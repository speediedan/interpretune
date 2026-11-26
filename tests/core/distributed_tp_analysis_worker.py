"""One rank of the two-rank check of the analysis runner's loop under placement ``tp``.

The module is a stand-in whose analysis step makes a collective on every batch (an ``all_reduce`` of the batch), as a
split model's forward does, so a rank that ran a step alone, or skipped one, would wait forever and the test would time
out. The loader shuffles, and each rank seeds its own random stream differently, so only the runner's shared order
makes the ranks see the same batches.

Three runs of ``core_analysis_loop``:

- ``fresh``: a random fingerprint. Every rank returns the same rows, in the same order, and only rank 0 writes them (the
  other rank's store directory and generator cache stay empty).
- ``keyed``, twice with one deterministic fingerprint and generator cache directory: the second run is served from rank
  0's cache, so its generator never starts, and no rank runs a step.

Then the J-lens weight seam on a tiny Llama whose LM head and embedding are split across the ranks (the LM head
column-wise over the vocabulary, the embedding row-wise): ``resolve_unembed_and_norm_scale`` and
``resolve_embedding_weight`` return the whole weights, a folded readout direction equals one process's, and inside
``gathered_weights`` each weight is gathered once.

Writes ``<out_dir>/rank<rank>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


class _Store:
    def __init__(self, root: Path) -> None:
        self.save_dir, self.cache_dir, self.dataset = root / "store", root / "cache", None


class _Module:
    """What ``core_analysis_loop`` reads from a module, with an analysis step that makes a collective per batch."""

    def __init__(self, root: Path) -> None:
        self.analysis_cfg = SimpleNamespace(output_store=_Store(root), output_schema=None, op=None)
        self.it_cfg = SimpleNamespace(profiler_cfg=SimpleNamespace(which=None))
        self.global_step, self.current_epoch, self.steps, self.dtype = 1, 0, 0, torch.float32

    def batch_to_device(self, batch):
        return batch

    def analysis_step(self, batch, batch_idx):
        self.steps += 1
        total = batch.sum().to(torch.float64)
        dist.all_reduce(total)  # what a split layer's forward does: every rank must make it
        for row in batch.tolist():
            yield {"row": row, "shared": float(total)}

    def on_analysis_start(self):
        pass

    def on_analysis_end(self):
        pass

    def on_analysis_epoch_end(self):
        pass


class _DataModule:
    def test_dataloader(self):
        return DataLoader(torch.arange(10), batch_size=3, shuffle=True)


def _analysis_runs(run, out_dir: Path) -> dict:
    from datasets import Features, Value

    from interpretune.runners import analysis

    features = Features({"row": Value("int64"), "shared": Value("float64")})
    analysis.dataset_features_and_format = lambda module, kwargs: (features, {}, kwargs)
    record: dict = {}
    root = out_dir / f"rank{run.rank}"
    module = _Module(root / "fresh")
    store = analysis.core_analysis_loop(module, _DataModule(), max_epochs=2, dist_run=run)
    record["fresh"] = {
        "rows": store.dataset.with_format(None).to_dict(),
        "steps": module.steps,
        "store_written": (root / "fresh" / "store").exists(),
        "cache_files": sorted(p.name for p in (root / "fresh" / "cache").rglob("*.arrow")),
    }
    shared_cache = out_dir / "shared_cache"  # every rank names the same directory; only rank 0 may write it
    for attempt in ("keyed_first", "keyed_second"):
        module = _Module(root / attempt)
        store = analysis.core_analysis_loop(
            module,
            _DataModule(),
            max_epochs=1,
            dist_run=run,
            dataset_fingerprint="tp-analysis-check",
            generator_cache_dir=shared_cache,
        )
        record[attempt] = {"rows": store.dataset.with_format(None).to_dict(), "steps": module.steps}
    return record


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _weight_seam(run) -> dict:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    from interpretune.analysis.optools import (
        fold_norm_into_unembed_rows,
        resolve_embedding_weight,
        resolve_unembed_and_norm_scale,
    )
    from interpretune.runners import distributed

    reference, placed = _tiny_llama(), _tiny_llama()
    mesh = init_device_mesh("cpu", (run.world_size,))
    parallelize_module(
        placed,
        mesh,
        {
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
            "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        },
    )
    split = sorted(n for n, p in placed.named_parameters() if type(p.data).__name__ == "DTensor")
    ref_info = resolve_unembed_and_norm_scale(SimpleNamespace(model=reference))
    tokens = torch.tensor([3, 17, 40])
    calls = {"n": 0}
    original = torch.distributed.tensor.DTensor.full_tensor

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    torch.distributed.tensor.DTensor.full_tensor = counting
    try:
        with distributed.gathered_weights():
            info = resolve_unembed_and_norm_scale(SimpleNamespace(model=placed))
            again = resolve_unembed_and_norm_scale(SimpleNamespace(model=placed))
            embed = resolve_embedding_weight(SimpleNamespace(model=placed))
            inside = calls["n"]
        outside_first = resolve_unembed_and_norm_scale(SimpleNamespace(model=placed)).w_u
        outside_second = resolve_unembed_and_norm_scale(SimpleNamespace(model=placed)).w_u
    finally:
        torch.distributed.tensor.DTensor.full_tensor = original
    return {
        "split_params": split,
        "lm_head_local_rows": int(placed.lm_head.weight.to_local().shape[0]),
        "unembed_whole": torch.equal(info.w_u, reference.lm_head.weight.detach()),
        "unembed_type": type(info.w_u).__name__,
        "norm_scale_equal": torch.equal(info.norm_scale, ref_info.norm_scale),
        "norm_kind": info.norm_kind,
        "embedding_whole": torch.equal(embed, reference.model.embed_tokens.weight.detach()),
        "readout_equal": torch.equal(
            fold_norm_into_unembed_rows(info, tokens, apply_norm=True),
            fold_norm_into_unembed_rows(ref_info, tokens, apply_norm=True),
        ),
        "gathered_once": inside == 2 and again.w_u is info.w_u,  # the unembedding and the embedding, once each
        "gathers_outside": calls["n"] - inside,
        "outside_fresh": outside_first is not outside_second,
        "shard_0": isinstance(placed.model.embed_tokens.weight.placements[0], Shard),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from interpretune.runners.distributed import init_process_group, resolve_placement

    run = resolve_placement("tp")
    assert run is not None
    init_process_group(run)
    torch.manual_seed(1 + run.rank)  # each rank's own stream differs: only the shared sampler seed can align them
    out_dir = Path(args.out_dir)
    record = _analysis_runs(run, out_dir)
    record["weights"] = _weight_seam(run)
    Path(out_dir, f"rank{run.rank}.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
