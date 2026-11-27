"""One rank of the two-rank check of the analysis runner's loop under placement ``ddp`` (FSDP2 replication).

The module holds a small linear layer replicated with ``fully_shard`` over a ``(replicate, shard)`` mesh, as
``place_module`` places it under ``ddp`` (the layer its own unit, since the step calls it directly). Its analysis step
runs a backward pass through the layer, so with FSDP2's gradient reduction left on, every backward would wait on the
other rank. Ten examples in batches of two make five batches, three on rank 0 and two on rank 1, so a rank waiting on
the other would wait forever and the test would time out.

Runs of ``core_analysis_loop``:

- ``ordered``: an unshuffled loader. The merged store equals rank 0's one-process run of an unplaced copy with the same
  weights, row for row (each row is one batch: its examples, the step's output and its batch index).
- ``shuffled``: a shuffled loader over two epochs. Both ranks return the same merged store, every example once per
  epoch, with batch indices in order.
- ``limited``: ``limit_analysis_batches=3`` keeps the first three batches of the whole sequence.
- Refusals: an op declaring ``op_state``, and a deterministic ``dataset_fingerprint``.

The step records its output with ``log_analysis``, so each run's ``analysis_metrics`` (per epoch) are recorded too:
every rank's equal the one-process run's, although each rank analyzed only its own batches.

Writes ``<out_dir>/rank<rank>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from interpretune.base.components.mixins import AnalysisStepMixin


class _Store:
    def __init__(self, root: Path) -> None:
        self.save_dir, self.cache_dir, self.dataset = root / "store", root / "cache", None


class _Module(nn.Module):
    """What ``core_analysis_loop`` reads from a module, with an analysis step that runs a backward pass."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.net = nn.Linear(3, 4)
        self.analysis_cfg = SimpleNamespace(output_store=_Store(root), output_schema=None, op=None)
        self.it_cfg = SimpleNamespace(profiler_cfg=SimpleNamespace(which=None))
        self.global_step, self.current_epoch, self.dtype = 1, 0, torch.float32

    # the analysis-metric accumulator the loop resets, flushes and reports, as every analysis module has it
    log_analysis = AnalysisStepMixin.log_analysis
    reset_analysis_metrics = AnalysisStepMixin.reset_analysis_metrics
    flush_analysis_metrics = AnalysisStepMixin.flush_analysis_metrics
    analysis_metrics = AnalysisStepMixin.analysis_metrics

    def batch_to_device(self, batch):
        return batch

    def analysis_step(self, batch, batch_idx):
        with torch.enable_grad():
            x = batch.to(torch.float32).unsqueeze(-1).repeat(1, 3)
            out = self.net(x).pow(2).sum()
            out.backward()  # FSDP2 would reduce these gradients across the replicas with its sync on
        self.log_analysis("out", out)
        yield {"examples": batch.tolist(), "out": float(out.detach()), "batch_idx": batch_idx}

    def on_analysis_start(self):
        pass

    def on_analysis_end(self):
        pass

    def on_analysis_epoch_end(self):
        pass


class _DataModule:
    def __init__(self, shuffle: bool) -> None:
        self.shuffle = shuffle

    def test_dataloader(self):
        return DataLoader(torch.arange(10), batch_size=2, shuffle=self.shuffle)


def _rows(store) -> dict:
    return store.dataset.with_format(None).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from datasets import Features, Sequence, Value

    from interpretune.runners import analysis
    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException

    features = Features({"examples": Sequence(Value("int64")), "out": Value("float64"), "batch_idx": Value("int64")})
    analysis.dataset_features_and_format = lambda module, kwargs: (features, {}, kwargs)
    run = resolve_placement("ddp")
    assert run is not None
    init_process_group(run)
    out_dir = Path(args.out_dir)
    root = out_dir / f"rank{run.rank}"
    torch.manual_seed(1 + run.rank)  # ranks initialize differently; placement broadcasts rank 0's weights
    module = _Module(root / "ordered")
    reference_weights = {k: v.detach().clone() for k, v in module.state_dict().items()}
    place_module(module, run, fsdp_plan={"net": {}})  # its own unit: the step calls it directly, not the root
    record: dict = {"param_types": sorted({type(p.data).__name__ for p in module.parameters()})}
    store = analysis.core_analysis_loop(module, _DataModule(shuffle=False), dist_run=run)
    record["ordered"] = _rows(store)
    record["ordered_metrics"] = module.analysis_metrics
    record["share_dirs_left"] = sorted(p.name for p in (root / "ordered").iterdir() if ".share" in p.name)
    if run.rank == 0:
        reference = _Module(root / "reference")
        reference.load_state_dict(reference_weights)
        record["reference"] = _rows(analysis.core_analysis_loop(reference, _DataModule(shuffle=False)))
        record["reference_metrics"] = reference.analysis_metrics
    module.analysis_cfg.output_store = _Store(root / "shuffled")
    record["shuffled"] = _rows(
        analysis.core_analysis_loop(module, _DataModule(shuffle=True), max_epochs=2, dist_run=run)
    )
    record["shuffled_metrics"] = module.analysis_metrics
    module.analysis_cfg.output_store = _Store(root / "limited")
    limited = analysis.core_analysis_loop(module, _DataModule(shuffle=False), limit_analysis_batches=3, dist_run=run)
    record["limited"] = _rows(limited)
    refusals = {}
    module.analysis_cfg.op = SimpleNamespace(name="running_mean", op_state={"scope": "run"})
    try:
        analysis.core_analysis_loop(module, _DataModule(shuffle=False), dist_run=run)
    except MisconfigurationException as err:
        refusals["op_state"] = str(err)
    module.analysis_cfg.op = None
    module.analysis_cfg.output_store = _Store(root / "keyed")
    try:
        analysis.core_analysis_loop(module, _DataModule(shuffle=False), dist_run=run, dataset_fingerprint="k")
    except MisconfigurationException as err:
        refusals["fingerprint"] = str(err)
    record["refusals"] = refusals
    Path(out_dir, f"rank{run.rank}.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
