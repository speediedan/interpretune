"""One rank of the two-rank check of cross-batch op state under the analysis runner's placement ``ddp``.

Three real analysis ops over a linear layer replicated by ``place_module``: ``encode`` (stateless: a forward pass per
example), ``running_sum`` (stateful: its ``op_state`` carries the running total of ``encode``'s outputs over every batch
so far, which it writes into each row) and ``scale`` (stateless: ``encode``'s outputs times the running total). Ten
examples in batches of two make five batches, three on rank 0 and two on rank 1, so each rank's own running total
differs from one process's after the first batch.

- ``terminal`` (``encode.running_sum``): the stateful op is last, so it is replayed after the merge.
- ``consumed`` (``encode.running_sum.scale``): a later op reads the running total, so ``encode.running_sum`` runs in the
  serial prefix pass and only ``scale`` runs on the shares.

For each, the merged store on both ranks must equal rank 0's one-process run of an unplaced copy with the same weights.
The analysis config is a stand-in with the real config's op-state lifecycle (``AnalysisCfg.op_state_for`` and
``reset_op_state``); ``init_analysis_cfgs`` is stubbed, since it would apply a full config to the module.

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


def _ops():
    from interpretune.analysis.inputs import OpStateSpec
    from interpretune.analysis.ops.base import AnalysisOp, ColCfg, CompositeAnalysisOp, OpSchema

    def col() -> ColCfg:
        return ColCfg(datasets_dtype="float32", required=False)

    def encode_impl(module, analysis_batch, batch, batch_idx):
        x = batch.to(torch.float32).unsqueeze(-1).repeat(1, 3)
        analysis_batch["x"] = module.net(x).sum(-1).detach()
        analysis_batch["examples"] = batch.tolist()
        return analysis_batch

    def running_sum_impl(analysis_batch, analysis_inputs=None):
        state = analysis_inputs.op_state
        total = float(state.get("total") or 0.0) + float(analysis_batch["x"].sum())
        state.set("total", total)
        analysis_batch["running"] = torch.tensor(total)
        return analysis_batch

    def scale_impl(analysis_batch):
        analysis_batch["scaled"] = analysis_batch["x"] * analysis_batch["running"]
        return analysis_batch

    encode = AnalysisOp("encode", "encode", OpSchema({"x": col()}), OpSchema({}))
    running = AnalysisOp(
        "running_sum",
        "running total",
        OpSchema({"running": col()}),
        OpSchema({"x": col()}),
        op_state=OpStateSpec(fields=("total",)),
    )
    scale = AnalysisOp("scale", "scale", OpSchema({"scaled": col()}), OpSchema({"x": col(), "running": col()}))
    for op, impl in ((encode, encode_impl), (running, running_sum_impl), (scale, scale_impl)):
        op._impl = impl

    def composite(name, members):
        built = CompositeAnalysisOp.__new__(CompositeAnalysisOp)  # unregistered parts: no schema compilation
        AnalysisOp.__init__(built, name, name, OpSchema({}), OpSchema({}))
        built.composition, built.composition_name = members, name
        return built

    return composite("terminal", [encode, running]), composite("consumed", [encode, running, scale])


class _Cfg:
    """The analysis config surface the runner and ops read, with the real config's op-state lifecycle."""

    def __init__(self, op, root: Path) -> None:
        from interpretune.config.analysis import AnalysisCfg

        self.op, self.output_schema, self.ignore_manual, self._op_states = op, None, False, {}
        self.output_store = SimpleNamespace(save_dir=root / "store", cache_dir=root / "cache", dataset=None)
        self._op_state_for, self._reset = AnalysisCfg.op_state_for, AnalysisCfg.reset_op_state

    def op_state_for(self, op):
        return self._op_state_for(self, op)

    def reset_op_state(self, *, epoch_boundary: bool = False):
        return self._reset(self, epoch_boundary=epoch_boundary)

    def save_batch(self, analysis_batch, batch, tokenizer=None):
        yield {k: analysis_batch[k] for k in ("examples", "x", "running", "scaled") if k in analysis_batch}


class _Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(3, 4)
        self.it_cfg = SimpleNamespace(profiler_cfg=SimpleNamespace(which=None))
        self.global_step, self.current_epoch, self.dtype, self.tokenizer = 1, 0, torch.float32, object()
        self.analysis_cfg = None

    def batch_to_device(self, batch):
        return batch

    def analysis_step(self, batch, batch_idx):
        from interpretune.analysis.execution import execute_analysis_step

        yield from execute_analysis_step(self, batch, batch_idx, analysis_cfg=self.analysis_cfg)

    def on_analysis_start(self):
        pass

    def on_analysis_end(self):
        pass

    def on_analysis_epoch_end(self):
        pass


class _DataModule:
    def test_dataloader(self):
        return DataLoader(torch.arange(10), batch_size=2, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from datasets import Features, Sequence, Value

    import interpretune.config.runner as runner_cfg
    from interpretune.runners import analysis
    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement

    runner_cfg.init_analysis_cfgs = lambda *a, **k: None  # the stand-in config is already applied
    columns = {"examples": Sequence(Value("int64")), "x": Sequence(Value("float32")), "running": Value("float32")}
    run = resolve_placement("ddp")
    assert run is not None
    init_process_group(run)
    out_dir = Path(args.out_dir)
    root = out_dir / f"rank{run.rank}"
    torch.manual_seed(1 + run.rank)  # placement broadcasts rank 0's weights
    module = _Module()
    weights = {k: v.detach().clone() for k, v in module.state_dict().items()}
    place_module(module, run, fsdp_plan={"net": {}})
    terminal, consumed = _ops()
    record: dict = {}
    for name, op, extra in (("terminal", terminal, {}), ("consumed", consumed, {"scaled": Sequence(Value("float32"))})):
        features = Features({**columns, **extra})
        analysis.dataset_features_and_format = lambda m, kwargs, f=features: (f, {}, kwargs)
        module.analysis_cfg = _Cfg(op, root / name)
        record[name] = analysis.core_analysis_loop(module, _DataModule(), dist_run=run).dataset.with_format(None)
        record[name] = record[name].to_dict()
        if run.rank == 0:
            reference = _Module()
            reference.load_state_dict(weights)
            reference.analysis_cfg = _Cfg(op, root / f"{name}_reference")
            record[f"{name}_reference"] = (
                analysis.core_analysis_loop(reference, _DataModule()).dataset.with_format(None).to_dict()
            )
    Path(out_dir, f"rank{run.rank}.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
