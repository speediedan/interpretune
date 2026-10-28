"""One process of the core loop's distributed parity runs (``tests/core/test_core_loop_distributed.py``).

Run under ``torch.distributed.run`` as one rank of a ``ddp`` or ``fsdp2`` run, or with ``--placement single`` as the
one-process reference, which trains on the global batch (the per-rank batch times the world size) that the ranks
share out. Each rank seeds its model differently and the reference is seeded as rank 0: the runner broadcasts rank 0's
weights before placement, and without that broadcast the ranks would not match it.

Writes ``<out_dir>/<placement>_rank<rank>.json`` with the per-step training losses, the epoch metrics the callbacks saw,
each rank's recorded device and the parameter types after placement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

PER_RANK_BATCH = 2


class LossRecorder:
    """Records each optimizer step's loss and the epoch metrics the loop hands its callbacks."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.epoch_metrics: list[dict[str, float]] = []

    def on_train_step_end(self, ctx, outputs):
        self.losses.append(float(outputs.detach()))

    def on_train_epoch_end(self, ctx, metrics):
        self.epoch_metrics.append(dict(metrics))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", required=True)
    parser.add_argument("--world-size", type=int, default=2, help="the ranks the reference's global batch stands for")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-units", nargs="*", default=None)
    args = parser.parse_args()

    from interpretune.config import SessionRunnerCfg
    from interpretune.runners import SessionRunner
    from tests.configuration import config_modules
    from tests.parity_acceptance.test_it_l import CoreCfg

    rank = int(os.environ.get("RANK", "0"))
    batch_size = PER_RANK_BATCH * (args.world_size if args.placement == "single" else 1)
    test_cfg = CoreCfg(
        force_prepare_data=False,
        max_epochs=1,
        limit_train_batches=3,
        limit_val_batches=2,
        dm_override_cfg={"train_batch_size": batch_size, "eval_batch_size": batch_size},
        # no dropout: each rank draws from its own RNG stream, the reference from one
        model_cfg={"model_args": {"max_seq_len": 200, "n_layers": 2, "dropout_p": 0.0}},
    )
    out_dir = Path(args.out_dir)
    torch.manual_seed(1 + rank)  # ranks initialize differently; placement must broadcast rank 0's weights
    it_session = config_modules(test_cfg, f"dist_{args.placement}", {}, out_dir / f"{args.placement}_rank{rank}")
    recorder = LossRecorder()
    runner_cfg = SessionRunnerCfg(
        it_session=it_session,
        max_epochs=test_cfg.max_epochs,
        limit_train_batches=test_cfg.limit_train_batches,
        limit_val_batches=test_cfg.limit_val_batches,
        callbacks=[recorder],
        placement=args.placement,
        shard_units=args.shard_units,
    )
    runner = SessionRunner(run_cfg=runner_cfg)
    param_types = sorted({type(p).__name__ for p in it_session.module.parameters()})
    runner.train()
    record = {
        "losses": recorder.losses,
        "epoch_metrics": recorder.epoch_metrics,
        "devices": list(runner.dist_run.devices) if runner.dist_run else [],
        "param_types": param_types,
    }
    (out_dir / f"{args.placement}_rank{rank}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
