"""One rank of the two-rank latent-tuning check (#834): FSDP2 latent-space fine-tuning through the core loop.

A SAELens composition (the tiny custom TransformerLens model of ``CoreSLCust``) with SAEs spliced in, so the SAE
parameters take the ``latent:sae`` role. finetuning-scheduler's ``TuningScheduler`` drives two phases through
``FTSCoreCallback``: the base model's last block first, then, by role, the spliced SAEs (``role:latent:sae``).

Run under ``torch.distributed.run`` as one rank of an ``fsdp2`` run. Writes ``<out_dir>/rank<rank>.json`` with the
per-step training losses, each phase's trainable parameter names (read at every epoch end), the final depth, each
rank's device and its peak allocated accelerator memory.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import torch

PER_RANK_BATCH = 2


class PhaseRecorder:
    """Records each optimizer step's loss and, at each epoch end, the trainable parameters of the module."""

    def __init__(self) -> None:
        self.module = None
        self.losses: list[float] = []
        self.trainable_by_epoch: list[list[str]] = []

    def on_train_step_end(self, ctx, outputs):
        self.losses.append(float(outputs.detach()))

    def on_train_epoch_end(self, ctx, metrics):
        names = sorted(n for n, p in self.module.named_parameters() if p.requires_grad)
        self.trainable_by_epoch.append(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--fsdp-plan", type=json.loads, required=True, help="the FSDP2 plan, as JSON")
    args = parser.parse_args()

    from interpretune.config import SessionRunnerCfg
    from interpretune.runners import SessionRunner
    from interpretune.session import ITSession
    from interpretune.tuning.core_loop import FTSCoreCallback
    from tests.configuration import config_modules
    from tests.core.cfg_aliases import CoreSLCust
    from tests.core.test_adapters_sae_lens import TestClassSAELens

    rank = int(os.environ.get("RANK", "0"))
    out_dir = Path(args.out_dir)
    test_cfg = CoreSLCust(
        phase="train",
        force_prepare_data=False,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches,
        limit_val_batches=2,
        dm_override_cfg={"train_batch_size": PER_RANK_BATCH, "eval_batch_size": PER_RANK_BATCH},
    )
    torch.manual_seed(1 + rank)  # ranks initialize differently; placement must broadcast rank 0's weights
    cfg = config_modules(test_cfg, "dist_latent", {}, out_dir / f"rank{rank}", cfg_only=True)
    cfg.module_cfg.add_saes_on_init = True
    cfg.module_cfg.sae_cfgs = [deepcopy(TestClassSAELens.test_sl_cust_config)]
    it_session = ITSession(cfg)
    module = it_session.module
    recorder = PhaseRecorder()
    recorder.module = module
    fts = FTSCoreCallback()  # its schedule names the placed module's parameters, set once they exist
    runner_cfg = SessionRunnerCfg(
        it_session=it_session,
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches,
        limit_val_batches=2,
        callbacks=[fts, recorder],
        placement="fsdp2",
        fsdp_plan=args.fsdp_plan,
    )
    runner = SessionRunner(run_cfg=runner_cfg)
    # the view names the bridge's parameters canonically (``blocks.1.attn.W_Q``), so phase 0 selects the last block
    # by a regex over those names; phase 1 adds the spliced SAEs by role
    n_layers = module.model.cfg.n_layers
    last_block = rf"blocks\.{n_layers - 1}\..*"
    fts.schedule = {
        0: {"params": [last_block], "max_transition_epoch": args.epochs // 2},
        1: {"params": ["role:latent:sae"]},
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    runner.train()
    record = {
        "losses": recorder.losses,
        "trainable_by_epoch": recorder.trainable_by_epoch,
        "phase0_selector": last_block,
        "sae_params": sorted(n for n, _ in module.named_parameters() if n.startswith("attached_saes.")),
        "depth": fts.tuning_scheduler.curr_depth if fts.tuning_scheduler is not None else None,
        "devices": list(runner.dist_run.devices) if runner.dist_run else [],
        "peak_allocated_gb": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }
    (out_dir / f"rank{rank}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
