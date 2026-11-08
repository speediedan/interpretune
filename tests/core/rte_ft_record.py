"""Callbacks that record a phased fine-tuning run for the RTE CLI test (#765, #847), one per training-loop
interface.

Each writes the same JSON record when training ends: every epoch's schedule depth, thawed parameters and validation
metrics, the final depth, and the best checkpoint's depth and score. A test adds one to a shipped run configuration,
so both interfaces are compared through the CLI that runs them. Under a launcher, ``{rank}`` in the output path is
replaced by the process's rank, and the core recorder also records the parameter types (``DTensor`` once sharded).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

METRICS = ("val_loss", "accuracy")


def _thawed(module: torch.nn.Module) -> list[str]:
    return sorted(n for n, p in module.named_parameters() if p.requires_grad)


def _write(
    out: str, epochs: list[dict[str, Any]], final_depth: int, best_depth: Any, best_score: Any, **extra: Any
) -> None:
    record = {
        "final_depth": final_depth,
        "best_depth": best_depth,
        "best_score": None if best_score is None else float(best_score),
        "epochs": epochs,
        **extra,
    }
    Path(out.format(rank=os.environ.get("RANK", "0"))).write_text(json.dumps(record, indent=1))


class CoreEpochRecorder:
    """Records a core-loop run driven by ``FTSCoreCallback``; place it after that callback, so an epoch is recorded
    after the transition it starts with."""

    def __init__(self, out: str) -> None:
        self.out = out
        self.epochs: list[dict[str, Any]] = []

    @staticmethod
    def _tuning_scheduler(ctx: Any) -> Any:
        from interpretune.tuning.core_loop import FTSCoreCallback

        fts = next(cb for cb in ctx.callbacks if isinstance(cb, FTSCoreCallback))
        assert fts.tuning_scheduler is not None
        return fts.tuning_scheduler

    def on_train_epoch_start(self, ctx: Any) -> None:
        depth = self._tuning_scheduler(ctx).curr_depth
        self.epochs.append({"epoch": ctx.epoch, "depth": depth, "thawed": _thawed(ctx.module)})

    def on_train_epoch_end(self, ctx: Any, metrics: dict[str, Any]) -> None:
        self.epochs[-1].update({k: float(metrics[k]) for k in METRICS if k in metrics})

    def on_train_end(self, ctx: Any) -> None:
        ts = self._tuning_scheduler(ctx)
        # the checkpoint policy was built from configuration inside the callback; the tuning scheduler holds the only
        # reference to it and has no public accessor
        checkpoint = ts._checkpoint
        param_types = sorted({type(p).__name__ for p in ctx.module.parameters()})
        _write(
            self.out, self.epochs, ts.curr_depth, checkpoint.best_depth, checkpoint.best_score, param_types=param_types
        )


try:
    from lightning.pytorch import Callback
except ImportError:  # pragma: no cover - the CLI test skips without Lightning
    Callback = object  # type: ignore[assignment,misc]


class LightningEpochRecorder(Callback):  # type: ignore[misc,valid-type]
    """Records a Lightning run driven by the ``FinetuningScheduler`` callback."""

    def __init__(self, out: str) -> None:
        self.out = out
        self.epochs: list[dict[str, Any]] = []

    @staticmethod
    def _callback(trainer: Any, kind: str) -> Any:
        import finetuning_scheduler as fts

        return next(cb for cb in trainer.callbacks if isinstance(cb, getattr(fts, kind)))

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        depth = self._callback(trainer, "FinetuningScheduler").curr_depth
        self.epochs.append({"epoch": trainer.current_epoch, "depth": depth, "thawed": _thawed(pl_module)})

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self.epochs and not trainer.sanity_checking:
            metrics = trainer.callback_metrics
            self.epochs[-1].update({k: float(metrics[k]) for k in METRICS if k in metrics})

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        checkpoint = self._callback(trainer, "FTSCheckpoint")
        depth = self._callback(trainer, "FinetuningScheduler").curr_depth
        _write(self.out, self.epochs, depth, checkpoint.best_ckpt_depth, checkpoint.best_model_score)
