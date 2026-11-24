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
    Path(out.format(rank=os.environ.get("RANK", "0"))).write_text(json.dumps(record, indent=1), encoding="utf-8")


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


def wrapped_model_names(module: torch.nn.Module) -> dict[str, str]:
    """Each of ``module``'s parameter names mapped to the name of the same tensor in the pretrained model the
    composition wraps, so thawed sets can be compared across compositions of one model.

    A TransformerLens view supplies the mapping itself (its bridge rewires the wrapped model, and splits a fused tensor
    into views); otherwise the wrapped model is the first ``transformers.PreTrainedModel`` under ``module`` and its
    tensors are matched by identity.
    """
    from transformers import PreTrainedModel
    from venngram.tuning import VIEW_ATTR

    view = getattr(module, VIEW_ATTR, None)
    if callable(getattr(view, "wrapped_names", None)):
        return view.wrapped_names()
    wrapped = next(m for m in module.modules() if isinstance(m, PreTrainedModel))
    by_id = {id(p): n for n, p in wrapped.named_parameters(remove_duplicate=False)}
    return {n: by_id[id(p)] for n, p in module.named_parameters() if id(p) in by_id}


class AddressedEpochRecorder(CoreEpochRecorder):
    """A ``CoreEpochRecorder`` that also records, by the wrapped model's names, each epoch's thawed tensors and the
    thawed tensors whose values the epoch changed: a thawed tensor no step reaches is thawed but never trained."""

    def on_train_epoch_start(self, ctx: Any) -> None:
        super().on_train_epoch_start(ctx)
        self._names = wrapped_model_names(ctx.module)
        self.epochs[-1]["wrapped"] = sorted({self._wrapped(n) for n in self.epochs[-1]["thawed"]})
        self._start = {n: p.detach().clone() for n, p in ctx.module.named_parameters() if p.requires_grad}

    def on_train_epoch_end(self, ctx: Any, metrics: dict[str, Any]) -> None:
        super().on_train_epoch_end(ctx, metrics)
        params = dict(ctx.module.named_parameters())
        changed = {n for n, before in self._start.items() if not torch.equal(before, params[n].detach())}
        self.epochs[-1]["stepped"] = sorted({self._wrapped(n) for n in changed})

    def _wrapped(self, name: str) -> str:
        return self._names.get(name, f"<unwrapped>{name}")


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
