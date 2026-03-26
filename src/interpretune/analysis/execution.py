"""Shared analysis execution helpers for generated and interactive workflows."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator, Mapping
from typing import Any

from transformers import BatchEncoding

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.helpers import AnalysisInputs, coerce_analysis_inputs, resolve_tokenizer
from interpretune.protocol import STEP_OUTPUT


def build_analysis_inputs(
    module: Any,
    analysis_cfg: Any | None = None,
    analysis_inputs: AnalysisInputs | Mapping[str, Any] | None = None,
) -> AnalysisInputs:
    """Combine config-backed and caller-provided analysis inputs into one scoped input object."""
    analysis_cfg = analysis_cfg or getattr(module, "analysis_cfg", None)
    config_inputs = AnalysisInputs(
        batch=getattr(analysis_cfg, "batch_inputs", None),
        run=getattr(analysis_cfg, "run_inputs", None),
        store=getattr(analysis_cfg, "input_store", None),
    )
    return config_inputs.merged(coerce_analysis_inputs(analysis_inputs))


@contextmanager
def activated_analysis_cfg(
    module: Any,
    analysis_cfg: Any | None,
    *,
    ignore_manual: bool | None = None,
) -> Generator[Any, None, None]:
    """Temporarily activate an analysis cfg on a module for manual or runner execution."""
    resolved_cfg = analysis_cfg or getattr(module, "analysis_cfg", None)
    if resolved_cfg is None:
        raise ValueError("An active analysis_cfg is required")

    from interpretune.config.runner import init_analysis_cfgs

    has_previous_cfg = hasattr(module, "analysis_cfg")
    previous_cfg = getattr(module, "analysis_cfg", None)
    if ignore_manual is None:
        ignore_manual = bool(getattr(resolved_cfg, "ignore_manual", False))

    init_analysis_cfgs(module, resolved_cfg, ignore_manual=ignore_manual)
    module.analysis_cfg = resolved_cfg
    try:
        yield resolved_cfg
    finally:
        if has_previous_cfg:
            module.analysis_cfg = previous_cfg
        else:
            delattr(module, "analysis_cfg")


def execute_analysis_op(
    module: Any,
    batch: BatchEncoding | None = None,
    batch_idx: int = 0,
    *,
    analysis_batch: AnalysisBatch | None = None,
    analysis_cfg: Any | None = None,
    analysis_inputs: AnalysisInputs | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> AnalysisBatch:
    """Execute the configured analysis op through a shared helper."""
    resolved_cfg = analysis_cfg or getattr(module, "analysis_cfg", None)
    if resolved_cfg is None or resolved_cfg.op is None:
        raise ValueError("execute_analysis_op requires an analysis_cfg with a resolved op")

    active_batch = analysis_batch or AnalysisBatch()
    with activated_analysis_cfg(module, resolved_cfg) as active_cfg:
        resolved_inputs = build_analysis_inputs(module, analysis_cfg=active_cfg, analysis_inputs=analysis_inputs)
        return active_cfg.op(
            module,
            active_batch,
            batch,
            batch_idx,
            analysis_inputs=resolved_inputs,
            **kwargs,
        )


def execute_analysis_step(
    module: Any,
    batch: BatchEncoding | None = None,
    batch_idx: int = 0,
    *,
    dataloader_idx: int = 0,
    analysis_batch: AnalysisBatch | None = None,
    analysis_cfg: Any | None = None,
    analysis_inputs: AnalysisInputs | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Generator[STEP_OUTPUT, None, None]:
    """Execute and serialize one analysis step through the shared helper path."""
    del dataloader_idx
    resolved_cfg = analysis_cfg or getattr(module, "analysis_cfg", None)
    if resolved_cfg is None:
        raise ValueError("execute_analysis_step requires an active analysis_cfg")

    result = execute_analysis_op(
        module,
        batch,
        batch_idx,
        analysis_batch=analysis_batch,
        analysis_cfg=resolved_cfg,
        analysis_inputs=analysis_inputs,
        **kwargs,
    )
    tokenizer = resolve_tokenizer(module)
    yield from resolved_cfg.save_batch(result, batch, tokenizer=tokenizer)


__all__ = [
    "AnalysisInputs",
    "activated_analysis_cfg",
    "build_analysis_inputs",
    "execute_analysis_op",
    "execute_analysis_step",
]
