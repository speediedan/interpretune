"""Bundled core op family: label/index preparation, model forward passes, and logit diffs.

Self-contained modulo the sanctioned op-authoring surfaces (:mod:`interpretune.analysis.optools`,
:mod:`interpretune.analysis.backends`); see the custom ops composition guide.
"""

from __future__ import annotations

from typing import Callable

import torch
from transformers import BatchEncoding

from interpretune.analysis.ops.base import get_batch_input
from interpretune.analysis.optools import (
    boolean_logits_to_avg_logit_diff,
    extract_logits,
    get_loss_preds_diffs,
    require_model_backend,
)
from interpretune.protocol import DefaultAnalysisBatchProtocol


def labels_to_ids_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding
) -> DefaultAnalysisBatchProtocol:
    """Implementation for converting string labels to tensor IDs."""
    if "labels" in batch:
        label_ids, orig_labels = module.labels_to_ids(batch.pop("labels"))
        analysis_batch.update(label_ids=label_ids, orig_labels=orig_labels)
    return analysis_batch


def get_answer_indices_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for extracting answer indices from batch."""

    # Check if answer_indices already exist
    if hasattr(analysis_batch, "answer_indices") and analysis_batch.answer_indices is not None:
        return analysis_batch

    # Check if we can get from input store
    if module.analysis_cfg.input_store and getattr(module.analysis_cfg.input_store, "answer_indices", None) is not None:
        answer_indices = module.analysis_cfg.input_store.answer_indices[batch_idx]
    else:
        # Otherwise compute it
        tokens = get_batch_input(batch).detach().cpu()  # type: ignore[attr-defined]  # BatchEncoding tensor has detach/cpu
        if module.datamodule.tokenizer.padding_side == "left":
            answer_indices = torch.full((tokens.size(0),), -1)  # type: ignore[attr-defined]  # BatchEncoding tensor has size
        else:
            nonpadding_mask = tokens != module.datamodule.tokenizer.pad_token_id
            # This could be more robust, test with various datasets and padding strategies
            answer_indices = torch.where(nonpadding_mask, 1, 0).sum(dim=1) - 1

    analysis_batch.update(answer_indices=answer_indices)
    return analysis_batch


def model_fwd_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for basic model forward pass."""
    # Ensure we have answer indices
    if not hasattr(analysis_batch, "answer_indices") or analysis_batch.answer_indices is None:
        # Declared required_op invoked via the public op surface
        # (see NOTE [Op-Driven Transitive Dependency Atomicity])
        import interpretune as it

        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    # Run forward pass
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")
    _backend = getattr(module, "_model_backend", None)
    if _backend is not None:
        answer_logits = _backend.fwd(model=module.model, batch=batch)
    else:
        answer_logits = extract_logits(module(**batch))

    analysis_batch.update(answer_logits=answer_logits)
    return analysis_batch


# Keep backward-compatible alias
model_forward_impl = model_fwd_impl


def model_fwd_w_cache_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for forward pass with activation caching (no latent model hooks)."""
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")

    model_backend = require_model_backend(module)
    answer_logits, cache = model_backend.fwd_w_cache(
        model=module.model,
        batch=batch,
        names_filter=module.analysis_cfg.names_filter,
    )

    # Declared required_op invoked via the public op surface
    # (see NOTE [Op-Driven Transitive Dependency Atomicity])
    import interpretune as it

    analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)
    analysis_batch.update(cache=cache, alive_latents={}, answer_logits=answer_logits)
    return analysis_batch


def logit_diffs_impl(
    module: torch.nn.Module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
    get_loss_preds_diffs: Callable = get_loss_preds_diffs,
) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing logit differences."""

    logits, indices = analysis_batch.answer_logits, analysis_batch.answer_indices
    assert logits is not None and indices is not None, "answer_logits and answer_indices must not be None"
    assert isinstance(logits, torch.Tensor) and isinstance(indices, torch.Tensor), "logits and indices must be tensors"
    indexed_logits = logits[torch.arange(get_batch_input(batch).size(0)), indices]  # type: ignore[attr-defined]  # BatchEncoding tensor has size
    answer_logits = torch.squeeze(indexed_logits, dim=1)
    loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(module, analysis_batch, answer_logits, logit_diff_fn)
    if logit_diffs.dim() == 0:
        logit_diffs.unsqueeze_(0)
    analysis_batch.update(loss=loss, logit_diffs=logit_diffs, preds=preds, answer_logits=answer_logits)
    return analysis_batch
