"""Bundled SAE op family: latent-model caching forward passes, alive-latent extraction, and ablation.

Self-contained modulo the sanctioned op-authoring surfaces (:mod:`interpretune.analysis.optools`,
:mod:`interpretune.analysis.backends`); see the custom ops composition guide.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import BatchEncoding

if TYPE_CHECKING:
    from transformer_lens.hook_points import HookPoint

from interpretune.analysis.ops.base import get_batch_input
from interpretune.analysis.optools import require_model_backend
from interpretune.protocol import DefaultAnalysisBatchProtocol


def ablate_sae_latent(
    sae_acts: torch.Tensor,
    hook: HookPoint,  # required by transformer_lens.hook_points._HookFunctionProtocol
    latent_idx: int | None = None,
    seq_pos: torch.Tensor | None = None,  # batched
) -> torch.Tensor:
    """Ablate a particular latent at a particular sequence position.

    If either argument is None, we ablate at all latents / sequence positions.
    """
    sae_acts[torch.arange(sae_acts.size(0)), seq_pos, latent_idx] = 0.0
    return sae_acts


def get_alive_latents_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for extracting alive latents from cache."""
    # Check if alive_latents already exist
    if hasattr(analysis_batch, "alive_latents") and analysis_batch.alive_latents is not None:
        return analysis_batch

    # Check if we can get from input store
    # TODO: remove this leaky abstraction, alive_latents should only be in analysis_batch, not accessed
    #       via analysis_cfg.input_store at the op level
    if module.analysis_cfg.input_store and module.analysis_cfg.input_store.alive_latents is not None:
        alive_latents = module.analysis_cfg.input_store.alive_latents[batch_idx]
    elif not hasattr(analysis_batch, "cache") or analysis_batch.cache is None:
        alive_latents = {}
    else:
        # Extract alive latents from the cache using the answer indices
        cache = analysis_batch.cache
        names_filter = module.analysis_cfg.names_filter
        answer_indices = analysis_batch.answer_indices

        filtered_acts = {name: acts for name, acts in cache.items() if names_filter(name)}
        alive_latents = {}
        for name, acts in filtered_acts.items():
            alive = (acts[torch.arange(acts.size(0)), answer_indices, :] > 0).any(dim=0).nonzero().squeeze(1).tolist()
            alive_latents[name] = alive

    analysis_batch.update(alive_latents=alive_latents)
    return analysis_batch


def model_fwd_w_cache_latent_models_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for forward pass with activation caching and latent model (SAE) hooks."""
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")

    model_backend = require_model_backend(module)
    latent_model_handles = getattr(module, "sae_handles", None)
    if not latent_model_handles:
        raise ValueError("model_fwd_w_cache_latent_models requires sae_handles on the module")

    answer_logits, cache = model_backend.fwd_w_cache_and_latent_models(
        model=module.model,
        batch=batch,
        latent_model_handles=latent_model_handles,
        names_filter=module.analysis_cfg.names_filter,
    )

    # Declared required_ops invoked via the public op surface
    # (see NOTE [Op-Driven Transitive Dependency Atomicity])
    import interpretune as it

    analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)
    analysis_batch.update(cache=cache)
    analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)  # type: ignore[call-arg]
    analysis_batch.update(answer_logits=answer_logits)
    return analysis_batch


# Keep backward-compatible alias
model_cache_forward_impl = model_fwd_w_cache_latent_models_impl


def model_ablation_impl(
    module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    batch_idx: int,
    ablate_latent_fn: Callable = ablate_sae_latent,
) -> DefaultAnalysisBatchProtocol:
    """Implementation for model ablation analysis."""
    # Declared required_ops invoked via the public op surface
    # (see NOTE [Op-Driven Transitive Dependency Atomicity])
    import interpretune as it

    # Ensure we have answer indices and alive latents
    if not hasattr(analysis_batch, "answer_indices") or analysis_batch.answer_indices is None:
        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")

    if not hasattr(analysis_batch, "alive_latents") or analysis_batch.alive_latents is None:
        # TODO: remove this leaky abstraction, alive_latents should only be in analysis_batch
        assert module.analysis_cfg.input_store and getattr(module.analysis_cfg.input_store, "alive_latents", None), (
            "alive_latents required for ablation op"
        )
        analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)  # type: ignore[call-arg]

    answer_indices = analysis_batch.answer_indices
    alive_latents = analysis_batch.alive_latents

    # Build hook configs for every (name, latent_idx) pair, then run them in batch.
    per_latent_logits: dict[str, dict[Any, torch.Tensor]] = defaultdict(dict)
    assert alive_latents is not None and isinstance(alive_latents, dict), "alive_latents must be a dict"

    hook_configs: list[list[tuple[str, Any]]] = []
    index_map: list[tuple[str, Any]] = []  # parallel list: (name, latent_idx) per config
    for name, alive in alive_latents.items():
        for latent_idx in alive:
            hook_configs.append([(name, partial(ablate_latent_fn, latent_idx=latent_idx, seq_pos=answer_indices))])
            index_map.append((name, latent_idx))

    model_backend = require_model_backend(module)
    all_logits = model_backend.fwd_w_hooks_batched(
        model=module.model,
        batch=batch,
        latent_model_handles=module.sae_handles,
        hook_configs=hook_configs,
        clear_contexts=True,
    )

    batch_indices = torch.arange(get_batch_input(batch).size(0))  # type: ignore[attr-defined]
    for (name, latent_idx), answer_logits in zip(index_map, all_logits, strict=True):
        per_latent_logits[name][latent_idx] = answer_logits[batch_indices, answer_indices, :]

    analysis_batch.update(answer_logits=per_latent_logits)
    return analysis_batch


def sae_correct_acts_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing correct activations from SAE outputs."""
    # Validate required inputs # TODO: refactor all required input checks to use shared AnalysisOp or Dispatcher logic
    required_inputs = ["logit_diffs", "answer_indices", "cache"]
    for key in required_inputs:
        if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
            raise ValueError(f"Missing required input '{key}' for {module.__class__.__name__}.sae_correct_acts")

    # Extract required data from analysis_batch
    cache = analysis_batch.cache
    logit_diffs = analysis_batch.logit_diffs
    answer_indices = analysis_batch.answer_indices

    # Ensure alive_latents are present
    if not hasattr(analysis_batch, "alive_latents") or analysis_batch.alive_latents is None:
        # Declared required_op invoked via the public op surface
        # (see NOTE [Op-Driven Transitive Dependency Atomicity])
        import interpretune as it

        analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)  # type: ignore[call-arg]

    assert isinstance(logit_diffs, torch.Tensor), "expected logit_diffs to be a Tensor"
    # Extract correct activations for examples with positive logit differences
    correct_mask = logit_diffs > 0
    # Handle scalar case
    if correct_mask.dim() == 0:
        correct_mask = correct_mask.unsqueeze(0)
    if logit_diffs.dim() == 0:
        logit_diffs = logit_diffs.unsqueeze(0)

    correct_activations = {}
    names_filter = module.analysis_cfg.names_filter  # type: ignore[attr-defined]
    assert cache is not None, "cache should not be None after validation"
    for name, acts in cache.items():
        if not names_filter(name):
            continue

        # Get activations at answer indices and select only for examples with positive logit diffs
        # Ensure index tensors are on the same device as acts to avoid cross-device indexing errors
        acts_device = acts.device
        assert answer_indices is not None and correct_mask is not None  # validated by caller
        acts_at_answer = acts[torch.arange(acts.size(0), device=acts_device), answer_indices.to(acts_device)]
        correct_activations[name] = acts_at_answer[correct_mask.to(acts_device)].cpu()

    analysis_batch.update(correct_activations=correct_activations)
    return analysis_batch
