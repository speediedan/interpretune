"""Bundled attribution op family: gradient- and ablation-based attribution.

Self-contained modulo the sanctioned op-authoring surfaces (:mod:`interpretune.analysis.optools`,
:mod:`interpretune.analysis.backends`); see the custom ops composition guide.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch
from transformers import BatchEncoding

from interpretune.analysis.ops.base import AnalysisBatch, get_batch_input
from interpretune.analysis.optools import (
    boolean_logits_to_avg_logit_diff,
    get_loss_preds_diffs,
    require_model_backend,
)
from interpretune.protocol import DefaultAnalysisBatchProtocol


def model_gradient_impl(
    module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    batch_idx: int,
    logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
    get_loss_preds_diffs: Callable = get_loss_preds_diffs,
) -> DefaultAnalysisBatchProtocol:
    """Implementation for gradient-based attribution.

    Defines a ``backward_fn`` closure that extracts answer logits, computes logit diffs,
    and returns their sum as the scalar to backpropagate.  The backend handles the entire
    forward + backward flow (enabling both eager and trace-based execution).
    """

    # Ensure we have answer indices
    if not hasattr(analysis_batch, "answer_indices") or analysis_batch.answer_indices is None:
        # Declared required_op invoked via the public op surface
        # (see NOTE [Op-Driven Transitive Dependency Atomicity])
        import interpretune as it

        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")

    answer_indices = analysis_batch.answer_indices

    # if we're running a manual analysis_step context, we may need to manually set hooks
    module.analysis_cfg.add_default_cache_hooks()
    # Verify hooks are configured
    assert all((module.analysis_cfg.fwd_hooks, module.analysis_cfg.bwd_hooks)), (
        "fwd_hooks and bwd_hooks required for gradient-based attribution op"
    )

    # TODO: In the future, we will likely use IT dispatch logic to control toggling autograd/inference mode etc.
    #       but for now controlling manually here
    # ---- backward_fn closure: captures op-specific state ---------------------
    # Applied to raw logits inside the backend.  Must use only standard PyTorch ops
    # so NNsight can trace through it (all operations intercepted via __torch_function__).
    def backward_fn(raw_logits: torch.Tensor) -> torch.Tensor:
        """Extract answer logits, compute logit diffs via get_loss_preds_diffs, return scalar."""
        sliced = raw_logits[torch.arange(raw_logits.size(0)), answer_indices]
        squeezed = torch.squeeze(sliced, dim=1)
        _, logit_diffs, _, _ = get_loss_preds_diffs(module, analysis_batch, squeezed, logit_diff_fn)
        return logit_diffs.sum()

    # ---- Run forward + backward via backend ----------------------------------
    model_backend = require_model_backend(module)
    raw_logits = model_backend.fwd_w_grads_and_latent_models(
        model=module.model,
        batch=batch,
        latent_model_handles=module.sae_handles,
        fwd_hooks=module.analysis_cfg.fwd_hooks,
        bwd_hooks=module.analysis_cfg.bwd_hooks,
        backward_fn=backward_fn,
    )

    # ---- Recompute metrics from returned real logits -------------------------
    answer_logits = torch.squeeze(
        raw_logits[torch.arange(get_batch_input(batch).size(0)), answer_indices],  # type: ignore[attr-defined]  # BatchEncoding tensor has size
        dim=1,
    )
    loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(module, analysis_batch, answer_logits, logit_diff_fn)
    if logit_diffs.dim() == 0:
        logit_diffs.unsqueeze_(0)

    analysis_batch.update(
        answer_logits=answer_logits,
        answer_indices=answer_indices,
        logit_diffs=logit_diffs,
        preds=preds,
        loss=loss,
        grad_cache=module.analysis_cfg.cache_dict,  # Store the gradient cache
    )
    return analysis_batch


def gradient_attribution_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing attribution values from gradients."""
    # TODO: change this to use shared superclass required input validation
    # Ensure required inputs exist
    required_inputs = ["answer_indices", "logit_diffs"]
    for key in required_inputs:
        if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
            raise ValueError(f"Missing required input '{key}' for gradient attribution")

    # Type checker assistance after validation
    assert analysis_batch.logit_diffs is not None, "logit_diffs validated above"
    assert isinstance(analysis_batch.logit_diffs, torch.Tensor), "logit_diffs should be tensor after validation"

    # TODO: switch to using grad_cache from analysis_batch once that functionality is implemented
    # Get cached activations (forwards) and gradients (backwards) from analysis_cfg.cache_dict
    # Prefer grad_cache on the analysis_batch, else fall back to module.analysis_cfg.cache_dict
    if getattr(analysis_batch, "grad_cache", None) is not None:
        cache_source = analysis_batch.grad_cache
    elif getattr(module.analysis_cfg, "cache_dict", None) is not None:
        cache_source = module.analysis_cfg.cache_dict
    else:
        raise ValueError(
            "No cache available: neither analysis_batch.grad_cache nor module.analysis_cfg.cache_dict is set"
        )

    # Wrap raw dicts into a backend-specific activation cache; already-wrapped caches pass through
    model_backend = require_model_backend(module)
    batch_cache_dict = model_backend.wrap_activation_cache(cache_source, module.model)
    batch_sz = get_batch_input(batch).size(0)  # type: ignore[attr-defined]  # BatchEncoding tensor has size

    # Get alive latents using GetAliveLatentsOp  # TODO: clean this up so no temp batch is required
    # Create a temporary analysis batch with the cache for GetAliveLatentsOp
    temp_batch = AnalysisBatch(cache=batch_cache_dict, answer_indices=analysis_batch.answer_indices)

    # Declared required_op invoked via the public op surface
    # (see NOTE [Op-Driven Transitive Dependency Atomicity])
    # TODO: refactor this to use the GetAliveLatentsOp? (which should then dispatch alive_latents implementation)
    import interpretune as it

    temp_batch = it.get_alive_latents(module, temp_batch, batch, batch_idx)  # type: ignore[arg-type]
    analysis_batch.alive_latents = temp_batch.alive_latents
    assert analysis_batch.alive_latents is not None, "alive_latents should be set after get_alive_latents call"

    # Compute attribution values and correct activations
    attribution_values: dict[str, torch.Tensor] = {}
    correct_activations: dict[str, torch.Tensor] = {}

    # Process each forward hook
    for fwd_name in [
        name
        for name in batch_cache_dict.keys()
        if module.analysis_cfg.names_filter(name) and not name.endswith("_grad")
    ]:
        # Check if we have gradient information for this hook
        grad_name = f"{fwd_name}_grad"
        if grad_name not in batch_cache_dict:
            continue

        # Initialize attribution tensor
        attribution_values[fwd_name] = torch.zeros(batch_sz, module.sae_handles[0].cfg.d_sae)

        # Get activations and gradients at the answer indices
        fwd_hook_acts = batch_cache_dict[fwd_name][torch.arange(batch_sz), analysis_batch.answer_indices]
        bwd_hook_grads = batch_cache_dict[grad_name][torch.arange(batch_sz), analysis_batch.answer_indices]

        # Ensure tensors have the right shape (add batch dimension if needed)
        for t in [fwd_hook_acts, bwd_hook_grads]:
            if t.dim() == 2:
                t.unsqueeze_(1)

        # Extract correct activations (for examples with positive logit differences)
        correct_activations[fwd_name] = torch.squeeze(fwd_hook_acts[(analysis_batch.logit_diffs > 0), :, :], dim=1)

        # Calculate attribution as activations × gradients for the alive latents
        alive_indices = analysis_batch.alive_latents[fwd_name]
        attribution_values[fwd_name][:, alive_indices] = torch.squeeze(
            (bwd_hook_grads[:, :, alive_indices] * fwd_hook_acts[:, :, alive_indices]).cpu(), dim=1
        )

    # Update the analysis batch with results
    analysis_batch.update(attribution_values=attribution_values, correct_activations=correct_activations)

    return analysis_batch


def ablation_attribution_impl(
    module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
    get_loss_preds_diffs: Callable = get_loss_preds_diffs,
) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing attribution values using latent ablation."""
    # Ensure we have required inputs
    required_inputs = ["answer_logits", "alive_latents", "logit_diffs"]
    for key in required_inputs:
        if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
            raise ValueError(f"Missing required input '{key}' for ablation attribution")

    # Initialize result structures
    attribution_values: dict[str, torch.Tensor] = {}
    per_latent = {
        "loss": defaultdict(dict),
        "logit_diffs": defaultdict(dict),
        "preds": defaultdict(dict),
        "answer_logits": defaultdict(dict),
    }

    # Process per-latent logits for each hook
    assert analysis_batch.answer_logits is not None and analysis_batch.alive_latents is not None, (
        "Missing required attributes in analysis_batch"
    )
    assert isinstance(analysis_batch.answer_logits, dict), "Expected answer_logits to be a dictionary"
    for act_name, logits in analysis_batch.answer_logits.items():
        attribution_values[act_name] = torch.zeros(get_batch_input(batch).size(0), module.sae_handles[0].cfg.d_sae)  # type: ignore[attr-defined]
        for latent_idx in analysis_batch.alive_latents[act_name]:
            # Calculate metrics for this latent using the instance's get_loss_preds_diffs method
            loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(
                module, analysis_batch, logits[latent_idx], logit_diff_fn
            )

            # Store per-latent metrics
            for metric_name, value in zip(per_latent.keys(), (loss, logit_diffs, preds, answer_logits)):
                per_latent[metric_name][act_name][latent_idx] = value

            # Calculate attribution values
            example_mask = (per_latent["logit_diffs"][act_name][latent_idx] > 0).cpu()
            per_latent["logit_diffs"][act_name][latent_idx] = (
                per_latent["logit_diffs"][act_name][latent_idx][example_mask].detach().cpu()
            )

            base_diffs = analysis_batch.logit_diffs
            assert base_diffs is not None, "Expected logit_diffs to be present in analysis_batch"
            assert isinstance(base_diffs, torch.Tensor), "Expected logit_diffs to be tensor at this point"
            for t in [example_mask, base_diffs]:
                if t.dim() == 0:
                    t.unsqueeze_(0)
            base_diffs = base_diffs.cpu()

            # Attribution is difference between base and ablated performance
            attribution_values[act_name][example_mask, latent_idx] = (
                base_diffs[example_mask] - per_latent["logit_diffs"][act_name][latent_idx]
            )

    # Update analysis batch with results
    for key in per_latent:
        analysis_batch.update(**{key: per_latent[key]})
    analysis_batch.update(attribution_values=attribution_values)

    return analysis_batch
