"""Definitions of specific analysis operations."""
from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Callable, Literal, Any
from collections import defaultdict
from functools import partial

import torch
from transformers import BatchEncoding
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float

import interpretune as it
from interpretune.protocol import DefaultAnalysisBatchProtocol
from interpretune.analysis.ops.base import AnalysisBatch


def boolean_logits_to_avg_logit_diff(
    logits: Float[torch.Tensor, "batch seq 2"],
    target_indices: torch.Tensor,
    reduction: Literal["mean", "sum"] | None = None,
    keep_as_tensor: bool = True,
) -> list[float] | float:
    """Returns the avg logit diff on a set of prompts, with fixed s2 pos and stuff."""
    incorrect_indices = 1 - target_indices
    correct_logits = torch.gather(logits, 2, torch.reshape(target_indices, (-1,1,1))).squeeze()
    incorrect_logits = torch.gather(logits, 2, torch.reshape(incorrect_indices, (-1,1,1))).squeeze()
    logit_diff = correct_logits - incorrect_logits
    if reduction is not None:
        logit_diff = logit_diff.mean() if reduction == "mean" else logit_diff.sum()
    return logit_diff if keep_as_tensor else logit_diff.tolist()


def get_loss_preds_diffs(module: torch.nn.Module,
                         analysis_batch: DefaultAnalysisBatchProtocol,
                         answer_logits: torch.Tensor,
                         logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff) -> tuple[
                             torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation for computing loss, predictions, and logit differences.

    Args:
        module: The module containing loss_fn and standardize_logits methods
        analysis_batch: The analysis batch containing labels and orig_labels
        answer_logits: The logits to analyze
        logit_diff_fn: Function to compute logit differences

    Returns:
        Tuple of (loss, logit_diffs, preds, answer_logits)
    """
    loss = module.loss_fn(answer_logits, analysis_batch.label_ids)
    answer_logits = module.standardize_logits(answer_logits)
    per_example_answers, _ = torch.max(answer_logits, dim=-2)
    preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
    logit_diffs = logit_diff_fn(answer_logits, target_indices=analysis_batch.orig_labels)
    return loss, logit_diffs, preds, answer_logits


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


def labels_to_ids_impl(module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding,
                       batch_idx: int) -> DefaultAnalysisBatchProtocol:
    """Implementation for converting string labels to tensor IDs."""
    if "labels" in batch:
        label_ids, orig_labels = module.labels_to_ids(batch.pop("labels"))
        analysis_batch.update(label_ids=label_ids, orig_labels=orig_labels)
    return analysis_batch


def get_answer_indices_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                          batch: BatchEncoding, batch_idx: int) -> DefaultAnalysisBatchProtocol:
    """Implementation for extracting answer indices from batch."""

    # Check if answer_indices already exist
    if hasattr(analysis_batch, 'answer_indices') and analysis_batch.answer_indices is not None:
        return analysis_batch

    # Check if we can get from input store
    if module.analysis_cfg.input_store and getattr(module.analysis_cfg.input_store, 'answer_indices', None) is not None:
        answer_indices = module.analysis_cfg.input_store.answer_indices[batch_idx]
    else:
        # Otherwise compute it
        tokens = batch["input"].detach().cpu()
        if module.datamodule.tokenizer.padding_side == "left":
            answer_indices = torch.full((tokens.size(0),), -1)
        else:
            nonpadding_mask = tokens != module.datamodule.tokenizer.pad_token_id
            # This could be more robust, test with various datasets and padding strategies
            answer_indices = torch.where(nonpadding_mask, 1, 0).sum(dim=1) - 1

    analysis_batch.update(answer_indices=answer_indices)
    return analysis_batch


def get_alive_latents_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                           batch: BatchEncoding, batch_idx: int) -> DefaultAnalysisBatchProtocol:
    """Implementation for extracting alive latents from cache."""
    # Check if alive_latents already exist
    if hasattr(analysis_batch, 'alive_latents') and analysis_batch.alive_latents is not None:
        return analysis_batch

    # Check if we can get from input store
    # TODO: remove this leaky abstraction, alive_latents should only be in analysis_batch, not accessed
    #       via analysis_cfg.input_store at the op level
    if module.analysis_cfg.input_store and module.analysis_cfg.input_store.alive_latents is not None:
        alive_latents = module.analysis_cfg.input_store.alive_latents[batch_idx]
    elif not hasattr(analysis_batch, 'cache') or analysis_batch.cache is None:
        alive_latents = {}
    else:
        # Extract alive latents from the cache using the answer indices
        cache = analysis_batch.cache
        names_filter = module.analysis_cfg.names_filter
        answer_indices = analysis_batch.answer_indices

        filtered_acts = {name: acts for name, acts in cache.items() if names_filter(name)}
        alive_latents = {}
        for name, acts in filtered_acts.items():
            alive = (acts[torch.arange(acts.size(0)),
                          answer_indices, :] > 0).any(dim=0).nonzero().squeeze().tolist()
            alive_latents[name] = alive

    analysis_batch.update(alive_latents=alive_latents)
    return analysis_batch


def model_forward_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                     batch: BatchEncoding, batch_idx: int) -> DefaultAnalysisBatchProtocol:
    """Implementation for basic model forward pass."""
    # Ensure we have answer indices
    if not hasattr(analysis_batch, 'answer_indices') or analysis_batch.answer_indices is None:
        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    # Run forward pass
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, 'forward')
    answer_logits = module(**batch)
    analysis_batch.update(answer_logits=answer_logits)
    return analysis_batch


def model_cache_forward_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                           batch: BatchEncoding, batch_idx: int) -> DefaultAnalysisBatchProtocol:
    """Implementation for forward pass with cache."""
    # Run with cache and SAEs
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, 'forward')
    answer_logits, cache = module.model.run_with_cache_with_saes(
        **batch, saes=module.sae_handles,
        names_filter=module.analysis_cfg.names_filter
    )

    # Get answer indices and alive latents
    analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)
    analysis_batch.update(cache=cache)
    analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)

    analysis_batch.update(answer_logits=answer_logits)
    return analysis_batch


def model_ablation_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                      batch: BatchEncoding, batch_idx: int,
                      ablate_latent_fn: Callable = ablate_sae_latent) -> DefaultAnalysisBatchProtocol:
    """Implementation for model ablation analysis."""
    # Ensure we have answer indices and alive latents
    if not hasattr(analysis_batch, 'answer_indices') or analysis_batch.answer_indices is None:
        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    if not hasattr(analysis_batch, 'alive_latents') or analysis_batch.alive_latents is None:
        # TODO: remove this leaky abstraction, alive_latents should only be in analysis_batch
        assert module.analysis_cfg.input_store and getattr(module.analysis_cfg.input_store, 'alive_latents', None), \
            "alive_latents required for ablation op"
        analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)

    answer_indices = analysis_batch.answer_indices
    alive_latents = analysis_batch.alive_latents

    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, 'forward')

    # Run ablation for each latent
    per_latent_logits: dict[str, dict[Any, torch.Tensor]] = defaultdict(dict)
    for name, alive in alive_latents.items():
        for latent_idx in alive:
            fwd_hooks_cfg = [(name, partial(ablate_latent_fn,
                                            latent_idx=latent_idx,
                                            seq_pos=answer_indices))]
            answer_logits = module.model.run_with_hooks_with_saes(
                **batch, saes=module.sae_handles,
                clear_contexts=True, fwd_hooks=fwd_hooks_cfg
            )
            per_latent_logits[name][latent_idx] = answer_logits[
                torch.arange(batch["input"].size(0)), answer_indices, :
            ]

    analysis_batch.update(answer_logits=per_latent_logits)
    return analysis_batch


def model_gradient_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                      batch: BatchEncoding, batch_idx: int,
                      logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
                      get_loss_preds_diffs: Callable = get_loss_preds_diffs) -> DefaultAnalysisBatchProtocol:
    """Implementation for gradient-based attribution."""

    # Ensure we have answer indices
    if not hasattr(analysis_batch, 'answer_indices') or analysis_batch.answer_indices is None:
        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    answer_indices = analysis_batch.answer_indices

    # if we're running a manual analysis_step context, we may need to manually set hooks
    module.analysis_cfg.add_default_cache_hooks()
    # Verify hooks are configured
    assert all((module.analysis_cfg.fwd_hooks, module.analysis_cfg.bwd_hooks)), \
        "fwd_hooks and bwd_hooks required for gradient-based attribution op"

    # TODO: In the future, we will likely use IT dispatch logic to control toggling autograd/inference mode etc.
    #       but for now controlling manually here
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, 'forward')
    # Run with hooks and compute gradients
    with torch.set_grad_enabled(True):
        with module.model.saes(saes=module.sae_handles):
            with module.model.hooks(
                fwd_hooks=module.analysis_cfg.fwd_hooks, bwd_hooks=module.analysis_cfg.bwd_hooks
            ):
                answer_logits = module.model(**batch)
                answer_logits = torch.squeeze(
                    answer_logits[torch.arange(batch["input"].size(0)), answer_indices],
                    dim=1
                )
                # Compute loss and logit differences using the instance's get_loss_preds_diffs method
                loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(
                    module, analysis_batch, answer_logits, logit_diff_fn
                )

                # Compute gradients
                logit_diffs.sum().backward()
                if logit_diffs.dim() == 0:
                    logit_diffs.unsqueeze_(0)
    analysis_batch.update(
        answer_logits=answer_logits,
        answer_indices=answer_indices,
        logit_diffs=logit_diffs,
        preds=preds,
        loss=loss,
        grad_cache=module.analysis_cfg.cache_dict  # Store the gradient cache
    )
    return analysis_batch


def logit_diffs_impl(module: torch.nn.Module, analysis_batch: DefaultAnalysisBatchProtocol,
                   batch: BatchEncoding, batch_idx: int,
                   logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
                   get_loss_preds_diffs: Callable = get_loss_preds_diffs) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing logit differences."""

    logits, indices = analysis_batch.answer_logits, analysis_batch.answer_indices
    answer_logits = torch.squeeze(logits[torch.arange(batch["input"].size(0)), indices], dim=1)
    loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(
        module, analysis_batch, answer_logits, logit_diff_fn
    )
    if logit_diffs.dim() == 0:
        logit_diffs.unsqueeze_(0)
    analysis_batch.update(loss=loss, logit_diffs=logit_diffs, preds=preds, answer_logits=answer_logits)
    return analysis_batch


def sae_correct_acts_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                        batch: BatchEncoding, batch_idx: int) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing correct activations from SAE outputs."""
    # Validate required inputs # TODO: refactor all required input checks to use shared AnalysisOp or Dispatcher logic
    required_inputs = ['logit_diffs', 'answer_indices', 'cache']
    for key in required_inputs:
        if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
            raise ValueError(f"Missing required input '{key}' for {module.__class__.__name__}.sae_correct_acts")

    # Extract required data from analysis_batch
    cache = analysis_batch.cache
    logit_diffs = analysis_batch.logit_diffs
    answer_indices = analysis_batch.answer_indices

    # Ensure alive_latents are present
    if not hasattr(analysis_batch, 'alive_latents') or analysis_batch.alive_latents is None:
        analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)

    # Extract correct activations for examples with positive logit differences
    correct_mask = logit_diffs > 0
    # Handle scalar case
    if correct_mask.dim() == 0:
        correct_mask = correct_mask.unsqueeze(0)
    if logit_diffs.dim() == 0:
        logit_diffs = logit_diffs.unsqueeze(0)

    correct_activations = {}
    names_filter = module.analysis_cfg.names_filter
    for name, acts in cache.items():
        if not names_filter(name):
            continue

        # Get activations at answer indices and select only for examples with positive logit diffs
        acts_at_answer = acts[torch.arange(acts.size(0)), answer_indices]
        correct_activations[name] = acts_at_answer[correct_mask].cpu()

    analysis_batch.update(correct_activations=correct_activations)
    return analysis_batch


def gradient_attribution_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                            batch: BatchEncoding, batch_idx: int) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing attribution values from gradients."""
    # TODO: change this to use shared superclass required input validation
    # Ensure required inputs exist
    required_inputs = ['answer_indices', 'logit_diffs']
    for key in required_inputs:
        if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
            raise ValueError(f"Missing required input '{key}' for gradient attribution")

    # TODO: switch to using grad_cache from analysis_batch once that functionality is implemented
    # Get cached activations (forwards) and gradients (backwards) from analysis_cfg.cache_dict
    from transformer_lens import ActivationCache
    # Prefer grad_cache on the analysis_batch, else fall back to module.analysis_cfg.cache_dict
    if getattr(analysis_batch, "grad_cache", None) is not None:
        cache_source = analysis_batch.grad_cache
    elif getattr(module.analysis_cfg, "cache_dict", None) is not None:
        cache_source = module.analysis_cfg.cache_dict
    else:
        raise ValueError(
            "No cache available: neither analysis_batch.grad_cache nor module.analysis_cfg.cache_dict is set"
        )

    # If it's not already an ActivationCache, wrap it
    if isinstance(cache_source, ActivationCache):
        batch_cache_dict = cache_source
    else:
        batch_cache_dict = ActivationCache(cache_source, module.model)
    batch_sz = batch["input"].size(0)

    # Get alive latents using GetAliveLatentsOp  # TODO: clean this up so no temp batch is required
    # Create a temporary analysis batch with the cache for GetAliveLatentsOp
    temp_batch = AnalysisBatch(
        cache=batch_cache_dict,
        answer_indices=analysis_batch.answer_indices
    )

    # TODO: refactor this to use the GetAliveLatentsOp? (which should then dispatch alive_latents implementation)
    temp_batch = it.get_alive_latents(module, temp_batch, batch, batch_idx)
    analysis_batch.alive_latents = temp_batch.alive_latents

    # Compute attribution values and correct activations
    attribution_values: dict[str, torch.Tensor] = {}
    correct_activations: dict[str, torch.Tensor] = {}

    # Process each forward hook
    for fwd_name in [name for name in batch_cache_dict.keys()
                     if module.analysis_cfg.names_filter(name) and not name.endswith("_grad")]:
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
        correct_activations[fwd_name] = torch.squeeze(
            fwd_hook_acts[(analysis_batch.logit_diffs > 0), :, :], dim=1
        )

        # Calculate attribution as activations × gradients for the alive latents
        alive_indices = analysis_batch.alive_latents[fwd_name]
        attribution_values[fwd_name][:, alive_indices] = torch.squeeze(
            (bwd_hook_grads[:, :, alive_indices] *
             fwd_hook_acts[:, :, alive_indices]).cpu(), dim=1
        )

    # Update the analysis batch with results
    analysis_batch.update(
        attribution_values=attribution_values,
        correct_activations=correct_activations
    )

    return analysis_batch


def ablation_attribution_impl(module, analysis_batch: DefaultAnalysisBatchProtocol,
                            batch: BatchEncoding, batch_idx: int,
                            logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
                            get_loss_preds_diffs: Callable = get_loss_preds_diffs) -> DefaultAnalysisBatchProtocol:
    """Implementation for computing attribution values using latent ablation."""
    # Ensure we have required inputs
    required_inputs = ['answer_logits', 'alive_latents', 'logit_diffs']
    for key in required_inputs:
        if not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
            raise ValueError(f"Missing required input '{key}' for ablation attribution")

    # Initialize result structures
    attribution_values: dict[str, torch.Tensor] = {}
    per_latent = {"loss": defaultdict(dict), "logit_diffs": defaultdict(dict),
                  "preds": defaultdict(dict), "answer_logits": defaultdict(dict)}

    # Process per-latent logits for each hook
    for act_name, logits in analysis_batch.answer_logits.items():
        attribution_values[act_name] = torch.zeros(batch["input"].size(0), module.sae_handles[0].cfg.d_sae)
        for latent_idx in analysis_batch.alive_latents[act_name]:
            # Calculate metrics for this latent using the instance's get_loss_preds_diffs method
            loss, logit_diffs, preds, answer_logits = get_loss_preds_diffs(
                module, analysis_batch, logits[latent_idx], logit_diff_fn
            )

            # Store per-latent metrics
            for metric_name, value in zip(
                per_latent.keys(),
                (loss, logit_diffs, preds, answer_logits)
            ):
                per_latent[metric_name][act_name][latent_idx] = value

            # Calculate attribution values
            example_mask = (per_latent["logit_diffs"][act_name][latent_idx] > 0).cpu()
            per_latent["logit_diffs"][act_name][latent_idx] = (
                per_latent["logit_diffs"][act_name][latent_idx][example_mask].detach().cpu()
            )

            base_diffs = analysis_batch.logit_diffs
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
