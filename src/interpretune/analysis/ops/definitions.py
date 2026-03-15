"""Definitions of specific analysis operations."""

from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Callable, Literal, Any, TYPE_CHECKING
from collections import defaultdict
from functools import partial
import json

import torch
from transformers import BatchEncoding
from jaxtyping import Float

if TYPE_CHECKING:
    from transformer_lens.hook_points import HookPoint

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.base import get_batch_input
from interpretune.analysis.backends import require_analysis_backend
from interpretune.protocol import DefaultAnalysisBatchProtocol
import interpretune as it


def boolean_logits_to_avg_logit_diff(
    logits: Float[torch.Tensor, "batch seq 2"],  # type: ignore
    target_indices: torch.Tensor,
    reduction: Literal["mean", "sum"] | None = None,
) -> torch.Tensor:
    """Returns the avg logit diff on a set of prompts, with fixed s2 pos and stuff."""
    incorrect_indices = 1 - target_indices
    correct_logits = torch.gather(logits, 2, torch.reshape(target_indices, (-1, 1, 1))).squeeze()
    incorrect_logits = torch.gather(logits, 2, torch.reshape(incorrect_indices, (-1, 1, 1))).squeeze()
    logit_diff = correct_logits - incorrect_logits
    if reduction is not None:
        logit_diff = logit_diff.mean() if reduction == "mean" else logit_diff.sum()
    return logit_diff


def get_loss_preds_diffs(
    module: torch.nn.Module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    answer_logits: torch.Tensor,
    logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation for computing loss, predictions, and logit differences.

    Args:
        module: The module containing loss_fn and standardize_logits methods
        analysis_batch: The analysis batch containing labels and orig_labels
        answer_logits: The logits to analyze
        logit_diff_fn: Function to compute logit differences

    Returns:
        Tuple of (loss, logit_diffs, preds, answer_logits)
    """
    loss = module.loss_fn(answer_logits, analysis_batch.label_ids)  # type: ignore[attr-defined]
    answer_logits = module.standardize_logits(answer_logits)  # type: ignore[attr-defined]
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


def _extract_logits(output: Any) -> torch.Tensor:
    """Extract logits tensor from a model forward-pass result.

    TransformerLens models may return raw logit tensors, while HuggingFace models
    (used by the NNsight backend) return ``ModelOutput`` objects with a
    ``.logits`` attribute.  This helper normalizes both cases.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"Cannot extract logits from model output of type {type(output).__name__}")


def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        return logits.detach().cpu()
    if logits.dim() == 2:
        return logits[-1].detach().cpu()
    if logits.dim() >= 3:
        return logits[0, -1].detach().cpu()
    raise ValueError(f"Unsupported logits rank for feature intervention output: {logits.dim()}")


def _mean_target_logit_delta(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_ids: torch.Tensor | None,
) -> torch.Tensor:
    if target_ids is not None and torch.numel(target_ids) > 0:
        target_ids = target_ids.to(dtype=torch.long).reshape(-1)
        return (post_logits.index_select(0, target_ids) - pre_logits.index_select(0, target_ids)).mean()
    return (post_logits - pre_logits).mean()


def model_forward_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for basic model forward pass."""
    # Ensure we have answer indices
    if not hasattr(analysis_batch, "answer_indices") or analysis_batch.answer_indices is None:
        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    # Run forward pass
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")
    _backend = getattr(module, "_model_backend", None)
    if _backend is not None:
        answer_logits = _backend.fwd(model=module.model, batch=batch)
    else:
        answer_logits = _extract_logits(module(**batch))

    analysis_batch.update(answer_logits=answer_logits)
    return analysis_batch


def model_cache_forward_impl(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Implementation for forward pass with cache."""
    # Run with cache and SAEs
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")
    answer_logits, cache = module.model_backend.fwd_w_cache_and_latent_models(
        model=module.model,
        batch=batch,
        latent_model_handles=module.sae_handles,
        names_filter=module.analysis_cfg.names_filter,
    )

    # Get answer indices and alive latents
    analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)
    analysis_batch.update(cache=cache)
    # See NOTE [Op-Driven Transitive Dependency Atomicity]
    analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)  # type: ignore[call-arg]

    analysis_batch.update(answer_logits=answer_logits)
    return analysis_batch


def model_ablation_impl(
    module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    batch_idx: int,
    ablate_latent_fn: Callable = ablate_sae_latent,
) -> DefaultAnalysisBatchProtocol:
    """Implementation for model ablation analysis."""
    # Ensure we have answer indices and alive latents
    if not hasattr(analysis_batch, "answer_indices") or analysis_batch.answer_indices is None:
        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    if not hasattr(analysis_batch, "alive_latents") or analysis_batch.alive_latents is None:
        # TODO: remove this leaky abstraction, alive_latents should only be in analysis_batch
        assert module.analysis_cfg.input_store and getattr(module.analysis_cfg.input_store, "alive_latents", None), (
            "alive_latents required for ablation op"
        )
        # See NOTE [Op-Driven Transitive Dependency Atomicity]
        analysis_batch = it.get_alive_latents(module, analysis_batch, batch, batch_idx)  # type: ignore[call-arg]

    answer_indices = analysis_batch.answer_indices
    alive_latents = analysis_batch.alive_latents

    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")

    # Build hook configs for every (name, latent_idx) pair, then run them in batch.
    per_latent_logits: dict[str, dict[Any, torch.Tensor]] = defaultdict(dict)
    assert alive_latents is not None and isinstance(alive_latents, dict), "alive_latents must be a dict"

    hook_configs: list[list[tuple[str, Any]]] = []
    index_map: list[tuple[str, Any]] = []  # parallel list: (name, latent_idx) per config
    for name, alive in alive_latents.items():
        for latent_idx in alive:
            hook_configs.append([(name, partial(ablate_latent_fn, latent_idx=latent_idx, seq_pos=answer_indices))])
            index_map.append((name, latent_idx))

    all_logits = module.model_backend.fwd_w_hooks_batched(
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
        analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)

    answer_indices = analysis_batch.answer_indices

    # if we're running a manual analysis_step context, we may need to manually set hooks
    module.analysis_cfg.add_default_cache_hooks()
    # Verify hooks are configured
    assert all((module.analysis_cfg.fwd_hooks, module.analysis_cfg.bwd_hooks)), (
        "fwd_hooks and bwd_hooks required for gradient-based attribution op"
    )

    # TODO: In the future, we will likely use IT dispatch logic to control toggling autograd/inference mode etc.
    #       but for now controlling manually here
    if module.analysis_cfg.auto_prune_batch_encoding and isinstance(batch, BatchEncoding):
        batch = module.auto_prune_batch(batch, "forward")

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
    raw_logits = module.model_backend.fwd_w_grads_and_latent_models(
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
        # See NOTE [Op-Driven Transitive Dependency Atomicity]
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
    batch_cache_dict = module.model_backend.wrap_activation_cache(cache_source, module.model)
    batch_sz = get_batch_input(batch).size(0)  # type: ignore[attr-defined]  # BatchEncoding tensor has size

    # Get alive latents using GetAliveLatentsOp  # TODO: clean this up so no temp batch is required
    # Create a temporary analysis batch with the cache for GetAliveLatentsOp
    temp_batch = AnalysisBatch(cache=batch_cache_dict, answer_indices=analysis_batch.answer_indices)

    # TODO: refactor this to use the GetAliveLatentsOp? (which should then dispatch alive_latents implementation)
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


def concept_direction_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Compute a normalized embedding direction between two concept groups."""
    analysis_backend = require_analysis_backend(module)
    tokenizer = analysis_backend.get_tokenizer(module)
    embed_weight = analysis_backend.get_embedding_weight(module)
    group_a = list(getattr(analysis_batch, "concept_group_a", []) or [])
    group_b = list(getattr(analysis_batch, "concept_group_b", []) or [])
    if not group_a or not group_b:
        raise ValueError("concept_direction requires non-empty concept_group_a and concept_group_b")

    group_a_ids = analysis_backend.token_strings_to_ids(tokenizer, group_a)
    group_b_ids = analysis_backend.token_strings_to_ids(tokenizer, group_b)
    group_a_embed = embed_weight[torch.tensor(group_a_ids, device=embed_weight.device)].float().mean(dim=0)
    group_b_embed = embed_weight[torch.tensor(group_b_ids, device=embed_weight.device)].float().mean(dim=0)
    direction_vector = group_a_embed - group_b_embed
    direction_norm = torch.linalg.vector_norm(direction_vector)
    if torch.isfinite(direction_norm) and direction_norm.item() > 0:
        direction_vector = direction_vector / direction_norm

    analysis_batch.update(
        concept_direction=direction_vector.detach().cpu(),
        concept_label=f"{' / '.join(group_a)} -> {' / '.join(group_b)}",
        concept_metadata=json.dumps(
            {
                "group_a": group_a,
                "group_b": group_b,
                "group_a_token_ids": group_a_ids,
                "group_b_token_ids": group_b_ids,
            }
        ),
    )
    return analysis_batch


def compute_attribution_graph_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Generate and decompose a circuit-tracer attribution graph."""
    analysis_backend = require_analysis_backend(module)
    prompt = kwargs.pop("prompt", None) or analysis_backend.resolve_prompt(module, analysis_batch, batch)
    graph = module.generate_attribution_graph(prompt, **kwargs)
    analysis_batch.update(**analysis_backend.decompose_graph(graph, extra_metadata={"batch_idx": batch_idx}))
    return analysis_batch


def extract_top_features_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Extract the top scoring features from analysis-batch feature rows."""
    active_features = torch.as_tensor(getattr(analysis_batch, "active_features", []), dtype=torch.long)
    selected_features = torch.as_tensor(getattr(analysis_batch, "selected_features", []), dtype=torch.long)
    if active_features.numel() == 0:
        analysis_batch.update(
            top_feature_ids=torch.empty((0, 3), dtype=torch.long),
            top_feature_scores=torch.empty((0,), dtype=torch.float32),
        )
        return analysis_batch
    active_features = active_features.reshape(-1, 3)

    score_values = getattr(analysis_batch, "node_influence_scores", None)
    if score_values is None:
        score_values = getattr(analysis_batch, "activation_values", None)
    scores = torch.as_tensor(score_values, dtype=torch.float32)
    if scores.dim() > 1:
        scores = scores.reshape(-1)

    feature_rows = active_features
    if selected_features.numel() > 0 and selected_features.shape[0] == scores.shape[0]:
        feature_rows = require_analysis_backend(module).select_feature_rows(active_features, selected_features)
    elif active_features.shape[0] != scores.shape[0]:
        raise ValueError(
            "extract_top_features requires active_features to match score length directly or via selected_features"
        )

    top_n = min(int(kwargs.get("top_n", scores.shape[0])), scores.shape[0])
    top_indices = torch.argsort(scores, descending=True)[:top_n]
    analysis_batch.update(
        top_feature_ids=feature_rows.index_select(0, top_indices).detach().cpu(),
        top_feature_scores=scores.index_select(0, top_indices).detach().cpu(),
    )
    return analysis_batch


def graph_prune_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Prune a structured circuit-tracer graph and refresh decomposed outputs."""
    analysis_backend = require_analysis_backend(module)
    graph = analysis_backend.hydrate_graph_from_batch(analysis_batch)
    pruned_graph = analysis_backend.build_pruned_graph(
        graph,
        node_threshold=float(kwargs.get("node_threshold", 0.8)),
        edge_threshold=float(kwargs.get("edge_threshold", 0.98)),
    )
    analysis_batch.update(
        **analysis_backend.decompose_graph(pruned_graph, extra_metadata={"batch_idx": batch_idx, "pruned": True})
    )
    return analysis_batch


def graph_node_influence_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Compute feature-node influence scores from a structured graph."""
    analysis_backend = require_analysis_backend(module)
    graph = analysis_backend.hydrate_graph_from_batch(analysis_batch)
    node_scores, node_feature_ids = analysis_backend.compute_node_influence_scores(graph)
    analysis_batch.update(
        node_influence_scores=node_scores,
        node_feature_ids=node_feature_ids,
    )
    return analysis_batch


def feature_intervention_forward_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Run circuit-tracer feature interventions against the module replacement model.

    This op currently implements forward-only intervention analysis and stores an Arrow-safe summary of the intervention
    tuples so downstream AnalysisStore consumers can inspect or rehydrate the canonical intervention list.
    """
    analysis_backend = require_analysis_backend(module)

    replacement_model = getattr(module, "replacement_model", None)
    if replacement_model is None:
        raise ValueError("feature_intervention_forward requires module.replacement_model")

    prompt = kwargs.pop("prompt", None) or analysis_backend.resolve_prompt(module, analysis_batch, batch)
    settings = analysis_backend.resolve_feature_intervention_settings(module, kwargs)
    interventions, intervention_payload = analysis_backend.build_feature_interventions(analysis_batch, settings)

    pre_logits_raw, _ = replacement_model.get_activations(prompt)
    pre_logits = _last_token_logits(pre_logits_raw)

    if interventions:
        post_logits_raw, _ = replacement_model.feature_intervention(
            prompt,
            interventions,
            **analysis_backend.feature_intervention_call_kwargs(settings),
        )
        post_logits = _last_token_logits(post_logits_raw)
    else:
        post_logits = pre_logits.clone()

    target_ids = getattr(analysis_batch, "logit_target_ids", None)
    target_ids_tensor = None if target_ids is None else torch.as_tensor(target_ids, dtype=torch.long).reshape(-1)
    logit_diff = _mean_target_logit_delta(pre_logits, post_logits, target_ids_tensor)

    analysis_batch.update(
        **intervention_payload,
        pre_intervention_logits=pre_logits,
        post_intervention_logits=post_logits,
        logit_diff=logit_diff.detach().cpu(),
    )
    return analysis_batch
