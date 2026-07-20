"""Definitions of specific analysis operations."""

from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL

from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch
from jaxtyping import Float
from transformers import BatchEncoding

if TYPE_CHECKING:
    from transformer_lens.hook_points import HookPoint

from interpretune.analysis.ops.base import AnalysisBatch, get_batch_input
from interpretune.analysis.backends import (
    FeatureSelectionSpec,
    require_analysis_backend,
    resolve_interventions,
)
from interpretune.analysis.ops.helpers import (
    _apply_optional_feature_sign_filter,
    _augment_feature_rows_for_selection,
    _extract_concept_latent_state_from_cache,
    _flatten_concept_store_rows,
    _resolve_concept_cache_key,
    _select_top_feature_indices,
    apply_feature_selection_filter,
    extract_logits,
    last_token_logits,
    load_json_field,
    mean_target_logit_delta,
    require_model_backend,
    resolve_feature_score_source,
    resolve_embedding_weight,
    resolve_aggregate_input,
    resolve_tokenizer,
    token_strings_to_last_ids,
    weighted_mean,
)
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


def extract_concept_latent_state_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Extract per-example latent rows from the configured cache key for downstream concept-direction ops."""
    context_enhanced = bool(kwargs.get("context_enhanced", False))
    context_scale = float(kwargs.get("context_scale", 1.0))
    use_answer_state_as_basis = bool(kwargs.get("use_answer_state_as_basis", False))

    latent_states, cache_key = _extract_concept_latent_state_from_cache(
        analysis_batch,
        context_enhanced=context_enhanced,
        context_scale=context_scale,
        use_answer_state_as_basis=use_answer_state_as_basis,
    )
    update_kwargs: dict[str, Any] = {
        "concept_latent_state": latent_states,
        "concept_cache_key": cache_key,
        "use_answer_state_as_basis": use_answer_state_as_basis,
    }
    raw_context_token_indices = analysis_batch.get("context_token_indices")
    if raw_context_token_indices is not None:
        update_kwargs["context_token_indices"] = (
            torch.as_tensor(raw_context_token_indices, dtype=torch.long).reshape(-1).detach().cpu()
        )
    analysis_batch.update(**update_kwargs)
    return analysis_batch


def extract_concept_latent_examples_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Filter and annotate concept latent rows for downstream concept-direction aggregation.

    Consumes ``concept_latent_state`` rows produced by the upstream ``extract_concept_latent_state`` op.
    That op must run first to populate ``concept_latent_state`` on the batch.

    Aggregation modes (selected via ``concept_aggregate_output_mode`` on the batch):

    * ``"streaming"`` (default): emit only per-batch tensors. Cross-batch aggregation is performed
      incrementally inside :func:`concept_direction_impl` using running per-group weighted sums
      stored on ``analysis_inputs.store``. This avoids materializing the full latent-row payload
      and keeps per-batch payload sizes constant. Recommended for all new callers and any pipeline
      where the full set of selected examples does not need to be retained for later inspection.
    * ``"in_memory"`` (legacy): accumulate the full per-batch row collections on
      ``analysis_inputs.store`` (``concept_latent_state_rows``, ``concept_group_id_rows``,
      ``concept_group_name_rows``, ``concept_example_logit_diff_rows``,
      ``concept_example_weight_rows``, optionally ``concept_context_indices_rows``) and re-emit
      them on every returned batch. Each per-batch call appends to a Python list and re-binds
      it on the store and on ``analysis_batch``; the underlying tensor data is shared by reference,
      but the list overhead grows linearly per batch and the runner's per-batch payloads end up
      holding O(N\u00b2) cumulative list references for ``N`` batches. This mode remains useful for
      callers that need access to the full row collection (e.g. parity tests, pre-computed
      aggregate inputs to ``concept_direction``), but its scalability depends on store
      caching/persistence; do not use it for large concept-example sets without that backing.

    The legacy mode is preserved to keep existing tests and notebook diagnostics that consume the
    aggregate row tensors directly working unchanged.
    """

    group_a_name = str(analysis_batch.get("concept_group_a_name") or "group_a")
    group_b_name = str(analysis_batch.get("concept_group_b_name") or "group_b")
    keep_correct_only = bool(analysis_batch.get("concept_correct_only", True))
    weight_by_logit_diff = bool(analysis_batch.get("concept_weight_by_logit_diff", False))
    aggregate_output = bool(analysis_batch.get("concept_aggregate_output", True))
    aggregate_mode = str(analysis_batch.get("concept_aggregate_output_mode", "streaming"))

    orig_labels = analysis_batch.orig_labels
    logit_diffs = analysis_batch.logit_diffs
    if orig_labels is None or logit_diffs is None:
        raise ValueError("extract_concept_latent_examples requires orig_labels and logit_diffs")

    cache_key = _resolve_concept_cache_key(analysis_batch)
    latent_states = analysis_batch.get("concept_latent_state")
    if latent_states is None:
        raise ValueError(
            "extract_concept_latent_examples requires 'concept_latent_state' on the batch. "
            "Run extract_concept_latent_state first."
        )
    latent_states = torch.as_tensor(latent_states).detach().cpu().float()

    labels = torch.as_tensor(orig_labels, dtype=torch.long).reshape(-1).detach().cpu()
    diffs = torch.as_tensor(logit_diffs, dtype=torch.float32).reshape(-1).detach().cpu()
    raw_context_token_indices = analysis_batch.get("context_token_indices")
    raw_group_a_label_ids = analysis_batch.get("concept_group_a_label_ids")
    raw_group_b_label_ids = analysis_batch.get("concept_group_b_label_ids")
    group_a_label_ids = (
        torch.empty((0,), dtype=torch.long)
        if raw_group_a_label_ids is None
        else torch.as_tensor(raw_group_a_label_ids, dtype=torch.long).reshape(-1)
    )
    group_b_label_ids = (
        torch.empty((0,), dtype=torch.long)
        if raw_group_b_label_ids is None
        else torch.as_tensor(raw_group_b_label_ids, dtype=torch.long).reshape(-1)
    )

    if latent_states.shape[0] != labels.shape[0]:
        raise ValueError(
            "extract_concept_latent_examples requires the latent rows to align with orig_labels "
            f"({latent_states.shape[0]} vs {labels.shape[0]})"
        )

    group_ids = torch.full((labels.shape[0],), -1, dtype=torch.long)
    if group_a_label_ids.numel() > 0:
        group_ids[torch.isin(labels, group_a_label_ids)] = 0
    if group_b_label_ids.numel() > 0:
        group_ids[torch.isin(labels, group_b_label_ids)] = 1

    selection_mask = group_ids >= 0
    correct_mask = diffs > 0
    if keep_correct_only:
        selection_mask &= correct_mask

    feature_shape = tuple(latent_states.shape[1:])
    empty_states = torch.empty((0, *feature_shape), dtype=latent_states.dtype)
    selected_states = latent_states[selection_mask] if selection_mask.any() else empty_states
    selected_group_ids = group_ids[selection_mask]
    selected_logit_diffs = diffs[selection_mask].detach().cpu()
    selected_context_indices: torch.Tensor | None = None
    if raw_context_token_indices is not None:
        context_token_indices = torch.as_tensor(raw_context_token_indices, dtype=torch.long).reshape(-1).detach().cpu()
        if context_token_indices.shape[0] != labels.shape[0]:
            raise ValueError(
                "extract_concept_latent_examples requires context_token_indices to align with orig_labels "
                f"({context_token_indices.shape[0]} vs {labels.shape[0]})"
            )
        selected_context_indices = (
            context_token_indices[selection_mask] if selection_mask.any() else torch.empty((0,), dtype=torch.long)
        )
    if weight_by_logit_diff:
        selected_weights = selected_logit_diffs.abs()
    else:
        selected_weights = torch.ones(selected_logit_diffs.shape, dtype=selected_logit_diffs.dtype)
    selected_group_names = [group_a_name if int(group_id) == 0 else group_b_name for group_id in selected_group_ids]

    aggregated_updates: dict[str, Any] = {}
    analysis_inputs = kwargs.get("analysis_inputs")
    store = getattr(analysis_inputs, "store", None) if analysis_inputs is not None else None
    if aggregate_output and aggregate_mode == "in_memory" and store is not None:
        aggregate_rows = (
            ("concept_latent_state_rows", selected_states),
            ("concept_group_id_rows", selected_group_ids),
            ("concept_group_name_rows", selected_group_names),
            ("concept_example_logit_diff_rows", selected_logit_diffs),
            ("concept_example_weight_rows", selected_weights),
        )
        for field_name, row_value in aggregate_rows:
            existing_rows = [] if batch_idx == 0 else list(getattr(store, field_name, []) or [])
            existing_rows.append(row_value)
            try:
                setattr(store, field_name, existing_rows)
            except Exception:
                pass
            aggregated_updates[field_name] = existing_rows
        if selected_context_indices is not None:
            context_rows = [] if batch_idx == 0 else list(getattr(store, "concept_context_indices_rows", []) or [])
            context_rows.append(selected_context_indices)
            try:
                setattr(store, "concept_context_indices_rows", context_rows)
            except Exception:
                pass
            aggregated_updates["concept_context_indices_rows"] = context_rows

    analysis_batch.update(
        concept_latent_state=selected_states,
        concept_group_id=selected_group_ids,
        concept_group_name=selected_group_names,
        concept_example_logit_diff=selected_logit_diffs,
        concept_example_weight=selected_weights,
        concept_cache_key=cache_key,
        concept_group_a_name=group_a_name,
        concept_group_b_name=group_b_name,
        concept_correct_mask=correct_mask.detach().cpu(),
        concept_context_indices=selected_context_indices,
        **aggregated_updates,
    )
    return analysis_batch


def model_fwd_impl(
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

    analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)
    analysis_batch.update(cache=cache, alive_latents={}, answer_logits=answer_logits)
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

    analysis_batch = it.get_answer_indices(module, analysis_batch, batch, batch_idx)
    analysis_batch.update(cache=cache)
    # See NOTE [Op-Driven Transitive Dependency Atomicity]
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
        # See NOTE [Op-Driven Transitive Dependency Atomicity]
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
    model_backend = require_model_backend(module)
    batch_cache_dict = model_backend.wrap_activation_cache(cache_source, module.model)
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


def _parse_streaming_per_batch_inputs(
    module,
    analysis_batch: AnalysisBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Parse per-batch concept inputs into normalized (states, group_ids, weights) cpu tensors.

    Inputs may be either a per-batch tensor (the standard streaming case) or a list of per-batch
    tensors with potentially non-uniform shapes (e.g. when concept_direction is invoked once with a
    store-backed input column). Returns ``(states, gids, weights, feature_shape)``.
    """
    per_batch_states = resolve_aggregate_input(module, analysis_batch, "concept_latent_state")
    per_batch_group_ids = resolve_aggregate_input(module, analysis_batch, "concept_group_id")
    if per_batch_states is None or per_batch_group_ids is None:
        raise ValueError(
            "concept_direction streaming mode requires per-batch concept_latent_state and concept_group_id"
        )
    per_batch_weights = resolve_aggregate_input(module, analysis_batch, "concept_example_weight")
    if isinstance(per_batch_states, (list, tuple)) and not isinstance(per_batch_states, torch.Tensor):
        states_t, gids_t, weights_t, _ = _flatten_concept_store_rows(
            per_batch_states, per_batch_group_ids, None, per_batch_weights
        )
        states_t = states_t.float().detach().cpu()
        gids_t = gids_t.long().reshape(-1).detach().cpu()
        weights_t = weights_t.float().reshape(-1).detach().cpu()
    else:
        states_t = torch.as_tensor(per_batch_states, dtype=torch.float32).detach().cpu()
        gids_t = torch.as_tensor(per_batch_group_ids, dtype=torch.long).reshape(-1).detach().cpu()
        if per_batch_weights is None:
            weights_t = torch.ones(gids_t.shape, dtype=torch.float32)
        else:
            weights_t = torch.as_tensor(per_batch_weights, dtype=torch.float32).reshape(-1).detach().cpu()
    feature_shape = tuple(states_t.shape[1:])
    return states_t, gids_t, weights_t, feature_shape


def _accumulate_streaming_group_means(
    states_t: torch.Tensor,
    gids_t: torch.Tensor,
    weights_t: torch.Tensor,
    feature_shape: tuple[int, ...],
    store,
    batch_idx: int,
) -> None:
    """Update per-group running weighted state sums and weight totals on the store."""
    for group_value, attr_state, attr_weight in (
        (0, "concept_running_state_sum_a", "concept_running_weight_a"),
        (1, "concept_running_state_sum_b", "concept_running_weight_b"),
    ):
        existing_state = getattr(store, attr_state, None) if batch_idx > 0 else None
        existing_weight = getattr(store, attr_weight, None) if batch_idx > 0 else None
        mask = gids_t == group_value
        if mask.any():
            weight_view = weights_t[mask].view(-1, *([1] * len(feature_shape)))
            batch_state_sum = (states_t[mask] * weight_view).sum(dim=0)
            batch_weight_sum = weights_t[mask].sum()
        else:
            batch_state_sum = torch.zeros(feature_shape, dtype=torch.float32) if existing_state is None else None
            batch_weight_sum = torch.zeros((), dtype=torch.float32)
        if existing_state is None:
            new_state = batch_state_sum
        elif batch_state_sum is None:
            new_state = existing_state
        else:
            new_state = torch.as_tensor(existing_state, dtype=torch.float32) + batch_state_sum
        if existing_weight is None:
            new_weight = batch_weight_sum
        else:
            new_weight = torch.as_tensor(existing_weight, dtype=torch.float32) + batch_weight_sum
        try:
            if new_state is not None:
                setattr(store, attr_state, new_state)
            setattr(store, attr_weight, new_weight)
        except Exception:
            pass


def _drain_streaming_paired_buffers(
    store,
    feature_shape: tuple[int, ...],
) -> None:
    """Pop matched (a, b) prefix pairs from pending buffers and accumulate weighted residuals.

    Pair index = stable iteration order across batches (matches the legacy in_memory contract,
    where ``group_a_states[i]`` is paired with ``group_b_states[i]``). For each matched prefix
    pair, computes the rejection residual ``a - ((a·b)/(b·b)) b`` and adds the pair-weight-mean
    times that residual to ``concept_running_residual_sum``.
    """
    pending_a = getattr(store, "concept_pending_a_states", None)
    pending_a_w = getattr(store, "concept_pending_a_weights", None)
    pending_b = getattr(store, "concept_pending_b_states", None)
    pending_b_w = getattr(store, "concept_pending_b_weights", None)
    if pending_a is None or pending_b is None:
        return
    n_pairs = min(int(pending_a.shape[0]), int(pending_b.shape[0]))
    if n_pairs <= 0:
        return

    matched_a = pending_a[:n_pairs]
    matched_b = pending_b[:n_pairs]
    matched_a_w = pending_a_w[:n_pairs] if pending_a_w is not None else torch.ones(n_pairs, dtype=torch.float32)
    matched_b_w = pending_b_w[:n_pairs] if pending_b_w is not None else torch.ones(n_pairs, dtype=torch.float32)

    flat_dim = int(torch.tensor(feature_shape).prod().item()) if feature_shape else 1
    a_flat = matched_a.reshape(n_pairs, flat_dim)
    b_flat = matched_b.reshape(n_pairs, flat_dim)
    # Per-row dot products (vectorized).
    bb = (b_flat * b_flat).sum(dim=1).clamp_min(1e-12)
    ab = (a_flat * b_flat).sum(dim=1)
    proj_scale = (ab / bb).view(n_pairs, 1)
    residuals_flat = a_flat - proj_scale * b_flat
    residuals = residuals_flat.reshape(n_pairs, *feature_shape)
    pair_weights = (matched_a_w + matched_b_w) / 2

    weight_view = pair_weights.view(-1, *([1] * len(feature_shape)))
    batch_residual_sum = (residuals * weight_view).sum(dim=0)
    batch_pair_weight_sum = pair_weights.sum()

    existing_residual_sum = getattr(store, "concept_running_residual_sum", None)
    existing_pair_weight = getattr(store, "concept_running_pair_weight", None)
    new_residual_sum = (
        batch_residual_sum
        if existing_residual_sum is None
        else torch.as_tensor(existing_residual_sum, dtype=torch.float32) + batch_residual_sum
    )
    new_pair_weight = (
        batch_pair_weight_sum
        if existing_pair_weight is None
        else torch.as_tensor(existing_pair_weight, dtype=torch.float32) + batch_pair_weight_sum
    )
    try:
        setattr(store, "concept_running_residual_sum", new_residual_sum)
        setattr(store, "concept_running_pair_weight", new_pair_weight)
    except Exception:
        pass

    # Trim consumed prefixes; keep unmatched suffix for future batches.
    try:
        setattr(store, "concept_pending_a_states", pending_a[n_pairs:])
        setattr(store, "concept_pending_b_states", pending_b[n_pairs:])
        if pending_a_w is not None:
            setattr(store, "concept_pending_a_weights", pending_a_w[n_pairs:])
        if pending_b_w is not None:
            setattr(store, "concept_pending_b_weights", pending_b_w[n_pairs:])
    except Exception:
        pass


def _accumulate_streaming_paired_rejection(
    states_t: torch.Tensor,
    gids_t: torch.Tensor,
    weights_t: torch.Tensor,
    feature_shape: tuple[int, ...],
    store,
    batch_idx: int,
) -> None:
    """Append per-group rows to pending buffers, then drain matched pairs into running residuals."""
    for group_value, states_attr, weights_attr in (
        (0, "concept_pending_a_states", "concept_pending_a_weights"),
        (1, "concept_pending_b_states", "concept_pending_b_weights"),
    ):
        mask = gids_t == group_value
        if not mask.any():
            continue
        new_states = states_t[mask]
        new_weights = weights_t[mask]
        existing_states = getattr(store, states_attr, None) if batch_idx > 0 else None
        existing_weights = getattr(store, weights_attr, None) if batch_idx > 0 else None
        if existing_states is None or int(getattr(existing_states, "shape", [0])[0]) == 0:
            combined_states = new_states
            combined_weights = new_weights
        else:
            combined_states = torch.cat([torch.as_tensor(existing_states, dtype=torch.float32), new_states], dim=0)
            combined_weights = torch.cat([torch.as_tensor(existing_weights, dtype=torch.float32), new_weights], dim=0)
        try:
            setattr(store, states_attr, combined_states)
            setattr(store, weights_attr, combined_weights)
        except Exception:
            pass

    _drain_streaming_paired_buffers(store, feature_shape)


def _resolve_streaming_group_names(
    module,
    analysis_batch: AnalysisBatch,
    gids_t: torch.Tensor,
) -> tuple[str, str]:
    """Resolve group display names from the batch or per-batch group-name input."""
    group_a_name = str(analysis_batch.get("concept_group_a_name") or "group_a")
    group_b_name = str(analysis_batch.get("concept_group_b_name") or "group_b")
    if (
        analysis_batch.get("concept_group_a_name") is not None
        and analysis_batch.get("concept_group_b_name") is not None
    ):
        return group_a_name, group_b_name
    per_batch_group_names = resolve_aggregate_input(module, analysis_batch, "concept_group_name")
    if per_batch_group_names is None:
        return group_a_name, group_b_name
    flattened_names: list[str] = []
    if isinstance(per_batch_group_names, (list, tuple)):
        iterable_names = per_batch_group_names
    else:
        try:
            iterable_names = list(per_batch_group_names)
        except TypeError:
            iterable_names = [per_batch_group_names]
    for entry in iterable_names:
        if isinstance(entry, (list, tuple)):
            flattened_names.extend(str(n) for n in entry)
        else:
            flattened_names.append(str(entry))
    paired = list(zip(flattened_names, gids_t.tolist(), strict=False))
    if analysis_batch.get("concept_group_a_name") is None:
        a_matches = [n for n, gid in paired if gid == 0 and n]
        if a_matches:
            group_a_name = a_matches[0]
    if analysis_batch.get("concept_group_b_name") is None:
        b_matches = [n for n, gid in paired if gid == 1 and n]
        if b_matches:
            group_b_name = b_matches[0]
    return group_a_name, group_b_name


def _concept_direction_streaming(
    module,
    analysis_batch: AnalysisBatch,
    batch_idx: int,
    store,
) -> AnalysisBatch:
    """Streaming/incremental concept-direction accumulator.

    Updates per-group running aggregator state on the shared analysis store using this batch's
    per-batch latent rows, then recomputes the current concept direction. The final batch's
    emitted direction is the converged result. Storage contract field names are defined in
    :mod:`interpretune.analysis.ops.helpers` (``CONCEPT_STREAMING_*`` constants).

    Supported direction modes:

    * ``mean_difference``, ``single_group``: maintain per-group running weighted state sums and
      weight totals (``concept_running_state_sum_{a,b}``, ``concept_running_weight_{a,b}``).
    * ``paired_rejection``: additionally maintain per-group pending buffers
      (``concept_pending_{a,b}_states``, ``concept_pending_{a,b}_weights``); on each batch, drain
      matched (a, b) prefix pairs (paired by stable iteration order, matching the legacy in_memory
      contract) and accumulate weighted residuals into ``concept_running_residual_sum`` /
      ``concept_running_pair_weight``.
    """
    direction_mode = str(analysis_batch.get("concept_direction_mode", "mean_difference"))
    states_t, gids_t, weights_t, feature_shape = _parse_streaming_per_batch_inputs(module, analysis_batch)

    if direction_mode in ("mean_difference", "single_group"):
        _accumulate_streaming_group_means(states_t, gids_t, weights_t, feature_shape, store, batch_idx)

        state_sum_a = getattr(store, "concept_running_state_sum_a", None)
        weight_a = getattr(store, "concept_running_weight_a", None)
        state_sum_b = getattr(store, "concept_running_state_sum_b", None)
        weight_b = getattr(store, "concept_running_weight_b", None)
        if state_sum_a is None or weight_a is None or float(weight_a) <= 0:
            raise ValueError("concept_direction streaming mode requires at least one group A example")
        mean_a = torch.as_tensor(state_sum_a, dtype=torch.float32) / torch.as_tensor(
            weight_a, dtype=torch.float32
        ).clamp_min(1e-12)
        if direction_mode == "mean_difference":
            if state_sum_b is None or weight_b is None or float(weight_b) <= 0:
                raise ValueError("mean_difference requires examples from both concept groups")
            mean_b = torch.as_tensor(state_sum_b, dtype=torch.float32) / torch.as_tensor(
                weight_b, dtype=torch.float32
            ).clamp_min(1e-12)
            direction_vector = mean_a - mean_b
        else:  # single_group
            direction_vector = mean_a
    elif direction_mode == "paired_rejection":
        _accumulate_streaming_paired_rejection(states_t, gids_t, weights_t, feature_shape, store, batch_idx)
        residual_sum = getattr(store, "concept_running_residual_sum", None)
        pair_weight = getattr(store, "concept_running_pair_weight", None)
        if residual_sum is None or pair_weight is None or float(pair_weight) <= 0:
            # No matched pairs accumulated yet (e.g. all of group_a in batch 0, group_b later).
            # Emit a zero direction; subsequent batches will produce the converged result.
            direction_vector = torch.zeros(feature_shape, dtype=torch.float32)
        else:
            direction_vector = torch.as_tensor(residual_sum, dtype=torch.float32) / torch.as_tensor(
                pair_weight, dtype=torch.float32
            ).clamp_min(1e-12)
    else:
        raise ValueError(f"Unsupported concept_direction_mode in streaming: {direction_mode}")

    direction_norm = torch.linalg.vector_norm(direction_vector)
    if torch.isfinite(direction_norm) and direction_norm.item() > 0:
        direction_vector = direction_vector / direction_norm

    group_a_name, group_b_name = _resolve_streaming_group_names(module, analysis_batch, gids_t)
    concept_label = analysis_batch.get("concept_label")
    resolved_label = concept_label
    if resolved_label is None:
        resolved_label = group_a_name if direction_mode == "single_group" else f"{group_a_name} -> {group_b_name}"

    analysis_batch.update(
        concept_direction=direction_vector.detach().cpu(),
        concept_label=resolved_label,
        concept_direction_mode=direction_mode,
        concept_group_a_name=group_a_name,
        concept_group_b_name=group_b_name,
        concept_aggregate_output_mode="streaming",
    )
    return analysis_batch


def concept_direction_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Compute a concept direction from latent-example rows, or fall back to token-group embeddings.

    Aggregation modes (selected via ``concept_aggregate_output_mode`` on the batch):

    * ``"streaming"``: maintain per-group running weighted state sums and weight totals on
      ``analysis_inputs.store`` (``concept_running_state_sum_a``, ``concept_running_weight_a``,
      ``concept_running_state_sum_b``, ``concept_running_weight_b``). Each per-batch invocation
      updates the running aggregates from this batch's per-batch ``concept_latent_state`` /
      ``concept_group_id`` / ``concept_example_weight`` tensors and recomputes the current concept
      direction from the accumulated sums; the final batch's emitted direction is the converged
      result. Memory cost is O(d_model * num_groups) instead of O(num_examples * d_model). For
      ``paired_rejection``, additionally maintains pending per-group buffers
      (``concept_pending_{a,b}_states``, ``concept_pending_{a,b}_weights``) plus running residual
      and pair-weight totals (``concept_running_residual_sum``, ``concept_running_pair_weight``);
      pairs are matched by stable iteration order (matching the legacy in_memory contract). The
      full storage-contract field set is exported via
      :data:`interpretune.analysis.ops.helpers.CONCEPT_STREAMING_STATE_FIELDS`.
    * ``"in_memory"`` (legacy): consume the aggregate row tensors emitted by
      :func:`extract_concept_latent_examples_impl` in legacy mode and compute the direction over
      the full materialized example set. Supports all direction modes.

    If ``concept_aggregate_output_mode`` is not set, behavior is determined by what is on the
    batch: aggregate row fields trigger the legacy path; per-batch fields with a writable store
    trigger streaming. If neither is available, fall back to a token-group embedding direction
    computed from the model's input embedding matrix.
    """
    aggregate_mode = analysis_batch.get("concept_aggregate_output_mode")
    analysis_inputs = kwargs.get("analysis_inputs")
    store = getattr(analysis_inputs, "store", None) if analysis_inputs is not None else None

    use_streaming = aggregate_mode == "streaming"
    if not use_streaming and aggregate_mode is None:
        # Auto-detect: prefer legacy path if aggregate rows are already present
        legacy_rows_present = resolve_aggregate_input(module, analysis_batch, "concept_latent_state_rows") is not None
        per_batch_state_present = resolve_aggregate_input(module, analysis_batch, "concept_latent_state") is not None
        use_streaming = (not legacy_rows_present) and per_batch_state_present and store is not None

    if use_streaming:
        return _concept_direction_streaming(module, analysis_batch, batch_idx, store)

    latent_state_rows = resolve_aggregate_input(module, analysis_batch, "concept_latent_state_rows")
    group_id_rows = resolve_aggregate_input(module, analysis_batch, "concept_group_id_rows")
    if latent_state_rows is None or group_id_rows is None:
        latent_state_rows = resolve_aggregate_input(module, analysis_batch, "concept_latent_state")
        group_id_rows = resolve_aggregate_input(module, analysis_batch, "concept_group_id")
    if latent_state_rows is not None and group_id_rows is not None:
        direction_mode = str(analysis_batch.get("concept_direction_mode", "mean_difference"))
        concept_label = analysis_batch.get("concept_label")
        group_name_rows = resolve_aggregate_input(module, analysis_batch, "concept_group_name_rows")
        if group_name_rows is None:
            group_name_rows = resolve_aggregate_input(module, analysis_batch, "concept_group_name")
        example_weight_rows = resolve_aggregate_input(module, analysis_batch, "concept_example_weight_rows")
        if example_weight_rows is None:
            example_weight_rows = resolve_aggregate_input(module, analysis_batch, "concept_example_weight")

        latent_states, group_ids, example_weights, flattened_group_names = _flatten_concept_store_rows(
            latent_state_rows,
            group_id_rows,
            group_name_rows,
            example_weight_rows,
        )

        group_a_mask = group_ids == 0
        group_b_mask = group_ids == 1
        if not group_a_mask.any():
            raise ValueError("concept_direction requires at least one example from concept group A")

        if direction_mode == "mean_difference":
            if not group_b_mask.any():
                raise ValueError("mean_difference requires at least one example from each concept group")
            direction_vector = weighted_mean(
                latent_states[group_a_mask], example_weights[group_a_mask]
            ) - weighted_mean(latent_states[group_b_mask], example_weights[group_b_mask])
        elif direction_mode == "paired_rejection":
            if not group_b_mask.any():
                raise ValueError("paired_rejection requires at least one example from each concept group")
            group_a_states = latent_states[group_a_mask]
            group_b_states = latent_states[group_b_mask]
            group_a_weights = example_weights[group_a_mask]
            group_b_weights = example_weights[group_b_mask]
            if group_a_states.shape[0] != group_b_states.shape[0]:
                raise ValueError("paired_rejection requires equal numbers of group-a and group-b latent examples")
            residuals = []
            pair_weights = []
            for state_a, state_b, weight_a, weight_b in zip(
                group_a_states, group_b_states, group_a_weights, group_b_weights, strict=True
            ):
                denom = torch.dot(state_b, state_b).clamp_min(1e-12)
                proj = (torch.dot(state_a, state_b) / denom) * state_b
                residuals.append(state_a - proj)
                pair_weights.append((weight_a + weight_b) / 2)
            direction_vector = weighted_mean(torch.stack(residuals), torch.stack(pair_weights))
        elif direction_mode == "single_group":
            direction_vector = weighted_mean(latent_states[group_a_mask], example_weights[group_a_mask])
        else:
            raise ValueError(f"Unsupported concept_direction_mode: {direction_mode}")

        direction_norm = torch.linalg.vector_norm(direction_vector)
        if torch.isfinite(direction_norm) and direction_norm.item() > 0:
            direction_vector = direction_vector / direction_norm

        group_a_name = str(analysis_batch.get("concept_group_a_name") or "group_a")
        group_b_name = str(analysis_batch.get("concept_group_b_name") or "group_b")
        if flattened_group_names:
            paired = zip(flattened_group_names, group_ids.tolist(), strict=False)
            group_a_matches = [name for name, group_id in paired if group_id == 0 and name]
            paired = zip(flattened_group_names, group_ids.tolist(), strict=False)
            group_b_matches = [name for name, group_id in paired if group_id == 1 and name]
            if group_a_matches:
                group_a_name = group_a_matches[0]
            if group_b_matches:
                group_b_name = group_b_matches[0]

        resolved_label = concept_label
        if resolved_label is None:
            resolved_label = group_a_name if direction_mode == "single_group" else f"{group_a_name} -> {group_b_name}"

        analysis_batch.update(
            concept_direction=direction_vector.detach().cpu(),
            concept_label=resolved_label,
            concept_direction_mode=direction_mode,
            concept_group_a_name=group_a_name,
            concept_group_b_name=group_b_name,
        )
        return analysis_batch

    tokenizer = resolve_tokenizer(module)
    embed_weight = resolve_embedding_weight(module)
    raw_group_a = analysis_batch.get("concept_group_a")
    raw_group_b = analysis_batch.get("concept_group_b")
    group_a = list(raw_group_a or [])
    group_b = list(raw_group_b or [])
    direction_mode = str(analysis_batch.get("concept_direction_mode", "mean_difference"))
    concept_label = analysis_batch.get("concept_label")
    if not group_a:
        raise ValueError("concept_direction requires non-empty concept_group_a")

    group_a_ids = token_strings_to_last_ids(tokenizer, group_a)
    group_a_embed = embed_weight[torch.tensor(group_a_ids, device=embed_weight.device)].float()
    if group_b:
        group_b_ids = token_strings_to_last_ids(tokenizer, group_b)
        group_b_embed = embed_weight[torch.tensor(group_b_ids, device=embed_weight.device)].float()
    else:
        group_b_ids = []
        group_b_embed = None

    if direction_mode == "mean_difference":
        if group_b_embed is None:
            raise ValueError("mean_difference requires non-empty concept_group_b")
        direction_vector = group_a_embed.mean(dim=0) - group_b_embed.mean(dim=0)
    elif direction_mode == "paired_rejection":
        if group_b_embed is None:
            raise ValueError("paired_rejection requires non-empty concept_group_b")
        if len(group_a_ids) != len(group_b_ids):
            raise ValueError("paired_rejection requires concept groups of equal length")
        residuals = []
        for embed_a, embed_b in zip(group_a_embed, group_b_embed):
            denom = torch.dot(embed_b, embed_b).clamp_min(1e-12)
            proj = (torch.dot(embed_a, embed_b) / denom) * embed_b
            residuals.append(embed_a - proj)
        direction_vector = torch.stack(residuals).mean(dim=0)
    elif direction_mode == "single_group":
        direction_vector = group_a_embed.mean(dim=0)
    else:
        raise ValueError(f"Unsupported concept_direction_mode: {direction_mode}")

    direction_norm = torch.linalg.vector_norm(direction_vector)
    if torch.isfinite(direction_norm) and direction_norm.item() > 0:
        direction_vector = direction_vector / direction_norm

    analysis_batch.update(
        concept_direction=direction_vector.detach().cpu(),
        concept_label=(
            concept_label
            or (
                " / ".join(group_a)
                if direction_mode == "single_group"
                else f"{' / '.join(group_a)} -> {' / '.join(group_b)}"
            )
        ),
        concept_group_a_token_ids=group_a_ids,
        concept_group_b_token_ids=group_b_ids,
        concept_direction_mode=direction_mode,
    )
    return analysis_batch


def model_fwd_intervention_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Apply generalized hook-point interventions and return pre/post logits.

    Delegates the intervention mechanics to the model backend's
    ``fwd_w_intervention`` method so the same op works for both
    NNsight (traced execution) and TransformerLens (eager hook execution).
    """

    model_backend = require_model_backend(module)
    interventions = resolve_interventions(
        analysis_batch=analysis_batch,
        resolve_field=lambda field_name: resolve_aggregate_input(module, analysis_batch, field_name),
        load_json_field=lambda field_name: load_json_field(module, analysis_batch, field_name),
        kwargs=kwargs,
    )
    use_latent_models = bool(
        resolve_aggregate_input(module, analysis_batch, "use_latent_models") or kwargs.get("use_latent_models", False)
    )
    latent_model_handles = getattr(module, "sae_handles", None) if use_latent_models else None

    if (
        getattr(module, "analysis_cfg", None)
        and module.analysis_cfg.auto_prune_batch_encoding
        and isinstance(batch, BatchEncoding)
    ):
        batch = module.auto_prune_batch(batch, "forward")

    with torch.no_grad():
        pre_logits, post_logits = model_backend.fwd_w_intervention(
            model=module.model,
            batch=batch,
            interventions=interventions,
            latent_model_handles=latent_model_handles,
        )

    # Extract last-token logits
    pre_lt = last_token_logits(pre_logits)
    post_lt = last_token_logits(post_logits)

    target_ids = resolve_aggregate_input(module, analysis_batch, "logit_target_ids")
    if target_ids is None:
        concept_a_ids = analysis_batch.get("concept_group_a_token_ids")
        concept_b_ids = analysis_batch.get("concept_group_b_token_ids")
        real_ids = list(concept_a_ids or []) + list(concept_b_ids or [])
        if real_ids:
            target_ids = torch.tensor(real_ids, dtype=torch.long)
    target_ids_tensor = None if target_ids is None else torch.as_tensor(target_ids, dtype=torch.long).reshape(-1)
    logit_diff = mean_target_logit_delta(pre_lt, post_lt, target_ids_tensor)

    analysis_batch.update(
        pre_intervention_logits=pre_lt.detach().cpu(),
        post_intervention_logits=post_lt.detach().cpu(),
        logit_diff=logit_diff.detach().cpu(),
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
    concept_direction = analysis_batch.get("concept_direction")
    concept_label = analysis_batch.get("concept_label")
    concept_group_a_token_ids = analysis_batch.get("concept_group_a_token_ids")
    concept_group_b_token_ids = analysis_batch.get("concept_group_b_token_ids")
    concept_direction_mode = analysis_batch.get("concept_direction_mode")
    if concept_direction is not None and "attribution_targets" not in kwargs:
        kwargs["attribution_targets"] = analysis_backend.build_concept_attribution_targets(
            module,
            prompt,
            concept_direction,
            concept_label,
            concept_group_a_token_ids=concept_group_a_token_ids,
            concept_group_b_token_ids=concept_group_b_token_ids,
            concept_direction_mode=concept_direction_mode,
        )

    # Only forward kwargs consumed by graph construction. Composite pipeline
    # calls also carry downstream stage params such as top_n and
    # intervention_scale_factor that must remain available for later ops.
    attribution_graph_kwargs = {
        name: kwargs[name]
        for name in (
            "attribution_targets",
            "max_n_logits",
            "desired_logit_prob",
            "batch_size",
            "max_feature_nodes",
            "offload",
            "verbose",
            "update_interval",
        )
        if name in kwargs
    }

    graph = module.generate_attribution_graph(prompt, **attribution_graph_kwargs)
    extra_metadata: dict[str, Any] = {}
    extra_metadata["batch_idx"] = batch_idx
    if concept_label is not None:
        extra_metadata["concept_label"] = concept_label
    analysis_batch.update(**analysis_backend.decompose_graph(graph, extra_metadata=extra_metadata))

    # Resolve virtual logit_target_ids from concept-direction graphs.
    # Circuit-tracer assigns virtual IDs (>= vocab_size) to custom concept targets.
    # Replace them with the real concept group token IDs so downstream ops can index logits.
    logit_target_ids = getattr(analysis_batch, "logit_target_ids", None)
    if logit_target_ids is not None and concept_direction is not None:
        ids_tensor = torch.as_tensor(logit_target_ids, dtype=torch.long).reshape(-1)
        vocab_size = getattr(analysis_batch, "graph_vocab_size", None)
        if vocab_size is not None and (ids_tensor >= int(vocab_size)).any():
            real_ids = list(concept_group_a_token_ids or []) + list(concept_group_b_token_ids or [])
            if not real_ids:
                raise ValueError(
                    "logit_target_ids contain virtual IDs (>= vocab_size) but no concept group "
                    "token IDs are available for resolution. Provide concept_group_a_token_ids / "
                    "concept_group_b_token_ids or explicit logit_target_ids."
                )
            analysis_batch.update(logit_target_ids=torch.tensor(real_ids, dtype=torch.long))

    return analysis_batch


def extract_top_features_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    top_n: int | None = None,
    **kwargs,
) -> AnalysisBatch:
    """Extract the top scoring features from analysis-batch feature rows.

    An optional ``feature_selection`` kwarg (:class:`FeatureSelectionSpec`) pre-filters
    ``active_features`` rows before score sorting.  The filter uses **OR** semantics —
    a row is kept if it matches *any* criterion in the spec.
    """
    feature_selection: FeatureSelectionSpec | None = kwargs.get("feature_selection", None)

    active_features = torch.as_tensor(getattr(analysis_batch, "active_features", []), dtype=torch.long)
    selected_features = torch.as_tensor(getattr(analysis_batch, "selected_features", []), dtype=torch.long)
    activation_values = getattr(analysis_batch, "activation_values", None)
    activation_tensor = (
        None if activation_values is None else torch.as_tensor(activation_values, dtype=torch.float32).reshape(-1)
    )
    if active_features.numel() == 0:
        analysis_batch.update(
            top_feature_ids=torch.empty((0, 3), dtype=torch.long),
            top_feature_scores=torch.empty((0,), dtype=torch.float32),
        )
        return analysis_batch
    active_features = active_features.reshape(-1, 3)

    score_source = resolve_feature_score_source(
        kwargs.get("score_source") or (feature_selection.score_source if feature_selection else None)
    )
    score_values = getattr(analysis_batch, score_source, None) if score_source else None
    if score_source is not None and score_values is None:
        raise ValueError(f"extract_top_features score_source '{score_source}' is not present on analysis_batch")
    if score_values is None:
        score_values = getattr(analysis_batch, "node_influence_scores", None)
    if score_values is None:
        score_values = getattr(analysis_batch, "activation_values", None)
    scores = torch.as_tensor(score_values, dtype=torch.float32)
    if scores.dim() > 1:
        scores = scores.reshape(-1)

    feature_rows = active_features
    aligned_activation_values = None
    if selected_features.numel() > 0 and selected_features.shape[0] == scores.shape[0]:
        feature_rows = require_analysis_backend(module).select_feature_rows(active_features, selected_features)
        if activation_tensor is not None and activation_tensor.shape[0] == active_features.shape[0]:
            aligned_activation_values = activation_tensor.index_select(0, selected_features.reshape(-1))
        elif activation_tensor is not None and activation_tensor.shape[0] == selected_features.shape[0]:
            aligned_activation_values = activation_tensor
    elif active_features.shape[0] != scores.shape[0]:
        raise ValueError(
            "extract_top_features requires active_features to match score length directly or via selected_features"
        )
    elif activation_tensor is not None and activation_tensor.shape[0] == active_features.shape[0]:
        aligned_activation_values = activation_tensor

    if feature_selection is not None:
        feature_rows, scores, aligned_activation_values = _augment_feature_rows_for_selection(
            feature_rows,
            scores,
            aligned_activation_values,
            feature_selection,
        )

    # ---- apply optional pre-filter before score ranking ----
    if feature_selection is not None:
        sel_mask = apply_feature_selection_filter(feature_rows, feature_selection)
        if sel_mask.any():
            sel_idx = sel_mask.nonzero(as_tuple=False).reshape(-1)
            feature_rows = feature_rows.index_select(0, sel_idx)
            scores = scores.index_select(0, sel_idx)
            if aligned_activation_values is not None:
                aligned_activation_values = aligned_activation_values.index_select(0, sel_idx)

        feature_rows, scores, aligned_activation_values = _apply_optional_feature_sign_filter(
            feature_rows,
            scores,
            aligned_activation_values,
            feature_selection,
        )

    rank_by_abs = bool(kwargs.get("rank_by_abs", feature_selection.rank_by_abs if feature_selection else False))
    top_indices = _select_top_feature_indices(
        feature_rows,
        scores,
        top_n,
        feature_selection,
        rank_scores=scores.abs() if rank_by_abs else None,
    )
    update_payload: dict[str, Any] = {
        "top_feature_ids": feature_rows.index_select(0, top_indices).detach().cpu(),
        "top_feature_scores": scores.index_select(0, top_indices).detach().cpu(),
    }
    if aligned_activation_values is not None:
        update_payload["top_feature_activation_values"] = (
            aligned_activation_values.index_select(0, top_indices).detach().cpu()
        )
    analysis_batch.update(**update_payload)
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
    update_payload: dict[str, Any] = {
        "node_influence_scores": node_scores,
        "node_feature_ids": node_feature_ids,
    }
    signed_score_fn = getattr(analysis_backend, "compute_signed_node_influence_scores", None)
    if callable(signed_score_fn):
        signed_scores = signed_score_fn(graph)
        update_payload["node_signed_influence_scores"] = signed_scores
    analysis_batch.update(
        **update_payload,
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

    This op currently implements forward-only intervention analysis and stores both the circuit-tracer tuple payload and
    a canonical feature-target InterventionDict summary for AnalysisStore consumers.
    """
    analysis_backend = require_analysis_backend(module)

    replacement_model = getattr(module, "replacement_model", None)
    if replacement_model is None:
        raise ValueError("feature_intervention_forward requires module.replacement_model")

    prompt = kwargs.pop("prompt", None)
    if prompt is None:
        prompt = analysis_backend.resolve_prompt(module, analysis_batch, batch)
    settings = analysis_backend.resolve_feature_intervention_settings(module, kwargs)
    if (
        getattr(analysis_batch, "top_feature_ids", None) is None
        and getattr(analysis_batch, "active_features", None) is not None
    ):
        selection_kwargs = {
            name: kwargs[name] for name in ("feature_selection", "score_source", "rank_by_abs") if name in kwargs
        }
        analysis_batch = extract_top_features_impl(
            module,
            analysis_batch,
            batch,
            batch_idx,
            top_n=kwargs.get("top_n"),
            **selection_kwargs,
        )
    feature_rows = analysis_batch.require(
        "top_feature_ids",
        message="feature_intervention_forward requires top_feature_ids in analysis_batch or scoped inputs",
    )
    feature_scores = analysis_batch.get("top_feature_scores")
    feature_activation_values = analysis_batch.get("top_feature_activation_values")
    target_ids = analysis_batch.get("logit_target_ids")

    # If no explicit logit_target_ids, try to resolve from concept group token IDs
    if target_ids is None:
        concept_a_ids = analysis_batch.get("concept_group_a_token_ids")
        concept_b_ids = analysis_batch.get("concept_group_b_token_ids")
        real_ids = list(concept_a_ids or []) + list(concept_b_ids or [])
        if real_ids:
            target_ids = torch.tensor(real_ids, dtype=torch.long)

    intervention_inputs = {
        "top_feature_ids": feature_rows,
        "top_feature_scores": feature_scores,
        "top_feature_activation_values": feature_activation_values,
        "logit_target_ids": target_ids,
    }
    analysis_batch.update(
        **{
            key: value
            for key, value in intervention_inputs.items()
            if value is not None and getattr(analysis_batch, key, None) is None
        }
    )
    interventions, intervention_payload = analysis_backend.build_feature_interventions(intervention_inputs, settings)

    pre_logits_raw, _ = replacement_model.get_activations(prompt)
    pre_logits = last_token_logits(pre_logits_raw)
    intervention_activation_cache = None

    if interventions:
        post_logits_raw, intervention_activation_cache = replacement_model.feature_intervention(
            prompt,
            interventions,
            **analysis_backend.feature_intervention_call_kwargs(settings),
        )
        post_logits = last_token_logits(post_logits_raw)
    else:
        post_logits = pre_logits.clone()

    target_ids_tensor = None
    if target_ids is not None:
        target_ids_tensor = torch.as_tensor(target_ids, dtype=torch.long).reshape(-1)
    logit_diff = mean_target_logit_delta(pre_logits, post_logits, target_ids_tensor)

    analysis_batch.update(
        **intervention_payload,
        pre_intervention_logits=pre_logits,
        post_intervention_logits=post_logits,
        logit_diff=logit_diff.detach().cpu(),
    )
    if intervention_activation_cache is not None:
        analysis_batch.update(intervention_activation_cache=intervention_activation_cache)
    return analysis_batch
