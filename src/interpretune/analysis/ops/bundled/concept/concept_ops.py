"""Bundled concept-direction op family.

Latent-state extraction, concept-example selection, concept-direction aggregation (streaming and
in-memory), and generalized hook-point interventions. Self-contained modulo the sanctioned
op-authoring surfaces (:mod:`interpretune.analysis.optools`, :mod:`interpretune.analysis.backends`);
see the custom ops composition guide. This family is also the source of the first published hub seed
collection, so it must remain publishable as a standalone ``kinds: [ops]`` repo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchEncoding

from interpretune.analysis.backends import require_intervention_support, resolve_interventions
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.backends.capabilities import BackendCapability
from interpretune.analysis.optools import (
    require_backend_capability,
    last_token_logits,
    load_json_field,
    mean_target_logit_delta,
    require_model_backend,
    resolve_aggregate_input,
    resolve_embedding_weight,
    resolve_tokenizer,
    token_strings_to_last_ids,
    weighted_mean,
)

# ---------------------------------------------------------------------------
# Concept-direction streaming-state storage contract
# ---------------------------------------------------------------------------
# These names are the formal storage contract used by the ``concept_direction``
# op's incremental aggregator, and they are the ``op_state.fields`` this family's
# YAML declares (``test_bundled_concept_streaming`` pins the two in sync, with
# these constants as the source of truth). State is read/written through the
# declared container at ``analysis_inputs.op_state``, whose lifecycle the runner
# owns: cleared at run start, at epoch boundaries only when the op asks
# (``reset_each_epoch``), released at run end. Treat as a stable surface;
# undeclared names raise rather than being silently created.
#
# Modes:
#   - ``mean_difference`` / ``single_group``: only the per-group running
#     weighted state sums and weight totals are used.
#   - ``paired_rejection``: additionally maintains pending per-group buffers
#     (for matching pair rows that arrive in different batches), a running
#     residual sum, and a running pair-weight total.
CONCEPT_STREAMING_GROUP_FIELDS: tuple[str, ...] = (
    "concept_running_state_sum_a",
    "concept_running_weight_a",
    "concept_running_state_sum_b",
    "concept_running_weight_b",
)

CONCEPT_STREAMING_PAIRED_REJECTION_FIELDS: tuple[str, ...] = (
    "concept_pending_a_states",
    "concept_pending_a_weights",
    "concept_pending_b_states",
    "concept_pending_b_weights",
    "concept_running_residual_sum",
    "concept_running_pair_weight",
)

CONCEPT_STREAMING_STATE_FIELDS: tuple[str, ...] = (
    *CONCEPT_STREAMING_GROUP_FIELDS,
    *CONCEPT_STREAMING_PAIRED_REJECTION_FIELDS,
)

# Legacy ``in_memory`` aggregation for extract_concept_latent_examples: per-batch row lists
# accumulated across batches. Same cross-batch state mechanism, different op, so it is declared
# separately. `concept_context_indices_rows` is only populated when context indices are supplied.
CONCEPT_AGGREGATE_ROW_FIELDS: tuple[str, ...] = (
    "concept_latent_state_rows",
    "concept_group_id_rows",
    "concept_group_name_rows",
    "concept_example_logit_diff_rows",
    "concept_example_weight_rows",
    "concept_context_indices_rows",
)


def reset_concept_streaming_state(state: Any) -> None:
    """Clear all concept_direction streaming aggregator fields.

    ``state`` is the op's declared ``op_state`` container (``analysis_inputs.op_state``). The runner
    already clears it at run start and releases it at run end, so this is for manual loops that reuse
    one ``AnalysisCfg`` across independent concept-direction runs. Accepts ``None`` for convenience.
    """
    if state is None:
        return
    state.clear()


def resolve_concept_cache_key(analysis_batch: AnalysisBatch) -> str:
    """Return the configured cache key for concept latent extraction (default ``unembed.hook_in``)."""
    return str(analysis_batch.get("concept_cache_key") or "unembed.hook_in")


def _resolve_context_token_indices(
    analysis_batch: AnalysisBatch,
    answer_index_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_context_token_indices = analysis_batch.get("context_token_indices")
    if raw_context_token_indices is None:
        raw_indices = answer_index_tensor - 1
        valid_mask = raw_indices >= 0
        return raw_indices.clamp(min=0), valid_mask

    context_index_tensor = torch.as_tensor(
        raw_context_token_indices,
        dtype=torch.long,
        device=answer_index_tensor.device,
    ).reshape(-1)
    if context_index_tensor.shape != answer_index_tensor.shape:
        raise ValueError(
            "extract_concept_latent_state requires context_token_indices to align with answer_indices "
            f"({tuple(context_index_tensor.shape)} vs {tuple(answer_index_tensor.shape)})"
        )
    valid_mask = context_index_tensor >= 0
    return context_index_tensor.clamp(min=0), valid_mask


def project_context_enhanced_states(
    answer_states: torch.Tensor,
    context_states: torch.Tensor,
    *,
    context_scale: float = 1.0,
    use_answer_state_as_basis: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project context-enhanced latent rows using either context or answer as the basis.

    The answer state is always scaled first so the existing context-basis path remains
    unchanged. When ``use_answer_state_as_basis`` is true, the context state is projected
    onto that scaled answer-state basis instead.
    """

    scaled_answer = float(context_scale) * answer_states
    if use_answer_state_as_basis:
        projection_source = context_states
        projection_basis = scaled_answer
    else:
        projection_source = scaled_answer
        projection_basis = context_states

    dot_num = (projection_source * projection_basis).sum(dim=-1, keepdim=True)
    dot_den = (projection_basis * projection_basis).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    projected_states = (dot_num / dot_den) * projection_basis
    return scaled_answer, dot_num, dot_den, projected_states


# TODO: This may be better cast as a separate op itself rather than a helper, we should revisit
def extract_concept_latent_state_from_cache(
    analysis_batch: AnalysisBatch,
    context_enhanced: bool = False,
    context_scale: float = 1.0,
    use_answer_state_as_basis: bool = False,
) -> tuple[torch.Tensor, str]:
    """Pull per-example concept latent rows out of an activation cache at the configured cache key."""
    cache = analysis_batch.cache
    answer_indices = analysis_batch.answer_indices
    if cache is None or answer_indices is None:
        raise ValueError("extract_concept_latent_state requires cache and answer_indices")

    cache_key = resolve_concept_cache_key(analysis_batch)
    if cache_key not in cache:
        raise ValueError(f"extract_concept_latent_state could not find cache key '{cache_key}'")

    cache_tensor = torch.as_tensor(cache[cache_key])
    if cache_tensor.dim() < 2:
        raise ValueError(f"Expected cached latent states for '{cache_key}' to be rank >= 2, got {cache_tensor.dim()}")

    if cache_tensor.dim() >= 3:
        index_tensor = torch.as_tensor(answer_indices, dtype=torch.long, device=cache_tensor.device).reshape(-1)
        batch_indices = torch.arange(cache_tensor.size(0), device=cache_tensor.device)
        latent_states = cache_tensor[batch_indices, index_tensor].detach().cpu().float()

        if context_enhanced:
            context_indices, valid = _resolve_context_token_indices(analysis_batch, index_tensor)
            context_states = cache_tensor[batch_indices, context_indices].detach().cpu().float()

            _scaled_answer, _dot_num, _dot_den, projected = project_context_enhanced_states(
                latent_states,
                context_states,
                context_scale=context_scale,
                use_answer_state_as_basis=use_answer_state_as_basis,
            )

            valid_expanded = valid.unsqueeze(-1).expand_as(latent_states)
            latent_states = torch.where(valid_expanded, projected, latent_states)
    else:
        latent_states = cache_tensor.detach().cpu().float()

    return latent_states, cache_key


def flatten_concept_store_rows(
    latent_state_rows: Any,
    group_id_rows: Any,
    group_name_rows: Any = None,
    example_weight_rows: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Flatten stored concept-direction rows while skipping empty per-batch entries."""

    def _ensure_row_sequence(values: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, (list, tuple)):
            return list(values)
        if hasattr(values, "__iter__") and not isinstance(values, (Mapping, str, bytes, torch.Tensor)):
            return list(values)
        return [values]

    latent_rows = _ensure_row_sequence(latent_state_rows)
    group_rows = _ensure_row_sequence(group_id_rows)
    name_rows = _ensure_row_sequence(group_name_rows)
    weight_rows = _ensure_row_sequence(example_weight_rows)

    flattened_states: list[torch.Tensor] = []
    flattened_groups: list[torch.Tensor] = []
    flattened_weights: list[torch.Tensor] = []
    flattened_names: list[str] = []

    for row_idx, state_row in enumerate(latent_rows):
        state_tensor = torch.as_tensor(state_row, dtype=torch.float32).detach().cpu()
        if state_tensor.numel() == 0:
            continue
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        if row_idx >= len(group_rows):
            raise ValueError("concept_direction requires concept_group_id rows for every concept_latent_state row")
        group_tensor = torch.as_tensor(group_rows[row_idx], dtype=torch.long).detach().cpu().reshape(-1)
        if group_tensor.numel() == 0:
            continue
        if state_tensor.shape[0] != group_tensor.shape[0]:
            raise ValueError(
                "concept_direction requires concept_latent_state and concept_group_id row lengths to match"
            )

        if row_idx < len(weight_rows) and weight_rows[row_idx] is not None:
            weight_tensor = torch.as_tensor(weight_rows[row_idx], dtype=torch.float32).detach().cpu().reshape(-1)
            if weight_tensor.numel() == 0:
                weight_tensor = torch.ones(state_tensor.shape[0], dtype=torch.float32)
        else:
            weight_tensor = torch.ones(state_tensor.shape[0], dtype=torch.float32)
        if weight_tensor.shape[0] != state_tensor.shape[0]:
            raise ValueError(
                "concept_direction requires concept_example_weight row lengths to match concept_latent_state"
            )

        row_names: list[str] = []
        if row_idx < len(name_rows):
            raw_names = name_rows[row_idx]
            if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
                row_names = [str(item) for item in raw_names]
            elif raw_names is not None:
                row_names = [str(raw_names)]
        if row_names and len(row_names) != state_tensor.shape[0]:
            raise ValueError("concept_direction requires concept_group_name row lengths to match concept_latent_state")
        if not row_names:
            row_names = [""] * state_tensor.shape[0]

        flattened_states.append(state_tensor)
        flattened_groups.append(group_tensor)
        flattened_weights.append(weight_tensor)
        flattened_names.extend(row_names)

    if not flattened_states:
        raise ValueError("concept_direction requires at least one non-empty concept_latent_state row")

    return (
        torch.cat(flattened_states, dim=0),
        torch.cat(flattened_groups, dim=0),
        torch.cat(flattened_weights, dim=0),
        flattened_names,
    )


def concept_target_token_ids(module: Any, concept_direction: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    """Project a concept direction onto the token embedding table and return the top token ids."""
    embed_weight = resolve_embedding_weight(module).float()
    direction = torch.as_tensor(concept_direction, dtype=embed_weight.dtype, device=embed_weight.device).reshape(-1)
    if embed_weight.dim() != 2:
        raise ValueError("Embedding weight must be rank-2 to derive concept target token ids")
    if embed_weight.shape[1] != direction.shape[0]:
        raise ValueError(
            "Concept direction dimensionality must match the embedding dimension "
            f"({direction.shape[0]} vs {embed_weight.shape[1]})"
        )

    direction_norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction_norm) or direction_norm.item() <= 0:
        raise ValueError("Concept direction must have finite non-zero norm")
    direction = direction / direction_norm

    embed_norms = torch.linalg.vector_norm(embed_weight, dim=1, keepdim=True).clamp_min(1e-12)
    scores = (embed_weight / embed_norms) @ direction
    top_k = max(1, min(int(top_k), int(scores.shape[0])))
    return torch.topk(scores, k=top_k).indices.detach().cpu()


# ---------------------------------------------------------------------------
# Op implementations
# ---------------------------------------------------------------------------


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

    latent_states, cache_key = extract_concept_latent_state_from_cache(
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
      held in that op's declared ``op_state``. This avoids materializing the full latent-row payload
      and keeps per-batch payload sizes constant. Recommended for all new callers and any pipeline
      where the full set of selected examples does not need to be retained for later inspection.
    * ``"in_memory"`` (legacy): accumulate the full per-batch row collections in this op's declared
      ``op_state`` (``concept_latent_state_rows``, ``concept_group_id_rows``,
      ``concept_group_name_rows``, ``concept_example_logit_diff_rows``,
      ``concept_example_weight_rows``, optionally ``concept_context_indices_rows``) and re-emit
      them on every returned batch. Each per-batch call appends to a Python list and re-binds
      it in op state and on ``analysis_batch``; the underlying tensor data is shared by reference,
      but the list overhead grows linearly per batch and the runner's per-batch payloads end up
      holding O(N²) cumulative list references for ``N`` batches. This mode remains useful for
      callers that need access to the full row collection (e.g. parity tests, pre-computed
      aggregate inputs to ``concept_direction``); do not use it for large concept-example sets.

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

    cache_key = resolve_concept_cache_key(analysis_batch)
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
    op_state = getattr(analysis_inputs, "op_state", None) if analysis_inputs is not None else None
    if aggregate_output and aggregate_mode == "in_memory":
        if op_state is None:
            raise ValueError(
                "extract_concept_latent_examples with aggregate_output and concept_aggregate_output_mode="
                "'in_memory' accumulates rows across batches, which requires this op's declared op_state. "
                "Run it through an AnalysisCfg (a runner, or interpretune.analysis.execution."
                "execute_analysis_op) so the state container has a lifecycle owner."
            )
        aggregate_rows = (
            ("concept_latent_state_rows", selected_states),
            ("concept_group_id_rows", selected_group_ids),
            ("concept_group_name_rows", selected_group_names),
            ("concept_example_logit_diff_rows", selected_logit_diffs),
            ("concept_example_weight_rows", selected_weights),
        )
        for field_name, row_value in aggregate_rows:
            existing_rows = list(op_state.get(field_name) or [])
            existing_rows.append(row_value)
            op_state.set(field_name, existing_rows)
            aggregated_updates[field_name] = existing_rows
        if selected_context_indices is not None:
            context_rows = list(op_state.get("concept_context_indices_rows") or [])
            context_rows.append(selected_context_indices)
            op_state.set("concept_context_indices_rows", context_rows)
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
        states_t, gids_t, weights_t, _ = flatten_concept_store_rows(
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
    op_state,
) -> None:
    """Update per-group running weighted state sums and weight totals in the declared op state.

    "Is there prior state" is now simply "has the field been set", so no ``batch_idx`` heuristic is
    involved: the lifecycle owner decides when accumulation starts over.
    """
    for group_value, attr_state, attr_weight in (
        (0, "concept_running_state_sum_a", "concept_running_weight_a"),
        (1, "concept_running_state_sum_b", "concept_running_weight_b"),
    ):
        existing_state = op_state.get(attr_state)
        existing_weight = op_state.get(attr_weight)
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
        if new_state is not None:
            op_state.set(attr_state, new_state)
        op_state.set(attr_weight, new_weight)


def _drain_streaming_paired_buffers(
    op_state,
    feature_shape: tuple[int, ...],
) -> None:
    """Pop matched (a, b) prefix pairs from pending buffers and accumulate weighted residuals.

    Pair index = stable iteration order across batches (matches the legacy in_memory contract,
    where ``group_a_states[i]`` is paired with ``group_b_states[i]``). For each matched prefix
    pair, computes the rejection residual ``a - ((a·b)/(b·b)) b`` and adds the pair-weight-mean
    times that residual to ``concept_running_residual_sum``.
    """
    pending_a = op_state.get("concept_pending_a_states")
    pending_a_w = op_state.get("concept_pending_a_weights")
    pending_b = op_state.get("concept_pending_b_states")
    pending_b_w = op_state.get("concept_pending_b_weights")
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

    existing_residual_sum = op_state.get("concept_running_residual_sum")
    existing_pair_weight = op_state.get("concept_running_pair_weight")
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
    op_state.update(
        concept_running_residual_sum=new_residual_sum,
        concept_running_pair_weight=new_pair_weight,
    )

    # Trim consumed prefixes; keep unmatched suffix for future batches.
    op_state.update(
        concept_pending_a_states=pending_a[n_pairs:],
        concept_pending_b_states=pending_b[n_pairs:],
    )
    if pending_a_w is not None:
        op_state.set("concept_pending_a_weights", pending_a_w[n_pairs:])
    if pending_b_w is not None:
        op_state.set("concept_pending_b_weights", pending_b_w[n_pairs:])


def _accumulate_streaming_paired_rejection(
    states_t: torch.Tensor,
    gids_t: torch.Tensor,
    weights_t: torch.Tensor,
    feature_shape: tuple[int, ...],
    op_state,
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
        existing_states = op_state.get(states_attr)
        existing_weights = op_state.get(weights_attr)
        if existing_states is None or int(getattr(existing_states, "shape", [0])[0]) == 0:
            combined_states = new_states
            combined_weights = new_weights
        else:
            combined_states = torch.cat([torch.as_tensor(existing_states, dtype=torch.float32), new_states], dim=0)
            combined_weights = torch.cat([torch.as_tensor(existing_weights, dtype=torch.float32), new_weights], dim=0)
        op_state.set(states_attr, combined_states)
        op_state.set(weights_attr, combined_weights)

    _drain_streaming_paired_buffers(op_state, feature_shape)


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
    op_state,
) -> AnalysisBatch:
    """Streaming/incremental concept-direction accumulator.

    Updates per-group running aggregator state in the op's declared ``op_state`` container using
    this batch's per-batch latent rows, then recomputes the current concept direction. The final
    batch's emitted direction is the converged result. Storage contract field names are defined at
    the top of this module (``CONCEPT_STREAMING_*`` constants).

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
        _accumulate_streaming_group_means(states_t, gids_t, weights_t, feature_shape, op_state)

        state_sum_a = op_state.get("concept_running_state_sum_a")
        weight_a = op_state.get("concept_running_weight_a")
        state_sum_b = op_state.get("concept_running_state_sum_b")
        weight_b = op_state.get("concept_running_weight_b")
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
        _accumulate_streaming_paired_rejection(states_t, gids_t, weights_t, feature_shape, op_state)
        residual_sum = op_state.get("concept_running_residual_sum")
        pair_weight = op_state.get("concept_running_pair_weight")
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

    * ``"streaming"``: maintain per-group running weighted state sums and weight totals in this op's
      declared ``op_state`` (``concept_running_state_sum_a``, ``concept_running_weight_a``,
      ``concept_running_state_sum_b``, ``concept_running_weight_b``). Each per-batch invocation
      updates the running aggregates from this batch's per-batch ``concept_latent_state`` /
      ``concept_group_id`` / ``concept_example_weight`` tensors and recomputes the current concept
      direction from the accumulated sums; the final batch's emitted direction is the converged
      result. Memory cost is O(d_model * num_groups) instead of O(num_examples * d_model). For
      ``paired_rejection``, additionally maintains pending per-group buffers
      (``concept_pending_{a,b}_states``, ``concept_pending_{a,b}_weights``) plus running residual
      and pair-weight totals (``concept_running_residual_sum``, ``concept_running_pair_weight``);
      pairs are matched by stable iteration order (matching the legacy in_memory contract). The
      full storage-contract field set is exported via :data:`CONCEPT_STREAMING_STATE_FIELDS`.
    * ``"in_memory"`` (legacy): consume the aggregate row tensors emitted by
      :func:`extract_concept_latent_examples_impl` in legacy mode and compute the direction over
      the full materialized example set. Supports all direction modes.

    If ``concept_aggregate_output_mode`` is not set, behavior is determined by what is on the
    batch: aggregate row fields trigger the legacy path; per-batch fields with a bound op-state
    container trigger streaming. If neither is available, fall back to a token-group embedding
    direction computed from the model's input embedding matrix.
    """
    aggregate_mode = analysis_batch.get("concept_aggregate_output_mode")
    analysis_inputs = kwargs.get("analysis_inputs")
    op_state = getattr(analysis_inputs, "op_state", None) if analysis_inputs is not None else None

    use_streaming = aggregate_mode == "streaming"
    if not use_streaming and aggregate_mode is None:
        # Auto-detect: prefer legacy path if aggregate rows are already present
        legacy_rows_present = resolve_aggregate_input(module, analysis_batch, "concept_latent_state_rows") is not None
        per_batch_state_present = resolve_aggregate_input(module, analysis_batch, "concept_latent_state") is not None
        use_streaming = (not legacy_rows_present) and per_batch_state_present and op_state is not None

    if use_streaming:
        if op_state is None:
            # Previously this reached the accumulators with store=None, where the writes were
            # swallowed by `except Exception: pass` and the failure surfaced several frames later as
            # "requires at least one group A example". Say what is actually missing.
            raise ValueError(
                "concept_direction streaming mode accumulates across batches, which requires this op's "
                "declared op_state. Run it through an AnalysisCfg (a runner, or "
                "interpretune.analysis.execution.execute_analysis_op) so the state container has a "
                "lifecycle owner."
            )
        return _concept_direction_streaming(module, analysis_batch, op_state)

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

        latent_states, group_ids, example_weights, flattened_group_names = flatten_concept_store_rows(
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
    require_backend_capability(model_backend, BackendCapability.INTERVENTION, "concept_intervention")
    interventions = resolve_interventions(
        analysis_batch=analysis_batch,
        resolve_field=lambda field_name: resolve_aggregate_input(module, analysis_batch, field_name),
        load_json_field=lambda field_name: load_json_field(module, analysis_batch, field_name),
        kwargs=kwargs,
    )
    # Gate the scope and mode axes HERE, before the backend canonicalizes against hook shapes: a backend
    # that cannot honour a scope or a mode must refuse by name rather than apply the one it has.
    require_intervention_support(model_backend, interventions, backend_name=type(model_backend).__name__)
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


__all__ = [
    "CONCEPT_AGGREGATE_ROW_FIELDS",
    "CONCEPT_STREAMING_GROUP_FIELDS",
    "CONCEPT_STREAMING_PAIRED_REJECTION_FIELDS",
    "CONCEPT_STREAMING_STATE_FIELDS",
    "concept_direction_impl",
    "concept_target_token_ids",
    "extract_concept_latent_examples_impl",
    "extract_concept_latent_state_from_cache",
    "extract_concept_latent_state_impl",
    "flatten_concept_store_rows",
    "model_fwd_intervention_impl",
    "project_context_enhanced_states",
    "reset_concept_streaming_state",
    "resolve_concept_cache_key",
]
