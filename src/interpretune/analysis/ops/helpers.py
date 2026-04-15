"""Shared helpers for analysis op implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interpretune.analysis.ops.base import AnalysisBatch

import torch

from interpretune.analysis.backends import (
    FeatureSelectionSpec,
    apply_feature_selection_filter,
    get_analysis_backend,
    get_model_backend,
)


_MISSING = object()


# TODO: Split feature-selection / top-feature ranking helpers into a dedicated module once the
# constrained-selection and intervention APIs stabilize.


def _mean_with_fallback(values: torch.Tensor, mask: torch.Tensor, *, default: float = 0.0) -> float:
    if values.numel() == 0:
        return default
    if mask.numel() > 0 and mask.any():
        return float(values[mask].mean().item())
    return float(values.mean().item())


def _apply_feature_activation_overrides(
    feature_rows: torch.Tensor,
    activation_values: torch.Tensor | None,
    feature_selection: FeatureSelectionSpec,
) -> torch.Tensor | None:
    if not feature_selection.activation_overrides:
        return activation_values

    if activation_values is None:
        activation_values = torch.zeros(feature_rows.shape[0], dtype=torch.float32)
    else:
        activation_values = activation_values.clone()

    for (layer, feature_id), value in feature_selection.activation_overrides.items():
        match_mask = (feature_rows[:, 0] == int(layer)) & (feature_rows[:, 2] == int(feature_id))
        if match_mask.any():
            activation_values[match_mask] = float(value)
    return activation_values


def _augment_feature_rows_for_selection(
    feature_rows: torch.Tensor,
    scores: torch.Tensor,
    activation_values: torch.Tensor | None,
    feature_selection: FeatureSelectionSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    requested_pairs = list(dict.fromkeys(feature_selection.layer_feature_pairs))
    if not requested_pairs:
        return (
            feature_rows,
            scores,
            _apply_feature_activation_overrides(feature_rows, activation_values, feature_selection),
        )

    existing_triples = {tuple(int(value) for value in row) for row in feature_rows.tolist()}
    all_positions = (
        torch.unique(feature_rows[:, 1], sorted=True)
        if feature_rows.shape[0] > 0
        else torch.empty((0,), dtype=torch.long)
    )

    appended_rows: list[tuple[int, int, int]] = []
    appended_scores: list[float] = []
    appended_activations: list[float] = []

    for layer, feature_id in requested_pairs:
        layer_number = int(layer)
        feature_number = int(feature_id)
        same_layer_mask = feature_rows[:, 0] == layer_number
        layer_positions = (
            torch.unique(feature_rows[same_layer_mask, 1], sorted=True) if same_layer_mask.any() else all_positions
        )
        if layer_positions.numel() == 0:
            layer_positions = torch.tensor([0], dtype=torch.long)

        score_baseline = _mean_with_fallback(scores, same_layer_mask, default=0.0)

        activation_baseline: float | None = None
        if activation_values is not None and activation_values.shape[0] == feature_rows.shape[0]:
            override_key = (layer_number, feature_number)
            if override_key in feature_selection.activation_overrides:
                activation_baseline = float(feature_selection.activation_overrides[override_key])
            else:
                activation_baseline = _mean_with_fallback(activation_values, same_layer_mask, default=0.0)

        for position in layer_positions.tolist():
            triple = (layer_number, int(position), feature_number)
            if triple in existing_triples:
                continue
            existing_triples.add(triple)
            appended_rows.append(triple)
            appended_scores.append(score_baseline)
            if activation_baseline is not None:
                appended_activations.append(activation_baseline)

    if appended_rows:
        feature_rows = torch.cat((feature_rows, torch.tensor(appended_rows, dtype=feature_rows.dtype)), dim=0)
        scores = torch.cat((scores, torch.tensor(appended_scores, dtype=scores.dtype)), dim=0)
        if activation_values is not None and appended_activations:
            activation_values = torch.cat(
                (activation_values, torch.tensor(appended_activations, dtype=activation_values.dtype)),
                dim=0,
            )

    activation_values = _apply_feature_activation_overrides(feature_rows, activation_values, feature_selection)
    return feature_rows, scores, activation_values


# Re-export for backwards compatibility
__all__ = ["FeatureSelectionSpec", "apply_feature_selection_filter"]


AnalysisScope = str
DEFAULT_ANALYSIS_SCOPES: tuple[AnalysisScope, ...] = ("analysis_batch", "batch", "run", "row", "store")


def extract_logits(output: Any) -> torch.Tensor:
    """Extract a logits tensor from framework-specific model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"Cannot extract logits from model output of type {type(output).__name__}")


def last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the final-token logits as a detached CPU tensor."""
    if logits.dim() == 1:
        return logits.detach().cpu()
    if logits.dim() == 2:
        return logits[-1].detach().cpu()
    if logits.dim() >= 3:
        return logits[0, -1].detach().cpu()
    raise ValueError(f"Unsupported logits rank for feature intervention output: {logits.dim()}")


def mean_target_logit_delta(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Return the mean delta over requested target ids, or over all logits if none are given."""
    if target_ids is not None and torch.numel(target_ids) > 0:
        target_ids = target_ids.to(dtype=torch.long).reshape(-1)
        vocab_size = pre_logits.size(0)
        oob = target_ids >= vocab_size
        if oob.any():
            raise ValueError(
                f"logit_target_ids contain out-of-bounds indices (>= vocab_size {vocab_size}): "
                f"{target_ids[oob].tolist()}. Virtual IDs from concept-direction targets must be "
                "resolved before intervention."
            )
        return (post_logits.index_select(0, target_ids) - pre_logits.index_select(0, target_ids)).mean()
    return (post_logits - pre_logits).mean()


def stack_column_tensors(values: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Normalize dataset or run-input column values into a tensor."""

    def _combine_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
        if tensors[0].ndim > 1:
            try:
                return torch.cat(tensors, dim=0)
            except RuntimeError:
                return torch.stack(tensors)
        return torch.stack(tensors)

    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype) if dtype is not None else values
    if isinstance(values, list | tuple):
        values = list(values)
        if not values:
            target_dtype = dtype if dtype is not None else torch.float32
            return torch.empty((0,), dtype=target_dtype)
        if all(isinstance(value, torch.Tensor) for value in values):
            tensors = [value.detach().cpu() for value in values]
            stacked = _combine_tensors(tensors)
            return stacked.to(dtype=dtype) if dtype is not None else stacked
        tensor_values = []
        for value in values:
            tensor_value = torch.as_tensor(value)
            tensor_values.append(tensor_value.detach().cpu())
        stacked = _combine_tensors(tensor_values)
        return stacked.to(dtype=dtype) if dtype is not None else stacked
    return torch.as_tensor(values, dtype=dtype)


def require_model_backend(module: Any) -> Any:
    """Return a model backend from either ``_model_backend`` or ``model_backend``."""
    backend = get_model_backend(module)
    if backend is None:
        raise ValueError("Target module must expose a model backend for this operation")
    return backend


def resolve_aggregate_input(module: Any, analysis_batch: Any, field_name: str) -> Any:
    """Resolve whole-column aggregation inputs, preferring explicit run or batch values over the input store."""
    analysis_cfg = getattr(module, "analysis_cfg", None)
    batch_inputs = getattr(analysis_cfg, "batch_inputs", {}) or {}
    run_inputs = getattr(analysis_cfg, "run_inputs", {}) or {}

    for scoped_values in (batch_inputs, run_inputs):
        if field_name in scoped_values and scoped_values[field_name] is not None:
            return scoped_values[field_name]

    if hasattr(analysis_batch, "keys") and field_name in analysis_batch.keys():
        return getattr(analysis_batch, field_name)

    input_store = getattr(analysis_cfg, "input_store", None)
    if input_store is not None:
        dataset = getattr(input_store, "dataset", None)
        raw_column_names = getattr(dataset, "column_names", None) if dataset is not None else None
        column_names = list(raw_column_names) if raw_column_names is not None else []
        if field_name in column_names:
            return input_store[field_name]
        store_value = getattr(input_store, field_name, None)
        if store_value is not None:
            return store_value

    return None


def load_json_field(module: Any, analysis_batch: Any, field_name: str) -> Any:
    """Resolve an aggregate input field and decode JSON string payloads when present."""

    raw_value = resolve_aggregate_input(module, analysis_batch, field_name)
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    return raw_value


def weighted_mean(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute a stable weighted mean over state rows."""
    weights = weights.to(device=states.device, dtype=states.dtype).reshape(-1)
    weight_sum = weights.sum().clamp_min(1e-12)
    return (states * weights.unsqueeze(-1)).sum(dim=0) / weight_sum


def _resolve_attr_path(root: Any, *path: str) -> Any | None:
    current = root
    for attr_name in path:
        current = getattr(current, attr_name, None)
        if current is None:
            return None
    return current


def _flatten_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, torch.Tensor):
        return [int(value) for value in tokenized.reshape(-1).tolist()]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list):
        if tokenized and isinstance(tokenized[0], list):
            return [int(value) for sublist in tokenized for value in sublist]
        return [int(value) for value in tokenized]
    return [int(tokenized)]


def _value_for_batch(store_value: Any, batch_idx: int | None) -> Any:
    if batch_idx is None or isinstance(store_value, (str, bytes, Mapping)):
        return store_value
    if isinstance(store_value, torch.Tensor):
        return store_value if store_value.dim() == 0 else store_value[batch_idx]
    if isinstance(store_value, (list, tuple)):
        return store_value[batch_idx]
    if hasattr(store_value, "__getitem__"):
        try:
            return store_value[batch_idx]
        except Exception:
            return store_value
    return store_value


def _lookup_mapping_or_attr(container: Any, field_name: str) -> Any:
    if container is None:
        return _MISSING
    if isinstance(container, Mapping):
        try:
            value = container[field_name] if field_name in container else _MISSING
        except Exception:
            value = _MISSING
        return _MISSING if value is None else value

    value = getattr(container, field_name, _MISSING)
    if value is not _MISSING and value is not None:
        return value

    dataset = getattr(container, "dataset", None)
    column_names = getattr(dataset, "column_names", []) if dataset is not None else []
    if field_name not in column_names:
        return _MISSING

    try:
        value = container[field_name]
    except Exception:
        return _MISSING
    return _MISSING if value is None else value


def _lookup_store_row_value(store: Any, field_name: str, batch_idx: int | None) -> Any:
    if store is None or batch_idx is None:
        return _MISSING

    dataset = getattr(store, "dataset", None)
    column_names = getattr(dataset, "column_names", []) if dataset is not None else []
    if dataset is not None and field_name in column_names:
        try:
            row = dataset[batch_idx]
        except Exception:
            row = None
        if isinstance(row, Mapping):
            value = row[field_name] if field_name in row else _MISSING
            if value is not None and value is not _MISSING:
                return value

    store_value = _lookup_mapping_or_attr(store, field_name)
    if store_value is _MISSING:
        return _MISSING
    return _value_for_batch(store_value, batch_idx)


@dataclass(kw_only=True)
class AnalysisInputs:
    """Explicit scoped analysis inputs used during op execution."""

    row: Mapping[str, Any] | None = None
    batch: Mapping[str, Any] | None = None
    run: Mapping[str, Any] | None = None
    store: Any = None

    def merged(self, other: "AnalysisInputs | Mapping[str, Any] | None") -> "AnalysisInputs":
        other_inputs = coerce_analysis_inputs(other)
        if other_inputs is None:
            return AnalysisInputs(row=self.row, batch=self.batch, run=self.run, store=self.store)

        def merge_mapping(
            base: Mapping[str, Any] | None,
            override: Mapping[str, Any] | None,
        ) -> Mapping[str, Any] | None:
            if base and override:
                return {**base, **override}
            return override if override is not None else base

        return AnalysisInputs(
            row=merge_mapping(self.row, other_inputs.row),
            batch=merge_mapping(self.batch, other_inputs.batch),
            run=merge_mapping(self.run, other_inputs.run),
            store=other_inputs.store if other_inputs.store is not None else self.store,
        )

    def resolve_scope(self, scope: AnalysisScope, field_name: str, batch_idx: int | None = None) -> Any:
        if scope == "row":
            value = _lookup_mapping_or_attr(self.row, field_name)
            if value is not _MISSING:
                return value
            return _lookup_store_row_value(self.store, field_name, batch_idx)
        if scope == "batch":
            return _lookup_mapping_or_attr(self.batch, field_name)
        if scope == "run":
            return _lookup_mapping_or_attr(self.run, field_name)
        if scope == "store":
            return _lookup_mapping_or_attr(self.store, field_name)
        raise ValueError(f"Unsupported analysis input scope: {scope}")


@dataclass(kw_only=True)
class AnalysisValueResolver:
    """Resolve analysis values using explicit precedence across analysis scopes."""

    analysis_batch: Any
    analysis_inputs: AnalysisInputs
    batch_idx: int | None = None

    def resolve(
        self,
        field_name: str,
        *,
        default: Any = None,
        scopes: tuple[AnalysisScope, ...] = DEFAULT_ANALYSIS_SCOPES,
    ) -> Any:
        for scope in scopes:
            if scope == "analysis_batch":
                value = _lookup_mapping_or_attr(self.analysis_batch, field_name)
            else:
                value = self.analysis_inputs.resolve_scope(scope, field_name, batch_idx=self.batch_idx)
            if value is not _MISSING:
                return value
        return default


def coerce_analysis_inputs(value: AnalysisInputs | Mapping[str, Any] | None) -> AnalysisInputs | None:
    """Normalize user-provided analysis inputs into an AnalysisInputs object."""
    if value is None:
        return None
    if isinstance(value, AnalysisInputs):
        return value
    if isinstance(value, Mapping):
        return AnalysisInputs(run=dict(value))
    raise TypeError(f"Unsupported analysis_inputs value: {type(value).__name__}")


def get_analysis_resolver(
    analysis_batch: Any,
    module: Any,
    *,
    batch_idx: int | None = None,
    analysis_inputs: AnalysisInputs | Mapping[str, Any] | None = None,
) -> AnalysisValueResolver:
    """Build a resolver that combines config-backed and explicit analysis input scopes."""
    analysis_cfg = getattr(module, "analysis_cfg", None)
    config_inputs = AnalysisInputs(
        batch=getattr(analysis_cfg, "batch_inputs", None),
        run=getattr(analysis_cfg, "run_inputs", None),
        store=getattr(analysis_cfg, "input_store", None),
    )
    resolved_inputs = config_inputs.merged(analysis_inputs)
    return AnalysisValueResolver(analysis_batch=analysis_batch, analysis_inputs=resolved_inputs, batch_idx=batch_idx)


def get_input_store_value(
    module: Any,
    field_name: str,
    batch_idx: int | None = None,
    *,
    scope: AnalysisScope = "row",
) -> Any:
    """Read a scoped value from ``module.analysis_cfg.input_store`` when present."""
    return get_analysis_resolver(None, module, batch_idx=batch_idx).resolve(field_name, default=None, scopes=(scope,))


def get_analysis_value(
    analysis_batch: Any,
    module: Any,
    field_name: str,
    batch_idx: int | None = None,
    default: Any = None,
    *,
    analysis_inputs: AnalysisInputs | Mapping[str, Any] | None = None,
    scopes: tuple[AnalysisScope, ...] = DEFAULT_ANALYSIS_SCOPES,
) -> Any:
    """Resolve a field from the active analysis context.

    This compatibility helper now delegates to :class:`AnalysisValueResolver`. New code should prefer using
    :func:`get_analysis_resolver` directly when multiple scoped lookups are needed inside an op implementation.
    """
    resolver = get_analysis_resolver(
        analysis_batch,
        module,
        batch_idx=batch_idx,
        analysis_inputs=analysis_inputs,
    )
    return resolver.resolve(field_name, default=default, scopes=scopes)


def resolve_tokenizer(module: Any) -> Any:
    """Resolve a tokenizer from a generic module or its analysis backend."""
    analysis_backend = get_analysis_backend(module)
    if analysis_backend is not None:
        try:
            return analysis_backend.get_tokenizer(module)
        except (AttributeError, ValueError):
            pass

    for path in (
        ("replacement_model", "tokenizer"),
        ("model", "tokenizer"),
        ("datamodule", "tokenizer"),
        ("tokenizer",),
    ):
        value = _resolve_attr_path(module, *path)
        if value is not None:
            return value

    raise ValueError("A tokenizer is required for this analysis operation")


def resolve_embedding_weight(module: Any) -> torch.Tensor:
    """Resolve an embedding weight matrix from a generic module or its analysis backend."""
    analysis_backend = get_analysis_backend(module)
    if analysis_backend is not None:
        try:
            return analysis_backend.get_embedding_weight(module)
        except (AttributeError, ValueError):
            pass

    for path in (
        ("replacement_model", "unembed_weight"),
        ("model", "unembed_weight"),
        ("replacement_model", "embed_weight"),
        ("model", "embed_weight"),
        ("replacement_model", "W_E"),
        ("model", "W_E"),
        ("replacement_model", "embed", "W_E"),
        ("model", "embed", "W_E"),
    ):
        value = _resolve_attr_path(module, *path)
        if isinstance(value, torch.Tensor):
            return value

    for attr_name in ("replacement_model", "model"):
        model = getattr(module, attr_name, None)
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embedding_layer = get_input_embeddings()
            weight = getattr(embedding_layer, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight

    raise ValueError("An embedding weight matrix is required for concept_direction")


def token_strings_to_ids(tokenizer: Any, token_strings: list[str]) -> list[int]:
    """Resolve token strings to token ids using either the vocab or tokenizer call path."""
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    token_ids: list[int] = []
    for token_str in token_strings:
        if token_str in vocab:
            token_ids.append(int(vocab[token_str]))
            continue
        tokenized = tokenizer(token_str, add_special_tokens=False)["input_ids"]
        token_ids.extend(_flatten_token_ids(tokenized))
    if not token_ids:
        raise ValueError("Unable to resolve any token ids for the provided concept groups")
    return token_ids


def token_strings_to_last_ids(tokenizer: Any, token_strings: list[str]) -> list[int]:
    """Resolve each token string to its terminal token id.

    This preserves one id per input token string, which is required for paired concept-direction constructions such as
    vector rejection.
    """

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    token_ids: list[int] = []
    for token_str in token_strings:
        if token_str in vocab:
            token_ids.append(int(vocab[token_str]))
            continue
        tokenized = tokenizer(token_str, add_special_tokens=False)["input_ids"]
        flattened = _flatten_token_ids(tokenized)
        if not flattened:
            raise ValueError(f"Unable to resolve a terminal token id for {token_str!r}")
        token_ids.append(int(flattened[-1]))
    if not token_ids:
        raise ValueError("Unable to resolve any token ids for the provided concept groups")
    return token_ids


def decode_token_ids(tokenizer: Any, token_ids: torch.Tensor | list[int]) -> list[str]:
    """Decode individual token ids to token strings when possible."""
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(tokenizer.convert_ids_to_tokens(int(token_id))) for token_id in ids]
    return [str(tokenizer.decode([int(token_id)], skip_special_tokens=False)) for token_id in ids]


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


def _resolve_concept_cache_key(analysis_batch: AnalysisBatch) -> str:
    return str(analysis_batch.get("concept_cache_key") or "unembed.hook_in")


# TODO: This may be better cast as a separate op itself rather than a helper, we should revisit
def _extract_concept_latent_state_from_cache(
    analysis_batch: AnalysisBatch,
    context_enhanced: bool = False,
    context_scale: float = 1.0,
) -> tuple[torch.Tensor, str]:
    cache = analysis_batch.cache
    answer_indices = analysis_batch.answer_indices
    if cache is None or answer_indices is None:
        raise ValueError("extract_concept_latent_state requires cache and answer_indices")

    cache_key = _resolve_concept_cache_key(analysis_batch)
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
            prev_indices = index_tensor - 1
            valid = prev_indices >= 0
            prev_clamped = prev_indices.clamp(min=0)
            context_states = cache_tensor[batch_indices, prev_clamped].detach().cpu().float()

            scaled_answer = context_scale * latent_states
            dot_num = (scaled_answer * context_states).sum(dim=-1, keepdim=True)
            dot_den = (context_states * context_states).sum(dim=-1, keepdim=True).clamp(min=1e-12)
            projected = (dot_num / dot_den) * context_states

            valid_expanded = valid.unsqueeze(-1).expand_as(latent_states)
            latent_states = torch.where(valid_expanded, projected, latent_states)
    else:
        latent_states = cache_tensor.detach().cpu().float()

    return latent_states, cache_key


def _flatten_concept_store_rows(
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


__all__ = [
    "AnalysisInputs",
    "AnalysisValueResolver",
    "concept_target_token_ids",
    "coerce_analysis_inputs",
    "DEFAULT_ANALYSIS_SCOPES",
    "decode_token_ids",
    "extract_logits",
    "get_analysis_value",
    "get_analysis_resolver",
    "get_input_store_value",
    "last_token_logits",
    "load_json_field",
    "mean_target_logit_delta",
    "require_model_backend",
    "resolve_embedding_weight",
    "resolve_aggregate_input",
    "resolve_tokenizer",
    "stack_column_tensors",
    "token_strings_to_ids",
    "token_strings_to_last_ids",
    "weighted_mean",
    "_resolve_concept_cache_key",
    "_extract_concept_latent_state_from_cache",
    "_flatten_concept_store_rows",
]
