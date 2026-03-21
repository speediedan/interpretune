"""Shared helpers for analysis op implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from interpretune.analysis.backends import get_analysis_backend, get_model_backend


_MISSING = object()
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
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype) if dtype is not None else values
    if isinstance(values, list | tuple):
        values = list(values)
        if not values:
            target_dtype = dtype if dtype is not None else torch.float32
            return torch.empty((0,), dtype=target_dtype)
        if all(isinstance(value, torch.Tensor) for value in values):
            stacked = torch.stack([value.detach().cpu() for value in values])
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
    "mean_target_logit_delta",
    "require_model_backend",
    "resolve_embedding_weight",
    "resolve_aggregate_input",
    "resolve_tokenizer",
    "stack_column_tensors",
    "token_strings_to_ids",
    "token_strings_to_last_ids",
    "weighted_mean",
]
