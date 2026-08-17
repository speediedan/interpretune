"""The Interpretune op-authoring toolkit.

This module is the sanctioned shared surface for analysis-op implementations. Bundled, local, and
hub op collections may all import from it: interpretune is by definition installed wherever an op
runs, so depending on this module is a declared, supported contract rather than a reach into package
internals. Op YAML ``importable_params`` entries may also reference callables in this namespace.

Everything exported here is expected to remain stable for op authors (within the project's pre-MVP
caveats); anything not exported is internal. Backend-specific behavior stays behind the
:mod:`interpretune.analysis.backends` capability seam, which op implementations may also use.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal

import torch
from jaxtyping import Float

from interpretune.analysis.backends import get_analysis_backend, get_model_backend
from interpretune.analysis.inputs import _resolve_attr_path
from interpretune.protocol import DefaultAnalysisBatchProtocol

FEATURE_SCORE_SOURCE_ALIASES: dict[str, str] = {
    "influence": "node_influence_scores",
    "absolute_influence": "node_influence_scores",
    "signed_influence": "node_signed_influence_scores",
    "gradient": "node_logit_diff_gradient_scores",
    "gradients": "node_logit_diff_gradient_scores",
    "logit_diff_gradient": "node_logit_diff_gradient_scores",
    "target_logit_diff_gradient": "node_logit_diff_gradient_scores",
}


def resolve_feature_score_source(score_source: str | None) -> str | None:
    """Normalize user-facing score-source aliases to analysis-batch field names."""
    if score_source is None:
        return None
    return FEATURE_SCORE_SOURCE_ALIASES.get(score_source, score_source)


# ---------------------------------------------------------------------------
# Tensor / logits utilities
# ---------------------------------------------------------------------------


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


def weighted_mean(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute a stable weighted mean over state rows."""
    weights = weights.to(device=states.device, dtype=states.dtype).reshape(-1)
    weight_sum = weights.sum().clamp_min(1e-12)
    return (states * weights.unsqueeze(-1)).sum(dim=0) / weight_sum


# ---------------------------------------------------------------------------
# Classification-logit utilities (op ``importable_params`` targets)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model-access resolution (backend-agnostic, per the composition guide)
# ---------------------------------------------------------------------------


def require_model_backend(module: Any) -> Any:
    """Return a model backend from either ``_model_backend`` or ``model_backend``."""
    backend = get_model_backend(module)
    if backend is None:
        raise ValueError("Target module must expose a model backend for this operation")
    return backend


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


# ---------------------------------------------------------------------------
# Tokenization utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scoped-input conveniences for op implementations
# ---------------------------------------------------------------------------
# Sanctioned bridge for whole-column aggregate inputs until the declared op-state /
# declared-inputs work lands; new op code should prefer the bound ``AnalysisBatch``
# access surface for batch-scoped values.


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


__all__ = [
    "boolean_logits_to_avg_logit_diff",
    "decode_token_ids",
    "extract_logits",
    "FEATURE_SCORE_SOURCE_ALIASES",
    "get_loss_preds_diffs",
    "last_token_logits",
    "load_json_field",
    "mean_target_logit_delta",
    "require_model_backend",
    "resolve_aggregate_input",
    "resolve_embedding_weight",
    "resolve_feature_score_source",
    "resolve_tokenizer",
    "stack_column_tensors",
    "token_strings_to_ids",
    "token_strings_to_last_ids",
    "weighted_mean",
]
