"""Shared helpers for analysis op implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from interpretune.analysis.backends import get_analysis_backend


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


def get_input_store_value(module: Any, field_name: str, batch_idx: int | None = None) -> Any:
    """Read a value from module.analysis_cfg.input_store when present."""
    analysis_cfg = getattr(module, "analysis_cfg", None)
    input_store = getattr(analysis_cfg, "input_store", None)
    if input_store is None:
        return None
    store_value = getattr(input_store, field_name, None)
    if store_value is None and hasattr(input_store, "dataset"):
        dataset = getattr(input_store, "dataset", None)
        column_names = getattr(dataset, "column_names", []) if dataset is not None else []
        if field_name in column_names:
            try:
                store_value = input_store[field_name]
            except Exception:
                store_value = None
    if store_value is None:
        return None
    return _value_for_batch(store_value, batch_idx)


def get_analysis_value(
    analysis_batch: Any,
    module: Any,
    field_name: str,
    batch_idx: int | None = None,
    default: Any = None,
) -> Any:
    """Resolve a field from the in-flight batch first, then fall back to input_store."""
    value = getattr(analysis_batch, field_name, None)
    if value is None and isinstance(analysis_batch, Mapping):
        value = analysis_batch.get(field_name)
    if value is not None:
        return value
    value = get_input_store_value(module, field_name, batch_idx=batch_idx)
    return default if value is None else value


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
    "concept_target_token_ids",
    "decode_token_ids",
    "extract_logits",
    "get_analysis_value",
    "get_input_store_value",
    "last_token_logits",
    "mean_target_logit_delta",
    "resolve_embedding_weight",
    "resolve_tokenizer",
    "token_strings_to_ids",
    "token_strings_to_last_ids",
]
