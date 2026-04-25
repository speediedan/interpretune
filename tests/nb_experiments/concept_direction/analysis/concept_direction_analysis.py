"""Utilities for embed-vs-store concept-direction parity debugging.

This module stays test-side on purpose. It reconstructs prompt/token alignment and context-enhanced extraction math from
existing analysis-batch inputs so parity tests can emit actionable diagnostics without widening the production op
contracts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PRESERVE_ARTIFACTS_ENV = "IT_PARITY_PRESERVE_ARTIFACTS"
PRESERVE_ARTIFACT_DIR_ENV = "IT_PARITY_PRESERVE_ARTIFACT_DIR"
ARTIFACT_GENERATION_ENV = "IT_CONCEPT_DIRECTION_ARTIFACT_GENERATION"
ARTIFACT_METADATA_KEY = "artifact_metadata"
DEFAULT_CONCEPT_DIRECTION_ARTIFACT_ROOT = "/tmp/it_concept_direction_analysis_artifacts"
PARITY_REPORT_FILE_BASENAME = "concept_direction_parity_report"
PIPELINE_STATE_FILE_BASENAME = "concept_direction_pipeline_state_artifacts"
REFERENCE_GRAPH_REPORT_BASENAME = "concept_direction_reference_graph_report"
REFERENCE_GRAPH_REPORT_FILE = "concept_direction_reference_graph_report.json"
DEFAULT_RANDOM_PERTURBATION_SEED = 17
DEFAULT_RANDOM_PERTURBATION_SCALE = 10.0

_REPORT_BASENAME_TO_KIND = {
    PARITY_REPORT_FILE_BASENAME: "parity_report",
    PIPELINE_STATE_FILE_BASENAME: "pipeline_state_artifacts",
    REFERENCE_GRAPH_REPORT_BASENAME: "reference_graph_report",
}
_VERSIONED_REPORT_PATTERN = re.compile(
    r"^(?P<basename>concept_direction_(?:parity_report|pipeline_state_artifacts|reference_graph_report))"
    r"(?:_(?P<generation>\d{8}_\d{6}))?\.json$"
)


def normalize_prompt_entity_text(entity_name: str) -> str:
    """Normalize token-like entity strings into prompt surface text."""

    raw_value = str(entity_name)
    normalized = raw_value.replace("▁", " ").replace("Ġ", " ")
    normalized = " ".join(normalized.split())
    return normalized or raw_value


def build_classification_prompt_text(entity_name: str, question: str) -> str:
    """Build a classification prompt with a normalized probe surface."""

    return f"{question} {normalize_prompt_entity_text(entity_name)} : "


def _tensor_sha256(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _tensor_summary(value: Any, *, include_values_below: int | None = None) -> dict[str, Any]:
    tensor = torch.as_tensor(value).detach().cpu().float()
    summary = {
        "shape": list(tensor.shape),
        "norm": float(torch.linalg.vector_norm(tensor).item()),
        "mean_abs": float(tensor.abs().mean().item()) if tensor.numel() else 0.0,
        "max_abs": float(tensor.abs().max().item()) if tensor.numel() else 0.0,
        "sha256": _tensor_sha256(tensor),
    }
    if include_values_below is not None and tensor.numel() <= include_values_below:
        summary["values"] = tensor.tolist()
    return summary


def cosine_similarity_value(left: Any, right: Any) -> float:
    """Return cosine similarity for two vectors after flattening on CPU."""

    left_tensor = torch.as_tensor(left, dtype=torch.float32).detach().cpu().reshape(1, -1)
    right_tensor = torch.as_tensor(right, dtype=torch.float32).detach().cpu().reshape(1, -1)
    return float(torch.nn.functional.cosine_similarity(left_tensor, right_tensor).item())


def build_random_vector_perturbation(
    base_direction: Any,
    *,
    scale: float,
    seed: int,
    orthogonalize: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a reproducible large random-vector perturbation of a concept direction."""

    base_tensor = torch.as_tensor(base_direction, dtype=torch.float32).detach().cpu().reshape(-1)
    base_norm = torch.linalg.vector_norm(base_tensor)
    if not torch.isfinite(base_norm) or base_norm.item() <= 0:
        raise ValueError("build_random_vector_perturbation requires a non-zero base direction")

    normalized_base = base_tensor / base_norm
    generator = torch.Generator(device=normalized_base.device)
    generator.manual_seed(int(seed))
    sampled_random_vector = torch.randn(normalized_base.shape, generator=generator, dtype=torch.float32)
    sampled_random_norm = torch.linalg.vector_norm(sampled_random_vector)
    if not torch.isfinite(sampled_random_norm) or sampled_random_norm.item() <= 0:
        raise ValueError("build_random_vector_perturbation generated a degenerate sampled random vector")
    sampled_random_unit = sampled_random_vector / sampled_random_norm

    random_vector = sampled_random_vector

    if orthogonalize:
        random_vector = random_vector - torch.dot(random_vector, normalized_base) * normalized_base
    random_norm = torch.linalg.vector_norm(random_vector)
    if not torch.isfinite(random_norm) or random_norm.item() <= 0:
        raise ValueError("build_random_vector_perturbation generated a degenerate random vector")

    random_unit = random_vector / random_norm
    perturbed_raw = normalized_base + float(scale) * random_unit
    perturbed_norm = torch.linalg.vector_norm(perturbed_raw)
    if not torch.isfinite(perturbed_norm) or perturbed_norm.item() <= 0:
        raise ValueError("build_random_vector_perturbation produced a zero-norm perturbed direction")

    perturbed_direction = perturbed_raw / perturbed_norm
    metadata = {
        "seed": int(seed),
        "scale": float(scale),
        "orthogonalize": bool(orthogonalize),
        "base_direction": _tensor_summary(normalized_base, include_values_below=64),
        "sampled_random_unit": _tensor_summary(sampled_random_unit, include_values_below=64),
        "random_unit": _tensor_summary(random_unit, include_values_below=64),
        "perturbed_raw": _tensor_summary(perturbed_raw, include_values_below=64),
        "perturbed_direction": _tensor_summary(perturbed_direction, include_values_below=64),
        "sampled_random_direction_cosine_to_base": cosine_similarity_value(normalized_base, sampled_random_unit),
        "perturbation_basis_cosine_to_base": cosine_similarity_value(normalized_base, random_unit),
        "cosine_to_base": cosine_similarity_value(normalized_base, perturbed_direction),
    }
    return perturbed_direction, metadata


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            converted = tokenizer.convert_ids_to_tokens(list(token_ids))
        except Exception:
            return [str(tokenizer.convert_ids_to_tokens(int(token_id))) for token_id in token_ids]
        if isinstance(converted, list):
            return [str(token) for token in converted]
    return [str(tokenizer.decode([int(token_id)], skip_special_tokens=False)) for token_id in token_ids]


def build_indexed_token_debug_rows(
    input_ids: Sequence[int],
    input_tokens: Sequence[str],
    *,
    highlights: Mapping[str, int | Sequence[int] | None] | None = None,
) -> list[dict[str, Any]]:
    """Build a token-by-token debug table with named highlight markers."""

    highlight_map: dict[int, list[str]] = {}
    for label, raw_indices in (highlights or {}).items():
        if raw_indices is None:
            continue
        if isinstance(raw_indices, int):
            indices = [raw_indices]
        else:
            indices = [int(index) for index in raw_indices]
        for index in indices:
            if index < 0:
                continue
            highlight_map.setdefault(int(index), []).append(str(label))

    rows: list[dict[str, Any]] = []
    for index, (token_id, token_text) in enumerate(zip(input_ids, input_tokens, strict=False)):
        marks = highlight_map.get(index, [])
        rows.append(
            {
                "index": int(index),
                "token_id": int(token_id),
                "token_text": str(token_text),
                "marks": list(marks),
                "marks_text": " | ".join(marks),
            }
        )
    return rows


def build_prompt_alignment_artifact(
    snapshot: PromptAlignmentSnapshot,
    *,
    probe_surface_text: str | None = None,
    cache_rendered_prompt: str | None = None,
    cache_input_ids: Sequence[int] | None = None,
    cache_input_tokens: Sequence[str] | None = None,
    cache_answer_index: int | None = None,
    context_token_index: int | None = None,
    context_token_source: str | None = None,
) -> dict[str, Any]:
    """Augment a prompt-alignment snapshot with cache-space token diagnostics."""

    record = snapshot.to_dict()
    if probe_surface_text is not None:
        record["probe_surface_text"] = probe_surface_text
    if cache_input_ids is None:
        cache_input_ids = snapshot.input_ids[: snapshot.answer_index]
    if cache_input_tokens is None:
        cache_input_tokens = snapshot.input_tokens[: snapshot.answer_index]

    cache_input_ids_list = [int(value) for value in cache_input_ids]
    cache_input_tokens_list = [str(value) for value in cache_input_tokens]
    record["cache_rendered_prompt"] = cache_rendered_prompt or snapshot.rendered_prompt
    record["cache_input_ids"] = cache_input_ids_list
    record["cache_input_tokens"] = cache_input_tokens_list
    record["cache_answer_index"] = cache_answer_index
    record["context_token_index"] = context_token_index
    record["context_token_source"] = context_token_source
    record["full_prompt_token_debug"] = build_indexed_token_debug_rows(
        snapshot.input_ids,
        snapshot.input_tokens,
        highlights={
            "probe": ()
            if snapshot.probe_start_index is None or snapshot.probe_end_index is None
            else range(snapshot.probe_start_index, snapshot.probe_end_index + 1),
            "answer": snapshot.answer_index,
            "answer_previous": snapshot.previous_token_index,
            "context": context_token_index,
        },
    )
    record["cache_prompt_token_debug"] = build_indexed_token_debug_rows(
        cache_input_ids_list,
        cache_input_tokens_list,
        highlights={
            "cache_answer": cache_answer_index,
            "context": context_token_index,
        },
    )
    if cache_answer_index is not None and 0 <= cache_answer_index < len(cache_input_ids_list):
        record["cache_answer_token_id"] = int(cache_input_ids_list[cache_answer_index])
        record["cache_answer_token_text"] = str(cache_input_tokens_list[cache_answer_index])
    else:
        record["cache_answer_token_id"] = None
        record["cache_answer_token_text"] = None
    if context_token_index is not None and 0 <= context_token_index < len(cache_input_ids_list):
        record["context_token_id"] = int(cache_input_ids_list[context_token_index])
        record["context_token_text"] = str(cache_input_tokens_list[context_token_index])
    else:
        record["context_token_id"] = None
        record["context_token_text"] = None
    return record


def build_context_extraction_artifact(
    snapshot: ContextEnhancedExtractionSnapshot,
    *,
    prompt_alignment_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach prompt-space token diagnostics to a context-enhanced extraction snapshot."""

    record = snapshot.to_dict()
    if prompt_alignment_artifact is not None:
        for field_name in (
            "cache_rendered_prompt",
            "cache_input_ids",
            "cache_input_tokens",
            "cache_prompt_token_debug",
            "full_prompt_token_debug",
            "cache_answer_index",
            "cache_answer_token_id",
            "cache_answer_token_text",
            "context_token_index",
            "context_token_source",
            "context_token_id",
            "context_token_text",
            "probe_text",
            "probe_surface_text",
            "answer_text",
        ):
            if field_name in prompt_alignment_artifact:
                record[field_name] = prompt_alignment_artifact[field_name]
    return record


def compute_concept_direction_geometry(
    latent_states: Any,
    group_ids: Any,
    *,
    direction_mode: str,
    example_weights: Any | None = None,
) -> dict[str, Any]:
    """Reconstruct weighted direction geometry before normalization."""

    latent_states_tensor = torch.as_tensor(latent_states)
    states = (
        torch.as_tensor(latent_states, dtype=torch.float32)
        .detach()
        .cpu()
        .reshape(
            -1,
            latent_states_tensor.shape[-1],
        )
    )
    groups = torch.as_tensor(group_ids, dtype=torch.long).detach().cpu().reshape(-1)
    if states.shape[0] != groups.shape[0]:
        raise ValueError(
            "compute_concept_direction_geometry requires latent_states and group_ids to align "
            f"({states.shape[0]} vs {groups.shape[0]})"
        )
    if example_weights is None:
        weights = torch.ones((states.shape[0],), dtype=torch.float32)
    else:
        weights = torch.as_tensor(example_weights, dtype=torch.float32).detach().cpu().reshape(-1)
        if weights.shape[0] != states.shape[0]:
            raise ValueError(
                "compute_concept_direction_geometry requires example_weights to align with latent_states "
                f"({weights.shape[0]} vs {states.shape[0]})"
            )

    group_a_mask = groups == 0
    group_b_mask = groups == 1
    if not group_a_mask.any():
        raise ValueError("compute_concept_direction_geometry requires at least one group-a example")

    residuals: torch.Tensor | None = None
    pair_weights: torch.Tensor | None = None
    if direction_mode == "mean_difference":
        if not group_b_mask.any():
            raise ValueError("mean_difference requires group-b examples")
        group_a_mean = torch.sum(states[group_a_mask] * weights[group_a_mask].unsqueeze(-1), dim=0) / weights[
            group_a_mask
        ].sum().clamp_min(1e-12)
        group_b_mean = torch.sum(states[group_b_mask] * weights[group_b_mask].unsqueeze(-1), dim=0) / weights[
            group_b_mask
        ].sum().clamp_min(1e-12)
        raw_direction = group_a_mean - group_b_mean
    elif direction_mode == "paired_rejection":
        if not group_b_mask.any():
            raise ValueError("paired_rejection requires group-b examples")
        group_a_states = states[group_a_mask]
        group_b_states = states[group_b_mask]
        group_a_weights = weights[group_a_mask]
        group_b_weights = weights[group_b_mask]
        if group_a_states.shape[0] != group_b_states.shape[0]:
            raise ValueError("paired_rejection requires equal numbers of group-a and group-b examples")
        residual_rows: list[torch.Tensor] = []
        pair_weight_rows: list[torch.Tensor] = []
        for state_a, state_b, weight_a, weight_b in zip(
            group_a_states,
            group_b_states,
            group_a_weights,
            group_b_weights,
            strict=True,
        ):
            denom = torch.dot(state_b, state_b).clamp_min(1e-12)
            proj = (torch.dot(state_a, state_b) / denom) * state_b
            residual_rows.append(state_a - proj)
            pair_weight_rows.append((weight_a + weight_b) / 2)
        residuals = torch.stack(residual_rows)
        pair_weights = torch.stack(pair_weight_rows)
        assert pair_weights is not None
        raw_direction = torch.sum(residuals * pair_weights.unsqueeze(-1), dim=0) / pair_weights.sum().clamp_min(1e-12)
    elif direction_mode == "single_group":
        raw_direction = torch.sum(states[group_a_mask] * weights[group_a_mask].unsqueeze(-1), dim=0) / weights[
            group_a_mask
        ].sum().clamp_min(1e-12)
    else:
        raise ValueError(f"Unsupported direction_mode: {direction_mode}")

    raw_norm = torch.linalg.vector_norm(raw_direction)
    normalized_direction = (
        raw_direction / raw_norm if torch.isfinite(raw_norm) and raw_norm.item() > 0 else raw_direction
    )
    return {
        "raw_direction": raw_direction.detach().cpu(),
        "normalized_direction": normalized_direction.detach().cpu(),
        "raw_direction_summary": _tensor_summary(raw_direction, include_values_below=64),
        "normalized_direction_summary": _tensor_summary(normalized_direction, include_values_below=64),
        "latent_row_count": int(states.shape[0]),
        "latent_row_norms": [float(value) for value in torch.linalg.vector_norm(states, dim=-1).tolist()],
        "group_ids": [int(value) for value in groups.tolist()],
        "example_weights": [float(value) for value in weights.tolist()],
        "group_weight_sums": {
            "group_a": float(weights[group_a_mask].sum().item()),
            "group_b": float(weights[group_b_mask].sum().item()) if group_b_mask.any() else 0.0,
        },
        "pair_residual_norms": (
            []
            if residuals is None
            else [float(value) for value in torch.linalg.vector_norm(residuals, dim=-1).tolist()]
        ),
        "pair_weights": [] if pair_weights is None else [float(value) for value in pair_weights.tolist()],
    }


def build_concept_direction_stage_artifact(
    *,
    path_label: str,
    direction_mode: str,
    direction: Any,
    tokenizer: Any | None = None,
    group_a_token_ids: Sequence[int] = (),
    group_b_token_ids: Sequence[int] = (),
    geometry: Mapping[str, Any] | None = None,
    prompt_examples: Sequence[Mapping[str, Any]] | None = None,
    example_logit_diffs: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Serialize per-path direction-construction diagnostics for later comparison."""

    direction_tensor = torch.as_tensor(direction, dtype=torch.float32).detach().cpu().reshape(-1)
    artifact = {
        "path_label": path_label,
        "direction_mode": str(direction_mode),
        "direction": _tensor_summary(direction_tensor),
        "group_a_token_ids": [int(value) for value in group_a_token_ids],
        "group_b_token_ids": [int(value) for value in group_b_token_ids],
        "group_a_tokens": [] if tokenizer is None else _decode_token_ids(tokenizer, group_a_token_ids),
        "group_b_tokens": [] if tokenizer is None else _decode_token_ids(tokenizer, group_b_token_ids),
    }
    if example_logit_diffs is not None:
        artifact["example_logit_diffs"] = [float(value) for value in example_logit_diffs]
    if geometry is not None:
        geometry_payload: dict[str, Any] = {
            key: value for key, value in geometry.items() if key not in {"raw_direction", "normalized_direction"}
        }
        geometry_payload["raw_direction"] = geometry["raw_direction_summary"]
        geometry_payload["normalized_direction"] = geometry["normalized_direction_summary"]
        artifact["geometry"] = geometry_payload
    if prompt_examples is not None:
        artifact["prompt_examples"] = [dict(example) for example in prompt_examples]
    return artifact


def _coerce_input_ids(tokenizer: Any, rendered_prompt: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(rendered_prompt, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 0:
            return [int(input_ids.item())]
        if input_ids.dim() == 1:
            return [int(value) for value in input_ids.tolist()]
        return [int(value) for value in input_ids[0].tolist()]
    if input_ids and isinstance(input_ids[0], list):
        return [int(value) for value in input_ids[0]]
    return [int(value) for value in input_ids]


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if not input_ids:
        return []
    if isinstance(input_ids[0], list):
        return [int(value) for value in input_ids[0]]
    return [int(value) for value in input_ids]


def _coerce_offset_mapping(
    tokenizer: Any,
    rendered_prompt: str,
    *,
    add_special_tokens: bool,
) -> list[tuple[int, int]] | None:
    def _pairify(pairs: Any) -> list[tuple[int, int]]:
        return [(int(pair[0]), int(pair[1])) for pair in pairs]

    try:
        encoded = tokenizer(
            rendered_prompt,
            return_offsets_mapping=True,
            return_tensors=None,
            padding=False,
            add_special_tokens=add_special_tokens,
        )
    except Exception:
        return None

    offset_mapping = encoded.get("offset_mapping")
    if offset_mapping is None:
        return None
    if isinstance(offset_mapping, torch.Tensor):
        if offset_mapping.dim() == 3:
            return _pairify(offset_mapping[0].tolist())
        return _pairify(offset_mapping.tolist())
    if (
        offset_mapping
        and isinstance(offset_mapping[0], list)
        and offset_mapping[0]
        and isinstance(offset_mapping[0][0], (list, tuple))
    ):
        return _pairify(offset_mapping[0])
    return _pairify(offset_mapping)


def _resolve_text_span(
    tokenizer: Any,
    rendered_prompt: str,
    input_ids: Sequence[int],
    *,
    text: str,
    add_special_tokens: bool,
    search_end: int | None = None,
) -> tuple[tuple[int, ...], int | None, int | None]:
    if not text:
        return (), None, None

    char_start = rendered_prompt.rfind(text, 0, search_end) if search_end is not None else rendered_prompt.rfind(text)
    if char_start < 0:
        fallback_token_ids = tuple(_encode_text(tokenizer, text))
        fallback_start = find_last_subsequence(input_ids, fallback_token_ids)
        fallback_end = None if fallback_start is None else fallback_start + len(fallback_token_ids) - 1
        return fallback_token_ids, fallback_start, fallback_end

    char_end = char_start + len(text)
    offset_mapping = _coerce_offset_mapping(tokenizer, rendered_prompt, add_special_tokens=add_special_tokens)
    if offset_mapping is not None:
        matched_indices = [
            index
            for index, (token_start, token_end) in enumerate(offset_mapping)
            if token_end > token_start and token_end > char_start and token_start < char_end
        ]
        if matched_indices:
            span_start_index = matched_indices[0]
            span_end_index = matched_indices[-1]
            span_token_ids = tuple(int(value) for value in input_ids[span_start_index : span_end_index + 1])
            return span_token_ids, span_start_index, span_end_index

    prefix_ids = _coerce_input_ids(tokenizer, rendered_prompt[:char_start], add_special_tokens=add_special_tokens)
    span_end_ids = _coerce_input_ids(tokenizer, rendered_prompt[:char_end], add_special_tokens=add_special_tokens)
    span_start_index = len(prefix_ids)
    span_end_index = len(span_end_ids) - 1
    span_token_ids = tuple(int(value) for value in input_ids[span_start_index : span_end_index + 1])
    return span_token_ids, span_start_index, span_end_index


def _decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return _decode_token_ids(tokenizer, token_ids)


def _coerce_feature_row(row: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in row)
    if len(values) != 3:
        raise ValueError(f"Expected feature row of length 3, received {values}")
    return (values[0], values[1], values[2])


def find_last_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int | None:
    """Return the starting index of the final subsequence match, or ``None`` when absent."""
    if not sequence or not subsequence or len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return None


@dataclass(frozen=True)
class PromptAlignmentSnapshot:
    """Rendered-prompt token alignment for one classification/example prompt.

    When ``answer_text`` is provided, ``answer_index`` is the first token index of the
    answer subsequence inside the rendered prompt plus answer string. This lets parity
    tests verify that the probe token is aligned against the actual answer token rather
    than a nearby separator token such as ``:`` or a standalone space token.
    """

    rendered_prompt: str
    input_ids: tuple[int, ...]
    input_tokens: tuple[str, ...]
    probe_text: str
    probe_token_ids: tuple[int, ...]
    probe_start_index: int | None
    probe_end_index: int | None
    answer_text: str | None
    answer_token_ids: tuple[int, ...]
    answer_start_index: int | None
    answer_end_index: int | None
    answer_index: int
    answer_token_id: int
    answer_token_text: str
    previous_token_index: int | None
    previous_token_id: int | None
    previous_token_text: str | None
    intervening_token_ids: tuple[int, ...]
    intervening_token_texts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered_prompt": self.rendered_prompt,
            "input_ids": list(self.input_ids),
            "input_tokens": list(self.input_tokens),
            "probe_text": self.probe_text,
            "probe_token_ids": list(self.probe_token_ids),
            "probe_start_index": self.probe_start_index,
            "probe_end_index": self.probe_end_index,
            "answer_text": self.answer_text,
            "answer_token_ids": list(self.answer_token_ids),
            "answer_start_index": self.answer_start_index,
            "answer_end_index": self.answer_end_index,
            "answer_index": self.answer_index,
            "answer_token_id": self.answer_token_id,
            "answer_token_text": self.answer_token_text,
            "previous_token_index": self.previous_token_index,
            "previous_token_id": self.previous_token_id,
            "previous_token_text": self.previous_token_text,
            "intervening_token_ids": list(self.intervening_token_ids),
            "intervening_token_texts": list(self.intervening_token_texts),
        }


def build_prompt_alignment_snapshot(
    tokenizer: Any,
    rendered_prompt: str,
    *,
    probe_text: str,
    answer_text: str | None = None,
    add_special_tokens: bool,
) -> PromptAlignmentSnapshot:
    """Capture probe-token and answer-token alignment for one rendered prompt."""
    input_ids = _coerce_input_ids(tokenizer, rendered_prompt, add_special_tokens=add_special_tokens)
    input_tokens = _decode_ids(tokenizer, input_ids)
    answer_char_start = rendered_prompt.rfind(answer_text) if answer_text else None
    probe_token_ids, probe_start_index, probe_end_index = _resolve_text_span(
        tokenizer,
        rendered_prompt,
        input_ids,
        text=probe_text,
        add_special_tokens=add_special_tokens,
        search_end=answer_char_start,
    )
    answer_token_ids: tuple[int, ...] = ()
    answer_start_index = None
    answer_end_index = None
    if answer_text is not None:
        answer_token_ids, answer_start_index, answer_end_index = _resolve_text_span(
            tokenizer,
            rendered_prompt,
            input_ids,
            text=answer_text,
            add_special_tokens=add_special_tokens,
        )
    answer_index = len(input_ids) - 1 if answer_start_index is None else answer_start_index
    answer_token_id = int(input_ids[answer_index])
    answer_token_text = str(input_tokens[answer_index])
    previous_token_index = answer_index - 1 if answer_index > 0 else None
    previous_token_id = None if previous_token_index is None else int(input_ids[previous_token_index])
    previous_token_text = None if previous_token_index is None else str(input_tokens[previous_token_index])
    intervening_start = None if probe_end_index is None else probe_end_index + 1
    intervening_end = answer_index
    intervening_token_ids: tuple[int, ...] = ()
    intervening_token_texts: tuple[str, ...] = ()
    if intervening_start is not None and intervening_start < intervening_end:
        intervening_token_ids = tuple(int(value) for value in input_ids[intervening_start:intervening_end])
        intervening_token_texts = tuple(str(value) for value in input_tokens[intervening_start:intervening_end])
    return PromptAlignmentSnapshot(
        rendered_prompt=rendered_prompt,
        input_ids=tuple(input_ids),
        input_tokens=tuple(input_tokens),
        probe_text=probe_text,
        probe_token_ids=tuple(probe_token_ids),
        probe_start_index=probe_start_index,
        probe_end_index=probe_end_index,
        answer_text=answer_text,
        answer_token_ids=answer_token_ids,
        answer_start_index=answer_start_index,
        answer_end_index=answer_end_index,
        answer_index=answer_index,
        answer_token_id=answer_token_id,
        answer_token_text=answer_token_text,
        previous_token_index=previous_token_index,
        previous_token_id=previous_token_id,
        previous_token_text=previous_token_text,
        intervening_token_ids=intervening_token_ids,
        intervening_token_texts=intervening_token_texts,
    )


def resolve_prompt_alignment_context_index(snapshot: PromptAlignmentSnapshot) -> tuple[int | None, str]:
    """Resolve the semantic context-token index for a prompt-alignment snapshot."""

    if snapshot.probe_end_index is not None and snapshot.answer_index > snapshot.probe_end_index:
        return snapshot.probe_end_index, "probe_end"
    if snapshot.previous_token_index is not None:
        return snapshot.previous_token_index, "answer_previous"
    return None, "unavailable"


def _resolve_concept_cache_key(analysis_batch: Any) -> str:
    if hasattr(analysis_batch, "get"):
        return str(analysis_batch.get("concept_cache_key") or "unembed.hook_in")
    return str(getattr(analysis_batch, "concept_cache_key", None) or "unembed.hook_in")


def _get_analysis_value(analysis_batch: Any, field_name: str) -> Any:
    if hasattr(analysis_batch, "get"):
        return analysis_batch.get(field_name)
    return getattr(analysis_batch, field_name)


@dataclass(frozen=True)
class ContextEnhancedExtractionSnapshot:
    """Detailed math for the resolved context-token extraction path."""

    cache_key: str
    context_source: str
    answer_indices: tuple[int, ...]
    answer_states: torch.Tensor
    context_indices: tuple[int, ...]
    context_states: torch.Tensor
    scaled_answer: torch.Tensor
    dot_num: torch.Tensor
    dot_den: torch.Tensor
    projected_states: torch.Tensor
    final_latent_states: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "context_source": self.context_source,
            "answer_indices": list(self.answer_indices),
            "context_indices": list(self.context_indices),
            "answer_states": self.answer_states.tolist(),
            "context_states": self.context_states.tolist(),
            "scaled_answer": self.scaled_answer.tolist(),
            "dot_num": self.dot_num.reshape(-1).tolist(),
            "dot_den": self.dot_den.reshape(-1).tolist(),
            "projected_states": self.projected_states.tolist(),
            "final_latent_states": self.final_latent_states.tolist(),
        }


def capture_context_enhanced_extraction_snapshot(
    analysis_batch: Any,
    *,
    context_scale: float,
) -> ContextEnhancedExtractionSnapshot:
    """Reconstruct the current context-enhanced extraction math from an analysis batch."""
    cache = _get_analysis_value(analysis_batch, "cache")
    answer_indices = _get_analysis_value(analysis_batch, "answer_indices")
    if cache is None or answer_indices is None:
        raise ValueError("capture_context_enhanced_extraction_snapshot requires cache and answer_indices")

    cache_key = _resolve_concept_cache_key(analysis_batch)
    cache_tensor = torch.as_tensor(cache[cache_key])
    answer_index_tensor = torch.as_tensor(answer_indices, dtype=torch.long, device=cache_tensor.device).reshape(-1)
    batch_indices = torch.arange(cache_tensor.size(0), device=cache_tensor.device)

    answer_states = cache_tensor[batch_indices, answer_index_tensor].detach().cpu().float()
    raw_context_token_indices = _get_analysis_value(analysis_batch, "context_token_indices")
    if raw_context_token_indices is None:
        context_source = "answer_previous"
        raw_context_index_tensor = answer_index_tensor - 1
        valid_mask = raw_context_index_tensor >= 0
    else:
        context_source = "context_token_indices"
        raw_context_index_tensor = torch.as_tensor(
            raw_context_token_indices,
            dtype=torch.long,
            device=cache_tensor.device,
        ).reshape(-1)
        if raw_context_index_tensor.shape != answer_index_tensor.shape:
            raise ValueError(
                "capture_context_enhanced_extraction_snapshot requires context_token_indices to align with "
                f"answer_indices ({raw_context_index_tensor.shape} vs {answer_index_tensor.shape})"
            )
        valid_mask = raw_context_index_tensor >= 0
    context_index_tensor = raw_context_index_tensor.clamp(min=0)
    context_states = cache_tensor[batch_indices, context_index_tensor].detach().cpu().float()

    scaled_answer = context_scale * answer_states
    dot_num = (scaled_answer * context_states).sum(dim=-1, keepdim=True)
    dot_den = (context_states * context_states).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    projected_states = (dot_num / dot_den) * context_states
    final_latent_states = torch.where(
        valid_mask.unsqueeze(-1).expand_as(answer_states),
        projected_states,
        answer_states,
    )

    return ContextEnhancedExtractionSnapshot(
        cache_key=cache_key,
        context_source=context_source,
        answer_indices=tuple(int(index) for index in answer_index_tensor.detach().cpu().tolist()),
        answer_states=answer_states,
        context_indices=tuple(int(index) for index in context_index_tensor.detach().cpu().tolist()),
        context_states=context_states,
        scaled_answer=scaled_answer,
        dot_num=dot_num.detach().cpu().float(),
        dot_den=dot_den.detach().cpu().float(),
        projected_states=projected_states.detach().cpu().float(),
        final_latent_states=final_latent_states.detach().cpu().float(),
    )


@dataclass(frozen=True)
class FeatureParitySummary:
    """Set-level overlap summary for two top-feature collections."""

    left_label: str
    right_label: str
    left_only: tuple[tuple[int, int, int], ...]
    shared: tuple[tuple[int, int, int], ...]
    right_only: tuple[tuple[int, int, int], ...]
    jaccard: float
    shared_score_cosine: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_label": self.left_label,
            "right_label": self.right_label,
            "left_only": [list(row) for row in self.left_only],
            "shared": [list(row) for row in self.shared],
            "right_only": [list(row) for row in self.right_only],
            "jaccard": self.jaccard,
            "shared_score_cosine": self.shared_score_cosine,
        }


def compare_top_feature_sets(
    left_rows: Sequence[Sequence[int]],
    right_rows: Sequence[Sequence[int]],
    *,
    left_scores: Sequence[float] | torch.Tensor | None = None,
    right_scores: Sequence[float] | torch.Tensor | None = None,
    left_label: str,
    right_label: str,
) -> FeatureParitySummary:
    """Compute overlap diagnostics for two ordered top-feature collections."""

    left = [_coerce_feature_row(row) for row in left_rows]
    right = [_coerce_feature_row(row) for row in right_rows]
    left_set = set(left)
    right_set = set(right)
    shared = tuple(row for row in left if row in right_set)
    left_only = tuple(row for row in left if row not in right_set)
    right_only = tuple(row for row in right if row not in left_set)
    union_size = len(left_set | right_set)
    jaccard = 0.0 if union_size == 0 else len(shared) / union_size

    shared_score_cosine: float | None = None
    if shared and left_scores is not None and right_scores is not None:
        left_score_tensor = torch.as_tensor(left_scores, dtype=torch.float32).reshape(-1)
        right_score_tensor = torch.as_tensor(right_scores, dtype=torch.float32).reshape(-1)
        left_score_map = {row: float(left_score_tensor[index].item()) for index, row in enumerate(left)}
        right_score_map = {row: float(right_score_tensor[index].item()) for index, row in enumerate(right)}
        left_shared = torch.tensor([left_score_map[row] for row in shared], dtype=torch.float32)
        right_shared = torch.tensor([right_score_map[row] for row in shared], dtype=torch.float32)
        left_norm = torch.linalg.vector_norm(left_shared)
        right_norm = torch.linalg.vector_norm(right_shared)
        if left_norm.item() > 0 and right_norm.item() > 0:
            shared_score_cosine = float(
                torch.nn.functional.cosine_similarity(left_shared.unsqueeze(0), right_shared.unsqueeze(0)).item()
            )

    return FeatureParitySummary(
        left_label=left_label,
        right_label=right_label,
        left_only=left_only,
        shared=shared,
        right_only=right_only,
        jaccard=jaccard,
        shared_score_cosine=shared_score_cosine,
    )


__all__ = [
    "PRESERVE_ARTIFACTS_ENV",
    "PRESERVE_ARTIFACT_DIR_ENV",
    "ARTIFACT_GENERATION_ENV",
    "ARTIFACT_METADATA_KEY",
    "DEFAULT_CONCEPT_DIRECTION_ARTIFACT_ROOT",
    "PARITY_REPORT_FILE_BASENAME",
    "PIPELINE_STATE_FILE_BASENAME",
    "REFERENCE_GRAPH_REPORT_FILE",
    "REFERENCE_GRAPH_REPORT_BASENAME",
    "DEFAULT_RANDOM_PERTURBATION_SCALE",
    "DEFAULT_RANDOM_PERTURBATION_SEED",
    "build_classification_prompt_text",
    "build_concept_direction_stage_artifact",
    "build_context_extraction_artifact",
    "build_random_vector_perturbation",
    "build_indexed_token_debug_rows",
    "build_prompt_alignment_artifact",
    "cosine_similarity_value",
    "ContextEnhancedExtractionSnapshot",
    "compute_concept_direction_geometry",
    "FeatureParitySummary",
    "normalize_prompt_entity_text",
    "PromptAlignmentSnapshot",
    "build_prompt_alignment_snapshot",
    "build_versioned_artifact_file_name",
    "capture_context_enhanced_extraction_snapshot",
    "compare_top_feature_sets",
    "discover_concept_direction_artifacts",
    "find_last_subsequence",
    "resolve_prompt_alignment_context_index",
    "resolve_concept_direction_artifact_generation",
    "load_concept_direction_parity_report",
    "load_concept_direction_reference_graph_report",
    "resolve_concept_direction_parity_output_dir",
    "save_concept_direction_parity_report",
    "save_concept_direction_reference_graph_report",
]


def resolve_concept_direction_artifact_generation(generation: str | None = None) -> str:
    resolved_generation = generation or os.environ.get(ARTIFACT_GENERATION_ENV)
    if resolved_generation:
        return str(resolved_generation)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_versioned_artifact_file_name(base_name: str, *, generation: str | None = None) -> str:
    return f"{base_name}_{resolve_concept_direction_artifact_generation(generation)}.json"


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact payload must parse to a mapping: {path}")
    return payload


def _report_generation_from_file_name(path: Path) -> str | None:
    match = _VERSIONED_REPORT_PATTERN.match(path.name)
    if not match:
        return None
    return match.group("generation")


def _report_matches_base_name(path: Path, base_name: str) -> bool:
    match = _VERSIONED_REPORT_PATTERN.match(path.name)
    return bool(match and match.group("basename") == base_name)


def _artifact_sort_key(path: Path) -> tuple[int, str, str]:
    payload = _read_json_mapping(path)
    metadata = payload.get(ARTIFACT_METADATA_KEY)
    generation = None
    if isinstance(metadata, Mapping):
        generation = metadata.get("generation") or metadata.get("generated_at")
    generation = str(generation) if generation else _report_generation_from_file_name(path)
    return (0 if generation is None else 1, generation or "", path.name)


def _discover_latest_artifact_paths(
    artifact_root: Path,
    *,
    base_name: str,
    selected_names: Sequence[str] | None = None,
    generation: str | None = None,
) -> list[Path]:
    selected_name_set = set(selected_names or ())
    discovered: list[Path] = []
    for artifact_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        if selected_name_set and artifact_dir.name not in selected_name_set:
            continue
        candidates = sorted(path for path in artifact_dir.glob("*.json") if _report_matches_base_name(path, base_name))
        if generation is not None:
            candidates = [
                path
                for path in candidates
                if (_read_json_mapping(path).get(ARTIFACT_METADATA_KEY, {}) or {}).get("generation") == generation
                or _report_generation_from_file_name(path) == generation
            ]
        if not candidates:
            continue
        discovered.append(max(candidates, key=_artifact_sort_key))
    return discovered


def discover_concept_direction_artifacts(
    *,
    artifact_dir: str | os.PathLike[str] | None = None,
    selection: Mapping[str, Sequence[str] | None] | None = None,
    generation: str | None = None,
) -> dict[str, Any]:
    artifact_root = resolve_concept_direction_parity_output_dir(artifact_dir=artifact_dir, create=False)
    if not artifact_root.exists():
        raise FileNotFoundError(f"Concept-direction artifact root does not exist: {artifact_root}")

    resolved_generation = generation or os.environ.get(ARTIFACT_GENERATION_ENV)
    resolved_selection = {
        "parity_reports": tuple(selection.get("parity_reports") or ()) if selection else (),
        "reference_reports": tuple(selection.get("reference_reports") or ()) if selection else (),
        "notebook_artifacts": tuple(selection.get("notebook_artifacts") or ()) if selection else (),
    }

    report_paths = _discover_latest_artifact_paths(
        artifact_root,
        base_name=PARITY_REPORT_FILE_BASENAME,
        selected_names=resolved_selection["parity_reports"],
        generation=resolved_generation,
    )
    reference_report_paths = _discover_latest_artifact_paths(
        artifact_root,
        base_name=REFERENCE_GRAPH_REPORT_BASENAME,
        selected_names=resolved_selection["reference_reports"],
        generation=resolved_generation,
    )
    notebook_pipeline_state_paths = _discover_latest_artifact_paths(
        artifact_root,
        base_name=PIPELINE_STATE_FILE_BASENAME,
        selected_names=resolved_selection["notebook_artifacts"],
        generation=resolved_generation,
    )
    parity_report_names = {path.parent.name for path in report_paths}
    notebook_pipeline_state_paths = [
        path for path in notebook_pipeline_state_paths if path.parent.name not in parity_report_names
    ]
    pipeline_state_paths = {
        path.parent.name: path
        for path in _discover_latest_artifact_paths(
            artifact_root,
            base_name=PIPELINE_STATE_FILE_BASENAME,
            selected_names=tuple(parity_report_names),
            generation=resolved_generation,
        )
    }

    if not report_paths and not reference_report_paths and not notebook_pipeline_state_paths:
        raise FileNotFoundError(f"No concept-direction artifacts were found under {artifact_root}")

    return {
        "artifact_root": artifact_root,
        "artifact_generation": resolved_generation,
        "artifact_selection": {key: value or None for key, value in resolved_selection.items()},
        "report_paths": report_paths,
        "reference_report_paths": reference_report_paths,
        "notebook_pipeline_state_paths": notebook_pipeline_state_paths,
        "pipeline_state_paths": pipeline_state_paths,
    }


def _prepare_artifact_payload(
    report: Mapping[str, Any],
    *,
    artifact_name: str,
    base_name: str,
    generation: str,
) -> dict[str, Any]:
    payload = dict(report)
    payload[ARTIFACT_METADATA_KEY] = {
        "artifact_name": artifact_name,
        "artifact_kind": _REPORT_BASENAME_TO_KIND[base_name],
        "base_name": base_name,
        "file_name": build_versioned_artifact_file_name(base_name, generation=generation),
        "generation": generation,
        "generated_at": generation,
    }
    return payload


def resolve_concept_direction_parity_output_dir(
    *,
    artifact_name: str | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
    create: bool = True,
) -> Path:
    base_dir = artifact_dir or os.environ.get(PRESERVE_ARTIFACT_DIR_ENV) or DEFAULT_CONCEPT_DIRECTION_ARTIFACT_ROOT
    output_dir = Path(base_dir).expanduser().resolve()
    if artifact_name:
        output_dir = output_dir / artifact_name
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_concept_direction_parity_report(
    report: Mapping[str, Any],
    *,
    artifact_name: str,
    artifact_dir: str | os.PathLike[str] | None = None,
    file_name: str | None = None,
    generation: str | None = None,
) -> Path:
    output_dir = resolve_concept_direction_parity_output_dir(artifact_name=artifact_name, artifact_dir=artifact_dir)
    resolved_generation = resolve_concept_direction_artifact_generation(generation)
    resolved_file_name = file_name or build_versioned_artifact_file_name(
        PARITY_REPORT_FILE_BASENAME,
        generation=resolved_generation,
    )
    output_path = output_dir / resolved_file_name
    payload = _prepare_artifact_payload(
        report,
        artifact_name=artifact_name,
        base_name=PARITY_REPORT_FILE_BASENAME,
        generation=resolved_generation,
    )
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def load_concept_direction_parity_report(report_path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(report_path).expanduser().resolve().read_text(encoding="utf-8"))


def save_concept_direction_reference_graph_report(
    report: Mapping[str, Any],
    *,
    artifact_name: str,
    artifact_dir: str | os.PathLike[str] | None = None,
    file_name: str | None = None,
    generation: str | None = None,
) -> Path:
    output_dir = resolve_concept_direction_parity_output_dir(artifact_name=artifact_name, artifact_dir=artifact_dir)
    resolved_generation = resolve_concept_direction_artifact_generation(generation)
    resolved_file_name = file_name or build_versioned_artifact_file_name(
        REFERENCE_GRAPH_REPORT_BASENAME,
        generation=resolved_generation,
    )
    output_path = output_dir / resolved_file_name
    payload = _prepare_artifact_payload(
        report,
        artifact_name=artifact_name,
        base_name=REFERENCE_GRAPH_REPORT_BASENAME,
        generation=resolved_generation,
    )
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def load_concept_direction_reference_graph_report(report_path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(report_path).expanduser().resolve().read_text(encoding="utf-8"))


def save_concept_direction_pipeline_state_artifacts(
    report: Mapping[str, Any],
    *,
    artifact_name: str,
    artifact_dir: str | os.PathLike[str] | None = None,
    file_name: str | None = None,
    generation: str | None = None,
) -> Path:
    output_dir = resolve_concept_direction_parity_output_dir(artifact_name=artifact_name, artifact_dir=artifact_dir)
    resolved_generation = resolve_concept_direction_artifact_generation(generation)
    resolved_file_name = file_name or build_versioned_artifact_file_name(
        PIPELINE_STATE_FILE_BASENAME,
        generation=resolved_generation,
    )
    output_path = output_dir / resolved_file_name
    payload = _prepare_artifact_payload(
        report,
        artifact_name=artifact_name,
        base_name=PIPELINE_STATE_FILE_BASENAME,
        generation=resolved_generation,
    )
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path


def load_concept_direction_pipeline_state_artifacts(report_path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(report_path).expanduser().resolve().read_text(encoding="utf-8"))
