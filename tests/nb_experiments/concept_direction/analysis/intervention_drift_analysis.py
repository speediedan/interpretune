"""Utilities for analyzing selected-feature intervention drift artifacts.

This module is intended for manual parity debugging and notebook exploration. It stays out of the normal CI path and
focuses on loading preserved graph and activation artifacts, computing divergence summaries, and inspecting how drift
appears to propagate through retained feature rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from circuit_tracer.graph import Graph


PRESERVE_ARTIFACTS_ENV = "IT_PARITY_PRESERVE_ARTIFACTS"
PRESERVE_ARTIFACT_DIR_ENV = "IT_PARITY_PRESERVE_ARTIFACT_DIR"


def _json_debug_value(value: Any, *, max_items: int = 16) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, range):
        return {"kind": "range", "start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_debug_value(item, max_items=max_items) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return tensor_fingerprint(value, max_items=max_items)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [_json_debug_value(item, max_items=max_items) for item in list(value)[:max_items]]
        if len(value) > max_items:
            return {"kind": type(value).__name__, "length": len(value), "items": items}
        return items
    return str(value)


def tensor_fingerprint(value: Any, *, max_items: int = 16) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    flat = tensor.reshape(-1)
    hasher = hashlib.sha256()
    hasher.update(str(tensor.dtype).encode("utf-8"))
    hasher.update(json.dumps([int(dim) for dim in tensor.shape]).encode("utf-8"))
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    summary: dict[str, Any] = {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": hasher.hexdigest(),
    }
    if flat.numel() <= max_items:
        summary["values"] = flat.tolist()
    elif flat.numel() > 0 and torch.is_floating_point(flat):
        summary["min"] = float(flat.min().item())
        summary["max"] = float(flat.max().item())
        summary["mean"] = float(flat.float().mean().item())
    elif flat.numel() > 0:
        summary["min"] = int(flat.min().item())
        summary["max"] = int(flat.max().item())
    return summary


def snapshot_analysis_batch(
    analysis_batch: Any,
    fields: Sequence[str],
    *,
    max_items: int = 16,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if analysis_batch is None:
        return snapshot

    for field_name in fields:
        if isinstance(analysis_batch, Mapping):
            value = analysis_batch.get(field_name)
        else:
            value = getattr(analysis_batch, field_name, None)
            if value is None and hasattr(analysis_batch, "get"):
                try:
                    value = analysis_batch.get(field_name)
                except Exception:
                    value = None
        if value is not None:
            snapshot[field_name] = _json_debug_value(value, max_items=max_items)
    return snapshot


def snapshot_module_runtime_state(module: Any) -> dict[str, Any]:
    circuit_tracer_cfg = getattr(module, "circuit_tracer_cfg", None)
    nnsight_cfg = getattr(module, "nnsight_cfg", None)
    analysis_cfg = getattr(module, "analysis_cfg", None)
    replacement_model = getattr(module, "replacement_model", None)
    model = getattr(module, "model", None)

    def _first_parameter_state(obj: Any) -> dict[str, Any] | None:
        if obj is None or not hasattr(obj, "parameters"):
            return None
        try:
            parameter = next(obj.parameters())
        except Exception:
            return None
        return {
            "device": str(parameter.device),
            "dtype": str(parameter.dtype),
            "shape": [int(dim) for dim in parameter.shape],
        }

    target_op = getattr(analysis_cfg, "target_op", None)
    target_op_name = getattr(target_op, "name", None) or getattr(target_op, "__name__", None)

    return {
        "module_class": type(module).__name__,
        "module_device_type": getattr(module, "device_type", None),
        "analysis_cfg": {
            "target_op": target_op_name,
            "ignore_manual": getattr(analysis_cfg, "ignore_manual", None),
            "save_tokens": getattr(analysis_cfg, "save_tokens", None),
        }
        if analysis_cfg is not None
        else None,
        "nnsight_cfg": {
            "model_name": getattr(nnsight_cfg, "model_name", None),
            "device_map": getattr(nnsight_cfg, "device_map", None),
            "torch_dtype": getattr(nnsight_cfg, "torch_dtype", None),
            "dispatch": getattr(nnsight_cfg, "dispatch", None),
            "attn_implementation": getattr(nnsight_cfg, "attn_implementation", None),
            "tokenizer_kwargs": _json_debug_value(getattr(nnsight_cfg, "tokenizer_kwargs", None)),
        }
        if nnsight_cfg is not None
        else None,
        "circuit_tracer_cfg": {
            "backend": getattr(circuit_tracer_cfg, "backend", None),
            "model_name": getattr(circuit_tracer_cfg, "model_name", None),
            "transcoder_set": getattr(circuit_tracer_cfg, "transcoder_set", None),
            "dtype": _json_debug_value(getattr(circuit_tracer_cfg, "dtype", None)),
            "analysis_target_tokens": _json_debug_value(getattr(circuit_tracer_cfg, "analysis_target_tokens", None)),
            "target_token_ids": _json_debug_value(getattr(circuit_tracer_cfg, "target_token_ids", None)),
            "max_feature_nodes": getattr(circuit_tracer_cfg, "max_feature_nodes", None),
            "offload": getattr(circuit_tracer_cfg, "offload", None),
            "verbose": getattr(circuit_tracer_cfg, "verbose", None),
            "batch_size": getattr(circuit_tracer_cfg, "batch_size", None),
            "max_n_logits": getattr(circuit_tracer_cfg, "max_n_logits", None),
            "desired_logit_prob": getattr(circuit_tracer_cfg, "desired_logit_prob", None),
            "intervention_scale_factor": getattr(circuit_tracer_cfg, "intervention_scale_factor", None),
            "intervention_max_influence_norm_scale": getattr(
                circuit_tracer_cfg,
                "intervention_max_influence_norm_scale",
                None,
            ),
            "intervention_sign_aware_scale": getattr(circuit_tracer_cfg, "intervention_sign_aware_scale", None),
            "intervention_value": getattr(circuit_tracer_cfg, "intervention_value", None),
            "intervention_value_source": getattr(circuit_tracer_cfg, "intervention_value_source", None),
            "intervention_constrained_layers": _json_debug_value(
                getattr(circuit_tracer_cfg, "intervention_constrained_layers", None)
            ),
            "intervention_freeze_attention": _json_debug_value(
                getattr(circuit_tracer_cfg, "intervention_freeze_attention", None)
            ),
            "intervention_apply_activation_function": getattr(
                circuit_tracer_cfg, "intervention_apply_activation_function", None
            ),
            "intervention_sparse": getattr(circuit_tracer_cfg, "intervention_sparse", None),
            "intervention_return_activations": getattr(circuit_tracer_cfg, "intervention_return_activations", None),
        }
        if circuit_tracer_cfg is not None
        else None,
        "module_first_parameter": _first_parameter_state(model if model is not None else module),
        "replacement_model": {
            "class": type(replacement_model).__name__,
            "device": _json_debug_value(getattr(replacement_model, "device", None)),
            "dtype": _json_debug_value(getattr(replacement_model, "dtype", None)),
            "cfg_n_layers": getattr(getattr(replacement_model, "cfg", None), "n_layers", None),
            "model_first_parameter": _first_parameter_state(getattr(replacement_model, "model", None)),
        }
        if replacement_model is not None
        else None,
    }


def _ensure_1d_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().float().cpu()
    if logits.dim() > 1:
        return logits.squeeze(0)[-1]
    return logits


def _coerce_feature_row(feature_row: Sequence[int] | Sequence[float]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in feature_row)
    if len(values) != 3:
        raise ValueError(f"Expected a feature row of length 3, received {values}")
    return (values[0], values[1], values[2])


def _is_divergent(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    return ~torch.isclose(actual, expected, atol=atol, rtol=rtol)


def selected_feature_rows(graph: Graph) -> torch.Tensor:
    """Return selected feature rows as ``(layer, position, feature_id)`` triples."""

    selected_indices = graph.selected_features.detach().long().cpu()
    return graph.active_features.detach().long().cpu().index_select(0, selected_indices)


def find_selected_feature_index(feature_rows: torch.Tensor, feature_row: Sequence[int]) -> int:
    """Return the selected-feature node index for a concrete feature row."""

    layer, position, feature_id = (int(value) for value in feature_row)
    matches = (
        (feature_rows[:, 0] == layer) & (feature_rows[:, 1] == position) & (feature_rows[:, 2] == feature_id)
    ).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(f"Selected feature row {(layer, position, feature_id)} is not present in the graph")
    return int(matches[0].item())


@dataclass(frozen=True)
class FeatureRowDrift:
    """Detailed expected-vs-actual activation drift for a retained feature row."""

    feature_row: tuple[int, int, int]
    node_index: int
    baseline_activation: float
    expected_activation: float
    actual_activation: float
    expected_delta: float
    actual_delta: float
    abs_error: float
    diverged: bool
    sign_mismatch: bool

    @property
    def layer(self) -> int:
        return self.feature_row[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_row": list(self.feature_row),
            "node_index": self.node_index,
            "baseline_activation": self.baseline_activation,
            "expected_activation": self.expected_activation,
            "actual_activation": self.actual_activation,
            "expected_delta": self.expected_delta,
            "actual_delta": self.actual_delta,
            "abs_error": self.abs_error,
            "diverged": self.diverged,
            "sign_mismatch": self.sign_mismatch,
        }


@dataclass(frozen=True)
class LayerDriftSummary:
    """Per-layer summary for retained-feature activation drift."""

    layer: int
    divergent_feature_count: int
    total_feature_count: int
    max_abs_error: float
    mean_abs_error: float
    expected_abs_delta_sum: float
    actual_abs_delta_sum: float
    top_error_feature_row: tuple[int, int, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "divergent_feature_count": self.divergent_feature_count,
            "total_feature_count": self.total_feature_count,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "expected_abs_delta_sum": self.expected_abs_delta_sum,
            "actual_abs_delta_sum": self.actual_abs_delta_sum,
            "top_error_feature_row": list(self.top_error_feature_row) if self.top_error_feature_row else None,
        }


@dataclass(frozen=True)
class LogitDrift:
    """Detailed expected-vs-actual drift for one tracked graph logit target."""

    token_id: int
    token_label: str
    baseline_logit: float
    expected_logit: float
    actual_logit: float
    expected_delta: float
    actual_delta: float
    abs_error: float
    diverged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "token_label": self.token_label,
            "baseline_logit": self.baseline_logit,
            "expected_logit": self.expected_logit,
            "actual_logit": self.actual_logit,
            "expected_delta": self.expected_delta,
            "actual_delta": self.actual_delta,
            "abs_error": self.abs_error,
            "diverged": self.diverged,
        }


@dataclass(frozen=True)
class LogitDriftSummary:
    """Summary for tracked graph-logit divergence."""

    divergent_logit_count: int
    total_logit_count: int
    max_abs_error: float
    mean_abs_error: float
    top_errors: tuple[LogitDrift, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "divergent_logit_count": self.divergent_logit_count,
            "total_logit_count": self.total_logit_count,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "top_errors": [error.to_dict() for error in self.top_errors],
        }


@dataclass(frozen=True)
class IncomingSourceSummary:
    """Incoming source edge summary for a divergent retained feature row."""

    source_feature_row: tuple[int, int, int]
    source_node_index: int
    source_abs_error: float
    source_actual_delta: float
    source_expected_delta: float
    edge_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_feature_row": list(self.source_feature_row),
            "source_node_index": self.source_node_index,
            "source_abs_error": self.source_abs_error,
            "source_actual_delta": self.source_actual_delta,
            "source_expected_delta": self.source_expected_delta,
            "edge_weight": self.edge_weight,
        }


@dataclass(frozen=True)
class PropagationPathSummary:
    """Incoming-edge summary for one divergent retained feature row."""

    target_feature_row: tuple[int, int, int]
    target_node_index: int
    target_abs_error: float
    target_actual_delta: float
    target_expected_delta: float
    incoming_sources: tuple[IncomingSourceSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_feature_row": list(self.target_feature_row),
            "target_node_index": self.target_node_index,
            "target_abs_error": self.target_abs_error,
            "target_actual_delta": self.target_actual_delta,
            "target_expected_delta": self.target_expected_delta,
            "incoming_sources": [source.to_dict() for source in self.incoming_sources],
        }


@dataclass(frozen=True)
class InterventionDriftReport:
    """Computed divergence report for one selected-feature intervention."""

    feature_row: tuple[int, int, int]
    selected_feature_index: int
    total_feature_count: int
    divergent_feature_count: int
    max_activation_abs_error: float
    layer_summaries: tuple[LayerDriftSummary, ...]
    feature_errors: tuple[FeatureRowDrift, ...]
    logit_summary: LogitDriftSummary

    @property
    def layer_with_max_divergence(self) -> int | None:
        if not self.layer_summaries:
            return None
        return self.layer_summaries[0].layer

    def to_dict(self, *, top_feature_count: int = 12) -> dict[str, Any]:
        feature_errors = sorted(self.feature_errors, key=lambda error: error.abs_error, reverse=True)
        return {
            "feature_row": list(self.feature_row),
            "selected_feature_index": self.selected_feature_index,
            "total_feature_count": self.total_feature_count,
            "divergent_feature_count": self.divergent_feature_count,
            "max_activation_abs_error": self.max_activation_abs_error,
            "layer_with_max_divergence": self.layer_with_max_divergence,
            "layer_summaries": [summary.to_dict() for summary in self.layer_summaries],
            "top_feature_errors": [error.to_dict() for error in feature_errors[:top_feature_count]],
            "logit_summary": self.logit_summary.to_dict(),
        }


@dataclass(frozen=True)
class PreservedInterventionArtifacts:
    """On-disk preserved intervention artifacts and their recomputed report."""

    artifact_dir: Path
    graph: Graph
    feature_row: tuple[int, int, int]
    interventions: tuple[tuple[int, int, int, float], ...]
    baseline_activation_cache: torch.Tensor
    intervention_activation_cache: torch.Tensor
    baseline_logits: torch.Tensor
    intervention_logits: torch.Tensor
    activation_atol: float
    activation_rtol: float
    logit_atol: float
    logit_rtol: float
    metadata: dict[str, Any]
    report: InterventionDriftReport


def build_intervention_drift_report(
    graph: Graph,
    *,
    feature_row: Sequence[int],
    baseline_activation_cache: torch.Tensor,
    intervention_activation_cache: torch.Tensor,
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    activation_atol: float,
    activation_rtol: float,
    logit_atol: float,
    logit_rtol: float,
) -> InterventionDriftReport:
    """Compute retained-feature and tracked-logit drift for one intervention."""

    feature_rows = selected_feature_rows(graph)
    selected_feature_index = find_selected_feature_index(feature_rows, feature_row)
    feature_count = int(feature_rows.shape[0])
    adjacency_column = graph.adjacency_matrix.detach().float().cpu()[:, selected_feature_index]

    baseline_activation_cache = baseline_activation_cache.detach().float().cpu()
    intervention_activation_cache = intervention_activation_cache.detach().float().cpu()
    baseline_logits = _ensure_1d_logits(baseline_logits)
    intervention_logits = _ensure_1d_logits(intervention_logits)

    baseline_selected_activations = baseline_activation_cache[
        feature_rows[:, 0], feature_rows[:, 1], feature_rows[:, 2]
    ]
    intervention_selected_activations = intervention_activation_cache[
        feature_rows[:, 0], feature_rows[:, 1], feature_rows[:, 2]
    ]
    expected_selected_deltas = adjacency_column[:feature_count]
    expected_selected_activations = baseline_selected_activations + expected_selected_deltas
    actual_selected_deltas = intervention_selected_activations - baseline_selected_activations
    activation_abs_errors = (intervention_selected_activations - expected_selected_activations).abs()
    activation_diverged = _is_divergent(
        intervention_selected_activations,
        expected_selected_activations,
        atol=activation_atol,
        rtol=activation_rtol,
    )

    feature_errors = []
    for node_index, feature_row_values in enumerate(feature_rows.tolist()):
        expected_delta = float(expected_selected_deltas[node_index].item())
        actual_delta = float(actual_selected_deltas[node_index].item())
        feature_errors.append(
            FeatureRowDrift(
                feature_row=_coerce_feature_row(feature_row_values),
                node_index=node_index,
                baseline_activation=float(baseline_selected_activations[node_index].item()),
                expected_activation=float(expected_selected_activations[node_index].item()),
                actual_activation=float(intervention_selected_activations[node_index].item()),
                expected_delta=expected_delta,
                actual_delta=actual_delta,
                abs_error=float(activation_abs_errors[node_index].item()),
                diverged=bool(activation_diverged[node_index].item()),
                sign_mismatch=(expected_delta * actual_delta) < 0,
            )
        )

    layer_summaries = []
    unique_layers = torch.unique(feature_rows[:, 0], sorted=True).tolist()
    for layer in unique_layers:
        layer = int(layer)
        layer_mask = feature_rows[:, 0] == layer
        layer_errors = activation_abs_errors[layer_mask]
        layer_expected = expected_selected_deltas[layer_mask].abs()
        layer_actual = actual_selected_deltas[layer_mask].abs()
        layer_diverged = activation_diverged[layer_mask]
        layer_feature_errors = [feature_errors[index] for index, keep in enumerate(layer_mask.tolist()) if keep]
        top_error = max(layer_feature_errors, key=lambda error: error.abs_error, default=None)
        layer_summaries.append(
            LayerDriftSummary(
                layer=layer,
                divergent_feature_count=int(layer_diverged.sum().item()),
                total_feature_count=int(layer_mask.sum().item()),
                max_abs_error=float(layer_errors.max().item()) if layer_errors.numel() else 0.0,
                mean_abs_error=float(layer_errors.mean().item()) if layer_errors.numel() else 0.0,
                expected_abs_delta_sum=float(layer_expected.sum().item()) if layer_expected.numel() else 0.0,
                actual_abs_delta_sum=float(layer_actual.sum().item()) if layer_actual.numel() else 0.0,
                top_error_feature_row=top_error.feature_row if top_error else None,
            )
        )
    layer_summaries.sort(key=lambda summary: (summary.max_abs_error, summary.divergent_feature_count), reverse=True)

    logit_target_ids = graph.logit_token_ids.detach().long().cpu()
    logit_target_tokens = [target.token_str for target in graph.logit_targets]
    expected_logit_deltas = adjacency_column[-len(logit_target_ids) :]
    baseline_target_logits = baseline_logits.index_select(0, logit_target_ids)
    intervention_target_logits = intervention_logits.index_select(0, logit_target_ids)
    baseline_logits_demeaned = baseline_target_logits - baseline_logits.mean()
    intervention_logits_demeaned = intervention_target_logits - intervention_logits.mean()
    expected_logits_demeaned = baseline_logits_demeaned + expected_logit_deltas
    logit_abs_errors = (intervention_logits_demeaned - expected_logits_demeaned).abs()
    logit_diverged = _is_divergent(
        intervention_logits_demeaned,
        expected_logits_demeaned,
        atol=logit_atol,
        rtol=logit_rtol,
    )
    logit_errors = []
    for index, token_id in enumerate(logit_target_ids.tolist()):
        expected_delta = float(expected_logit_deltas[index].item())
        actual_delta = float((intervention_logits_demeaned[index] - baseline_logits_demeaned[index]).item())
        logit_errors.append(
            LogitDrift(
                token_id=int(token_id),
                token_label=logit_target_tokens[index],
                baseline_logit=float(baseline_logits_demeaned[index].item()),
                expected_logit=float(expected_logits_demeaned[index].item()),
                actual_logit=float(intervention_logits_demeaned[index].item()),
                expected_delta=expected_delta,
                actual_delta=actual_delta,
                abs_error=float(logit_abs_errors[index].item()),
                diverged=bool(logit_diverged[index].item()),
            )
        )
    logit_errors.sort(key=lambda error: error.abs_error, reverse=True)
    logit_summary = LogitDriftSummary(
        divergent_logit_count=int(logit_diverged.sum().item()),
        total_logit_count=len(logit_errors),
        max_abs_error=float(logit_abs_errors.max().item()) if logit_abs_errors.numel() else 0.0,
        mean_abs_error=float(logit_abs_errors.mean().item()) if logit_abs_errors.numel() else 0.0,
        top_errors=tuple(logit_errors[: min(10, len(logit_errors))]),
    )

    return InterventionDriftReport(
        feature_row=_coerce_feature_row(feature_row),
        selected_feature_index=selected_feature_index,
        total_feature_count=feature_count,
        divergent_feature_count=int(activation_diverged.sum().item()),
        max_activation_abs_error=float(activation_abs_errors.max().item()) if activation_abs_errors.numel() else 0.0,
        layer_summaries=tuple(layer_summaries),
        feature_errors=tuple(feature_errors),
        logit_summary=logit_summary,
    )


def build_layer_error_distribution(report: InterventionDriftReport) -> torch.Tensor:
    """Return a padded layer-by-layer matrix of absolute activation errors."""

    if not report.feature_errors:
        return torch.empty((0, 0), dtype=torch.float32)
    layers = sorted({error.layer for error in report.feature_errors})
    max_count = max(sum(1 for error in report.feature_errors if error.layer == layer) for layer in layers)
    distribution = torch.zeros((len(layers), max_count), dtype=torch.float32)
    for row_index, layer in enumerate(layers):
        layer_errors = [error.abs_error for error in report.feature_errors if error.layer == layer]
        distribution[row_index, : len(layer_errors)] = torch.tensor(layer_errors, dtype=torch.float32)
    return distribution


def summarize_propagation_paths(
    graph: Graph,
    report: InterventionDriftReport,
    *,
    target_layer: int,
    top_k_rows: int = 5,
    top_k_sources: int = 5,
) -> tuple[PropagationPathSummary, ...]:
    """Summarize top incoming retained-feature sources for divergent rows in one layer."""

    feature_rows = selected_feature_rows(graph)
    feature_count = int(feature_rows.shape[0])
    adjacency = graph.adjacency_matrix.detach().float().cpu()[:feature_count, :feature_count]
    errors_by_node = {error.node_index: error for error in report.feature_errors}
    target_errors = [error for error in report.feature_errors if error.layer == target_layer]
    target_errors.sort(key=lambda error: error.abs_error, reverse=True)
    path_summaries: list[PropagationPathSummary] = []

    for target_error in target_errors[:top_k_rows]:
        incoming_weights = adjacency[target_error.node_index].clone()
        if incoming_weights.numel() == 0:
            incoming_source_summaries: tuple[IncomingSourceSummary, ...] = ()
        else:
            incoming_weights[target_error.node_index] = 0
            source_limit = min(top_k_sources, incoming_weights.numel())
            if source_limit == 0:
                incoming_source_summaries = ()
            else:
                top_source_indices = torch.topk(incoming_weights.abs(), source_limit).indices.tolist()
                incoming_sources = []
                for source_index in top_source_indices:
                    source_error = errors_by_node[int(source_index)]
                    incoming_sources.append(
                        IncomingSourceSummary(
                            source_feature_row=source_error.feature_row,
                            source_node_index=source_error.node_index,
                            source_abs_error=source_error.abs_error,
                            source_actual_delta=source_error.actual_delta,
                            source_expected_delta=source_error.expected_delta,
                            edge_weight=float(incoming_weights[source_index].item()),
                        )
                    )
                incoming_source_summaries = tuple(incoming_sources)

        path_summaries.append(
            PropagationPathSummary(
                target_feature_row=target_error.feature_row,
                target_node_index=target_error.node_index,
                target_abs_error=target_error.abs_error,
                target_actual_delta=target_error.actual_delta,
                target_expected_delta=target_error.expected_delta,
                incoming_sources=incoming_source_summaries,
            )
        )

    return tuple(path_summaries)


def preservation_requested(env: Mapping[str, str] | None = None) -> bool:
    """Return whether preserved intervention artifacts are requested."""

    env = env or os.environ
    value = env.get(PRESERVE_ARTIFACTS_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def resolve_artifact_output_dir(
    *,
    artifact_name: str,
    base_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve the output directory for preserved intervention artifacts."""

    env = env or os.environ
    if not preservation_requested(env):
        return None

    default_output_root = Path(__file__).resolve().parent / "artifacts"
    output_root_value = (
        base_dir if base_dir is not None else env.get(PRESERVE_ARTIFACT_DIR_ENV, str(default_output_root))
    )
    output_root = Path(output_root_value)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return output_root / f"{artifact_name}_{timestamp}"


def _json_ready_interventions(interventions: Iterable[Sequence[int | float]]) -> list[list[int | float]]:
    return [[int(spec[0]), int(spec[1]), int(spec[2]), float(spec[3])] for spec in interventions]


def save_preserved_intervention_artifacts(
    artifact_dir: str | Path,
    *,
    graph: Graph,
    feature_row: Sequence[int],
    interventions: Iterable[Sequence[int | float]],
    baseline_activation_cache: torch.Tensor,
    intervention_activation_cache: torch.Tensor,
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    activation_atol: float,
    activation_rtol: float,
    logit_atol: float,
    logit_rtol: float,
    report: InterventionDriftReport,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a graph plus activation/logit tensors for manual drift analysis."""

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    graph_path = artifact_dir / "graph.pt"
    graph.to_pt(str(graph_path))

    bundle_path = artifact_dir / "bundle.pt"
    bundle = {
        "feature_row": [int(value) for value in feature_row],
        "interventions": _json_ready_interventions(interventions),
        "baseline_activation_cache": baseline_activation_cache.detach().cpu(),
        "intervention_activation_cache": intervention_activation_cache.detach().cpu(),
        "baseline_logits": _ensure_1d_logits(baseline_logits),
        "intervention_logits": _ensure_1d_logits(intervention_logits),
        "activation_atol": float(activation_atol),
        "activation_rtol": float(activation_rtol),
        "logit_atol": float(logit_atol),
        "logit_rtol": float(logit_rtol),
        "metadata": metadata or {},
    }
    torch.save(bundle, bundle_path)

    summary_path = artifact_dir / "summary.json"
    summary_payload = {
        "artifact_dir": str(artifact_dir),
        "feature_row": [int(value) for value in feature_row],
        "report": report.to_dict(),
        "metadata": metadata or {},
        "interventions": _json_ready_interventions(interventions),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    return artifact_dir


def _resolve_artifact_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    if candidate.name in {"bundle.pt", "graph.pt", "summary.json"}:
        return candidate.parent
    raise FileNotFoundError(f"Unable to resolve artifact directory from {candidate}")


def load_preserved_intervention_artifacts(path: str | Path) -> PreservedInterventionArtifacts:
    """Load preserved intervention artifacts from disk and recompute the report."""

    artifact_dir = _resolve_artifact_dir(path)
    graph = Graph.from_pt(str(artifact_dir / "graph.pt"), map_location="cpu")
    bundle = torch.load(artifact_dir / "bundle.pt", weights_only=False, map_location="cpu")
    feature_row = _coerce_feature_row(bundle["feature_row"])
    report = build_intervention_drift_report(
        graph,
        feature_row=feature_row,
        baseline_activation_cache=bundle["baseline_activation_cache"],
        intervention_activation_cache=bundle["intervention_activation_cache"],
        baseline_logits=bundle["baseline_logits"],
        intervention_logits=bundle["intervention_logits"],
        activation_atol=float(bundle["activation_atol"]),
        activation_rtol=float(bundle["activation_rtol"]),
        logit_atol=float(bundle["logit_atol"]),
        logit_rtol=float(bundle["logit_rtol"]),
    )
    interventions = tuple(
        (int(spec[0]), int(spec[1]), int(spec[2]), float(spec[3])) for spec in bundle.get("interventions", [])
    )
    return PreservedInterventionArtifacts(
        artifact_dir=artifact_dir,
        graph=graph,
        feature_row=feature_row,
        interventions=interventions,
        baseline_activation_cache=bundle["baseline_activation_cache"],
        intervention_activation_cache=bundle["intervention_activation_cache"],
        baseline_logits=bundle["baseline_logits"],
        intervention_logits=bundle["intervention_logits"],
        activation_atol=float(bundle["activation_atol"]),
        activation_rtol=float(bundle["activation_rtol"]),
        logit_atol=float(bundle["logit_atol"]),
        logit_rtol=float(bundle["logit_rtol"]),
        metadata=dict(bundle.get("metadata", {})),
        report=report,
    )


def build_cli_summary(
    artifacts: PreservedInterventionArtifacts,
    *,
    target_layer: int,
    top_k_rows: int,
    top_k_sources: int,
) -> dict[str, Any]:
    """Build a JSON-ready summary for CLI and notebook use."""

    propagation = summarize_propagation_paths(
        artifacts.graph,
        artifacts.report,
        target_layer=target_layer,
        top_k_rows=top_k_rows,
        top_k_sources=top_k_sources,
    )
    return {
        "artifact_dir": str(artifacts.artifact_dir),
        "feature_row": list(artifacts.feature_row),
        "metadata": artifacts.metadata,
        "report": artifacts.report.to_dict(),
        "target_layer": target_layer,
        "propagation": [summary.to_dict() for summary in propagation],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_path", help="Artifact directory or one of its files")
    parser.add_argument("--target-layer", type=int, default=33, help="Layer to inspect for propagation summaries")
    parser.add_argument("--top-k-rows", type=int, default=5, help="Number of target rows to summarize")
    parser.add_argument("--top-k-sources", type=int, default=5, help="Number of incoming sources per target row")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write the JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifacts = load_preserved_intervention_artifacts(args.artifact_path)
    summary = build_cli_summary(
        artifacts,
        target_layer=args.target_layer,
        top_k_rows=args.top_k_rows,
        top_k_sources=args.top_k_sources,
    )
    summary_json = json.dumps(summary, indent=2, default=str)
    print(summary_json)
    if args.output_json is not None:
        args.output_json.write_text(summary_json, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
