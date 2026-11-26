"""Aggregate checkpoint attribution measurements without changing their denominator.

Head loadings partition summed source edges, which need not equal the measured decision. This report retains both:
source and head contributions are aligned with each item's decision and divided by the sum of absolute decisions. Layers
are separate partitions, so their head shares must not be added together. A supplied bias is measured by the collector;
an absent bias stays unknown, and the difference from the decision remains explicitly unassigned.

This module aggregates already measured rows. It does not load a checkpoint or certify that its identity describes the
weights used by the collector. The collector must establish that independently, including the decision check.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
from typing import Any, Sequence

# The partition check's absolute floor, by the precision the edges were computed in. Each partition and edge_total are
# separate sums of the same edges, so they differ by that precision's rounding, and a floor below it refuses a correct
# measurement. In float32 the gap measured flat over the 5,130 partitions of a GPU replay of all 190 RTE validation
# rows (median 1.2e-5, max 2.3e-5); 5e-5 clears it while a 1e-3 mismatch still fails. No other precision has a measured
# floor, so a caller states one explicitly.
PARTITION_ATOL = {"float32": 5e-5}


@dataclass(frozen=True)
class SparseAttributionRow:
    """One prompt's raw signed contributions, before normalization.

    Feature coordinates are ``(layer, position, feature)``. Error edges are layer by position, embedding edges are
    position, and head edges are reported layer by head. ``edge_total`` is the shared total every reported attention
    layer partitions with its bypass. ``bias`` is optional measured evidence, never inferred from a remainder.
    """

    row_id: str
    checkpoint_sha256: str
    basis_id: str
    decision: float
    answer_position: int
    feature_nodes: Sequence[tuple[int, int, int]]
    feature_edges: Sequence[float]
    error_edges: Sequence[Sequence[float]]
    embedding_edges: Sequence[float]
    head_layers: Sequence[int]
    head_edges: Sequence[Sequence[float]]
    bypass_edges: Sequence[float]
    edge_total: float
    bias: float | None = None
    freeze_attention: bool = True
    freeze_norms: bool = True


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def _index(value: int, size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ValueError(f"{name} must be an integer in [0, {size}), got {value!r}")
    return value


def _report_metadata(
    checkpoint_sha256: str,
    basis_id: str,
    model_id: str,
    model_revision: str,
    target_token_ids: tuple[int, int],
    top_k: int,
    measurement_dtype: str,
    partition_atol: float | None,
    partition_rtol: float,
) -> tuple[float, float]:
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA256 digest")
    if not all(isinstance(v, str) and v.strip() for v in (basis_id, model_id, model_revision)):
        raise ValueError("basis_id, model_id and model_revision must be stated")
    if len(target_token_ids) != 2 or any(type(v) is not int or v < 0 for v in target_token_ids):
        raise ValueError("target_token_ids must name two nonnegative token ids")
    if target_token_ids[0] == target_token_ids[1]:
        raise ValueError("target_token_ids must name distinct positive and negative targets")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(measurement_dtype, str) or not measurement_dtype.strip():
        raise ValueError("measurement_dtype must name the precision the edges were computed in")
    if partition_atol is None:
        if measurement_dtype not in PARTITION_ATOL:
            raise ValueError(
                f"no measured partition_atol for measurement_dtype {measurement_dtype!r} (measured: "
                f"{sorted(PARTITION_ATOL)}); state one explicitly"
            )
        partition_atol = PARTITION_ATOL[measurement_dtype]
    atol, rtol = _finite(partition_atol, "partition_atol"), _finite(partition_rtol, "partition_rtol")
    if atol < 0 or rtol < 0:
        raise ValueError("partition tolerances must be nonnegative")
    return atol, rtol


def _row_identity(
    row: SparseAttributionRow,
    checkpoint_sha256: str,
    basis_id: str,
    bias_known: bool,
    seen: set[str],
) -> None:
    if not row.row_id or row.row_id in seen:
        raise ValueError(f"row_id must be nonempty and unique, got {row.row_id!r}")
    if row.checkpoint_sha256 != checkpoint_sha256 or row.basis_id != basis_id:
        raise ValueError(f"row {row.row_id!r} has a different checkpoint or basis")
    if (row.bias is not None) != bias_known:
        raise ValueError("all rows must either measure bias or leave it unknown")
    if type(row.freeze_attention) is not bool or type(row.freeze_norms) is not bool:
        raise ValueError("each row must record boolean freeze_attention and freeze_norms")


def _row_layout(row: SparseAttributionRow, n_layers: int, layers: tuple[int, ...], n_heads: int) -> None:
    n_pos = len(row.embedding_edges)
    if len(row.error_edges) != n_layers or not n_pos:
        raise ValueError(f"row {row.row_id!r} has different model dimensions")
    if any(len(part) != n_pos for part in row.error_edges):
        raise ValueError(f"row {row.row_id!r} has an incomplete error-node layout")
    if tuple(row.head_layers) != layers or len(row.head_edges) != len(layers):
        raise ValueError(f"row {row.row_id!r} has different head layers")
    if len(row.bypass_edges) != len(layers) or any(len(part) != n_heads for part in row.head_edges):
        raise ValueError(f"row {row.row_id!r} has an incomplete head/bypass layout")
    if len(row.feature_nodes) != len(row.feature_edges) or len(set(row.feature_nodes)) != len(row.feature_nodes):
        raise ValueError(f"row {row.row_id!r} needs one edge per distinct feature coordinate")
    _index(row.answer_position, n_pos, "answer position")
    for layer, position, feature in row.feature_nodes:
        _index(layer, n_layers, "feature layer")
        _index(position, n_pos, "feature position")
        if type(feature) is not int or feature < 0:
            raise ValueError(f"feature id must be a nonnegative integer, got {feature!r}")


def _decision(row: SparseAttributionRow) -> float:
    decision = _finite(row.decision, "decision")
    if decision == 0.0:
        raise ValueError(f"row {row.row_id!r} has zero decision, so its alignment is undefined")
    return decision


def _partition_error(
    row: SparseAttributionRow,
    source_total: float,
    edge_total: float,
    row_heads: list[list[float]],
    row_bypass: list[float],
    atol: float,
    rtol: float,
) -> float:
    totals = [source_total, *(math.fsum((*part, rest)) for part, rest in zip(row_heads, row_bypass))]
    for total in totals:
        if not math.isclose(total, edge_total, abs_tol=atol, rel_tol=rtol):
            raise ValueError(f"row {row.row_id!r}: partition {total} does not match edge_total {edge_total}")
    return max(abs(total - edge_total) for total in totals)


def checkpoint_attribution_report(
    rows: Sequence[SparseAttributionRow],
    *,
    checkpoint_sha256: str,
    basis_id: str,
    model_id: str,
    model_revision: str,
    target_token_ids: tuple[int, int],
    measurement_dtype: str,
    top_k: int = 10,
    partition_atol: float | None = None,
    partition_rtol: float = 1e-5,
) -> dict[str, Any]:
    """Summarize measured rows for one checkpoint and basis, refusing incomplete or incompatible partitions.

    The target is the first token's logit minus the second's. Top answer-position features are ranked by the
    magnitude of their aggregated signed share, so opposing features remain visible. The output is JSON-compatible;
    an unknown bias is ``None``. It reports the head layers it actually received and whether they cover the model.
    ``measurement_dtype`` names the precision the edges were computed in; it selects the partition check's absolute
    floor from ``PARTITION_ATOL`` unless ``partition_atol`` is given, and both are recorded in the report.
    """
    atol, rtol = _report_metadata(
        checkpoint_sha256,
        basis_id,
        model_id,
        model_revision,
        target_token_ids,
        top_k,
        measurement_dtype,
        partition_atol,
        partition_rtol,
    )
    if not rows:
        raise ValueError("a checkpoint report needs at least one measured row")

    reference = rows[0]
    n_layers = len(reference.error_edges)
    n_pos = len(reference.embedding_edges)
    layers = tuple(reference.head_layers)
    n_heads = len(reference.head_edges[0]) if reference.head_edges else 0
    if not n_layers or not n_pos or not layers or not n_heads or len(set(layers)) != len(layers):
        raise ValueError("nonempty source and head dimensions, with distinct head layers, are required")
    for layer in layers:
        _index(layer, n_layers, "head layer")
    bias_known = reference.bias is not None
    seen: set[str] = set()
    denominator = 0.0
    source_sums = dict.fromkeys(("features", "errors", "embeddings", "bias", "unassigned"), 0.0)
    feature_layers = [0.0] * n_layers
    error_layers = [0.0] * n_layers
    heads = [[0.0] * n_heads for _ in layers]
    bypass = [0.0] * len(layers)
    answer_features: dict[tuple[int, int], float] = {}
    max_partition_error = 0.0
    item_records = []

    for row in rows:
        _row_identity(row, checkpoint_sha256, basis_id, bias_known, seen)
        if (row.freeze_attention, row.freeze_norms) != (reference.freeze_attention, reference.freeze_norms):
            raise ValueError("all rows must use the same attribution regime")
        _row_layout(row, n_layers, layers, n_heads)
        decision = _decision(row)
        sign = 1.0 if decision > 0 else -1.0
        edge_total = _finite(row.edge_total, "edge_total")
        features = [_finite(v, "feature edge") for v in row.feature_edges]
        errors = [[_finite(v, "error edge") for v in part] for part in row.error_edges]
        embeddings = [_finite(v, "embedding edge") for v in row.embedding_edges]
        row_heads = [[_finite(v, "head edge") for v in part] for part in row.head_edges]
        row_bypass = [_finite(v, "bypass edge") for v in row.bypass_edges]
        bias = _finite(row.bias, "bias") if row.bias is not None else 0.0
        source_total = math.fsum((*features, *(v for part in errors for v in part), *embeddings))
        max_partition_error = max(
            max_partition_error, _partition_error(row, source_total, edge_total, row_heads, row_bypass, atol, rtol)
        )
        for (layer, position, feature), edge in zip(row.feature_nodes, features):
            feature_layers[layer] += sign * edge
            if position == row.answer_position:
                key = (layer, feature)
                answer_features[key] = answer_features.get(key, 0.0) + sign * edge
        for layer, part in enumerate(errors):
            error_layers[layer] += sign * math.fsum(part)
        for i, part in enumerate(row_heads):
            for head, value in enumerate(part):
                heads[i][head] += sign * value
            bypass[i] += sign * row_bypass[i]
        source_sums["features"] += sign * math.fsum(features)
        source_sums["errors"] += sign * math.fsum(v for part in errors for v in part)
        source_sums["embeddings"] += sign * math.fsum(embeddings)
        source_sums["bias"] += sign * bias
        source_sums["unassigned"] += sign * (decision - source_total - bias)
        denominator += abs(decision)
        item_records.append({"row_id": row.row_id, "decision": decision, "edge_total": edge_total})
        seen.add(row.row_id)

    top = sorted(answer_features, key=lambda key: (-abs(answer_features[key]), key))[:top_k]
    result = {
        "format": "interpretune.sparse-attribution-report/1",
        "checkpoint_sha256": checkpoint_sha256,
        "basis_id": basis_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "target_token_ids": list(target_token_ids),
        "attribution_regime": {
            "freeze_attention": reference.freeze_attention,
            "freeze_norms": reference.freeze_norms,
        },
        "items": len(rows),
        "normalization": "decision_aligned_ratio_of_sums",
        "sum_abs_decision": denominator,
        "source_shares": {k: v / denominator if k != "bias" or bias_known else None for k, v in source_sums.items()},
        "feature_shares_by_layer": [v / denominator for v in feature_layers],
        "error_shares_by_layer": [v / denominator for v in error_layers],
        "head_layers": list(layers),
        "complete_head_layers": set(layers) == set(range(n_layers)),
        "head_shares": [[v / denominator for v in part] for part in heads],
        "bypass_shares": [v / denominator for v in bypass],
        "top_answer_features": [
            {"layer": layer, "feature": feature, "share": answer_features[layer, feature] / denominator}
            for layer, feature in top
        ],
        "max_partition_error": max_partition_error,
        "measurement_dtype": measurement_dtype,
        "partition_atol": atol,
        "partition_rtol": rtol,
        "rows": item_records,
    }
    json.dumps(result, allow_nan=False)
    return result
