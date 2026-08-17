"""Feature-selection specs and filters shared by feature-attribution ops.

Part of the sanctioned :mod:`interpretune.analysis.backends` seam that op implementations (bundled,
local, or hub) may import.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class FeatureSelectionSpec:
    """Pre-filter specification for :func:`extract_top_features_impl`.

    All criteria use **OR** semantics: a feature row ``(layer, position, feature_id)``
    passes the filter if it matches *any* of the non-empty criteria.

    Numeric slice notation is supported for ``layers`` and ``positions`` — pass a
    Python ``slice`` object alongside (or instead of) explicit ``int`` lists. The
    slice is applied as a numeric range over the observed values in
    *active_features*, so ``slice(10, None)`` means "layer >= 10" and
    ``slice(0, 10)`` means "position >= 0 and < 10".

    Attributes:
        layers: Explicit layer indices to include.
        positions: Explicit token-position indices to include.
        feature_ids: Explicit feature-ID values to include.
        layer_slice: A ``slice`` expanded over observed layer values.
        position_slice: A ``slice`` expanded over observed position values.
        triples: Exact ``(layer, position, feature_id)`` tuples to include.
        layer_feature_pairs: Exact ``(layer, feature_id)`` pairs to include across any position.
        activation_overrides: Optional override activation values keyed by ``(layer, feature_id)``.
        score_source: Optional analysis-batch field name or alias to use for feature ranking. Supported aliases include
            ``"influence"``, ``"signed_influence"``, and the planned backward-pass ``"gradient"`` /
            ``"logit_diff_gradient"`` channel for gradients of selected feature activations with respect to a target
            logit difference.
        score_sign: Optional sign filter for score values: ``"any"``, ``"positive"``, or ``"negative"``.
        rank_by_abs: If true, rank by absolute score magnitude while preserving the original signed score values.
    """

    layers: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    feature_ids: list[int] = field(default_factory=list)
    layer_slice: slice | None = None
    position_slice: slice | None = None
    triples: list[tuple[int, int, int]] = field(default_factory=list)
    layer_feature_pairs: list[tuple[int, int]] = field(default_factory=list)
    activation_overrides: dict[tuple[int, int], float] = field(default_factory=dict)
    score_source: str | None = None
    score_sign: str = "any"
    rank_by_abs: bool = False


def _expand_slice(s: slice, observed: torch.Tensor) -> list[int]:
    """Expand a numeric ``slice`` into concrete observed values."""
    unique_vals = sorted(observed.unique().tolist())
    if not unique_vals:
        return []

    filtered = [
        int(value)
        for value in unique_vals
        if (s.start is None or value >= s.start) and (s.stop is None or value < s.stop)
    ]
    if s.step not in (None, 1):
        filtered = filtered[:: int(s.step)]
    return filtered


def apply_feature_selection_filter(
    active_features: torch.Tensor,
    spec: FeatureSelectionSpec,
) -> torch.Tensor:
    """Return a boolean mask (length *N*) selecting rows of *active_features* that match *spec*.

    ``active_features`` has shape ``(N, 3)`` with columns ``[layer, position, feature_id]``.
    """
    n = active_features.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)

    mask = torch.zeros(n, dtype=torch.bool)

    layers_col = active_features[:, 0]
    positions_col = active_features[:, 1]
    features_col = active_features[:, 2]

    # Explicit layer list
    if spec.layers:
        layer_set = torch.tensor(spec.layers, dtype=layers_col.dtype)
        mask |= torch.isin(layers_col, layer_set)

    # Layer slice
    if spec.layer_slice is not None:
        expanded = _expand_slice(spec.layer_slice, layers_col)
        if expanded:
            layer_set = torch.tensor(expanded, dtype=layers_col.dtype)
            mask |= torch.isin(layers_col, layer_set)

    # Explicit position list
    if spec.positions:
        pos_set = torch.tensor(spec.positions, dtype=positions_col.dtype)
        mask |= torch.isin(positions_col, pos_set)

    # Position slice
    if spec.position_slice is not None:
        expanded = _expand_slice(spec.position_slice, positions_col)
        if expanded:
            pos_set = torch.tensor(expanded, dtype=positions_col.dtype)
            mask |= torch.isin(positions_col, pos_set)

    # Explicit feature IDs
    if spec.feature_ids:
        fid_set = torch.tensor(spec.feature_ids, dtype=features_col.dtype)
        mask |= torch.isin(features_col, fid_set)

    # Exact (layer, position, feature_id) triples
    if spec.triples:
        triple_tensor = torch.tensor(spec.triples, dtype=active_features.dtype)  # (T, 3)
        # Compare every row against every triple: (N, 1, 3) == (T, 3) → (N, T, 3)
        matches = (active_features.unsqueeze(1) == triple_tensor.unsqueeze(0)).all(dim=2)  # (N, T)
        mask |= matches.any(dim=1)

    # Exact (layer, feature_id) pairs across any position
    if spec.layer_feature_pairs:
        pair_tensor = torch.tensor(spec.layer_feature_pairs, dtype=active_features.dtype)  # (P, 2)
        pair_rows = torch.stack((layers_col, features_col), dim=1)
        matches = (pair_rows.unsqueeze(1) == pair_tensor.unsqueeze(0)).all(dim=2)  # (N, P)
        mask |= matches.any(dim=1)

    return mask


def apply_feature_score_sign_filter(scores: torch.Tensor, score_sign: str = "any") -> torch.Tensor:
    """Return a boolean mask selecting feature scores with the requested sign."""
    if score_sign == "any":
        return torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if score_sign == "positive":
        return scores > 0
    if score_sign == "negative":
        return scores < 0
    raise ValueError("score_sign must be one of 'any', 'positive', or 'negative'")


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


def augment_feature_rows_for_selection(
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


def select_top_feature_indices(
    feature_rows: torch.Tensor,
    scores: torch.Tensor,
    top_n: int | None,
    feature_selection: FeatureSelectionSpec | None,
    *,
    rank_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    effective_rank_scores = scores if rank_scores is None else rank_scores
    ranked_indices = torch.argsort(effective_rank_scores, descending=True)
    selected_count = scores.shape[0] if top_n is None else min(int(top_n), scores.shape[0])
    if selected_count <= 0:
        return torch.empty((0,), dtype=torch.long)

    selected = ranked_indices[:selected_count].tolist()
    if feature_selection is None or not feature_selection.layer_feature_pairs:
        return torch.tensor(selected, dtype=torch.long)

    rank_by_index = {int(index): rank for rank, index in enumerate(ranked_indices.tolist())}
    guaranteed: list[int] = []
    for layer, feature_id in dict.fromkeys(feature_selection.layer_feature_pairs):
        match_mask = (feature_rows[:, 0] == int(layer)) & (feature_rows[:, 2] == int(feature_id))
        if not match_mask.any():
            continue
        pair_indices = match_mask.nonzero(as_tuple=False).reshape(-1)
        pair_scores = effective_rank_scores.index_select(0, pair_indices)
        best_pair_index = int(pair_indices[int(torch.argmax(pair_scores).item())].item())
        guaranteed.append(best_pair_index)

    if not guaranteed:
        return torch.tensor(selected, dtype=torch.long)

    guaranteed = sorted(dict.fromkeys(guaranteed), key=lambda index: rank_by_index[index])
    guaranteed_set = set(guaranteed)
    selected_set = set(selected)

    for guaranteed_index in guaranteed:
        if guaranteed_index in selected_set:
            continue
        replace_position = next(
            (position for position in range(len(selected) - 1, -1, -1) if selected[position] not in guaranteed_set),
            None,
        )
        if replace_position is None:
            selected.append(guaranteed_index)
            selected_set.add(guaranteed_index)
            continue
        removed_index = selected[replace_position]
        selected[replace_position] = guaranteed_index
        selected_set.discard(removed_index)
        selected_set.add(guaranteed_index)

    selected = sorted(dict.fromkeys(selected), key=lambda index: rank_by_index[index])
    return torch.tensor(selected, dtype=torch.long)


def apply_optional_feature_sign_filter(
    feature_rows: torch.Tensor,
    scores: torch.Tensor,
    activation_values: torch.Tensor | None,
    feature_selection: FeatureSelectionSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    score_mask = apply_feature_score_sign_filter(scores, feature_selection.score_sign)
    if feature_selection.score_sign == "any":
        return feature_rows, scores, activation_values
    selected_indices = score_mask.nonzero(as_tuple=False).reshape(-1)
    feature_rows = feature_rows.index_select(0, selected_indices)
    scores = scores.index_select(0, selected_indices)
    if activation_values is not None:
        activation_values = activation_values.index_select(0, selected_indices)
    return feature_rows, scores, activation_values
