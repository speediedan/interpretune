"""Convert public head-loading measurements into checkpoint report rows.

The decision must be measured independently from logits on the selected checkpoint. Head loadings supply the source
edges, which need not sum to that decision. This conversion neither restores nor verifies checkpoint weights; those
checks belong to the checkpoint collector before it calls this function.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from it_examples.utils.sparse_attribution_report import SparseAttributionRow


def head_loading_row(
    report: Mapping[str, Any],
    *,
    row_id: str,
    checkpoint_sha256: str,
    basis_id: str,
    decision: float,
    answer_position: int,
) -> SparseAttributionRow:
    """Retain the raw source coordinates, head partitions and measured regime of one public op result.

    No bias measurement is exposed by head loadings, so bias remains unknown. The checkpoint report validates layout,
    finite numbers and source/head partition sums when it aggregates the rows.
    """
    regime = json.loads(report["head_loading_regime_json"])
    if not isinstance(regime, dict) or set(regime) != {"freeze_attention", "freeze_norms"}:
        raise ValueError("head-loading regime must record freeze_attention and freeze_norms")
    if any(type(value) is not bool for value in regime.values()):
        raise ValueError("head-loading regime must contain boolean freeze flags")
    return SparseAttributionRow(
        row_id=row_id,
        checkpoint_sha256=checkpoint_sha256,
        basis_id=basis_id,
        decision=decision,
        answer_position=answer_position,
        feature_nodes=[tuple(node) for node in report["head_loading_feature_nodes"].tolist()],
        feature_edges=report["head_loading_feature_edges"].tolist(),
        error_edges=report["head_loading_error_edges"].tolist(),
        embedding_edges=report["head_loading_embedding_edges"].tolist(),
        head_layers=report["head_loading_layers"].tolist(),
        head_edges=report["head_loading_head_edges"].tolist(),
        bypass_edges=report["head_loading_bypass_edges"].tolist(),
        edge_total=float(report["head_loading_total"]),
        freeze_attention=regime["freeze_attention"],
        freeze_norms=regime["freeze_norms"],
    )
