"""Convert public head-loading measurements into checkpoint report rows.

The decision must be measured independently from logits on the selected checkpoint. Head loadings supply the source
edges, which need not sum to that decision. This conversion neither restores nor verifies checkpoint weights; those
checks belong to the checkpoint collector before it calls this function.
"""

from __future__ import annotations

import json
import hashlib
import math
from typing import Any, Mapping

from it_examples.utils.sparse_attribution_report import SparseAttributionRow


def _basis_layer_identity(tc: Any, layer: int) -> list[Any]:
    from circuit_tracer.transcoder import SingleLayerTranscoder
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    if type(tc) is not SingleLayerTranscoder or tc.layer_idx != layer or tc.lazy_encoder or tc.lazy_decoder:
        raise ValueError(f"basis layer {layer} must be a materialized SingleLayerTranscoder in layer order")
    if type(tc.activation_function) is not JumpReLU:
        raise ValueError(
            f"basis layer {layer} requires supported JumpReLU; got {type(tc.activation_function).__name__}"
        )
    bandwidth = float(tc.activation_function.bandwidth)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f"basis layer {layer} has invalid JumpReLU bandwidth")
    return [tc.d_model, tc.d_transcoder, bandwidth, tc.W_skip is not None]


def _basis_conventions(basis: Any, zero: Any) -> tuple[list[str], list[int | None]]:
    if not isinstance(zero, slice) or any(
        type(v) is not int and v is not None for v in (zero.start, zero.stop, zero.step)
    ):
        raise ValueError("basis fingerprinting requires an explicit integer zero-position slice")
    hooks = [basis.feature_input_hook, basis.feature_output_hook]
    if any(not isinstance(hook, str) or not hook for hook in hooks):
        raise ValueError("basis fingerprinting requires the feature input and output hooks")
    return hooks, [zero.start, zero.stop, zero.step]


def transcoder_basis_id(replacement_model: Any) -> str:
    """Fingerprint the actual per-layer JumpReLU basis, hooks and zero-position convention used for measurement.

    Names of Hub repositories are not evidence of loaded tensor contents. This hashes every transcoder parameter and
    buffer, including thresholds, and the non-tensor settings that change its feature basis. Lazy, cross-layer and
    unfamiliar activation implementations are refused until their identity can be recorded completely.
    """
    from circuit_tracer.transcoder import TranscoderSet
    from nnsight import Envoy

    from it_examples.utils.sparse_attribution_checkpoint import tensor_state_sha256

    basis = replacement_model.transcoders
    if isinstance(basis, Envoy):
        basis = basis._module
    if type(basis) is not TranscoderSet or len(basis) != replacement_model.cfg.n_layers:
        raise ValueError("basis fingerprinting requires one TranscoderSet layer per model layer")
    hooks, zero = _basis_conventions(basis, replacement_model.zero_positions)
    layers = [_basis_layer_identity(tc, layer) for layer, tc in enumerate(basis)]
    identity = {
        "format": "interpretune.transcoder-basis/1",
        "state_sha256": tensor_state_sha256(basis.state_dict()),
        "feature_hooks": hooks,
        "zero_positions": zero,
        "layers": layers,
    }
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


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
