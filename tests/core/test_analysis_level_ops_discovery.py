from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from interpretune.analysis.backends import AnalysisBackendCapability, BackendCapability, get_module_capabilities
from interpretune.analysis.ops.base import AnalysisBatch, AnalysisOp, CompositeAnalysisOp
from interpretune.analysis.ops.dispatcher import DISPATCHER


class _DummyBackend:
    def __init__(self, capabilities: frozenset[BackendCapability]):
        self.capabilities = capabilities


class _DummyModule(torch.nn.Module):
    def __init__(
        self,
        backend_capabilities: frozenset[BackendCapability] | None = None,
        analysis_capabilities: frozenset[AnalysisBackendCapability] | None = None,
    ) -> None:
        super().__init__()
        self._model_backend = _DummyBackend(backend_capabilities or frozenset())
        self.analysis_capabilities = analysis_capabilities or frozenset()


def test_analysis_level_ops_are_discoverable() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    assert op.name == "compute_attribution_graph"

    concept_op = DISPATCHER.get_op("concept_direction")
    assert isinstance(concept_op, AnalysisOp)
    assert concept_op.name == "concept_direction"


def test_analysis_level_composite_ops_resolve() -> None:
    op = DISPATCHER.get_op("intervention_from_concept")
    assert isinstance(op, CompositeAnalysisOp)
    assert [sub_op.name for sub_op in op.composition] == [
        "concept_direction",
        "compute_attribution_graph",
        "graph_node_influence",
        "extract_top_features",
        "feature_intervention_forward",
    ]


def test_required_capabilities_are_parsed() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    assert {cap.value for cap in op.required_capabilities} == {"attribution_graph"}

    intervention_op = DISPATCHER.get_op("feature_intervention_forward")
    assert isinstance(intervention_op, AnalysisOp)
    assert {cap.value for cap in intervention_op.required_capabilities} == {"feature_intervention"}


def test_get_module_capabilities_aggregates_backend_and_adapter_capabilities() -> None:
    module = _DummyModule(
        backend_capabilities=frozenset({BackendCapability.GRADIENTS}),
        analysis_capabilities=frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH}),
    )

    capabilities = get_module_capabilities(module)
    assert capabilities.model == frozenset({BackendCapability.GRADIENTS})
    assert capabilities.analysis == frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH})
    assert capabilities.values == frozenset({"gradients", "attribution_graph"})


def test_capability_validation_rejects_missing_module_capability() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    module = _DummyModule(backend_capabilities=frozenset({BackendCapability.GRADIENTS}))

    with pytest.raises(ValueError, match="requires capabilities"):
        op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)


def test_capability_validation_allows_matching_analysis_capability() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    module = _DummyModule(analysis_capabilities=frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH}))
    mock_impl = MagicMock(return_value=AnalysisBatch(ok=True))
    original_impl = op._impl
    try:
        op._impl = mock_impl

        result = cast(AnalysisBatch, op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0))

        assert result.ok is True
        mock_impl.assert_called_once()
    finally:
        op._impl = original_impl


def test_composite_ops_validate_capabilities_per_stage() -> None:
    op = DISPATCHER.get_op("intervention_from_concept")
    assert isinstance(op, CompositeAnalysisOp)
    module = _DummyModule(analysis_capabilities=frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH}))

    def noop_impl(module, analysis_batch, batch, batch_idx, **kwargs):
        return analysis_batch

    original_impls = [sub_op._impl for sub_op in op.composition[:-1]]
    try:
        for sub_op in op.composition[:-1]:
            sub_op._impl = noop_impl

        analysis_batch = AnalysisBatch(
            concept_group_a=["Paris"],
            concept_group_b=["London"],
            input_string="Paris London",
            adjacency_matrix=[[0.0]],
            active_features=[[0, 0, 0]],
            selected_features=[0],
            activation_values=[0.1],
            logit_target_ids=[0],
            logit_target_tokens=["Paris"],
            logit_probabilities=[1.0],
            input_tokens=[0],
            graph_cfg_json=(
                "{"
                '"n_layers": 1, "d_model": 1, "d_head": 1, "n_heads": 1, '
                '"d_mlp": 1, "d_vocab": 1, "tokenizer_name": "fake", '
                '"model_name": "fake", "original_architecture": "Fake"'
                "}"
            ),
            graph_scan_json='"scan"',
            graph_vocab_size=1,
            top_feature_ids=[[0, 0, 0]],
            top_feature_scores=[0.1],
        )

        with pytest.raises(ValueError, match="feature_intervention"):
            op(module=module, analysis_batch=analysis_batch, batch=None, batch_idx=0)
    finally:
        for sub_op, original_impl in zip(op.composition[:-1], original_impls, strict=True):
            sub_op._impl = original_impl
