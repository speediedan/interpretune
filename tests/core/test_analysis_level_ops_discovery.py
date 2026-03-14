from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from interpretune.analysis.backends import BackendCapability, get_module_capabilities
from interpretune.analysis.ops.base import AnalysisBatch, AnalysisOp, CompositeAnalysisOp
from interpretune.analysis.ops.dispatcher import DISPATCHER


class _DummyBackend:
    def __init__(self, capabilities: frozenset[BackendCapability]):
        self.capabilities = capabilities


class _DummyModule(torch.nn.Module):
    def __init__(
        self,
        backend_capabilities: frozenset[BackendCapability] | None = None,
        analysis_capabilities: frozenset[BackendCapability] | None = None,
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
    op = DISPATCHER.get_op("ct_full_analysis")
    assert isinstance(op, CompositeAnalysisOp)
    assert [sub_op.name for sub_op in op.composition] == [
        "concept_direction",
        "compute_attribution_graph",
        "graph_prune",
        "extract_top_features",
        "feature_intervention_forward",
    ]


def test_required_capabilities_are_parsed() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    assert {cap.value for cap in op.required_capabilities} == {"attribution"}

    intervention_op = DISPATCHER.get_op("feature_intervention_forward")
    assert isinstance(intervention_op, AnalysisOp)
    assert {cap.value for cap in intervention_op.required_capabilities} == {"feature_intervention"}


def test_get_module_capabilities_aggregates_backend_and_adapter_capabilities() -> None:
    module = _DummyModule(
        backend_capabilities=frozenset({BackendCapability.GRADIENTS}),
        analysis_capabilities=frozenset({BackendCapability.ATTRIBUTION}),
    )

    assert get_module_capabilities(module) == frozenset({BackendCapability.GRADIENTS, BackendCapability.ATTRIBUTION})


def test_capability_validation_rejects_missing_module_capability() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    module = _DummyModule(backend_capabilities=frozenset({BackendCapability.GRADIENTS}))

    with pytest.raises(ValueError, match="requires capabilities"):
        op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)


def test_capability_validation_allows_matching_analysis_capability() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    module = _DummyModule(analysis_capabilities=frozenset({BackendCapability.ATTRIBUTION}))
    mock_impl = MagicMock(return_value=AnalysisBatch(ok=True))
    op._impl = mock_impl

    result = cast(AnalysisBatch, op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0))

    assert result.ok is True
    mock_impl.assert_called_once()


def test_composite_ops_validate_capabilities_per_stage() -> None:
    op = DISPATCHER.get_op("ct_full_analysis")
    assert isinstance(op, CompositeAnalysisOp)
    module = _DummyModule(analysis_capabilities=frozenset({BackendCapability.ATTRIBUTION}))

    def noop_impl(module, analysis_batch, batch, batch_idx, **kwargs):
        return analysis_batch

    for sub_op in op.composition[:-1]:
        sub_op._impl = noop_impl

    analysis_batch = AnalysisBatch(
        concept_group_a=["Paris"],
        concept_group_b=["London"],
        graph_pt_bytes=b"graph-bytes",
        active_features=[[0, 0, 0]],
        activation_values=[0.1],
        top_feature_ids=[[0, 0, 0]],
        top_feature_scores=[0.1],
    )

    with pytest.raises(ValueError, match="feature_intervention"):
        op(module=module, analysis_batch=analysis_batch, batch=None, batch_idx=0)
