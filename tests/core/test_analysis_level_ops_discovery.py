from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from interpretune.analysis.backends import AnalysisBackendCapability, ModelBackendCapability, get_module_capabilities
from interpretune.analysis.ops.base import AnalysisBatch, AnalysisOp, CompositeAnalysisOp
from interpretune.analysis.ops.dispatcher import DISPATCHER


class _DummyBackend:
    def __init__(self, capabilities: frozenset[ModelBackendCapability]):
        self.capabilities = capabilities


class _DummyModule(torch.nn.Module):
    def __init__(
        self,
        backend_capabilities: frozenset[ModelBackendCapability] | None = None,
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
        backend_capabilities=frozenset({ModelBackendCapability.GRADIENTS}),
        analysis_capabilities=frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH}),
    )

    capabilities = get_module_capabilities(module)
    assert capabilities.model == frozenset({ModelBackendCapability.GRADIENTS})
    assert capabilities.analysis == frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH})
    assert capabilities.values == frozenset({"gradients", "attribution_graph"})


def test_capability_validation_rejects_missing_module_capability() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    module = _DummyModule(backend_capabilities=frozenset({ModelBackendCapability.GRADIENTS}))

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


class _InterveningBackend(_DummyBackend):
    def __init__(self, modes, scopes):
        from interpretune.analysis.backends import InterventionSupport

        super().__init__(frozenset({ModelBackendCapability.ACTIVATION_INTERVENTION}))
        self.intervention_support = InterventionSupport(position_scopes=frozenset(scopes), modes=frozenset(modes))


def _intervening_module(modes=("add",), scopes=("last_token",)) -> _DummyModule:
    module = _DummyModule()
    module._model_backend = _InterveningBackend(modes, scopes)
    return module


def _op(name="probe", modes=None, scopes=None, capabilities=None) -> AnalysisOp:
    from interpretune.analysis.ops.base import OpSchema

    op = AnalysisOp(
        name=name,
        description="",
        output_schema=OpSchema({}),
        required_capabilities=capabilities,
        required_intervention_modes=modes,
        required_position_scopes=scopes,
    )
    op._impl = lambda module, analysis_batch, batch, batch_idx, **kw: analysis_batch
    return op


class TestInterventionRequirementAxes:
    """`required_intervention_modes` / `required_position_scopes`: the INTERVENTION configurations an op needs."""

    def test_axes_normalize_to_the_vocabulary_and_refuse_unknown_values(self):
        from interpretune.analysis.backends import InterventionMode, PositionScope

        op = _op(modes=["patch", InterventionMode.ADD], scopes=["all_positions"])
        assert op.required_intervention_modes == frozenset({InterventionMode.PATCH, InterventionMode.ADD})
        assert op.required_position_scopes == frozenset({PositionScope.ALL_POSITIONS})
        with pytest.raises(ValueError, match="sorcery"):
            _op(modes=["sorcery"])

    def test_a_missing_mode_is_refused_by_axis_before_execution(self):
        op = _op(modes=["patch"])
        module = _intervening_module(modes=("add",), scopes=("last_token",))
        with pytest.raises(ValueError, match=r"requires intervention modes \['patch'\].*declares \['add'\].*mode axis"):
            op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)

    def test_a_missing_scope_is_refused_by_axis(self):
        op = _op(scopes=["all_positions"])
        module = _intervening_module(modes=("add",), scopes=("last_token",))
        with pytest.raises(ValueError, match=r"position_scopes \['all_positions'\].*declares \['last_token'\]"):
            op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)

    def test_declaring_an_axis_implies_the_intervention_surface(self):
        op = _op(modes=["add"])
        module = _DummyModule(backend_capabilities=frozenset({ModelBackendCapability.GRADIENTS}))
        with pytest.raises(ValueError, match="declares no activation_intervention surface"):
            op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)

    def test_declared_axes_pass_on_a_backend_that_honours_them(self):
        op = _op(modes=["patch"], scopes=["all_positions"])
        module = _intervening_module(modes=("add", "patch"), scopes=("last_token", "all_positions"))
        result = op(module=module, analysis_batch=AnalysisBatch(ok=True), batch=None, batch_idx=0)
        assert cast(AnalysisBatch, result).ok is True

    def test_composites_carry_the_union_of_their_parts_plus_their_own(self):
        from interpretune.analysis.backends import InterventionMode, PositionScope

        a = _op("a", modes=["add"])
        b = _op("b", scopes=["all_positions"])
        composite = CompositeAnalysisOp([a, b], name="a_then_b", required_intervention_modes=["patch"])
        assert composite.required_intervention_modes == frozenset({InterventionMode.ADD, InterventionMode.PATCH})
        assert composite.required_position_scopes == frozenset({PositionScope.ALL_POSITIONS})
        # and the composite refuses BEFORE its first part runs, on the union
        ran = []
        a._impl = lambda module, analysis_batch, batch, batch_idx, **kw: ran.append("a") or analysis_batch
        module = _intervening_module(modes=("add",), scopes=("last_token", "all_positions"))
        with pytest.raises(ValueError, match=r"missing \['patch'\]"):
            composite(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)
        assert ran == []

    def test_the_bundled_intervention_op_declares_the_surface(self):
        op = DISPATCHER.get_op("model_fwd_intervention")
        assert isinstance(op, AnalysisOp)
        # by VALUE: the suite can load the capabilities module twice, leaving value-equal, identity-distinct members
        assert ModelBackendCapability.ACTIVATION_INTERVENTION.value in {c.value for c in op.required_capabilities}
        assert not op.requires_intervention_axes, "the bundled op is payload-driven and fixes no mode itself"
