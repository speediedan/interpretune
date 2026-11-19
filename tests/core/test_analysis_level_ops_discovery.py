from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from interpretune.analysis.backends import (
    Capability,
    AttributionGraphSupport,
    FeatureTargetSupport,
    InterventionSupport,
    PositionScope,
    TechniqueSupport,
    get_module_capabilities,
)
from interpretune.analysis.ops.base import AnalysisBatch, AnalysisOp, CompositeAnalysisOp
from interpretune.analysis.ops.dispatcher import DISPATCHER


class _DummyBackend:
    def __init__(self, capabilities: frozenset[Capability]):
        self.capabilities = capabilities


class _GraphRecordWithoutMethodCheck(AttributionGraphSupport):
    """The attribution-graph record for a dummy that has none of the graph methods, which these gate tests never
    call."""

    def provider_protocol(self) -> type | None:
        return None


class _DummyAnalysisBackend:
    """Declares analysis surfaces with the records that must travel with them."""

    def __init__(self, capabilities: frozenset[Capability], techniques: tuple[str, ...] = ("attribution_graph",)):
        self.capabilities = capabilities
        self.technique_support = {
            name: _GraphRecordWithoutMethodCheck(requires_own_eager_attention=False) for name in techniques
        }
        self.intervention_support = InterventionSupport(
            ops=frozenset({"replace"}),
            position_scopes=frozenset({PositionScope.POSITIONS}),
            targets={"features": FeatureTargetSupport(value_sources=frozenset({"constant"}))},
        )


class _DummyModule(torch.nn.Module):
    def __init__(
        self,
        backend_capabilities: frozenset[Capability] | None = None,
        analysis_capabilities: frozenset[Capability] | None = None,
        techniques: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._model_backend = _DummyBackend(backend_capabilities or frozenset())
        if analysis_capabilities is not None or techniques:
            self._analysis_backend = _DummyAnalysisBackend(analysis_capabilities or frozenset(), techniques)


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


def test_required_capabilities_and_techniques_are_parsed() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    assert op.required_capabilities == frozenset()
    assert op.required_techniques == {"attribution_graph"}

    intervention_op = DISPATCHER.get_op("feature_intervention_forward")
    assert isinstance(intervention_op, AnalysisOp)
    assert {cap.value for cap in intervention_op.required_capabilities} == {"intervention"}


def test_get_module_capabilities_aggregates_backend_and_adapter_capabilities() -> None:
    module = _DummyModule(backend_capabilities=frozenset({Capability.GRADIENTS}), techniques=("attribution_graph",))

    capabilities = get_module_capabilities(module)
    assert capabilities.providers == {Capability.GRADIENTS: ("model_backend",)}
    assert capabilities.values == frozenset({"gradients"})
    assert set(capabilities.techniques) == {"attribution_graph"}
    assert capabilities.technique_for("attribution_graph")[0] == "analysis_backend"


def test_capability_validation_rejects_missing_module_capability() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    module = _DummyModule(backend_capabilities=frozenset({Capability.GRADIENTS}))

    with pytest.raises(ValueError, match="no component of this module implements technique 'attribution_graph'"):
        op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)


def test_capability_validation_allows_matching_analysis_capability() -> None:
    op = DISPATCHER.get_op("compute_attribution_graph")
    assert isinstance(op, AnalysisOp)
    module = _DummyModule(techniques=("attribution_graph",))
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
    module = _DummyModule(techniques=("attribution_graph",))

    def noop_impl(module, analysis_batch, batch, batch_idx, **kwargs):
        return analysis_batch

    original_impls = [sub_op._impl for sub_op in op.composition[:-1]]
    try:
        for sub_op in op.composition[:-1]:
            sub_op._impl = noop_impl

        analysis_batch = AnalysisBatch(
            concept_group_a=["Paris"],
            concept_group_b=["London"],
            concept_basis="embed",
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

        with pytest.raises(ValueError, match=r"feature_intervention_forward.*missing \[intervention\]"):
            op(module=module, analysis_batch=analysis_batch, batch=None, batch_idx=0)
    finally:
        for sub_op, original_impl in zip(op.composition[:-1], original_impls, strict=True):
            sub_op._impl = original_impl


class _InterveningBackend(_DummyBackend):
    def __init__(self, modes, scopes):
        from interpretune.analysis.backends import InterventionSupport

        super().__init__(frozenset({Capability.INTERVENTION}))
        self.intervention_support = InterventionSupport(position_scopes=frozenset(scopes), ops=frozenset(modes))


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
        required_tensor_ops=modes,
        required_position_scopes=scopes,
    )
    op._impl = lambda module, analysis_batch, batch, batch_idx, **kw: analysis_batch
    return op


class TestInterventionRequirementAxes:
    """`required_tensor_ops` / `required_position_scopes`: the INTERVENTION configurations an op needs."""

    def test_axes_normalize_to_the_vocabulary_and_refuse_unknown_values(self):
        from interpretune.analysis.backends import PositionScope

        op = _op(modes=["patch", "add"], scopes=["all_positions"])
        assert op.required_tensor_ops == frozenset({"patch", "add"})
        assert op.required_position_scopes == frozenset({PositionScope.ALL_POSITIONS})
        with pytest.raises(ValueError, match="sorcery"):
            _op(modes=["sorcery"])

    def test_a_missing_mode_is_refused_by_axis_before_execution(self):
        op = _op(modes=["patch"])
        module = _intervening_module(modes=("add",), scopes=("last_token",))
        with pytest.raises(ValueError, match=r"requires intervention ops \['patch'\].*declares \['add'\].*op axis"):
            op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)

    def test_a_missing_scope_is_refused_by_axis(self):
        op = _op(scopes=["all_positions"])
        module = _intervening_module(modes=("add",), scopes=("last_token",))
        with pytest.raises(ValueError, match=r"position_scopes \['all_positions'\].*declares \['last_token'\]"):
            op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)

    def test_declaring_an_axis_implies_the_intervention_surface(self):
        op = _op(modes=["add"])
        module = _DummyModule(backend_capabilities=frozenset({Capability.GRADIENTS}))
        with pytest.raises(ValueError, match="declares no intervention surface"):
            op(module=module, analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)

    def test_declared_axes_pass_on_a_backend_that_honours_them(self):
        op = _op(modes=["patch"], scopes=["all_positions"])
        module = _intervening_module(modes=("add", "patch"), scopes=("last_token", "all_positions"))
        result = op(module=module, analysis_batch=AnalysisBatch(ok=True), batch=None, batch_idx=0)
        assert cast(AnalysisBatch, result).ok is True

    def test_composites_carry_the_union_of_their_parts_plus_their_own(self):
        from interpretune.analysis.backends import PositionScope

        a = _op("a", modes=["add"])
        b = _op("b", scopes=["all_positions"])
        composite = CompositeAnalysisOp([a, b], name="a_then_b", required_tensor_ops=["patch"])
        assert composite.required_tensor_ops == frozenset({"add", "patch"})
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
        assert Capability.INTERVENTION.value in {c.value for c in op.required_capabilities}
        assert not op.requires_intervention_axes, "the bundled op is payload-driven and fixes no mode itself"


class _ToySupport(TechniqueSupport):
    """A third party's technique record: a subclass naming its technique, with no edit to the capability enum."""

    technique = "toy_technique"


class _ToyProvider(_DummyBackend):
    def __init__(self, records):
        super().__init__(frozenset())
        self.technique_support = records


def _technique_op(*techniques: str) -> AnalysisOp:
    from interpretune.analysis.ops.base import OpSchema

    op = AnalysisOp(name="toy_op", description="", output_schema=OpSchema({}), required_techniques=techniques)
    op._impl = lambda module, analysis_batch, batch, batch_idx, **kw: analysis_batch
    return op


class TestTechniquesBindToProviders:
    """An op names the techniques it executes; exactly one component must declare a typed record for each."""

    def test_a_third_party_technique_needs_no_enum_edit(self):
        with pytest.raises(ValueError, match="'toy_technique' is not a capability"):
            from interpretune.analysis.backends import normalize_backend_capability

            normalize_backend_capability("toy_technique")
        module = _DummyModule()
        module._model_backend = _ToyProvider({"toy_technique": _ToySupport()})
        _technique_op("toy_technique")._validate_capabilities(module)
        module._model_backend = _ToyProvider({})
        with pytest.raises(
            ValueError, match="toy_op.*no component of this module implements technique 'toy_technique'"
        ):
            _technique_op("toy_technique")._validate_capabilities(module)

    def test_core_surfaces_without_an_implementation_are_refused(self):
        surfaces = frozenset({Capability.INTERVENTION, Capability.GRADIENTS})
        module = _intervening_module()
        module._model_backend.capabilities = surfaces
        op = DISPATCHER.get_op("head_loadings")
        assert isinstance(op, AnalysisOp)
        with pytest.raises(ValueError, match="implements technique 'head_attribution'.*surfaces provided"):
            op._validate_capabilities(module)

    def test_two_providers_are_ambiguous(self):
        module = _DummyModule(techniques=("toy_technique",))
        module._analysis_backend.technique_support = {"toy_technique": _ToySupport()}
        module._model_backend = _ToyProvider({"toy_technique": _ToySupport()})
        with pytest.raises(
            ValueError, match=r"implemented by \['analysis_backend', 'model_backend'\]; name the provider"
        ):
            _technique_op("toy_technique")._validate_capabilities(module)
        caps = get_module_capabilities(module)
        assert caps.technique_for("toy_technique", provider="model_backend")[0] == "model_backend"

    @pytest.mark.parametrize(
        "records, match",
        [
            ({"toy_technique": None}, "not a TechniqueSupport record"),
            ({"toy_technique": {"requires_own_eager_attention": True}}, "not a TechniqueSupport record"),
            ({"attribution_graph": _ToySupport()}, "_ToySupport, which serves 'toy_technique'"),
            ({"attribution_graph": AttributionGraphSupport()}, "does not implement SupportsAttributionGraph"),
            (["attribution_graph"], "maps each technique name"),
        ],
    )
    def test_an_untyped_or_mismatched_record_is_refused(self, records, match):
        module = _DummyModule()
        module._model_backend = _ToyProvider(records)
        with pytest.raises(ValueError, match=match):
            get_module_capabilities(module)

    def test_a_retired_technique_capability_spelling_points_at_the_technique_field(self):
        with pytest.raises(ValueError, match="'attribution_graph' is a retired capability spelling.*`technique"):
            _op(capabilities=["attribution_graph"])

    def test_a_composite_refuses_a_missing_technique_before_its_first_part_runs(self):
        op = DISPATCHER.get_op("attribution_from_concept")
        assert isinstance(op, CompositeAnalysisOp)
        assert op.required_techniques == {"attribution_graph"}
        first = op.composition[0]
        original = first._impl
        first._impl = MagicMock(side_effect=AssertionError("the first part ran"))
        try:
            with pytest.raises(ValueError, match="attribution_from_concept.*technique 'attribution_graph'"):
                op(module=_DummyModule(), analysis_batch=AnalysisBatch(), batch=None, batch_idx=0)
        finally:
            first._impl = original
