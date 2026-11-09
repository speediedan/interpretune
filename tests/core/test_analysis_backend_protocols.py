"""The analysis-backend surface decomposes like the model-backend one: a core plus gated groups with records.

A ``Capability`` member an analysis backend declares names a ``Supports*`` group and answers whether it is
implemented; the group's configuration space is a typed record on the protocol. These tests pin the decomposition,
the records' refusals, and the provenance refusal that guards graph construction, each with a positive control.
"""

from __future__ import annotations

import types

import pytest
import torch

from interpretune.analysis.backends import (
    AnalysisBackend,
    Capability,
    AnalysisBackendCore,
    AttributionGraphSupport,
    FeatureTargetSupport,
    InterventionMode,
    InterventionSupport,
    ModuleCapabilities,
    PositionScope,
    SupportsAttributionGraph,
    SupportsFeatureInterventions,
    SupportsHeadAttribution,
)


class _CoreOnlyAnalysisBackend:
    """Implements the core and nothing gated."""

    @property
    def capabilities(self):
        return frozenset()

    @property
    def requires(self):
        return frozenset()

    def supports(self, capability):
        return False

    def get_tokenizer(self, module):
        return None

    def get_embedding_weight(self, module):
        return torch.zeros(1, 1)

    def token_strings_to_ids(self, tokenizer, token_strings):
        return []

    def resolve_prompt(self, module, analysis_batch, batch):
        return ""


class TestDecomposition:
    def test_a_core_only_backend_satisfies_the_core_and_neither_group(self):
        backend = _CoreOnlyAnalysisBackend()
        assert isinstance(backend, AnalysisBackendCore)
        assert not isinstance(backend, SupportsAttributionGraph)
        assert not isinstance(backend, SupportsFeatureInterventions)
        assert not isinstance(backend, SupportsHeadAttribution)
        assert not isinstance(backend, AnalysisBackend)

    def test_the_bundled_backend_satisfies_every_group(self):
        from interpretune.adapters.circuit_tracer.backends import CircuitTracerAnalysisBackend

        backend = CircuitTracerAnalysisBackend()
        for protocol in (
            AnalysisBackendCore,
            SupportsAttributionGraph,
            SupportsFeatureInterventions,
            SupportsHeadAttribution,
            AnalysisBackend,
        ):
            assert isinstance(backend, protocol), protocol.__name__
        assert backend.capabilities == {
            Capability.ATTRIBUTION_GRAPH,
            Capability.INTERVENTION,
            Capability.HEAD_ATTRIBUTION,
        }

    def test_head_attribution_is_not_required_of_every_full_backend(self):
        """A backend can carry graphs and feature interventions without these reports, and stays an
        ``AnalysisBackend``: the capability refuses the ops by name, so nothing else needs to change for it."""
        assert SupportsHeadAttribution not in AnalysisBackend.__mro__


_FEATURES = InterventionSupport(
    ops=frozenset({InterventionMode.REPLACE}),
    position_scopes=frozenset({PositionScope.POSITIONS}),
    targets={"features": FeatureTargetSupport(value_sources=frozenset({"constant"}))},
)


class TestRecordsTravelWithTheirSurface:
    def test_a_declared_analysis_surface_needs_its_record(self):
        with pytest.raises(ValueError, match="ATTRIBUTION_GRAPH is declared but no attribution_graph support record"):
            ModuleCapabilities(providers={Capability.ATTRIBUTION_GRAPH: ("analysis_backend",)})
        with pytest.raises(ValueError, match=r"INTERVENTION is provided by \['analysis_backend'\] and support records"):
            ModuleCapabilities(providers={Capability.INTERVENTION: ("analysis_backend",)})

    def test_a_record_without_its_surface_is_refused(self):
        with pytest.raises(ValueError, match=r"INTERVENTION is provided by \[\] and support records accompany"):
            ModuleCapabilities(providers={}, intervention={"analysis_backend": _FEATURES})

    def test_components_carry_their_records_together(self):
        caps = ModuleCapabilities(
            providers={
                Capability.GRADIENTS: ("model_backend",),
                Capability.ATTRIBUTION_GRAPH: ("analysis_backend",),
            },
            attribution_graph=AttributionGraphSupport(),
        )
        assert caps.supports(Capability.ATTRIBUTION_GRAPH)
        assert not caps.supports(Capability.INTERVENTION)


class TestFeatureTargetSupport:
    record = FeatureTargetSupport(
        value_sources=frozenset({"top_feature_scores", "constant"}),
        constrainable_layers=False,
        returns_activations=False,
    )

    def test_a_declared_configuration_is_accepted(self):
        assert self.record.refusal({"value_source": "top_feature_scores", "value": None}) is None

    def test_each_undeclared_configuration_is_refused_by_name(self):
        assert "value_source 'top_feature_activation_values' is not honoured" in self.record.refusal(
            {"value_source": "top_feature_activation_values"}
        )
        assert "needs an explicit intervention value" in self.record.refusal(
            {"value_source": "constant", "value": None}
        )
        assert "cannot restrict" in self.record.refusal(
            {"value_source": "constant", "value": 1.0, "constrained_layers": [0]}
        )
        assert "cannot return" in self.record.refusal(
            {"value_source": "constant", "value": 1.0, "return_activations": True}
        )
        assert "cannot hold norm denominators" in self.record.refusal(
            {"value_source": "constant", "value": 1.0, "freeze_norms": True}
        )

    def test_the_bundled_backend_refuses_through_its_record(self):
        from interpretune.adapters.circuit_tracer.backends import CircuitTracerAnalysisBackend

        backend = CircuitTracerAnalysisBackend()
        module = types.SimpleNamespace(circuit_tracer_cfg=types.SimpleNamespace())
        with pytest.raises(ValueError, match="feature intervention settings refused: value_source 'weird'"):
            backend.resolve_feature_intervention_settings(module, {"intervention_value_source": "weird"})
        settings = backend.resolve_feature_intervention_settings(
            module, {"intervention_value_source": "constant", "intervention_value": 2.0}
        )
        assert settings["value_source"] == "constant" and settings["value"] == 2.0

    def test_an_empty_record_is_refused(self):
        with pytest.raises(ValueError, match="at least one value source"):
            FeatureTargetSupport(value_sources=frozenset())


@pytest.mark.usefixtures("unpatched_gemma_eager_attention")
class TestAttributionGraphSupport:
    """The provenance check, with the planted foreign function as the positive control.

    The class-scoped fixture restores transformers' own gemma eager attention for the class duration: any
    TransformerLens bridge built earlier in the session replaces the process-wide module global and never puts it back,
    which the provenance guard then (correctly) refuses.
    """

    def _gemma3_model(self):
        import transformers.models.gemma3.modeling_gemma3 as modeling

        config = types.SimpleNamespace(_attn_implementation="eager")
        inner = modeling.Gemma3ForCausalLM.__new__(
            modeling.Gemma3ForCausalLM
        )  # no weights: only the type and config matter
        object.__setattr__(inner, "config", config)
        return types.SimpleNamespace(model=inner), modeling

    def test_the_modeling_modules_own_eager_attention_is_accepted(self):
        model, _ = self._gemma3_model()
        assert AttributionGraphSupport().refusal(model) is None

    def test_a_foreign_eager_attention_is_refused_by_name(self, monkeypatch):
        model, modeling = self._gemma3_model()

        def hooked_eager_attention_forward(*args, **kwargs):  # pragma: no cover - never called
            raise AssertionError

        monkeypatch.setattr(modeling, "eager_attention_forward", hooked_eager_attention_forward)
        why = AttributionGraphSupport().refusal(model)
        assert why is not None and "not the modeling module's own" in why
        assert "hooked_eager_attention_forward" in why and modeling.__name__ in why

    def test_a_non_eager_configuration_is_refused(self):
        model, _ = self._gemma3_model()
        model.model.config._attn_implementation = "sdpa"
        assert "configured as 'sdpa'" in AttributionGraphSupport().refusal(model)

    def test_an_attention_implementation_circuit_tracer_registered_is_accepted_by_provenance(self):
        """A circuit-tracer backend may switch the shared model to an eager variant it registers under its own name
        and build graphs through it; the guard accepts it because the registered function lives in circuit-tracer,
        not because of the name."""
        pytest.importorskip("circuit_tracer")
        from circuit_tracer.replacement_model import replacement_model_interp_engine as ct_engine

        ct_engine._register_frozen_attention()  # what circuit-tracer does when it builds that backend's model
        model, _ = self._gemma3_model()
        model.model.config._attn_implementation = ct_engine.FROZEN_ATTN_IMPL
        assert AttributionGraphSupport().refusal(model) is None
        # Control on the same stand-in: a name nobody registered is still refused.
        model.model.config._attn_implementation = "circuit_tracer_frozen_lookalike"
        assert "configured as 'circuit_tracer_frozen_lookalike'" in AttributionGraphSupport().refusal(model)

    def test_a_registered_implementation_from_another_package_is_still_refused(self, monkeypatch):
        """Provenance is the criterion: registering the same kind of variant from outside circuit-tracer earns
        nothing. Registered through the registry's global mapping under monkeypatch, so it is restored afterwards
        (the registry's own `pop` reaches only its local mapping)."""
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def foreign_eager(*args, **kwargs):  # pragma: no cover - never called
            raise AssertionError

        monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "it_test_foreign_eager", foreign_eager)
        assert "it_test_foreign_eager" in ALL_ATTENTION_FUNCTIONS.valid_keys()
        model, _ = self._gemma3_model()
        model.model.config._attn_implementation = "it_test_foreign_eager"
        assert "configured as 'it_test_foreign_eager'" in AttributionGraphSupport().refusal(model)

    def test_no_requirement_accepts_anything(self):
        assert AttributionGraphSupport(requires_own_eager_attention=False).refusal(object()) is None
