"""The analysis-backend surface decomposes like the model-backend one: a core plus gated groups with records.

A member of ``AnalysisBackendCapability`` names a ``Supports*`` group and answers whether it is implemented; the
group's configuration space is a typed record on the protocol. These tests pin the decomposition, the records'
refusals, and the provenance refusal that guards graph construction, each with a positive control.
"""

from __future__ import annotations

import types

import pytest
import torch

from interpretune.analysis.backends import (
    AnalysisBackend,
    AnalysisBackendCapability,
    AnalysisBackendCore,
    AttributionGraphSupport,
    FeatureInterventionSupport,
    ModelBackendCapability,
    ModuleCapabilities,
    SupportsAttributionGraph,
    SupportsFeatureInterventions,
)


class _CoreOnlyAnalysisBackend:
    """Implements the core and nothing gated."""

    @property
    def capabilities(self):
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
        assert not isinstance(backend, AnalysisBackend)

    def test_the_bundled_backend_satisfies_every_group(self):
        from interpretune.adapters.circuit_tracer.backends import CircuitTracerAnalysisBackend

        backend = CircuitTracerAnalysisBackend()
        for protocol in (AnalysisBackendCore, SupportsAttributionGraph, SupportsFeatureInterventions, AnalysisBackend):
            assert isinstance(backend, protocol), protocol.__name__
        assert backend.capabilities == {
            AnalysisBackendCapability.ATTRIBUTION_GRAPH,
            AnalysisBackendCapability.FEATURE_INTERVENTION,
        }


class TestRecordsTravelWithTheirSurface:
    def test_a_declared_analysis_surface_needs_its_record(self):
        with pytest.raises(ValueError, match="ATTRIBUTION_GRAPH is declared but no attribution_graph support record"):
            ModuleCapabilities(model=frozenset(), analysis=frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH}))
        with pytest.raises(ValueError, match="FEATURE_INTERVENTION is declared but no feature_intervention support"):
            ModuleCapabilities(model=frozenset(), analysis=frozenset({AnalysisBackendCapability.FEATURE_INTERVENTION}))

    def test_a_record_without_its_surface_is_refused(self):
        with pytest.raises(ValueError, match="feature_intervention support record is present but FEATURE_INTERVENTION"):
            ModuleCapabilities(
                model=frozenset(),
                analysis=frozenset(),
                feature_intervention=FeatureInterventionSupport(value_sources=frozenset({"constant"})),
            )

    def test_both_levels_carry_their_records_together(self):
        caps = ModuleCapabilities(
            model=frozenset({ModelBackendCapability.GRADIENTS}),
            analysis=frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH}),
            attribution_graph=AttributionGraphSupport(),
        )
        assert caps.supports(AnalysisBackendCapability.ATTRIBUTION_GRAPH)
        assert not caps.supports(AnalysisBackendCapability.FEATURE_INTERVENTION)


class TestFeatureInterventionSupport:
    record = FeatureInterventionSupport(
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
            FeatureInterventionSupport(value_sources=frozenset())


class TestAttributionGraphSupport:
    """The provenance check, with the planted foreign function as the positive control."""

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

    def test_no_requirement_accepts_anything(self):
        assert AttributionGraphSupport(requires_own_eager_attention=False).refusal(object()) is None
