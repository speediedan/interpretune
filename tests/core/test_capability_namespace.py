"""One capability namespace (#951): components declare capabilities, records and requirements, and the module's
aggregate refuses an unmet requirement and an ambiguous target by name.

Each refusal is paired with the composition that passes, so a refusal cannot pass because the aggregation refused
everything.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.backends import (
    Capability,
    FeatureTargetSupport,
    InterventionSupport,
    LatentModelSupport,
    PositionScope,
    get_module_capabilities,
)

_FEATURE_TUPLES = InterventionSupport(
    ops=frozenset({"replace"}),
    position_scopes=frozenset({PositionScope.POSITIONS}),
    targets={"features": FeatureTargetSupport(value_sources=frozenset({"constant"}))},
)


class _ModelBackend:
    def __init__(
        self, capabilities=frozenset({Capability.GRADIENTS, Capability.LATENT_MODELS, Capability.INTERVENTION})
    ):
        self.capabilities = frozenset(capabilities)
        self.requires = frozenset()
        self.latent_model_support = LatentModelSupport()
        self.intervention_support = InterventionSupport.every()


class _AnalysisBackend:
    def __init__(self, requires=frozenset({Capability.GRADIENTS, Capability.LATENT_MODELS})):
        self.capabilities = frozenset({Capability.INTERVENTION})
        self.requires = frozenset(requires)
        self.intervention_support = _FEATURE_TUPLES


class _Module:
    def __init__(self, model_backend=None, analysis_backend=None):
        if model_backend is not None:
            self._model_backend = model_backend
        if analysis_backend is not None:
            self._analysis_backend = analysis_backend


class TestRequirements:
    def test_a_requirement_another_component_meets_composes(self):
        caps = get_module_capabilities(_Module(_ModelBackend(), _AnalysisBackend()))
        assert caps.requires == {"analysis_backend": frozenset({Capability.GRADIENTS, Capability.LATENT_MODELS})}
        assert caps.providers[Capability.INTERVENTION] == ("model_backend", "analysis_backend")

    def test_an_unmet_requirement_is_refused_naming_the_component_and_what_is_missing(self):
        partial = _ModelBackend(capabilities={Capability.GRADIENTS, Capability.INTERVENTION})
        with pytest.raises(
            ValueError, match=r"the analysis_backend \(_AnalysisBackend\) .* requires \['LATENT_MODELS'\]"
        ):
            get_module_capabilities(_Module(partial, _AnalysisBackend()))
        with pytest.raises(ValueError, match=r"requires \['GRADIENTS', 'LATENT_MODELS'\] from another component"):
            get_module_capabilities(_Module(analysis_backend=_AnalysisBackend()))

    def test_a_component_cannot_meet_its_own_requirement(self):
        """``requires`` names what ANOTHER component provides: declaring a capability and requiring it is not a
        composition."""

        class _SelfReliant(_ModelBackend):
            def __init__(self):
                super().__init__()
                self.requires = frozenset({Capability.GRADIENTS})

        with pytest.raises(ValueError, match=r"the model_backend .* requires \['GRADIENTS'\] from another component"):
            get_module_capabilities(_Module(_SelfReliant()))

    def test_the_bundled_circuit_tracer_backend_requires_what_graphs_are_built_on(self):
        from interpretune.adapters.circuit_tracer.backends import CircuitTracerAnalysisBackend

        assert CircuitTracerAnalysisBackend().requires == {Capability.GRADIENTS, Capability.LATENT_MODELS}


class TestTargetsWithTwoProviders:
    def test_a_target_with_two_providers_is_kept_and_an_unqualified_request_refused(self):
        caps = get_module_capabilities(_Module(_ModelBackend(), _AnalysisBackend()))
        assert caps.target_providers("features") == ("model_backend", "analysis_backend")
        with pytest.raises(ValueError, match=r"'features' .* provided by \['model_backend', 'analysis_backend'\]"):
            caps.intervention_for("features")

    def test_a_qualified_request_resolves_to_its_provider(self):
        caps = get_module_capabilities(_Module(_ModelBackend(), _AnalysisBackend()))
        assert caps.intervention_for("features", provider="analysis_backend") is _FEATURE_TUPLES
        assert caps.target_detail("features", provider="analysis_backend").value_sources == {"constant"}
        assert caps.target_detail("features", provider="model_backend") is None

    def test_a_target_with_one_provider_needs_no_qualification(self):
        caps = get_module_capabilities(_Module(_ModelBackend(), _AnalysisBackend()))
        assert caps.intervention_for("activation") is caps.intervention["model_backend"]

    def test_a_provider_that_does_not_reach_the_target_is_refused(self):
        caps = get_module_capabilities(_Module(_ModelBackend(), _AnalysisBackend()))
        with pytest.raises(ValueError, match="'activation' intervention target is not provided by 'analysis_backend'"):
            caps.intervention_for("activation", provider="analysis_backend")
        with pytest.raises(ValueError, match="'heads' is not an intervention target"):
            caps.intervention_for("heads")

    def test_a_target_nothing_provides_is_refused(self):
        caps = get_module_capabilities(_Module(_ModelBackend(capabilities={Capability.GRADIENTS})))
        with pytest.raises(ValueError, match="no component of this module provides the 'activation' intervention"):
            caps.intervention_for("activation")

    def test_a_non_intervention_surface_with_two_providers_is_refused(self):
        class _AlsoLatent(_AnalysisBackend):
            def __init__(self):
                super().__init__(requires=frozenset())
                self.capabilities = frozenset({Capability.LATENT_MODELS})
                self.latent_model_support = LatentModelSupport()

        with pytest.raises(ValueError, match=r"LATENT_MODELS is provided by \['model_backend', 'analysis_backend'\]"):
            get_module_capabilities(_Module(_ModelBackend(), _AlsoLatent()))


class TestTheInterventionRecord:
    def test_targets_default_to_every_target_like_venngrams_support(self):
        record = InterventionSupport(ops={"add"}, position_scopes={"last_token"})
        assert set(record.targets) == {"activation", "units", "features"}

    def test_an_unknown_target_value_source_or_flow_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"unknown targets \['heads'\]"):
            InterventionSupport(ops={"add"}, position_scopes={"last_token"}, targets={"heads": None})
        with pytest.raises(ValueError, match=r"unknown value sources \['column'\]"):
            InterventionSupport(ops={"add"}, position_scopes={"last_token"}, values={"column"})
        with pytest.raises(ValueError, match=r"unknown flow \(`of`\)s \['hessian'\]"):
            InterventionSupport(ops={"add"}, position_scopes={"last_token"}, of={"hessian"})

    def test_a_detail_belongs_to_the_target_it_describes(self):
        detail = FeatureTargetSupport(value_sources=frozenset({"constant"}))
        with pytest.raises(ValueError, match="the 'activation' target carries a FeatureTargetSupport detail"):
            InterventionSupport(ops={"replace"}, position_scopes={"positions"}, targets={"activation": detail})

    def test_hook_backends_declare_hook_scopes_only(self):
        """Explicit positions are named by a feature tuple, never by a hook spec, so ``every()`` does not claim
        them."""
        assert InterventionSupport.every().position_scopes == {PositionScope.LAST_TOKEN, PositionScope.ALL_POSITIONS}
