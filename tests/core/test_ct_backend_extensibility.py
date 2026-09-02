"""The circuit-tracer seams must be enterable by a backend core does not name.

Before N2 these were identity branches (`if backend == "nnsight": ... else: TL`), which a third-party
adapter could not enter and which OVERWROTE any model backend already attached in the MRO. Both are
the shapes the per-adapter package work (#401) exists to remove: core owns protocols, capability
vocabulary and registries, and knows nothing about any particular hub adapter.
"""

from __future__ import annotations

import pytest

from interpretune.adapters.circuit_tracer.adapter import CT_BACKEND_REGISTRY, CT_MODEL_BACKEND_FACTORIES
from interpretune.config import CircuitTracerConfig


@pytest.fixture
def third_party_backend():
    """Register a backend core has never heard of, and clean up."""
    name = "a_third_party_backend"
    CT_BACKEND_REGISTRY[name] = "SomeReplacementModel"
    try:
        yield name
    finally:
        CT_BACKEND_REGISTRY.pop(name, None)
        CT_MODEL_BACKEND_FACTORIES.pop(name, None)


class TestConfigValidationIsRegistryDriven:
    def test_an_unregistered_backend_is_refused(self):
        with pytest.raises(ValueError, match="Registered backends"):
            CircuitTracerConfig(backend="not_a_registered_backend")

    def test_registration_alone_makes_a_backend_configurable(self, third_party_backend):
        """The whole point: no edit to core, no branch, just a registry entry."""
        assert CircuitTracerConfig(backend=third_party_backend).backend == third_party_backend

    def test_the_bundled_backends_still_validate(self):
        for name in ("transformerlens", "nnsight"):
            assert CircuitTracerConfig(backend=name).backend == name

    def test_it_is_refused_again_once_unregistered(self, third_party_backend):
        """Negative control: the acceptance above is caused by the registration, not by a widened check."""
        CT_BACKEND_REGISTRY.pop(third_party_backend)
        with pytest.raises(ValueError, match="Registered backends"):
            CircuitTracerConfig(backend=third_party_backend)


class TestAttachDoesNotOverride:
    """A model backend already attached in the MRO must survive circuit-tracer's init.

    This is the mechanism that lets a third adapter compose: it attaches its own backend, and
    circuit-tracer -- which arrives later in the MRO -- must leave it alone. Without this the
    composition order silently decides which backend wins.
    """

    @staticmethod
    def _module_with(backend):
        class _M:
            pass

        m = _M()
        if backend is not None:
            m._model_backend = backend
        return m

    def test_an_existing_backend_is_detected(self):
        from interpretune.analysis.backends.capabilities import get_model_backend

        sentinel = object()
        assert get_model_backend(self._module_with(sentinel)) is sentinel

    def test_absence_is_reported_as_none_not_an_error(self):
        """The attach path branches on this, so it must answer rather than raise on a bare module."""
        from interpretune.analysis.backends.capabilities import get_model_backend

        assert get_model_backend(self._module_with(None)) is None

    def test_a_registered_backend_may_supply_no_factory(self, third_party_backend):
        """Legitimate case: the backend's own adapter attaches it, so circuit-tracer must not require one."""
        assert third_party_backend in CT_BACKEND_REGISTRY
        assert CT_MODEL_BACKEND_FACTORIES.get(third_party_backend) is None


class TestCoreKnowsNothingOfAnyHubAdapter:
    """Zero references to a hub-delivered adapter anywhere in core.

    The rails (#125) exist so a component registers ITSELF. Anything core did on its behalf would recreate the
    privileged position #401 removed, and it would do so silently -- a name in a branch reads as support rather than as
    a special case.
    """

    def test_no_core_module_mentions_interp_engine(self):
        from pathlib import Path

        core = Path(__file__).parent.parent.parent / "src" / "interpretune"
        offenders = [
            str(p.relative_to(core))
            for p in core.rglob("*.py")
            if "interp_engine" in p.read_text() or "interp-engine" in p.read_text()
        ]
        assert not offenders, f"core references a hub adapter: {offenders}"

    def test_the_bundled_registry_names_only_bundled_backends(self):
        """A hub backend must not be pre-registered here; it registers itself from its entrypoint."""
        assert set(CT_BACKEND_REGISTRY) == {"transformerlens", "nnsight"}
