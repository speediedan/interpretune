"""The decomposed ``ModelBackend`` surface: required core + capability-gated method groups.

A partial backend (the hub-adapter case that motivated the split) satisfies ``ModelBackendCore``
plus whichever ``Supports*`` groups it truthfully claims; ops gate optional methods through
``require_backend_capability`` rather than discovering a missing method as an ``AttributeError``
mid-execution.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends.capabilities import BackendCapability
from interpretune.analysis.backends.protocols import (
    ModelBackend,
    ModelBackendCore,
    SupportsGradients,
    SupportsIntervention,
    SupportsLatentModels,
)
from interpretune.analysis.optools import require_backend_capability


class _CaptureOnlyBackend:
    """A truthful partial backend: core surface only, no optional groups."""

    @property
    def capabilities(self) -> frozenset[BackendCapability]:
        return frozenset()

    def supports(self, capability: BackendCapability) -> bool:
        return capability in self.capabilities

    def fwd(self, model, batch):
        return torch.zeros(1)

    def fwd_w_cache(self, model, batch, names_filter):
        return torch.zeros(1), {}

    def wrap_activation_cache(self, cache_dict, model):
        return cache_dict


class TestDecomposition:
    def test_partial_backend_satisfies_core_but_not_full_protocol(self):
        backend = _CaptureOnlyBackend()
        assert isinstance(backend, ModelBackendCore)
        assert not isinstance(backend, ModelBackend)

    def test_bundled_backends_satisfy_every_group(self):
        from interpretune.adapters.nnsight.backends import NNsightModelBackend
        from interpretune.adapters.transformer_lens.backends import TLModelBackend

        for backend in (TLModelBackend(), NNsightModelBackend(hook_resolver=None)):
            assert isinstance(backend, ModelBackend)
            for proto in (ModelBackendCore, SupportsLatentModels, SupportsGradients, SupportsIntervention):
                assert isinstance(backend, proto)
            for cap in (
                BackendCapability.LATENT_MODELS,
                BackendCapability.GRADIENTS,
                BackendCapability.INTERVENTION,
            ):
                assert backend.supports(cap)

    def test_batched_hooks_is_an_efficiency_claim_not_method_presence(self):
        """TL implements fwd_w_hooks_batched (sequential loop) and says so on the latent-model record."""
        from interpretune.adapters.transformer_lens.backends import TLModelBackend
        from interpretune.analysis.backends import LatentModelSupport

        backend = TLModelBackend()
        assert callable(backend.fwd_w_hooks_batched)
        assert backend.latent_model_support == LatentModelSupport(batched_hooks=False)


class TestCapabilityGate:
    def test_gate_passes_silently_when_claimed(self):
        from interpretune.adapters.transformer_lens.backends import TLModelBackend

        require_backend_capability(TLModelBackend(), BackendCapability.GRADIENTS, "gradient_attribution")

    def test_gate_names_the_op_the_capability_and_what_the_backend_claims(self):
        backend = _CaptureOnlyBackend()
        with pytest.raises(ValueError, match=r"some_op requires .*LATENT_MODELS.*_CaptureOnlyBackend"):
            require_backend_capability(backend, BackendCapability.LATENT_MODELS, "some_op")

    def test_gate_is_value_based_across_module_identity_splits(self):
        """A reloaded capabilities module yields value-equal but identity-distinct enum members.

        This repo's suite can load a module twice (documented importlib double-loading class), so the gate must pass for
        a backend that REPORTS the capability even when the two sides hold members from different load events.
        """
        import importlib.util

        from interpretune.analysis.backends import capabilities as cap_mod

        import sys

        spec = importlib.util.spec_from_file_location("_cap_copy_for_test", cap_mod.__file__)
        assert spec is not None and spec.loader is not None
        cap_copy = importlib.util.module_from_spec(spec)
        # dataclass field resolution looks the module up by name during exec
        sys.modules["_cap_copy_for_test"] = cap_copy
        try:
            spec.loader.exec_module(cap_copy)
        finally:
            sys.modules.pop("_cap_copy_for_test", None)
        other_member = cap_copy.BackendCapability.GRADIENTS
        assert other_member is not BackendCapability.GRADIENTS  # the split is real

        class _SplitBackend(_CaptureOnlyBackend):
            @property
            def capabilities(self):
                return frozenset({other_member})

        require_backend_capability(_SplitBackend(), BackendCapability.GRADIENTS, "some_op")
