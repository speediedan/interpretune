"""Tests for CircuitTracerAdapter backend integration.

This module tests the CircuitTracerAdapter with both TransformerLens and NNsight backends. Following the fixture usage
patterns established in test_adapters_transformer_lens.py.

GPU-dependent tests use RunIf(bf16_cuda=True) to require CUDA+bf16 support, consistent with the bidirectional mapping
tests in test_adapters_transformer_lens.py. Function-scoped fixtures ensure proper model cleanup between tests.
cleanup_cuda fixture ensures GPU memory is freed after each test.
"""

from __future__ import annotations

import os
import sys
import types

import httpx
import pytest

import nnsight

from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel

from interpretune.config import CircuitTracerConfig
from interpretune.adapters.circuit_tracer import ReplacementModelType
from tests.runif import RunIf


# =============================================================================
# Circuit Tracer Configuration Tests
# =============================================================================


@pytest.fixture
def third_party_backend(monkeypatch):
    """A registered backend other than nnsight, so the non-nnsight paths do not depend on the TL backend."""
    from interpretune.adapters.circuit_tracer.registry import CT_BACKEND_REGISTRY

    monkeypatch.setitem(CT_BACKEND_REGISTRY, "thirdparty", "ThirdPartyReplacementModel")
    return "thirdparty"


def _stub_transformer_lens(monkeypatch, mode: str) -> None:
    """``hooked`` has the class; ``missing`` lacks it; ``migration_error`` raises ImportError on access, as 4.0
    does."""
    stub = types.ModuleType("transformer_lens")
    if mode == "hooked":
        stub.HookedTransformer = type("HookedTransformer", (), {})
    elif mode == "migration_error":

        def __getattr__(name):
            raise ImportError(f"'{name}' was removed in TransformerLens 4.0.")

        stub.__getattr__ = __getattr__  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformer_lens", stub)


class TestCircuitTracerConfig:
    """Test CircuitTracerConfig backend configuration."""

    def test_default_backend_is_nnsight(self):
        """The TransformerLens backend needs HookedTransformer, which transformer-lens 4.0 removed."""
        cfg = CircuitTracerConfig()
        assert cfg.backend == "nnsight"

    @pytest.mark.parametrize("mode", ["missing", "migration_error"])
    def test_transformerlens_backend_is_refused_without_hooked_transformer(self, monkeypatch, mode):
        _stub_transformer_lens(monkeypatch, mode)
        with pytest.raises(ValueError, match=r"backend 'transformerlens' needs transformer_lens\.HookedTransformer"):
            CircuitTracerConfig(backend="transformerlens")

    def test_transformerlens_backend_is_accepted_where_hooked_transformer_exists(self, monkeypatch):
        """Positive control: the refusal keys on the class being absent, not on the backend name."""
        _stub_transformer_lens(monkeypatch, "hooked")
        assert CircuitTracerConfig(backend="transformerlens").backend == "transformerlens"

    def test_backend_validation_nnsight(self):
        """Verify 'nnsight' backend is valid."""
        cfg = CircuitTracerConfig(backend="nnsight")
        assert cfg.backend == "nnsight"

    def test_invalid_backend_raises_error(self):
        """Verify invalid backend value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            CircuitTracerConfig(backend="invalid_backend")

    def test_nnsight_remote_configuration(self):
        """Verify NNsight remote execution configuration."""
        cfg = CircuitTracerConfig(backend="nnsight", nnsight_remote=True, ndif_api_key="test_key_123")
        assert cfg.backend == "nnsight"
        assert cfg.nnsight_remote is True
        assert cfg.ndif_api_key == "test_key_123"

    def test_nnsight_local_configuration(self):
        """Verify NNsight local execution configuration."""
        cfg = CircuitTracerConfig(backend="nnsight", nnsight_remote=False)
        assert cfg.backend == "nnsight"
        assert cfg.nnsight_remote is False
        assert cfg.ndif_api_key is None

    def test_warning_on_nnsight_remote_with_a_non_nnsight_backend(self, third_party_backend):
        with pytest.warns(UserWarning, match="nnsight_remote=True but backend is not 'nnsight'"):
            CircuitTracerConfig(backend=third_party_backend, nnsight_remote=True)

    def test_warning_on_ndif_api_key_with_a_non_nnsight_backend(self, third_party_backend):
        with pytest.warns(UserWarning, match="ndif_api_key is set but backend is not 'nnsight'"):
            CircuitTracerConfig(backend=third_party_backend, ndif_api_key="test_key")

    def test_backend_serialization(self):
        """Verify backend configuration serializes correctly."""
        from dataclasses import asdict

        cfg = CircuitTracerConfig(backend="nnsight", nnsight_remote=True, ndif_api_key="test_key")
        serialized = asdict(cfg)
        assert serialized["backend"] == "nnsight"
        assert serialized["nnsight_remote"] is True
        assert serialized["ndif_api_key"] == "test_key"

    def test_all_backends_with_default_settings(self, third_party_backend):
        """Verify backends share the default attribution settings."""
        for backend in [third_party_backend, "nnsight"]:
            cfg = CircuitTracerConfig(backend=backend)
            assert cfg.max_n_logits == 10
            assert cfg.desired_logit_prob == 0.95
            assert cfg.batch_size == 256

    def test_backend_with_custom_attribution_params(self):
        """Verify backend configuration works with custom attribution parameters."""
        cfg = CircuitTracerConfig(
            backend="nnsight", max_n_logits=20, desired_logit_prob=0.99, batch_size=128, max_feature_nodes=4096
        )
        assert cfg.backend == "nnsight"
        assert cfg.max_n_logits == 20
        assert cfg.desired_logit_prob == 0.99
        assert cfg.batch_size == 128
        assert cfg.max_feature_nodes == 4096


# =============================================================================
# Circuit Tracer Backend Type Tests
# =============================================================================


class TestCircuitTracerBackendTypes:
    """Test backend type handling in CircuitTracerAdapter."""

    def test_backend_types_are_importable(self):
        """The adapter must import under transformer-lens 4.0, so it no longer imports the TL product."""
        assert NNSightReplacementModel is not None
        # The union type itself is checked at type-check time
        assert ReplacementModelType is not None


# =============================================================================
# Circuit Tracer TransformerLens Backend Adapter Integration Tests
# =============================================================================


# circuit-tracer's TransformerLens backend subclasses HookedTransformer, which TransformerLens 4.0 removes, so its
# registry compositions are disabled until circuit-tracer ports it onto TransformerBridge. Restore with them.
_CT_TL_DISABLED = pytest.mark.skip(
    reason="circuit-tracer TransformerLens backend disabled pending its TransformerLens 4.0 support; "
    "its compositions are commented out in tests/test_module_registry.yaml"
)


@_CT_TL_DISABLED
@pytest.mark.usefixtures("cleanup_cuda")
class TestCircuitTracerTLBackend:
    """TransformerLens backend integration tests for CircuitTracerAdapter.

    Verifies backend property access, replacement model loading, type matching, and config preservation. Uses the (core,
    transformer_lens, circuit_tracer) adapter combination. Requires CUDA with bf16 support (Gemma2 model).
    """

    @RunIf(bf16_cuda=True, requires_env="HF_GATED_PUBLIC_REPO_AUTH_KEY")
    def test_tl_backend_integration(self, get_it_session__ct_tl_gemma2__setup):
        """Verify TL backend initialization: property access, model loading, typing, and config preservation."""
        it_session = get_it_session__ct_tl_gemma2__setup.it_session

        # Backend property access
        assert it_session.module.circuit_tracer_cfg.backend == "transformerlens"

        # Replacement model loaded
        assert it_session.module.replacement_model is not None

        # Replacement model type matches backend
        assert type(it_session.module.replacement_model).__name__ == "TransformerLensReplacementModel"

        # Original HF config preserved after conversion
        assert hasattr(it_session.module.model, "config")
        assert it_session.module.model.config is not None


@_CT_TL_DISABLED
@pytest.mark.usefixtures("cleanup_cuda")
class TestCircuitTracerLightningTLBackendInitialization:
    """Test Lightning + TransformerLens backend initialization.

    Requires CUDA with bf16 support (Gemma2 model) and Lightning.
    """

    @RunIf(lightning=True, bf16_cuda=True, requires_env="HF_GATED_PUBLIC_REPO_AUTH_KEY")
    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__l_ct_tl_gemma2__setup"),
        ],
        ids=["lightning_transformerlens"],
    )
    def test_lightning_adapter_backend_support(self, session_fixture, request):
        """Verify Lightning adapter works with TransformerLens backend."""
        it_session = request.getfixturevalue(session_fixture).it_session

        # Verify Lightning module composition - check for Lightning-specific private attribute
        # Note: .trainer property raises RuntimeError when not attached, so check _trainer instead
        assert hasattr(it_session.module, "_trainer")
        # Backend should still be accessible
        assert it_session.module.circuit_tracer_cfg.backend == "transformerlens"


# =============================================================================
# NNsight Backend Tests
# =============================================================================


@pytest.mark.usefixtures("cleanup_cuda")
class TestCircuitTracerNNsightBackend:
    """NNsight backend integration tests for CircuitTracerAdapter.

    Verifies backend property access, replacement model loading and typing, and local mode configuration. Uses the
    (core, circuit_tracer) adapter combination with backend="nnsight". Requires CUDA with bf16 support (Gemma2 model).
    """

    @RunIf(bf16_cuda=True, requires_env="HF_GATED_PUBLIC_REPO_AUTH_KEY")
    def test_nnsight_backend_integration(self, get_it_session__ct_nnsight_gemma2__setup):
        """Verify NNsight backend initialization: property access, model loading, typing, and local mode config."""
        it_session = get_it_session__ct_nnsight_gemma2__setup.it_session

        # Backend property access
        assert it_session.module.circuit_tracer_cfg.backend == "nnsight"

        # Replacement model loaded and correctly typed
        assert it_session.module.replacement_model is not None
        assert isinstance(it_session.module.replacement_model, NNSightReplacementModel)

        # Local mode configuration
        assert it_session.module.circuit_tracer_cfg.nnsight_remote is False
        assert it_session.module.circuit_tracer_cfg.ndif_api_key is None

    def test_ndif_api_key_from_env(self, monkeypatch):
        """Verify NDIF_API_KEY is resolved from environment variable."""
        monkeypatch.setenv("NDIF_API_KEY", "test_env_key_123")
        cfg = CircuitTracerConfig(backend="nnsight", nnsight_remote=True)
        assert cfg.ndif_api_key == "test_env_key_123"


class TestNNsightBackendGemma3:
    """Unit tests for NNsight backend using Gemma3 model.

    Gemma3 is supported via the NNsight backend in circuit-tracer (no TL/HookedTransformer path exists for Gemma3). Uses
    gemma-scope transcoders from mwhanna/gemma-scope-2-1b-pt.
    """

    @RunIf(optional=True)
    def test_nnsight_backend_initialization_local(self, get_it_session__ct_nnsight_gemma3__setup):
        """Verify NNsight backend initializes correctly in local mode with Gemma3."""
        it_session = get_it_session__ct_nnsight_gemma3__setup.it_session

        assert it_session.module.circuit_tracer_cfg.backend == "nnsight"
        assert it_session.module.circuit_tracer_cfg.nnsight_remote is False
        assert isinstance(it_session.module.replacement_model, NNSightReplacementModel)

    @RunIf(optional=True)
    def test_gemma3_transcoder_set(self, get_it_session__ct_nnsight_gemma3__setup):
        """Verify Gemma3 uses full HF hub path for transcoder_set (not shortcut like Gemma2 'gemma')."""
        it_session = get_it_session__ct_nnsight_gemma3__setup.it_session

        transcoder_set = it_session.module.circuit_tracer_cfg.transcoder_set
        assert transcoder_set == "mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine"
        # Verify replacement model was loaded successfully with the transcoder set
        assert it_session.module.replacement_model is not None


# =============================================================================
# Remote Execution Tests (Commented out pending NNsight adapter support)
# =============================================================================


class TestNNsightRemoteExecution:
    """Tests for NNsight remote execution via NDIF.

    This test validates the API key flow and remote configuration using
    the NNsight replacement model with remote tracing capabilities.

    Requirements:
    - NDIF_API_KEY must be set (dotenv is loaded on tests module import)
    - Network access to NDIF service
    """

    @pytest.fixture
    def ensure_ndif_api_key(self):
        """Check if NDIF_API_KEY is set and configure nnsight.

        NNsight expects NDIF_API_KEY env var, but we use NDIF_API_KEY. This fixture also sets nnsight.CONFIG.API.APIKEY
        directly.
        """
        api_key = os.environ.get("NDIF_API_KEY")
        if not api_key:
            pytest.skip("NDIF_API_KEY not set, skipping remote test")
        # nnsight expects NDIF_API_KEY, but we configure directly
        nnsight.CONFIG.API.APIKEY = api_key
        return api_key

    @RunIf(optional=True)
    def test_remote_execution_api_key_flow(self, ensure_ndif_api_key, get_it_session__ct_nnsight_gemma2_remote__setup):
        """Verify remote execution: replacement_model loaded, activation tracing works.

        Tests that:
        1. CircuitTracerConfig reflects remote mode
        2. NNSightReplacementModel is properly loaded
        3. Internal activations can be traced/inspected via remote execution
        """
        from nnsight.intervention.backends.remote import RemoteException

        it_session = get_it_session__ct_nnsight_gemma2_remote__setup.it_session

        # CircuitTracerConfig should reflect remote
        assert it_session.module.circuit_tracer_cfg.nnsight_remote is True
        assert it_session.module.circuit_tracer_cfg.backend == "nnsight"

        # Verify replacement_model was properly loaded
        replacement_model = it_session.module.replacement_model
        assert replacement_model is not None
        assert isinstance(replacement_model, NNSightReplacementModel)

        # Verify we can trace internal activations remotely
        # NNSightReplacementModel inherits from nnsight.LanguageModel, supporting trace() context
        prompt = "The capital of France is"

        for attempt in range(2):
            try:
                with replacement_model.trace(prompt, remote=True):
                    # Access hidden states from an early layer
                    # Gemma2 uses model.layers[layer].output for hidden states
                    hidden = nnsight.save(replacement_model.model.layers[0].output[0])
                    # Access final logits
                    logits = nnsight.save(replacement_model.lm_head.output)
                break
            except httpx.TimeoutException as error:
                if attempt == 1:
                    pytest.skip(f"NDIF request timed out twice: {error}")
            except RemoteException as e:
                # Skip if model isn't available on NDIF (not scheduled/dedicated)
                if "not dedicated" in str(e) or "not scheduled" in str(e).lower():
                    pytest.skip(f"Model not available on NDIF: {e}")
                raise

        # Validate shapes - batch=1, seq_len varies with tokenization
        assert hidden.dim() == 3, f"Expected 3D hidden states, got shape {hidden.shape}"
        assert hidden.shape[0] == 1, "Expected batch size 1"

        # Logits should be [batch, seq, vocab]
        assert logits.dim() == 3, f"Expected 3D logits, got shape {logits.shape}"
        assert logits.shape[0] == 1, "Expected batch size 1"
