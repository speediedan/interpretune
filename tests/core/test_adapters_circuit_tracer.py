"""Tests for CircuitTracerAdapter backend integration.

This module tests the CircuitTracerAdapter with TransformerLens backend. NNsight backend tests are currently commented
out pending backend adapter support. Following the fixture usage patterns established in
test_adapters_transformer_lens.py.
"""

from __future__ import annotations

import pytest

from interpretune.config import CircuitTracerConfig
from interpretune.adapters.circuit_tracer import (
    ReplacementModelType,
    TransformerLensReplacementModel,
    NNSightReplacementModel,
)
from tests.runif import RunIf


# =============================================================================
# Circuit Tracer Configuration Tests
# =============================================================================


class TestCircuitTracerConfig:
    """Test CircuitTracerConfig backend configuration."""

    def test_default_backend_is_transformerlens(self):
        """Verify default backend is 'transformerlens'."""
        cfg = CircuitTracerConfig()
        assert cfg.backend == "transformerlens"

    def test_backend_validation_transformerlens(self):
        """Verify 'transformerlens' backend is valid."""
        cfg = CircuitTracerConfig(backend="transformerlens")
        assert cfg.backend == "transformerlens"

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
        cfg = CircuitTracerConfig(backend="nnsight", nnsight_remote=True, nnsight_api_key="test_key_123")
        assert cfg.backend == "nnsight"
        assert cfg.nnsight_remote is True
        assert cfg.nnsight_api_key == "test_key_123"

    def test_nnsight_local_configuration(self):
        """Verify NNsight local execution configuration."""
        cfg = CircuitTracerConfig(backend="nnsight", nnsight_remote=False)
        assert cfg.backend == "nnsight"
        assert cfg.nnsight_remote is False
        assert cfg.nnsight_api_key is None

    def test_warning_on_nnsight_remote_with_tl_backend(self):
        """Verify warning when nnsight_remote=True with transformerlens backend."""
        with pytest.warns(UserWarning, match="nnsight_remote=True but backend is not 'nnsight'"):
            CircuitTracerConfig(backend="transformerlens", nnsight_remote=True)

    def test_warning_on_nnsight_api_key_with_tl_backend(self):
        """Verify warning when nnsight_api_key provided with transformerlens backend."""
        with pytest.warns(UserWarning, match="nnsight_api_key is set but backend is not 'nnsight'"):
            CircuitTracerConfig(backend="transformerlens", nnsight_api_key="test_key")

    def test_backend_serialization(self):
        """Verify backend configuration serializes correctly."""
        from dataclasses import asdict

        cfg = CircuitTracerConfig(backend="nnsight", nnsight_remote=True, nnsight_api_key="test_key")
        serialized = asdict(cfg)
        assert serialized["backend"] == "nnsight"
        assert serialized["nnsight_remote"] is True
        assert serialized["nnsight_api_key"] == "test_key"

    def test_all_backends_with_default_settings(self):
        """Verify both backends work with default attribution settings."""
        for backend in ["transformerlens", "nnsight"]:
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

    def test_backend_type_alias_supports_both(self):
        """Verify ReplacementModelType union includes both backends."""
        # This is a type-level check - just verify the types are importable
        assert TransformerLensReplacementModel is not None
        assert NNSightReplacementModel is not None
        # The union type itself is checked at type-check time
        assert ReplacementModelType is not None


# =============================================================================
# Circuit Tracer TransformerLens Backend Adapter Integration Tests
# =============================================================================


class TestCircuitTracerTLBackendInitialization:
    """Test TransformerLens backend-specific initialization in CircuitTracerAdapter.

    These tests use the (core, transformer_lens, circuit_tracer) adapter combination.
    """

    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__ct_tl_gemma2__setup", marks=RunIf(standalone=True)),
        ],
        ids=["transformerlens"],
    )
    def test_backend_property_access(self, session_fixture, request):
        """Verify backend property returns correct value from config."""
        it_session = request.getfixturevalue(session_fixture).it_session

        # Access backend through circuit_tracer_cfg
        backend = it_session.module.circuit_tracer_cfg.backend
        assert backend == "transformerlens"

    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__ct_tl_gemma2__setup", marks=RunIf(standalone=True)),
        ],
        ids=["transformerlens"],
    )
    def test_replacement_model_loaded(self, request, session_fixture):
        """Verify replacement model is loaded for TransformerLens backend."""
        it_session = request.getfixturevalue(session_fixture).it_session

        # Replacement model should be loaded after setup
        assert it_session.module.replacement_model is not None

    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__ct_tl_gemma2__setup", marks=RunIf(standalone=True)),
        ],
        ids=["transformerlens"],
    )
    def test_replacement_model_type_matches_backend(self, session_fixture, request):
        """Verify replacement model type matches TransformerLens backend."""
        it_session = request.getfixturevalue(session_fixture).it_session

        backend = it_session.module.circuit_tracer_cfg.backend
        replacement_model = it_session.module.replacement_model

        assert backend == "transformerlens"
        assert isinstance(replacement_model, TransformerLensReplacementModel)

    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__ct_tl_gemma2__setup", marks=RunIf(standalone=True)),
        ],
        ids=["transformerlens"],
    )
    def test_model_config_preserved(self, session_fixture, request):
        """Verify original HF config is preserved after conversion."""
        it_session = request.getfixturevalue(session_fixture).it_session

        # The model should have a config attribute
        assert hasattr(it_session.module.model, "config")
        assert it_session.module.model.config is not None


class TestCircuitTracerTLBackendFunctionality:
    """Test TransformerLens backend-specific functionality beyond initialization."""

    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__ct_tl_gemma2__setup", marks=RunIf(standalone=True)),
        ],
        ids=["transformerlens"],
    )
    def test_transformerlens_backend_loads(self, request, session_fixture):
        """Verify TransformerLens backend loads successfully."""
        it_session = request.getfixturevalue(session_fixture).it_session

        assert it_session.module.circuit_tracer_cfg.backend == "transformerlens"
        assert isinstance(it_session.module.replacement_model, TransformerLensReplacementModel)

        # TODO: Enable Lightning tests after fixing Lightning adapter composition
        # class TestCircuitTracerLightningTLBackendInitialization:
        #     """Test Lightning + TransformerLens backend initialization."""
        #
        #     @RunIf(lightning=True)
        #     @pytest.mark.parametrize(
        #         "session_fixture",
        #         [
        #             pytest.param("get_it_session__l_ct_tl_gemma2__setup"),
        #         ],
        #         ids=["lightning_transformerlens"],
        #     )
        #     def test_lightning_adapter_backend_support(self, session_fixture, request):
        #         """Verify Lightning adapter works with TransformerLens backend."""
        #         it_session = request.getfixturevalue(session_fixture).it_session
        #
        #         # Should have Lightning-specific attributes
        #         assert hasattr(it_session.module, "trainer")
        #         # Backend should still be accessible
        #         assert it_session.module.circuit_tracer_cfg.backend == "transformerlens"

        # =============================================================================
        # NNsight Backend Tests (Commented out pending NNsight adapter support)
        # =============================================================================

        # TODO: Enable after NNsight adapter composition is implemented
        # The NNsight backend will use (core, circuit_tracer) adapter combination
        # without the transformer_lens adapter.

        # class TestCircuitTracerNNSightBackendInitialization:
        #     """Test NNsight backend-specific initialization in CircuitTracerAdapter.
        #
        #     These tests use the (core, circuit_tracer) adapter combination.
        #     """
        #
        #     @pytest.mark.parametrize(
        #         "session_fixture",
        #         [
        #             pytest.param("get_it_session__ct_nnsight_gemma2__setup"),
        #         ],
        #         ids=["nnsight"],
        #     )
        #     def test_nnsight_backend_property_access(self, session_fixture, request):
        #         """Verify backend property returns correct value from config."""
        #         it_session = request.getfixturevalue(session_fixture).it_session
        #
        #         backend = it_session.module.circuit_tracer_cfg.backend
        #         assert backend == "nnsight"
        #
        #     @pytest.mark.parametrize(
        #         "session_fixture",
        #         [
        #             pytest.param("get_it_session__ct_nnsight_gemma2__setup"),
        #         ],
        #         ids=["nnsight"],
        #     )
        #     def test_nnsight_replacement_model_loaded(self, request, session_fixture):
        #         """Verify replacement model is loaded for NNsight backend."""
        #         it_session = request.getfixturevalue(session_fixture).it_session
        #
        #         # Replacement model should be loaded after setup
        #         assert it_session.module.replacement_model is not None
        #         assert isinstance(it_session.module.replacement_model, NNSightReplacementModel)

        # class TestNNsightBackendGemma2:
        #     """Unit tests for NNsight backend using Gemma2 model."""
        #
        #     def test_nnsight_backend_initialization_local(self, get_it_session__ct_nnsight_gemma2__setup):
        #         """Verify NNsight backend initializes correctly in local mode with Gemma2."""
        #         it_session = get_it_session__ct_nnsight_gemma2__setup.it_session
        #
        #         assert it_session.module.circuit_tracer_cfg.backend == "nnsight"
        #         assert it_session.module.circuit_tracer_cfg.nnsight_remote is False
        #         assert isinstance(it_session.module.replacement_model, NNSightReplacementModel)
        #
        #     def test_nnsight_api_key_from_env(self, monkeypatch):
        #         """Verify NNSIGHT_API_KEY is resolved from environment variable."""
        #         # Set environment variable temporarily
        #         monkeypatch.setenv("NNSIGHT_API_KEY", "test_env_key_123")
        #
        #         # Create config with remote=True but no explicit API key
        #         cfg = CircuitTracerConfig(
        #             backend="nnsight",
        #             nnsight_remote=True
        #         )
        #
        #         # __post_init__ should have resolved API key from env
        #         assert cfg.nnsight_api_key == "test_env_key_123"
        #
        #     def test_nnsight_local_no_api_key_required(self, get_it_session__ct_nnsight_gemma2__setup):
        #         """Verify local mode doesn't require API key."""
        #         it_session = get_it_session__ct_nnsight_gemma2__setup.it_session
        #
        #         # Local mode should work without API key
        #         assert it_session.module.circuit_tracer_cfg.nnsight_remote is False
        #         assert it_session.module.circuit_tracer_cfg.nnsight_api_key is None
        #         assert it_session.module.replacement_model is not None


# =============================================================================
# Remote Execution Tests (Commented out pending NNsight adapter support)
# =============================================================================

# TODO: enable after local tests pass
# class TestNNsightRemoteExecution:
#     """Minimal test for NNsight remote execution.
#
#     This test validates the API key flow and remote configuration.
#     Uses smallest model and minimal operations to conserve API usage.
#     """
#
#     @RunIf(standalone=True)
#     @pytest.mark.skipif(
#         "NNSIGHT_API_KEY" not in __import__("os").environ,
#         reason="NNSIGHT_API_KEY environment variable not set"
#     )
#     def test_remote_execution_api_key_flow(self, get_it_session__ct_nnsight_gemma2_remote__setup, monkeypatch):
#         """Verify remote execution API key flows correctly from env to config to adapter.
#
#         This is a single minimal test to validate the full API key resolution chain:
#         1. NNSIGHT_API_KEY environment variable
#         2. CircuitTracerConfig.__post_init__ resolution
#         3. Adapter layer passing to NNsight backend
#
#         Note: This test will make a minimal remote API call to validate the flow.
#         """
#         # Ensure environment variable is set (should be from fixture, but verify)
#         import os
#         api_key = os.environ.get("NNSIGHT_API_KEY")
#         assert api_key is not None, "NNSIGHT_API_KEY must be set in environment"
#
#         it_session = get_it_session__ct_nnsight_gemma2_remote__setup.it_session
#
#         # Verify configuration
#         assert it_session.module.circuit_tracer_cfg.backend == "nnsight"
#         assert it_session.module.circuit_tracer_cfg.nnsight_remote is True
#
#         # API key should be resolved from environment
#         # (CircuitTracerConfig.__post_init__ handles this)
#         assert it_session.module.circuit_tracer_cfg.nnsight_api_key == api_key
#
#         # Verify model loaded (this validates API key was accepted)
#         assert it_session.module.replacement_model is not None
#         assert isinstance(it_session.module.replacement_model, NNSightReplacementModel)
#
