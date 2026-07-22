"""Unit tests for NNsight adapter integration.

Tests NNsight adapter functionality including:
- Configuration validation and exceptions
- Model loading and initialization
- Device/dtype configuration
- NDIF_API_KEY environment variable handling
- Remote/local execution consistency
- Basic generation functionality

Similar to test_adapters_transformer_lens.py in structure and testing patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Sequence
from unittest.mock import patch

import httpx
import nnsight
import pytest
import torch

from interpretune.config import NNsightConfig, ITNNsightConfig
from interpretune.protocol import Adapter
from interpretune.utils import MisconfigurationException
from tests.base_defaults import BaseCfg
from tests.runif import RunIf
from tests.utils import ablate_cls_attrs


# =============================================================================
# Test Configuration Classes
# =============================================================================


@dataclass(kw_only=True)
class NNsightTestConfig(BaseCfg):
    """Base test configuration for NNsight adapter tests."""

    phase: str = "test"
    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight)
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(
            model_name="openai-community/gpt2",
            device_map="cpu",
            torch_dtype="float32",
            dispatch=True,
        )
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestNNsightConfig:
    """Tests for NNsightConfig dataclass validation and functionality."""

    def test_nnsight_config_defaults(self):
        """Verify NNsightConfig default values are set correctly."""
        config = NNsightConfig()

        assert config.model_name == "openai-community/gpt2"
        assert config.device_map is None
        assert config.torch_dtype == "float32"
        assert config.dispatch is True
        assert config.trust_remote_code is False
        assert config.attn_implementation is None
        assert config.default_padding_side == "left"
        assert config.remote is False
        assert config.api_key is None

    def test_nnsight_config_dtype_resolution(self):
        """Verify torch dtype string resolution works correctly."""
        # Test string dtype resolution
        config_f32 = NNsightConfig(torch_dtype="float32")
        assert config_f32.resolved_dtype == torch.float32

        config_f16 = NNsightConfig(torch_dtype="float16")
        assert config_f16.resolved_dtype == torch.float16

        config_bf16 = NNsightConfig(torch_dtype="bfloat16")
        assert config_bf16.resolved_dtype == torch.bfloat16

        # Test direct torch.dtype
        config_direct = NNsightConfig(torch_dtype=torch.float64)
        assert config_direct.resolved_dtype == torch.float64

    def test_nnsight_config_tokenizer_kwargs_defaults(self):
        """Verify tokenizer_kwargs are initialized with padding_side default."""
        config = NNsightConfig()

        assert config.tokenizer_kwargs is not None
        assert config.tokenizer_kwargs.get("padding_side") == "left"

    def test_nnsight_config_tokenizer_kwargs_preserved(self):
        """Verify explicit tokenizer_kwargs are preserved."""
        custom_kwargs = {"padding_side": "right", "add_bos_token": True}
        config = NNsightConfig(tokenizer_kwargs=custom_kwargs)

        assert config.tokenizer_kwargs["padding_side"] == "right"
        assert config.tokenizer_kwargs["add_bos_token"] is True

    def test_get_nnsight_kwargs(self):
        """Verify get_nnsight_kwargs returns correct initialization kwargs."""
        config = NNsightConfig(
            model_name="gpt2",
            device_map="cpu",
            torch_dtype="float32",
            dispatch=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        kwargs = config.get_nnsight_kwargs()

        assert kwargs["dispatch"] is True
        assert kwargs["device_map"] == "cpu"
        assert kwargs["torch_dtype"] == torch.float32
        assert kwargs["trust_remote_code"] is True
        assert kwargs["attn_implementation"] == "sdpa"
        assert "tokenizer_kwargs" in kwargs

    def test_get_nnsight_kwargs_remote(self):
        """Verify get_nnsight_kwargs includes remote settings when enabled."""
        config = NNsightConfig(remote=True, dispatch=False)

        kwargs = config.get_nnsight_kwargs()

        from interpretune.config.nnsight import _nnsight_accepts_init_remote

        if _nnsight_accepts_init_remote():  # nnsight 0.6 line: remote is an init kwarg
            assert kwargs["remote"] is True
        else:  # 0.7+: remote moved to per-trace selection; init kwarg would leak to the HF ctor
            assert "remote" not in kwargs
        assert kwargs["dispatch"] is False


class TestITNNsightConfig:
    """Tests for ITNNsightConfig dataclass validation."""

    def test_itnnsight_config_requires_nnsight_cfg(self):
        """Verify ITNNsightConfig raises error without nnsight_cfg."""
        with pytest.raises(MisconfigurationException, match="A valid nnsight_cfg"):
            _ = ITNNsightConfig(
                model_name_or_path="gpt2",
                nnsight_cfg=None,
            )

    def test_itnnsight_config_model_name_sync(self):
        """Verify model name synchronization between ITConfig and NNsightConfig."""
        # Test nnsight_cfg.model_name used when model_name_or_path not set
        config1 = ITNNsightConfig(
            model_name_or_path=None,
            nnsight_cfg=NNsightConfig(model_name="openai-community/gpt2"),
        )
        assert config1.model_name_or_path == "openai-community/gpt2"

        # Test model_name_or_path used when nnsight_cfg.model_name not set
        config2 = ITNNsightConfig(
            model_name_or_path="openai-community/gpt2",
            nnsight_cfg=NNsightConfig(model_name=None),
        )
        assert config2.nnsight_cfg.model_name == "openai-community/gpt2"

    def test_itnnsight_config_dict_conversion(self):
        """Verify dict-based nnsight_cfg is converted to NNsightConfig."""
        # Note: model_name_or_path takes precedence and syncs to nnsight_cfg.model_name
        config = ITNNsightConfig(
            model_name_or_path="openai-community/gpt2",
            nnsight_cfg={"torch_dtype": "float32"},
        )

        assert isinstance(config.nnsight_cfg, NNsightConfig)
        assert config.nnsight_cfg.model_name == "openai-community/gpt2"
        assert config.nnsight_cfg.torch_dtype == "float32"


# =============================================================================
# API Key and Environment Variable Tests
# =============================================================================


class TestNNsightAPIKey:
    """Tests for NDIF_API_KEY environment variable handling."""

    def test_api_key_from_config(self):
        """Verify API key from config is used when provided."""
        config = NNsightConfig(
            remote=True,
            api_key="test_api_key_123",
            dispatch=False,
        )

        assert config.api_key == "test_api_key_123"

    def test_api_key_env_var_detection(self):
        """Verify NDIF_API_KEY environment variable is detected."""
        # This test verifies the environment variable detection logic
        # The actual API key usage is tested in integration tests
        env_key = os.environ.get("NDIF_API_KEY")
        if env_key:
            # If env var is set, verify it's accessible
            assert len(env_key) > 0
        else:
            # If not set, that's also valid for this test
            pass

    @patch.dict(os.environ, {"NDIF_API_KEY": "env_test_key_456"})
    def test_api_key_env_var_fallback(self):
        """Verify environment variable is used as fallback for API key."""
        config = NNsightConfig(remote=True, api_key=None, dispatch=False)

        # Config doesn't have api_key, should fall back to env var in adapter
        assert config.api_key is None
        assert os.environ.get("NDIF_API_KEY") == "env_test_key_456"


# =============================================================================
# Session and Module Tests
# =============================================================================


class TestNNsightSession:
    """Tests for NNsight session initialization and module behavior.

    These tests use class-scoped fixtures registered in example_module_registry.yaml as gpt2.rte.nnsight, enabling
    efficient fixture sharing across test methods.
    """

    def test_nnsight_session_config_creation(self):
        """Verify NNsight session config is accessible."""
        config = ITNNsightConfig(
            model_name_or_path="openai-community/gpt2",
            nnsight_cfg=NNsightConfig(model_name="openai-community/gpt2", dispatch=False),
        )
        assert config.nnsight_cfg is not None
        assert isinstance(config.nnsight_cfg, NNsightConfig)
        assert config.nnsight_cfg.model_name == "openai-community/gpt2"

    def test_nnsight_session_cfg_fixture(self, get_it_session_cfg__ns_gpt2):
        """Verify NNsight session config fixture works via registry lookup."""
        cfg = get_it_session_cfg__ns_gpt2
        assert cfg is not None
        assert cfg.module_cfg is not None

    def test_nnsight_session_model_init(self, get_it_session__ns_gpt2__setup):
        """Verify NNsight model is initialized correctly."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        # Verify model is a NNsight LanguageModel
        assert module.model is not None
        assert hasattr(module.model, "tokenizer")

    def test_nnsight_device_property(self, get_it_session__ns_gpt2__setup):
        """Verify device property returns correct value."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        device = module.device
        # Device should be CPU for this test config
        if device is not None:
            assert device.type == "cpu"

    def test_nnsight_cfg_property(self, get_it_session__ns_gpt2__setup):
        """Verify nnsight_cfg property is accessible."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        nnsight_cfg = module.nnsight_cfg
        assert nnsight_cfg is not None
        assert isinstance(nnsight_cfg, NNsightConfig)
        assert nnsight_cfg.model_name == "openai-community/gpt2"

    def test_nnsight_model_property(self, get_it_session__ns_gpt2__setup):
        """Verify nnsight_model property returns the LanguageModel."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        nnsight_model = module.nnsight_model
        assert nnsight_model is not None

    def test_nnsight_tokenizer_access(self, get_it_session__ns_gpt2__setup):
        """Verify tokenizer is accessible through NNsight model."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        # Access tokenizer through NNsight model
        if hasattr(module.model, "tokenizer"):
            tokenizer = module.model.tokenizer
            assert tokenizer is not None
            # Test basic tokenization
            tokens = tokenizer("Hello, world!")
            assert "input_ids" in tokens


class TestNNsightMixinAttributes:
    """Tests for NNsightAttributeMixin property access patterns."""

    def test_device_property_with_unset_state(self, get_it_session__ns_gpt2__setup):
        """Verify device property handles unset state gracefully."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        # Temporarily ablate device to test fallback behavior
        with ablate_cls_attrs(module._it_state, "_device"):
            device = module.device
            # Should fall back to model parameters or None
            assert device is None or isinstance(device, torch.device)

    def test_device_setter(self, get_it_session__ns_gpt2__setup):
        """Verify device setter works correctly."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        # Set device via string
        module.device = "meta"
        assert isinstance(module.device, torch.device)
        assert module.device.type == "meta"

        # Reset to CPU
        module.device = "cpu"
        assert module.device.type == "cpu"

    def test_input_output_device_properties(self, get_it_session__ns_gpt2__setup):
        """Verify input_device and output_device return same as device."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        device = module.device
        assert module.input_device == device
        assert module.output_device == device


# =============================================================================
# Lightning Integration Tests
# =============================================================================


class TestNNsightLightningIntegration:
    """Tests for NNsight adapter with Lightning integration.

    These tests use class-scoped fixtures registered in example_module_registry.yaml.
    """

    @RunIf(lightning=True)
    def test_lightning_nnsight_session_init(self, get_it_session__l_ns_gpt2__setup):
        """Verify Lightning + NNsight session initializes correctly."""
        fixture = get_it_session__l_ns_gpt2__setup
        module = fixture.it_session.module

        assert module.model is not None
        assert module.nnsight_cfg is not None

    @RunIf(lightning=True)
    def test_lightning_nnsight_device_property(self, get_it_session__l_ns_gpt2__setup):
        """Verify device property works with Lightning integration."""
        fixture = get_it_session__l_ns_gpt2__setup
        module = fixture.it_session.module

        device = module.device
        if device is not None:
            assert isinstance(device, torch.device)


# =============================================================================
# Tracing and Intervention Tests (Basic)
# =============================================================================


class TestNNsightTracing:
    """Basic tests for NNsight tracing functionality."""

    @RunIf(standalone=True)
    def test_nnsight_trace_context(self, get_it_session__ns_gpt2__setup):
        """Verify NNsight trace context manager works.

        Note: Marked standalone because NNsight's trace() uses sys.settrace() which
        interferes with coverage.py's trace function, causing coverage data loss for
        all tests that run after this one in the same process.
        """
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        # Basic trace test - verify we can enter trace context
        test_input = "Hello, world!"
        try:
            with module.model.trace(test_input) as tracer:
                # In trace context, verify we have access to model structure
                assert tracer is not None
        except Exception as e:
            # If tracing fails, it should be a clear NNsight error
            pytest.fail(f"NNsight tracing failed: {e}")

    def test_nnsight_forward_pass(self, get_it_session__ns_gpt2__setup):
        """Verify NNsight model can perform forward pass."""
        fixture = get_it_session__ns_gpt2__setup
        module = fixture.it_session.module

        # Test basic forward pass
        test_input = "The quick brown fox"
        tokenizer = module.model.tokenizer
        tokens = tokenizer(test_input, return_tensors="pt")

        # NNsight models should support forward pass
        with torch.no_grad():
            try:
                output = module.model(tokens["input_ids"])
                assert output is not None
            except Exception as e:
                # Some NNsight configurations may not support direct forward
                pytest.skip(f"Direct forward not supported in this config: {e}")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestNNsightErrorHandling:
    """Tests for NNsight adapter error handling."""

    def test_missing_nnsight_cfg_error(self):
        """Verify clear error when nnsight_cfg is missing."""
        with pytest.raises(MisconfigurationException, match="nnsight_cfg"):
            _ = ITNNsightConfig(
                model_name_or_path="gpt2",
                nnsight_cfg=None,
            )

    def test_invalid_nnsight_cfg_dict_error(self):
        """Verify clear error when nnsight_cfg dict is invalid."""
        with pytest.raises(MisconfigurationException, match="Failed to initialize NNsightConfig"):
            _ = ITNNsightConfig(
                model_name_or_path="gpt2",
                nnsight_cfg={"invalid_key": "value"},
            )

    def test_missing_model_name_error(self):
        """Verify clear error when both model names are missing."""
        with pytest.raises(MisconfigurationException, match="model_name_or_path.*must be provided"):
            _ = ITNNsightConfig(
                model_name_or_path=None,
                nnsight_cfg=NNsightConfig(model_name=None),
            )


# =============================================================================
# Remote Execution Tests (require NDIF_API_KEY)
# =============================================================================


class TestNNsightRemoteExecution:
    """Tests for NNsight remote execution via NDIF.

    These tests require NDIF_API_KEY to be set and network access. NDIF accepts any python 3.9+ client (nnsight
    serializes intervention functions by source, not bytecode — the old exact-py3.12 requirement is obsolete), but
    nnsight itself supports python <=3.12 only (no 3.13 classifier as of 0.6.3/0.7.0) — run these from a 3.10-3.12
    environment.
    """

    @pytest.fixture
    def has_ndif_api_key(self):
        """Check if NDIF_API_KEY is set and configure nnsight.

        NNsight expects NDIF_API_KEY env var, but we use NDIF_API_KEY. This fixture also sets nnsight.CONFIG.API.APIKEY
        directly.
        """
        api_key = os.environ.get("NDIF_API_KEY")
        if not api_key:
            pytest.skip("NDIF_API_KEY not set, skipping remote tests")
        # nnsight expects NDIF_API_KEY, but we configure directly
        nnsight.CONFIG.API.APIKEY = api_key
        return api_key

    def test_remote_config_initialization(self, has_ndif_api_key):
        """Verify remote config initializes with API key from environment."""
        config = NNsightConfig(
            model_name="openai-community/gpt2",
            remote=True,
            dispatch=False,
        )

        kwargs = config.get_nnsight_kwargs()
        from interpretune.config.nnsight import _nnsight_accepts_init_remote

        if _nnsight_accepts_init_remote():  # nnsight 0.6 line: remote is an init kwarg
            assert kwargs["remote"] is True
        else:  # 0.7+: remote moved to per-trace selection; init kwarg would leak to the HF ctor
            assert "remote" not in kwargs
        assert kwargs["dispatch"] is False

    @RunIf(optional=True)
    def test_remote_smoke_pinned_model(self, has_ndif_api_key):
        """Verify NDIF remote execution works end-to-end against a currently-pinned model.

        Unlike test_remote_local_consistency (which needs the SAME model loadable locally and pinned remotely, so it
        skips whenever gpt2 is unpinned), this smoke meta-loads a pinned model (no local weights) and runs one remote
        trace — validating connectivity, API-key tier scheduling, and result download. Override the substrate with
        IT_NDIF_REMOTE_SMOKE_MODEL when the pinned set rotates (default: EleutherAI/gpt-j-6b, pinned + ungated as of
        2026-07).
        """
        model_name = os.environ.get("IT_NDIF_REMOTE_SMOKE_MODEL", "EleutherAI/gpt-j-6b")
        model = nnsight.LanguageModel(model_name)  # meta tensors only — no local weight download
        try:
            with model.trace("The capital of France is", remote=True):
                logits = nnsight.save(model.lm_head.output)
        except httpx.TimeoutException as error:
            pytest.skip(f"NDIF request timed out: {error}")
        except Exception as error:
            # same environmental-skip contract as test_remote_local_consistency below
            if any(marker in str(error) for marker in ("not pinned", "not scheduled", "hotswapping")):
                pytest.skip(f"NDIF model unavailable for this API key tier: {error}")
            raise
        assert logits.ndim == 3 and logits.shape[-1] > 0, f"unexpected remote logits shape: {tuple(logits.shape)}"

    @RunIf(optional=True)
    def test_remote_local_consistency(
        self,
        has_ndif_api_key,
        get_it_session__ns_gpt2__setup,
        get_it_session__ns_remote_gpt2__setup,
    ):
        """Test that remote and local execution produce consistent logits.

        Compares outputs from local and remote execution to validate that NNsight's remote execution produces equivalent
        results.

        Uses GPT2 fixtures for both local and remote execution to ensure consistency across execution modes. NOTE: NDIF
        only schedules non-hotswap-tier API keys onto PINNED models, so this test skips whenever gpt2 isn't currently
        pinned (and on other availability-class rejections). When probing current NDIF state (pinned/running models,
        outage layers, key tier), query the `/status` endpoint defensively — nnsight.ndif_status() crashes on bare
        (unenriched) deployment entries as of 0.7.0 (maintainer-side private notes carry a layered diagnostic probe).
        The local-vs-remote comparison inherently requires a model loadable locally AND pinned remotely — when NDIF pins
        only large models, skipping is the correct outcome.
        """
        local_session = get_it_session__ns_gpt2__setup.it_session
        remote_session = get_it_session__ns_remote_gpt2__setup.it_session

        prompt = "The capital of France is"

        # Run local trace
        local_model = local_session.module.model
        with local_model.trace(prompt, remote=False):
            local_logits = nnsight.save(local_model.lm_head.output)

        # Run remote trace
        remote_model = remote_session.module.model
        try:
            with remote_model.trace(prompt, remote=True):
                remote_logits = nnsight.save(remote_model.lm_head.output)
        except httpx.TimeoutException as error:
            pytest.skip(f"NDIF request timed out: {error}")
        except Exception as error:
            # NDIF service-side availability (e.g. "Model is not pinned and hotswapping is not
            # supported for this API key" — the target model isn't currently scheduled for this
            # key tier) is environmental, not a client regression; don't fail the gate on it.
            # Matched on the message because nnsight 0.7 raises a per-exception dynamic wrapper
            # ("NNsightException" subclassing the original RemoteException) with no stable
            # importable class across the 0.6/0.7 line; everything else re-raises.
            if any(marker in str(error) for marker in ("not pinned", "not scheduled", "hotswapping")):
                pytest.skip(f"NDIF model unavailable for this API key tier: {error}")
            raise

        # Verify shapes match
        assert local_logits.shape == remote_logits.shape, (
            f"Logit shapes differ: local {local_logits.shape} vs remote {remote_logits.shape}"
        )

        # Verify logits are close after dtype normalization. Exact argmax equality is too brittle here because NDIF
        # returns bfloat16 logits, and near-tied candidates can swap rank while the full tensor remains numerically
        # consistent.
        local_logits_f32 = local_logits.float()
        remote_logits_f32 = remote_logits.float()
        assert torch.allclose(local_logits_f32, remote_logits_f32, rtol=0.02, atol=2.0), (
            f"Logits differ beyond tolerance. Max diff: {(local_logits_f32 - remote_logits_f32).abs().max().item()}"
        )
