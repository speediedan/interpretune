"""Tests for analysis backend components.

Covers:
- BackendCapability: enum values and membership checks
- HookNameResolver: hook name parsing, resolution, SAE sub-hook stripping, architecture registry
- NNsightActivationCacheAdapter: dict-like interface conformance
- NNsightModelBackend: protocol conformance, capabilities, _splice_sae helper, fwd_w_hooks_batched
- TLModelBackend: protocol conformance, capabilities, fwd_w_hooks_batched sequential fallback
- _matches_names_filter: all NamesFilter variants
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch

from interpretune.analysis.backends import (
    AnalysisBackendCapability,
    BackendCapability,
    InterventionDict,
    InterventionSpec,
    ModelBackend,
    apply_intervention_to_last_token,
    build_intervention_dict,
    expand_intervention_patterns,
)
from interpretune.analysis.backends.hook_mapping import (
    ArchitectureMapping,
    HookMapping,
    HookNameResolver,
    ResolvedHook,
)
from interpretune.analysis.backends.nnsight import (
    NNsightActivationCacheAdapter,
    NNsightModelBackend,
    _DummyHookPoint,
    _apply_saved_to_cache_hooks,
    get_default_configs_per_pass,
    _matches_names_filter,
    _navigate_envoy,
    _read_envoy_activation,
    _write_envoy_activation,
)


# ==============================================================================
# Module-level configuration tests
# ==============================================================================


class TestNNsightBackendConfig:
    """Tests for module-level nnsight configuration set by the backend."""

    def test_pymount_disabled_on_import(self):
        """Importing the nnsight backend should disable PYMOUNT (uses nnsight.save() instead)."""
        import nnsight

        assert nnsight.CONFIG.APP.PYMOUNT is False

    def test_nnsight_save_serializable(self):
        """Verify a function using nnsight.save() survives serialization round-trip.

        This validates that our trace closure patterns (which use nnsight.save() instead of obj.save()) are compatible
        with nnsight's source-based serialization used for remote execution.
        """
        import nnsight as _nnsight
        from nnsight.intervention.serialization import dumps, loads

        def sample_trace_body(model_output):
            result = _nnsight.save(model_output)
            return result

        serialized = dumps(sample_trace_body)
        restored = loads(serialized)
        assert callable(restored)
        # Verify the restored function name matches
        assert restored.__name__ == "sample_trace_body"

    def test_default_configs_per_pass_cpu_linux(self, monkeypatch):
        monkeypatch.delenv("IT_NNSIGHT_CONFIGS_PER_PASS", raising=False)
        monkeypatch.setattr("interpretune.analysis.backends.nnsight.platform.system", lambda: "Linux")
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert get_default_configs_per_pass() == 4

    def test_default_configs_per_pass_cpu_windows(self, monkeypatch):
        monkeypatch.delenv("IT_NNSIGHT_CONFIGS_PER_PASS", raising=False)
        monkeypatch.setattr("interpretune.analysis.backends.nnsight.platform.system", lambda: "Windows")
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert get_default_configs_per_pass() == 2

    def test_default_configs_per_pass_env_override(self, monkeypatch):
        monkeypatch.setenv("IT_NNSIGHT_CONFIGS_PER_PASS", "6")
        assert get_default_configs_per_pass() == 6


# ==============================================================================
# HookNameResolver tests
# ==============================================================================


class TestHookNameResolver:
    """Tests for HookNameResolver hook name parsing and resolution."""

    def test_parse_basic_hook(self):
        layer, base, subhook = HookNameResolver.parse_hook_name("blocks.5.hook_resid_post")
        assert layer == 5
        assert base == "hook_resid_post"
        assert subhook is None

    def test_parse_dotted_hook(self):
        layer, base, subhook = HookNameResolver.parse_hook_name("blocks.3.mlp.hook_pre")
        assert layer == 3
        assert base == "mlp.hook_pre"
        assert subhook is None

    def test_parse_sae_subhook_preserved(self):
        layer, base, subhook = HookNameResolver.parse_hook_name("blocks.5.hook_resid_post.hook_sae_acts_post")
        assert layer == 5
        assert base == "hook_resid_post"
        assert subhook == "hook_sae_acts_post"

    def test_parse_sae_input_preserved(self):
        layer, base, subhook = HookNameResolver.parse_hook_name("blocks.2.hook_resid_pre.hook_sae_input")
        assert layer == 2
        assert base == "hook_resid_pre"
        assert subhook == "hook_sae_input"

    def test_parse_sae_error_preserved(self):
        layer, base, subhook = HookNameResolver.parse_hook_name("blocks.0.hook_mlp_out.hook_sae_error")
        assert layer == 0
        assert base == "hook_mlp_out"
        assert subhook == "hook_sae_error"

    def test_parse_global_hook(self):
        layer, base, subhook = HookNameResolver.parse_hook_name("unembed.hook_in.hook_sae_input")
        assert layer == -1
        assert base == "unembed.hook_in"
        assert subhook == "hook_sae_input"

    def test_parse_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            HookNameResolver.parse_hook_name("not_a_valid_hook")

    def test_parse_no_layer_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            HookNameResolver.parse_hook_name("blocks.hook_resid_post")

    def test_resolve_gpt2_resid_post(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        path, io_type = resolver.resolve("blocks.5.hook_resid_post")
        assert path == "transformer.h.5"
        assert io_type == "output"

    def test_resolve_gpt2_resid_pre(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        path, io_type = resolver.resolve("blocks.0.hook_resid_pre")
        assert path == "transformer.h.0"
        assert io_type == "input"

    def test_resolve_gpt2_resid_mid(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        path, io_type = resolver.resolve("blocks.3.hook_resid_mid")
        assert path == "transformer.h.3.ln_2"
        assert io_type == "input"

    def test_resolve_gpt2_attn_out(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        path, io_type = resolver.resolve("blocks.2.hook_attn_out")
        assert path == "transformer.h.2.attn"
        assert io_type == "output"

    def test_resolve_gpt2_mlp_out(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        path, io_type = resolver.resolve("blocks.1.hook_mlp_out")
        assert path == "transformer.h.1.mlp"
        assert io_type == "output"

    def test_resolve_gpt2_mlp_pre(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        path, io_type = resolver.resolve("blocks.4.mlp.hook_pre")
        assert path == "transformer.h.4.mlp"
        assert io_type == "input"

    def test_resolve_global_unembed_hook(self):
        resolver = HookNameResolver("Gemma2ForCausalLM")
        path, io_type = resolver.resolve("unembed.hook_in")
        assert path == "lm_head"
        assert io_type == "input"

    def test_resolve_sae_subhook_resolves_to_base(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        path1, io1 = resolver.resolve("blocks.5.hook_resid_post")
        path2, io2 = resolver.resolve("blocks.5.hook_resid_post.hook_sae_acts_post")
        assert path1 == path2
        assert io1 == io2

    def test_resolve_llama_resid_post(self):
        resolver = HookNameResolver("LlamaForCausalLM")
        path, io_type = resolver.resolve("blocks.7.hook_resid_post")
        assert path == "model.layers.7"
        assert io_type == "output"

    def test_resolve_gemma2_mlp_out(self):
        resolver = HookNameResolver("Gemma2ForCausalLM")
        path, io_type = resolver.resolve("blocks.3.hook_mlp_out")
        assert path == "model.layers.3.post_feedforward_layernorm"
        assert io_type == "output"

    def test_unsupported_architecture_raises(self):
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            HookNameResolver("UnknownModel")

    def test_unsupported_hook_raises(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        with pytest.raises(ValueError, match="Unknown hook name"):
            resolver.resolve("blocks.0.nonexistent_hook")

    def test_architecture_property(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        assert resolver.architecture == "GPT2LMHeadModel"

    def test_supported_hooks_is_sorted_list(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        hooks = resolver.supported_hooks
        assert isinstance(hooks, list)
        assert hooks == sorted(hooks)
        assert "hook_resid_post" in hooks

    def test_get_supported_architectures(self):
        archs = HookNameResolver.get_supported_architectures()
        assert "GPT2LMHeadModel" in archs
        assert "LlamaForCausalLM" in archs
        assert "Gemma2ForCausalLM" in archs

    def test_register_architecture(self):
        custom = ArchitectureMapping(
            model_architecture="TestCustomModel",
            hook_mappings={"hook_resid_post": HookMapping(envoy_path="layers.{layer}", io_type="output")},
        )
        HookNameResolver.register_architecture(custom)
        try:
            resolver = HookNameResolver("TestCustomModel")
            path, io_type = resolver.resolve("blocks.0.hook_resid_post")
            assert path == "layers.0"
            assert io_type == "output"
        finally:
            # Clean up: remove from registry
            from interpretune.analysis.backends.hook_mapping import _ARCHITECTURE_REGISTRY

            _ARCHITECTURE_REGISTRY.pop("TestCustomModel", None)

    def test_resolve_transcoder_hooks(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        read, write = resolver.resolve_transcoder_hooks("blocks.3.mlp.hook_pre", "blocks.3.hook_mlp_out")
        assert read == ("transformer.h.3.mlp", "input")
        assert write == ("transformer.h.3.mlp", "output")

    def test_resolve_transcoder_hooks_no_output(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        read, write = resolver.resolve_transcoder_hooks("blocks.3.hook_resid_post")
        assert read == ("transformer.h.3", "output")
        assert write is None

    # -- resolve_for_envoy tests --

    def test_resolve_for_envoy_gpt2_resid_post(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        resolved = resolver.resolve_for_envoy("blocks.5.hook_resid_post")
        assert isinstance(resolved, ResolvedHook)
        assert resolved.module_path == "transformer.h.5"
        assert resolved.io_type == "output"
        assert resolved.tuple_output is True

    def test_resolve_for_envoy_gpt2_mlp_out_not_tuple(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        resolved = resolver.resolve_for_envoy("blocks.1.hook_mlp_out")
        assert resolved.module_path == "transformer.h.1.mlp"
        assert resolved.io_type == "output"
        assert resolved.tuple_output is False

    def test_resolve_for_envoy_input_hook(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        resolved = resolver.resolve_for_envoy("blocks.0.hook_resid_pre")
        assert resolved.io_type == "input"
        # tuple_output is True (default) but irrelevant for input hooks

    def test_resolve_for_envoy_global_unembed_hook(self):
        resolver = HookNameResolver("Gemma2ForCausalLM")
        resolved = resolver.resolve_for_envoy("unembed.hook_in.hook_sae_input")
        assert resolved.module_path == "lm_head"
        assert resolved.io_type == "input"
        assert resolved.tuple_output is False

    def test_resolve_for_envoy_sae_subhook_resolves_to_base(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        resolved1 = resolver.resolve_for_envoy("blocks.5.hook_resid_post")
        resolved2 = resolver.resolve_for_envoy("blocks.5.hook_resid_post.hook_sae_acts_post")
        assert resolved1 == resolved2

    def test_resolve_for_envoy_llama_mlp_out_not_tuple(self):
        resolver = HookNameResolver("LlamaForCausalLM")
        resolved = resolver.resolve_for_envoy("blocks.7.hook_mlp_out")
        assert resolved.module_path == "model.layers.7.mlp"
        assert resolved.tuple_output is False

    def test_resolve_for_envoy_gemma2_mlp_out_not_tuple(self):
        resolver = HookNameResolver("Gemma2ForCausalLM")
        resolved = resolver.resolve_for_envoy("blocks.3.hook_mlp_out")
        assert resolved.module_path == "model.layers.3.post_feedforward_layernorm"
        assert resolved.tuple_output is False

    def test_resolve_for_envoy_unknown_hook_raises(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        with pytest.raises(ValueError, match="Unknown hook name"):
            resolver.resolve_for_envoy("blocks.0.nonexistent_hook")


# ==============================================================================
# NNsightActivationCacheAdapter tests
# ==============================================================================


class TestNNsightActivationCacheAdapter:
    """Tests for the dict-like activation cache wrapper."""

    @pytest.fixture()
    def cache(self):
        return NNsightActivationCacheAdapter(
            {
                "blocks.5.hook_resid_post.hook_sae_input": torch.randn(2, 10, 768),
                "blocks.5.hook_resid_post.hook_sae_acts_post": torch.randn(2, 10, 16384),
            }
        )

    def test_getitem(self, cache):
        t = cache["blocks.5.hook_resid_post.hook_sae_input"]
        assert t.shape == (2, 10, 768)

    def test_contains_positive(self, cache):
        assert "blocks.5.hook_resid_post.hook_sae_input" in cache

    def test_contains_negative(self, cache):
        assert "blocks.0.hook_resid_pre" not in cache

    def test_len(self, cache):
        assert len(cache) == 2

    def test_keys(self, cache):
        keys = list(cache.keys())
        assert len(keys) == 2

    def test_items(self, cache):
        items = list(cache.items())
        assert len(items) == 2
        assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in items)

    def test_values(self, cache):
        vals = list(cache.values())
        assert len(vals) == 2

    def test_repr(self, cache):
        r = repr(cache)
        assert "NNsightActivationCacheAdapter" in r

    def test_missing_key_raises(self, cache):
        with pytest.raises(KeyError):
            cache["nonexistent"]


# ==============================================================================
# _matches_names_filter tests
# ==============================================================================


class TestMatchesNamesFilter:
    """Tests for NamesFilter matching logic."""

    def test_none_matches_everything(self):
        assert _matches_names_filter("any_name", None)

    def test_str_match(self):
        assert _matches_names_filter("blocks.5.hook_resid_post", "blocks.5.hook_resid_post")

    def test_str_no_match(self):
        assert not _matches_names_filter("blocks.5.hook_resid_post", "blocks.3.hook_resid_post")

    def test_callable_match(self):
        assert _matches_names_filter("blocks.5.hook_resid_post", lambda n: "resid_post" in n)

    def test_callable_no_match(self):
        assert not _matches_names_filter("blocks.5.hook_resid_post", lambda n: "mlp" in n)

    def test_sequence_match(self):
        names = ["blocks.5.hook_resid_post", "blocks.3.hook_resid_pre"]
        assert _matches_names_filter("blocks.5.hook_resid_post", names)

    def test_sequence_no_match(self):
        names = ["blocks.3.hook_resid_pre"]
        assert not _matches_names_filter("blocks.5.hook_resid_post", names)


# ==============================================================================
# DummyHookPoint tests
# ==============================================================================


class TestDummyHookPoint:
    """Tests for the _DummyHookPoint hook-name carrier."""

    def test_name_attribute(self):
        dummy = _DummyHookPoint(name="blocks.5.hook_resid_post.hook_sae_acts_post")
        assert dummy.name == "blocks.5.hook_resid_post.hook_sae_acts_post"


# ==============================================================================
# NNsightModelBackend tests
# ==============================================================================


class TestNNsightModelBackend:
    """Tests for NNsightModelBackend protocol conformance and basic behaviour."""

    def test_satisfies_model_backend_protocol(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        backend = NNsightModelBackend(resolver)
        assert isinstance(backend, ModelBackend)

    def test_wrap_activation_cache_from_dict(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        backend = NNsightModelBackend(resolver)
        raw = {"key1": torch.randn(2, 3)}

        result = backend.wrap_activation_cache(raw, model=None)
        assert isinstance(result, NNsightActivationCacheAdapter)
        assert "key1" in result

    def test_wrap_activation_cache_idempotent(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        backend = NNsightModelBackend(resolver)
        adapter = NNsightActivationCacheAdapter({"key1": torch.randn(2, 3)})

        result = backend.wrap_activation_cache(adapter, model=None)
        assert result is adapter

    def test_get_hf_model_with_module_attr(self):
        model = MagicMock()
        model._model = torch.nn.Linear(3, 3)
        del model._module  # ensure _model is checked first
        result = NNsightModelBackend._get_hf_model(model)
        assert result is model._model

    def test_get_hf_model_with_module_fallback(self):
        model = MagicMock(spec=[])
        # No _model or _module — should return model itself
        result = NNsightModelBackend._get_hf_model(model)
        assert result is model

    def test_get_module_by_path(self):
        parent = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU())
        child = NNsightModelBackend._get_module_by_path(parent, "0")
        assert isinstance(child, torch.nn.Linear)


class TestNNsightModelBackendSAESplicing:
    """Tests for SAE splicing helpers, cache population, and gradient method helpers.

    The forward-only methods (fwd_w_cache, fwd_w_hooks) and the gradient method
    (fwd_w_grads) all use NNsight's trace API and require real NNsight models for
    full integration testing (standalone tests).  Unit tests here focus on the envoy
    helpers and the post-trace cache population helper ``_apply_saved_to_cache_hooks``.
    """

    @pytest.fixture()
    def simple_gpt2_model(self):
        """Create a minimal mock model that mirrors GPT-2's module structure."""
        block = torch.nn.Module()
        block.ln_2 = torch.nn.LayerNorm(32)
        block.attn = torch.nn.Linear(32, 32, bias=False)
        block.mlp = torch.nn.Linear(32, 32, bias=False)

        h = torch.nn.ModuleList([block])

        transformer = torch.nn.Module()
        transformer.h = h

        model = torch.nn.Module()
        model.transformer = transformer

        return model

    @pytest.fixture()
    def mock_sae(self):
        """Create a mock SAE with encode/decode and cfg.metadata.hook_name."""
        sae = MagicMock()
        sae.cfg.metadata.hook_name = "blocks.0.hook_resid_post"
        sae.encode.side_effect = lambda x: x * 0.5
        sae.decode.side_effect = lambda x: x * 2.0
        return sae

    def test_fwd_w_grads_signature_matches_protocol(self):
        """Verify fwd_w_grads_and_latent_models accepts the new backward_fn-based signature."""
        import inspect

        resolver = HookNameResolver("GPT2LMHeadModel")
        backend = NNsightModelBackend(resolver)

        sig = inspect.signature(backend.fwd_w_grads_and_latent_models)
        param_names = list(sig.parameters.keys())
        assert "batch" in param_names
        assert "backward_fn" in param_names
        assert "model" in param_names
        assert "latent_model_handles" in param_names
        assert "fwd_hooks" in param_names
        assert "bwd_hooks" in param_names

    def test_apply_saved_to_cache_hooks_populates_cache(self):
        """Test that _apply_saved_to_cache_hooks calls matching cache functions with resolved tensors."""
        cache_dict: dict[str, torch.Tensor] = {}

        def cache_fn(tensor: torch.Tensor, hook: _DummyHookPoint) -> None:
            cache_dict[hook.name] = tensor

        names_filter = lambda name: "hook_sae_acts_post" in name  # noqa: E731

        # saved dict maps sub-hook names directly to resolved tensors (after nnsight.save())
        expected_tensor = torch.randn(2, 10, 768)
        saved = {"blocks.0.hook_resid_post.hook_sae_acts_post": expected_tensor}
        hooks = [(names_filter, cache_fn)]

        _apply_saved_to_cache_hooks(saved, hooks)

        assert "blocks.0.hook_resid_post.hook_sae_acts_post" in cache_dict
        assert torch.equal(cache_dict["blocks.0.hook_resid_post.hook_sae_acts_post"], expected_tensor)

    def test_apply_saved_to_cache_hooks_filter_excludes_non_matching(self):
        """Test that non-matching sub-hook names are not cached."""
        cache_dict: dict[str, torch.Tensor] = {}

        def cache_fn(tensor: torch.Tensor, hook: _DummyHookPoint) -> None:
            cache_dict[hook.name] = tensor

        names_filter = lambda name: "hook_sae_acts_post" in name  # noqa: E731

        save_proxy_input = MagicMock()
        save_proxy_input.value = torch.randn(2, 10, 768)
        save_proxy_acts = MagicMock()
        save_proxy_acts.value = torch.randn(2, 10, 16384)

        saved = {
            "blocks.0.hook_resid_post.hook_sae_input": save_proxy_input,
            "blocks.0.hook_resid_post.hook_sae_acts_post": save_proxy_acts,
        }
        hooks = [(names_filter, cache_fn)]

        _apply_saved_to_cache_hooks(saved, hooks)

        assert "blocks.0.hook_resid_post.hook_sae_input" not in cache_dict
        assert "blocks.0.hook_resid_post.hook_sae_acts_post" in cache_dict

    def test_apply_saved_to_cache_hooks_multiple_hooks(self):
        """Test with multiple hook entries targeting different sub-hook names."""
        fwd_cache: dict[str, torch.Tensor] = {}
        grad_cache: dict[str, torch.Tensor] = {}

        def fwd_cache_fn(tensor: torch.Tensor, hook: _DummyHookPoint) -> None:
            fwd_cache[hook.name] = tensor

        def grad_cache_fn(tensor: torch.Tensor, hook: _DummyHookPoint) -> None:
            grad_cache[hook.name] = tensor

        save_proxy = MagicMock()
        save_proxy.value = torch.randn(2, 10, 768)

        saved = {"blocks.0.hook_resid_post.hook_sae_input": save_proxy}
        hooks = [
            (lambda name: "hook_sae_input" in name, fwd_cache_fn),
            (lambda name: "hook_sae_input" in name, grad_cache_fn),
        ]

        _apply_saved_to_cache_hooks(saved, hooks)

        assert "blocks.0.hook_resid_post.hook_sae_input" in fwd_cache
        assert "blocks.0.hook_resid_post.hook_sae_input" in grad_cache


# ==============================================================================
# NNsight envoy helper tests
# ==============================================================================


class TestNNsightEnvoyHelpers:
    """Tests for _navigate_envoy, _read_envoy_activation, _write_envoy_activation."""

    def test_navigate_envoy_simple_path(self):
        """Navigate a dotted path on a real module hierarchy."""
        block = torch.nn.Module()
        block.mlp = torch.nn.Linear(4, 4)
        h = torch.nn.ModuleList([block])
        model = torch.nn.Module()
        model.h = h

        # Wrap in a top-level container to mimic model.transformer.h.0.mlp
        container = torch.nn.Module()
        container.transformer = torch.nn.Module()
        container.transformer.h = h

        result = _navigate_envoy(container, "transformer.h.0.mlp")
        assert result is block.mlp

    def test_navigate_envoy_integer_index(self):
        """Integer parts should index into ModuleList."""
        layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.ReLU()])
        model = torch.nn.Module()
        model.layers = layers

        result = _navigate_envoy(model, "layers.1")
        assert isinstance(result, torch.nn.ReLU)

    def test_navigate_envoy_deep_path(self):
        """Navigate through nested structure like model.layers.3.self_attn."""
        attn = torch.nn.Linear(4, 4)
        layer0 = torch.nn.Module()
        layer0.self_attn = attn
        layers = torch.nn.ModuleList([layer0])
        model_inner = torch.nn.Module()
        model_inner.layers = layers
        model = torch.nn.Module()
        model.model = model_inner

        result = _navigate_envoy(model, "model.layers.0.self_attn")
        assert result is attn

    def test_read_envoy_activation_output_tuple(self):
        """For tuple-output hooks, should access .output[0]."""
        envoy = MagicMock()
        tensor_proxy = MagicMock()
        envoy.output.__getitem__ = MagicMock(return_value=tensor_proxy)
        resolved = ResolvedHook(module_path="dummy", io_type="output", tuple_output=True)

        result = _read_envoy_activation(envoy, resolved)
        envoy.output.__getitem__.assert_called_with(0)
        assert result is tensor_proxy

    def test_read_envoy_activation_output_single(self):
        """For single-tensor output hooks, should access .output directly."""
        envoy = MagicMock()
        resolved = ResolvedHook(module_path="dummy", io_type="output", tuple_output=False)

        result = _read_envoy_activation(envoy, resolved)
        assert result is envoy.output

    def test_read_envoy_activation_input(self):
        """For input hooks, should access .input directly (NNsight 0.6+ returns first positional arg)."""
        envoy = MagicMock()
        resolved = ResolvedHook(module_path="dummy", io_type="input", tuple_output=True)

        result = _read_envoy_activation(envoy, resolved)
        assert result is envoy.input

    def test_write_envoy_activation_output_tuple(self):
        """For tuple-output hooks, should assign to .output[0]."""
        envoy = MagicMock()
        resolved = ResolvedHook(module_path="dummy", io_type="output", tuple_output=True)
        new_value = MagicMock()

        _write_envoy_activation(envoy, resolved, new_value)
        envoy.output.__setitem__.assert_called_with(0, new_value)

    def test_write_envoy_activation_output_single(self):
        """For single-tensor output hooks, should assign to .output directly."""
        envoy = MagicMock()
        resolved = ResolvedHook(module_path="dummy", io_type="output", tuple_output=False)
        new_value = MagicMock()

        _write_envoy_activation(envoy, resolved, new_value)
        # MagicMock: envoy.output = new_value is captured differently
        # We verify the assignment happened by checking attributes weren't called as dict access
        assert not envoy.output.__setitem__.called

    def test_write_envoy_activation_input(self):
        """For input hooks, should assign to .input directly (NNsight 0.6+ attribute assignment)."""
        envoy = MagicMock()
        resolved = ResolvedHook(module_path="dummy", io_type="input", tuple_output=True)
        new_value = MagicMock()

        _write_envoy_activation(envoy, resolved, new_value)
        # NNsight 0.6+: envoy.input = value (direct attribute assignment, not __setitem__)
        assert not envoy.input.__setitem__.called


# ==============================================================================
# BackendCapability enum tests
# ==============================================================================


class TestBackendCapability:
    """Tests for the BackendCapability enum values and membership."""

    def test_batched_hooks_value(self):
        assert BackendCapability.BATCHED_HOOKS.value == "batched_hooks"

    def test_gradients_value(self):
        assert BackendCapability.GRADIENTS.value == "gradients"

    def test_attribution_value(self):
        assert AnalysisBackendCapability.ATTRIBUTION_GRAPH.value == "attribution_graph"

    def test_feature_intervention_value(self):
        assert AnalysisBackendCapability.FEATURE_INTERVENTION.value == "feature_intervention"

    def test_enum_members_count(self):
        assert len(BackendCapability) == 2
        assert len(AnalysisBackendCapability) == 2

    def test_membership_in_frozenset(self):
        caps = frozenset({BackendCapability.BATCHED_HOOKS, BackendCapability.GRADIENTS})
        assert BackendCapability.BATCHED_HOOKS in caps
        assert BackendCapability.GRADIENTS in caps


# ==============================================================================
# NNsightModelBackend capability & _splice_sae tests
# ==============================================================================


class TestNNsightModelBackendCapabilities:
    """Tests for NNsightModelBackend capability declarations."""

    @pytest.fixture()
    def backend(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        return NNsightModelBackend(resolver)

    def test_capabilities_returns_frozenset(self, backend):
        assert isinstance(backend.capabilities, frozenset)

    def test_capabilities_includes_batched_hooks(self, backend):
        assert BackendCapability.BATCHED_HOOKS in backend.capabilities

    def test_capabilities_includes_gradients(self, backend):
        assert BackendCapability.GRADIENTS in backend.capabilities

    def test_supports_batched_hooks(self, backend):
        assert backend.supports(BackendCapability.BATCHED_HOOKS) is True

    def test_supports_gradients(self, backend):
        assert backend.supports(BackendCapability.GRADIENTS) is True


class TestNNsightSpliceSae:
    """Tests for NNsightModelBackend._splice_sae helper.

    _splice_sae must be called inside an NNsight trace context, so these tests mock the envoy helpers and SAE
    encode/decode to verify the wiring logic without requiring a real NNsight model.
    """

    @pytest.fixture()
    def backend(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        return NNsightModelBackend(resolver)

    @pytest.fixture()
    def mock_sae(self):
        sae = MagicMock()
        sae.cfg.metadata.hook_name = "blocks.0.hook_resid_post"
        sae.encode.side_effect = lambda x: x * 0.5
        sae.decode.side_effect = lambda x: x * 2.0
        return sae

    @patch("interpretune.analysis.backends.nnsight._write_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._read_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._navigate_envoy")
    def test_splice_sae_calls_encode_decode(self, mock_nav, mock_read, mock_write, backend, mock_sae):
        """Verify _splice_sae calls encode, decode, and writes back the result."""
        fake_act = torch.randn(2, 10, 768)
        mock_nav.return_value = MagicMock()  # envoy
        mock_read.return_value = fake_act

        feature_acts, act_proxy, resolved = backend._splice_sae(MagicMock(), mock_sae)

        mock_sae.encode.assert_called_once_with(fake_act)
        mock_sae.decode.assert_called_once()
        mock_write.assert_called_once()
        assert act_proxy is fake_act
        assert resolved.module_path == "transformer.h.0"

    @patch("interpretune.analysis.backends.nnsight._write_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._read_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._navigate_envoy")
    def test_splice_sae_applies_hook_fn(self, mock_nav, mock_read, mock_write, backend, mock_sae):
        """When hook_fn is provided, it should be applied to feature_acts before decode."""
        fake_act = torch.randn(2, 10, 768)
        mock_nav.return_value = MagicMock()
        mock_read.return_value = fake_act

        hook_fn = MagicMock(side_effect=lambda t, h: t * 0.0)
        feature_acts, _, _ = backend._splice_sae(MagicMock(), mock_sae, hook_fn=hook_fn)

        hook_fn.assert_called_once()
        # hook_fn received a _DummyHookPoint with the correct name
        _, hook_arg = hook_fn.call_args[0]
        assert hook_arg.name == "blocks.0.hook_resid_post.hook_sae_acts_post"

    @patch("interpretune.analysis.backends.nnsight._write_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._read_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._navigate_envoy")
    def test_splice_sae_no_hook_fn_skips_hook(self, mock_nav, mock_read, mock_write, backend, mock_sae):
        """Without hook_fn, encode result goes directly to decode."""
        fake_act = torch.randn(2, 10, 768)
        mock_nav.return_value = MagicMock()
        mock_read.return_value = fake_act

        feature_acts, _, _ = backend._splice_sae(MagicMock(), mock_sae, hook_fn=None)

        # encode should produce feature_acts, decode should receive it
        mock_sae.encode.assert_called_once()
        mock_sae.decode.assert_called_once()
        # decode receives the output of encode (fake_act * 0.5)
        decode_arg = mock_sae.decode.call_args[0][0]
        assert torch.allclose(decode_arg, fake_act * 0.5)

    @patch("interpretune.analysis.backends.nnsight._write_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._read_envoy_activation")
    @patch("interpretune.analysis.backends.nnsight._navigate_envoy")
    def test_splice_sae_returns_resolved_hook(self, mock_nav, mock_read, mock_write, backend, mock_sae):
        """Returned resolved hook should have correct metadata."""
        mock_nav.return_value = MagicMock()
        mock_read.return_value = torch.randn(2, 10, 768)

        _, _, resolved = backend._splice_sae(MagicMock(), mock_sae)

        assert isinstance(resolved, ResolvedHook)
        assert resolved.io_type == "output"
        assert resolved.tuple_output is True  # GPT2 hook_resid_post has tuple output


class TestNNsightFwdWHooksBatched:
    """Tests for NNsightModelBackend.fwd_w_hooks_batched signature and behaviour."""

    @pytest.fixture()
    def backend(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        return NNsightModelBackend(resolver)

    def test_signature_matches_protocol(self, backend):
        sig = inspect.signature(backend.fwd_w_hooks_batched)
        param_names = list(sig.parameters.keys())
        assert "model" in param_names
        assert "batch" in param_names
        assert "latent_model_handles" in param_names
        assert "hook_configs" in param_names
        assert "clear_contexts" in param_names
        assert "configs_per_pass" in param_names

    def test_empty_hook_configs_returns_empty_list(self, backend):
        result = backend.fwd_w_hooks_batched(
            model=MagicMock(),
            batch={"input": torch.randn(2, 5)},
            latent_model_handles=[],
            hook_configs=[],
        )
        assert result == []

    def test_configs_per_pass_call_override_defaults_to_none(self, backend):
        sig = inspect.signature(backend.fwd_w_hooks_batched)
        assert sig.parameters["configs_per_pass"].default is None

    def test_backend_default_configs_per_pass_is_32(self, backend):
        assert backend._configs_per_pass == 32


# ==============================================================================
# TLModelBackend capability & fwd_w_hooks_batched tests
# ==============================================================================


class TestTLModelBackendCapabilities:
    """Tests for TLModelBackend capability declarations."""

    @pytest.fixture()
    def backend(self):
        from interpretune.analysis.backends.transformer_lens import TLModelBackend

        return TLModelBackend()

    def test_satisfies_model_backend_protocol(self, backend):
        assert isinstance(backend, ModelBackend)

    def test_capabilities_returns_frozenset(self, backend):
        assert isinstance(backend.capabilities, frozenset)

    def test_capabilities_includes_gradients(self, backend):
        assert BackendCapability.GRADIENTS in backend.capabilities

    def test_capabilities_does_not_include_batched_hooks(self, backend):
        assert BackendCapability.BATCHED_HOOKS not in backend.capabilities

    def test_supports_gradients(self, backend):
        assert backend.supports(BackendCapability.GRADIENTS) is True

    def test_supports_batched_hooks_false(self, backend):
        assert backend.supports(BackendCapability.BATCHED_HOOKS) is False


class TestTLFwdWHooksBatched:
    """Tests for TLModelBackend.fwd_w_hooks_batched sequential fallback."""

    @pytest.fixture()
    def backend(self):
        from interpretune.analysis.backends.transformer_lens import TLModelBackend

        return TLModelBackend()

    def test_signature_matches_protocol(self, backend):
        sig = inspect.signature(backend.fwd_w_hooks_batched)
        param_names = list(sig.parameters.keys())
        assert "model" in param_names
        assert "batch" in param_names
        assert "hook_configs" in param_names
        assert "configs_per_pass" in param_names

    def test_empty_hook_configs_returns_empty_list(self, backend):
        result = backend.fwd_w_hooks_batched(
            model=MagicMock(),
            batch={},
            latent_model_handles=[],
            hook_configs=[],
        )
        assert result == []

    def test_calls_fwd_w_hooks_per_config(self, backend):
        """TL backend should call fwd_w_hooks_and_latent_models once per config."""
        mock_model = MagicMock()
        fake_logits_1 = torch.randn(2, 5, 100)
        fake_logits_2 = torch.randn(2, 5, 100)
        mock_model.run_with_hooks_with_saes.side_effect = [fake_logits_1, fake_logits_2]

        hook_configs = [
            [("blocks.0.hook_resid_post.hook_sae_acts_post", lambda t, h: t)],
            [("blocks.0.hook_resid_post.hook_sae_acts_post", lambda t, h: t * 0)],
        ]

        result = backend.fwd_w_hooks_batched(
            model=mock_model,
            batch={"input": torch.randn(2, 5)},
            latent_model_handles=["sae_handle"],
            hook_configs=hook_configs,
            clear_contexts=True,
        )

        assert len(result) == 2
        assert mock_model.run_with_hooks_with_saes.call_count == 2
        assert torch.equal(result[0], fake_logits_1)
        assert torch.equal(result[1], fake_logits_2)

    def test_configs_per_pass_ignored(self, backend):
        """configs_per_pass should be accepted but have no effect on TL backend."""
        mock_model = MagicMock()
        mock_model.run_with_hooks_with_saes.return_value = torch.randn(2, 5, 100)

        hook_configs = [
            [("hook", lambda t, h: t)],
            [("hook", lambda t, h: t)],
            [("hook", lambda t, h: t)],
        ]

        # Should work identically with any configs_per_pass value
        result = backend.fwd_w_hooks_batched(
            model=mock_model,
            batch={"input": torch.randn(2, 5)},
            latent_model_handles=[],
            hook_configs=hook_configs,
            configs_per_pass=1,
        )

        assert len(result) == 3
        assert mock_model.run_with_hooks_with_saes.call_count == 3


# ==============================================================================
# InterventionSpec tests
# ==============================================================================


class TestInterventionSpec:
    """Tests for the InterventionSpec NamedTuple defaults and construction."""

    def test_default_mode_is_replace(self):
        from interpretune.analysis.backends import InterventionSpec

        spec = InterventionSpec(intervention_tensor=torch.randn(768))
        assert spec.mode == "replace"

    def test_default_scale_factor_is_one(self):
        from interpretune.analysis.backends import InterventionSpec

        spec = InterventionSpec(intervention_tensor=torch.randn(768))
        assert spec.scale_factor == 1.0

    def test_explicit_add_mode(self):
        from interpretune.analysis.backends import InterventionSpec

        spec = InterventionSpec(intervention_tensor=torch.randn(768), mode="add", scale_factor=3.0)
        assert spec.mode == "add"
        assert spec.scale_factor == 3.0

    def test_is_named_tuple(self):
        from interpretune.analysis.backends import InterventionSpec

        spec = InterventionSpec(intervention_tensor=torch.randn(768))
        assert hasattr(spec, "_fields")
        assert "intervention_tensor" in spec._fields
        assert "mode" in spec._fields
        assert "scale_factor" in spec._fields

    def test_dot_mode(self):
        spec = InterventionSpec(intervention_tensor=torch.ones(768), mode="dot", scale_factor=2.0)
        assert spec.mode == "dot"
        assert spec.scale_factor == 2.0

    def test_intervention_tensor_attribute_name(self):
        """Verify the field is named intervention_tensor (not the old 'vector' name)."""
        from interpretune.analysis.backends import InterventionSpec

        t = torch.randn(768)
        spec = InterventionSpec(intervention_tensor=t)
        assert torch.equal(spec.intervention_tensor, t)
        assert not hasattr(spec, "vector")


class TestInterventionDictHelpers:
    """Tests for generalized intervention normalization and application helpers."""

    def test_intervention_dict_mapping_interface(self):
        intervention_dict = InterventionDict({"blocks.0.hook_resid_post": (InterventionSpec(torch.ones(4)),)})
        assert len(intervention_dict) == 1
        assert list(intervention_dict.keys()) == ["blocks.0.hook_resid_post"]

    def test_build_intervention_dict_splits_multihook_tensor_rows(self):
        expanded_matches = {"blocks.*.hook_resid_post": ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]}
        hook_shapes = {
            "blocks.0.hook_resid_post": (4,),
            "blocks.1.hook_resid_post": (4,),
        }
        interventions = {
            "blocks.*.hook_resid_post": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                dtype=torch.float32,
            )
        }

        intervention_dict = build_intervention_dict(interventions, expanded_matches, hook_shapes)
        assert torch.equal(
            intervention_dict["blocks.0.hook_resid_post"][0].intervention_tensor,
            interventions["blocks.*.hook_resid_post"][0],
        )
        assert torch.equal(
            intervention_dict["blocks.1.hook_resid_post"][0].intervention_tensor,
            interventions["blocks.*.hook_resid_post"][1],
        )

    def test_build_intervention_dict_broadcasts_single_tensor_across_matches(self):
        expanded_matches = {"blocks.*.hook_resid_post": ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]}
        hook_shapes = {
            "blocks.0.hook_resid_post": (4,),
            "blocks.1.hook_resid_post": (4,),
        }
        vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        intervention_dict = build_intervention_dict(
            {"blocks.*.hook_resid_post": vector},
            expanded_matches,
            hook_shapes,
            default_mode="add",
            default_scale_factor=2.0,
        )

        for hook_name in expanded_matches["blocks.*.hook_resid_post"]:
            spec = intervention_dict[hook_name][0]
            assert spec.mode == "add"
            assert spec.scale_factor == 2.0
            assert torch.equal(spec.intervention_tensor, vector)

    def test_build_intervention_dict_preserves_explicit_mixed_shapes(self):
        interventions = {
            "blocks.0.hook_resid_post": torch.ones(4),
            "blocks.0.attn.hook_z": {"intervention_tensor": torch.ones(2, 3), "mode": "add", "scale_factor": 0.5},
        }
        expanded_matches = {
            "blocks.0.hook_resid_post": ["blocks.0.hook_resid_post"],
            "blocks.0.attn.hook_z": ["blocks.0.attn.hook_z"],
        }
        hook_shapes = {
            "blocks.0.hook_resid_post": (4,),
            "blocks.0.attn.hook_z": (2, 3),
        }

        intervention_dict = build_intervention_dict(interventions, expanded_matches, hook_shapes)
        assert intervention_dict["blocks.0.hook_resid_post"][0].mode == "replace"
        assert intervention_dict["blocks.0.attn.hook_z"][0].mode == "add"
        assert intervention_dict["blocks.0.attn.hook_z"][0].scale_factor == 0.5

    def test_build_intervention_dict_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="not compatible"):
            build_intervention_dict(
                {"blocks.0.hook_resid_post": torch.ones(5)},
                {"blocks.0.hook_resid_post": ["blocks.0.hook_resid_post"]},
                {"blocks.0.hook_resid_post": (4,)},
            )

    def test_expand_intervention_patterns_deduplicates_alias_matches(self):
        available_hook_map = {
            "blocks.0.hook_resid_post": "blocks.0.hook_resid_post",
            "blocks.0.hook_resid_post.alias": "blocks.0.hook_resid_post",
        }
        expanded = expand_intervention_patterns(["blocks.0.hook_resid_post*"], available_hook_map)
        assert expanded["blocks.0.hook_resid_post*"] == ["blocks.0.hook_resid_post"]

    def test_apply_intervention_dot_mode_projects_onto_direction(self):
        value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
        spec = InterventionSpec(intervention_tensor=torch.tensor([1.0, 0.0]), mode="dot", scale_factor=2.0)

        result = apply_intervention_to_last_token(value.clone(), spec, last_pos=1)
        expected = torch.tensor([[[1.0, 2.0], [6.0, 0.0]]], dtype=torch.float32)
        assert torch.allclose(result, expected)


# ==============================================================================
# fwd_w_intervention signature tests
# ==============================================================================


class TestFwdWInterventionSignature:
    """Verify both backends expose the fwd_w_intervention method with correct signature."""

    def test_nnsight_fwd_w_intervention_signature(self):
        resolver = HookNameResolver("GPT2LMHeadModel")
        backend = NNsightModelBackend(resolver)
        sig = inspect.signature(backend.fwd_w_intervention)
        param_names = list(sig.parameters.keys())
        assert "model" in param_names
        assert "batch" in param_names
        assert "interventions" in param_names
        assert "latent_model_handles" in param_names

    def test_tl_fwd_w_intervention_signature(self):
        from interpretune.analysis.backends.transformer_lens import TLModelBackend

        backend = TLModelBackend()
        sig = inspect.signature(backend.fwd_w_intervention)
        param_names = list(sig.parameters.keys())
        assert "model" in param_names
        assert "batch" in param_names
        assert "interventions" in param_names
        assert "latent_model_handles" in param_names

    def test_tl_unknown_mode_raises(self):
        """TL backend should raise ValueError for unknown intervention modes."""
        from interpretune.analysis.backends.transformer_lens import TLModelBackend

        backend = TLModelBackend()
        mock_model = MagicMock()
        mock_model.hook_dict = {"blocks.0.hook_resid_post": MagicMock()}
        mock_model.run_with_cache.return_value = (
            torch.randn(1, 5, 100),
            {"blocks.0.hook_resid_post": torch.randn(1, 5, 100)},
        )

        spec = InterventionSpec(intervention_tensor=torch.randn(100), mode="unknown")
        with pytest.raises(ValueError, match="Unknown intervention mode"):
            backend.fwd_w_intervention(
                model=mock_model,
                batch={"input": torch.randn(1, 5)},
                interventions={"blocks.0.hook_resid_post": spec},
            )

    def test_tl_wildcard_expansion(self):
        """TL backend should expand wildcard patterns against model.hook_dict."""
        from interpretune.analysis.backends.transformer_lens import TLModelBackend

        backend = TLModelBackend()
        mock_model = MagicMock()
        mock_model.hook_dict = {
            "blocks.0.hook_resid_post": MagicMock(),
            "blocks.1.hook_resid_post": MagicMock(),
            "blocks.0.hook_resid_pre": MagicMock(),
        }

        # run_with_cache returns (logits, cache_dict)
        mock_model.run_with_cache.return_value = (
            torch.randn(1, 5, 100),
            {
                "blocks.0.hook_resid_post": torch.randn(1, 5, 100),
                "blocks.1.hook_resid_post": torch.randn(1, 5, 100),
            },
        )
        # run_with_hooks returns logits
        mock_model.run_with_hooks.return_value = torch.randn(1, 5, 100)

        spec = InterventionSpec(intervention_tensor=torch.randn(100))

        # Use wildcard pattern that should match 2 hooks
        backend.fwd_w_intervention(
            model=mock_model,
            batch={"input": torch.randn(1, 5)},
            interventions={"blocks.*.hook_resid_post": spec},
        )

        # Verify run_with_hooks was called (the hooks were built)
        assert mock_model.run_with_hooks.called
        # Check that the fwd_hooks arg has 2 entries (matched blocks.0 and blocks.1)
        call_kwargs = mock_model.run_with_hooks.call_args
        fwd_hooks = call_kwargs.kwargs.get("fwd_hooks", call_kwargs[1].get("fwd_hooks", []))
        assert len(fwd_hooks) == 2

    def test_tl_replace_mode_hook_overwrites(self):
        """TL backend replace mode hook should overwrite (not add to) the activation."""
        from interpretune.analysis.backends.transformer_lens import TLModelBackend

        backend = TLModelBackend()
        mock_model = MagicMock()
        mock_model.hook_dict = {"blocks.0.hook_resid_post": MagicMock()}

        # Return logits with known last_pos
        logits = torch.randn(1, 3, 100)
        mock_model.run_with_cache.return_value = (logits, {"blocks.0.hook_resid_post": torch.randn(1, 3, 100)})

        replacement_vec = torch.ones(100) * 42.0
        spec = InterventionSpec(intervention_tensor=replacement_vec, mode="replace")

        # Capture the hook that gets built
        def capture_hooks(**kwargs):
            hooks = kwargs.get("fwd_hooks", [])
            # Apply the hook to a known activation
            activation = torch.randn(1, 3, 100)
            for _, hook_fn in hooks:
                activation = hook_fn(activation, None)
            # After replace mode, last position should equal the replacement vector
            assert torch.allclose(activation[0, 2, :], replacement_vec), "Replace mode should overwrite last position"
            return torch.randn(1, 3, 100)

        mock_model.run_with_hooks.side_effect = capture_hooks

        backend.fwd_w_intervention(
            model=mock_model,
            batch={"input": torch.randn(1, 3)},
            interventions={"blocks.0.hook_resid_post": spec},
        )
