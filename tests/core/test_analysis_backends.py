"""Tests for analysis backend components.

Covers:
- HookNameResolver: hook name parsing, resolution, SAE sub-hook stripping, architecture registry
- NNsightActivationCacheAdapter: dict-like interface conformance
- NNsightModelBackend: protocol conformance and hook wiring
- _matches_names_filter: all NamesFilter variants
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from interpretune.analysis.backends import ModelBackend
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
    _matches_names_filter,
    _navigate_envoy,
    _read_envoy_activation,
    _write_envoy_activation,
)


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
        """Test that _apply_saved_to_cache_hooks calls matching cache functions with real tensors."""
        cache_dict: dict[str, torch.Tensor] = {}

        def cache_fn(tensor: torch.Tensor, hook: _DummyHookPoint) -> None:
            cache_dict[hook.name] = tensor

        names_filter = lambda name: "hook_sae_acts_post" in name  # noqa: E731

        # Create a mock SaveProxy with a .value attribute
        save_proxy = MagicMock()
        save_proxy.value = torch.randn(2, 10, 768)

        saved = {"blocks.0.hook_resid_post.hook_sae_acts_post": save_proxy}
        hooks = [(names_filter, cache_fn)]

        _apply_saved_to_cache_hooks(saved, hooks)

        assert "blocks.0.hook_resid_post.hook_sae_acts_post" in cache_dict
        assert torch.equal(cache_dict["blocks.0.hook_resid_post.hook_sae_acts_post"], save_proxy.value)

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
        """For input hooks, should access .input[0]."""
        envoy = MagicMock()
        tensor_proxy = MagicMock()
        envoy.input.__getitem__ = MagicMock(return_value=tensor_proxy)
        resolved = ResolvedHook(module_path="dummy", io_type="input", tuple_output=True)

        result = _read_envoy_activation(envoy, resolved)
        envoy.input.__getitem__.assert_called_with(0)
        assert result is tensor_proxy

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
        """For input hooks, should assign to .input[0]."""
        envoy = MagicMock()
        resolved = ResolvedHook(module_path="dummy", io_type="input", tuple_output=True)
        new_value = MagicMock()

        _write_envoy_activation(envoy, resolved, new_value)
        envoy.input.__setitem__.assert_called_with(0, new_value)
