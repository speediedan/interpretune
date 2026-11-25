"""A prefix key/value cache handed to a model backend's traces.

An attribution that scores the last position can run its forward and backward pass on that token alone, reading the
prefix from a key/value cache built without gradients. These tests hold the nnsight backend to the full-sequence values
at that position on every trace it runs, and every other backend to a refusal by name.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.adapters.nnsight.backends import NNsightModelBackend
from interpretune.analysis.backends.hook_mapping import HookNameResolver
from interpretune.analysis.backends.prefix_cache import fresh_prefix_cache, prefix_cache_kwargs
from interpretune.analysis.backends.readouts import Readout

IDS = torch.tensor([[464, 3139, 286, 4881, 318]])  # gpt2's "The capital of France is"
A, B = 1000, 2000
SITES = ["blocks.2.hook_out", "blocks.5.hook_in", "blocks.8.hook_out"]


def _readout() -> Readout:
    return Readout(lambda logits, activations: (logits[:, -1, A] - logits[:, -1, B]).sum())


def _edit() -> dict:
    vector = torch.randn(768, generator=torch.Generator().manual_seed(1))
    return {
        "blocks.5.hook_in": {
            "intervention_tensor": vector,
            "op": "add",
            "scale_factor": 1.0,
            "position_scope": "last_token",
        }
    }


@pytest.fixture(scope="module")
def lm():
    from nnsight import LanguageModel

    lm = LanguageModel("gpt2", device_map="cpu", dispatch=True)
    # gpt2's safetensors file leaves every mapped tensor misaligned, and on Apple silicon a one-row CPU matmul with a
    # large misaligned weight returns NaN or garbage (pytorch/pytorch#191238). A cached forward is one row, so the
    # weights are copied to aligned memory until torch handles it.
    for parameter in lm._module.parameters():
        parameter.data = parameter.data.clone()
    return lm


@pytest.fixture(scope="module")
def backend():
    return NNsightModelBackend(HookNameResolver("GPT2LMHeadModel"))


def _prefix_cache(lm, ids: torch.Tensor = IDS):
    with torch.no_grad():
        return lm._module(ids[:, :-1], use_cache=True).past_key_values


def _cached_batch(lm) -> dict:
    return {"input_ids": IDS[:, -1:], "past_key_values": _prefix_cache(lm)}


class TestNNsightPrefixCache:
    def test_a_cached_gradient_equals_the_full_sequence_gradient_at_its_position(self, lm, backend):
        full, _, full_value = backend.bwd_w_intervention(lm, {"input_ids": IDS}, {}, _readout(), capture=SITES)
        cached, _, cached_value = backend.bwd_w_intervention(lm, _cached_batch(lm), {}, _readout(), capture=SITES)
        torch.testing.assert_close(cached_value, full_value, atol=1e-3, rtol=1e-4)
        for site in SITES:
            assert cached[site].shape[1] == 1
            torch.testing.assert_close(cached[site][:, -1], full[site][:, -1], atol=1e-5, rtol=1e-4)
        # positive control: the last token run without its prefix scores a different value, so the match above is the
        # cache's doing and a dropped cache would fail this test
        uncached, _, uncached_value = backend.bwd_w_intervention(
            lm, {"input_ids": IDS[:, -1:]}, {}, _readout(), capture=SITES
        )
        assert abs(float(uncached_value) - float(full_value)) > 1e-1

    def test_every_trace_starts_from_the_cache_as_handed(self, lm, backend):
        """An edit runs a baseline trace and an edited one; both must read the same prefix, and the caller's cache
        keeps its length."""
        batch = _cached_batch(lm)
        prefix_len = batch["past_key_values"].get_seq_length()
        _, full_edited, _ = backend.bwd_w_intervention(
            lm, {"input_ids": IDS}, _edit(), _readout(), capture=["blocks.2.hook_out"]
        )
        for _ in range(2):
            _, edited, _ = backend.bwd_w_intervention(lm, batch, _edit(), _readout(), capture=["blocks.2.hook_out"])
            torch.testing.assert_close(
                edited["blocks.2.hook_out"][:, -1], full_edited["blocks.2.hook_out"][:, -1], atol=1e-5, rtol=1e-4
            )
            assert batch["past_key_values"].get_seq_length() == prefix_len

    def test_a_cached_forward_equals_the_full_forward_at_its_position(self, lm, backend):
        full_logits, full_acts = backend.fwd_w_cache(lm, {"input_ids": IDS}, SITES)
        cached_logits, cached_acts = backend.fwd_w_cache(lm, _cached_batch(lm), SITES)
        assert cached_logits.shape[1] == 1
        torch.testing.assert_close(cached_logits[:, -1].detach(), full_logits[:, -1].detach(), atol=1e-3, rtol=1e-4)
        for site in SITES:
            torch.testing.assert_close(cached_acts[site][:, -1], full_acts[site][:, -1], atol=1e-4, rtol=1e-4)

    def test_batched_configs_each_run_from_the_cache(self, lm, backend):
        """Nnsight merges a trace's invocations into one forward, so with a cache each config runs a trace of its
        own from the cache as handed."""
        full_logits = backend.fwd(lm, {"input_ids": IDS})
        results = backend.fwd_w_hooks_batched(lm, _cached_batch(lm), [], [[], []], configs_per_pass=2)
        assert len(results) == 2
        for logits in results:
            torch.testing.assert_close(logits[:, -1].detach(), full_logits[:, -1].detach(), atol=1e-3, rtol=1e-4)


class TestFreshPrefixCache:
    @staticmethod
    def _dynamic(layers: int = 2):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for layer in range(layers):
            cache.update(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4), layer)
        return cache

    def test_extending_the_copy_leaves_the_handed_cache_and_shares_its_prefix(self):
        cache = self._dynamic()
        fresh = fresh_prefix_cache(cache)
        assert fresh.layers[0].keys is cache.layers[0].keys  # the prefix tensors are shared, not copied
        fresh.update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4), 0)
        assert fresh.get_seq_length() == 4 and cache.get_seq_length() == 3
        assert bool((cache.layers[0].keys == 1).all())

    def test_a_static_cache_is_refused_by_name(self):
        from transformers import GPT2Config
        from transformers.cache_utils import StaticCache

        cache = StaticCache(config=GPT2Config(n_layer=2, n_embd=8, n_head=2), max_cache_len=4)
        with pytest.raises(TypeError, match=r"can hold only .*DynamicLayer.*got \['StaticLayer'\]"):
            fresh_prefix_cache(cache)

    def test_an_object_that_is_not_a_cache_is_refused_by_name(self):
        legacy = ((torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4)),)
        with pytest.raises(TypeError, match="must be a transformers Cache .* got tuple"):
            fresh_prefix_cache(legacy)

    def test_forward_kwargs_carry_a_fresh_cache_and_turn_it_on(self):
        cache = self._dynamic()
        kwargs = prefix_cache_kwargs({"input_ids": IDS[:, -1:], "past_key_values": cache})
        assert set(kwargs) == {"past_key_values", "use_cache"} and kwargs["use_cache"] is True
        assert kwargs["past_key_values"] is not cache
        assert prefix_cache_kwargs({"input_ids": IDS}) == {}


class TestTransformerLensRefusesAPrefixCache:
    @pytest.mark.parametrize(
        "call",
        [
            lambda backend, batch: backend.fwd(None, batch),
            lambda backend, batch: backend.fwd_w_cache(None, batch, None),
            lambda backend, batch: backend.fwd_w_intervention(None, batch, {}),
            lambda backend, batch: backend.bwd_w_intervention(None, batch, {}, _readout()),
        ],
        ids=["fwd", "fwd_w_cache", "fwd_w_intervention", "bwd_w_intervention"],
    )
    def test_a_batch_carrying_a_cache_is_refused_by_name(self, call):
        from interpretune.adapters.transformer_lens.backends import TLModelBackend

        batch = {"input": IDS[:, -1:], "past_key_values": TestFreshPrefixCache._dynamic()}
        with pytest.raises(
            NotImplementedError, match=r"TransformerLens bridge backend cannot run a trace from a prefix"
        ):
            call(TLModelBackend(), batch)
