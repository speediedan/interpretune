"""#273: the hook map's `tuple_output` flags must be measured, not trusted.

The static flags describe the transformers version the map was written against. transformers 5.x
decoder blocks return plain tensors where 4.x returned tuples, and against a plain tensor
`envoy.output[0]` silently READS batch row 0 (caching a (seq, d) activation that downstream shape
logic misreads) and on the write path OVERWRITES batch row 0 -- a wrong-answer defect, not a crash.
Found by the first consumer of `blocks.N.hook_resid_post` output hooks on the NNsight path (the
J-space demo section): every prior demo used input-io hooks, which take a different code path.

These tests assert calibration agrees with what the test MEASURES ITSELF on the same model, so they
stay true whichever way a future transformers release moves.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends.hook_mapping import HookNameResolver


@pytest.fixture(scope="module")
def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=32, n_positions=16)
    return GPT2LMHeadModel(config).eval()


def _measure_directly(model, submodule_path: str) -> bool:
    """Ground truth: is this module's forward output a tuple, measured with a plain hook?"""
    seen: dict[str, bool] = {}

    def record(_m, _a, out):
        seen["is_tuple"] = isinstance(out, tuple)

    handle = model.get_submodule(submodule_path).register_forward_hook(record)
    with torch.no_grad():
        model(torch.zeros(1, 2, dtype=torch.long))
    handle.remove()
    return seen["is_tuple"]


class TestTupleOutputCalibration:
    def test_calibration_agrees_with_direct_measurement(self, tiny_gpt2):
        resolver = HookNameResolver("GPT2LMHeadModel")
        measured = resolver.calibrate_tuple_outputs(tiny_gpt2)
        assert measured, "probe reached no output hooks at all"
        assert measured["hook_resid_post"] == _measure_directly(tiny_gpt2, "transformer.h.0")
        assert measured["hook_mlp_out"] == _measure_directly(tiny_gpt2, "transformer.h.0.mlp")

    def test_resolve_for_envoy_prefers_measured_over_static(self, tiny_gpt2):
        """The defect: the static flag said tuple, the model disagreed, and `output[0]` ate batch row 0."""
        resolver = HookNameResolver("GPT2LMHeadModel")
        static_flag = resolver.resolve_for_envoy("blocks.0.hook_resid_post").tuple_output
        resolver.calibrate_tuple_outputs(tiny_gpt2)
        calibrated = resolver.resolve_for_envoy("blocks.0.hook_resid_post").tuple_output
        assert calibrated == _measure_directly(tiny_gpt2, "transformer.h.0")
        # On transformers 5.x these genuinely differ, which is what makes this a regression test and
        # not a tautology. If a future release reverts to tuples, the assert above still holds and
        # this informational check simply stops distinguishing.
        if not calibrated:
            assert static_flag != calibrated, "static flag now agrees; the map default was updated?"

    def test_probe_failure_leaves_static_flags_in_force(self):
        class ExplodingModel:
            def get_submodule(self, _path):
                raise AttributeError("no such module")

            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

        resolver = HookNameResolver("GPT2LMHeadModel")
        before = resolver.resolve_for_envoy("blocks.0.hook_resid_post").tuple_output
        measured = resolver.calibrate_tuple_outputs(ExplodingModel())
        assert measured == {}
        assert resolver.resolve_for_envoy("blocks.0.hook_resid_post").tuple_output == before

    def test_backend_calibration_is_idempotent(self, tiny_gpt2, monkeypatch):
        from interpretune.adapters.nnsight.backends import NNsightModelBackend, get_default_configs_per_pass

        backend = NNsightModelBackend(
            HookNameResolver("GPT2LMHeadModel"), configs_per_pass=get_default_configs_per_pass()
        )
        calls = {"n": 0}
        real = backend._resolver.calibrate_tuple_outputs

        def counting(model):
            calls["n"] += 1
            return real(model)

        monkeypatch.setattr(backend._resolver, "calibrate_tuple_outputs", counting)
        monkeypatch.setattr(NNsightModelBackend, "_get_hf_model", staticmethod(lambda m: tiny_gpt2))
        backend._ensure_tuple_calibration(object())
        backend._ensure_tuple_calibration(object())
        assert calls["n"] == 1
