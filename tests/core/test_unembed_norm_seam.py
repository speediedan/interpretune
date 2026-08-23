"""#225: the sanctioned seam for unembed + final-norm-scale resolution.

The per-family conventions (HF gemma `1 + weight`, other RMSNorms `weight`, TL folded-at-load,
LayerNorm `weight` + centering-left-to-caller) previously lived only in the jlens op collection's
private copy; this seam is their single home, and #330's investigation is what validates each row.
Tests use real transformers norm classes rather than stubs wherever the convention depends on the
CLASS (kind detection reads the class name), and stubs where only the attribute shape matters.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.optools import UnembedNormInfo, resolve_unembed_and_norm_scale

VOCAB, D = 16, 8


class _Cfg:
    def __init__(self, model_type):
        self.model_type = model_type


class _Head:
    def __init__(self):
        torch.manual_seed(0)
        self.weight = torch.randn(VOCAB, D)


def _hf_module(model_type: str, norm, inner_attr: str = "model", head_attr: str = "lm_head"):
    inner = type("Inner", (), {})()
    setattr(inner, {"model": "norm", "transformer": "ln_f", "gpt_neox": "final_layer_norm"}[inner_attr], norm)
    model = type("Model", (), {})()
    model.config = _Cfg(model_type)
    setattr(model, head_attr, _Head())
    setattr(model, inner_attr, inner)
    module = type("Module", (), {})()
    module.model = model
    return module


class TestHFFamilies:
    def test_gemma_rmsnorm_scale_is_one_plus_weight(self):
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

        norm = Gemma2RMSNorm(D)
        norm.weight.data = torch.full((D,), 0.5)
        info = resolve_unembed_and_norm_scale(_hf_module("gemma2", norm))
        assert info.norm_kind == "rmsnorm"
        torch.testing.assert_close(info.norm_scale, torch.full((D,), 1.5))
        assert info.w_u.shape == (VOCAB, D)

    def test_llama_rmsnorm_scale_is_weight_directly(self):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        norm = LlamaRMSNorm(D)
        norm.weight.data = torch.full((D,), 0.5)
        info = resolve_unembed_and_norm_scale(_hf_module("llama", norm))
        assert info.norm_kind == "rmsnorm"
        torch.testing.assert_close(info.norm_scale, torch.full((D,), 0.5))

    def test_gpt2_layernorm_reports_layernorm_kind(self):
        norm = torch.nn.LayerNorm(D)
        info = resolve_unembed_and_norm_scale(_hf_module("gpt2", norm, inner_attr="transformer"))
        assert info.norm_kind == "layernorm"
        torch.testing.assert_close(info.norm_scale, norm.weight)

    def test_pythia_embed_out_and_final_layer_norm_resolve(self):
        norm = torch.nn.LayerNorm(D)
        info = resolve_unembed_and_norm_scale(
            _hf_module("gpt_neox", norm, inner_attr="gpt_neox", head_attr="embed_out")
        )
        assert info.norm_kind == "layernorm" and info.w_u.shape == (VOCAB, D)

    def test_missing_final_norm_returns_none_scale_not_a_guess(self):
        model = type("Model", (), {})()
        model.config = _Cfg("mystery")
        model.lm_head = _Head()
        module = type("Module", (), {})()
        module.model = model
        info = resolve_unembed_and_norm_scale(module)
        assert info.norm_scale is None and info.norm_kind == "none"


class TestTransformerLensOrientation:
    def test_w_u_is_transposed_to_vocab_by_d(self):
        class TLModel:
            def __init__(self):
                torch.manual_seed(1)
                self.W_U = torch.randn(D, VOCAB)  # TL stores (d, vocab)

        module = type("Module", (), {})()
        module.model = TLModel()
        info = resolve_unembed_and_norm_scale(module)
        assert info.w_u.shape == (VOCAB, D)
        torch.testing.assert_close(info.w_u, module.model.W_U.T)
        assert info.norm_scale is None

    def test_tl_ln_final_weight_used_as_stored_with_kind_from_bias_presence(self):
        class LNFinal:
            def __init__(self, with_bias):
                self.w = torch.full((D,), 1.25)  # gemma-on-TL: the +1 is already folded at load
                if with_bias:
                    self.b = torch.zeros(D)

        for with_bias, kind in ((False, "rmsnorm"), (True, "layernorm")):

            class TLModel:
                def __init__(self):
                    self.W_U = torch.randn(D, VOCAB)
                    self.ln_final = LNFinal(with_bias)

            module = type("Module", (), {})()
            module.model = TLModel()
            info = resolve_unembed_and_norm_scale(module)
            assert info.norm_kind == kind
            torch.testing.assert_close(info.norm_scale, torch.full((D,), 1.25))


def test_no_unembed_surface_raises_rather_than_guessing():
    """Returning an embedding matrix would be right only for tied weights, wrongly silent elsewhere."""
    module = type("Module", (), {})()
    module.model = type("Model", (), {})()
    with pytest.raises(ValueError, match="neither an HF-style"):
        resolve_unembed_and_norm_scale(module)


def test_named_tuple_surface_is_stable():
    """The collection imports this by name; field renames are a compat break worth failing on."""
    assert UnembedNormInfo._fields == ("w_u", "norm_scale", "norm_kind")
