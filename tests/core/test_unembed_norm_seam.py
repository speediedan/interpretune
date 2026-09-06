"""The sanctioned seam for unembed + final-norm-scale resolution.

The per-family conventions (HF gemma `1 + weight`, other RMSNorms `weight`, TransformerLens
folded-at-load, LayerNorm `weight` plus centering) previously lived only in an op collection's
private copy, where each consumer was free to get a different row wrong. This seam is their single
home, so a convention is asserted once here rather than rediscovered per caller.

Tests use real transformers norm classes rather than stubs wherever the convention depends on the
CLASS (kind detection reads the class name), and stubs where only the attribute shape matters.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.optools import (
    UnembedNormInfo,
    _rmsnorm_scale,
    fold_norm_into_unembed_rows,
    resolve_unembed_and_norm_scale,
)

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


class TestRMSNormOffsetIsPerFamilyNotPerPrefix:
    """The gemma line splits on whether its RMSNorm applies `(1 + weight)` or `weight`.

    A prefix test was correct for every family that existed when it was written and silently wrong for
    `gemma3n` and the whole `gemma4` line. The cases below that assert `weight` for a `gemma*` family are
    the ones that fail against a prefix rule, so they are what keeps the fix from being reverted by a
    reasonable-looking simplification.
    """

    WEIGHT = 0.5

    @pytest.mark.parametrize("model_type", ["gemma", "gemma2", "gemma3", "gemma3_text"])
    def test_families_that_apply_the_offset(self, model_type):
        scale = _rmsnorm_scale(torch.full((D,), self.WEIGHT), model_type)
        torch.testing.assert_close(scale, torch.full((D,), 1.0 + self.WEIGHT))

    @pytest.mark.parametrize(
        "model_type",
        ["gemma3n", "gemma3n_text", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_assistant"],
    )
    def test_gemma_families_that_do_NOT_apply_the_offset(self, model_type):
        """These are the cases a `startswith("gemma")` rule gets wrong, so they are the regression."""
        scale = _rmsnorm_scale(torch.full((D,), self.WEIGHT), model_type)
        torch.testing.assert_close(scale, torch.full((D,), self.WEIGHT))

    @pytest.mark.parametrize("model_type", ["llama", "qwen3", "mistral", ""])
    def test_families_outside_the_gemma_namespace_apply_weight_directly(self, model_type):
        scale = _rmsnorm_scale(torch.full((D,), self.WEIGHT), model_type)
        torch.testing.assert_close(scale, torch.full((D,), self.WEIGHT))

    def test_an_unrecognized_gemma_family_warns_rather_than_guessing_silently(self):
        """Both guesses are wrong for some member of this namespace and neither is visible in the output.

        The value still has to be something, so it assumes `weight`; the point is that the assumption is
        announced rather than made silently.
        """
        with pytest.warns(UserWarning, match="unrecognized gemma-family model_type"):
            scale = _rmsnorm_scale(torch.full((D,), self.WEIGHT), "gemma5_hypothetical")
        torch.testing.assert_close(scale, torch.full((D,), self.WEIGHT))

    def test_a_recognized_family_does_not_warn(self):
        """The positive control: if everything warned, the warning above would carry no information."""
        import warnings

        for model_type in ("gemma3", "gemma4", "llama"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _rmsnorm_scale(torch.full((D,), self.WEIGHT), model_type)

    def test_the_offset_decision_reaches_the_resolved_seam(self):
        """End to end through `resolve_unembed_and_norm_scale`, not just the helper."""
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

        norm = Gemma3RMSNorm(D)
        norm.weight.data = torch.full((D,), self.WEIGHT)
        offset = resolve_unembed_and_norm_scale(_hf_module("gemma3", norm))
        plain = resolve_unembed_and_norm_scale(_hf_module("gemma4", norm))
        torch.testing.assert_close(offset.norm_scale, torch.full((D,), 1.0 + self.WEIGHT))
        torch.testing.assert_close(plain.norm_scale, torch.full((D,), self.WEIGHT))
        assert offset.norm_kind == plain.norm_kind == "rmsnorm"

    def test_a_gemma_family_with_no_rmsnorm_neither_warns_nor_reaches_for_one(self):
        """Two gemma families ship no RMSNorm class at all, so the seam must not fire on their name.

        The unrecognized-family warning keys on the family name, which makes it tempting to check the name first. It is
        deliberately reachable only from inside the rmsnorm branch, so a gemma-family model whose final norm is absent
        or of another kind passes through silently.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            info = resolve_unembed_and_norm_scale(_hf_module("gemma4_assistant", object()))
        assert info.norm_scale is None and info.norm_kind == "none"

    def test_a_layernorm_is_untouched_by_the_rmsnorm_rule(self):
        """The offset question is RMSNorm-only; a LayerNorm in any family applies its weight directly."""
        norm = torch.nn.LayerNorm(D)
        norm.weight.data = torch.full((D,), self.WEIGHT)
        info = resolve_unembed_and_norm_scale(_hf_module("gemma3", norm, inner_attr="transformer"))
        assert info.norm_kind == "layernorm"
        torch.testing.assert_close(info.norm_scale, torch.full((D,), self.WEIGHT))


class TestFoldNormIntoUnembedRows:
    """The folded row must reproduce the model's OWN readout direction, per norm kind.

    Each case checks the composition against the norm module actually applied, so a wrong convention fails on a number
    rather than on a restatement of the convention being tested.
    """

    @staticmethod
    def _x():
        torch.manual_seed(1)
        return torch.randn(D)

    def test_rmsnorm_folded_row_reproduces_the_readout_up_to_the_input_scalar(self):
        w_u = torch.randn(VOCAB, D)
        scale = torch.linspace(0.3, 2.7, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind="rmsnorm")
        x, c = self._x(), 5

        rms = x.pow(2).mean().sqrt()
        readout = w_u[c] @ (scale * x / rms)
        folded = fold_norm_into_unembed_rows(info, [c])[0]

        torch.testing.assert_close(folded @ x / rms, readout)

    def test_layernorm_folded_row_needs_the_centering_to_reproduce_the_readout(self):
        w_u = torch.randn(VOCAB, D)
        scale = torch.linspace(0.3, 2.7, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind="layernorm")
        x, c = self._x(), 5

        std = x.var(unbiased=False).sqrt()
        readout = w_u[c] @ (scale * (x - x.mean()) / std)
        folded = fold_norm_into_unembed_rows(info, [c])[0]
        torch.testing.assert_close(folded @ x / std, readout)

        # The positive control: the same row WITHOUT centering does not reproduce it, so the
        # assertion above is capable of failing when the projector is dropped.
        uncentered = w_u[c] * scale
        assert not torch.isclose(uncentered @ x / std, readout, atol=1e-4, rtol=1e-4)

    def test_rmsnorm_rows_are_not_centered(self):
        w_u = torch.randn(VOCAB, D)
        scale = torch.linspace(0.3, 2.7, D)
        rms_row = fold_norm_into_unembed_rows(UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind="rmsnorm"), [3])[0]
        ln_row = fold_norm_into_unembed_rows(UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind="layernorm"), [3])[0]
        assert rms_row.mean().abs() > 1e-3, "an RMSNorm row must keep its uniform component"
        torch.testing.assert_close(ln_row.mean(), torch.zeros(()), atol=1e-6, rtol=0)

    def test_apply_norm_false_is_the_probing_shorthand(self):
        w_u = torch.randn(VOCAB, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=torch.linspace(0.3, 2.7, D), norm_kind="rmsnorm")
        torch.testing.assert_close(fold_norm_into_unembed_rows(info, [2, 7], apply_norm=False), w_u[[2, 7]])

    def test_absent_scale_returns_raw_rows_rather_than_guessing_one(self):
        w_u = torch.randn(VOCAB, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=None, norm_kind="none")
        torch.testing.assert_close(fold_norm_into_unembed_rows(info, [1]), w_u[[1]])

    def test_rows_are_returned_per_id_in_order(self):
        w_u = torch.randn(VOCAB, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=None, norm_kind="none")
        rows = fold_norm_into_unembed_rows(info, [4, 1, 4])
        assert rows.shape == (3, D)
        torch.testing.assert_close(rows[0], rows[2])

    def test_empty_token_group_raises(self):
        info = UnembedNormInfo(w_u=torch.randn(VOCAB, D), norm_scale=None, norm_kind="none")
        with pytest.raises(ValueError, match="at least one token id"):
            fold_norm_into_unembed_rows(info, [])
