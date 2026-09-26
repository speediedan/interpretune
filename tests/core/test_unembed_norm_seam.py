"""The sanctioned seam for unembed + final-norm-scale resolution.

The per-family conventions (HF gemma `1 + weight`, other RMSNorms `weight`, TransformerLens
folded-at-load, LayerNorm `weight` plus centering) previously lived only in an op collection's
private copy, where each consumer was free to get a different row wrong. This seam is their single
home, so a convention is asserted once here rather than rediscovered per caller.

Tests use real transformers norm classes rather than stubs wherever the convention depends on what the
norm COMPUTES (kind and scale are both read by evaluating it), and stubs where only the attribute shape matters.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.optools import (
    UnembedNormInfo,
    _final_norm_kind,
    _rmsnorm_scale,
    fold_norm_into_unembed_rows,
    jlens_basis_name,
    jlens_direction_rows,
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


class _Wrapped(torch.nn.Module):
    """A norm behind a wrapper that hides its class, the way a TransformerBridge component does."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.weight = inner.weight
        self.bias = getattr(inner, "bias", None)

    def forward(self, x):
        return self.inner(x)


def _layernorm(dtype=torch.float32):
    torch.manual_seed(1)
    norm = torch.nn.LayerNorm(D).to(dtype)
    norm.weight.data = (0.5 + torch.rand(D)).to(dtype)
    norm.bias.data = (torch.arange(D).float() * 0.25 - 1.0).to(dtype)
    return norm


class TestNormKindIsReadFromBehaviourNotClassName:
    """The seam's real inputs arrive wrapped (``NormalizationBridge`` under a TransformerBridge, ``Envoy`` under
    nnsight), so a class-name test read every wrapped LayerNorm as an RMSNorm and returned its bias as the
    scale."""

    @pytest.mark.parametrize("name", ["NormalizationBridge", "Envoy"])
    def test_a_wrapped_layernorm_keeps_its_kind_scale_and_bias(self, name):
        norm = _layernorm()
        wrapped = type(name, (_Wrapped,), {})(norm)
        info = resolve_unembed_and_norm_scale(_hf_module("gpt2", wrapped, inner_attr="transformer"))
        assert info.norm_kind == "layernorm"
        torch.testing.assert_close(info.norm_scale, norm.weight)
        torch.testing.assert_close(info.norm_bias, norm.bias)

    def test_a_wrapped_rmsnorm_is_still_probed_for_its_applied_scale(self):
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

        norm = Gemma2RMSNorm(D)
        norm.weight.data = torch.full((D,), 0.5)
        info = resolve_unembed_and_norm_scale(_hf_module("gemma2", type("NormalizationBridge", (_Wrapped,), {})(norm)))
        assert info.norm_kind == "rmsnorm"
        torch.testing.assert_close(info.norm_scale, torch.full((D,), 1.5))

    def test_a_real_nnsight_envoy_over_a_layernorm(self):
        envoy = pytest.importorskip("nnsight.intervention.envoy")
        norm = _layernorm()
        info = resolve_unembed_and_norm_scale(_hf_module("gpt2", envoy.Envoy(norm), inner_attr="transformer"))
        assert info.norm_kind == "layernorm"
        torch.testing.assert_close(info.norm_scale.float(), norm.weight.float())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_the_probe_holds_in_reduced_precision(self, dtype):
        """The probe and its shift are exact in bf16, so a LayerNorm's centering cancels the shift there too."""
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        rms = LlamaRMSNorm(D).to(dtype)
        assert _final_norm_kind(_layernorm(dtype), _layernorm(dtype).weight) == "layernorm"
        assert _final_norm_kind(rms, rms.weight) == "rmsnorm"

    def test_a_norm_that_cannot_be_evaluated_is_refused_by_name(self):
        class Unprobeable(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("needs an attention mask")

        with pytest.raises(RuntimeError, match=r"could not classify the final norm 'Unprobeable'"):
            _final_norm_kind(Unprobeable(), torch.ones(D))

    def test_an_all_zero_output_is_refused_rather_than_defaulted(self):
        class Zeros(torch.nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        with pytest.raises(RuntimeError, match=r"returned all zeros"):
            _final_norm_kind(Zeros(), torch.ones(D))


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
    """The collection imports this by name; field renames are a compat break worth failing on.

    Additive fields with defaults (like ``norm_bias``) are not: every existing construction and
    reader keeps working, which the keyword-built cases below rely on.
    """
    assert UnembedNormInfo._fields == ("w_u", "norm_scale", "norm_kind", "norm_bias")


class TestRMSNormOffsetIsPerFamilyNotPerPrefix:
    """The gemma line splits on whether its RMSNorm applies `(1 + weight)` or `weight`.

    A prefix test was correct for every family that existed when it was written and silently wrong for
    `gemma3n` and the whole `gemma4` line. That split is still real and is why the scale is now READ from
    the norm module rather than declared from a table: a table has to be right about every family that
    exists and every family that will exist, and being wrong about one is invisible in the output.

    These cases therefore assert the same facts as before against the new mechanism. The ones asserting
    `weight` for a `gemma*` family remain the regression: they fail against a prefix rule, and they now
    also fail against any reintroduced table that gets an entry wrong.
    """

    WEIGHT = 0.5
    OFFSET_FAMILIES = [("gemma", "Gemma"), ("gemma2", "Gemma2"), ("gemma3", "Gemma3")]
    PLAIN_GEMMA_FAMILIES = [("gemma3n", "Gemma3n"), ("gemma4", "Gemma4")]
    OUTSIDE_FAMILIES = [("llama", "Llama"), ("qwen3", "Qwen3"), ("mistral", "Mistral")]

    def _norm(self, model_type: str, prefix: str):
        module = pytest.importorskip(f"transformers.models.{model_type}.modeling_{model_type}")
        cls = getattr(module, f"{prefix}RMSNorm", None)
        if cls is None:
            pytest.skip(f"{prefix}RMSNorm absent from this transformers build")
        norm = cls(D)
        with torch.no_grad():
            norm.weight.data = torch.full((D,), self.WEIGHT).to(norm.weight.dtype)
        return norm

    @pytest.mark.parametrize("model_type, prefix", OFFSET_FAMILIES)
    def test_families_that_apply_the_offset(self, model_type, prefix):
        norm = self._norm(model_type, prefix)
        torch.testing.assert_close(_rmsnorm_scale(norm, norm.weight).float(), torch.full((D,), 1.0 + self.WEIGHT))

    @pytest.mark.parametrize("model_type, prefix", PLAIN_GEMMA_FAMILIES)
    def test_gemma_families_that_do_NOT_apply_the_offset(self, model_type, prefix):
        """These are the cases a `startswith("gemma")` rule gets wrong, so they are the regression."""
        norm = self._norm(model_type, prefix)
        torch.testing.assert_close(_rmsnorm_scale(norm, norm.weight).float(), torch.full((D,), self.WEIGHT))

    @pytest.mark.parametrize("model_type, prefix", OUTSIDE_FAMILIES)
    def test_families_outside_the_gemma_namespace_apply_weight_directly(self, model_type, prefix):
        norm = self._norm(model_type, prefix)
        torch.testing.assert_close(_rmsnorm_scale(norm, norm.weight).float(), torch.full((D,), self.WEIGHT))

    def test_a_family_no_table_could_know_about_needs_no_special_case(self):
        """Replaces a test that asserted a WARNING for an unrecognized gemma family.

        That warning existed because a table had to guess and the guess was wrong for some member of the namespace.
        Reading the scale from the module removes the guess, so there is nothing to announce: a family nobody has heard
        of is handled exactly like one everybody has.
        """
        import warnings

        class FutureRMSNorm(torch.nn.Module):
            """A plausible later variant applying `(1 + weight)` under a name no table carries."""

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.full((D,), TestRMSNormOffsetIsPerFamilyNotPerPrefix.WEIGHT))
                self.eps = 1e-6

            def forward(self, x):
                x = x.float()
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)

        norm = FutureRMSNorm()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scale = _rmsnorm_scale(norm, norm.weight)
        torch.testing.assert_close(scale.float(), torch.full((D,), 1.0 + self.WEIGHT))

    def test_the_scale_follows_the_MODULE_and_not_the_declared_family(self):
        """The strongest form of this class's claim, and it was not expressible against a table.

        One module, resolved under two different `model_type` names. A table answers from the name and
        would give two different scales; reading the module gives the same one both times, because the
        name was never the thing that decided it.
        """
        norm = self._norm("gemma3", "Gemma3")
        as_gemma3 = resolve_unembed_and_norm_scale(_hf_module("gemma3", norm))
        as_gemma4 = resolve_unembed_and_norm_scale(_hf_module("gemma4", norm))
        torch.testing.assert_close(as_gemma3.norm_scale, as_gemma4.norm_scale)
        torch.testing.assert_close(as_gemma3.norm_scale.float(), torch.full((D,), 1.0 + self.WEIGHT))
        assert as_gemma3.norm_kind == as_gemma4.norm_kind == "rmsnorm"

    def test_a_gemma_family_with_no_rmsnorm_does_not_reach_for_one(self):
        """A gemma-family model whose final norm is absent or of another kind passes through silently."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            info = resolve_unembed_and_norm_scale(_hf_module("gemma4_assistant", object()))
        assert info.norm_scale is None and info.norm_kind == "none"

    def test_a_layernorm_is_untouched_by_the_rmsnorm_rule(self):
        """The offset question is RMSNorm-only, and the probe does not apply: a constant vector centers to zero and
        a LayerNorm returns its bias.

        The declared path survives on that arm.
        """
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
        folded = fold_norm_into_unembed_rows(info, [c], apply_norm=True)[0]

        torch.testing.assert_close(folded @ x / rms, readout)

    def test_layernorm_folded_row_needs_the_centering_to_reproduce_the_readout(self):
        w_u = torch.randn(VOCAB, D)
        scale = torch.linspace(0.3, 2.7, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind="layernorm")
        x, c = self._x(), 5

        std = x.var(unbiased=False).sqrt()
        readout = w_u[c] @ (scale * (x - x.mean()) / std)
        folded = fold_norm_into_unembed_rows(info, [c], apply_norm=True)[0]
        torch.testing.assert_close(folded @ x / std, readout)

        # The positive control: the same row WITHOUT centering does not reproduce it, so the
        # assertion above is capable of failing when the projector is dropped.
        uncentered = w_u[c] * scale
        assert not torch.isclose(uncentered @ x / std, readout, atol=1e-4, rtol=1e-4)

    def test_rmsnorm_rows_are_not_centered(self):
        w_u = torch.randn(VOCAB, D)
        scale = torch.linspace(0.3, 2.7, D)
        rms_row = fold_norm_into_unembed_rows(
            UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind="rmsnorm"), [3], apply_norm=True
        )[0]
        ln_row = fold_norm_into_unembed_rows(
            UnembedNormInfo(w_u=w_u, norm_scale=scale, norm_kind="layernorm"), [3], apply_norm=True
        )[0]
        assert rms_row.mean().abs() > 1e-3, "an RMSNorm row must keep its uniform component"
        torch.testing.assert_close(ln_row.mean(), torch.zeros(()), atol=1e-6, rtol=0)

    def test_apply_norm_false_is_the_probing_shorthand(self):
        w_u = torch.randn(VOCAB, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=torch.linspace(0.3, 2.7, D), norm_kind="rmsnorm")
        torch.testing.assert_close(fold_norm_into_unembed_rows(info, [2, 7], apply_norm=False), w_u[[2, 7]])

    def test_absent_scale_returns_raw_rows_rather_than_guessing_one(self):
        w_u = torch.randn(VOCAB, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=None, norm_kind="none")
        torch.testing.assert_close(fold_norm_into_unembed_rows(info, [1], apply_norm=True), w_u[[1]])

    def test_rows_are_returned_per_id_in_order(self):
        w_u = torch.randn(VOCAB, D)
        info = UnembedNormInfo(w_u=w_u, norm_scale=None, norm_kind="none")
        rows = fold_norm_into_unembed_rows(info, [4, 1, 4], apply_norm=True)
        assert rows.shape == (3, D)
        torch.testing.assert_close(rows[0], rows[2])

    def test_empty_token_group_raises(self):
        info = UnembedNormInfo(w_u=torch.randn(VOCAB, D), norm_scale=None, norm_kind="none")
        with pytest.raises(ValueError, match="at least one token id"):
            fold_norm_into_unembed_rows(info, [], apply_norm=True)


class TestInnerModelSelectionNeverTruthTestsAModule:
    """The inner-model selection must not truth-test a module, in either failing direction.

    Both are silent-at-the-seam and loud somewhere else: an `or` chain skips a falsy-but-real submodule
    and resolves the wrong norm, or raises inside a wrapper whose `__len__` delegates to a module that
    has none. The first was latent for every zero-length container; the second took down a notebook test
    on the nnsight path when a published op collection began calling this seam.
    """

    def test_a_wrapper_whose_len_raises_does_not_break_selection(self):
        """The nnsight `Envoy` shape: `__len__` delegates to a module that has no length.

        Fails against a truthiness-based selection with `TypeError: ... has no len()`, raised by the
        `or` itself rather than by anything this seam meant to do.
        """
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        class RaisesOnLen(torch.nn.Module):
            def __init__(self, norm):
                super().__init__()
                self.norm = norm

            def __len__(self):
                raise TypeError("object of type 'Gemma2Model' has no len()")

        norm = LlamaRMSNorm(D)
        norm.weight.data = torch.full((D,), 0.5)
        model = type("Model", (), {})()
        model.config = _Cfg("llama")
        model.lm_head = _Head()
        model.model = RaisesOnLen(norm)
        module = type("Module", (), {})()
        module.model = model

        info = resolve_unembed_and_norm_scale(module)
        assert info.norm_kind == "rmsnorm"
        torch.testing.assert_close(info.norm_scale, torch.full((D,), 0.5))

    def test_an_empty_container_submodule_is_not_skipped(self):
        """A real submodule that happens to be falsy must still be selected.

        `nn.Sequential` and `nn.ModuleList` define `__len__`, so an empty one is falsy. An `or` chain
        falls through to the next candidate and resolves a norm from the wrong object, with no error.
        """
        assert not bool(torch.nn.Sequential()), "premise: a zero-length container is falsy"

        class FalsyButReal(torch.nn.Module):
            """Length zero, like an empty container, while still carrying the norm.

            Subclassing `nn.Sequential` does NOT work here: assigning the norm registers it in
            `_modules`, so `len()` becomes 1 and the object is truthy. That version of this test passed
            against the unfixed code, which is the only reason it was caught.
            """

            def __len__(self):
                return 0

        inner = FalsyButReal()
        inner.norm = torch.nn.LayerNorm(D)
        inner.norm.weight.data = torch.full((D,), 0.25)
        assert not bool(inner), "premise: this stand-in is falsy"
        model = type("Model", (), {})()
        model.config = _Cfg("gpt2")
        model.lm_head = _Head()
        model.model = inner
        # a decoy on the fallback path: if selection truth-tests, it lands here instead
        model.transformer = type("Decoy", (), {})()
        model.transformer.ln_f = torch.nn.LayerNorm(D)
        model.transformer.ln_f.weight.data = torch.full((D,), 99.0)
        module = type("Module", (), {})()
        module.model = model

        info = resolve_unembed_and_norm_scale(module)
        torch.testing.assert_close(info.norm_scale, torch.full((D,), 0.25))
        assert float(info.norm_scale.max()) != 99.0, "selection fell through to the decoy"


class TestMultimodalTextDecoderNorm:
    """A multimodal checkpoint keeps its text decoder's final norm one level below the backbone.

    Gemma 3 at 4b and up is `Gemma3ForConditionalGeneration`, whose backbone (`model.model`) carries no `norm`
    of its own; the norm lives at `model.model.language_model.norm`. Missing it raised nothing: the model
    read as norm-less, `jlens_read` dropped the final norm's gain from its readout, and every norm-aware
    direction came back unfolded while still being labelled `jlens_norm_aware`.
    """

    WEIGHT = 0.5

    def test_norm_nested_under_the_text_decoder_is_found(self):
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

        norm = Gemma3RMSNorm(D)
        with torch.no_grad():
            norm.weight.fill_(self.WEIGHT)
        text_decoder = type("TextDecoder", (), {})()
        text_decoder.norm = norm
        backbone = type("Backbone", (), {})()
        backbone.language_model = text_decoder
        model = type("Model", (), {})()
        model.config = _Cfg("gemma3")
        model.lm_head = _Head()
        model.model = backbone
        module = type("Module", (), {})()
        module.model = model

        info = resolve_unembed_and_norm_scale(module)
        assert info.norm_kind == "rmsnorm"
        torch.testing.assert_close(info.norm_scale.float(), torch.full((D,), 1.0 + self.WEIGHT))

    def test_real_gemma3_conditional_generation_resolves_like_its_text_only_twin(self):
        """The real architecture, built tiny from a local config: no download, no gated weights."""
        from types import SimpleNamespace

        from transformers import Gemma3Config, Gemma3ForCausalLM, Gemma3ForConditionalGeneration

        text = dict(
            vocab_size=VOCAB,
            hidden_size=D,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        vision = dict(
            hidden_size=D,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
        )
        torch.manual_seed(0)
        multimodal = Gemma3ForConditionalGeneration(
            Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4)
        )
        text_only = Gemma3ForCausalLM(multimodal.config.get_text_config())
        assert getattr(multimodal.model, "norm", None) is None, "premise: the backbone itself carries no norm"
        with torch.no_grad():
            multimodal.model.language_model.norm.weight.fill_(self.WEIGHT)
            text_only.model.norm.weight.fill_(self.WEIGHT)

        mm_info = resolve_unembed_and_norm_scale(SimpleNamespace(model=multimodal))
        txt_info = resolve_unembed_and_norm_scale(SimpleNamespace(model=text_only))
        assert mm_info.norm_kind == txt_info.norm_kind == "rmsnorm"
        torch.testing.assert_close(mm_info.norm_scale.float(), txt_info.norm_scale.float())
        rows = fold_norm_into_unembed_rows(mm_info, [3], apply_norm=True)
        torch.testing.assert_close(rows, mm_info.w_u[[3]].float() * (1.0 + self.WEIGHT))


class TestTheBasisMustBeStated:
    """`apply_norm` is required, and the one construction is shared rather than rewritten.

    The signature carried `= True` while the docstring said no default is safe. An op took the basis
    from that default without passing the flag, two layers away, so it could not be asked for the other
    basis and recorded neither. Requiring the parameter makes the omission unrepresentable rather than
    merely wrong: a call site that does not state its basis does not run.
    """

    @staticmethod
    def _info():
        w_u = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        return UnembedNormInfo(w_u=w_u, norm_scale=torch.tensor([2.0, 0.5, 1.0]), norm_kind="rmsnorm")

    def test_omitting_the_basis_is_a_type_error_not_a_default(self):
        with pytest.raises(TypeError):
            fold_norm_into_unembed_rows(self._info(), [1])  # type: ignore[call-arg]

    @pytest.mark.parametrize("apply_norm,expected", [(True, "jlens_norm_aware"), (False, "jlens_paper")])
    def test_the_basis_has_a_name_to_record(self, apply_norm, expected):
        assert jlens_basis_name(apply_norm) == expected

    @pytest.mark.parametrize("apply_norm", [True, False])
    def test_the_shared_construction_matches_the_expression_it_replaced(self, apply_norm):
        """`jlens_direction_rows` is `fold(...) @ J`, which is what all three sites open-coded."""
        info, j = self._info(), torch.randn(3, 5)

        got = jlens_direction_rows(info, [0, 2], j, apply_norm=apply_norm)
        want = fold_norm_into_unembed_rows(info, [0, 2], apply_norm=apply_norm) @ j

        torch.testing.assert_close(got, want)

    def test_the_two_bases_actually_differ(self):
        """The complement: if they agreed, recording which one produced a result would be pointless.

        They coincide only when the scale is uniform, so this uses a non-uniform one -- otherwise the
        test would pass while asserting nothing about the distinction it exists to protect.
        """
        info, j = self._info(), torch.randn(3, 5)

        folded = jlens_direction_rows(info, [0, 2], j, apply_norm=True)
        raw = jlens_direction_rows(info, [0, 2], j, apply_norm=False)

        assert not torch.allclose(folded, raw), (
            "the folded and unfolded bases produced identical directions, so this fixture cannot "
            "distinguish them and the surrounding assertions prove nothing"
        )

    def test_a_group_mean_is_the_same_before_or_after_the_lens(self):
        """The collection averaged rows then composed; composing then averaging is the same map."""
        info, j = self._info(), torch.randn(3, 5)

        rows_first = fold_norm_into_unembed_rows(info, [0, 2, 3], apply_norm=True).mean(dim=0) @ j
        lens_first = jlens_direction_rows(info, [0, 2, 3], j, apply_norm=True).mean(dim=0)

        torch.testing.assert_close(rows_first, lens_first)
