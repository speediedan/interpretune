"""The activation-point vocabulary: parse, resolve, and the properties the five tables used to get wrong."""

from __future__ import annotations

import pytest

from interpretune.analysis.points import (
    ActivationPoint,
    Contribution,
    Slot,
    TensorRef,
    UnknownPointError,
    Unresolvable,
    component_map_for,
    known_architectures,
    parse,
    resolve,
)


class TestParse:
    def test_component_spelling_round_trips(self):
        p = parse("blocks.5.ln2.hook_out")
        assert p == ActivationPoint("ln2", Slot.OUT, 5)
        assert p.canonical == "blocks.5.ln2.hook_out"

    def test_semantic_names_are_the_same_record_as_their_component_spelling(self):
        assert parse("blocks.5.hook_resid_pre") == parse("blocks.5.hook_in")
        assert parse("blocks.5.hook_resid_post") == parse("blocks.5.hook_out")
        assert parse("blocks.5.hook_resid_mid") == parse("blocks.5.ln2.hook_in")
        assert parse("blocks.5.attn.hook_z") == parse("blocks.5.attn.o.hook_in")

    def test_contributions_are_semantic_not_component(self):
        p = parse("blocks.5.hook_mlp_out")
        assert p.contribution is Contribution.MLP and p.component == "mlp"
        assert parse("blocks.5.mlp.hook_out").contribution is None

    def test_pre_norm_legacy_inputs_carry_a_caution(self):
        p = parse("blocks.5.hook_mlp_in")
        assert p == ActivationPoint("ln2", Slot.IN, 5, caution=p.caution) and "BEFORE the block norm" in p.caution

    def test_subhook_suffix_is_kept(self):
        p = parse("blocks.5.hook_resid_post.hook_sae_acts_post")
        assert p.subhook == "hook_sae_acts_post" and p.canonical == "blocks.5.hook_out.hook_sae_acts_post"

    def test_globals(self):
        assert parse("unembed.hook_out") == ActivationPoint("unembed", Slot.OUT, None)
        assert parse("hook_embed") == parse("embed.hook_out")

    @pytest.mark.parametrize("bad", ["blocks.x.hook_in", "blocks.5.", "blocks.5.mlp.hook_sideways", "hook_normalized"])
    def test_unknown_spellings_are_refused_by_name(self, bad):
        with pytest.raises(UnknownPointError):
            parse(bad)


class TestResolve:
    @pytest.fixture(params=known_architectures())
    def cmap(self, request):
        return component_map_for(request.param)

    def test_block_in_and_out_resolve_on_every_architecture(self, cmap):
        for name in ("blocks.3.hook_in", "blocks.3.hook_out", "unembed.hook_in", "unembed.hook_out"):
            res = resolve(parse(name), cmap)
            assert isinstance(res, TensorRef), (cmap.architecture, name, res)

    def test_the_output_distribution_is_addressable(self, cmap):
        res = resolve(parse("unembed.hook_out"), cmap)
        assert isinstance(res, TensorRef) and res.io == "output" and res.module_path.endswith("lm_head")

    def test_a_norm_is_three_tensors(self, cmap):
        i, n, o = (resolve(parse(f"blocks.2.ln2.hook_{s}"), cmap) for s in ("in", "normalized", "out"))
        assert isinstance(i, TensorRef) and i.io == "input" and not i.derived
        assert isinstance(o, TensorRef) and o.io == "output" and not o.derived
        assert isinstance(n, TensorRef) and n.derivation == "pre_gain_normalized"

    def test_contribution_is_post_norm_only_on_sandwich_norm_models(self, cmap):
        attn = resolve(parse("blocks.4.hook_attn_out"), cmap)
        mlp = resolve(parse("blocks.4.hook_mlp_out"), cmap)
        assert isinstance(attn, TensorRef) and isinstance(mlp, TensorRef)
        if cmap.sandwich_norms:
            assert attn.module_path.endswith("post_attention_layernorm")
            assert mlp.module_path.endswith("post_feedforward_layernorm")
        else:
            assert attn.module_path == cmap.module_for("attn", 4)
            assert mlp.module_path == cmap.module_for("mlp", 4)

    def test_unknown_component_is_a_value_with_a_reason(self, cmap):
        res = resolve(parse("blocks.1.router.hook_out"), cmap)
        assert isinstance(res, Unresolvable) and "router" in res.reason and res.alternatives

    def test_mlp_pre_activation_is_the_projection_output(self, cmap):
        """TransformerLens defines `mlp.hook_pre` as `mlp.in.hook_out`, the pre-activation, not the MLP's input."""
        res = resolve(parse("blocks.0.mlp.hook_pre"), cmap)
        assert isinstance(res, TensorRef) and res.io == "output" and res.module_path == cmap.module_for("mlp.in", 0)


class TestBundledMapAgreesWithTransformerLens:
    """The independent oracle for the bundled documents: TransformerLens' own per-architecture component mapping.

    A convergence case that resolves both sides through the same resolver cannot catch a wrong row (both sides
    move together), and the cross-backend parity module never touches the resolver. This does: every component
    both sources carry must name the same module with the same kind.
    """

    @pytest.fixture(scope="class")
    def tl_gpt2_map(self):
        pytest.importorskip("transformer_lens")
        from transformer_lens.model_bridge import TransformerBridge

        from interpretune.analysis.points import from_transformer_lens

        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        return from_transformer_lens(bridge.adapter, "GPT2LMHeadModel")

    def test_every_shared_component_names_the_same_module_and_kind(self, tl_gpt2_map):
        bundled = component_map_for("GPT2LMHeadModel")
        shared = sorted(set(bundled.components) & set(tl_gpt2_map.components))
        assert len(shared) >= 10, f"too few shared components to be a meaningful check: {shared}"
        mismatches = [
            (k, bundled.components[k], tl_gpt2_map.components[k])
            for k in shared
            if bundled.components[k] != tl_gpt2_map.components[k]
        ]
        assert not mismatches, mismatches

    def test_the_bundled_map_carries_nothing_transformer_lens_does_not(self, tl_gpt2_map):
        """A bundled row with no TL counterpart is a row nothing independent vouches for; there should be none on
        gpt2."""
        extra = sorted(set(component_map_for("GPT2LMHeadModel").components) - set(tl_gpt2_map.components))
        assert not extra, extra


class TestResolverLayerShape:
    """The resolver refuses a layer on a global point and a missing layer on a block point.

    Found by the first hub
    adapter enumerating the vocabulary by probing: the lenient forms produced nonsense paths that grew with the
    vocabulary until every broad filter selected one and was refused.
    """

    def test_a_block_point_needs_a_layer(self):
        from interpretune.analysis.backends.hook_mapping import HookNameResolver

        with pytest.raises(ValueError, match="needs a layer"):
            HookNameResolver("GPT2LMHeadModel").resolve("attn.hook_z")

    def test_a_global_point_takes_no_layer(self):
        from interpretune.analysis.backends.hook_mapping import HookNameResolver

        with pytest.raises(ValueError, match="takes no layer"):
            HookNameResolver("GPT2LMHeadModel").resolve("blocks.5.unembed.hook_out")

    def test_the_right_shapes_still_resolve(self):
        from interpretune.analysis.backends.hook_mapping import HookNameResolver

        r = HookNameResolver("GPT2LMHeadModel")
        assert r.resolve("blocks.5.attn.hook_z") == ("transformer.h.5.attn.c_proj", "input")
        assert r.resolve("unembed.hook_out") == ("lm_head", "output")
