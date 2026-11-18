"""Component maps as the vocabulary consumes them: venngram's documents, resolved for the vocabulary and bound to a
model where one exists.

The schema, its reader and its evolution policy are venngram's and tested there. What is pinned here is what
Interpretune adds on top: resolution through declared kinds, row applicability, stacks and contributions; binding an
optional row to the layers a model has; the refusals an unbound map gives instead of assuming a layer; and the one
lookup every consumer reads (venngram's registry, with a verified row for every supported architecture).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest import mock

import pytest
import yaml
from venngram.maps import ComponentMapModelMismatch, check_map_against_model, load_component_map
from venngram.maps.check import derive_rmsnorm_offset

from interpretune.analysis.points import bind_component_map, component_map_for, from_venngram, parse, resolve
from interpretune.analysis.points.resolution import TensorRef, Unresolvable
from tests.core.mapped_architectures import (
    EXPECTED_RMSNORM_OFFSET,
    QWEN3_5_FULL_ATTENTION,
    SUPPORTED_ARCHITECTURES,
    bound_map,
    tiny_model,
)

ROWS = {
    "embed": {"module": "model.embed", "kind": "embed"},
    "blocks.{i}": {"module": "model.layers.{i}", "kind": "block"},
    "blocks.{i}.ln1": {"module": "model.layers.{i}.input_layernorm", "kind": "norm"},
    "blocks.{i}.attn": {"module": "model.layers.{i}.self_attn", "kind": "attn"},
    "blocks.{i}.mlp": {"module": "model.layers.{i}.mlp", "kind": "mlp"},
    "ln_final": {"module": "model.norm", "kind": "norm"},
    "unembed": {"module": "lm_head", "kind": "unembed"},
}


def _doc(**overrides) -> dict:
    doc = {"schema_version": 3, "architecture": "ToyForCausalLM", "components": {k: dict(v) for k, v in ROWS.items()}}
    doc.update(overrides)
    return doc


def _load(tmp_path: Path, doc: dict, name: str = "map.yaml"):
    """A document read through venngram's reader, as the unbound map the vocabulary resolves through."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return from_venngram(load_component_map(path))


class _Tree:
    """A module tree from a nested dict; lists become indexable stacks."""

    def __init__(self, spec):
        for key, value in spec.items():
            if isinstance(value, dict):
                value = _Tree(value)
            elif isinstance(value, list):
                value = [_Tree(v) if isinstance(v, dict) else v for v in value]
            setattr(self, key, value)


def _model(n_layers=2, with_attn=lambda i: True):
    layers = [
        {"input_layernorm": {}, "mlp": {}, **({"self_attn": {}} if with_attn(i) else {})} for i in range(n_layers)
    ]
    return _Tree({"model": {"embed": {}, "layers": layers, "norm": {}}, "lm_head": {}})


class TestKindsAreData:
    def test_a_declared_kind_resolves_with_its_own_output_rule(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.mixer": {"module": "model.layers.{i}.mixer", "kind": "mixer"}}
        cmap = _load(tmp_path, _doc(kinds={"mixer": {"output": "tuple0", "sublayer": True}}, components=rows))
        res = resolve(parse("blocks.0.mixer.hook_out"), cmap)
        assert isinstance(res, TensorRef) and res.module_path == "model.layers.0.mixer" and res.tuple_output is True
        assert cmap.kinds["mixer"].sublayer is True and cmap.kinds["attn"].tuple_output is True


class TestRowApplicability:
    def test_a_layer_the_row_excludes_is_refused_by_name_not_resolved_to_a_plausible_path(self, tmp_path):
        """The hybrid-stack case: attention on odd layers, a mixer on even ones."""
        rows = {
            **ROWS,
            "blocks.{i}.attn": {"module": "model.layers.{i}.self_attn", "kind": "attn", "layers": "odd"},
            "blocks.{i}.mixer": {"module": "model.layers.{i}.mamba", "kind": "mixer", "layers": "even"},
        }
        cmap = _load(tmp_path, _doc(kinds={"mixer": {"sublayer": True}}, components=rows))
        odd = resolve(parse("blocks.1.attn.hook_out"), cmap)
        assert isinstance(odd, TensorRef) and odd.module_path == "model.layers.1.self_attn"
        even = resolve(parse("blocks.0.attn.hook_out"), cmap)
        assert isinstance(even, Unresolvable) and "covers odd layers, not layer 0" in even.reason
        assert isinstance(resolve(parse("blocks.0.mixer.hook_out"), cmap), TensorRef)


class TestOptionalRowsResolveOnlyAgainstAModel:
    """An optional row applies wherever its module exists, which depends on the model, so an unbound map refuses it
    by name and a bound one resolves it on exactly the layers that have it."""

    def _optional(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.attn": {**ROWS["blocks.{i}.attn"], "optional": True}}
        return _load(tmp_path, _doc(components=rows))

    def test_the_unbound_map_refuses_by_name_and_never_assumes_layer_0(self, tmp_path):
        cmap = self._optional(tmp_path)
        for layer in (0, 1):
            res = resolve(parse(f"blocks.{layer}.attn.hook_out"), cmap)
            assert isinstance(res, Unresolvable) and "resolves only against a model" in res.reason
        with pytest.raises(ValueError, match="resolves only against a model"):
            cmap.first_layer("attn")
        # a row that is not optional is unaffected
        assert isinstance(resolve(parse("blocks.0.mlp.hook_out"), cmap), TensorRef)

    def test_binding_gives_the_row_the_layers_its_module_was_found_on(self, tmp_path):
        bound = bind_component_map(self._optional(tmp_path), _model(n_layers=4, with_attn=lambda i: i % 2 == 1))
        assert bound.bound and not any(e.optional for e in bound.components.values())
        assert bound.components["blocks.{i}.attn"].layers.spec == (1, 3) and bound.first_layer("attn") == 1
        assert resolve(parse("blocks.3.attn.hook_out"), bound).module_path == "model.layers.3.self_attn"
        other = resolve(parse("blocks.2.attn.hook_out"), bound)
        assert isinstance(other, Unresolvable) and "layers [1, 3], not layer 2" in other.reason

    def test_two_depths_bind_two_coverages(self, tmp_path):
        """Coverage is the model's, not the architecture's: the same document binds differently at another
        depth."""
        cmap = self._optional(tmp_path)
        shallow = bind_component_map(cmap, _model(n_layers=2, with_attn=lambda i: i % 2 == 1))
        deep = bind_component_map(cmap, _model(n_layers=6, with_attn=lambda i: i % 2 == 1))
        assert shallow.components["blocks.{i}.attn"].layers.spec == (1,)
        assert deep.components["blocks.{i}.attn"].layers.spec == (1, 3, 5)
        assert shallow.cache_key() != deep.cache_key()

    def test_an_optional_row_found_on_no_layer_is_refused(self, tmp_path):
        with pytest.raises(ComponentMapModelMismatch, match="optional row's module .* exists on no layer"):
            bind_component_map(self._optional(tmp_path), _model(with_attn=lambda i: False))

    def test_a_bound_map_cannot_be_registered_as_the_architectures(self, tmp_path):
        from interpretune.analysis.points import register

        bound = bind_component_map(self._optional(tmp_path), _model())
        with pytest.raises(ValueError, match="is bound to a model; register the document"):
            register(bound)


class TestBindingChecksTheMapAgainstTheModel:
    def test_a_correct_map_binds(self, tmp_path):
        assert bind_component_map(_load(tmp_path, _doc()), _model()).bound

    def test_a_row_covering_a_layer_that_lacks_the_module_is_refused_naming_it(self, tmp_path):
        with pytest.raises(ComponentMapModelMismatch, match=r"(?s)1 problems.*blocks\.\{i\}\.attn at layer 0"):
            bind_component_map(_load(tmp_path, _doc()), _model(with_attn=lambda i: i == 1))

    def test_a_declared_derived_property_that_disagrees_with_the_model_is_refused(self, tmp_path):
        class OffsetNorm:
            def forward(self, x):
                return x * (1.0 + self.weight)

        model = _model()
        model.model.norm = OffsetNorm()
        cmap = _load(tmp_path, _doc(properties={"rmsnorm_offset": False}))
        with pytest.raises(ComponentMapModelMismatch, match="properties.rmsnorm_offset"):
            bind_component_map(cmap, model)


class TestStacks:
    def _two_stacks(self, tmp_path):
        rows = {
            **ROWS,
            "encoder.blocks.{i}": {"module": "encoder.block.{i}", "kind": "block"},
            "encoder.blocks.{i}.attn": {"module": "encoder.block.{i}.layer.0.SelfAttention", "kind": "attn"},
            "encoder.embed": {"module": "encoder.embed_tokens", "kind": "embed"},
        }
        return _load(tmp_path, _doc(stacks={"encoder": {"module": "encoder.block.{i}"}}, components=rows))

    def test_the_prefix_parses_and_round_trips(self):
        point = parse("encoder.blocks.3.attn.hook_out")
        assert point.stack == "encoder" and point.layer == 3 and point.canonical == "encoder.blocks.3.attn.hook_out"
        bare = parse("blocks.3.attn.hook_out")
        assert bare.stack is None and bare.canonical == "blocks.3.attn.hook_out"

    def test_a_semantic_name_takes_the_prefix_too(self):
        point = parse("encoder.blocks.0.hook_resid_pre")
        assert point.stack == "encoder" and point.component == ""

    def test_the_primary_stack_stays_bare_and_the_second_resolves_under_its_name(self, tmp_path):
        cmap = self._two_stacks(tmp_path)
        primary = resolve(parse("blocks.0.attn.hook_out"), cmap)
        encoder = resolve(parse("encoder.blocks.0.attn.hook_out"), cmap)
        assert isinstance(primary, TensorRef) and primary.module_path == "model.layers.0.self_attn"
        assert isinstance(encoder, TensorRef) and encoder.module_path == "encoder.block.0.layer.0.SelfAttention"
        assert cmap.block_components() == cmap.block_components("blocks")
        assert "attn" in cmap.block_components("encoder") and "encoder.embed" in cmap.global_components()

    def test_an_undeclared_stack_is_refused_by_name(self, tmp_path):
        cmap = self._two_stacks(tmp_path)
        res = resolve(parse("decoder.blocks.0.attn.hook_out"), cmap)
        assert isinstance(res, Unresolvable) and "declares no block stack 'decoder'" in res.reason


class TestTheMultimodalMaps:
    def test_gemma3_vision_tower_resolves_and_the_language_model_spelling_is_unchanged(self):
        cmap = component_map_for("Gemma3ForConditionalGeneration")
        assert "vision" in cmap.stacks
        vision = resolve(parse("vision.blocks.2.attn.hook_out"), cmap)
        text = resolve(parse("blocks.1.attn.hook_out"), cmap)
        assert isinstance(vision, TensorRef) and vision.module_path == "model.vision_tower.encoder.layers.2.self_attn"
        assert isinstance(text, TensorRef) and text.module_path == "model.language_model.layers.1.self_attn"
        projector = resolve(parse("projector.hook_out"), cmap)
        assert isinstance(projector, TensorRef) and cmap.kinds["projector"].tuple_output is False


class TestContributionsAreKeyedBySublayerKind:
    """Contributions route to a sublayer's post-norm by the kind's position in `properties.sublayers`, so a third
    sublayer kind needs no enum member and no resolver branch; the classic two route exactly as before."""

    def _three_sublayers(self, tmp_path, sandwich: bool):
        rows = {
            **ROWS,
            "blocks.{i}.cross_attn": {"module": "model.layers.{i}.cross_attn", "kind": "attn"},
            "blocks.{i}.ln2": {"module": "model.layers.{i}.ln2", "kind": "norm"},
            "blocks.{i}.ln3": {"module": "model.layers.{i}.ln3", "kind": "norm"},
        }
        if sandwich:
            rows.update(
                {
                    "blocks.{i}.ln1_post": {"module": "model.layers.{i}.ln1_post", "kind": "norm"},
                    "blocks.{i}.ln2_post": {"module": "model.layers.{i}.ln2_post", "kind": "norm"},
                    "blocks.{i}.ln3_post": {"module": "model.layers.{i}.ln3_post", "kind": "norm"},
                }
            )
        props = {"sandwich_norms": sandwich, "sublayers": ["attn", "cross_attn", "mlp"]}
        return _load(tmp_path, _doc(properties=props, components=rows))

    def test_the_four_points_parse_and_keep_every_existing_spelling(self):
        p = parse("blocks.2.hook_cross_attn_out")
        assert (p.component, p.slot.value, p.contribution) == ("cross_attn", "out", "cross_attn")
        assert parse("blocks.2.hook_cross_attn_in").component == "cross_attn"
        assert parse("blocks.2.hook_resid_after_1").component == "ln3"
        assert parse("blocks.2.hook_resid_after_0").component == parse("blocks.2.hook_resid_mid").component == "ln2"
        assert (
            parse("blocks.2.hook_attn_out").contribution == "attn"
            and parse("blocks.2.hook_mlp_out").contribution == "mlp"
        )

    def test_routing_follows_the_declared_order_on_a_sandwich_norm_block(self, tmp_path):
        cmap = self._three_sublayers(tmp_path, sandwich=True)
        attn = resolve(parse("blocks.0.hook_attn_out"), cmap)
        cross = resolve(parse("blocks.0.hook_cross_attn_out"), cmap)
        mlp = resolve(parse("blocks.0.hook_mlp_out"), cmap)
        assert [r.module_path for r in (attn, cross, mlp)] == [
            "model.layers.0.ln1_post",
            "model.layers.0.ln2_post",
            "model.layers.0.ln3_post",
        ]

    def test_a_pre_norm_block_reads_the_raw_sublayer_output(self, tmp_path):
        cmap = self._three_sublayers(tmp_path, sandwich=False)
        cross = resolve(parse("blocks.0.hook_cross_attn_out"), cmap)
        assert isinstance(cross, TensorRef) and cross.module_path == "model.layers.0.cross_attn"
        after = resolve(parse("blocks.0.hook_resid_after_1"), cmap)
        assert isinstance(after, TensorRef) and after.module_path == "model.layers.0.ln3" and after.io == "input"

    def test_the_classic_two_route_exactly_as_before(self):
        gemma = component_map_for("Gemma2ForCausalLM")
        assert resolve(parse("blocks.3.hook_attn_out"), gemma).module_path.endswith("post_attention_layernorm")
        assert resolve(parse("blocks.3.hook_mlp_out"), gemma).module_path.endswith("post_feedforward_layernorm")
        gpt2 = component_map_for("GPT2LMHeadModel")
        assert resolve(parse("blocks.3.hook_mlp_out"), gpt2).module_path == "transformer.h.3.mlp"

    def test_a_declared_kind_can_be_a_contribution_without_any_enum_change(self, tmp_path):
        rows = {
            **ROWS,
            "blocks.{i}.mixer": {"module": "model.layers.{i}.mixer", "kind": "mixer"},
            "blocks.{i}.ln1_post": {"module": "model.layers.{i}.ln1_post", "kind": "norm"},
        }
        cmap = _load(
            tmp_path,
            _doc(
                kinds={"mixer": {"sublayer": True}},
                properties={"sandwich_norms": True, "sublayers": ["mixer", "mlp"]},
                components=rows,
            ),
        )
        from interpretune.analysis.points.vocabulary import ActivationPoint, Slot

        point = ActivationPoint("mixer", Slot.OUT, 0, "mixer")
        assert resolve(point, cmap).module_path == "model.layers.0.ln1_post"


class TestSpellingsUnderAStack:
    def test_a_stacked_name_keeps_its_prefix_in_every_spelling(self):
        from interpretune.analysis.points.vocabulary import spellings

        out = spellings("vision.blocks.2.attn.hook_out")
        assert "vision.blocks.2.attn.hook_out" in out
        assert all(s.startswith("vision.blocks.2.") for s in out), out
        bare = spellings("blocks.2.attn.hook_out")
        assert all(s.startswith("blocks.2.") for s in bare), bare


class TestQwen3_5HybridAttention:
    """Qwen3.5 alternates three linear-attention layers with one full-attention layer, so its attention has two
    modules, and which layers have which depends on the checkpoint's depth: the map marks both optional.

    Bound to a model, each layer type resolves to its own module and the other is refused by name, never mapped.
    """

    FULL, LINEAR = 7, 6

    @pytest.fixture(scope="class")
    def cmap(self):
        return bound_map("Qwen3_5ForConditionalGeneration")

    def test_binding_finds_the_full_attention_layers(self, cmap):
        assert cmap.components["blocks.{i}.attn"].layers.spec == QWEN3_5_FULL_ATTENTION
        linear = cmap.components["blocks.{i}.linear_attn"].layers.spec
        assert set(linear) == set(range(32)) - set(QWEN3_5_FULL_ATTENTION)

    def test_each_attention_kind_resolves_on_its_own_layers(self, cmap):
        full = resolve(parse(f"blocks.{self.FULL}.attn.hook_out"), cmap)
        assert isinstance(full, TensorRef) and full.module_path == f"model.language_model.layers.{self.FULL}.self_attn"
        assert full.tuple_output is True
        lin = resolve(parse(f"blocks.{self.LINEAR}.linear_attn.hook_out"), cmap)
        assert isinstance(lin, TensorRef)
        assert lin.module_path == f"model.language_model.layers.{self.LINEAR}.linear_attn"
        assert lin.tuple_output is False

    def test_the_other_kind_is_refused_by_name_on_each_layer(self, cmap):
        no_attn = resolve(parse(f"blocks.{self.LINEAR}.attn.hook_out"), cmap)
        assert isinstance(no_attn, Unresolvable) and f"not layer {self.LINEAR}" in no_attn.reason
        no_lin = resolve(parse(f"blocks.{self.FULL}.linear_attn.out.hook_out"), cmap)
        assert isinstance(no_lin, Unresolvable) and f"not layer {self.FULL}" in no_lin.reason

    def test_unbound_the_attention_is_refused_as_needing_a_model(self):
        cmap = component_map_for("Qwen3_5ForConditionalGeneration")
        res = resolve(parse(f"blocks.{self.FULL}.attn.hook_out"), cmap)
        assert isinstance(res, Unresolvable) and "resolves only against a model" in res.reason

    def test_the_old_output_projection_name_is_refused_naming_the_registry_one(self, cmap):
        """`linear_attn.o` became `linear_attn.out`, the registry's row name; the old spelling has no alias."""
        res = resolve(parse(f"blocks.{self.LINEAR}.linear_attn.o.hook_out"), cmap)
        assert isinstance(res, Unresolvable) and "named 'linear_attn.out'" in res.reason
        new = resolve(parse(f"blocks.{self.LINEAR}.linear_attn.out.hook_out"), cmap)
        assert new.module_path == f"model.language_model.layers.{self.LINEAR}.linear_attn.out_proj"

    @pytest.mark.parametrize("layer", [FULL, LINEAR])
    def test_the_residual_points_hold_on_both_layer_types(self, cmap, layer):
        mid = resolve(parse(f"blocks.{layer}.hook_resid_mid"), cmap)
        assert mid.module_path == f"model.language_model.layers.{layer}.post_attention_layernorm" and mid.io == "input"
        post = resolve(parse(f"blocks.{layer}.hook_resid_post"), cmap)
        assert post.module_path == f"model.language_model.layers.{layer}" and post.io == "output"

    def test_the_final_norm_and_unembed_resolve(self, cmap):
        assert resolve(parse("ln_final.hook_out"), cmap).module_path == "model.language_model.norm"
        assert resolve(parse("unembed.hook_out"), cmap).module_path == "lm_head"

    def test_the_vision_tower_resolves(self, cmap):
        res = resolve(parse("vision.blocks.0.attn.hook_out"), cmap)
        assert isinstance(res, TensorRef) and res.module_path == "model.visual.blocks.0.attn"

    def test_the_seam_reads_the_final_norms_applied_scale(self):
        """The model's own norm, nested under `language_model`, probed for the scale it applies: `1 + weight`."""
        torch = pytest.importorskip("torch")
        tr = pytest.importorskip("transformers")
        from interpretune.analysis.optools import resolve_unembed_and_norm_scale

        small = dict(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=2, vocab_size=64)
        model = tr.Qwen3_5ForConditionalGeneration(
            tr.Qwen3_5Config(
                text_config=dict(
                    num_hidden_layers=4,
                    head_dim=8,
                    linear_num_key_heads=2,
                    linear_num_value_heads=2,
                    linear_key_head_dim=8,
                    linear_value_head_dim=8,
                    **small,
                ),
                vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16),
            )
        )
        with torch.no_grad():
            model.model.language_model.norm.weight.fill_(0.5)
        info = resolve_unembed_and_norm_scale(type("Module", (), {"model": model})())
        assert info.norm_kind == "rmsnorm"
        torch.testing.assert_close(info.norm_scale.float(), torch.full((16,), 1.5))


class TestEverySupportedArchitectureIsServed:
    """One lookup, venngram's: every supported architecture has a verified registry row at the CI warm generation,
    so a registry regression fails here by name rather than falling back to a copy nobody checks."""

    @pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
    def test_a_verified_row_serves_the_architecture(self, architecture):
        from venngram.maps import registered_architectures
        from venngram.registry import rows

        # an in-process registration would win over the registry and mask a missing row, so there must be none
        assert architecture not in registered_architectures()
        statuses = [row.status for row in rows("component_map", architecture)]
        assert "verified" in statuses, f"{architecture}: registry rows {statuses}, none verified"
        assert component_map_for(architecture, strict=True).architecture == architecture

    def test_an_unknown_architecture_is_refused_naming_the_lookup(self):
        with pytest.raises(KeyError, match="no component map for architecture 'NoSuchForCausalLM'"):
            component_map_for("NoSuchForCausalLM")

    def test_strict_lookup_refuses_a_retired_map(self, tmp_path):
        from venngram.maps import register, unregister
        from venngram.maps.schema import ComponentMapDeprecationWarning

        path = tmp_path / "retired.yaml"
        path.write_text(yaml.safe_dump(_doc(architecture="RetiredToyModel", deprecated_since="0.9.0")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComponentMapDeprecationWarning)
            register(load_component_map(path))
        try:
            assert component_map_for("RetiredToyModel").deprecated_since == "0.9.0"
            with pytest.raises(ValueError, match="deprecated since 0.9.0"):
                component_map_for("RetiredToyModel", strict=True)
        finally:
            unregister("RetiredToyModel")


class TestEverySupportedMapDescribesItsModel:
    """The entry gate for the supported set: each map is bound to its transformers class on every run."""

    @pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
    def test_the_map_binds_to_the_class(self, architecture):
        cmap = component_map_for(architecture)
        model = tiny_model(architecture)
        assert check_map_against_model(cmap, model) == []
        assert bind_component_map(cmap, model).bound
        assert derive_rmsnorm_offset(model, cmap) is EXPECTED_RMSNORM_OFFSET[architecture]


def _tl_adapter(architecture: str):
    """TransformerLens' own architecture adapter for a supported architecture, built from the tiny config alone.

    No weights and no Hub call: the adapter is selected from a bridge config mapped off the HF config, the same
    way the bridge's boot path selects it, so the oracle covers every supported map at the cost of a config.
    """
    pytest.importorskip("transformer_lens")
    import transformer_lens.model_bridge  # noqa: F401  (the factory module imports it, circularly, if it is first)
    from transformer_lens.config import TransformerBridgeConfig
    from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
    from transformer_lens.model_bridge.sources.transformers import map_default_transformer_lens_config

    hf_config = tiny_model(architecture).config
    cfg = TransformerBridgeConfig.from_dict(dict(map_default_transformer_lens_config(hf_config).__dict__))
    cfg.architecture = architecture
    for nested in ("text_config", "vision_config"):
        if hasattr(hf_config, nested):
            setattr(cfg, nested, getattr(hf_config, nested))
    if architecture not in TL_LIBRARY_CHECK_NOT_NEEDED:
        return ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter

    with mock.patch.object(ArchitectureAdapter, "_check_required_libraries", lambda self: None):
        return ArchitectureAdapterFactory.select_architecture_adapter(cfg)


#: Architectures whose TransformerLens adapter refuses to construct without an optional library that its component
#: mapping does not use, with the reason. The check is suppressed for those alone, so the oracle reads the real
#: adapter instead of being skipped in every environment that lacks the library (the pinned and CI ones do).
TL_LIBRARY_CHECK_NOT_NEEDED = {
    "Qwen3_5ForConditionalGeneration": "torchvision, for TransformerLens' image processor at run time",
}

_GATED_DELTA_NET = {
    "blocks.{i}.linear_attn.conv",
    "blocks.{i}.linear_attn.in_a",
    "blocks.{i}.linear_attn.in_b",
    "blocks.{i}.linear_attn.in_qkv",
    "blocks.{i}.linear_attn.in_z",
    "blocks.{i}.linear_attn.norm",
    "blocks.{i}.linear_attn.out",
}

#: Registry rows that have no TransformerLens counterpart, by architecture, each with the reason it is expected.
#: TransformerLens bridges a vision tower as one opaque component, so the multimodal maps' vision stacks and projectors
#: are the registry's alone, and its gated delta net bridge exposes none of the linear attention's submodules. Every
#: other row must have a counterpart: a row nothing independent vouches for is the drift this oracle exists to catch,
#: so a new unmatched row fails by name rather than being allowed.
EXPECTED_WITHOUT_TL_COUNTERPART = {
    "GPT2LMHeadModel": set(),
    "LlamaForCausalLM": set(),
    "Gemma2ForCausalLM": set(),
    "Gemma3ForCausalLM": set(),
    "Gemma3ForConditionalGeneration": {
        "projector",
        "vision.blocks.{i}",
        "vision.blocks.{i}.attn",
        "vision.blocks.{i}.attn.k",
        "vision.blocks.{i}.attn.o",
        "vision.blocks.{i}.attn.q",
        "vision.blocks.{i}.attn.v",
        "vision.blocks.{i}.ln1",
        "vision.blocks.{i}.ln2",
        "vision.blocks.{i}.mlp",
        "vision.blocks.{i}.mlp.in",
        "vision.blocks.{i}.mlp.out",
        "vision.embed",
        "vision.ln_final",
    },
    "Qwen3_5ForCausalLM": _GATED_DELTA_NET,
    "Qwen3_5ForConditionalGeneration": _GATED_DELTA_NET
    | {
        "projector",
        "vision.blocks.{i}",
        "vision.blocks.{i}.attn",
        "vision.blocks.{i}.attn.o",
        "vision.blocks.{i}.attn.qkv",
        "vision.blocks.{i}.ln1",
        "vision.blocks.{i}.ln2",
        "vision.blocks.{i}.mlp",
        "vision.blocks.{i}.mlp.in",
        "vision.blocks.{i}.mlp.out",
        "vision.embed",
        "vision.pos_embed",
    },
}


class TestEverySupportedMapAgreesWithTransformerLens:
    """The independent oracle for the registry's maps: TransformerLens' own per-architecture component mapping.

    A convergence case that resolves both sides through the same component map cannot catch a wrong row, because both
    sides move together; the nnsight backend's hook names derive from these maps, so its HF-reference cases have exactly
    that shape. This test compares each map with a source that shares nothing with it. The row counts are asserted, so
    the oracle's own coverage is a measured number rather than a belief.
    """

    @pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
    def test_every_shared_row_names_the_same_module_and_kind(self, architecture):
        from venngram.maps.transformer_lens import from_transformer_lens

        served = component_map_for(architecture)
        derived = from_transformer_lens(_tl_adapter(architecture), architecture)
        shared = sorted(set(served.components) & set(derived.components))
        # module and kind only: TransformerLens has no per-row layer predicate (it marks a hybrid stack's attention
        # bridges optional), so a row's coverage is vouched for by binding it to the class, not here
        mismatches = [
            (k, served.components[k], derived.components[k])
            for k in shared
            if (served.components[k].module, served.components[k].kind)
            != (derived.components[k].module, derived.components[k].kind)
        ]
        assert not mismatches, mismatches
        expected_shared = set(served.components) - EXPECTED_WITHOUT_TL_COUNTERPART[architecture]
        assert set(shared) == expected_shared, (
            f"{architecture}: TransformerLens vouches for {len(shared)} of {len(expected_shared)} rows; "
            f"uncompared: {sorted(expected_shared - set(shared))}"
        )
        assert served.properties.get("sandwich_norms", False) == derived.properties["sandwich_norms"]

    @pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
    def test_the_map_carries_nothing_unvouched_for(self, architecture):
        """A row with no TransformerLens counterpart is expected only where the table above says why."""
        from venngram.maps.transformer_lens import from_transformer_lens

        derived = from_transformer_lens(_tl_adapter(architecture), architecture)
        extra = set(component_map_for(architecture).components) - set(derived.components)
        assert extra == EXPECTED_WITHOUT_TL_COUNTERPART[architecture], sorted(
            extra ^ EXPECTED_WITHOUT_TL_COUNTERPART[architecture]
        )

    def test_every_supported_architecture_has_an_expectation_here(self):
        assert set(SUPPORTED_ARCHITECTURES) == set(EXPECTED_WITHOUT_TL_COUNTERPART)
