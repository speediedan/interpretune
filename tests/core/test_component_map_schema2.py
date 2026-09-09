"""Schema 2 of the component map: kinds as data, row applicability, additional stacks, typed properties, and a
load-time check of the document against its model. Every case here is a decision from the architecture spike, pinned."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from interpretune.analysis.points import parse, resolve
from interpretune.analysis.points.component_map import (
    COMPONENT_MAP_SCHEMA_VERSION,
    LayerSet,
    component_map_for,
    load_component_map_file,
)
from interpretune.analysis.points.resolution import TensorRef, Unresolvable
from interpretune.analysis.points.validation import (
    ComponentMapModelMismatch,
    check_map_against_model,
    derive_rmsnorm_offset,
    validate_map_against_model,
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
    doc = {"schema_version": 2, "architecture": "ToyForCausalLM", "components": {k: dict(v) for k, v in ROWS.items()}}
    doc.update(overrides)
    return doc


def _load(tmp_path: Path, doc: dict, name: str = "map.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return load_component_map_file(path)


class TestKindsAreData:
    def test_a_declared_kind_resolves_with_its_own_output_rule(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.mixer": {"module": "model.layers.{i}.mixer", "kind": "mixer"}}
        cmap = _load(tmp_path, _doc(kinds={"mixer": {"output": "tuple0", "sublayer": True}}, components=rows))
        res = resolve(parse("blocks.0.mixer.hook_out"), cmap)
        assert isinstance(res, TensorRef) and res.module_path == "model.layers.0.mixer" and res.tuple_output is True
        assert cmap.kinds["mixer"].sublayer is True and cmap.kinds["attn"].tuple_output is True

    def test_an_undeclared_kind_is_refused_and_names_the_declaration(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.mixer": {"module": "model.layers.{i}.mixer", "kind": "mixer"}}
        with pytest.raises(ValueError, match="unknown component kind 'mixer'.*declare a new kind under `kinds:`"):
            _load(tmp_path, _doc(components=rows))

    def test_a_kind_with_an_unknown_output_rule_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="expected 'tensor' or 'tuple0'"):
            _load(tmp_path, _doc(kinds={"mixer": {"output": "list"}}))

    def test_a_schema_1_document_cannot_declare_kinds(self, tmp_path):
        with pytest.raises(ValueError, match="`kinds` is a schema-2 key"):
            _load(tmp_path, _doc(schema_version=1, kinds={"mixer": {}}))


class TestRowApplicability:
    @pytest.mark.parametrize(
        "spec, covered, excluded",
        [([0, 2], (0, 2), (1, 3)), ("1-2", (1, 2), (0, 3)), ("odd", (1, 3), (0, 2)), ("even", (0, 2), (1, 3))],
    )
    def test_the_three_forms(self, spec, covered, excluded):
        layers = LayerSet.parse(spec, "t")
        assert all(layers.covers(i) for i in covered) and not any(layers.covers(i) for i in excluded)

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

    def test_a_malformed_predicate_is_refused(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.attn": {**ROWS["blocks.{i}.attn"], "layers": "some"}}
        with pytest.raises(ValueError, match="`layers` must be"):
            _load(tmp_path, _doc(components=rows))

    def test_a_schema_1_document_cannot_carry_a_predicate(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.attn": {**ROWS["blocks.{i}.attn"], "layers": "odd"}}
        with pytest.raises(ValueError, match="`layers` is a schema-2 key"):
            _load(tmp_path, _doc(schema_version=1, components=rows))


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

    def test_a_row_naming_an_undeclared_stack_is_refused_at_load(self, tmp_path):
        rows = {**ROWS, "vision.blocks.{i}": {"module": "v.layers.{i}", "kind": "block"}}
        with pytest.raises(ValueError, match="names the block stack 'vision', which the document does not declare"):
            _load(tmp_path, _doc(components=rows))

    def test_the_primary_name_cannot_be_declared_and_a_template_needs_the_index(self, tmp_path):
        with pytest.raises(ValueError, match="not a valid stack name"):
            _load(tmp_path, _doc(stacks={"blocks": {"module": "x.{i}"}}))
        with pytest.raises(ValueError, match="needs a `module` template containing"):
            _load(tmp_path, _doc(stacks={"vision": {"module": "model.vision_tower"}}))


class TestProperties:
    def test_sublayers_is_declared_and_typed(self, tmp_path):
        cmap = _load(tmp_path, _doc(properties={"sublayers": ["mixer"]}, kinds={"mixer": {"sublayer": True}}))
        assert cmap.sublayers == ("mixer",)
        assert _load(tmp_path, _doc()).sublayers == ("attn", "mlp")
        with pytest.raises(ValueError, match="'sublayers' must be a list of kind names"):
            _load(tmp_path, _doc(properties={"sublayers": [1]}))


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


class TestTheMapIsCheckedAgainstTheModel:
    def test_a_correct_map_has_no_problems(self, tmp_path):
        assert check_map_against_model(_load(tmp_path, _doc()), _model()) == []

    def test_a_row_covering_a_layer_that_lacks_the_module_is_named(self, tmp_path):
        problems = check_map_against_model(_load(tmp_path, _doc()), _model(with_attn=lambda i: i % 2 == 1))
        assert [(p.row, p.layer) for p in problems] == [("blocks.{i}.attn", 0)]
        assert "does not exist here" in problems[0].reason

    def test_a_row_excluding_a_layer_that_has_the_module_is_named(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.attn": {**ROWS["blocks.{i}.attn"], "layers": "odd"}}
        problems = check_map_against_model(_load(tmp_path, _doc(components=rows)), _model())
        assert [(p.row, p.layer) for p in problems] == [("blocks.{i}.attn", 0)]
        assert "excludes this layer" in problems[0].reason

    def test_the_predicate_that_matches_the_model_passes(self, tmp_path):
        rows = {**ROWS, "blocks.{i}.attn": {**ROWS["blocks.{i}.attn"], "layers": "odd"}}
        assert check_map_against_model(_load(tmp_path, _doc(components=rows)), _model(with_attn=lambda i: i % 2)) == []

    def test_validate_raises_listing_every_problem(self, tmp_path):
        with pytest.raises(ComponentMapModelMismatch, match=r"(?s)1 problems.*blocks\.\{i\}\.attn at layer 0"):
            validate_map_against_model(_load(tmp_path, _doc()), _model(with_attn=lambda i: i == 1))

    def test_a_declared_derived_property_that_disagrees_with_the_model_is_refused(self, tmp_path):
        class OffsetNorm:
            def forward(self, x):
                return x * (1.0 + self.weight)

        class PlainNorm:
            def forward(self, x):
                return x * self.weight

        model = _model()
        model.model.norm = OffsetNorm()
        cmap = _load(tmp_path, _doc(properties={"rmsnorm_offset": False}))
        assert derive_rmsnorm_offset(model, cmap) is True
        problems = check_map_against_model(cmap, model)
        assert [p.row for p in problems] == ["properties.rmsnorm_offset"]
        model.model.norm = PlainNorm()
        assert check_map_against_model(cmap, model) == []


class TestTheBundledMultimodalMapUnderSchema2:
    def test_the_vision_tower_resolves_and_the_language_model_spelling_is_unchanged(self):
        cmap = component_map_for("Gemma3ForConditionalGeneration")
        assert cmap.schema_version == COMPONENT_MAP_SCHEMA_VERSION and "vision" in cmap.stacks
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


def _tiny(architecture: str):
    """The transformers class for a bundled architecture, instantiated tiny on the meta device."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    small = dict(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=2, vocab_size=64)
    builders = {
        "GPT2LMHeadModel": lambda: tr.GPT2LMHeadModel(tr.GPT2Config(n_layer=3, n_embd=16, n_head=2, vocab_size=64)),
        "LlamaForCausalLM": lambda: tr.LlamaForCausalLM(tr.LlamaConfig(num_hidden_layers=3, **small)),
        "Gemma2ForCausalLM": lambda: tr.Gemma2ForCausalLM(tr.Gemma2Config(num_hidden_layers=3, head_dim=8, **small)),
        "Gemma3ForCausalLM": lambda: tr.Gemma3ForCausalLM(
            tr.Gemma3TextConfig(num_hidden_layers=3, head_dim=8, **small)
        ),
        "Gemma3ForConditionalGeneration": lambda: tr.Gemma3ForConditionalGeneration(
            tr.Gemma3Config(
                text_config=dict(num_hidden_layers=2, head_dim=8, **small),
                vision_config=dict(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    image_size=16,
                    patch_size=8,
                ),
            )
        ),
    }
    if architecture not in builders:
        pytest.fail(
            f"{architecture} is bundled but has no tiny builder here: entering the bundled set means adding one, so "
            "the document is checked against its class on every run (docs/activation_point_vocabulary.md, "
            "'Entering and leaving the bundled set')."
        )
    with torch.device("meta"):
        return builders[architecture]()


EXPECTED_RMSNORM_OFFSET = {
    "GPT2LMHeadModel": False,
    "LlamaForCausalLM": False,
    "Gemma2ForCausalLM": True,
    "Gemma3ForCausalLM": True,
    "Gemma3ForConditionalGeneration": True,
}


class TestEveryBundledMapDescribesItsModel:
    """The entry gate for the bundled set: a document is checked against its transformers class on every run.

    Before this test the claim "every bundled map describes its model" rested on one ad hoc meta-device run, which
    is indistinguishable from nobody having checked to anyone reading the suite.
    """

    @pytest.mark.parametrize("architecture", sorted(EXPECTED_RMSNORM_OFFSET))
    def test_the_document_describes_the_class(self, architecture):
        from interpretune.analysis.points.component_map import known_architectures

        assert architecture in known_architectures()
        cmap = component_map_for(architecture)
        model = _tiny(architecture)
        assert check_map_against_model(cmap, model) == []
        assert derive_rmsnorm_offset(model, cmap) is EXPECTED_RMSNORM_OFFSET[architecture]

    def test_every_bundled_architecture_is_covered_here(self):
        """A document added under data/ without a builder above fails this test by name, which is the point."""
        from interpretune.analysis.points.component_map import known_architectures

        assert set(known_architectures()) == set(EXPECTED_RMSNORM_OFFSET), (
            "a bundled map has no entry in EXPECTED_RMSNORM_OFFSET / _tiny; entering the bundled set means adding "
            "both so the document is checked against its class"
        )


def _tl_adapter(architecture: str):
    """TransformerLens' own architecture adapter for a bundled architecture, built from the tiny config alone.

    No weights and no Hub call: the adapter is selected from a bridge config mapped off the HF config, the same
    way the bridge's boot path selects it, so the oracle covers every bundled map at the cost of a config.
    """
    pytest.importorskip("transformer_lens")
    from transformer_lens.config import TransformerBridgeConfig
    from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
    from transformer_lens.model_bridge.sources.transformers import map_default_transformer_lens_config

    hf_config = _tiny(architecture).config
    cfg = TransformerBridgeConfig.from_dict(dict(map_default_transformer_lens_config(hf_config).__dict__))
    cfg.architecture = architecture
    for nested in ("text_config", "vision_config"):
        if hasattr(hf_config, nested):
            setattr(cfg, nested, getattr(hf_config, nested))
    return ArchitectureAdapterFactory.select_architecture_adapter(cfg)


#: Bundled rows that have no TransformerLens counterpart, by architecture, each with the reason it is expected.
#: TransformerLens bridges a vision tower as one opaque component, so the multimodal document's vision stack and
#: projector are ours alone. Every other bundled row must have a counterpart: a row nothing independent vouches
#: for is the drift this oracle exists to catch, so a new unmatched row fails by name rather than being allowed.
EXPECTED_WITHOUT_TL_COUNTERPART = {
    "GPT2LMHeadModel": set(),
    "LlamaForCausalLM": set(),
    "Gemma2ForCausalLM": set(),
    "Gemma3ForCausalLM": set(),
    "Gemma3ForConditionalGeneration": {
        "projector",
        "vision.blocks.{i}",
        "vision.blocks.{i}.attn",
        "vision.blocks.{i}.attn.o",
        "vision.blocks.{i}.ln1",
        "vision.blocks.{i}.ln2",
        "vision.blocks.{i}.mlp",
        "vision.blocks.{i}.mlp.in",
        "vision.blocks.{i}.mlp.out",
        "vision.embed",
        "vision.ln_final",
    },
}


class TestEveryBundledMapAgreesWithTransformerLens:
    """The independent oracle for the bundled documents: TransformerLens' own per-architecture component mapping.

    A convergence case that resolves both sides through the same component map cannot catch a wrong row, because
    both sides move together; the nnsight backend's hook names derive from these documents, so its HF-reference
    cases have exactly that shape. This test compares each document with a source that shares nothing with it.

    It once covered gpt2 only, from a full weight-loading boot, and the derivation silently skipped two bridge
    classes the RMSNorm architectures use, so extending it to the other four maps compared seven rows of fourteen
    and would have reported agreement over the whole. The derivation now refuses a class it does not know, and
    the row counts are asserted here, so the oracle's own coverage is a measured number rather than a belief.
    """

    @pytest.mark.parametrize("architecture", sorted(EXPECTED_RMSNORM_OFFSET))
    def test_every_shared_row_names_the_same_module_and_kind(self, architecture):
        from interpretune.analysis.points.component_map import from_transformer_lens

        bundled = component_map_for(architecture)
        derived = from_transformer_lens(_tl_adapter(architecture), architecture)
        shared = sorted(set(bundled.components) & set(derived.components))
        mismatches = [
            (k, bundled.components[k], derived.components[k])
            for k in shared
            if bundled.components[k] != derived.components[k]
        ]
        assert not mismatches, mismatches
        # the oracle's reach, as a number: every text row is compared, not a subset that happens to overlap
        expected_shared = set(bundled.components) - EXPECTED_WITHOUT_TL_COUNTERPART[architecture]
        assert set(shared) == expected_shared, (
            f"{architecture}: TransformerLens vouches for {len(shared)} of {len(expected_shared)} bundled rows; "
            f"uncompared: {sorted(expected_shared - set(shared))}"
        )
        assert bundled.properties.get("sandwich_norms", False) == derived.properties["sandwich_norms"]

    @pytest.mark.parametrize("architecture", sorted(EXPECTED_RMSNORM_OFFSET))
    def test_the_bundled_map_carries_nothing_unvouched_for(self, architecture):
        """A bundled row with no TransformerLens counterpart is expected only where the table above says why."""
        from interpretune.analysis.points.component_map import from_transformer_lens

        derived = from_transformer_lens(_tl_adapter(architecture), architecture)
        extra = set(component_map_for(architecture).components) - set(derived.components)
        assert extra == EXPECTED_WITHOUT_TL_COUNTERPART[architecture], sorted(
            extra ^ EXPECTED_WITHOUT_TL_COUNTERPART[architecture]
        )

    def test_every_bundled_architecture_has_an_expectation_here(self):
        from interpretune.analysis.points.component_map import known_architectures

        assert set(known_architectures()) == set(EXPECTED_WITHOUT_TL_COUNTERPART)

    def test_an_unknown_bridge_class_is_refused_by_name(self):
        """The derivation names a class it cannot place instead of skipping it, which is how its coverage once
        shrank without a symptom."""
        from interpretune.analysis.points.component_map import from_transformer_lens

        class NovelBridge:
            name = "novel"
            submodules = {}

        class _Adapter:
            component_mapping = {"novel": NovelBridge()}

        with pytest.raises(ValueError, match="'NovelBridge' at 'novel' is a bridge class this derivation does not"):
            from_transformer_lens(_Adapter(), "GPT2LMHeadModel")

    def test_the_derived_map_is_a_schema_one_document(self):
        """The derived map uses no schema-2 key, so it declares schema 1 whatever the bundled document declares;
        the comparison is row by row and does not depend on the two agreeing on a version."""
        from interpretune.analysis.points.component_map import from_transformer_lens

        assert from_transformer_lens(_tl_adapter("GPT2LMHeadModel"), "GPT2LMHeadModel").schema_version == 1


class TestAdditiveChangesNeedNoBump:
    def test_adding_semantic_points_left_the_version_at_two(self):
        """The additive-change rule, pinned: four semantic points landed after schema 2 with no bump, because a
        semantic point is a code-only addition that no reader must understand to resolve the rows it has.

        A change that makes an older reader resolve the WRONG rows (a new applicability key) is what bumps; a name is
        not.
        """
        from interpretune.analysis.points.component_map import COMPONENT_MAP_SCHEMA_VERSION
        from interpretune.analysis.points.vocabulary import semantic_names

        assert COMPONENT_MAP_SCHEMA_VERSION == 2
        assert {"hook_cross_attn_in", "hook_cross_attn_out"} <= set(semantic_names())
