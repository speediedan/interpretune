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

    def test_the_map_describes_the_real_class(self):
        torch = pytest.importorskip("torch")
        tr = pytest.importorskip("transformers")
        small = dict(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=2, vocab_size=64)
        with torch.device("meta"):
            model = tr.Gemma3ForConditionalGeneration(
                tr.Gemma3Config(
                    text_config=dict(num_hidden_layers=2, **small),
                    vision_config=dict(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=3,
                        num_attention_heads=2,
                        image_size=16,
                        patch_size=8,
                    ),
                )
            )
        cmap = component_map_for("Gemma3ForConditionalGeneration")
        assert check_map_against_model(cmap, model) == []
        assert derive_rmsnorm_offset(model, cmap) is True
