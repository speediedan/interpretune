"""Exporting interventions as a venngram book (#887): each round trip applies as the payload did, or refuses by
name."""

import pytest
import torch
import yaml

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends.capabilities import InterventionSupport, PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec, _apply_op_to_region
from interpretune.analysis.venngram import (
    check_recipe,
    export_book,
    load_recipe,
    recipe_interventions,
    support_mapping,
    venngram_support,
)
from interpretune.utils import MisconfigurationException

WIDTH = 4
SITE = "mlp.hook_out"
HOOK = f"blocks.3.{SITE}"
VECTOR = torch.tensor([1.0, -2.0, 0.5, 0.0])
PAIR = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
SUBSPACE = torch.cat([PAIR, torch.tensor([[0.0, 0.0, 1.0, 0.0]])])


def _book(interventions, **kwargs):
    kwargs.setdefault("basis", "jlens_folded")
    return export_book(interventions, name="exported", model_id="m", revision="r1", **kwargs)


def _round_trip(interventions, **kwargs):
    """The payload a written and re-read book applies, with the recipe it read."""
    recipe = load_recipe(yaml.safe_load(yaml.safe_dump(_book(interventions, **kwargs)))).recipe
    return recipe_interventions(recipe, site_widths={SITE: WIDTH}), recipe


def _apply(specs, region):
    for spec in specs:
        region = _apply_op_to_region(region, spec)
    return region


REGION = torch.tensor([[0.5, 1.5, -2.0, 3.0], [1.0, -1.0, 0.25, 2.0]])  # (batch, width), one position


class TestRoundTrip:
    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param(InterventionSpec(torch.tensor(2.0), "add", 3.0), id="add-dose"),
            pytest.param(InterventionSpec(VECTOR, "add", 0.25), id="add-operand"),
            pytest.param(InterventionSpec(torch.tensor(2.0), "replace", 5.0), id="replace-dose"),
            pytest.param(InterventionSpec(VECTOR, "replace", 5.0), id="replace-operand"),
            pytest.param(InterventionSpec(torch.tensor(3.0), "scale", 0.5), id="scale-dose"),
            pytest.param(InterventionSpec(VECTOR, "project"), id="project"),
            pytest.param(InterventionSpec(VECTOR, "reject", 0.5), id="reject-vector"),
            pytest.param(InterventionSpec(SUBSPACE, "reject"), id="reject-subspace"),
            pytest.param(InterventionSpec(PAIR, "patch"), id="patch"),
            pytest.param(InterventionSpec(PAIR, "clamp", clamp_min=0.0), id="clamp-min"),
            pytest.param(InterventionSpec(VECTOR, "clamp", clamp_min=0.0, clamp_max=0.1), id="clamp"),
        ],
    )
    def test_each_mode_applies_as_the_payload_did(self, spec):
        payload, _ = _round_trip({HOOK: spec})
        assert list(payload) == [HOOK] and len(payload[HOOK]) == 1
        expected = _apply([spec], REGION.clone())
        assert not torch.equal(expected, REGION), "the case must change the activation for the comparison to mean much"
        torch.testing.assert_close(_apply(payload[HOOK], REGION.clone()), expected)

    def test_a_vector_scale_is_a_unit_set_of_the_units_it_changes(self):
        spec = InterventionSpec(torch.tensor([1.0, 2.0, 1.0, 0.5]), "scale", 1.0)
        payload, recipe = _round_trip({HOOK: spec})
        assert recipe.stages[0].site == SITE  # a unit set names its layers, so its stage names none
        assert recipe.stages[0].ops[0].unit_doses(recipe.book) == {(3, 1): 2.0, (3, 3): 0.5}
        assert payload[HOOK][0].intervention_tensor.tolist() == spec.intervention_tensor.tolist()

    def test_a_vector_add_at_a_unit_site_is_a_unit_set_with_its_scale_folded_in(self):
        spec = InterventionSpec(torch.tensor([0.0, 2.0, 0.0, 0.0]), "add", 0.5)
        payload, recipe = _round_trip({HOOK: spec}, unit_sites={SITE}, basis=None)
        assert recipe.operands == {} and recipe.stages[0].ops[0].unit_doses(recipe.book) == {(3, 1): 1.0}
        torch.testing.assert_close(_apply(payload[HOOK], REGION.clone()), _apply([spec], REGION.clone()))

    @pytest.mark.parametrize(
        "spec, dose",
        [
            pytest.param(InterventionSpec(torch.ones(WIDTH), "scale"), 1.0, id="identity-scale"),
            pytest.param(InterventionSpec(torch.full((WIDTH,), 2.0), "scale", 0.5), 1.0, id="scale"),
        ],
    )
    def test_a_constant_vector_scale_is_a_dose(self, spec, dose):
        _, recipe = _round_trip({HOOK: spec})
        op = recipe.stages[0].ops[0]
        assert op.target is None and op.value.dose == dose

    def test_order_and_scopes_survive(self):
        first = InterventionSpec(VECTOR, "add", position_scope=PositionScope.ALL_POSITIONS)
        second = InterventionSpec(torch.tensor(2.0), "scale", position_scope="all_positions")
        third = InterventionSpec(VECTOR, "reject", position_scope=PositionScope.LAST_TOKEN)
        other = InterventionSpec(torch.tensor(1.0), "add")
        payload, recipe = _round_trip({HOOK: [first, second, third], "blocks.4.hook_resid_pre": other})
        assert [(s.site, s.tokens.kind, [o.op for o in s.ops]) for s in recipe.stages] == [
            (HOOK, "all", ["add", "scale"]),
            (HOOK, "last", ["reject"]),
            ("blocks.4.hook_resid_pre", "last", ["add"]),
        ]
        assert [s.op for s in payload[HOOK]] == [
            "add",
            "scale",
            "reject",
        ]
        assert [s.position_scope for s in payload[HOOK]] == [
            PositionScope.ALL_POSITIONS,
            PositionScope.ALL_POSITIONS,
            PositionScope.LAST_TOKEN,
        ]

    def test_a_payload_read_from_a_recipe_exports_back_to_the_same_payload(self):
        first, _ = _round_trip({HOOK: [InterventionSpec(VECTOR, "add", 0.5)]})
        second, _ = _round_trip(first)
        assert second[HOOK][0].intervention_tensor.tolist() == first[HOOK][0].intervention_tensor.tolist()
        assert second[HOOK][0].scale_factor == first[HOOK][0].scale_factor

    def test_a_raw_tensor_takes_the_payload_default_mode(self):
        payload, _ = _round_trip({HOOK: VECTOR})
        assert payload[HOOK][0].op == "replace"


class TestOperands:
    def test_an_operand_above_the_inline_limit_goes_to_a_sidecar(self, tmp_path):
        from venngram.operands import INLINE_LIMIT

        spec = InterventionSpec(torch.arange(INLINE_LIMIT + 1, dtype=torch.float32), "add")
        with pytest.raises(MisconfigurationException, match="pass `sidecar_dir`"):
            _book({HOOK: spec})
        document = _book({HOOK: spec}, sidecar_dir=tmp_path)
        path = tmp_path / "book.yaml"
        path.write_text(yaml.safe_dump(document))
        recipe = load_recipe(path).recipe
        operand = recipe.operands["operand_0"]
        assert operand.file == "exported.operands.safetensors" and operand.values is None
        values, shape = operand.load(tmp_path)
        assert shape == (INLINE_LIMIT + 1,) and values[-1] == INLINE_LIMIT

    def test_a_small_operand_stays_inline_beside_a_large_one(self, tmp_path):
        from venngram.operands import INLINE_LIMIT

        big = InterventionSpec(torch.zeros(INLINE_LIMIT + 1), "add")
        document = _book({HOOK: [InterventionSpec(VECTOR, "add"), big]}, sidecar_dir=tmp_path)
        operands = document["treatments"]["exported"]["operands"]
        assert list(operands) == ["operand_0", "operand_1"]
        assert "values" in operands["operand_0"] and "file" in operands["operand_1"]


class TestRefusals:
    @pytest.mark.parametrize(
        "interventions, match",
        [
            pytest.param({}, "no interventions", id="empty"),
            pytest.param({"hook_embed": VECTOR}, "names none", id="no-layer"),
            pytest.param({"blocks.*.hook_resid_pre": VECTOR}, "expand a wildcard", id="wildcard"),
            pytest.param({HOOK: InterventionSpec(torch.empty(0), "add")}, "empty tensor", id="empty-tensor"),
            pytest.param({HOOK: InterventionSpec(torch.zeros(2, 2, WIDTH), "add")}, "takes a dose", id="add-rank"),
            pytest.param({HOOK: InterventionSpec(PAIR, "scale")}, "takes a dose or a vector", id="scale-rank"),
            pytest.param({HOOK: InterventionSpec(PAIR, "replace")}, "operand of rank", id="replace-rank"),
            pytest.param({HOOK: InterventionSpec(VECTOR, "project", 0.5)}, "no `scale_factor`", id="project-scale"),
            pytest.param({HOOK: InterventionSpec(PAIR, "patch", 0.5)}, "no `scale_factor`", id="patch-scale"),
            pytest.param(
                {HOOK: InterventionSpec(VECTOR, "project", use_intervention_tensor_as_basis=False)},
                "has no spelling",
                id="project-reversed",
            ),
            pytest.param({HOOK: InterventionSpec(SUBSPACE, "project")}, "operand of rank", id="project-subspace"),
            pytest.param({HOOK: InterventionSpec(VECTOR, "patch")}, "operand of rank", id="patch-rank"),
            pytest.param({HOOK: InterventionSpec(SUBSPACE, "patch")}, "needs a pair", id="patch-stack"),
            pytest.param({HOOK: InterventionSpec(VECTOR, "reject", 1.5)}, "`fraction` in", id="reject-fraction"),
            pytest.param({HOOK: InterventionSpec(VECTOR, "clamp")}, "neither bound", id="clamp"),
        ],
    )
    def test_refused_by_name(self, interventions, match):
        with pytest.raises(MisconfigurationException, match=match):
            _book(interventions)

    def test_an_operand_needs_its_basis(self):
        with pytest.raises(MisconfigurationException, match="pass `basis`"):
            _book({HOOK: InterventionSpec(VECTOR, "add")}, basis=None)

    def test_the_refusal_names_the_hook_and_spec(self):
        specs = [InterventionSpec(VECTOR, "add"), InterventionSpec(VECTOR, "clamp")]
        with pytest.raises(MisconfigurationException, match=rf"`{HOOK}` spec 1"):
            _book({HOOK: specs})


def test_an_exported_book_passes_venngrams_conformance_under_the_published_record(tmp_path):
    from venngram import conformance, store

    class _Capture:
        capturable = frozenset({SITE})

    support = venngram_support(InterventionSupport.every(), _Capture())
    document = _book(
        {
            HOOK: [
                InterventionSpec(VECTOR, "reject", 0.5),
                InterventionSpec(torch.tensor([1.0, 2.0, 1.0, 1.0]), "scale"),
            ]
        }
    )
    assert check_recipe(load_recipe(document).recipe, support) == []
    root = tmp_path / "published"
    root.mkdir()
    (root / "book.yaml").write_text(yaml.safe_dump(document))
    (root / "support.yaml").write_text(yaml.safe_dump(support_mapping(support)))
    store.publish(
        "local", str(root), root, documents=[("address_book", "book.yaml"), ("support_record", "support.yaml")]
    )
    record = conformance.read_support(str(root / "support.yaml"))
    report = conformance.check(f"local:{root}", supports=[record])
    assert report.ok, [(d.path, d.error, dict(d.refusals)) for d in report.documents]
    assert any(d.kind == "address_book" and d.refusals for d in report.documents)  # the book was judged, not skipped
