"""Applying a recipe through native interventions (#887): unit doses scattered per layer, refused when unmapped."""

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.testing.venngram_books import treatment_book
from interpretune.analysis.backends.capabilities import PositionScope
from interpretune.analysis.backends.interventions import apply_intervention
from interpretune.analysis.venngram import load_recipe, recipe_interventions
from interpretune.utils import MisconfigurationException

RECIPE = {
    "schema_version": 2,
    "name": "dose-the-units",
    "model": {"id": "m", "revision": "r1"},
    "book": {
        "schema_version": 1,
        "name": "units",
        "portability": "checkpoint",
        "applies_to": {"models": [{"id": "m", "revision": "r1"}]},
        "sets": {"chess": {"point": "mlp.out.hook_in", "layers": [2, 15], "units": {2: [7], 15: [4041]}}},
    },
    "stages": [
        {
            "site": "mlp.out.hook_in",
            "tokens": "last",
            "ops": [{"op": "add", "target": {"set": "chess"}, "value": {"doses": {2: {7: 0.5}, 15: {4041: -1.25}}}}],
        }
    ],
}


def _recipe(doc=RECIPE):
    return load_recipe(treatment_book(doc)).recipe


def test_unit_doses_scatter_per_layer_and_apply_through_the_native_path():
    payload = recipe_interventions(_recipe(), site_widths={"mlp.out.hook_in": 8192})
    assert sorted(payload) == ["blocks.15.mlp.out.hook_in", "blocks.2.mlp.out.hook_in"]
    (spec,) = payload["blocks.15.mlp.out.hook_in"]
    assert spec.op == "add" and spec.position_scope is PositionScope.LAST_TOKEN
    assert spec.intervention_tensor.shape == (8192,) and spec.intervention_tensor[4041] == -1.25
    assert torch.count_nonzero(spec.intervention_tensor) == 1

    value = torch.zeros(1, 3, 8192)
    out = apply_intervention(value, spec, last_pos=2)
    assert out[0, 2, 4041] == -1.25 and out[0, 0, 4041] == 0.0  # last token only


def test_a_unit_outside_the_supplied_width_and_a_missing_width_refuse_by_name():
    with pytest.raises(MisconfigurationException, match="unit 4041 at layer 15 is outside the site width 1024"):
        recipe_interventions(_recipe(), site_widths={"mlp.out.hook_in": 1024})
    with pytest.raises(MisconfigurationException, match="no width was supplied for this site"):
        recipe_interventions(_recipe(), site_widths={})


def test_a_unit_replace_changes_only_the_named_units_and_positions():
    doc = {
        **RECIPE,
        "stages": [
            {
                **RECIPE["stages"][0],
                "ops": [{"op": "replace", "target": {"set": "chess"}, "value": {"dose": 2.0}}],
            }
        ],
    }
    payload = recipe_interventions(_recipe(doc), site_widths={"mlp.out.hook_in": 8192})
    (spec,) = payload["blocks.2.mlp.out.hook_in"]
    value = torch.arange(2 * 3 * 8192, dtype=torch.float32).reshape(2, 3, 8192)
    expected = value.clone()
    expected[:, 1, 7] = 2.0
    actual = apply_intervention(value.clone(), spec, last_pos=1)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, value)


BASIS = "jlens_folded"


def _activation_recipe(ops, operands, site="blocks.3.hook_resid_post", tokens="all"):
    return _recipe(
        {
            **RECIPE,
            "operands": {name: {**operand, "basis": BASIS} for name, operand in operands.items()},
            "stages": [{"site": site, "tokens": tokens, "ops": ops}],
        }
    )


def _apply(payload, value, pattern="blocks.3.hook_resid_post"):
    for spec in payload[pattern]:
        value = apply_intervention(value, spec, last_pos=value.shape[1] - 1)
    return value


def test_a_unit_scale_multiplies_the_named_units_and_keeps_the_rest():
    doc = {
        **RECIPE,
        "stages": [
            {**RECIPE["stages"][0], "ops": [{"op": "scale", "target": {"set": "chess"}, "value": {"dose": 3.0}}]}
        ],
    }
    payload = recipe_interventions(_recipe(doc), site_widths={"mlp.out.hook_in": 8192})
    (spec,) = payload["blocks.2.mlp.out.hook_in"]
    assert spec.op == "scale" and spec.intervention_tensor[7] == 3.0
    assert torch.count_nonzero(spec.intervention_tensor - 1.0) == 1  # ones everywhere else
    value = torch.ones(1, 3, 8192)
    out = apply_intervention(value, spec, last_pos=2)
    assert out[0, 2, 7] == 3.0 and out[0, 2, 8] == 1.0 and out[0, 0, 7] == 1.0


def test_each_operand_op_maps_onto_its_native_mode_in_recipe_order():
    operands = {
        "v": {"values": [1.0, 0.0, 0.0, 0.0], "shape": [4]},
        "pair": {"values": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "shape": [2, 4]},
        "span": {"values": [1.0, 0.0, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0], "shape": [2, 4]},
    }
    ops = [
        {"op": "add", "operand": "v", "value": {"dose": 0.5}},
        {"op": "replace", "operand": "v", "value": {"dose": 2.0}},
        {"op": "scale", "value": {"dose": 3.0}},
        {"op": "project", "operand": "v"},
        {"op": "reject", "operand": "span", "params": {"fraction": 0.25}},
        {"op": "patch", "operand": "pair"},
        {"op": "clamp", "operand": "v", "params": {"min": -1.0, "max": 1.0}},
    ]
    specs = recipe_interventions(_activation_recipe(ops, operands), site_widths={})["blocks.3.hook_resid_post"]
    assert [s.op for s in specs] == [
        "add",
        "replace",
        "scale",
        "project",
        "reject",
        "patch",
        "clamp",
    ]
    add, replace, scale, _, reject, patch, clamp = specs
    assert add.scale_factor == 0.5 and torch.equal(add.intervention_tensor, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(replace.intervention_tensor, torch.tensor([2.0, 0.0, 0.0, 0.0]))  # replace ignores scale_factor
    assert scale.intervention_tensor.item() == 3.0
    assert reject.scale_factor == 0.25 and reject.intervention_tensor.shape == (2, 4)
    assert patch.intervention_tensor.shape == (2, 4)
    assert (clamp.clamp_min, clamp.clamp_max) == (-1.0, 1.0)
    assert all(s.position_scope is PositionScope.ALL_POSITIONS for s in specs)


def test_a_full_reject_through_the_native_path_empties_the_span():
    operands = {"v": {"values": [0.6, 0.8, 0.0, 0.0], "shape": [4]}}
    payload = recipe_interventions(
        _activation_recipe([{"op": "reject", "operand": "v"}], operands, tokens="last"), site_widths={}
    )
    value = torch.tensor([[[5.0, -3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]])
    out = _apply(payload, value.clone())
    assert torch.allclose(out[0, 1] @ torch.tensor([0.6, 0.8, 0.0, 0.0]), torch.tensor(0.0), atol=1e-6)
    assert torch.equal(out[0, 0], value[0, 0])  # last token only
    assert torch.equal(out[0, 1, 2:], value[0, 1, 2:])  # the orthogonal complement is untouched


def test_a_sidecar_operand_loads_relative_to_the_recipe(tmp_path):
    import yaml
    from venngram.operands import write_operands

    sha = write_operands(tmp_path / "dirs.safetensors", {"concept": ([0.0, 2.0, 0.0, 0.0], [4])})
    doc = {
        **RECIPE,
        "operands": {
            "concept": {"file": "dirs.safetensors", "key": "concept", "sha256": sha, "shape": [4], "basis": BASIS}
        },
        "stages": [{"site": "blocks.1.hook_resid_post", "tokens": "all", "ops": [{"op": "add", "operand": "concept"}]}],
    }
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(treatment_book(doc)), encoding="utf-8")
    payload = recipe_interventions(load_recipe(tmp_path / "recipe.yaml").recipe, site_widths={})
    (spec,) = payload["blocks.1.hook_resid_post"]
    assert torch.equal(spec.intervention_tensor, torch.tensor([0.0, 2.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "ops, operands, site, widths, match",
    [
        (
            [{"op": "reject", "operand": "v"}],
            {"v": {"values": [1.0, 0.0], "shape": [2]}},
            "hook_resid_post",
            {},
            "names no layer",
        ),
        (
            [{"op": "project", "operand": "s"}],
            {"s": {"values": [1.0, 0.0, 0.0, 1.0], "shape": [2, 2]}},
            "blocks.0.hook_resid_post",
            {},
            "projecting onto a subspace has no native translation",
        ),
        (
            [{"op": "project", "operand": "v", "params": {"fraction": 0.5}}],
            {"v": {"values": [1.0, 0.0], "shape": [2]}},
            "blocks.0.hook_resid_post",
            {},
            "fraction below 1 has no native translation",
        ),
        (
            [{"op": "add", "operand": "v"}],
            {"v": {"values": [1.0, 0.0], "shape": [2]}},
            "blocks.0.hook_resid_post",
            {"hook_resid_post": 768},
            "operand `v` has width 2 but the site width is 768",
        ),
    ],
)
def test_untranslatable_activation_ops_refuse_by_name(ops, operands, site, widths, match):
    with pytest.raises(MisconfigurationException, match=match):
        recipe_interventions(_activation_recipe(ops, operands, site=site), site_widths=widths)


GRADIENT_RECIPE = {
    **RECIPE,
    "book": {
        **RECIPE["book"],
        "schema_version": 2,
        "sets": {"grads": {"point": "mlp.out.hook_in", "layers": [2], "units": [7], "of": "grad"}},
    },
    "stages": [
        {
            "site": "mlp.out.hook_in",
            "tokens": "all",
            "ops": [{"op": "scale", "target": {"set": "grads"}, "value": {"dose": 0.0}}],
        }
    ],
}


def test_an_op_on_a_gradient_is_refused_rather_than_applied_to_the_activation_value():
    """#938: a target set addressing `of: grad` names the gradient through those units; the forward surface writes
    values, so translating the op would scale the activation instead."""
    from interpretune.analysis.backends.capabilities import InterventionSupport
    from interpretune.analysis.venngram import check_recipe, venngram_support

    recipe = _recipe(GRADIENT_RECIPE)
    with pytest.raises(MisconfigurationException, match="addresses the gradient"):
        recipe_interventions(recipe, site_widths={"mlp.out.hook_in": 8192})
    record = InterventionSupport(position_scopes={PositionScope.ALL_POSITIONS}, ops={"scale"})
    support = venngram_support(record)
    assert support.of == frozenset({"value"})  # the record's flows are carried into the declaration
    refusals = check_recipe(recipe, support)
    assert any("does not act on" in r for r in refusals) and any("addresses the gradient" in r for r in refusals)
    # declaring `grad` (as both model backends do) passes venngram's check; translating for values still refuses
    both = InterventionSupport(position_scopes={PositionScope.ALL_POSITIONS}, ops={"scale"}, of={"value", "grad"})
    assert venngram_support(both).of == frozenset({"value", "grad"})
    assert [r for r in check_recipe(recipe, venngram_support(both)) if "addresses the gradient" in r]
    assert check_recipe(recipe, venngram_support(both), flow="grad") == []


def test_an_op_on_a_gradient_translates_for_the_backward_surface():
    """#970: with `flow="grad"` the same unit op becomes the spec it would be on values, for
    `bwd_w_intervention`."""
    payload = recipe_interventions(_recipe(GRADIENT_RECIPE), site_widths={"mlp.out.hook_in": 16}, flow="grad")
    (spec,) = payload["blocks.2.mlp.out.hook_in"]
    assert spec.op == "scale" and spec.position_scope is PositionScope.ALL_POSITIONS
    assert spec.intervention_tensor[7] == 0.0 and int((spec.intervention_tensor == 1.0).sum()) == 15


def test_a_value_op_is_refused_by_the_backward_translation_and_an_unknown_flow_by_name():
    with pytest.raises(MisconfigurationException, match="acts on the activation value, and this translation is for"):
        recipe_interventions(_recipe(), site_widths={"mlp.out.hook_in": 8192}, flow="grad")
    with pytest.raises(MisconfigurationException, match="flow 'gradient' is not a flow"):
        recipe_interventions(_recipe(), site_widths={"mlp.out.hook_in": 8192}, flow="gradient")


def _composed_recipe(target, value):
    """A recipe whose op targets ``target``, a composition of the book's unit sets at one site (#905,
    venngram#80)."""
    point = {"point": "mlp.out.hook_in", "layers": [12]}
    sets = {
        "a": {**point, "units": {12: [1, 2, 3]}},
        "b": {**point, "units": {12: [3, 4]}},
        "both": "@a & @b",
        "rest": "@a - @b",
    }
    doc = {
        **RECIPE,
        "book": {**RECIPE["book"], "schema_version": 2, "sets": sets},
        "stages": [{**RECIPE["stages"][0], "ops": [{"op": "add", "target": {"set": target}, "value": value}]}],
    }
    return _recipe(doc)


def test_a_composed_unit_target_applies_through_the_native_path():
    """An intersection dosed once doses exactly its one member; it was refused as a feature target before."""
    payload = recipe_interventions(_composed_recipe("both", {"dose": 0.5}), site_widths={"mlp.out.hook_in": 16})
    (spec,) = payload["blocks.12.mlp.out.hook_in"]
    assert spec.op == "add" and spec.intervention_tensor[3] == 0.5
    assert torch.count_nonzero(spec.intervention_tensor) == 1


def test_a_doses_table_on_a_composed_target_doses_only_its_members():
    """``@a - @b`` is units 1 and 2: a table entry for unit 4 (in ``b``) is not a member, so nothing doses it."""
    recipe = _composed_recipe("rest", {"doses": {12: {1: 0.5, 2: -0.25, 4: 9.0}}})
    (spec,) = recipe_interventions(recipe, site_widths={"mlp.out.hook_in": 16})["blocks.12.mlp.out.hook_in"]
    assert spec.intervention_tensor[1] == 0.5 and spec.intervention_tensor[2] == -0.25
    assert spec.intervention_tensor[4] == 0.0 and torch.count_nonzero(spec.intervention_tensor) == 2


def test_a_table_value_on_a_composed_target_reads_its_column_for_every_member():
    """``@a - @b`` is units 1 and 2: each takes its own value from the column, and a row for unit 4 (in ``b``) is not a
    member, so nothing doses it. It was refused before venngram listed a composition's members (venngram#131)."""
    value = {"from": {"table": "scores", "column": "score"}}
    tables = {"scores": {"score": {(12, 1): 0.5, (12, 2): -0.25, (12, 4): 9.0}}}
    payload = recipe_interventions(_composed_recipe("rest", value), site_widths={"mlp.out.hook_in": 16}, tables=tables)
    (spec,) = payload["blocks.12.mlp.out.hook_in"]
    assert spec.intervention_tensor[1] == 0.5 and spec.intervention_tensor[2] == -0.25
    assert spec.intervention_tensor[4] == 0.0 and torch.count_nonzero(spec.intervention_tensor) == 2


def test_a_table_value_on_a_composed_target_lists_the_members_its_column_misses():
    """A composed target's member the column has no row for is refused rather than dropped, as for a single set."""
    value = {"from": {"table": "scores", "column": "score"}}
    tables = {"scores": {"score": {(12, 1): 0.5}}}
    with pytest.raises(MisconfigurationException, match=r"no value for \(layer, unit\) \[\(12, 2\)\]"):
        recipe_interventions(_composed_recipe("rest", value), site_widths={"mlp.out.hook_in": 16}, tables=tables)


def _book_with_tables(value, *, tables=True, sets=None, site="mlp.out.hook_in"):
    """A schema 13 book whose treatment reads ``value`` and whose ``tables`` section carries ``finding`` (#905)."""
    model = {"id": "m", "revision": "r1"}
    finding = {
        "model": model,
        "source": {"name": "finder", "commit": "0f3a9c1", "license": "MIT"},
        "columns": {"dose": {"description": "signed steering dose"}, "score": {"description": "a finder score"}},
        "rows": [{"layer": 12, "unit": u, "dose": d, "score": -d} for u, d in ((1, 1.5), (2, -0.5), (4, 9.0))],
    }
    book = {
        "schema_version": 13,
        "name": "book-tables",
        "portability": "checkpoint",
        "applies_to": {"models": [model]},
        "sets": sets or {"a": {"point": "mlp.out.hook_in", "units": {12: [1, 2]}}},
        "treatments": {
            "steer": {
                "model": model,
                "stages": [{"site": site, "ops": [{"op": "add", "target": {"set": "a"}, "value": value}]}],
            }
        },
    }
    if tables:
        book["tables"] = {"finding": finding}
    return load_recipe(book).recipe


def test_a_table_value_reads_the_books_own_table_when_the_caller_supplies_none():
    """The book names the table its treatment reads, so the caller passes no ``tables`` at all (#905)."""
    recipe = _book_with_tables({"from": {"table": "finding", "column": "dose"}})
    (spec,) = recipe_interventions(recipe, site_widths={"mlp.out.hook_in": 16})["blocks.12.mlp.out.hook_in"]
    assert spec.intervention_tensor[1] == 1.5 and spec.intervention_tensor[2] == -0.5
    assert spec.intervention_tensor[4] == 0.0 and torch.count_nonzero(spec.intervention_tensor) == 2


def test_a_book_tables_transform_reads_a_second_column_of_the_same_table():
    """``sign_of`` reads its column from the book's table as the value does."""
    value = {"from": {"table": "finding", "column": "dose"}, "transform": [{"sign_of": {"column": "score"}}]}
    (spec,) = recipe_interventions(_book_with_tables(value), site_widths={"mlp.out.hook_in": 16})[
        "blocks.12.mlp.out.hook_in"
    ]
    assert spec.intervention_tensor[1] == -1.5 and spec.intervention_tensor[2] == 0.5


def test_a_caller_table_replaces_the_books_table_of_that_name():
    recipe = _book_with_tables({"from": {"table": "finding", "column": "dose"}})
    tables = {"finding": {"dose": {(12, 1): 3.0, (12, 2): 4.0}}}
    (spec,) = recipe_interventions(recipe, site_widths={"mlp.out.hook_in": 16}, tables=tables)[
        "blocks.12.mlp.out.hook_in"
    ]
    assert spec.intervention_tensor[1] == 3.0 and spec.intervention_tensor[2] == 4.0


def test_a_table_neither_the_book_nor_the_caller_supplies_is_refused_by_name():
    recipe = _book_with_tables({"from": {"table": "finding", "column": "dose"}}, tables=False)
    with pytest.raises(MisconfigurationException, match="reads table `finding`, which neither the book"):
        recipe_interventions(recipe, site_widths={"mlp.out.hook_in": 16})


def test_a_feature_target_reads_the_books_table_as_a_unit_target_does():
    """A feature set's members are ``(layer, feature)`` pairs, and the book's table is read for them the same
    way."""
    from interpretune.analysis.venngram.values import member_values

    sets = {"a": {"features": "1,2", "layers": [12]}}
    recipe = _book_with_tables({"from": {"table": "finding", "column": "dose"}}, sets=sets, site="hook_resid_post")
    assert member_values(recipe, recipe.stages[0].ops[0]) == {(12, 1): 1.5, (12, 2): -0.5}


def test_a_composed_gradient_target_is_refused_like_a_single_one():
    """A composition of ``of: grad`` unit sets addresses the gradient as each of its terms does (#938, #999)."""
    point = {"point": "mlp.out.hook_in", "layers": [12], "of": "grad"}
    sets = {"a": {**point, "units": {12: [1, 2, 3]}}, "b": {**point, "units": {12: [3, 4]}}, "both": "@a & @b"}
    doc = {
        **RECIPE,
        "book": {**RECIPE["book"], "schema_version": 2, "sets": sets},
        "stages": [{**RECIPE["stages"][0], "ops": [{"op": "add", "target": {"set": "both"}, "value": {"dose": 0.5}}]}],
    }
    with pytest.raises(MisconfigurationException, match="addresses the gradient"):
        recipe_interventions(_recipe(doc), site_widths={"mlp.out.hook_in": 16})


def test_position_scoped_recipe_repeats_its_operand_without_touching_other_positions():
    doc = {**RECIPE, "stages": [{**RECIPE["stages"][0], "tokens": {"positions": [2, 0]}}]}
    payload = recipe_interventions(_recipe(doc), site_widths={"mlp.out.hook_in": 8192})
    (spec,) = payload["blocks.2.mlp.out.hook_in"]
    assert spec.positions == (2, 0)
    result = apply_intervention(torch.zeros(1, 4, 8192), spec, last_pos=3)
    assert result[0, 2, 7] == result[0, 0, 7] == 0.5
    assert torch.count_nonzero(result) == 2
