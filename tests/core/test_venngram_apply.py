"""Applying a recipe through native interventions (#887): unit doses scattered per layer, refused when unmapped."""

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends.capabilities import PositionScope
from interpretune.analysis.backends.interventions import apply_intervention
from interpretune.analysis.venngram import load_recipe, recipe_interventions
from interpretune.utils import MisconfigurationException

RECIPE = {
    "schema_version": 2,
    "kind": "recipe",
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
    return load_recipe(doc).recipe


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


def test_an_unapplied_op_refuses_rather_than_narrowing():
    doc = {
        **RECIPE,
        "stages": [
            {
                **RECIPE["stages"][0],
                "ops": [{"op": "replace", "target": {"set": "chess"}, "value": {"dose": 2.0}}],
            }
        ],
    }
    with pytest.raises(MisconfigurationException, match="a unit-targeted `replace` has no native translation"):
        recipe_interventions(_recipe(doc), site_widths={"mlp.out.hook_in": 8192})


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
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(doc))
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
    # declaring `grad` (as no backend does yet) passes venngram's check; the apply path still refuses
    both = InterventionSupport(position_scopes={PositionScope.ALL_POSITIONS}, ops={"scale"}, of={"value", "grad"})
    assert venngram_support(both).of == frozenset({"value", "grad"})
    assert [r for r in check_recipe(recipe, venngram_support(both)) if "addresses the gradient" in r]
