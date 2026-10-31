"""Applying a recipe through native interventions (#887): unit doses scattered per layer, refused when unmapped."""

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends.capabilities import InterventionMode, PositionScope
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
    assert spec.mode is InterventionMode.ADD and spec.position_scope is PositionScope.LAST_TOKEN
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
                "ops": [{"op": "scale", "target": {"set": "chess"}, "value": {"dose": 2.0}}],
            }
        ],
    }
    with pytest.raises(MisconfigurationException, match="only unit-targeted `add` doses"):
        recipe_interventions(_recipe(doc), site_widths={"mlp.out.hook_in": 8192})
