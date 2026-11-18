"""Selected-coordinate interventions preserve their complement, token scope and autograd paths."""

import pytest
import torch

from interpretune.analysis.backends.interventions import (
    InterventionSpec,
    apply_intervention,
    apply_op,
    build_intervention_dict,
)


@pytest.mark.parametrize("scope", ["last_token", "all_positions"])
@pytest.mark.parametrize("op", ["add", "scale", "replace", "clamp"])
def test_selected_units_match_a_manual_coordinate_edit(scope, op):
    original = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) - 6
    selected = (3, 1)
    operand = torch.tensor([2.0, -1.0]) if op != "clamp" else torch.eye(2)
    spec = InterventionSpec(operand, op=op, position_scope=scope, unit_indices=selected, clamp_min=-2.0, clamp_max=2.0)
    # Exercise the canonicalizer too: a low-level op that works only before payload validation is unreachable.
    canonical = build_intervention_dict({"hook": spec}, {"hook": ["hook"]}, {"hook": (4,)})["hook"][0]
    expected = original.clone()
    region = expected if scope == "all_positions" else expected[:, 1, :]
    for unit, dose in zip(selected, [2.0, -1.0], strict=True):
        if op == "add":
            region[..., unit] += dose
        elif op == "scale":
            region[..., unit] *= dose
        elif op == "replace":
            region[..., unit] = dose
        else:
            region[..., unit].clamp_(-2.0, 2.0)
    actual = apply_intervention(original.clone(), canonical, last_pos=1)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, original)
    assert torch.equal(actual[..., [0, 2]], original[..., [0, 2]])
    if scope == "all_positions":
        torch.testing.assert_close(apply_op(original, canonical), expected)


def test_a_selected_replacement_preserves_the_complements_gradient():
    value = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).requires_grad_()
    operand = torch.tensor([3.0, 5.0], requires_grad=True)
    spec = InterventionSpec(operand, unit_indices=(3, 1), position_scope="all_positions")
    out = apply_intervention(value.clone(), spec, last_pos=1)
    out.sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([[[1.0, 0.0, 1.0, 0.0]]]).expand_as(value))
    torch.testing.assert_close(operand.grad, torch.tensor([2.0, 2.0]))


@pytest.mark.parametrize("indices", [(), (-1,), (4,), (1, 1), (True,), (1.5,)])
def test_invalid_unit_indices_are_refused_before_execution(indices):
    spec = InterventionSpec(torch.tensor(1.0), unit_indices=indices)
    with pytest.raises(ValueError, match="unit_indices"):
        build_intervention_dict({"hook": spec}, {"hook": ["hook"]}, {"hook": (4,)})


def test_mapping_payloads_keep_their_selection_after_canonicalization():
    spec = build_intervention_dict(
        {"hook": {"intervention_tensor": [5.0], "unit_indices": [2]}},
        {"hook": ["hook"]},
        {"hook": (4,)},
    )["hook"][0]
    assert spec.unit_indices == (2,)
    original = torch.ones(1, 2, 4)
    expected = original.clone()
    expected[:, 1, 2] = 5.0
    assert torch.equal(apply_intervention(original, spec, last_pos=1), expected)


@pytest.mark.parametrize("op", ["add", "scale", "replace"])
def test_export_and_reimport_preserve_selected_values_including_zeros(op):
    from interpretune.analysis.venngram import export_book, load_recipe, recipe_interventions

    hook = "blocks.0.hook_resid_post"
    spec = InterventionSpec(torch.tensor([0.0, 2.0]), op=op, unit_indices=(3, 1))
    doc = export_book({hook: spec}, name="selected", model_id="m", revision="r")
    recipe = load_recipe(doc).recipe
    (restored,) = recipe_interventions(recipe, site_widths={"hook_resid_post": 4})[hook]
    x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    torch.testing.assert_close(
        apply_intervention(x.clone(), spec, last_pos=1), apply_intervention(x.clone(), restored, last_pos=1)
    )
    assert recipe.stages[0].ops[0].unit_doses(recipe.book) == {(0, 3): 0.0, (0, 1): 2.0}


def test_export_refuses_a_selected_clamp_the_recipe_registry_cannot_express():
    from interpretune.analysis.venngram import export_book
    from interpretune.utils import MisconfigurationException

    spec = InterventionSpec(torch.eye(1), op="clamp", unit_indices=(2,), clamp_min=0.0)
    with pytest.raises(MisconfigurationException, match="venngram's clamp target is activation"):
        export_book({"blocks.0.hook_resid_post": spec}, name="selected", model_id="m", revision="r")
