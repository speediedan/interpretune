"""Unit addresses preserve the point, layer, coordinate set and flow through execution."""

import pytest
import torch
from torch import nn

from interpretune.analysis.backends.capabilities import InterventionSupport
from interpretune.analysis.backends.interventions import apply_intervention, require_intervention_support
from interpretune.analysis.venngram import normalized_unit_doses, resolve_unit_sites, unit_interventions
from interpretune.tuning.interventions import (
    InterventionCfg,
    UnitIntervention,
    attach_configured_interventions,
    attach_intervention,
)
from interpretune.utils import MisconfigurationException


POINT = "hook_resid_post"
WIDTHS = {POINT: 4}


def test_set_algebra_keeps_each_layers_own_units():
    sets = {
        "chosen": {"point": POINT, "layers": [0, 1], "units": {0: [1, 3], 1: [2]}},
        "control": {"point": POINT, "layers": [0], "units": [1]},
    }
    selection = resolve_unit_sites("@chosen - @control", n_layers=2, site_widths=WIDTHS, sets=sets)
    assert selection.units == {f"blocks.0.{POINT}": (3,), f"blocks.1.{POINT}": (2,)}
    assert selection.flow == "value"


def test_slices_and_negative_indices_use_the_supplied_checkpoint_width():
    selected = resolve_unit_sites(f"point:{POINT} & layers:-1: & units:-2:", n_layers=3, site_widths=WIDTHS)
    assert selected.units == {f"blocks.2.{POINT}": (2, 3)}


@pytest.mark.parametrize(
    "address, match",
    [
        (f"point:{POINT} & layers:0", "must name `units`"),
        ("units:[1]", "must name `units` and a `point`"),
        (f"point:{POINT} & layers:0 & units:[9]", "outside"),
        (f"point:{POINT} & layers:0 & units:[1] & features:sae", "no unit-site meaning"),
        (f"point:{POINT} & layers:0 & units:[1] & space:jlens", "no unit-site meaning"),
    ],
)
def test_unsupported_unit_addresses_are_refused(address, match):
    with pytest.raises((MisconfigurationException, ValueError), match=match):
        resolve_unit_sites(address, n_layers=2, site_widths=WIDTHS)


def test_a_width_is_required_rather_than_inferred_from_the_last_index():
    with pytest.raises(MisconfigurationException, match="positive last-axis width"):
        resolve_unit_sites(f"point:{POINT} & units:[1]", n_layers=2, site_widths={})


@pytest.mark.parametrize("op", ["add", "scale", "replace", "clamp"])
def test_direct_payloads_cover_all_four_modes_with_per_layer_values(op):
    payload = unit_interventions(
        {"point": POINT, "layers": [0, 1], "units": {0: [3], 1: [1, 2]}},
        n_layers=2,
        site_widths=WIDTHS,
        op=op,
        **({"clamp_min": -0.5, "clamp_max": 0.5} if op == "clamp" else {"value": {0: 2.0, 1: [-1.0, 3.0]}}),
    )
    original = torch.tensor([[[-2.0, -1.0, 2.0, 4.0]]])
    expected = original.clone()
    if op == "add":
        expected[..., 1] -= 1.0
        expected[..., 2] += 3.0
    elif op == "scale":
        expected[..., 1] *= -1.0
        expected[..., 2] *= 3.0
    elif op == "replace":
        expected[..., 1], expected[..., 2] = -1.0, 3.0
    else:
        expected[..., 1:3].clamp_(-0.5, 0.5)
    actual = apply_intervention(original.clone(), payload[f"blocks.1.{POINT}"], last_pos=0)
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, original)


def test_a_gradient_address_cannot_silently_become_a_forward_payload():
    address = f"point:{POINT} & layers:0 & units:[1] & of:grad"
    with pytest.raises(MisconfigurationException, match="acts on 'grad'.*flow is 'value'"):
        unit_interventions(address, n_layers=2, site_widths=WIDTHS, value=2.0)
    assert unit_interventions(address, n_layers=2, site_widths=WIDTHS, value=2.0, flow="grad")


def test_a_backend_must_declare_unit_targets():
    class Backend:
        intervention_support = InterventionSupport.every(targets={"activation": None})

    payload = unit_interventions(f"point:{POINT} & units:[1]", n_layers=1, site_widths=WIDTHS, value=2.0)
    with pytest.raises(NotImplementedError, match="does not declare the 'units'"):
        require_intervention_support(Backend(), payload, backend_name="activation-only")


@pytest.mark.parametrize("unit_dim", [0, 1])
def test_normalized_doses_match_the_column_norm_definition_in_both_layouts(unit_dim):
    weight = torch.tensor([[3.0, 0.0, 8.0], [4.0, 2.0, 6.0]])
    actual = normalized_unit_doses(
        torch.tensor([2.0, -3.0]),
        units=(2, 0),
        down_projection=weight if unit_dim == 1 else weight.T,
        unit_dim=unit_dim,
        median_residual_norm=5.0,
    )
    # The selected column norms are 10 and 5, so normalization gives 1 and -3.
    torch.testing.assert_close(actual, torch.tensor([1.0, -3.0]))


def test_normalization_refuses_a_zero_column():
    with pytest.raises(MisconfigurationException, match="nonzero finite column norms"):
        normalized_unit_doses(1.0, units=(1,), down_projection=torch.zeros(2, 3), unit_dim=1, median_residual_norm=5.0)


@pytest.mark.parametrize("op", ["add", "scale", "replace", "clamp"])
def test_unit_modules_match_manual_edits_and_preserve_the_complement(op):
    kwargs = {"clamp_min": -1.0, "clamp_max": 1.0} if op == "clamp" else {"operand": torch.tensor([2.0, -3.0])}
    module = UnitIntervention(op, [3, 1], **kwargs)
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) - 2
    expected = x.clone()
    for unit, operand in [(3, 2.0), (1, -3.0)]:
        if op == "add":
            expected[..., unit] += operand
        elif op == "scale":
            expected[..., unit] *= operand
        elif op == "replace":
            expected[..., unit] = operand
        else:
            expected[..., unit].clamp_(-1.0, 1.0)
    torch.testing.assert_close(module(x), expected)
    assert torch.equal(module(x)[..., [0, 2]], x[..., [0, 2]])


@pytest.mark.parametrize("op", ["add", "scale"])
def test_a_learned_unit_operand_has_only_selected_parameters_and_trains(op):
    module = UnitIntervention(op, [1, 3])
    x = torch.ones(1, 2, 4, requires_grad=True)
    assert torch.equal(module(x), x)
    assert sum(p.numel() for p in module.parameters()) == 2
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    module(x).sum().backward()
    optimizer.step()
    actual = module(x)
    assert not torch.equal(actual, x)
    assert torch.equal(actual[..., [0, 2]], x[..., [0, 2]])


def test_a_configured_unit_address_splices_the_mlp_input_and_trains_only_two_units():
    from transformers import GPT2Config, GPT2LMHeadModel

    root = nn.Module()
    root.model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=4, n_inner=8, vocab_size=8)).eval()
    address = "point:mlp.out.hook_in & layers:0 & units:[1, 6]"
    (steer,) = attach_configured_interventions(root, [InterventionCfg(name="s", site=address)])
    assert isinstance(steer, UnitIntervention)
    ids = torch.tensor([[1, 2, 3]])
    before = root.model(ids).logits
    with torch.no_grad():
        steer.operand.value.fill_(3.0)
    after = root.model(ids).logits
    assert not torch.equal(before, after)
    assert sum(p.numel() for p in steer.parameters()) == 2
    with pytest.raises(ValueError, match="requires a UnitIntervention with units"):
        attach_intervention(root, "wrong", UnitIntervention("add", [2, 3]), address)


def test_a_book_entry_gives_a_configured_splice_its_site_and_units(tmp_path):
    """A site that references an address book's named set takes the set's units too (#876): the configured
    intervention is the same two-unit splice the literal address gives."""
    import yaml
    from transformers import GPT2Config, GPT2LMHeadModel

    book = {"schema_version": 1, "name": "units", "sets": {"steer": "point:mlp.out.hook_in & layers:0 & units:[1, 6]"}}
    path = tmp_path / "units.yaml"
    path.write_text(yaml.safe_dump(book))
    root = nn.Module()
    root.model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=4, n_inner=8, vocab_size=8)).eval()
    (steer,) = attach_configured_interventions(root, [InterventionCfg(name="s", site="@steer", book=str(path))])
    assert isinstance(steer, UnitIntervention) and steer.unit_indices == (1, 6)
    assert sum(p.numel() for p in steer.parameters()) == 2
    with pytest.raises(ValueError, match="requires a UnitIntervention with units"):
        from interpretune.tuning.sites import book_sets

        attach_intervention(root, "full", nn.Identity(), "@steer", sets=book_sets(book))
