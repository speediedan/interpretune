"""Latent identity and set-algebra contracts, independent of a backend's hook implementation."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from interpretune.analysis.venngram.latents import LatentBinding, LatentCatalog, LatentFrame, SubspaceBinding
from interpretune.utils import MisconfigurationException


def binding(name="sae", layer=1, *, block_size=1, point="hook_resid_post", space=None):
    model = nn.Linear(6, 4, bias=False)
    rows = model.weight.T
    return LatentBinding(name, layer, point, model, rows, block_size, space)


def test_blocks_keep_the_loaded_model_and_select_whole_decoder_row_groups():
    model = binding("bsf", block_size=3)
    catalog = LatentCatalog([model])
    (target,) = catalog.resolve("features:bsf & layers:1 & units:[1]", n_layers=3).targets
    assert catalog.binding(target).model is model.model
    assert target.unit == 1
    assert_close(catalog.basis(target), model.decoder_rows[3:6])


def test_an_sae_feature_and_a_block_use_the_same_address_grammar():
    catalog = LatentCatalog([binding(), binding("bsf", block_size=3)])
    for name, count in (("sae", 1), ("bsf", 3)):
        (target,) = catalog.resolve(f"features:{name} & units:[1]", n_layers=3).targets
        assert catalog.basis(target).shape == (count, 4)


def test_aliases_for_external_feature_ids_share_one_canonical_identity():
    catalog = LatentCatalog(
        [binding()],
        external_ids={
            "gpt2-small/1-res-example/2": "features:sae & layers:1 & units:[2]",
            "release/sae_id": "features:sae & layers:1",
            "ct-layer1-feature2": "features:sae & layers:1 & units:[2]",
        },
    )
    direct = catalog.resolve("features:sae & units:[2]", n_layers=3)
    for alias in ("gpt2-small/1-res-example/2", "ct-layer1-feature2"):
        assert catalog.resolve(f"features:{alias}", n_layers=3) == direct
    assert catalog.resolve("features:release/sae_id & units:[2]", n_layers=3) == direct


def test_book_algebra_per_layer_units_and_negative_block_indices():
    catalog = LatentCatalog([binding("bsf", layer=layer, block_size=3) for layer in (0, 2)])
    sets = {"all": "features:bsf", "last": "features:bsf & units:[-1]"}
    resolved = catalog.resolve("(@all - @last) & layers:even", n_layers=3, sets=sets)
    assert [(t.layer, t.unit) for t in resolved.targets] == [(0, 0), (2, 0)]
    resolved = catalog.resolve({"features": "bsf", "units": {0: [1], 2: "0:1"}}, n_layers=3)
    assert [(t.layer, t.unit) for t in resolved.targets] == [(0, 1), (2, 0)]


def test_feature_block_unit_slices_use_the_block_count():
    catalog = LatentCatalog([binding("bsf", block_size=3)])
    resolved = catalog.resolve("features:bsf & units:-1:", n_layers=3)
    assert [t.unit for t in resolved.targets] == [1]
    with pytest.raises(MisconfigurationException, match="(outside|range|index|indices)"):
        catalog.resolve("features:bsf & units:[2]", n_layers=3)


def test_subspace_identity_is_indivisible_and_book_filters_compose():
    basis = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    catalog = LatentCatalog([binding()], [SubspaceBinding("plane", 1, "hook_resid_post", basis)])
    resolved = catalog.resolve("@plane & layers:1", n_layers=3, sets={"plane": "subspace:plane"})
    assert len(resolved.targets) == 1
    assert_close(catalog.basis(resolved.targets[0]), basis)
    with pytest.raises(MisconfigurationException, match="indivisible"):
        catalog.resolve("subspace:plane & units:[0]", n_layers=3)
    selected = catalog.resolve("@sae & units:[0]", n_layers=3, sets={"sae": "features:sae"})
    assert len(selected.targets) == 1


def test_semantic_contribution_and_raw_module_output_remain_distinct():
    catalog = LatentCatalog([binding(point="hook_mlp_out"), binding("raw", point="mlp.hook_out")])
    result = catalog.resolve("point:hook_mlp_out", n_layers=3)
    assert len(result.targets) == 6
    assert {t.name for t in result.targets} == {"sae"}


def test_point_aliases_select_the_same_binding():
    catalog = LatentCatalog([binding(point="hook_resid_post")])
    assert catalog.resolve("point:hook_resid_post", n_layers=3) == catalog.resolve("point:hook_out", n_layers=3)


def test_flow_is_preserved_and_a_mixture_is_refused():
    catalog = LatentCatalog([binding()])
    assert catalog.resolve("features:sae & of:grad", n_layers=3).flow == "grad"
    with pytest.raises(MisconfigurationException, match="(mix|disagree|both)"):
        catalog.resolve("(features:sae & of:grad) | (features:sae & of:value)", n_layers=3)


def test_expect_counts_blocks_and_missing_book_references_are_refused():
    catalog = LatentCatalog([binding("bsf", block_size=3)])
    assert len(catalog.resolve({"features": "bsf", "expect": 2}, n_layers=3).targets) == 2
    with pytest.raises(MisconfigurationException, match="expected 6"):
        catalog.resolve({"features": "bsf", "expect": 6}, n_layers=3)
    with pytest.raises(MisconfigurationException, match="no set is named"):
        catalog.resolve("@missing", n_layers=3)


@pytest.mark.parametrize("address", ["features:missing", "subspace:missing", "role:base", "stack:vision"])
def test_unknown_or_unimplemented_qualifiers_are_refused(address):
    with pytest.raises(MisconfigurationException):
        LatentCatalog([binding()]).resolve(address, n_layers=3)


def test_external_aliases_cannot_cycle_shadow_models_or_change_flow():
    with pytest.raises(MisconfigurationException, match="shadow"):
        LatentCatalog([binding()], external_ids={"sae": "features:sae"})
    aliases = {"a": "features:b", "b": "features:a"}
    with pytest.raises(MisconfigurationException, match="refers to itself"):
        LatentCatalog([binding()], external_ids=aliases).resolve("features:a", n_layers=3)
    with pytest.raises(MisconfigurationException, match="gradient flow"):
        LatentCatalog([binding()], external_ids={"a": "features:sae & of:grad"}).resolve("features:a", n_layers=3)


def test_catalog_refuses_ambiguous_locations_invalid_block_widths_and_wrong_depth():
    with pytest.raises(MisconfigurationException, match="duplicate"):
        LatentCatalog([binding(), binding(space="jlens")])
    with pytest.raises(MisconfigurationException, match="divide"):
        binding(block_size=4)
    with pytest.raises(MisconfigurationException, match="exceeds depth"):
        LatentCatalog([binding(layer=3)]).resolve("features:sae", n_layers=3)


def test_rotating_a_blocks_rows_changes_coordinates_but_preserves_its_span():
    model = binding("bsf", block_size=3)
    catalog = LatentCatalog([model])
    (target,) = catalog.resolve("features:bsf & units:[0]", n_layers=3).targets
    rows = catalog.basis(target).double()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rotated = rotation @ rows
    assert_close(torch.linalg.pinv(rows) @ rows, torch.linalg.pinv(rotated) @ rotated)


@pytest.fixture
def framed_block():
    """A real decoder with a known plane and an independently specified orthonormal frame."""
    decoder = nn.Linear(2, 4, bias=False, dtype=torch.float64)
    with torch.no_grad():
        decoder.weight.copy_(torch.tensor([[3.0, 0.0], [4.0, 0.0], [0.0, 2.0], [0.0, 0.0]]))
    model = LatentBinding("block", 0, "hook_resid_post", decoder, decoder.weight.T, 2)
    catalog = LatentCatalog([model])
    (target,) = catalog.resolve("features:block & units:[0]", n_layers=1).targets
    rows = torch.tensor([[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    frame = LatentFrame("semantic", "v1", target, rows)
    return model, target, frame


def test_a_frame_changes_coordinates_without_changing_block_identity(framed_block):
    model, target, frame = framed_block
    other = LatentFrame("semantic", "v2", target, frame.rows.flip(0))
    catalog = LatentCatalog([model], frames=[frame, other])
    values = model.model(torch.tensor([2.0, -3.0], dtype=torch.float64))
    assert_close(values, torch.tensor([6.0, 8.0, -6.0, 0.0], dtype=torch.float64))
    assert_close(catalog.coordinates(target, values, frame="semantic", version="v1"), values.new_tensor([10, -6]))
    assert_close(catalog.coordinates(target, values, frame="semantic", version="v2"), values.new_tensor([-6, 10]))
    assert catalog.resolve("features:block & units:[0]", n_layers=1).targets == (target,)


def test_coordinate_replacement_preserves_the_complement_and_autograd(framed_block):
    model, target, frame = framed_block
    catalog = LatentCatalog([model], frames=[frame])
    values = torch.tensor([[[7.0, 1.0, 5.0, 11.0]]], dtype=torch.float64, requires_grad=True)
    coordinates = torch.tensor([[[2.0, -3.0]]], dtype=torch.float64, requires_grad=True)
    edited = catalog.replace_coordinates(target, values, coordinates, frame="semantic", version="v1")
    # The normal to the first frame row is fixed, and the fourth axis lies outside the decoder plane.
    assert_close(edited, values.new_tensor([[[5.2, -1.4, -3.0, 11.0]]]))
    edited.sum().backward()
    assert_close(values.grad, values.new_tensor([[[0.16, -0.12, 0.0, 1.0]]]))
    assert_close(coordinates.grad, values.new_tensor([[[1.4, 1.0]]]))


def test_a_frame_spans_a_named_subspace_and_dependent_decoder_rows(framed_block):
    model, target, frame = framed_block
    subspace = SubspaceBinding("plane", 0, "hook_resid_post", model.decoder_rows)
    catalog = LatentCatalog(subspaces=[subspace])
    (plane,) = catalog.resolve("subspace:plane", n_layers=1).targets
    payload = LatentFrame("plane-coordinates", "1", plane, frame.rows)
    assert LatentCatalog(subspaces=[subspace], frames=[payload]).frame(plane, name="plane-coordinates", version="1")
    with torch.no_grad():
        model.model.weight[:, 1].copy_(model.model.weight[:, 0] * 2)
    line = LatentFrame("line", "1", target, frame.rows[:1])
    assert LatentCatalog([model], frames=[line]).frame(target, name="line", version="1") is line
    with pytest.raises(MisconfigurationException, match="target rank is 1"):
        LatentCatalog([model], frames=[frame])


def test_frame_span_checks_are_invariant_to_decoder_rotations_and_scaling(framed_block):
    model, target, frame = framed_block
    with torch.no_grad():
        original = model.decoder_rows.clone()
        model.model.weight.copy_(torch.stack([original[1] * 1e-7, -original[0] * 1e-7], dim=1))
    catalog = LatentCatalog([model], frames=[frame])
    assert catalog.frame(target, name="semantic", version="v1") is frame


def test_same_width_and_rank_do_not_license_a_frame_outside_the_span(framed_block):
    model, target, _ = framed_block
    wrong = LatentFrame("wrong", "1", target, torch.eye(4, dtype=torch.float64)[[0, 2]])
    with pytest.raises(MisconfigurationException, match="does not span its target"):
        LatentCatalog([model], frames=[wrong])
    incomplete = LatentFrame("incomplete", "1", target, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    with pytest.raises(MisconfigurationException, match="target rank is 2"):
        LatentCatalog([model], frames=[incomplete])


def test_frame_lookup_never_guesses_a_version_or_an_owner(framed_block):
    model, target, frame = framed_block
    catalog = LatentCatalog([model, binding("other", layer=0)], frames=[frame])
    with pytest.raises(MisconfigurationException, match="unknown latent frame 'semantic' version 'latest'"):
        catalog.frame(target, name="semantic", version="latest")
    with pytest.raises(MisconfigurationException, match="unknown latent frame 'missing'"):
        catalog.frame(target, name="missing", version="v1")
    (other,) = catalog.resolve("features:other & units:[0]", n_layers=1).targets
    with pytest.raises(MisconfigurationException, match="belongs to"):
        catalog.frame(other, name="semantic", version="v1")
    with pytest.raises(MisconfigurationException, match="duplicate latent frame"):
        LatentCatalog([model], frames=[frame, frame])


def test_frames_are_snapshots_and_still_revalidate_mutable_payloads(framed_block):
    model, target, frame = framed_block
    supplied = frame.rows.clone()
    snapshot = LatentFrame("snapshot", "1", target, supplied)
    supplied.zero_()
    catalog = LatentCatalog([model], frames=[snapshot])
    assert_close(catalog.frame(target, name="snapshot", version="1").rows, frame.rows)
    snapshot.rows.zero_()
    with pytest.raises(MisconfigurationException, match="not orthonormal"):
        catalog.coordinates(target, torch.zeros(4), frame="snapshot", version="1")


def test_frames_revalidate_the_loaded_models_current_span(framed_block):
    model, target, frame = framed_block
    catalog = LatentCatalog([model], frames=[frame])
    with torch.no_grad():
        model.model.weight[3, 1] = 2
    with pytest.raises(MisconfigurationException, match="does not span its target"):
        catalog.frame(target, name="semantic", version="v1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "nonempty name"),
        ({"version": " "}, "nonempty version"),
        ({"atol": float("nan")}, "atol"),
        ({"rank_rtol": 0}, "rank_rtol"),
        ({"rows": torch.zeros(2, 4)}, "orthonormal"),
        ({"rows": torch.full((2, 4), float("inf"))}, "finite"),
        ({"rows": torch.eye(2, dtype=torch.int64)}, "floating-point"),
    ],
)
def test_invalid_frame_payloads_are_refused(framed_block, kwargs, message):
    _, target, frame = framed_block
    values = {"name": "frame", "version": "1", "target": target, "rows": frame.rows} | kwargs
    with pytest.raises(MisconfigurationException, match=message):
        LatentFrame(**values)


def test_coordinate_edits_refuse_wrong_shapes_nonfinite_values_and_implicit_casts(framed_block):
    model, target, frame = framed_block
    catalog = LatentCatalog([model], frames=[frame])
    for value in (torch.ones(3), torch.tensor(1.0), torch.ones(4, dtype=torch.int64), torch.full((4,), float("nan"))):
        with pytest.raises(MisconfigurationException, match="finite floating-point values"):
            catalog.coordinates(target, value, frame="semantic", version="v1")
    values = torch.ones(2, 4, dtype=torch.float64)
    for coords in (torch.ones(2), torch.ones(2, 2, dtype=torch.float32), torch.full((2, 2), float("inf"))):
        with pytest.raises(MisconfigurationException, match="finite coordinates"):
            catalog.replace_coordinates(target, values, coords, frame="semantic", version="v1")
