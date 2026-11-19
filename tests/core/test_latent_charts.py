"""Chart contracts: chart coordinates read and edited through an explicit frame, with registered kinds only."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from interpretune.analysis.venngram import charts
from interpretune.analysis.venngram.charts import AffineChart, LatentChart, charts_from_book, register_chart_kind
from interpretune.analysis.venngram.latents import LatentBinding, LatentCatalog, LatentFrame
from interpretune.utils import MisconfigurationException

F64 = torch.float64
# The frame's rows: the decoder plane's two orthonormal directions in a four-wide activation space.
ROWS = torch.tensor([[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=F64)


@pytest.fixture
def block():
    """A two-row decoder block, its frame, and values whose frame coordinates are (10, -6) plus an off-block 11."""
    decoder = nn.Linear(2, 4, bias=False, dtype=F64)
    with torch.no_grad():
        decoder.weight.copy_(torch.tensor([[3.0, 0.0], [4.0, 0.0], [0.0, 2.0], [0.0, 0.0]]))
    model = LatentBinding("block", 0, "hook_resid_post", decoder, decoder.weight.T, 2)
    (target,) = LatentCatalog([model]).resolve("features:block & units:[0]", n_layers=1).targets
    frame = LatentFrame("semantic", "v1", target, ROWS)
    values = torch.tensor([6.0, 8.0, -6.0, 11.0], dtype=F64)
    return model, target, frame, values


def plane(target, *, origin=(0.0, 0.0), axes=((1.0, 0.0), (0.0, 1.0)), **overrides):
    fields = dict(
        name="hue",
        version="v1",
        kind="pca_plane",
        target=target,
        frame="semantic",
        frame_version="v1",
        intrinsic_dim=2,
        periodic={1: 2 * math.pi},
        params={"origin": torch.tensor(origin, dtype=F64), "axes": torch.tensor(axes, dtype=F64)},
    )
    fields.update(overrides)
    return LatentChart(**fields)


def line(target, *, origin=(0.0, 0.0)):
    return LatentChart(
        "depth",
        "v1",
        "affine",
        target,
        "semantic",
        "v1",
        1,
        params={"origin": torch.tensor(origin, dtype=F64), "axes": torch.tensor([[1.0, 0.0]], dtype=F64)},
    )


def expected(values, old, new):
    """The edit an exact frame write makes: the frame coordinates move from ``old`` to ``new``, nothing else does."""
    return values + (torch.tensor(new, dtype=F64) - torch.tensor(old, dtype=F64)) @ ROWS


def test_a_plane_chart_reads_polar_coordinates_through_its_frame(block):
    model, target, frame, values = block
    catalog = LatentCatalog(
        [model], frames=[frame], charts=[plane(target), plane(target, origin=(4.0, 2.0), version="v2")]
    )
    read = catalog.chart_coordinates(target, values, chart="hue", version="v1")
    assert_close(read, torch.tensor([math.sqrt(136), math.atan2(-6, 10)], dtype=F64))
    read = catalog.chart_coordinates(target, values, chart="hue", version="v2")
    assert_close(read, torch.tensor([10.0, math.atan2(-8, 6)], dtype=F64))


def test_rotation_turns_the_plane_and_keeps_everything_off_the_target(block):
    model, target, frame, values = block
    catalog = LatentCatalog([model], frames=[frame], charts=[plane(target)])
    edit = dict(chart="hue", version="v1", op="rotate")
    # A quarter turn takes frame coordinates (10, -6) to (6, 10); the fourth activation axis is untouched.
    assert_close(catalog.apply_chart(target, values, by={1: math.pi / 2}, **edit), expected(values, [10, -6], [6, 10]))
    assert_close(catalog.apply_chart(target, values, by={1: 2 * math.pi}, **edit), values)


def test_scale_norm_set_and_a_path_across_the_seam(block):
    model, target, frame, values = block
    catalog = LatentCatalog([model], frames=[frame], charts=[plane(target)])
    edit = dict(chart="hue", version="v1")
    halved = catalog.apply_chart(target, values, op="scale_norm", factor=0.5, **edit)
    assert_close(halved, expected(values, [10, -6], [5, -3]))
    placed = catalog.apply_chart(target, values, op="set", value=[2.0, math.pi / 2], **edit)
    assert_close(placed, expected(values, [10, -6], [0, 2]))
    # Halfway from 350 to 10 degrees is 0 degrees, the short way round, not 180.
    waypoints = [[1.0, math.radians(350)], [1.0, math.radians(10)]]
    for t, angle in ((0.0, -10.0), (0.5, 0.0), (1.0, 10.0)):
        moved = catalog.apply_chart(target, values, op="path", waypoints=waypoints, t=t, **edit)
        point = [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
        assert_close(moved, expected(values, [10, -6], point))


def test_an_affine_chart_keeps_the_frame_coordinates_off_the_chart(block):
    model, target, frame, values = block
    catalog = LatentCatalog([model], frames=[frame], charts=[line(target, origin=(1.0, 0.0))])
    assert_close(catalog.chart_coordinates(target, values, chart="depth", version="v1"), values.new_tensor([9.0]))
    doubled = catalog.apply_chart(target, values, chart="depth", version="v1", op="scale_norm", factor=2)
    assert_close(doubled, expected(values, [10, -6], [19, -6]))
    with pytest.raises(MisconfigurationException, match="does not support 'rotate'; it supports"):
        catalog.apply_chart(target, values, chart="depth", version="v1", op="rotate", by={0: 1.0})


def test_chart_edits_keep_autograd_through_the_frame(block):
    model, target, frame, values = block
    catalog = LatentCatalog([model], frames=[frame], charts=[plane(target)])
    values = values.clone().requires_grad_(True)
    catalog.apply_chart(target, values, chart="hue", version="v1", op="rotate", by={1: math.pi / 2}).sum().backward()
    # A fixed rotation is linear in the values: v' = v M with M = I + R^T (Q^T - I) R.
    turn = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=F64)
    linear = torch.eye(4, dtype=F64) + ROWS.T @ (turn.T - torch.eye(2, dtype=F64)) @ ROWS
    assert_close(values.grad, linear @ torch.ones(4, dtype=F64))


def test_unregistered_kinds_are_refused_and_registration_is_explicit(block, monkeypatch):
    model, target, frame, values = block
    with pytest.raises(MisconfigurationException, match=r"chart kind 'spline' is not registered \(registered: \["):
        plane(target, kind="spline")
    monkeypatch.setattr(charts, "_CHART_KINDS", dict(charts._CHART_KINDS))

    class Ray(AffineChart):
        name = "ray"

    register_chart_kind(Ray())
    with pytest.raises(MisconfigurationException, match="'ray' is already registered"):
        register_chart_kind(Ray())
    ray = LatentChart("r", "v1", "ray", target, "semantic", "v1", 1, params=line(target).params)
    catalog = LatentCatalog([model], frames=[frame], charts=[ray])
    assert_close(catalog.chart_coordinates(target, values, chart="r", version="v1"), values.new_tensor([10.0]))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"periodic": {}}, "the second periodic with period 2.pi"),
        ({"intrinsic_dim": 0}, "positive integer intrinsic_dim"),
        ({"periodic": {5: 1.0}}, "periodic dimension 5 is outside"),
        ({"periodic": {1: -1.0}}, "positive finite period"),
        ({"params": {"origin": torch.zeros(2, dtype=F64)}}, r"takes params \['axes', 'origin'\]"),
        ({"params": {"origin": torch.zeros(2, dtype=F64), "axes": torch.ones(2, 2, dtype=F64)}}, "not orthonormal"),
        (
            {"params": {"origin": torch.zeros(3, dtype=F64), "axes": torch.eye(2, dtype=F64)}},
            r"'origin' with shape \(2,\)",
        ),
        ({"frame_version": "v9"}, "unknown latent frame 'semantic' version 'v9'"),
    ],
)
def test_invalid_charts_are_refused_by_name(block, overrides, message):
    model, target, frame, _ = block
    with pytest.raises(MisconfigurationException, match=message):
        LatentCatalog([model], frames=[frame], charts=[plane(target, **overrides)])


def test_chart_lookup_and_op_arguments_never_guess(block):
    model, target, frame, values = block
    catalog = LatentCatalog([model], frames=[frame], charts=[plane(target)])
    with pytest.raises(MisconfigurationException, match="unknown latent chart 'hue' version 'v2'"):
        catalog.chart_coordinates(target, values, chart="hue", version="v2")
    with pytest.raises(MisconfigurationException, match="duplicate latent chart"):
        LatentCatalog([model], frames=[frame], charts=[plane(target), plane(target)])
    edit = dict(chart="hue", version="v1")
    with pytest.raises(MisconfigurationException, match="does not support 'flip'"):
        catalog.apply_chart(target, values, op="flip", **edit)
    with pytest.raises(MisconfigurationException, match=r"takes \['factor'\], got \['by'\]"):
        catalog.apply_chart(target, values, op="scale_norm", by={1: 1.0}, **edit)
    with pytest.raises(MisconfigurationException, match="rotates only its periodic dimensions"):
        catalog.apply_chart(target, values, op="rotate", by={0: 1.0}, **edit)
    with pytest.raises(MisconfigurationException, match=r"t must lie in \[0, 1\]"):
        catalog.apply_chart(target, values, op="path", waypoints=[[1.0, 0.0], [1.0, 1.0]], t=1.5, **edit)


def test_charts_travel_in_an_address_book(block):
    from interpretune.analysis.venngram.sites import _venngram

    model, target, frame, values = block
    payload = {
        "version": "v1",
        "kind": "pca_plane",
        "target": {"kind": "feature", "name": "block", "layer": 0, "point": "hook_resid_post", "unit": 0},
        "frame": {"name": "semantic", "version": "v1"},
        "intrinsic_dim": 2,
        "periodic": {1: 2 * math.pi},
        "params": {"origin": [0.0, 0.0], "axes": [[1.0, 0.0], [0.0, 1.0]]},
    }
    doc = {"schema_version": 1, "sets": {"hued": "features:block & units:[0]"}, "charts": {"hue": payload}}
    book = _venngram("book").AddressBook.from_mapping(doc)
    (chart,) = charts_from_book(book)
    catalog = LatentCatalog([model], frames=[frame], charts=[chart])
    (resolved,) = catalog.resolve("@hued", n_layers=1, sets=book.sets()).targets
    assert resolved == target
    assert_close(catalog.chart_coordinates(target, values, chart="hue", version="v1")[0], values.new_tensor(136**0.5))
    with pytest.raises(MisconfigurationException, match=r"unknown keys \['basis'\]"):
        LatentChart.from_payload("hue", {**payload, "basis": []})
