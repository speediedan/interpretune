"""Linear-space contracts: features expressed in a mapped space read and written through a site exactly."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.testing import assert_close

from interpretune.analysis.venngram.charts import LatentChart
from interpretune.analysis.venngram.latents import LatentBinding, LatentCatalog, LatentFrame
from interpretune.analysis.venngram.spaces import LinearSpace
from interpretune.utils import MisconfigurationException

F64 = torch.float64
# A three-wide site read into a two-dimensional space; (0, 1, -1) is the direction the map does not see.
MAP = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=F64)
NULL = torch.tensor([0.0, 1.0, -1.0], dtype=F64)
H = torch.tensor([1.0, 2.0, 3.0], dtype=F64)


def test_read_and_an_exact_pseudoinverse_write():
    space = LinearSpace("jlens", "v1", MAP)
    assert_close(space.read(H), H.new_tensor([2.0, 5.0]))
    # The pseudoinverse is M^T (M M^T)^-1 = [[1/2, 0], [0, 1/2], [0, 1/2]], so moving the first coordinate by 2 moves
    # the first site coordinate by 1.
    written = space.write(H, H.new_tensor([4.0, 5.0]))
    assert_close(written, H.new_tensor([2.0, 2.0, 3.0]))
    assert_close(space.read(written), H.new_tensor([4.0, 5.0]))
    assert_close(written @ NULL, H @ NULL)


def test_a_declared_inverse_is_used_rather_than_the_pseudoinverse():
    inverse = torch.tensor([[0.5, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=F64)
    space = LinearSpace("jlens", "v1", MAP, inverse)
    written = space.write(H, H.new_tensor([2.0, 6.0]))
    assert_close(written, H.new_tensor([1.0, 3.0, 3.0]))
    assert_close(space.read(written), H.new_tensor([2.0, 6.0]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"matrix": torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=F64)}, "has rank 1 for 2 space dimensions"),
        ({"inverse": torch.zeros(3, 2, dtype=F64)}, "cannot be written exactly"),
        ({"inverse": torch.zeros(2, 3, dtype=F64)}, r"inverse shape \(2, 3\) is not \(3, 2\)"),
        ({"matrix": torch.tensor([[1.0, math.inf]], dtype=F64)}, "finite floating-point matrix"),
        ({"version": " "}, "nonempty version"),
    ],
)
def test_maps_that_cannot_be_written_exactly_are_refused(kwargs, message):
    fields = {"name": "jlens", "version": "v1", "matrix": MAP, **kwargs}
    with pytest.raises(MisconfigurationException, match=message):
        LinearSpace(**fields)


@pytest.fixture
def mapped_block():
    """A two-row block whose decoder rows live in the two-dimensional space, with a frame over it."""
    decoder = nn.Linear(2, 2, bias=False, dtype=F64)
    with torch.no_grad():
        decoder.weight.copy_(torch.eye(2, dtype=F64))
    model = LatentBinding("jbsf", 0, "hook_resid_post", decoder, decoder.weight.T, 2, space="jlens")
    (target,) = LatentCatalog([model]).resolve("features:jbsf & space:jlens", n_layers=1).targets
    frame = LatentFrame("axes", "v1", target, torch.eye(2, dtype=F64))
    return model, target, frame


def test_framed_coordinates_are_read_and_replaced_through_the_site(mapped_block):
    model, target, frame = mapped_block
    catalog = LatentCatalog([model], frames=[frame], spaces=[LinearSpace("jlens", "v1", MAP)])
    h = H.clone().requires_grad_(True)
    assert_close(catalog.site_coordinates(target, h, frame="axes", version="v1"), H.new_tensor([2.0, 5.0]))
    edited = catalog.replace_site_coordinates(target, h, H.new_tensor([4.0, 5.0]), frame="axes", version="v1")
    assert_close(edited, H.new_tensor([2.0, 2.0, 3.0]))
    assert_close(catalog.site_coordinates(target, edited, frame="axes", version="v1"), H.new_tensor([4.0, 5.0]))
    edited.sum().backward()
    # The written value is h + (c - M h) pinv(M)^T with c fixed, so its gradient is the column sums of I - pinv(M) M.
    projector = torch.eye(3, dtype=F64) - torch.linalg.pinv(MAP) @ MAP
    assert_close(h.grad, projector.sum(0))


def test_a_chart_edit_reaches_the_site_through_the_space(mapped_block):
    model, target, frame = mapped_block
    plane = LatentChart(
        "turn",
        "v1",
        "pca_plane",
        target,
        "axes",
        "v1",
        2,
        periodic={1: 2 * math.pi},
        params={"origin": torch.zeros(2, dtype=F64), "axes": torch.eye(2, dtype=F64)},
    )
    catalog = LatentCatalog([model], frames=[frame], charts=[plane], spaces=[LinearSpace("jlens", "v1", MAP)])
    turned = catalog.apply_site_chart(target, H, chart="turn", version="v1", op="rotate", by={1: math.pi / 2})
    # A quarter turn takes space coordinates (2, 5) to (-5, 2).
    assert_close(catalog.to_space(target, turned), H.new_tensor([-5.0, 2.0]))
    assert_close(turned @ NULL, H @ NULL)


def test_spaces_are_registered_explicitly_and_must_fit_their_bindings(mapped_block):
    model, target, frame = mapped_block
    catalog = LatentCatalog([model], frames=[frame])
    with pytest.raises(MisconfigurationException, match=r"linear space 'jlens' is not registered .*registered: \[\]"):
        catalog.site_coordinates(target, H, frame="axes", version="v1")
    wide = LinearSpace("jlens", "v1", torch.eye(3, dtype=F64))
    with pytest.raises(MisconfigurationException, match="rows of width 2, but linear space 'jlens' has dimension 3"):
        LatentCatalog([model], spaces=[wide])
    with pytest.raises(MisconfigurationException, match="duplicate linear space 'jlens'"):
        LatentCatalog([model], spaces=[LinearSpace("jlens", "v1", MAP), LinearSpace("jlens", "v2", MAP)])


def test_a_binding_in_the_site_space_reads_the_site_directly():
    decoder = nn.Linear(3, 3, bias=False, dtype=F64)
    model = LatentBinding("sae", 0, "hook_resid_post", decoder, torch.eye(3, dtype=F64), 3)
    (target,) = LatentCatalog([model]).resolve("features:sae", n_layers=1).targets
    frame = LatentFrame("axes", "v1", target, torch.eye(3, dtype=F64))
    catalog = LatentCatalog([model], frames=[frame], spaces=[LinearSpace("jlens", "v1", MAP)])
    assert_close(catalog.site_coordinates(target, H, frame="axes", version="v1"), H)
