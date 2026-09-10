"""`clamp` bounds coordinates into a range; it does not assign them.

The distinction is the reason this is its own mode rather than a parameterization of `patch`. No choice
of bounds makes a clamp swap two coordinates, and no choice of targets makes an assignment leave an
in-range activation alone. They share an English word and nothing else, which is how they nearly shared
a name.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends.capabilities import InterventionMode
from interpretune.analysis.backends.interventions import (
    InterventionSpec,
    _apply_mode_to_region,
    _apply_span_clamp,
)


@pytest.fixture
def basis() -> torch.Tensor:
    """Two NON-orthonormal rows, so a `V^T` implementation gives a different answer than `V^+`."""
    return torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])


def _spec(basis: torch.Tensor, **kw) -> InterventionSpec:
    return InterventionSpec(basis, mode=InterventionMode.CLAMP, **kw)


class TestClampBoundsRatherThanAssigns:
    def test_an_out_of_range_coordinate_is_moved_to_the_bound(self, basis):
        h = torch.tensor([[5.0, 0.0, 2.0]])
        out = _apply_span_clamp(_spec(basis, clamp_min=-1.0, clamp_max=1.0), input_value=h, target=basis)
        coords = out.reshape(1, -1) @ torch.linalg.pinv(basis.transpose(0, 1)).transpose(0, 1)
        assert torch.all(coords <= 1.0 + 1e-5) and torch.all(coords >= -1.0 - 1e-5)

    def test_an_in_range_activation_is_untouched(self, basis):
        """The property that separates bounding from assigning: assignment always moves the coordinate."""
        h = torch.tensor([[0.2, 0.1, 2.0]])
        out = _apply_span_clamp(_spec(basis, clamp_min=-1.0, clamp_max=1.0), input_value=h, target=basis)
        assert torch.allclose(out, h, atol=1e-6)

    def test_the_orthogonal_complement_survives(self, basis):
        """Only the coordinates along the span move; the third axis is outside it and must not."""
        h = torch.tensor([[5.0, -5.0, 2.0]])
        out = _apply_span_clamp(_spec(basis, clamp_min=-1.0, clamp_max=1.0), input_value=h, target=basis)
        assert out[0, 2] == pytest.approx(2.0, abs=1e-6)

    def test_one_sided_bounds_are_independent(self, basis):
        """A floor alone must not impose a ceiling, and vice versa: `min`/`max` are separately optional."""
        h = torch.tensor([[9.0, 0.0, 0.0]])
        floored = _apply_span_clamp(_spec(basis, clamp_min=-1.0), input_value=h, target=basis)
        assert torch.allclose(floored, h, atol=1e-6), "a floor moved a coordinate that was above it"
        capped = _apply_span_clamp(_spec(basis, clamp_max=1.0), input_value=h, target=basis)
        assert not torch.allclose(capped, h, atol=1e-6)

    def test_it_is_idempotent(self, basis):
        """Clamping a clamped activation changes nothing: the second pass finds every coordinate in range."""
        h = torch.tensor([[5.0, -5.0, 2.0]])
        once = _apply_span_clamp(_spec(basis, clamp_min=-1.0, clamp_max=1.0), input_value=h, target=basis)
        twice = _apply_span_clamp(_spec(basis, clamp_min=-1.0, clamp_max=1.0), input_value=once, target=basis)
        assert torch.allclose(once, twice, atol=1e-6)

    def test_coordinates_are_pseudoinverse_not_transpose(self, basis):
        """With a non-orthonormal basis the two differ, and `V^T` bounds a quantity the caller did not name."""
        h = torch.tensor([[5.0, 5.0, 0.0]])
        out = _apply_span_clamp(_spec(basis, clamp_min=-1.0, clamp_max=1.0), input_value=h, target=basis)
        flat = h.reshape(1, -1)
        transpose_coords = flat @ basis.transpose(0, 1)
        pinv_coords = flat @ torch.linalg.pinv(basis.transpose(0, 1)).transpose(0, 1)
        assert not torch.allclose(transpose_coords, pinv_coords, atol=1e-3), "fixture is not discriminating"
        wrong = flat + (torch.clamp(transpose_coords, -1.0, 1.0) - transpose_coords) @ basis
        assert not torch.allclose(out, wrong.reshape(h.shape), atol=1e-4)


class TestClampRefusesAProvableNoOp:
    """A clamp with no band is the identity for every input, and completes returning plausible logits."""

    def test_neither_bound_is_refused_by_name(self, basis):
        h = torch.tensor([[5.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match=r"neither `clamp_min` nor `clamp_max`"):
            _apply_span_clamp(_spec(basis), input_value=h, target=basis)

    def test_an_inverted_range_is_refused_rather_than_ordered(self, basis):
        h = torch.tensor([[5.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match=r"clamp_min=.* greater than clamp_max"):
            _apply_span_clamp(_spec(basis, clamp_min=1.0, clamp_max=-1.0), input_value=h, target=basis)

    def test_the_refusal_is_not_merely_a_width_check(self, basis):
        """POSITIVE CONTROL: a spec identical but for a band must succeed, so the refusal is about the band."""
        h = torch.tensor([[5.0, 0.0, 0.0]])
        out = _apply_span_clamp(_spec(basis, clamp_max=1.0), input_value=h, target=basis)
        assert not torch.allclose(out, h)

    def test_a_width_mismatch_is_refused_rather_than_broadcast(self):
        wrong = torch.tensor([[1.0, 0.0]])
        h = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="width"):
            _apply_span_clamp(_spec(wrong, clamp_max=1.0), input_value=h, target=wrong)


class TestClampReachesTheSharedDispatch:
    def test_it_dispatches_through_the_shared_region_path(self, basis):
        """Reaching the mode by direct call only is how `patch` nearly shipped unreachable (#320)."""
        h = torch.tensor([[5.0, -5.0, 2.0]])
        spec = _spec(basis, clamp_min=-1.0, clamp_max=1.0)
        direct = _apply_span_clamp(spec, input_value=h, target=basis)
        dispatched = _apply_mode_to_region(h, spec)
        assert torch.allclose(direct, dispatched, atol=1e-6)
