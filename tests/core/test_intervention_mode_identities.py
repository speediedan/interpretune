"""The `reject`/`clamp` mode contract, stated algebraically.

#423's cross-tree oracle handed the adapter lane four identities that pin the mode CLASS rather than one case; they land
here as in-tree tests rather than living only in the adapter's suite. Each expectation is recomputed from an independent
expression (a fresh pseudoinverse, never the implementation's own arithmetic), and the fixture is guarded to be
discriminating: with an orthonormal basis the transpose and the pseudoinverse agree, and every test below would pass for
the wrong implementation.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends.interventions import (
    InterventionSpec,
    _apply_span_clamp,
    _apply_span_rejection,
)


@pytest.fixture
def pair() -> tuple[torch.Tensor, torch.Tensor]:
    """A non-orthonormal pair plus an activation with an out-of-span component."""
    v = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]])
    h = torch.tensor([5.0, -3.0, 2.0, 1.0])
    return v, h


def _pinv_coords(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """`V^+ h`, independently of either implementation: `V` has rows, so the matrix is `V^T`."""
    return torch.linalg.pinv(v.transpose(0, 1)) @ h


def _reject(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    spec = InterventionSpec(intervention_tensor=v, mode="reject", scale_factor=1.0)
    return _apply_span_rejection(spec, input_value=h.reshape(1, -1), target=v)[0]


def _clamp(v: torch.Tensor, h: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    spec = InterventionSpec(intervention_tensor=v, mode="clamp", clamp_min=lo, clamp_max=hi)
    return _apply_span_clamp(spec, input_value=h.reshape(1, -1), target=v)[0]


class TestTheFixtureIsDiscriminating:
    def test_transpose_and_pseudoinverse_coordinates_differ(self, pair):
        """Without this, every identity below would also hold for a `V^T` implementation."""
        v, h = pair
        transpose_coords = h @ v.transpose(0, 1)
        assert not torch.allclose(transpose_coords, _pinv_coords(v, h), atol=1e-3)


class TestRejectIdentities:
    def test_reject_is_idempotent(self, pair):
        v, h = pair
        torch.testing.assert_close(_reject(v, _reject(v, h)), _reject(v, h), atol=1e-5, rtol=1e-5)

    def test_reject_empties_the_span(self, pair):
        """The coordinates of a rejected activation are zero: nothing projectable remains."""
        v, h = pair
        torch.testing.assert_close(_pinv_coords(v, _reject(v, h)), torch.zeros(2), atol=1e-5, rtol=1e-5)


class TestClampIdentities:
    def test_clamp_reaches_its_targets_exactly(self, pair):
        """The coordinates of a clamped activation are the clipped coordinates, no further, no less."""
        v, h = pair
        out = _clamp(v, h, lo=-1.0, hi=1.0)
        expected = torch.clamp(_pinv_coords(v, h), min=-1.0, max=1.0)
        torch.testing.assert_close(_pinv_coords(v, out), expected, atol=1e-5, rtol=1e-5)

    def test_clamp_preserves_the_orthogonal_complement(self, pair):
        """Rejecting after clamping equals rejecting directly: the clamp moved nothing outside the span.

        This separates a correct span operation from a plausible per-row approximation first, and it
        holds regardless of `k` or conditioning, which is why it belongs first in any suite.
        """
        v, h = pair
        torch.testing.assert_close(_reject(v, _clamp(v, h, lo=-1.0, hi=1.0)), _reject(v, h), atol=1e-5, rtol=1e-5)
