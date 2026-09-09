"""The `reject` intervention mode: remove the activation's component inside a span.

`reject` is the complement of `project`, and complements are the failure this file exists to catch: a
caller reaching for "project out" who gets `project` receives exactly the opposite component, with no
error and a plausible activation. So the cases below pin the mathematical relationship rather than only
that the activation moved.

Every expectation is computed from an independent expression (`h - M pinv(M) h` with `M = V^T`) rather
than from the implementation's own arithmetic, so a transcription error in the implementation cannot
make its own test pass.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends.capabilities import InterventionMode
from interpretune.analysis.backends.interventions import (
    InterventionSpec,
    _apply_mode_to_region,
    _apply_span_rejection,
)

D = 6


def _oracle(v: torch.Tensor, h: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """`h - alpha * V V^+ h`, written independently of the implementation."""
    m = v.transpose(0, 1).double()
    return (h.double() - alpha * (m @ (torch.linalg.pinv(m) @ h.double()))).float()


def _reject(v: torch.Tensor, h: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    spec = InterventionSpec(intervention_tensor=v, mode="reject", scale_factor=alpha)
    return _apply_span_rejection(spec, input_value=h.reshape(1, -1), target=v)[0]


@pytest.fixture
def pair():
    torch.manual_seed(0)
    return torch.randn(2, D), torch.randn(D)


class TestAgainstAnIndependentExpression:
    def test_matches_the_oracle(self, pair):
        v, h = pair
        torch.testing.assert_close(_reject(v, h), _oracle(v, h), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("k", [1, 2, 3, 5])
    def test_any_rank_is_accepted_unlike_patch(self, k):
        """`patch` requires exactly two vectors because a swap needs a partner; a rejection does not."""
        torch.manual_seed(k)
        v, h = torch.randn(k, D), torch.randn(D)
        torch.testing.assert_close(_reject(v, h), _oracle(v, h), atol=1e-5, rtol=1e-5)

    def test_a_single_vector_may_be_passed_unstacked(self):
        torch.manual_seed(3)
        v, h = torch.randn(D), torch.randn(D)
        torch.testing.assert_close(_reject(v, h), _oracle(v.reshape(1, D), h), atol=1e-5, rtol=1e-5)


class TestTheDefiningProperties:
    def test_the_span_component_is_gone(self, pair):
        """The operation's whole point, stated as a property rather than as an expected value."""
        v, h = pair
        m = v.transpose(0, 1).double()
        coords_after = torch.linalg.pinv(m) @ _reject(v, h).double()
        assert float(coords_after.abs().max()) < 1e-5

    def test_it_is_idempotent(self, pair):
        """Rejecting twice removes nothing further, because there is nothing left in the span."""
        v, h = pair
        once = _reject(v, h)
        torch.testing.assert_close(_reject(v, once), once, atol=1e-5, rtol=1e-5)

    def test_reject_and_project_are_complements_not_inverses(self, pair):
        """The relationship a caller most needs and the one most easily got backwards.

        At scale 1.0 the two modes partition the activation: what `project` keeps is exactly what
        `reject` removes. If this ever fails, one of the two has silently become the other.
        """
        v, h = pair
        single = v[:1]
        rejected = _reject(single, h)
        kept = h - rejected
        m = single.transpose(0, 1).double()
        in_span = (m @ (torch.linalg.pinv(m) @ h.double())).float()
        torch.testing.assert_close(kept, in_span, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("alpha, description", [(0.0, "identity"), (0.5, "half"), (1.0, "full")])
    def test_scale_factor_is_a_removal_fraction_not_a_magnitude(self, pair, alpha, description):
        """Graded ablation is one mode with a parameter rather than three modes."""
        v, h = pair
        torch.testing.assert_close(_reject(v, h, alpha), _oracle(v, h, alpha), atol=1e-5, rtol=1e-5)

    def test_scale_zero_leaves_the_activation_untouched(self, pair):
        v, h = pair
        torch.testing.assert_close(_reject(v, h, 0.0), h, atol=1e-6, rtol=0)


class TestItIsNotAnotherModeWearingTheName:
    """Positive controls.

    A mode that ignored its tensor entirely would pass 'the activation changed'.
    """

    def test_a_different_basis_gives_a_different_result(self, pair):
        v, h = pair
        torch.manual_seed(99)
        other = torch.randn(2, D)
        assert not torch.allclose(_reject(v, h), _reject(other, h), atol=1e-4)

    def test_it_differs_from_project_on_the_same_inputs(self, pair):
        """If these agreed, one mode would be dead and every test above would still pass."""
        v, h = pair
        rejected = _apply_mode_to_region(
            h.reshape(1, -1), InterventionSpec(intervention_tensor=v[:1], mode="reject", scale_factor=1.0)
        )
        projected = _apply_mode_to_region(
            h.reshape(1, -1),
            InterventionSpec(
                # `project` takes an UNSTACKED (d,) vector, which it broadcasts to the activation shape,
                # where `reject` takes a stacked (k, d) basis because it accepts any rank. The asymmetry
                # is real and worth knowing at a call site; it is not something this test can fix.
                intervention_tensor=v[0],
                mode="project",
                scale_factor=1.0,
                use_intervention_tensor_as_basis=True,
            ),
        )
        assert not torch.allclose(rejected, projected, atol=1e-4)

    def test_a_width_mismatch_is_refused_rather_than_broadcast(self):
        torch.manual_seed(5)
        v, h = torch.randn(2, D + 2), torch.randn(D)
        with pytest.raises(ValueError, match="width"):
            _reject(v, h)


class TestModeRegistration:
    def test_reject_is_a_declared_mode(self):
        assert InterventionMode("reject") is InterventionMode.REJECT

    def test_it_dispatches_through_the_shared_region_path(self, pair):
        """Routing through `_apply_mode_to_region` is what makes both position scopes work for free."""
        v, h = pair
        spec = InterventionSpec(intervention_tensor=v, mode="reject", scale_factor=1.0)
        torch.testing.assert_close(
            _apply_mode_to_region(h.reshape(1, -1), spec)[0], _oracle(v, h), atol=1e-5, rtol=1e-5
        )
