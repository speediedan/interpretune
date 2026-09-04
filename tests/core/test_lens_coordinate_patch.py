"""#273 direction iii: lens-coordinate patching, the J-space paper's write approach.

`h <- h + V(sigma(c) - c)` swaps a concept pair in lens coordinates. The property that distinguishes it
from naive steering (`h <- h + alpha * v`) is that the component orthogonal to the pair is untouched.
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends.interventions import InterventionSpec, apply_intervention


def _patch(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    value = torch.zeros(1, 2, h.shape[-1], dtype=h.dtype)
    value[:, 1, :] = h
    spec = InterventionSpec(intervention_tensor=torch.stack([v_s, v_t]), mode="patch", scale_factor=scale)
    return apply_intervention(value, spec, last_pos=1)[0, 1]


class TestLensCoordinatePatch:
    def test_orthonormal_pair_exchanges_coordinates(self):
        torch.manual_seed(0)
        basis, _ = torch.linalg.qr(torch.randn(6, 6))
        v_s, v_t = basis[:, 0], basis[:, 1]
        h = 3.0 * v_s + 1.0 * v_t + 5.0 * basis[:, 2]
        out = _patch(h, v_s, v_t)
        assert out @ v_s == pytest.approx(1.0, abs=1e-4)
        assert out @ v_t == pytest.approx(3.0, abs=1e-4)

    def test_orthogonal_component_is_untouched(self):
        """The defining property.

        Naive steering cannot offer it: `alpha * v` perturbs whatever it overlaps.
        """
        torch.manual_seed(0)
        basis, _ = torch.linalg.qr(torch.randn(6, 6))
        v_s, v_t = basis[:, 0], basis[:, 1]
        h = 3.0 * v_s + 1.0 * v_t + 5.0 * basis[:, 2] - 2.0 * basis[:, 3]
        out = _patch(h, v_s, v_t)
        for idx in (2, 3, 4, 5):
            assert out @ basis[:, idx] == pytest.approx(float(h @ basis[:, idx]), abs=1e-4)

    def test_non_orthonormal_pair_still_lands_on_sigma_of_c(self):
        """Why the implementation uses the pseudoinverse rather than a transpose.

        Orthogonal preservation holds either way -- the update lies in span(V) by construction -- so it cannot
        distinguish the two. The coordinates can: only `V^+` makes the patched activation satisfy `V^+ h' == sigma(c)`.
        A transpose leaves it elsewhere while still looking plausible, which is precisely why this is asserted rather
        than assumed.
        """
        torch.manual_seed(0)
        v_s = torch.randn(6)
        v_t = v_s + 0.3 * torch.randn(6)  # strongly correlated: transpose and pinv genuinely differ
        assert abs(float(torch.nn.functional.cosine_similarity(v_s, v_t, dim=0))) > 0.5, "pair must be oblique"
        h = torch.randn(6)

        v_matrix = torch.stack([v_s, v_t]).transpose(0, 1)
        coords = torch.linalg.pinv(v_matrix) @ h
        out = _patch(h, v_s, v_t)
        after = torch.linalg.pinv(v_matrix) @ out
        assert after.tolist() == pytest.approx(coords.flip(-1).tolist(), abs=1e-3)

        # the transpose alternative reaches different coordinates, so the assertion above has teeth
        naive = h + v_matrix @ ((v_matrix.transpose(0, 1) @ h).flip(-1) - v_matrix.transpose(0, 1) @ h)
        assert (torch.linalg.pinv(v_matrix) @ naive).tolist() != pytest.approx(coords.flip(-1).tolist(), abs=1e-3)

    def test_scale_factor_scales_the_swapped_coordinates(self):
        """The paper's optional alpha.

        It reports oversteering as a real failure mode, so 1.0 is the default.
        """
        torch.manual_seed(0)
        basis, _ = torch.linalg.qr(torch.randn(6, 6))
        v_s, v_t = basis[:, 0], basis[:, 1]
        h = 3.0 * v_s + 1.0 * v_t
        out = _patch(h, v_s, v_t, scale=2.0)
        assert out @ v_s == pytest.approx(2.0, abs=1e-4)
        assert out @ v_t == pytest.approx(6.0, abs=1e-4)

    def test_a_single_vector_is_rejected_rather_than_guessed(self):
        """A swap needs a pair.

        Accepting one vector would have to invent the partner.
        """
        h = torch.randn(1, 2, 6)
        spec = InterventionSpec(intervention_tensor=torch.randn(6), mode="patch")
        with pytest.raises(ValueError, match="exactly two lens vectors"):
            apply_intervention(h, spec, last_pos=1)

    def test_width_mismatch_names_both_widths(self):
        value = torch.randn(1, 2, 6)
        spec = InterventionSpec(intervention_tensor=torch.randn(2, 5), mode="patch")
        with pytest.raises(ValueError, match="width 5 .*6-dimensional"):
            apply_intervention(value, spec, last_pos=1)

    def test_batch_rows_are_patched_independently(self):
        torch.manual_seed(0)
        basis, _ = torch.linalg.qr(torch.randn(6, 6))
        v_s, v_t = basis[:, 0], basis[:, 1]
        value = torch.zeros(2, 2, 6)
        value[0, 1, :] = 3.0 * v_s + 1.0 * v_t
        value[1, 1, :] = -2.0 * v_s + 7.0 * v_t
        spec = InterventionSpec(intervention_tensor=torch.stack([v_s, v_t]), mode="patch")
        out = apply_intervention(value, spec, last_pos=1)
        assert out[0, 1] @ v_s == pytest.approx(1.0, abs=1e-4)
        assert out[1, 1] @ v_s == pytest.approx(7.0, abs=1e-4)
        assert out[1, 1] @ v_t == pytest.approx(-2.0, abs=1e-4)

    def test_unknown_mode_still_raises(self):
        """Adding a mode must not turn a typo into silent no-op behaviour."""
        value = torch.randn(1, 2, 6)
        spec = InterventionSpec(intervention_tensor=torch.randn(6), mode="patchh")
        with pytest.raises(ValueError, match="Unknown intervention mode"):
            apply_intervention(value, spec, last_pos=1)


class TestPatchReachesThroughTheValidatedPath:
    """The low-level tests above call `apply_intervention` directly and so bypass validation.

    That is exactly how this nearly shipped unreachable: `_validate_intervention_spec` gates every
    intervention arriving through the op surface, and it both allowlists the mode and requires the tensor
    to broadcast INTO the target slice. A `(2, d_model)` pair does not, so patch was rejected two ways
    while eight green low-level tests said otherwise.
    """

    def test_mode_is_accepted_by_the_validator(self):
        from interpretune.analysis.backends.interventions import _validate_intervention_spec

        spec = InterventionSpec(intervention_tensor=torch.randn(2, 6), mode="patch")
        validated = _validate_intervention_spec(spec, target_shape=(6,), hook_name="blocks.0.hook_resid_post")
        assert validated.mode == "patch"
        assert tuple(validated.intervention_tensor.shape) == (2, 6)

    def test_validator_checks_the_pair_width_against_the_hook(self):
        from interpretune.analysis.backends.interventions import _validate_intervention_spec

        spec = InterventionSpec(intervention_tensor=torch.randn(2, 5), mode="patch")
        with pytest.raises(ValueError, match="not compatible with hook"):
            _validate_intervention_spec(spec, target_shape=(6,), hook_name="blocks.0.hook_resid_post")

    def test_validator_rejects_a_single_vector(self):
        from interpretune.analysis.backends.interventions import _validate_intervention_spec

        spec = InterventionSpec(intervention_tensor=torch.randn(6), mode="patch")
        with pytest.raises(ValueError, match="exactly two lens vectors"):
            _validate_intervention_spec(spec, target_shape=(6,), hook_name="blocks.0.hook_resid_post")

    @pytest.mark.parametrize("mode", ["replace", "add", "project"])
    def test_pre_existing_modes_are_unchanged_by_the_new_branch(self, mode):
        """The patch branch must not relax validation for the modes that were already there."""
        from interpretune.analysis.backends.interventions import _validate_intervention_spec

        assert (
            _validate_intervention_spec(
                InterventionSpec(intervention_tensor=torch.randn(6), mode=mode),
                target_shape=(6,),
                hook_name="h",
            ).mode
            == mode
        )
        # a (2, 6) tensor is still incompatible for these -- only patch takes a pair
        with pytest.raises(ValueError, match="not compatible with hook"):
            _validate_intervention_spec(
                InterventionSpec(intervention_tensor=torch.randn(2, 6), mode=mode), target_shape=(6,), hook_name="h"
            )

    def test_validator_still_rejects_an_unknown_mode(self):
        from interpretune.analysis.backends.interventions import _validate_intervention_spec

        with pytest.raises(ValueError, match="Unknown intervention mode"):
            _validate_intervention_spec(
                InterventionSpec(intervention_tensor=torch.randn(6), mode="patchh"), target_shape=(6,), hook_name="h"
            )
