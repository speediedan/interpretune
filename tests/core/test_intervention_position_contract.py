"""#441: an intervention changes exactly the positions its scope selects, and no other.

The last-token slice is correct by construction, but every backend calls it by convention and nothing verified
that the OTHER positions were left alone: a backend applying an intervention to every prompt position produces
entirely plausible logits, and a value check on the last position passes it. This is the shared contract, asserted
on the COUNT and the IDENTITY of changed positions across every mode and both scopes, at the one function every
bundled backend routes through. The conformance suite asserts the same set through each backend's runner path
(`test_last_token_scope_moves_exactly_the_final_position`, `test_all_positions_scope_moves_every_real_position`).
"""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.backends import InterventionMode, PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec, apply_intervention

BATCH, SEQ, D = 2, 5, 8
LAST = 3  # not the final index: left padding makes the last REAL token an argument, not a shape


def _activation() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D)


def _spec(mode: str, scope: str) -> InterventionSpec:
    torch.manual_seed(1)
    vector = torch.randn(D)
    tensor = torch.stack([vector, torch.roll(vector, 3)]) if mode == "patch" else vector
    # `clamp` needs a BAND, and a tight one: this test asserts which positions moved, so a band wide
    # enough to leave the activation inside it would report "nothing changed" for the right positions
    # and pass while testing nothing. A clamp with no band at all is refused by name rather than
    # silently behaving as the identity, which is why it cannot simply be omitted here.
    extra = {"clamp_min": -0.05, "clamp_max": 0.05} if mode == "clamp" else {}
    return InterventionSpec(intervention_tensor=tensor, mode=mode, scale_factor=2.0, position_scope=scope, **extra)


def _changed_positions(before: torch.Tensor, after: torch.Tensor) -> set[int]:
    moved = ~torch.isclose(before, after, rtol=0, atol=1e-6)
    return {int(p) for p in torch.nonzero(moved.any(dim=-1).any(dim=0)).flatten().tolist()}


@pytest.mark.parametrize("mode", [m.value for m in InterventionMode])
class TestExactlyTheSelectedPositionsChange:
    def test_last_token_changes_the_last_real_position_and_nothing_else(self, mode):
        before = _activation()
        after = apply_intervention(before.clone(), _spec(mode, PositionScope.LAST_TOKEN.value), last_pos=LAST)
        changed = _changed_positions(before, after)
        assert changed == {LAST}, f"mode {mode!r}: changed positions {sorted(changed)}, expected exactly {{{LAST}}}"

    def test_all_positions_changes_every_position(self, mode):
        before = _activation()
        after = apply_intervention(before.clone(), _spec(mode, PositionScope.ALL_POSITIONS.value), last_pos=LAST)
        assert _changed_positions(before, after) == set(range(SEQ)), f"mode {mode!r} left a position untouched"


class TestTheCountIsWhatCatchesAWholePromptBackend:
    """Positive control: a backend that applied the last-token intervention everywhere would pass a last-position
    value check and fail this contract on the count."""

    def test_a_whole_prompt_application_fails_the_last_token_contract(self):
        before = _activation()
        spec = _spec("add", PositionScope.LAST_TOKEN.value)
        everywhere = before.clone()
        everywhere[:, :, :] = (
            everywhere + spec.intervention_tensor * spec.scale_factor
        )  # what interp-engine did by default
        # the last position is exactly what the contract asks for...
        expected_last = apply_intervention(before.clone(), spec, last_pos=LAST)[:, LAST]
        assert torch.allclose(everywhere[:, LAST], expected_last)
        # ...and the count is what tells the two operations apart
        assert _changed_positions(before, everywhere) != {LAST}
