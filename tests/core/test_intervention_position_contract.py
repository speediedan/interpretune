"""An intervention changes exactly the positions its scope selects, and no other.

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

from interpretune.analysis.backends import PositionScope
from interpretune.analysis.backends.interventions import InterventionSpec, apply_intervention

BATCH, SEQ, D = 2, 5, 8
#: The core ops that change the activation's value; `replace_with_error` keeps it by construction (only its gradient
#: differs), so no position can read as changed under it.
VALUE_OPS = ("add", "clamp", "patch", "project", "reject", "replace", "scale")
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
    return InterventionSpec(intervention_tensor=tensor, op=mode, scale_factor=2.0, position_scope=scope, **extra)


def _changed_positions(before: torch.Tensor, after: torch.Tensor) -> set[int]:
    moved = ~torch.isclose(before, after, rtol=0, atol=1e-6)
    return {int(p) for p in torch.nonzero(moved.any(dim=-1).any(dim=0)).flatten().tolist()}


@pytest.mark.parametrize("mode", VALUE_OPS)
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


@pytest.mark.parametrize("mode", VALUE_OPS)
def test_explicit_positions_change_only_the_selected_set_for_every_value_op(mode):
    before = _activation()
    one = _spec(mode, "last_token")
    operands = torch.stack([one.intervention_tensor, torch.roll(one.intervention_tensor, 1, dims=-1)])
    spec = one._replace(position_scope="positions", positions=(3, 1), intervention_tensor=operands)
    after = apply_intervention(before.clone(), spec, last_pos=LAST)
    assert _changed_positions(before, after) == {1, 3}
    assert torch.equal(after[:, [0, 2, 4]], before[:, [0, 2, 4]])


def test_position_operand_order_and_padding_coordinates_are_preserved():
    before = torch.zeros(2, 5, 3)
    spec = InterventionSpec(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), position_scope="positions", positions=(3, 0)
    )
    result = apply_intervention(before.clone(), spec, last_pos=4)
    torch.testing.assert_close(result[:, 3], torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    torch.testing.assert_close(result[:, 0], torch.tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]))
    assert torch.equal(result[:, [1, 2, 4]], before[:, [1, 2, 4]])
    # Position zero means the padded tensor's first column for every row, independent of last_pos.
    assert torch.equal(result, apply_intervention(before.clone(), spec, last_pos=1))


def test_positions_compose_with_unit_indices():
    # Each position's operand row spans only the selected units, so the width check must narrow to them once: a
    # validation that narrowed again inside the per-position check would refuse unit 3 against a width of 2.
    before = torch.zeros(2, 5, 4)
    spec = InterventionSpec(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        op="replace",
        position_scope="positions",
        positions=(4, 0),
        unit_indices=(3, 1),
    )
    result = apply_intervention(before.clone(), spec, last_pos=2)
    expected = before.clone()
    expected[:, 4, 3], expected[:, 4, 1] = 1.0, 2.0
    expected[:, 0, 3], expected[:, 0, 1] = 3.0, 4.0
    assert torch.equal(result, expected)
    value = before.clone()
    with pytest.raises(ValueError, match="unit_indices"):
        apply_intervention(value, spec._replace(unit_indices=(4, 1)), last_pos=2)
    assert torch.equal(value, before)


@pytest.mark.parametrize("positions", [None, (), (1, 1), (-1,), (True,), (1.5,), "1"])
def test_invalid_position_sets_are_refused_before_mutation(positions):
    before = _activation()
    spec = InterventionSpec(torch.ones(2, D), position_scope="positions", positions=positions)
    value = before.clone()
    with pytest.raises(ValueError, match="position"):
        apply_intervention(value, spec, last_pos=LAST)
    assert torch.equal(value, before)


@pytest.mark.parametrize(
    "positions,tensor", [((0, SEQ), torch.ones(2, D)), ((0, 1), torch.ones(3, D)), ((0, 1), torch.ones(2, D + 1))]
)
def test_out_of_range_positions_and_mismatched_operands_are_atomic(positions, tensor):
    before = _activation()
    value = before.clone()
    with pytest.raises(ValueError, match="positions|shape"):
        apply_intervention(
            value, InterventionSpec(tensor, position_scope="positions", positions=positions), last_pos=LAST
        )
    assert torch.equal(value, before)


def test_positions_on_another_scope_are_refused():
    with pytest.raises(ValueError, match="only valid"):
        apply_intervention(_activation(), InterventionSpec(torch.ones(D), positions=(1,)), last_pos=LAST)


def test_position_metadata_survives_wildcard_canonicalization_and_scope_gating():
    from interpretune.analysis.backends.interventions import build_intervention_dict, require_intervention_support
    from interpretune.analysis.backends.capabilities import InterventionSupport
    from types import SimpleNamespace

    raw = {
        "blocks.*.hook_out": {
            "intervention_tensors": [torch.ones(2, D), torch.full((2, D), 2.0)],
            "positions": [3, 1],
            "position_scope": "positions",
            "op": "add",
        }
    }
    hooks = ["blocks.0.hook_out", "blocks.1.hook_out"]
    result = build_intervention_dict(raw, {"blocks.*.hook_out": hooks}, dict.fromkeys(hooks, (D,)))
    assert all(spec.positions == (3, 1) for specs in result.values() for spec in specs)
    limited = SimpleNamespace(intervention_support=InterventionSupport(position_scopes={"last_token"}, ops={"add"}))
    with pytest.raises(NotImplementedError, match="position_scope='positions'"):
        require_intervention_support(limited, raw, backend_name="limited")


def test_shorthand_carries_position_indices_to_canonicalization():
    from interpretune.analysis.backends.interventions import resolve_interventions, build_intervention_dict

    hook = "blocks.0.hook_out"
    fields = {
        "intervention_hook_pattern": hook,
        "intervention_tensor": torch.ones(2, D),
        "intervention_position_scope": "positions",
        "intervention_positions": [4, 1],
    }
    payload = resolve_interventions(analysis_batch={}, resolve_field=fields.get, load_json_field=lambda _: None)
    (spec,) = build_intervention_dict(payload, {hook: [hook]}, {hook: (D,)})[hook]
    assert spec.positions == (4, 1)


class TestAnOperandPerBatchRow:
    """``operand_per_row`` reads a positions operand as ``(batch, n_positions, ...)``: row $i$'s operands edit
    prompt $i$ only, for an edit whose value depends on the prompt."""

    @pytest.mark.parametrize("mode", VALUE_OPS)
    def test_each_row_gets_exactly_what_running_it_alone_gives(self, mode):
        before = _activation()
        one = _spec(mode, "last_token")
        rows = torch.stack([one.intervention_tensor, -0.5 * one.intervention_tensor])
        per_position = torch.stack([rows, torch.roll(rows, 1, dims=-1)], dim=1)  # (batch, n_positions, ...)
        spec = one._replace(
            position_scope="positions", positions=(3, 1), intervention_tensor=per_position, operand_per_row=True
        )
        after = apply_intervention(before.clone(), spec, last_pos=LAST)
        for row in range(BATCH):
            alone = one._replace(position_scope="positions", positions=(3, 1), intervention_tensor=per_position[row])
            expected = apply_intervention(before[row : row + 1].clone(), alone, last_pos=LAST)
            torch.testing.assert_close(after[row : row + 1], expected, rtol=0, atol=0)
        assert _changed_positions(before, after) == {1, 3}

    def test_rows_differ_where_their_operands_differ(self):
        before = torch.zeros(2, 4, 3)
        operands = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]])
        spec = InterventionSpec(operands, position_scope="positions", positions=(2,), operand_per_row=True)
        result = apply_intervention(before.clone(), spec, last_pos=3)
        torch.testing.assert_close(result[:, 2], torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
        assert torch.equal(result[:, [0, 1, 3]], before[:, [0, 1, 3]])

    def test_a_span_operand_keeps_its_own_leading_axis(self):
        # (batch, n_positions, k, d): each row removes its own two-dimensional span. Inferring the batch axis from
        # the shape could not tell this from a shared (n_positions, k, d) operand, which is why the flag is explicit.
        torch.manual_seed(2)
        before = torch.randn(2, 4, D)
        spans = torch.randn(2, 1, 2, D)
        spec = InterventionSpec(spans, op="reject", position_scope="positions", positions=(1,), operand_per_row=True)
        result = apply_intervention(before.clone(), spec, last_pos=3)
        for row in range(2):
            basis = torch.linalg.qr(spans[row, 0].T)[0]
            torch.testing.assert_close(result[row, 1] @ basis, torch.zeros(2), atol=1e-5, rtol=0)

    @pytest.mark.parametrize("operands", [torch.ones(3, 2, D), torch.ones(BATCH, 3, D), torch.ones(BATCH, 2, D + 1)])
    def test_a_mismatched_batch_position_or_width_axis_is_refused_before_mutation(self, operands):
        before = _activation()
        value = before.clone()
        spec = InterventionSpec(operands, position_scope="positions", positions=(0, 2), operand_per_row=True)
        with pytest.raises(ValueError, match="batch rows|per position|shape"):
            apply_intervention(value, spec, last_pos=LAST)
        assert torch.equal(value, before)

    def test_the_flag_on_another_scope_is_refused(self):
        spec = InterventionSpec(torch.ones(BATCH, D), position_scope="all_positions", operand_per_row=True)
        with pytest.raises(ValueError, match="operand_per_row is only valid"):
            apply_intervention(_activation(), spec, last_pos=LAST)

    def test_a_mapping_payload_carries_the_flag_through_canonicalization(self):
        from interpretune.analysis.backends.interventions import build_intervention_dict

        hook = "blocks.0.hook_out"
        raw = {
            hook: {
                "intervention_tensor": torch.ones(BATCH, 2, D),
                "positions": [3, 1],
                "position_scope": "positions",
                "operand_per_row": True,
            }
        }
        (spec,) = build_intervention_dict(raw, {hook: [hook]}, {hook: (D,)})[hook]
        assert spec.operand_per_row and spec.positions == (3, 1)
