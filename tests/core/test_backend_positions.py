"""Every model backend counts positions over a padded row's real tokens, the convention TransformerLens 4.0
applies."""

from __future__ import annotations

import torch

from interpretune.adapters.nnsight.backends import _install_mask_derived_positions
from interpretune.analysis.backends.positions import accepts_position_ids, mask_derived_position_ids

LEFT_PADDED = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])


class _Recorder(torch.nn.Module):
    """A forward that records the position ids it received."""

    def __init__(self):
        super().__init__()
        self.seen: list[torch.Tensor | None] = []

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        self.seen.append(position_ids)
        return input_ids


class _NoPositions(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        return input_ids


def test_a_left_padded_row_starts_its_real_tokens_at_zero():
    ids = mask_derived_position_ids(LEFT_PADDED)
    assert ids.tolist() == [[1, 1, 0, 1, 2], [0, 1, 2, 3, 4]]


def test_an_unpadded_row_is_arange():
    torch.testing.assert_close(mask_derived_position_ids(torch.ones(1, 4, dtype=torch.long)), torch.arange(4)[None])


def test_the_signature_check_reads_the_forward():
    assert accepts_position_ids(_Recorder().forward)
    assert not accepts_position_ids(_NoPositions().forward)


class TestTheNNsightPreHook:
    def test_a_padded_forward_receives_mask_derived_positions(self):
        model = _Recorder()
        _install_mask_derived_positions(model)
        model(input_ids=torch.zeros_like(LEFT_PADDED), attention_mask=LEFT_PADDED)
        torch.testing.assert_close(model.seen[-1], mask_derived_position_ids(LEFT_PADDED))

    def test_positions_the_caller_passed_are_kept(self):
        model = _Recorder()
        _install_mask_derived_positions(model)
        explicit = torch.arange(5)[None].expand(2, 5)
        model(input_ids=torch.zeros_like(LEFT_PADDED), attention_mask=LEFT_PADDED, position_ids=explicit)
        assert model.seen[-1] is explicit

    def test_an_unpadded_forward_is_left_to_the_model(self):
        model = _Recorder()
        _install_mask_derived_positions(model)
        model(input_ids=torch.zeros(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long))
        assert model.seen[-1] is None

    def test_a_cached_decoding_step_is_left_alone(self):
        """The mask spans the cached prefix as well as the one new token, so positions derived from it would not
        match the input; the model derives them from the cache instead."""
        model = _Recorder()
        _install_mask_derived_positions(model)
        model(input_ids=torch.zeros(2, 1, dtype=torch.long), attention_mask=LEFT_PADDED)
        assert model.seen[-1] is None

    def test_installing_twice_registers_one_hook(self):
        model = _Recorder()
        _install_mask_derived_positions(model)
        _install_mask_derived_positions(model)
        assert len(model._forward_pre_hooks) == 1

    def test_a_model_without_position_ids_gets_no_hook(self):
        model = _NoPositions()
        _install_mask_derived_positions(model)
        assert len(model._forward_pre_hooks) == 0
