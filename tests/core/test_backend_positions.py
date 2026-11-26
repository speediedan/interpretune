"""Every model backend counts positions over a padded row's real tokens, the convention TransformerLens 4.0
applies."""

from __future__ import annotations

from copy import deepcopy

import torch

from interpretune.analysis.backends.positions import (
    accepts_position_ids,
    install_mask_derived_positions,
    mask_derived_position_ids,
)

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


class TestMaskDerivedPositionHook:
    def test_a_padded_forward_receives_mask_derived_positions(self):
        model = _Recorder()
        install_mask_derived_positions(model)
        model(torch.zeros_like(LEFT_PADDED), LEFT_PADDED)
        torch.testing.assert_close(model.seen[-1], mask_derived_position_ids(LEFT_PADDED))

    def test_positions_the_caller_passed_are_kept(self):
        model = _Recorder()
        install_mask_derived_positions(model)
        explicit = torch.arange(5)[None].expand(2, 5)
        model(input_ids=torch.zeros_like(LEFT_PADDED), attention_mask=LEFT_PADDED, position_ids=explicit)
        assert model.seen[-1] is explicit
        model(torch.zeros_like(LEFT_PADDED), LEFT_PADDED, explicit)
        assert model.seen[-1] is explicit

    def test_an_unpadded_forward_is_left_to_the_model(self):
        model = _Recorder()
        install_mask_derived_positions(model)
        model(input_ids=torch.zeros(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long))
        assert model.seen[-1] is None

    def test_a_cached_decoding_step_is_left_alone(self):
        """The mask spans the cached prefix as well as the one new token, so positions derived from it would not
        match the input; the model derives them from the cache instead."""
        model = _Recorder()
        install_mask_derived_positions(model)
        model(input_ids=torch.zeros(2, 1, dtype=torch.long), attention_mask=LEFT_PADDED)
        assert model.seen[-1] is None

    def test_installing_twice_registers_one_hook(self):
        model = _Recorder()
        install_mask_derived_positions(model)
        install_mask_derived_positions(model)
        assert len(model._forward_pre_hooks) == 1

    def test_a_model_without_position_ids_gets_no_hook(self):
        model = _NoPositions()
        install_mask_derived_positions(model)
        assert len(model._forward_pre_hooks) == 0


@torch.inference_mode()
def test_core_and_nnsight_match_unpadded_gpt2_rows():
    from nnsight import TransformersModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from interpretune.adapters.nnsight.backends import NNsightModelBackend
    from interpretune.base.modules import BaseITModule
    from interpretune.config import ITConfig

    config = GPT2Config(
        n_layer=2, n_embd=32, n_head=4, n_positions=16, vocab_size=16, pad_token_id=0, bos_token_id=1, eos_token_id=2
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = GPT2LMHeadModel(config).eval()
    input_ids = torch.tensor([[0, 0, 3, 5, 7], [3, 5, 7, 9, 11]])
    mask = LEFT_PADDED
    kwargs = {"input_ids": input_ids, "attention_mask": mask}
    expected = torch.cat([reference(row[real.bool()][None]).logits[:, -1] for row, real in zip(input_ids, mask)])
    assert not torch.allclose(reference(**kwargs).logits[0, -1], expected[0])

    class CoreHF(BaseITModule):
        def model_init(self):
            self.model = deepcopy(reference)

    core = CoreHF(ITConfig(log_to_stream=False, log_to_file=False)).eval()
    word_level = WordLevel({str(i): i for i in range(config.vocab_size)}, unk_token="0")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(word_level), pad_token="0", eos_token="2")
    nnsight = TransformersModel(deepcopy(reference), tokenizer=tokenizer, task="text-generation")
    backend = NNsightModelBackend.for_model(nnsight)
    assert backend is not None

    torch.testing.assert_close(core(**kwargs).logits[:, -1], expected)
    torch.testing.assert_close(backend.fwd(nnsight, kwargs)[:, -1], expected)
