"""The ``rte`` component's data path honours its datamodule configuration (#858, #861).

``dataloader_kwargs`` reach the training loader alone (evaluation splits score every example, in order), and
``max_seq_length`` bounds every encoded example by trimming its context field rather than the template or question.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def rte_datamodule_cls():
    from tests.rte_component import RTEBoolqDataModule

    return RTEBoolqDataModule


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


def _bare_datamodule(cls, dataloader_kwargs):
    dm = cls.__new__(cls)
    dm.dataset = {"train": list(range(10)), "validation": list(range(10))}
    dm.data_collator = None
    dm.itdm_cfg = SimpleNamespace(dataloader_kwargs=dataloader_kwargs, train_batch_size=4, eval_batch_size=4)
    return dm


def test_dataloader_kwargs_shuffle_and_drop_last_apply_to_train_only(rte_datamodule_cls):
    dm = _bare_datamodule(
        rte_datamodule_cls, {"num_workers": 0, "pin_memory": False, "shuffle": True, "drop_last": True}
    )
    train = dm.train_dataloader()
    assert isinstance(train.sampler, RandomSampler) and train.drop_last
    assert len(train) == 2
    for loader in (dm.val_dataloader(), dm.test_dataloader(), dm.predict_dataloader()):
        assert isinstance(loader.sampler, SequentialSampler) and not loader.drop_last
        assert len(loader) == 3


def _sequence_for(field1: str, field2: str) -> str:
    return f"Context: {field1}\nQuestion: {field2}? Answer:"


def test_fit_sequences_trims_only_the_context(rte_datamodule_cls, gpt2_tokenizer):
    long_context = " ".join(f"word{i}" for i in range(200))
    pairs = [("A short premise.", "Is it short"), (long_context, "Is it long")]
    sequences = [_sequence_for(*pair) for pair in pairs]
    max_seq_length = 64
    assert len(gpt2_tokenizer(sequences[1])["input_ids"]) > max_seq_length

    fitted = rte_datamodule_cls.fit_sequences(sequences, pairs, _sequence_for, gpt2_tokenizer, max_seq_length)

    assert fitted[0] == sequences[0]
    assert len(gpt2_tokenizer(fitted[1])["input_ids"]) <= max_seq_length
    assert fitted[1].startswith("Context: word0 word1")
    assert fitted[1].endswith("\nQuestion: Is it long? Answer:")


def test_fit_sequences_refuses_a_prompt_over_budget_without_its_context(rte_datamodule_cls, gpt2_tokenizer):
    pairs = [("some context", " ".join(["why"] * 40))]
    sequences = [_sequence_for(*pairs[0])]
    with pytest.raises(ValueError, match="without its context"):
        rte_datamodule_cls.fit_sequences(sequences, pairs, _sequence_for, gpt2_tokenizer, 16)
