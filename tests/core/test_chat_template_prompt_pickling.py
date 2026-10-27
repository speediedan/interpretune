"""A bound ``ChatTemplatePromptConfig`` stays loadable with ``weights_only=True`` (#765).

A Lightning datamodule's hyperparameters hold its config, so a checkpoint pickles the prompt config with whatever
tokenizer the datamodule bound to it. Lightning loads checkpoints with ``weights_only=True`` and refused the tokenizer,
which failed the gemma-3 RTE example's best-checkpoint restore at its first phase transition.
"""

from __future__ import annotations

import copy

import torch

from interpretune.config import ChatTemplatePromptConfig


class _Tokenizer:
    """A stand-in for a tokenizer: any object outside the safe globals."""

    chat_template = "{{ messages }}"


def test_a_pickled_config_leaves_the_bound_tokenizer_out(tmp_path):
    cfg = ChatTemplatePromptConfig(add_generation_prompt=False)
    tokenizer = _Tokenizer()
    cfg.bind_tokenizer(tokenizer)
    path = tmp_path / "ckpt.pt"
    torch.save({"datamodule_hyper_parameters": {"prompt_cfg": cfg}}, path)
    loaded = torch.load(path, weights_only=True)["datamodule_hyper_parameters"]["prompt_cfg"]
    assert isinstance(loaded, ChatTemplatePromptConfig)
    assert loaded.add_generation_prompt is False
    assert loaded._tokenizer is None
    assert cfg._tokenizer is tokenizer  # pickling does not unbind the original


def test_a_deepcopy_shares_the_bound_tokenizer():
    cfg = ChatTemplatePromptConfig(cust_task_prompt={"context": "c"})
    tokenizer = _Tokenizer()
    cfg.bind_tokenizer(tokenizer)
    clone = copy.deepcopy(cfg)
    assert clone._tokenizer is tokenizer
    assert clone.cust_task_prompt == cfg.cust_task_prompt and clone.cust_task_prompt is not cfg.cust_task_prompt
