"""``ChatTemplatePromptConfig.chat_template_kwargs`` reaches the tokenizer's template (#869).

Qwen3's chat template reads ``enable_thinking``: left at its default, the model may open a reasoning block at the answer
position, so a one-token Yes or No answer is read from the wrong token. Both template paths pass the variables through.
"""

from __future__ import annotations

from interpretune.config import ChatTemplatePromptConfig


class _RecordingTokenizer:
    """Records the keyword arguments each ``apply_chat_template`` call receives."""

    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "rendered"


def test_both_template_paths_pass_the_template_variables():
    cfg = ChatTemplatePromptConfig(chat_template_kwargs={"enable_thinking": False})
    tokenizer = _RecordingTokenizer()
    cfg.bind_tokenizer(tokenizer)
    assert cfg.model_chat_template_fn("Answer Yes or No.") == "rendered"
    assert cfg.apply_chat_template_fn(tokenizer, "Answer Yes or No.") == "rendered"
    assert [call["enable_thinking"] for call in tokenizer.calls] == [False, False]


def test_no_template_variables_by_default():
    cfg = ChatTemplatePromptConfig()
    tokenizer = _RecordingTokenizer()
    cfg.bind_tokenizer(tokenizer)
    cfg.model_chat_template_fn("Answer Yes or No.")
    assert "enable_thinking" not in tokenizer.calls[0]
