from dataclasses import dataclass
from it_examples.experiments.rte_boolq import RTEBoolqPromptConfig

####################################
# Gemma (unified for Gemma2/Gemma3)
####################################
# Gemma2 and Gemma3 instruction-tuned models use the same <start_of_turn>/<end_of_turn> chat template.
# TODO: consider using HF `tokenizer.apply_chat_template` api instead of manual template construction
#       (see docs/apply_chat_template_proposal.md for details)


@dataclass(kw_only=True)
class GemmaPromptConfig:
    # see https://huggingface.co/google/gemma-2-2b-it and https://huggingface.co/google/gemma-3-1b-it
    B_TURN: str = "<start_of_turn>"
    E_TURN: str = "<end_of_turn>"
    USER_ROLE: str = "user"
    ASSISTANT_ROLE: str = "model"

    def __post_init__(self) -> None:
        self.USER_ROLE_START = self.B_TURN + self.USER_ROLE + "\n"
        self.USER_ROLE_END = self.E_TURN + self.B_TURN + self.ASSISTANT_ROLE + "\n"

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: str | None = None) -> str:
        if tokenization_pattern == "gemma-chat":
            sequence = self.USER_ROLE_START + f"{task_prompt.strip()} {self.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence


# Backward-compatible aliases
Gemma2PromptConfig = GemmaPromptConfig
Gemma3PromptConfig = GemmaPromptConfig


@dataclass(kw_only=True)
class RTEBoolqGemmaPromptConfig(GemmaPromptConfig, RTEBoolqPromptConfig): ...


# Backward-compatible aliases
RTEBoolqGemma2PromptConfig = RTEBoolqGemmaPromptConfig
RTEBoolqGemma3PromptConfig = RTEBoolqGemmaPromptConfig


####################################
# Llama3
####################################


@dataclass(kw_only=True)
class Llama3PromptConfig:
    # see https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md for more details
    sys_prompt: str = "You are a helpful assistant."
    B_TEXT: str = "<|begin_of_text|>"
    E_TEXT: str = "<|end_of_text|>"
    B_HEADER: str = "<|start_header_id|>"
    E_HEADER: str = "<|end_header_id|>"
    E_TURN: str = "<|eot_id|>"
    # tool tags, see https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
    # for tool prompt format details
    TOOL_TAG: str = "<|python_tag|>"
    E_TOOL_MSG: str = "<|eom_id|>"
    SYS_ROLE: str = "system"
    USER_ROLE: str = "user"
    ASSISTANT_ROLE: str = "assistant"
    TOOL_ROLE: str = "ipython"

    def __post_init__(self) -> None:
        self.SYS_ROLE_HEADER = self.B_HEADER + self.SYS_ROLE + self.E_HEADER
        self.USER_ROLE_HEADER = self.B_HEADER + self.USER_ROLE + self.E_HEADER
        self.ASSISTANT_ROLE_HEADER = self.B_HEADER + self.ASSISTANT_ROLE + self.E_HEADER
        self.SYS_ROLE_START = (
            self.B_TEXT + self.SYS_ROLE_HEADER + "\n" + self.sys_prompt + self.E_TURN + self.USER_ROLE_HEADER + "\n"
        )
        self.USER_ROLE_END = self.E_TURN + self.ASSISTANT_ROLE_HEADER + "\n"

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: str | None = None) -> str:
        if tokenization_pattern == "llama3-chat":
            sequence = self.SYS_ROLE_START + f"{task_prompt.strip()} {self.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence


@dataclass(kw_only=True)
class RTEBoolqLlama3PromptConfig(Llama3PromptConfig, RTEBoolqPromptConfig): ...
