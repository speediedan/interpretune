from dataclasses import dataclass
from typing import Optional
from it_examples.experiments.rte_boolq import RTEBoolqPromptConfig

####################################
# Gemma2
####################################
# TODO: add option for using HF `tokenizer.apply_chat_template` api?
#       We usually want full control so lower priority atm, but will likely be valuable in the future.

@dataclass(kw_only=True)
class Gemma2PromptConfig:
    # see https://huggingface.co/google/gemma-2-2b-it for more details
    B_TURN: str = "<start_of_turn>"
    E_TURN: str = "<end_of_turn>"
    USER_ROLE: str = "user"
    ASSISTANT_ROLE: str = "model"

    def __post_init__(self) -> None:
        self.USER_ROLE_START = self.B_TURN + self.USER_ROLE + "\n"
        self.USER_ROLE_END = self.E_TURN + self.B_TURN + self.ASSISTANT_ROLE + "\n"

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: Optional[str] = None) -> str:
        if tokenization_pattern == "gemma2-chat":
            sequence = self.USER_ROLE_START + f"{task_prompt.strip()} {self.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence

@dataclass(kw_only=True)
class RTEBoolqGemma2PromptConfig(Gemma2PromptConfig, RTEBoolqPromptConfig):
    ...

# class PromptMeta(type):
#     def __new__(mcs, name, bases, classdict, **kwargs):
#         #component, input, ctx = mcs._validate_build_ctx(kwargs)
#         # TODO: add runtime checks for adherence to IT protocol here?
#         #composition_classes = mcs._map_composition_target(component, ctx)
#         #bases = (input, *composition_classes)
#         built_class = super().__new__(mcs, name, bases, classdict)
#         #built_class._orig_module_name = input.__qualname__
#         #built_class._composed_classes = composition_classes
#         return built_class

# RTEBoolqGemma2PromptConfig = dataclass(kw_only=True)(
#     type('RTEBoolqGemma2PromptConfig', (Gemma2PromptConfig, RTEBoolqPromptConfig), {})
# )

# class ComposedPromptClsFactory:
#     def __new__(cls, name: str, bases: Tuple[type, ...]) -> type:
#         return dataclass(kw_only=True)(type.__new__(cls, name, bases, {}))


####################################
# Llama3
####################################

@dataclass(kw_only=True)
class Llama3PromptConfig:
    # see https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md for more details
    sys_prompt: str = ("You are a helpful assistant.")
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
        self.SYS_ROLE_START = self.B_TEXT + self.SYS_ROLE_HEADER + "\n" + self.sys_prompt + self.E_TURN + \
            self.USER_ROLE_HEADER + "\n"
        self.USER_ROLE_END = self.E_TURN + self.ASSISTANT_ROLE_HEADER + "\n"

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: Optional[str] = None) -> str:
        if tokenization_pattern == "llama3-chat":
            sequence = self.SYS_ROLE_START + f"{task_prompt.strip()} {self.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence

@dataclass(kw_only=True)
class RTEBoolqLlama3PromptConfig(Llama3PromptConfig, RTEBoolqPromptConfig):
    ...
