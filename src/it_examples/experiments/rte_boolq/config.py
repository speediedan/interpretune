import os
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pprint import pformat

import torch

from interpretune.base.config.datamodule import PromptConfig
from interpretune.base.config.module import ITConfig
from interpretune.adapters.transformer_lens import ITLensConfig
from interpretune.base.config.mixins import ZeroShotClassificationConfig, BaseGenerationConfig, HFGenerationConfig

@dataclass(kw_only=True)
class RTEBoolqEntailmentMapping:
    entailment_mapping: Tuple = ("Yes", "No")  # RTE style, invert mapping for BoolQ
    entailment_mapping_indices: Optional[torch.Tensor] = None


@dataclass(kw_only=True)
class RTEBoolqZeroShotClassificationConfig(RTEBoolqEntailmentMapping, ZeroShotClassificationConfig):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=HFGenerationConfig)

    def __repr__(self):
        return f"Zero-Shot Classification Config: {os.linesep}{pformat(self.__dict__)}"


@dataclass(kw_only=True)
class RTEBoolqPromptConfig(PromptConfig):
    ctx_question_join: str = 'Does the previous passage imply that '
    question_suffix: str = '? Answer with only one word, either Yes or No.'
    cust_task_prompt: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True)
class RTEBoolqLlama3PromptConfig(RTEBoolqPromptConfig):
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


# add our custom model attributes
@dataclass(kw_only=True)
class RTEBoolqConfig(RTEBoolqEntailmentMapping, ITConfig):
    ...


@dataclass(kw_only=True)
class RTEBoolqTLConfig(RTEBoolqEntailmentMapping, ITLensConfig):
    ...
