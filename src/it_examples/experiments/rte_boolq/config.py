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
class Llama2PromptConfig(RTEBoolqPromptConfig):
    sys_prompt: str = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as"
    " possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic,"
    " dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
    " If a question does not make any sense, or is not factually coherent, explain why instead of answering something"
    " not correct. If you don't know the answer to a question, please don't share false information.")
    B_INST: str = "[INST]"
    E_INST: str = "[/INST]"
    B_SYS: str = "<<SYS>>\n"
    E_SYS: str = "\n<</SYS>>\n\n"

    def __post_init__(self) -> None:
        self.SYS_PREFIX_START = f"{self.B_INST} " + self.B_SYS
        self.SYS_PREFIX = self.SYS_PREFIX_START + self.sys_prompt + self.E_SYS


# add our custom model attributes
@dataclass(kw_only=True)
class RTEBoolqConfig(RTEBoolqEntailmentMapping, ITConfig):
    ...


@dataclass(kw_only=True)
class RTEBoolqTLConfig(RTEBoolqEntailmentMapping, ITLensConfig):
    ...
