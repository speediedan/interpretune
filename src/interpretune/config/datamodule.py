import os
from typing import Optional, Any, Dict, Tuple, List
import logging
from dataclasses import dataclass, field

from interpretune.config import ITSerializableCfg, ITSharedConfig
from interpretune.utils import rank_zero_warn


log = logging.getLogger(__name__)

################################################################################
# ITDatamodule Configuration Dataclasses
################################################################################

@dataclass(kw_only=True)
class PromptConfig(ITSerializableCfg):
    cust_task_prompt: Dict[str, Any] = field(default_factory=dict)

    def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: Optional[str] = None) -> str:
        return task_prompt.strip()

@dataclass(kw_only=True)
class TokenizationConfig(ITSerializableCfg):
    tokenizers_parallelism: bool = True
    local_fast_tokenizer_path: Optional[str] = None
    cust_tokenization_pattern: Optional[str] = None
    special_tokens_dict: Dict[str, Any] = field(default_factory=dict)
    max_seq_length: int = 2048   # TODO: force this to be set rather than allowing a default?


@dataclass(kw_only=True)
class DatasetProcessingConfig(ITSerializableCfg):
    remove_unused_columns: bool = True
    text_fields: Optional[Tuple] = None
    dataset_path: Optional[str] = None
    enable_datasets_cache: Optional[bool] = False  # disable caching unless explicitly set to improve reproducibility
    data_collator_cfg: Dict[str, Any] = field(default_factory=dict)
    signature_columns: Optional[List] = field(default_factory=list)
    prepare_data_map_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ITDataModuleConfig(ITSharedConfig, TokenizationConfig, DatasetProcessingConfig):
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    train_batch_size: int = 32
    eval_batch_size: int = 32
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    # note that for prompt_cfg, we:
    #   1. use (data)classes to minimize special character yaml parsing complications (can override w/ diff init_args)
    #   2. do not provide a default dataclass to avoid current dataclass subclass limitations
    prompt_cfg: PromptConfig = field(default_factory=PromptConfig)

    def __post_init__(self) -> None:
        # TODO: validate prompt_cfg validity
        self.dataloader_kwargs = {
            "num_workers": self.dataloader_kwargs.get("num_workers", 0),
            "pin_memory": self.dataloader_kwargs.get("pin_memory", False),
        }
        if not self.data_collator_cfg:
            self.data_collator_cfg = {"collator_class": "transformers.DataCollatorWithPadding"}
        default_dataset_save_path = f"{os.environ['HOME']}/.cache/huggingface/datasets/{self.task_name}"
        self.dataset_path = self.dataset_path or default_dataset_save_path

    def _cross_validate(self, it_cfg: ITSerializableCfg) -> None:
        # inspect tokenizer, tokenizer_name, model_name_or_path here, updating datamodule config before instantiation
        # if a value is missing in the datamodule config but present in the module config
        # we first inspect to see if we have a fallback `model_name_or_path`
        for dm_fallback_attr in ["tokenizer", "tokenizer_name", "model_name_or_path"]:
            if getattr(self, dm_fallback_attr) is None and getattr(it_cfg, dm_fallback_attr, None) is not None:
                rank_zero_warn(f"Since no datamodule `{dm_fallback_attr}` was provided, attempting to use fallback"
                               f" configuration, setting `{dm_fallback_attr}` to {getattr(it_cfg, dm_fallback_attr)}.")
                setattr(self, dm_fallback_attr, getattr(it_cfg, dm_fallback_attr))
