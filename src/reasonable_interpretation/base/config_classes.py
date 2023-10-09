import os
from typing import Any, Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field

import torch
from lightning.pytorch.utilities import rank_zero_warn, rank_zero_info
import numpy as np

TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"


@dataclass
class LMGenerationConfig:
    max_new_tokens: int = 5  # nb maxing logits over multiple tokens (n<=5) will yield a very slight perf gain versus 1
    do_sample: bool = True
    top_p: float = 1.0
    temperature: float = 1.0
    use_cache: bool = True
    top_k: int = 50
    repetition_penalty: float = 1.0
    output_attentions: bool = False
    output_hidden_states: bool = False
    length_penalty: float = 1.0
    output_scores: bool = True
    return_dict_in_generate: bool = True

@dataclass
class RIZeroShotClassificationConfig:
    enabled: bool = False
    entailment_mapping: Tuple = ("Yes", "No")  # RTE style, invert mapping for BoolQ
    entailment_mapping_indices: Optional[torch.Tensor] = None
    lm_generation_cfg: LMGenerationConfig = field(default_factory=lambda: LMGenerationConfig())

@dataclass
class DebugLMConfig:
    enabled: bool = False
    debug_raw_preds: np.ndarray = field(default_factory=lambda: np.array([]))
    debug_raw_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    debug_raw_sequences: Optional[List[str]] = None
    record_memory_history: bool = False  # only enable for debugging/memory analysis
    raw_debug_sequences: List = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.raw_debug_sequences) == 0 and self.enabled:
            self.raw_debug_sequences = ['What is the color of a banana?', 'List the first 5 letters in the alphabet.',
                                        'How many days in a week?', 'How old is Barack Obama?']

@dataclass
class PromptConfig:
    ctx_question_join: str = 'Does the previous passage imply that '
    question_suffix: str = '? Answer with only one word, either Yes or No.'

# TODO: once 3.10 is the minimum, make kw_only to leverage dataclass inheritence for shared fields between LM/LDM
@dataclass
class RIDataModuleConfig:
    model_name_or_path: str
    task_name: str = DEFAULT_TASK
    os_env_model_auth_key: Optional[str] = None
    tokenizer_id_overrides: Optional[Dict] = field(default_factory=dict)
    max_seq_length: int = 2048
    train_batch_size: int = 32
    eval_batch_size: int = 32
    remove_unused_columns: bool = True
    tokenizers_parallelism: bool = True
    text_fields: Optional[Tuple] = None
    defer_model_init: Optional[bool] = False
    local_fast_tokenizer_path: Optional[str] = None
    dataset_path: Optional[str] = None
    cust_tokenization_pattern: Optional[str] = None
    prepare_validation_set_only: Optional[bool] = False
    enable_datasets_cache: Optional[bool] = False  # disable caching unless explicitly set to improve reproducibility
    TASK_TEXT_FIELD_MAP = {"rte": ("premise", "hypothesis"), "boolq": ("passage", "question")}
    # note that for prompt_cfg, we:
    #   1. use (data)classes to minimize special character yaml parsing complications (can override w/ diff init_args)
    #   2. do not provide a default dataclass to avoid current dataclass subclass limitations
    prompt_cfg: Optional[Any] = None
    signature_columns: Optional[List] = field(default_factory=list)
    cust_task_prompt: Dict[str, Any] = field(default_factory=dict)
    special_tokens_dict: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_collator_cfg: Dict[str, Any] = field(default_factory=dict)
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TODO: validate prompt_cfg validity
        if self.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(f"Invalid task_name {self.task_name!r}. Proceeding with the default task: {DEFAULT_TASK!r}")
            self.task_name = DEFAULT_TASK
        self.text_fields = self.TASK_TEXT_FIELD_MAP[self.task_name]
        self.dataloader_kwargs = {
            "num_workers": self.dataloader_kwargs.get("num_workers", 0),
            "pin_memory": self.dataloader_kwargs.get("pin_memory", False),
        }
        default_dataset_save_path = f"{os.environ['HOME']}/.cache/huggingface/datasets/{self.task_name}"
        self.dataset_path = self.dataset_path or default_dataset_save_path
        if self.defer_model_init:
            assert self.signature_columns is not None, ("`signature_columns` must be specified if `defer_model_init` "
                                                        "is set to True")

@dataclass
class RIConfig:
    """Dataclass to encapsulate the RIModuleinternal state."""
    model_name_or_path: str
    task_name: str = DEFAULT_TASK
    os_env_model_auth_key: Optional[str] = None
    tokenizer_id_overrides: Optional[Dict] = field(default_factory=dict)
    experiment_tag: Optional[str] = "default"
    torch_dtype: Optional[Union[str, torch.dtype]] = None
    log_env_details: Optional[bool] = True
    model_class: Optional[torch.nn.Module] = None
    activation_checkpointing: Optional[bool] = True
    defer_model_init: Optional[bool] = False
    use_model_cache: Optional[bool] = False
    zero_shot_cfg: RIZeroShotClassificationConfig = field(default_factory=lambda: RIZeroShotClassificationConfig())
    debug_lm_cfg: DebugLMConfig = field(default_factory=lambda: DebugLMConfig())
    optimizer_init: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler_init: Dict[str, Any] = field(default_factory=dict)
    pl_lrs_cfg: Dict[str, Any] = field(default_factory=dict)
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    special_tokens_dict: Dict = field(default_factory=dict)
    auto_model_cfg: Dict[str, Any] = field(default_factory=dict)
    bitsandbytesconfig: Dict[str, Any] = field(default_factory=dict)
    lora_cfg: Dict[str, Any] = field(default_factory=dict)
    dynamic_module_cfg: Dict[str, Any] = field(default_factory=dict)
    from_pretrained_cfg: Dict[str, Any] = field(default_factory=dict)

    def _torch_dtype_serde(self) -> Optional[torch.dtype]:
        if self.from_pretrained_cfg.get('torch_dtype', None):
            if hasattr(torch, self.from_pretrained_cfg['torch_dtype']):
                return getattr(torch, self.from_pretrained_cfg.pop('torch_dtype'))
            elif hasattr(torch, self.from_pretrained_cfg['torch_dtype'].split(".")[-1]):
                return getattr(torch, self.from_pretrained_cfg.pop('torch_dtype').split(".")[-1])
            else:
                rank_zero_warn(f"The provided `torch_dtype` {self.from_pretrained_cfg.pop('torch_dtype')} could "
                                "not be resolved, attempting to proceed with `torch_dtype` unset.")
        return None

    def __post_init__(self) -> None:
        if self.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(f"Invalid task_name {self.task_name!r}. Proceeding with the default task: {DEFAULT_TASK!r}")
            self.task_name = DEFAULT_TASK
        self.num_labels = 0 if self.zero_shot_cfg.enabled else TASK_NUM_LABELS[self.task_name]
        # always use env specified access token, if loading a checkpoint with another session's token, don't trust it
        if 'token' in self.from_pretrained_cfg:
            del self.from_pretrained_cfg['token']
        self.torch_dtype = self._torch_dtype_serde()
        if self.torch_dtype and self.bitsandbytesconfig:
            rank_zero_info(f'Ignoring torch_dtype option `{self.torch_dtype}` because quantization config was passed.')
            self.torch_dtype = None
