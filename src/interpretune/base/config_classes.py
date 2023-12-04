import os
from typing import Any, Dict, Optional, Tuple, List, Union, Literal
from dataclasses import dataclass, field
import logging

import yaml
import torch
import numpy as np


from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch.utilities import rank_zero_warn, rank_zero_info
else:
    from interpretune.utils.logging import rank_zero_warn, rank_zero_info  # type: ignore[no-redef]


log = logging.getLogger(__name__)

@dataclass(kw_only=True)
class ITSerializableCfg(yaml.YAMLObject):
    ...

def it_cfg_mapping_representer(dumper, data):
    return dumper.represent_mapping('!InterpretuneCfg', data.__dict__)


@dataclass(kw_only=True)
class LMGenerationConfig(ITSerializableCfg):
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


@dataclass(kw_only=True)
class ITZeroShotClassificationConfig(ITSerializableCfg):
    enabled: bool = False
    lm_generation_cfg: LMGenerationConfig = field(default_factory=lambda: LMGenerationConfig())

yaml.add_representer(ITSerializableCfg, it_cfg_mapping_representer)

@dataclass(kw_only=True)
class ITLensFromPretrainedConfig(ITSerializableCfg):
    enabled: bool = False
    model_name: str = "gpt2-small"
    fold_ln: Optional[bool] = True
    center_writing_weights: Optional[bool] = True
    center_unembed: Optional[bool] = True
    refactor_factored_attn_matrices: Optional[bool] = False
    checkpoint_index: Optional[int] = None
    checkpoint_value: Optional[int] = None
    # only supporting str for device for now due to omegaconf container dumping limitations
    device: Optional[str] = None
    n_devices: Optional[int] = 1
    move_to_device: Optional[bool] = True
    fold_value_biases: Optional[bool] = True
    default_prepend_bos: Optional[bool] = True
    default_padding_side: Optional[Literal["left", "right"]] = "right"
    dtype: Optional[str] = "float32"


@dataclass(kw_only=True)
class DebugLMConfig(ITSerializableCfg):
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


@dataclass(kw_only=True)
class PromptConfig(ITSerializableCfg):
    cust_task_prompt: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ITSharedConfig(ITSerializableCfg):
    model_name_or_path: str = ''
    task_name: str = ''
    os_env_model_auth_key: Optional[str] = None
    tokenizer_id_overrides: Optional[Dict] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    defer_model_init: Optional[bool] = False


@dataclass(kw_only=True)
class TokenizationConfig(ITSerializableCfg):
    tokenizers_parallelism: bool = True
    local_fast_tokenizer_path: Optional[str] = None
    cust_tokenization_pattern: Optional[str] = None
    special_tokens_dict: Dict[str, Any] = field(default_factory=dict)
    max_seq_length: int = 2048


@dataclass(kw_only=True)
class DatasetProcessingConfig(ITSerializableCfg):
    remove_unused_columns: bool = True
    text_fields: Optional[Tuple] = None
    dataset_path: Optional[str] = None
    prepare_validation_set_only: Optional[bool] = False
    enable_datasets_cache: Optional[bool] = False  # disable caching unless explicitly set to improve reproducibility
    data_collator_cfg: Dict[str, Any] = field(default_factory=dict)
    signature_columns: Optional[List] = field(default_factory=list)


@dataclass(kw_only=True)
class ITDataModuleConfig(ITSharedConfig, TokenizationConfig, DatasetProcessingConfig):
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    train_batch_size: int = 32
    eval_batch_size: int = 32
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    # note that for prompt_cfg, we:
    #   1. use (data)classes to minimize special character yaml parsing complications (can override w/ diff init_args)
    #   2. do not provide a default dataclass to avoid current dataclass subclass limitations
    prompt_cfg: PromptConfig = field(default_factory=lambda: PromptConfig())

    def __post_init__(self) -> None:
        # TODO: validate prompt_cfg validity
        self.dataloader_kwargs = {
            "num_workers": self.dataloader_kwargs.get("num_workers", 0),
            "pin_memory": self.dataloader_kwargs.get("pin_memory", False),
        }
        default_dataset_save_path = f"{os.environ['HOME']}/.cache/huggingface/datasets/{self.task_name}"
        self.dataset_path = self.dataset_path or default_dataset_save_path
        if self.defer_model_init:
            assert self.signature_columns is not None, ("`signature_columns` must be specified if `defer_model_init` "
                                                        "is set to True")


@dataclass(kw_only=True)
class ModelConfig(ITSerializableCfg):
    model_class: Optional[torch.nn.Module] = None
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    auto_model_cfg: Dict[str, Any] = field(default_factory=dict)
    from_pretrained_cfg: Dict[str, Any] = field(default_factory=dict)
    dynamic_module_cfg: Dict[str, Any] = field(default_factory=dict)
    use_model_cache: Optional[bool] = False


@dataclass(kw_only=True)
class OptimizerSchedulerConfig(ITSerializableCfg):
    optimizer_init: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler_init: Dict[str, Any] = field(default_factory=dict)
    pl_lrs_cfg: Dict[str, Any] = field(default_factory=dict)
    # Whether to enable gradients for the input embeddings. Useful for finetuning adapter weights w/ a frozen model.
    enable_input_require_grads: Optional[bool] = True


@dataclass(kw_only=True)
class MemoryEfficiencyConfig(ITSerializableCfg):
    torch_dtype: Optional[Union[str, torch.dtype]] = None
    lora_cfg: Dict[str, Any] = field(default_factory=dict)
    bitsandbytesconfig: Dict[str, Any] = field(default_factory=dict)
    activation_checkpointing: Optional[bool] = True


@dataclass(kw_only=True)
class ITConfig(ITSharedConfig, ModelConfig, OptimizerSchedulerConfig, MemoryEfficiencyConfig):
    """Dataclass to encapsulate the ITModuleinternal state."""
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    experiment_tag: Optional[str] = "default"
    log_env_details: Optional[bool] = True
    debug_lm_cfg: DebugLMConfig = field(default_factory=lambda: DebugLMConfig())
    zero_shot_cfg: ITZeroShotClassificationConfig = field(default_factory=lambda: ITZeroShotClassificationConfig())
    # TODO: support only creation of HookedTransformer with pretrained method for now, later support direct creation
    tlens_from_pretrained_cfg: ITLensFromPretrainedConfig = field(default_factory=lambda: ITLensFromPretrainedConfig())

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
        if 'token' in self.from_pretrained_cfg:
            del self.from_pretrained_cfg['token']
        self.torch_dtype = self._torch_dtype_serde()
        if self.torch_dtype and self.bitsandbytesconfig:
            rank_zero_info(f'Ignoring torch_dtype option `{self.torch_dtype}` because quantization config was passed.')
            self.torch_dtype = None
