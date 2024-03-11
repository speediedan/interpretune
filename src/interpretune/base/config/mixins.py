from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import torch

from interpretune.base.config.shared import ITSerializableCfg
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.import_utils import _resolve_torch_dtype



@dataclass(kw_only=True)
class BaseGenerationConfig(ITSerializableCfg):
    max_new_tokens: int = 5  # nb maxing logits over multiple tokens (n<=5) will yield a very slight perf gain versus 1
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0


@dataclass(kw_only=True)
class HFGenerationConfig(BaseGenerationConfig):
    use_cache: bool = True
    repetition_penalty: float = 1.0
    output_attentions: bool = False
    output_hidden_states: bool = False
    length_penalty: float = 1.0
    output_scores: bool = True
    return_dict_in_generate: bool = True


@dataclass(kw_only=True)
class ZeroShotClassificationConfig(ITSerializableCfg):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=lambda: HFGenerationConfig())


@dataclass(kw_only=True)
class HFFromPretrainedConfig(ITSerializableCfg):
    """
    HFFromPretrainedConfig: Configuration for loading a pretrained model from Huggingface along with configuration
    options contingent on the HF pretrained model type.
    """
    pretrained_kwargs: Dict[str, Any] = field(default_factory=dict)
    dynamic_module_cfg: Dict[str, Any] = field(default_factory=dict)
    use_model_cache: Optional[bool] = False
    model_head: str = ''
    lora_cfg: Dict[str, Any] = field(default_factory=dict)
    bitsandbytesconfig: Dict[str, Any] = field(default_factory=dict)
    activation_checkpointing: bool = False
    # Whether to enable gradients for the input embeddings. Useful for finetuning adapter weights w/ a frozen model.
    enable_input_require_grads: bool = True
    default_head: str = "transformers.AutoModelForSequenceClassification"

    def __post_init__(self):
        if self.pretrained_kwargs.get('token', None):
            del self.pretrained_kwargs['token']

    def _torch_dtype_serde(self) -> Optional[torch.dtype]:
        if self.pretrained_kwargs and self.pretrained_kwargs.get('torch_dtype', None):
            if resolved_dtype := _resolve_torch_dtype(self.pretrained_kwargs['torch_dtype']):
                del self.pretrained_kwargs['torch_dtype']
                return resolved_dtype
            else:
                rank_zero_warn(f"The provided `torch_dtype` {self.pretrained_kwargs.pop('torch_dtype')} could not"
                               " be resolved, attempting to proceed with `torch_dtype` unset.")
