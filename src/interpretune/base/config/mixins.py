from typing import Any, Dict, Optional, NamedTuple
from dataclasses import dataclass, field

import torch
from transformers.generation.configuration_utils import GenerationConfig

from interpretune.base.config.shared import ITSerializableCfg
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.import_utils import _resolve_torch_dtype


class ITExtension(NamedTuple):
    ext_attr: str
    ext_fqn: str


@dataclass(kw_only=True)
class BaseGenerationConfig(ITSerializableCfg):
    ...  # core generation config that may be extended and include common `ITConfig` logic in the future


@dataclass(kw_only=True)
class CoreGenerationConfig(BaseGenerationConfig):
    max_new_tokens: int = 5  # nb maxing logits over multiple tokens (n<=5) will yield a very slight perf gain versus 1
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0

@dataclass(kw_only=True)
class HFGenerationConfig(BaseGenerationConfig):
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        default_overrides = {"return_dict_in_generate": True, "output_logits": True}
        hf_overrides = self.kwargs.pop('cust_hf_overrides', default_overrides)
        valid_hf_keys = [k for k in GenerationConfig().__dict__.keys()  if not k.startswith("_")]
        # we defer to HF's default generation config for all supported `GenerationConfig` settings except for attributes
        # specified in the default or provided (`cust_hf_overrides`) override config
        for k, v in self.kwargs.items():
            if k in valid_hf_keys:
                setattr(self, k, v)
        for k, v in hf_overrides.items():
            if k not in self.kwargs.keys() and k in valid_hf_keys:
                setattr(self, k, v)
        self.kwargs = {}

@dataclass(kw_only=True)
class ZeroShotClassificationConfig(ITSerializableCfg):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=lambda: HFGenerationConfig())
    # for generate methods that don't also perform data preparation, filter out inputs that the model's generate
    # function does not support
    input_inspection_enabled: bool = True

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
    default_head: str = "transformers.AutoModelForCausalLM"

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
