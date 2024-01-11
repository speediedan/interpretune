import os
import warnings
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass, field

import torch

from interpretune.config_classes.shared import ITSerializableCfg, ITSharedConfig
from interpretune.analysis.debug_generation import DebugLMConfig
from interpretune.analysis.memprofiler import MemProfilerCfg
from interpretune.utils.logging import rank_zero_info, rank_zero_warn


# TODO: add core helper log/log_dict methods for core context usage
for warnf in [".*For Lightning compatibility, this noop .*",]:
    warnings.filterwarnings("once", warnf)


################################################################################
# ITModule Configuration Dataclasses
# Each core ITModule component has its own configuration dataclass. The core
# component configurations are then composed to form the ITModule configuration.
# This approach offers immense configuration flexibility while reducing the
# configuration noise that would be incurred by organizing the configuration
# space using a single flat configuration dataclass.
################################################################################

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
class ModelConfig(ITSerializableCfg):
    model_class: Optional[torch.nn.Module] = None
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    auto_model_cfg: Dict[str, Any] = field(default_factory=dict)
    cust_fwd_kwargs: Dict[str, Any] = field(default_factory=dict)
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
    lora_cfg: Dict[str, Any] = field(default_factory=dict)
    bitsandbytesconfig: Dict[str, Any] = field(default_factory=dict)
    activation_checkpointing: Optional[bool] = False

@dataclass(kw_only=True)
class AutoCompatConfig(ITSerializableCfg):
    ret_callable: Optional[bool] = False
    ret_val: Optional[Any] = None

@dataclass(kw_only=True)
class ITConfig(ITSharedConfig, ModelConfig, OptimizerSchedulerConfig, MemoryEfficiencyConfig):
    """Dataclass to encapsulate the ITModuleinternal state."""
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    experiment_tag: Optional[str] = "default"
    log_env_details: Optional[bool] = True
    lightning_compat_attrs: Dict[str, AutoCompatConfig] = \
        field(default_factory=lambda: {'log': AutoCompatConfig(), 'log_dict': AutoCompatConfig(),})
    core_log_dir: Optional[str | os.PathLike] = None
    memprofiler_cfg: MemProfilerCfg = field(default_factory=lambda: MemProfilerCfg())
    debug_lm_cfg: DebugLMConfig = field(default_factory=lambda: DebugLMConfig())
    zero_shot_cfg: ITZeroShotClassificationConfig = field(default_factory=lambda: ITZeroShotClassificationConfig())
    # TODO: support only creation of HookedTransformer with pretrained method for now, later support direct creation
    tlens_from_pretrained_cfg: ITLensFromPretrainedConfig = field(default_factory=lambda: ITLensFromPretrainedConfig())

    def _pop_dtype_msg(self) -> None:
        rank_zero_warn(f"The provided `torch_dtype` {self.from_pretrained_cfg.pop('torch_dtype')} could not be "
                       "resolved, attempting to proceed with `torch_dtype` unset.")

    def _torch_dtype_serde(self) -> Optional[torch.dtype]:
        if self.from_pretrained_cfg.get('torch_dtype', None):
            if isinstance(self.from_pretrained_cfg['torch_dtype'], str):
                if hasattr(torch, self.from_pretrained_cfg['torch_dtype']):
                    return getattr(torch, self.from_pretrained_cfg.pop('torch_dtype'))
                elif hasattr(torch, self.from_pretrained_cfg['torch_dtype'].split(".")[-1]):
                    return getattr(torch, self.from_pretrained_cfg.pop('torch_dtype').split(".")[-1])
                else:
                    self._pop_dtype_msg()
            elif isinstance(self.from_pretrained_cfg['torch_dtype'], torch.dtype):
                return self.from_pretrained_cfg.pop('torch_dtype')
            else:
                self._pop_dtype_msg()

    def __post_init__(self) -> None:
        if 'token' in self.from_pretrained_cfg:
            del self.from_pretrained_cfg['token']
        self._torch_dtype = self._torch_dtype_serde()
        if self._torch_dtype and self.bitsandbytesconfig:
            rank_zero_info(f'Ignoring torch_dtype option `{self._torch_dtype}` because quantization config was passed.')
            self._torch_dtype = 'see quantization config'
