import os
#import warnings
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

import torch

from interpretune.base.config.shared import ITSerializableCfg, ITSharedConfig
from interpretune.base.config.mixins import ZeroShotClassificationConfig, HFFromPretrainedConfig
from interpretune.analysis.debug_generation import DebugLMConfig
from interpretune.analysis.memprofiler import MemProfilerCfg
from interpretune.utils.logging import rank_zero_info
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.types import LRSchedulerConfig, Optimizable
# from interpretune.utils.warnings import dummy_method_warn_fingerprint

# # TODO: add core helper log/log_dict methods for core context usage
# for warnf in [f".*{dummy_method_warn_fingerprint}.*",]:
#     warnings.filterwarnings("once", warnf)


################################################################################
# ITModule Configuration Dataclasses
# Each core ITModule component has its own configuration dataclass. The core
# component configurations are then composed to form the ITModule configuration.
# This approach offers immense configuration flexibility while reducing the
# configuration noise that would be incurred by organizing the configuration
# space using a single flat configuration dataclass.
################################################################################


@dataclass(kw_only=True)
class ModelConfig(ITSerializableCfg):
    model_class: Optional[torch.nn.Module] = None
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    cust_fwd_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class OptimizerSchedulerConfig(ITSerializableCfg):
    optimizer_init: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler_init: Dict[str, Any] = field(default_factory=dict)
    pl_lrs_cfg: Dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class AutoCompatConfig(ITSerializableCfg):
    ret_callable: Optional[bool] = False
    ret_val: Optional[Any] = None

@dataclass(kw_only=True)
class ITConfig(ITSharedConfig, ModelConfig, OptimizerSchedulerConfig):
    """Dataclass to encapsulate the ITModuleinternal state."""
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    experiment_tag: Optional[str] = "default"
    log_env_details: Optional[bool] = True
    core_log_dir: Optional[str | os.PathLike] = None
    memprofiler_cfg: MemProfilerCfg = field(default_factory=lambda: MemProfilerCfg())
    debug_lm_cfg: DebugLMConfig = field(default_factory=lambda: DebugLMConfig())
    zero_shot_cfg: ZeroShotClassificationConfig = field(default_factory=lambda: ZeroShotClassificationConfig())
    hf_from_pretrained_cfg: Optional[HFFromPretrainedConfig] = None
    compatibility_attrs: Dict[str, AutoCompatConfig] = \
        field(default_factory=lambda: {'log': AutoCompatConfig(), 'log_dict': AutoCompatConfig(),})

    def __post_init__(self) -> None:
        # _torch_dtype may have been resolved and set in a subclass already
        if self.hf_from_pretrained_cfg:
            if not hasattr(self, '_torch_dtype'):
                self._torch_dtype = self.hf_from_pretrained_cfg._torch_dtype_serde()
            if self._torch_dtype and self.hf_from_pretrained_cfg.bitsandbytesconfig:
                rank_zero_info(f'Specified torch_dtype `{self._torch_dtype}` being overridden by quantization config.')
                self._torch_dtype = 'see quantization config'

@dataclass
class ITState:
    """Dataclass to encapsulate the ITModule internal state and keep top-level namespace as clean as possible."""
    _it_lr_scheduler_configs: List[LRSchedulerConfig] = None
    _it_optimizers: List[Optimizable] = None  # initialized via core IT module `configure_optimizers` hook
    _datamodule: Optional[ITDataModule] = None  # datamodule handle attached after init
    _device: Optional[torch.device] = None  # root device (sometimes used if not handled by Lightning)
    _extensions: Dict[str, Any] = field(default_factory=dict)
    _session_complete: bool = False
    _init_hparams: Dict[str, Any] = field(default_factory=dict)
    # TODO: should we leave initialization of the below to the relevant property dispatch functions?
    _current_epoch: int = 0
    _global_step: int = 0
