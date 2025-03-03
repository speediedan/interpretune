from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
import os
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field

import torch

from interpretune.config import (ITSerializableCfg, ITSharedConfig, AutoCompConf, ExtensionConf, HFFromPretrainedConfig,
                                 GenerativeClassificationConfig)
from interpretune.utils import rank_zero_info
from interpretune.protocol import LRSchedulerConfig, Optimizable

if TYPE_CHECKING:
    from interpretune.protocol import AnalysisCfgProtocol
    from interpretune.base import ITDataModule


################################################################################
# ITModule Configuration Dataclasses
# Each core ITModule component has its own configuration dataclass. The core
# component configurations are then composed to form the ITModule configuration.
# This approach offers immense configuration flexibility while reducing the
# configuration noise that would be incurred by organizing the configuration
# space using a single flat configuration dataclass.
################################################################################

@dataclass(kw_only=True)
class ModelConf(ITSerializableCfg):
    model_class: torch.nn.Module | None = None
    model_cfg: dict[str, Any] = field(default_factory=dict)
    cust_fwd_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class OptimizerSchedulerConf(ITSerializableCfg):
    optimizer_init: dict[str, Any] = field(default_factory=dict)
    lr_scheduler_init: dict[str, Any] = field(default_factory=dict)
    pl_lrs_cfg: dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class MixinsConf(ITSerializableCfg):
    analysis_cfg: AnalysisCfgProtocol | None = None
    generative_step_cfg: GenerativeClassificationConfig = field(default_factory=GenerativeClassificationConfig)
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = None

@dataclass(kw_only=True)
class LoggingConf(ITSerializableCfg):
    experiment_tag: str | None = "default"
    log_env_details: bool | None = True
    core_log_dir: str | os.PathLike | None = None

@dataclass(kw_only=True)
class AutoCompatConfig(ITSerializableCfg):
    ret_callable: bool | None = False
    ret_val: Any | None = None

@dataclass(kw_only=True)
class CompatConf(ITSerializableCfg):
    compatibility_attrs: dict[str, AutoCompatConfig] = field(default_factory=lambda: {'log': AutoCompatConfig(),
                                                                                      'log_dict': AutoCompatConfig(),})

@dataclass(kw_only=True)
class ITConfig(ITSharedConfig, ModelConf, OptimizerSchedulerConf, MixinsConf, LoggingConf, CompatConf, AutoCompConf,
               ExtensionConf):
    #"""Dataclass to encapsulate the ITModuleinternal state."""
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    # dynamic fields added via ExtensionConf contingent on supported extension availability
    # debug_lm_cfg: DebugLMConfig = field(default_factory=DebugLMConfig)
    # memprofiler_cfg: MemProfilerCfg = field(default_factory=MemProfilerCfg)

    def __post_init__(self) -> None:
        # _torch_dtype may have been resolved and set in a subclass already
        if self.hf_from_pretrained_cfg:
            if not hasattr(self, '_torch_dtype'):
                self._torch_dtype = self.hf_from_pretrained_cfg._torch_dtype_serde()
            if self._torch_dtype and self.hf_from_pretrained_cfg.bitsandbytesconfig:
                rank_zero_info(f'Specified torch_dtype `{self._torch_dtype}` being overridden by quantization config.')
                self._torch_dtype = None

@dataclass
class ITState:
    """Dataclass to encapsulate the ITModule internal state and keep top-level namespace as clean as possible."""
    _it_lr_scheduler_configs: list[LRSchedulerConfig] = None
    _it_optimizers: list[Optimizable] = None  # initialized via core IT module `configure_optimizers` hook
    _datamodule: ITDataModule | None = None  # datamodule handle attached after init
    _device: torch.device | None = None  # root device (sometimes used if not handled by Lightning)
    _extensions: dict[str, Any] = field(default_factory=dict)
    _session_complete: bool = False
    _init_hparams: dict[str, Any] = field(default_factory=dict)
    # TODO: should we leave initialization of the below to the relevant property dispatch functions?
    _current_epoch: int = 0
    _global_step: int = 0
