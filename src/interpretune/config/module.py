from typing import Any, TYPE_CHECKING, Optional, Tuple, Type
from dataclasses import dataclass, field

import torch

from interpretune.config import (
    ITSerializableCfg,
    ITSharedConfig,
    AutoCompConf,
    ExtensionConf,
    HFFromPretrainedConfig,
    GenerativeClassificationConfig,
)
from interpretune.utils import rank_zero_info
from interpretune.utils.repr_helpers import (
    summarize_obj,
    state_to_dict,
    state_to_summary,
    state_repr,
)
from interpretune.protocol import LRSchedulerConfig, Optimizable, StrOrPath

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
    model_class: Type[torch.nn.Module] | None = None
    model_cfg: dict[str, Any] = field(default_factory=dict)
    cust_fwd_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class OptimizerSchedulerConf(ITSerializableCfg):
    optimizer_init: dict[str, Any] = field(default_factory=dict)
    lr_scheduler_init: dict[str, Any] = field(default_factory=dict)
    pl_lrs_cfg: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ClassificationConf(ITSerializableCfg):
    classification_mapping: Optional[Tuple] = None
    classification_mapping_indices: Optional[torch.Tensor] = None


@dataclass(kw_only=True)
class MixinsConf(ITSerializableCfg):
    analysis_cfg: Optional["AnalysisCfgProtocol"] = None
    generative_step_cfg: GenerativeClassificationConfig = field(default_factory=GenerativeClassificationConfig)
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = None


@dataclass(kw_only=True)
class LoggingConf(ITSerializableCfg):
    experiment_tag: str | None = "default"
    log_env_details: bool | None = True
    core_log_dir: StrOrPath | None = None


@dataclass(kw_only=True)
class AutoCompatConfig(ITSerializableCfg):
    ret_callable: bool | None = False
    ret_val: Any | None = None


@dataclass(kw_only=True)
class CompatConf(ITSerializableCfg):
    compatibility_attrs: dict[str, AutoCompatConfig] = field(
        default_factory=lambda: {
            "log": AutoCompatConfig(),
            "log_dict": AutoCompatConfig(),
        }
    )


@dataclass(kw_only=True)
class ITConfig(
    ITSharedConfig,
    ModelConf,
    OptimizerSchedulerConf,
    ClassificationConf,
    MixinsConf,
    LoggingConf,
    CompatConf,
    AutoCompConf,
    ExtensionConf,
):
    # """Dataclass to encapsulate the ITModule internal state."""
    # See NOTE [Interpretune Dataclass-Oriented Configuration]
    # dynamic fields added via ExtensionConf contingent on supported extension availability
    # debug_lm_cfg: DebugLMConfig = field(default_factory=DebugLMConfig)
    # memprofiler_cfg: MemProfilerCfg = field(default_factory=MemProfilerCfg)

    def __post_init__(self) -> None:
        # `_dtype` may have been resolved and set in a subclass already
        if self.hf_from_pretrained_cfg:
            if not hasattr(self, "_dtype"):
                self._dtype = self.hf_from_pretrained_cfg._dtype_serde()
            if self._dtype and self.hf_from_pretrained_cfg.bitsandbytesconfig:
                rank_zero_info(f"Specified dtype `{self._dtype}` being overridden by quantization config.")
                self._dtype = None


@dataclass
class ITState:
    """Dataclass to encapsulate the ITModule internal state and keep top-level namespace as clean as possible."""

    _it_lr_scheduler_configs: list[LRSchedulerConfig] = field(default_factory=list)
    _it_optimizers: list[Optimizable] = field(default_factory=list)  # init'd via `configure_optimizers`
    _log_dir: Optional[StrOrPath] = None
    _datamodule: Optional["ITDataModule"] = None  # datamodule handle attached after init
    _device: torch.device | None = None  # root device (sometimes used if not handled by Lightning)
    _extensions: dict[str, Any] = field(default_factory=dict)
    _session_complete: bool = False
    _init_hparams: dict[str, Any] = field(default_factory=dict)
    # TODO: should we leave initialization of the below to the relevant property dispatch functions?
    _current_epoch: int = 0
    _global_step: int = 0

    # Object summarization mapping used by shared repr helpers.
    # a simple mapping of attribute_name -> label is used
    _obj_summ_map = {
        "_device": "device",
        "_datamodule": "datamodule",
        "_it_optimizers": "optimizers",
        "_it_lr_scheduler_configs": "schedulers",
        "_current_epoch": "epoch",
        "_global_step": "step",
        "_session_complete": "session_complete",
        "_log_dir": "log_dir",
        "_extensions": "extensions",
        "_init_hparams": "init_cfg",
    }

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary dict of the ITState."""
        return state_to_dict(
            self,
            custom_key_transforms={
                "_init_hparams": self._init_hparams_transform,
                "_extensions": self._extensions_transform,
            },
        )

    @staticmethod
    def _init_hparams_transform(v: dict[str, Any]) -> dict:
        """Custom transform for summarizing the `_init_hparams` mapping.

        - empty -> {}
        - nested dict -> "{...}" (non-empty) or {} (empty)
        - string -> short repr
        - otherwise -> summarize_obj
        """
        if not v:
            return {}
        init_summary: dict[str, Any] = {}
        for ik, iv in v.items():
            if isinstance(iv, dict):
                init_summary[ik] = "{...}" if len(iv) > 0 else {}
            else:
                init_summary[ik] = summarize_obj(iv)
        return init_summary

    @staticmethod
    def _extensions_transform(v: Any) -> Any:
        """Transform for summarizing the `_extensions` mapping."""
        try:
            return {ek: summarize_obj(ev) for ek, ev in (v or {}).items()}
        except Exception:
            return summarize_obj(v)

    def __repr__(self) -> str:
        try:
            inner = state_to_summary(self.to_dict(), self)
            return state_repr(inner, self.__class__.__name__)
        except Exception:
            return state_repr("", self.__class__.__name__)
