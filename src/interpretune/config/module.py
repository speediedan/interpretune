# N.B. we need to avoid annotations import here due to jsonargparse validation issues that emerge when it is used
# from __future__ import annotations
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING, Tuple, Type
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
from interpretune.config.profiling import ProfilerCfg
from interpretune.utils import rank_zero_info, MisconfigurationException
from interpretune.utils.repr_helpers import (
    summarize_obj,
    state_to_dict,
    state_to_summary,
    state_repr,
)
from interpretune.protocol import LRSchedulerConfig, Optimizable, StrOrPath
from interpretune.tuning.interventions import InterventionCfg, as_intervention_cfg

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
    """Model identity and loading options."""

    model_class: Type[torch.nn.Module] | None = None
    model_cfg: dict[str, Any] = field(default_factory=dict)
    cust_fwd_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class OptimizerSchedulerConf(ITSerializableCfg):
    """Optimizer and LR-scheduler construction, given as class paths plus init args."""

    optimizer_init: dict[str, Any] = field(default_factory=dict)
    lr_scheduler_init: dict[str, Any] = field(default_factory=dict)
    pl_lrs_cfg: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ClassificationConf(ITSerializableCfg):
    """Classification setup: the label mapping and how predictions are derived."""

    classification_mapping: Tuple | None = None
    classification_mapping_indices: torch.Tensor | None = None


@dataclass(kw_only=True)
class MixinsConf(ITSerializableCfg):
    """Which optional mixins the composed module should include."""

    analysis_cfg: "AnalysisCfgProtocol | None" = None
    generative_step_cfg: GenerativeClassificationConfig = field(default_factory=GenerativeClassificationConfig)
    hf_from_pretrained_cfg: HFFromPretrainedConfig | None = None


@dataclass(kw_only=True)
class LoggingConf(ITSerializableCfg):
    """Logging destinations and verbosity for a run."""

    experiment_tag: str | None = "default"
    log_env_details: bool | None = True
    core_log_dir: StrOrPath | None = None
    logging_level: str | int = "INFO"
    log_to_stream: bool = True
    log_to_file: bool = True


@dataclass(kw_only=True)
class AutoCompatConfig(ITSerializableCfg):
    """Declares the auto-composition target so a config subclass is synthesized to match its fields."""

    ret_callable: bool | None = False
    ret_val: Any | None = None


@dataclass(kw_only=True)
class CompatConf(ITSerializableCfg):
    """Compatibility toggles for behaviors that differ across supported backend versions."""

    compatibility_attrs: dict[str, AutoCompatConfig] = field(default_factory=dict)


@dataclass(kw_only=True)
class TuningConf(ITSerializableCfg):
    """Tunable modules the module attaches once its model is built (#812)."""

    # interventions spliced in at a site and declared `intervention:<kind>` (`interpretune.tuning.interventions`)
    tuning_interventions: list[InterventionCfg] = field(default_factory=list)


@dataclass(kw_only=True)
class ITConfig(
    ITSharedConfig,
    ModelConf,
    OptimizerSchedulerConf,
    ClassificationConf,
    MixinsConf,
    LoggingConf,
    CompatConf,
    TuningConf,
    AutoCompConf,
    ExtensionConf,
):
    """The module-side configuration: model, optimization, classification, logging and mixins.

    Composed from the grouped configs above rather than declaring every field flat, so an adapter can
    extend one group without widening the whole surface.
    """

    # PyTorch profiler selection for the core runners (interpretune#11); default off. MemProfiler keeps its own
    # extension field, and _check_profiler_selection refuses enabling both.
    profiler_cfg: ProfilerCfg = field(default_factory=ProfilerCfg)

    # The finetuning-scheduler view to attach in place of the composition's own (interpretune#841): a `class_path`
    # and optional `init_args` mapping, instantiated with the module as the view's root. For example
    # `finetuning_scheduler.tunable_view.CanonicalView` schedules a composition by its own (physical) parameter names,
    # and `interpretune.adapters.transformer_lens.tunable_view.TransformerBridgeView` with `implicit_norm_thaw: false`
    # thaws only the norms a schedule names. Unset, the composition's `build_tunable_view()` decides.
    tunable_view: dict[str, Any] | None = None

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
        self._check_profiler_selection()
        self._check_tunable_view()
        self._coerce_tuning_interventions()

    def _coerce_tuning_interventions(self) -> None:
        """Build each ``tuning_interventions`` mapping into an :class:`InterventionCfg` (see
        :func:`~interpretune.tuning.interventions.as_intervention_cfg`)."""
        try:
            self.tuning_interventions = [as_intervention_cfg(e) for e in self.tuning_interventions or ()]
        except TypeError as err:
            raise MisconfigurationException(f"it_cfg.tuning_interventions: {err}") from err

    def _check_tunable_view(self) -> None:
        """Refuse a ``tunable_view`` that is not a ``class_path`` (and optional ``init_args``) mapping, or that
        sets the view's root, which is always the module."""
        if self.tunable_view is None:
            return
        cfg = self.tunable_view
        if not isinstance(cfg, Mapping) or not isinstance(cfg.get("class_path"), str):
            raise MisconfigurationException(
                f"it_cfg.tunable_view must be a mapping with a `class_path` (and optional `init_args`), got {cfg!r}."
            )
        if extra := set(cfg) - {"class_path", "init_args"}:
            raise MisconfigurationException(
                f"it_cfg.tunable_view takes `class_path` and `init_args`, not {sorted(extra)}."
            )
        init_args = cfg.get("init_args") or {}
        if not isinstance(init_args, Mapping):
            raise MisconfigurationException(f"it_cfg.tunable_view's `init_args` must be a mapping, got {init_args!r}.")
        if "root" in init_args:
            raise MisconfigurationException(
                "it_cfg.tunable_view's `init_args` cannot set `root`: the view's root is always the module."
            )

    def _check_profiler_selection(self) -> None:
        """Refuse two profilers observing one module: the PyTorch profiler and an enabled MemProfiler.

        Both instrument the same forward and backward passes, so running them together perturbs each other's
        measurements; exactly one may be active.
        """
        # memprofiler_cfg is attached by ExtensionConf only when the bundled MemProfiler extension is available.
        mem_cfg = getattr(self, "memprofiler_cfg", None)
        if self.profiler_cfg.which == "pytorch" and mem_cfg is not None and getattr(mem_cfg, "enabled", False):
            raise MisconfigurationException(
                "profiler_cfg selects the PyTorch profiler while it_cfg.memprofiler_cfg is enabled: two profilers "
                "would observe the same module. Enable exactly one."
            )


@dataclass
class ITState:
    """Dataclass to encapsulate the ITModule internal state and keep top-level namespace as clean as possible."""

    _it_lr_scheduler_configs: list[LRSchedulerConfig] = field(default_factory=list)
    _it_optimizers: list[Optimizable] = field(default_factory=list)  # init'd via `configure_optimizers`
    _log_dir: StrOrPath | None = None
    _datamodule: "ITDataModule | None" = None  # datamodule handle attached after init
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
