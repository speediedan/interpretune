from __future__ import annotations
import tempfile
import warnings
from datetime import datetime
from typing import Any, Union, TYPE_CHECKING, cast, Optional
from functools import reduce, partial
from pathlib import Path
from copy import deepcopy
import inspect

import torch

from interpretune.base import ITStateMixin, ITDataModule
from interpretune.extensions import MemProfiler
from interpretune.utils import (
    rank_zero_info,
    rank_zero_warn,
    rank_zero_debug,
    MisconfigurationException,
    _resolve_torch_dtype,
    collect_env_info,
    dummy_method_warn_fingerprint,
)
from interpretune.base.metadata import ITClassMetadata
from interpretune.protocol import (
    Optimizable,
    LRSchedulerConfig,
    StrOrPath,
    LRSchedulerProtocolUnion,
)

if TYPE_CHECKING:
    from interpretune.config import ITConfig, ITState


# TODO: add core helper log/log_dict methods for core context usage
for warnf in [
    f".*{dummy_method_warn_fingerprint}.*",
]:
    warnings.filterwarnings("once", warnf)

# simple barebones interface encapsulating the data preparation, data setup, model setup and optimizer/scheduler
# configuration processes. Intended for maximal interactive experimental flexibility without any iterative assumptions.
# Interpretune executes these hooks after your IT data and model modules are instantiated. The intention is to
# conveniently handle common LLM interpretability experimental setup to facilitate iterative exploratory analysis
# with raw PyTorch in a way that is compatible with wide range of frameworks and research packages. This allows the same
# code to be reused for maximally flexible interactive experimentation and interleaved framework-based tuning, testing
# and prediction/interpretation tasks.


def _dummy_notify(method: str, ret_callable: bool, ret_val: Any, *args, **kwargs) -> Any | None:
    rank_zero_warn(
        f"The `{method}` method is not defined for this module. For framework compatibility, this noop "
        "method will be used. This warning will only be issued once by default."
    )
    out = lambda *args, **kwargs: ret_val if ret_callable else ret_val
    return out


class BaseConfigImpl:
    """ " Methods for adapting the configuration and logging of BaseITModule."""

    # if you override these in your module, ensure you cooperatively call super() if you want to retain
    # the relevant BaseITModule hook functionality
    # proper initialization of these variables should be done in the child class
    it_cfg: ITConfig
    memprofiler: MemProfiler
    model: torch.nn.Module
    cuda_allocator_history: bool
    _it_state: ITState

    @staticmethod
    def _make_config_serializable(config_to_clean: Any, target_keys: str | list) -> dict:
        serial_cfg = deepcopy(config_to_clean)
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        for k in target_keys:
            fqn_l = k.split(".")
            try:
                setattr(reduce(getattr, fqn_l[:-1], serial_cfg), fqn_l[-1], repr(reduce(getattr, fqn_l, serial_cfg)))
            except AttributeError as ae:
                rank_zero_info(
                    f"Attempted to clean a key that was not present, continuing without cleaning that key: {ae}"
                )
        return serial_cfg

    def _init_dirs_and_hooks(self) -> None:
        self._create_experiment_dir()
        if self.cuda_allocator_history:
            self.memprofiler.init_cuda_snapshots_dir()
        # TODO: add save_hyperparameters/basic logging func for raw pytorch
        # (override w/ a framework-specific version where appropriate)
        # self.save_hyperparameters(self._it_state._init_hparams)

    def _create_experiment_dir(self) -> None:
        # we only want to create the core experiment-specific dir for frameworks that aren't adding their own
        log_dir = getattr(self._it_state, "_log_dir", None)
        if log_dir is None:
            return
        # coerce to pathlib.Path so we can safely call mkdir; original value may be str/PathLike
        path_dir = Path(log_dir) / self._it_state._init_hparams["experiment_id"]
        path_dir.mkdir(exist_ok=True, parents=True)
        self._it_state._log_dir = path_dir

    def _capture_hyperparameters(self) -> None:
        # subclasses may have provided their own hparams so we update rather than override
        model_config = {}
        if self.it_cfg.hf_from_pretrained_cfg:
            model_config = (
                self._make_config_serializable(
                    self.model.config,
                    ["quantization_config.bnb_4bit_compute_dtype", "torch_dtype", "_pre_quantization_dtype"],
                ),
            )
        # if `model.config `exists, any provided `model_cfg` should already be merged with it
        elif getattr(self.model, "config", None) is None:
            model_config = self.it_cfg.model_cfg
        self._it_state._init_hparams.update(
            {
                "optimizer_init": self.it_cfg.optimizer_init,
                "lr_scheduler_init": self.it_cfg.lr_scheduler_init,
                "pl_lrs_cfg": self.it_cfg.pl_lrs_cfg,
                "hf_from_pretrained_cfg": self.it_cfg.hf_from_pretrained_cfg,
                "model_config": model_config,
                "model_name_or_path": self.it_cfg.model_name_or_path,
                "task_name": self.it_cfg.task_name,
                "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.it_cfg.experiment_tag}",
            }
        )
        self._it_state._init_hparams["env_info"] = collect_env_info() if self.it_cfg.log_env_details else None


class PropertyDispatcher:
    _it_state: ITState
    # These attributes are provided by the composed BaseITModule at runtime.
    it_cfg: "ITConfig"
    model: torch.nn.Module

    # Consolidated class-level metadata to reduce attribute clutter
    _it_cls_metadata = ITClassMetadata(
        core_to_framework_attrs_map={},
        property_composition={},
    )

    # Below is an experimental feature that enables us to conditionally defer property definitions to other
    # adapter definitions. The intention is to conditionally enhance functionality (e.g., add a setter method
    # where one doesn't exist in a given adapter implementation) while still maximizing compatibility in
    # deferring to the adapter property implementation in contexts where it would be supported. This
    # functionality can be disabled on a property basis by setting `enabled=False` at the cost of potentially reduced
    # compatibility because IT will not dispatch to the adapter's implementation of the IT-enhanced property.

    """Property dispatcher."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cached_mro = inspect.getmro(type(self))
        prop_comp = type(self)._it_cls_metadata.property_composition
        self._enabled_overrides = [
            p for p, cfg in prop_comp.items() if cfg["enabled"] and cfg["target"] in self._cached_mro
        ]

    def _maybe_dispatch(self, non_dispatch_val: Any | None = None) -> Any | None:
        """_summary_

        Args:
            non_dispatch_val (Optional[Any], optional): The value to return if we are not dispatching. Defaults to None.

        Returns:
            Optional[Any]: _description_
        """
        current_frame = inspect.currentframe()
        if current_frame is None or current_frame.f_back is None:
            return non_dispatch_val
        f_back = current_frame.f_back
        if f_back.f_code is None:
            return non_dispatch_val
        overridden_method = f_back.f_code.co_name
        if overridden_method in self._enabled_overrides:
            return (
                type(self)._it_cls_metadata.property_composition[overridden_method]["dispatch"].__get__(self._it_state)
            )
        else:
            return non_dispatch_val

    def _core_or_framework(self, c2f_map_key: str):
        c2f_map = type(self)._it_cls_metadata.core_to_framework_attrs_map
        if not (c2f := c2f_map.get(c2f_map_key, None)):  # short-circuit w/o mapping dispatch
            return getattr(self._it_state, c2f_map_key, None)
        try:
            attr_val = getattr(self._it_state, c2f_map_key, None) or reduce(getattr, c2f[0].split("."), self)
        except (AttributeError, RuntimeError) as resolution_error:
            rank_zero_debug(f"{c2f[2]}: {resolution_error}")
            attr_val = c2f[1]
        return attr_val

    @property
    def core_log_dir(self) -> Optional[StrOrPath]:
        result = self._core_or_framework(c2f_map_key="_log_dir")
        return cast(Optional[StrOrPath], result)

    @property
    def datamodule(self) -> ITDataModule | None:
        result = self._core_or_framework(c2f_map_key="_datamodule")
        return cast(Union[ITDataModule, None], result)

    @property
    def session_complete(self) -> bool:
        return self._it_state._session_complete

    @property
    def cuda_allocator_history(self) -> bool:
        return self.it_cfg.memprofiler_cfg.enabled and self.it_cfg.memprofiler_cfg.cuda_allocator_history

    @property
    def torch_dtype(self) -> Union[torch.dtype, "str"] | None:
        try:
            if dtype := getattr(self.it_cfg, "_torch_dtype", None):
                return dtype
            if getattr(self, "model", None):
                dtype = getattr(self.model, "_torch_dtype", None) or getattr(self.model, "dtype", None)
        except AttributeError:
            dtype = None
        return dtype

    @property
    def device(self) -> torch.device | None:
        try:
            if self._it_state._device is not None:
                # dispatch a framework's property implementation if appropriate, otherwise use `self._it_state._device`
                device = self._maybe_dispatch(self._it_state._device)
            else:
                # check for `model.device`
                device = reduce(getattr, "model.device".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a device reference (has it been set yet?): {ae}")
            device = None
        return cast(Union[torch.device, None], device)

    @device.setter
    def device(self, value: str | torch.device | None) -> None:
        if value is not None and not isinstance(value, torch.device):
            value = torch.device(value)
        self._it_state._device = value

    @torch_dtype.setter
    def torch_dtype(self, value: Union[torch.dtype, "str"] | None) -> None:
        if value is not None and not isinstance(value, torch.dtype):
            value = _resolve_torch_dtype(value)
        self.it_cfg._torch_dtype = value

    def _hook_output_handler(self, hook_name: str, output: Any) -> None:
        if hook_name == "configure_optimizers":
            self._it_init_optimizers_and_schedulers(output)  # type: ignore[attr-defined]  # provided by mixing class
        else:
            rank_zero_warn(f"Output received for hook `{hook_name}` which is not yet supported.")


class CoreHelperAttributes:
    """Mixin class for adding arbitrary core helper attributes to core (non-framework adapted) IT classes."""

    def __init__(self, *args, **kwargs) -> None:
        # we need to initialize internal state before `ITStateMixin`'s __init__ is invoked so use this static method
        ITStateMixin._init_internal_state(self)
        # for core/non-framework modules, we configure a _log_dir rather than relying on the trainer to do so
        # if using a framework (e.g. Lightning) module, we can access the trainer log_dir via the `core_log_dir`
        # property or continue to rely on the trainer log_dir directly
        if it_cfg := kwargs.get("it_cfg", None):
            self._it_state._log_dir = Path(it_cfg.core_log_dir or tempfile.gettempdir())
            ca = it_cfg.compatibility_attrs
        else:
            raise MisconfigurationException("CoreHelperAttributes requires an ITConfig.")
        self._supported_helper_attrs = {
            k: partial(_dummy_notify, method=k, ret_callable=v.ret_callable, ret_val=v.ret_val) for k, v in ca.items()
        }
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # NOTE: dynamically stub a specified subset of framework module attributes (if they don't already exist) to
        #       extend the cases where framework methods can be iteratively used for core context experimentation
        if "_supported_helper_attrs" in self.__dict__:
            _helper_attrs = self.__dict__["_supported_helper_attrs"]
            if name in _helper_attrs:
                return _helper_attrs[name]()
        # the unresolved attribute wasn't ours, pass it to the next __getattr__ in __mro__
        return super().__getattr__(name)  # type: ignore[attr-defined]

    @property
    def current_epoch(self) -> int:
        return self._core_or_framework(c2f_map_key="_current_epoch")

    @property
    def optimizers(self) -> Optional[list[Optimizable]]:
        return self._core_or_framework(c2f_map_key="_it_optimizers")

    @property
    def lr_scheduler_configs(self) -> list[LRSchedulerConfig]:
        return self._core_or_framework(c2f_map_key="_it_lr_scheduler_configs")

    @property
    def lr_schedulers(self) -> None | LRSchedulerProtocolUnion | list[LRSchedulerProtocolUnion]:
        """Returns the learning rate scheduler(s) that are being used during training.

        Returns:
            A single scheduler (either an ``LRScheduler`` or ``ReduceLROnPlateau``),
            a list of such schedulers, or ``None`` if no scheduler configs are available.
        """
        if not self.lr_scheduler_configs:
            return None

        scheds: list[LRSchedulerProtocolUnion] = [config.scheduler for config in self.lr_scheduler_configs]
        if len(scheds) == 1:
            return scheds[0]
        return scheds

    # N.B. Some frameworks (e.g. Lightning) do not support directly setting `current_epoch` and `global_step`, instead
    # using `self.fit_loop.epoch_progress.current.completed` and `self.fit_loop.epoch_loop.global_step` respectively.
    # Instead of mocking analogous loop attributes, when using this mixin, we allow setting the values directly for
    # convenience.

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        self._it_state._current_epoch = value

    @property
    def global_step(self) -> int:
        return self._core_or_framework(c2f_map_key="_global_step")

    @global_step.setter
    def global_step(self, value: int) -> None:
        self._it_state._global_step = value


# adapted from pytorch/core/optimizer.py initialization methods
class OptimizerScheduler:
    """ " Barebones interface to setup optimizers and schedulers for manual optimization with core IT modules."""

    # proper initialization of these variables should be done in the child class
    _it_state: ITState

    def _it_init_optimizers_and_schedulers(self, optim_conf: dict[str, Any] | list | Optimizable | tuple) -> None:
        if optim_conf is None:
            rank_zero_info(  # TODO: maybe set a debug level instead?
                "`configure_optimizers` returned `None`, Interpretune will not configure an optimizer or scheduler.",
            )
            optims, lrs_configs = [], []
        else:
            optims, lrs = OptimizerScheduler._configure_optimizers(optim_conf)
            lrs_configs = OptimizerScheduler._configure_schedulers_manual_opt(lrs)
        self._it_state._it_optimizers, self._it_state._it_lr_scheduler_configs = optims, lrs_configs

    @staticmethod
    def _configure_schedulers_manual_opt(schedulers: list) -> list[LRSchedulerConfig]:
        """Convert each scheduler into `LRSchedulerConfig` structure with relevant information, when using manual
        optimization."""
        lr_scheduler_configs = []
        for scheduler in schedulers:
            if isinstance(scheduler, dict):
                # We do not filter out keys that are invalid with manual optimization to maximize flexibility (since
                # this is a core IT module context)
                # invalid_keys = {"reduce_on_plateau", "monitor", "strict", "interval"}

                config = LRSchedulerConfig(**{key: scheduler[key] for key in scheduler})
            else:
                config = LRSchedulerConfig(scheduler)
            lr_scheduler_configs.append(config)
        return lr_scheduler_configs

    @staticmethod
    def _configure_optimizers(optim_conf: dict[str, Any] | list | Optimizable | tuple) -> tuple[list, list]:
        # for basic optimizer/scheduler init, the proposed IT protocol uses the convenient Lightning optimizer/scheduler
        # formats. Note we do not subject configuration to Lightning validation constraints to increase flexibility.
        optimizers, lr_schedulers = [], []

        # single output, single optimizer
        if isinstance(optim_conf, Optimizable):  # TODO: switch to ParamGroupAddable protocol for FTS instead?
            optimizers = [optim_conf]
        # two lists, optimizer + lr schedulers
        elif (
            isinstance(optim_conf, (list, tuple))
            and len(optim_conf) == 2
            and isinstance(optim_conf[0], list)
            and all(isinstance(opt, Optimizable) for opt in optim_conf[0])
        ):
            opt, sch = optim_conf
            optimizers = opt
            lr_schedulers = sch if isinstance(sch, list) else [sch]
        # single dictionary
        elif isinstance(optim_conf, dict):
            optimizers = [optim_conf["optimizer"]]
            monitor = optim_conf.get("monitor", None)
            if monitor:
                rank_zero_warn(
                    "Interpretune does not support `monitor` in `configure_optimizers` with the core IT module context."
                )
            lr_schedulers = [optim_conf["lr_scheduler"]] if "lr_scheduler" in optim_conf else []
        # multiple dictionaries
        elif isinstance(optim_conf, (list, tuple)) and all(isinstance(d, dict) for d in optim_conf):
            optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
            scheduler_dict = (
                lambda scheduler: dict(scheduler) if isinstance(scheduler, dict) else {"scheduler": scheduler}
            )
            lr_schedulers = [
                scheduler_dict(opt_dict["lr_scheduler"]) for opt_dict in optim_conf if "lr_scheduler" in opt_dict
            ]
        # single list or tuple, multiple optimizer
        elif isinstance(optim_conf, (list, tuple)) and all(isinstance(opt, Optimizable) for opt in optim_conf):
            optimizers = list(optim_conf)
        # unknown configuration
        else:
            raise MisconfigurationException(
                "Unknown configuration for model optimizers."
                " Output from `model.configure_optimizers()` should be one of:\n"
                " * `Optimizer`\n"
                " * [`Optimizer`]\n"
                " * ([`Optimizer`], [`LRScheduler`])\n"
                ' * {"optimizer": `Optimizer`, (optional) "lr_scheduler": `LRScheduler`}\n'
            )
        return optimizers, lr_schedulers


class BaseITComponents(BaseConfigImpl, PropertyDispatcher, OptimizerScheduler):  # type: ignore[misc]
    ...
