import tempfile
import os
from datetime import datetime
from typing import Any, Dict, List, Union, Tuple, Optional, NamedTuple
from functools import reduce, partial
from pathlib import Path
from copy import deepcopy

import torch

from interpretune.base.config.module import ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.logging import rank_zero_info, rank_zero_warn
from interpretune.utils.exceptions import MisconfigurationException
from interpretune.analysis.debug_generation import DebugGeneration
from interpretune.analysis.memprofiler import MemProfiler
from interpretune.utils.types import LRSchedulerConfig, Optimizer, Optimizable, LRScheduler
from interpretune.utils.logging import collect_env_info, rank_zero_debug

# simple barebones interface encapsulating the data preparation, data setup, model setup and optimizer/scheduler
# configuration processes. Intended for maximal interactive experimental flexibility without any iterative assumptions.
# Interpretune executes these hooks after your IT data and model modules are instantiated. The intention is to
# convienently handle common LLM interpretability experimental setup to facilitate iterative exploratory analysis
# with raw PyTorch in a way that is compatible with the Lightning framework. This allows the same code to be reused for
# maximally flexible interactive experimentation and interleaved Lightning-based tuning, testing and
# prediction/interpretation tasks.

CORE_TO_LIGHTNING_ATTRS_MAP = {
    "_log_dir": ("trainer.model._trainer.log_dir", None, "No log_dir has been set yet"),
    "_datamodule": ("trainer.datamodule", None, "Could not find datamodule reference (has it been attached yet?)"),
    "_tl_cfg": ("model.cfg", None, ""),
    "_current_epoch": ("trainer.current_epoch", 0, ""),
    "_global_step": ("trainer.global_step", 0, ""),
}

class ITExtension(NamedTuple):
    ext_attr: str
    ext_class: Any

SUPPORTED_EXTENSIONS = (ITExtension("debug_lm", DebugGeneration), ITExtension("memprofiler", MemProfiler))


def _dummy_notify(method: str, ret_callable: bool, rv: Any, *args, **kwargs) -> Optional[Any]:
    rank_zero_warn(f"The `{method}` method is not defined for this module. For Lightning compatibility, this noop "
                    "method will be used. This warning will only be issued once by default.")
    out = lambda *args, **kwargs: rv if ret_callable else rv
    return out


class ConfigAdapter:
    """" Methods for adapting the configuration and logging of BaseITModule."""

    # if you override these in your LightningModule, ensure you cooperatively call super() if you want to retain
    # the relevant BaseITModule hook functionality
    # proper initialization of these variables should be done in the child class
    it_cfg: ITConfig
    memprofiler: MemProfiler
    model: torch.nn.Module
    cuda_allocator_history: bool
    init_hparams: Dict[str, Any]
    # conditionally initialized in core subclasses ITModule and ITLensModule (via CoreHelperAttributes)
    _log_dir: Optional[os.PathLike]

    @staticmethod
    def _make_config_serializable(config_to_clean: Any, target_keys: Union[str, List]) -> Dict:
        serial_cfg = deepcopy(config_to_clean)
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        for k in target_keys:
            fqn_l = k.split(".")
            try:
                setattr(reduce(getattr, fqn_l[:-1], serial_cfg), fqn_l[-1], repr(reduce(getattr, fqn_l, serial_cfg)))
            except AttributeError as ae:
                rank_zero_info("Attempted to clean a key that was not present, continuing without cleaning that key: "
                               f"{ae}")
        return serial_cfg

    def _init_dirs_and_hooks(self) -> None:
        self._create_experiment_dir()
        if self.cuda_allocator_history:
            self.memprofiler.init_cuda_snapshots_dir()
        # TODO: add save_hyperparameters/basic logging func for raw pytorch
        # (override w/ lightning version where appropriate)
        #self.save_hyperparameters(self.init_hparams)

    def _create_experiment_dir(self) -> None:
        # we only want to create the core experiment-specific dir for non-lightning modules
        if getattr(self, '_log_dir', None):
            self._log_dir = self._log_dir / self.init_hparams['experiment_id']
            self._log_dir.mkdir(exist_ok=True, parents=True)

    def _capture_hyperparameters(self) -> None:
        # subclasses may have provided their own hparams so we update rather than override
        model_config = {}
        if self.it_cfg.hf_from_pretrained_cfg:
            model_config = self._make_config_serializable(self.model.config,
                                                          ['quantization_config.bnb_4bit_compute_dtype',
                                                           'torch_dtype', '_pre_quantization_dtype']),
        self.init_hparams.update({
            "optimizer_init": self.it_cfg.optimizer_init,
            "lr_scheduler_init": self.it_cfg.lr_scheduler_init,
            "pl_lrs_cfg": self.it_cfg.pl_lrs_cfg,
            "hf_from_pretrained_cfg": self.it_cfg.hf_from_pretrained_cfg,
            # "dynamic_module_cfg": self.it_cfg.dynamic_module_cfg,
            # "quantization_cfg": self.it_cfg.lora_cfg,
            # "auto_model_cfg": self.it_cfg.auto_model_cfg, # TODO: cleanup/consolidate saving configs/dedup
            "model_config": model_config,
            "model_name_or_path": self.it_cfg.model_name_or_path,
            "task_name": self.it_cfg.task_name,
            "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.it_cfg.experiment_tag}",
            })
        self.init_hparams["env_info"] = collect_env_info() if self.it_cfg.log_env_details else None


class PropertyDispatcher:

    def connect_extensions(self):
        for ext_name, ext_class in SUPPORTED_EXTENSIONS:
            if getattr(self.it_cfg, f'{ext_name}_cfg').enabled:
                setattr(self, ext_name, ext_class())
                getattr(self, ext_name).connect(self)

    def _core_or_lightning(self, c2l_map_key: str):
        c2l = CORE_TO_LIGHTNING_ATTRS_MAP[c2l_map_key]
        try:
            attr_val = getattr(self, c2l_map_key, None) or reduce(getattr, c2l[0].split("."), self)
        except AttributeError as ae:
            rank_zero_debug(f"{c2l[2]}: {ae}")
            attr_val = c2l[1]
        return attr_val

    @property
    def core_log_dir(self) -> Optional[str | os.PathLike]:
        return self._core_or_lightning(c2l_map_key="_log_dir")

    @property
    def datamodule(self) -> Optional[ITDataModule]:
        return self._core_or_lightning(c2l_map_key="_datamodule")

    @property
    def session_complete(self) -> bool:
        return self._session_complete

    @property
    def cuda_allocator_history(self) -> bool:
        return self.it_cfg.memprofiler_cfg.enabled and self.it_cfg.memprofiler_cfg.cuda_allocator_history

    @property
    def torch_dtype(self) -> Optional[Union[torch.dtype, 'str']]:
        try:
            if dtype := getattr(self.it_cfg, "_torch_dtype", None):
                return dtype
            if getattr(self, 'model', None):
                dtype = getattr(self.model, "_torch_dtype", None) or getattr(self.model, "dtype", None)
        except AttributeError:
            dtype = None
        return dtype

    def _hook_output_handler(self, hook_name: str, output: Any) -> None:
        if hook_name == "configure_optimizers":
            self._it_init_optimizers_and_schedulers(output)
        elif hook_name == "on_train_epoch_start":
            pass  # TODO: remove if decided that no need to connect output of this hook
        else:
            rank_zero_warn(f"Output received for hook `{hook_name}` which is not yet supported.")


class CoreHelperAttributes:

    _log_dir: Optional[os.PathLike]

    """Mixin class for adding arbitrary core helper attributes to core (non-Lightning) IT classes."""
    def __init__(self, *args, **kwargs) -> None:
        # for core/non-lightning modules, we configure a _log_dir rather than relying on the trainer to do so
        # if using a Lightning module, we can access the trainer log_dir via the `core_log_dir` property
        # or continue to rely on the trainer log_dir directly
        if it_cfg := kwargs.get('it_cfg', None):
            self._log_dir = Path(it_cfg.core_log_dir or tempfile.gettempdir())
            ca = it_cfg.lightning_compat_attrs
        else:
            raise MisconfigurationException("CoreHelperAttributes requires an ITConfig.")
        self._supported_helper_attrs = {k: partial(_dummy_notify, k, v.ret_callable, v.ret_val) for k,v in ca.items()}
        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # NOTE: somewhat hacky way to dynamically stub a specified subset of Lightning module attributes
        # (if they don't already exist) to extend the cases where Lightning methods can be iteratively used for core
        # context experimentation
        if '_supported_helper_attrs' in self.__dict__:
            _helper_attrs= self.__dict__['_supported_helper_attrs']
            if name in _helper_attrs:
                return _helper_attrs[name]
        return super().__getattr__(name)  # the unresolved attribute wasn't ours, pass it on to PyTorch's __getattr__

    @property
    def current_epoch(self) -> int:
        return self._core_or_lightning(c2l_map_key="_current_epoch")

    # N.B. Lightning does not support directly setting `current_epoch` and `global_step`, instead using
    # `self.fit_loop.epoch_progress.current.completed` and `self.fit_loop.epoch_loop.global_step` respectively. Instead
    # of mocking analagous loop attributes, when using this mixin (and not Lightning), we allow settting the values
    # directly for convenience
    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        self._current_epoch = value

    @property
    def global_step(self) -> int:
        return self._core_or_lightning(c2l_map_key="_global_step")

    @global_step.setter
    def global_step(self, value: int) -> None:
        self._global_step = value

    @property
    def device(self) -> Optional[torch.device]:
        try:
            device = getattr(self, "_device", None) or reduce(getattr, "model.device".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a device reference (has it been set yet?): {ae}")
            device = None
        return device


# adapted from pytorch/core/optimizer.py initialization methods
class OptimizerScheduler:
    """" Barebones interface to setup optimizers and schedulers for manual optimization with core IT modules."""

    # proper initialization of these variables should be done in the child class
    it_optimizers: List[Optimizable]
    it_lr_scheduler_configs: List[LRSchedulerConfig]
    it_lr_schedulers: List[LRScheduler]

    def _it_init_optimizers_and_schedulers(self, optim_conf: Union[Dict[str, Any], List, Optimizer, Tuple]) \
        -> Tuple[List[Optimizer], List[LRSchedulerConfig]]:

        if optim_conf is None:
            rank_zero_info(  # TODO: maybe set a debug level instead?
                "`configure_optimizers` returned `None`, Interpretune will not configure an optimizer or scheduler.",
            )

        optims, lrs = OptimizerScheduler._configure_optimizers(optim_conf)
        lrs_configs = OptimizerScheduler._configure_schedulers_manual_opt(lrs)
        lrs = [lrs.scheduler for lrs in lrs_configs]
        self.it_optimizers, self.it_lr_scheduler_configs, self.it_lr_schedulers = optims, lrs_configs, lrs

    @staticmethod
    def _configure_schedulers_manual_opt(schedulers: list) -> List[LRSchedulerConfig]:
        """Convert each scheduler into `LRSchedulerConfig` structure with relevant information, when using manual
        optimization."""
        lr_scheduler_configs = []
        for scheduler in schedulers:
            if isinstance(scheduler, dict):
                # We do not filter out keys that are invalid with manual optimization to maximize flexibility (since
                # this is a core IT module context)
                #invalid_keys = {"reduce_on_plateau", "monitor", "strict", "interval"}

                config = LRSchedulerConfig(**{key: scheduler[key] for key in scheduler})
            else:
                config = LRSchedulerConfig(scheduler)
            lr_scheduler_configs.append(config)
        return lr_scheduler_configs

    @staticmethod
    def _configure_optimizers(
        optim_conf: Union[Dict[str, Any], List, Optimizer, Tuple]
    ) -> Tuple[List, List]:
        # for basic optimizer/scheduler init, supporting Lightning optimizer/scheduler formats, Note we do not subject
        # configuration to Lightning validation constraints to increase flexibility in exploratory experimental mode.
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
                rank_zero_warn("Interpretune does not support `monitor` in `configure_optimizers` with the core IT"
                               " module context.")
            lr_schedulers = [optim_conf["lr_scheduler"]] if "lr_scheduler" in optim_conf else []
        # multiple dictionaries
        elif isinstance(optim_conf, (list, tuple)) and all(isinstance(d, dict) for d in optim_conf):
            optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
            scheduler_dict = lambda scheduler: dict(scheduler) if isinstance(scheduler, dict) else \
                {"scheduler": scheduler}
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


class CoreComponents(ConfigAdapter, PropertyDispatcher, OptimizerScheduler):
    ...
