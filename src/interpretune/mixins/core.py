import tempfile
from typing import Any, Dict, List, Union, Tuple, Optional
from functools import reduce, partial
from pathlib import Path
from contextlib import contextmanager
from functools import wraps

import torch
from transformer_lens import HookedTransformerConfig

from interpretune.utils.logging import rank_zero_info, rank_zero_warn
from interpretune.utils.exceptions import MisconfigurationException
from interpretune.utils.types import LRSchedulerConfig, Optimizer, Optimizable, LRScheduler

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

def _dummy_notify(method: str, ret_callable: bool, rv: Any, *args, **kwargs) -> Optional[Any]:
    rank_zero_warn(f"The `{method}` method is not defined for this module. For Lightning compatibility, this noop "
                    "method will be used. This warning will only be issued once by default.")
    out = lambda *args, **kwargs: rv if ret_callable else rv
    return out

class CoreHelperAttributeMixin:
    """Mixin class for adding arbitrary core helper attributes to core (non-Lightning) IT classes."""
    def __init__(self, *args, **kwargs) -> None:
        # for core/non-lightning modules, we configure a _log_dir rather than relying on the trainer to do so
        # if using a Lightning module, we can access the trainer log_dir via the `core_log_dir` property
        # or continue to rely on the trainer log_dir directly
        if it_cfg := kwargs.get('it_cfg', None):
            self._log_dir = Path(it_cfg.core_log_dir or tempfile.gettempdir())
            ca = it_cfg.lightning_compat_attrs
        else:
            raise MisconfigurationException("CoreHelperAttributeMixin requires an ITConfig.")
        self._supported_helper_attrs = {k: partial(_dummy_notify, k, v.ret_callable, v.ret_val) for k,v in ca.items()}
        # set initial defaults for non-Lightning modules
        self._current_epoch = 0
        self._global_step = 0
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

    @property
    def global_step(self) -> int:
        return self._core_or_lightning(c2l_map_key="_global_step")

    @property
    def device(self) -> Optional[torch.device]:
        try:
            device = getattr(self, "_device", None) or reduce(getattr, "model.device".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a device reference (has it been set yet?): {ae}")
            device = None
        return device


class TLensAttributeMixin:
    @property
    def tl_cfg(self) -> Optional[HookedTransformerConfig]:
        return self._core_or_lightning(c2l_map_key="_tl_cfg")

    @property
    def device(self) -> Optional[torch.device]:
        try:
            device = getattr(self, "_device", None) or getattr(self.tl_cfg, "device", None) or \
                reduce(getattr, "model.device".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a device reference (has it been set yet?): {ae}")
            device = None
        return device

# adapted from pytorch/core/optimizer.py initialization methods
class OptimizerSchedulerInitMixin:
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

        optims, lrs = OptimizerSchedulerInitMixin._configure_optimizers(optim_conf)
        lrs_configs = OptimizerSchedulerInitMixin._configure_schedulers_manual_opt(lrs)
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


class ProfilerHooksMixin:

    @contextmanager
    @staticmethod
    def memprofile_ctx(memprofiler, phase: str, epoch_idx: Optional[int] = None, step_idx: Optional[int] = None):
        try:
            memprofiler.snap(phase=phase, epoch_idx=epoch_idx, step_idx=step_idx, step_ctx="start")
            yield
        finally:
            memprofiler.snap(phase=phase, epoch_idx=epoch_idx, step_idx=step_idx, step_ctx="end", reset_mem_hooks=True)

    @staticmethod
    def memprofilable(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'memprofiler'):
                return func(self, *args, **kwargs)
            phase = func.__name__
            # for increased generality, we derive a profile `step_idx` based on a profiler snap counter rather than
            # parsing `args` if a `batch_idx` kwarg isn't found
            step_idx = kwargs.get("batch_idx", None)
            with ProfilerHooksMixin.memprofile_ctx(self.memprofiler, phase=phase, step_idx=step_idx):
                if self.memprofiler.memprofiler_cfg.enable_saved_tensors_hooks and \
                    self.memprofiler._enabled[(phase, 'start')]:
                    with torch.autograd.graph.saved_tensors_hooks(*self.memprofiler._saved_tensors_funcs):
                        return func(self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)
        return wrapper
