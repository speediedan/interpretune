from typing import Any, Dict, List, Union, Tuple

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
