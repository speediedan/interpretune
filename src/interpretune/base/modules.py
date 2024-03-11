import warnings
from typing import Any, Optional, List
from abc import ABC

import torch

from interpretune.base.config.module import ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.base.hooks import BaseITHooks
from interpretune.base.components.core import CoreComponents, CoreHelperAttributes
from interpretune.base.components.mixins import CoreMixins
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.analysis.debug_generation import DebugGeneration
from interpretune.analysis.memprofiler import MemProfiler
from interpretune.utils.types import LRSchedulerConfig, Optimizable, LRScheduler



# TODO: add core helper log/log_dict methods for core context usage
for warnf in [".*For Lightning compatibility, this noop .*",]:
    warnings.filterwarnings("once", warnf)


class BaseITModule(ABC, CoreMixins, CoreComponents, BaseITHooks, torch.nn.Module):

    def __init__(
        self,
        it_cfg: ITConfig,
        *args,
        **kwargs
    ):
        """In this example, this :class:`~lightning.pytorch.core.module.LightningModule` is initialized by composing
        the ./config/fts_defaults.yaml default configuration with various scheduled fine-tuning yaml configurations
        via the :class:`~lightning.pytorch.cli.LightningCLI` but it can be used like any other
        :class:`~lightning.pytorch.core.module.LightningModule` as well.

        Args:
            it_cfg (ITConfig): Configuration for this
                :class:`~lightning.pytorch.core.module.LightningModule`.
        """
        # See NOTE [Interpretune Dataclass-Oriented Configuration]
        super().__init__(*args, **kwargs)
        self._init_direct_attrs()
        self.it_cfg: ITConfig = self._before_it_cfg_init(it_cfg)
        self.connect_extensions()
        self.model_init()

    def _init_direct_attrs(self):
        """Direct base module attribute initialization."""
        # TODO: maybe move these to the top of the class instead of a separate function
        self.model: torch.nn.Module = None
        self.it_optimizers: List[Optimizable] = None  # initialized via core IT module `configure_optimizers` hook
        self.it_lr_scheduler_configs: List[LRSchedulerConfig] = None
        self.it_lr_schedulers: List[LRScheduler] = None
        self.memprofiler: Optional[MemProfiler] = None
        self.debug_lm: Optional[DebugGeneration] = None
        self._datamodule: Optional[ITDataModule] = None  # datamodule handle attached after init
        self._device: Optional[torch.device] = None  # root device (sometimes used if not handled by Lightning)
        self._session_complete: bool = False
        self.init_hparams = {}

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        """Optionally modify configuration before it_cfg is initialized."""
        return it_cfg

    def model_init(self) -> None:
        if self.cuda_allocator_history:
            torch.cuda.memory._record_memory_history()
        if self.it_cfg.hf_from_pretrained_cfg:
            self.hf_pretrained_model_init()
        else:
            self.custom_model_init()
        self._capture_hyperparameters()
        self.load_metric()

    def custom_model_init(self) -> None:
        """Optionally load a custom configured model instead of using HF pretrained-based initialization."""
        # see transformer_lens plugin for an example

    def load_metric(self) -> None:
        """Optionally load a metric at the end of model initialization."""

    def on_session_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if getattr(self, 'memprofiler', None):
            self.memprofiler.dump_memory_stats()
        self._session_complete = True


class ITModule(CoreHelperAttributes, BaseITModule):
    """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation model
    on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """
    ...


if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import LightningModule

    class ITLightningModule(BaseITModule, LightningModule):
        """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation
        model on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
        implementations of a given model and the `SuperGLUE Hugging Face dataset.

        <https://huggingface.co/datasets/super_glue#data-instances>`_.
        """
        def on_train_start(self) -> None:
            # ensure model is in training mode (e.g. needed for some edge cases w/ skipped sanity checking)
            self.model.train()
            return super().on_train_start()
else:
    ITLightningModule = object
