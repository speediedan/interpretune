import warnings
from typing import Any, Optional

import torch

from interpretune.base.config.module import ITConfig
from interpretune.base.hooks import BaseITHooks
from interpretune.base.components.core import CoreComponents, CoreHelperAttributes
from interpretune.base.components.mixins import CoreMixins
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.analysis.debug_generation import DebugGeneration
from interpretune.analysis.memprofiler import MemProfiler



# TODO: add core helper log/log_dict methods for core context usage
for warnf in [".*For Lightning compatibility, this noop .*",]:
    warnings.filterwarnings("once", warnf)

#class BaseITModule(ABC, CoreMixins, CoreComponents, BaseITHooks, torch.nn.Module):
class BaseITModule(CoreMixins, CoreComponents, BaseITHooks, torch.nn.Module):

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
        self.model: torch.nn.Module = None
        self.memprofiler: Optional[MemProfiler] = None
        self.debug_lm: Optional[DebugGeneration] = None
        self.it_cfg: ITConfig = self._before_it_cfg_init(it_cfg)
        self.connect_extensions()
        self._dispatch_model_init()

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        """Optionally modify configuration before it_cfg is initialized."""
        return it_cfg

    def _dispatch_model_init(self) -> None:
        if self.cuda_allocator_history:
            torch.cuda.memory._record_memory_history()
        self.auto_model_init()
        if self.model:
            self.post_auto_model_init()  # allow modification of a configuration-drive model
        else:
            self.model_init()  # load a custom model
        self._capture_hyperparameters()
        self.load_metric()

    def auto_model_init(self) -> None:
        """Can be overridden by subclasses to automatically initialize model from a configuration (e.g.
        hf_from_pretrained_cfg, tl_from_config etc.)."""
        if self.it_cfg.hf_from_pretrained_cfg:
            self.hf_pretrained_model_init()

    def post_auto_model_init(self) -> None:
        """Optionally modify init of an existing configuration-driven model."""

    def model_init(self) -> None:
        """Optionally load a custom model ."""

    def load_metric(self) -> None:
        """Optionally load a metric at the end of model initialization."""

    def on_session_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if getattr(self, 'memprofiler', None):
            self.memprofiler.dump_memory_stats()
        self._it_state._session_complete = True


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
