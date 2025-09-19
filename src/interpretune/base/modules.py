from typing import Any, Optional

import torch

from interpretune.config import ITConfig
from interpretune.base import BaseITHooks, BaseITComponents, BaseITMixins

# # TODO: add core helper log/log_dict methods for core context usage
# for warnf in [".*For framework compatibility, this noop .*",]:
#     warnings.filterwarnings("once", warnf)


class BaseITModule(BaseITMixins, BaseITComponents, BaseITHooks, torch.nn.Module):  # type: ignore[misc]
    def __init__(self, it_cfg: ITConfig, *args, **kwargs):
        """"""
        # See NOTE [Interpretune Dataclass-Oriented Configuration]
        super().__init__(*args, **kwargs)
        self.model: torch.nn.Module = None  # type: ignore[assignment]  # initialized later in lifecycle
        self.it_cfg: ITConfig = self._before_it_cfg_init(it_cfg)
        self._connect_extensions()
        self._dispatch_model_init()

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        """Optionally modify configuration before it_cfg is initialized."""
        return it_cfg

    def _dispatch_model_init(self) -> None:
        if self.cuda_allocator_history:
            torch.cuda.memory._record_memory_history()
        self.auto_model_init()
        if self.model:
            self.post_auto_model_init()  # allow modification of a configuration-driven model
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
        if getattr(self, "memprofiler", None):
            self.memprofiler.dump_memory_stats()
        self._it_state._session_complete = True

    def __repr__(self) -> str:
        try:
            state_summary = getattr(self, "_it_state", None)
            state_str = repr(state_summary) if state_summary is not None else "ITState(<no state>)"
            model_name = getattr(self.model, "__class__", None)
            model_repr = model_name.__name__ if model_name is not None else str(self.model)
            return f"{self.__class__.__name__}({state_str}, model={model_repr})"
        except Exception:
            return super().__repr__()
