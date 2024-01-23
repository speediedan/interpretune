from typing import Any, Optional
import torch
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.config.module import ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.import_utils import instantiate_class
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.types import STEP_OUTPUT, OptimizerLRScheduler

class BaseITHooks:
    """" LightningModule hooks implemented by BaseITModule."""

    # if you override these in your LightningModule, ensure you cooperatively call super() if you want to retain
    # the relevant BaseITModule hook functionality
    # proper initialization of these variables should be done in the child class
    it_cfg: ITConfig
    session_complete: bool
    _datamodule: ITDataModule
    model: torch.nn.Module

    def setup(self, *args, **kwargs) -> None:
        # TODO: add super() calls to these methods once Interpretunable protocol is defined
        # super().setup(*args, **kwargs)
        # if we're setting up a Lightning module, datamodule is not provided in setup and will be accessed via Trainer
        if datamodule := kwargs.get("datamodule", None):
            self._datamodule = datamodule
        self._init_dirs_and_hooks()

    def configure_optimizers(self) -> Optional[OptimizerLRScheduler]:
        """Optional because it is not mandatory in the context of core IT modules (required for Lightning
        modules)."""
        # With FTS >= 2.0, ``FinetuningScheduler`` simplifies initial optimizer configuration by ensuring the optimizer
        # configured here will optimize the parameters (and only those parameters) scheduled to be optimized in phase 0
        # of the current fine-tuning schedule. This auto-configuration can be disabled if desired by setting
        # ``enforce_phase0_params`` to ``False``.
        self._set_input_require_grads()
        optimizer, scheduler = None, None
        if self.it_cfg.optimizer_init:  # in case this hook is manually invoked by the user
            optimizer = instantiate_class(args=self.model.parameters(), init=self.it_cfg.optimizer_init)
        if self.it_cfg.lr_scheduler_init:
            scheduler = {
                "scheduler": instantiate_class(args=optimizer, init=self.it_cfg.lr_scheduler_init),
                **self.it_cfg.pl_lrs_cfg,
            }
        return [optimizer], [scheduler]

    # N.B. we call `on_session_end` at the end of train, test and predict session types only. This is because Lightning
    # calls both `on_train_end` and `on_validation_end` with most training sessions when running both a fit and
    # evaluation loop as is usually the case) but only `on_test_end` with the test stage.
    # The `on_run_end` hook for Lightning is roughly analogous to Interpretune's `on_session_end` hook but is not
    # a hook Lightning makes available to users.
    def on_train_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def on_test_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def on_predict_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def forward(self, **inputs: Any) -> STEP_OUTPUT:
        return self.model(**inputs, **self.it_cfg.cust_fwd_kwargs)

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        rank_zero_warn("`training_step` must be implemented to be used with the Interpretune.")

    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        rank_zero_warn("`validation_step` must be implemented to be used with the Interpretune.")

    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        rank_zero_warn("`validation_step` must be implemented to be used with the Interpretune.")

    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        rank_zero_warn("`prediction_step` must be implemented to be used with the Interpretune.")
