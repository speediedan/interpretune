from typing import Any, Optional
import torch
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.utils.import_utils import instantiate_class
from interpretune.config.module import ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.base.mixins.core import ProfilerHooksMixin
from interpretune.utils.logging import rank_zero_info
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

    @ProfilerHooksMixin.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
         # TODO: decide whether to build a closure for the core training_step to enable identical
         # core/lightning module training_steps in more cases (need to be explicit about the compatibility constraints)
        outputs = self(**batch)
        loss, _other_outputs = outputs[0], outputs[1:]
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @ProfilerHooksMixin.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        # TODO: condition this on a metric being configured
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    @ProfilerHooksMixin.memprofilable
    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.it_cfg.zero_shot_cfg.enabled:
            self.zero_shot_test_step(batch, batch_idx)
        else:
            self.default_test_step(batch, batch_idx)

    @ProfilerHooksMixin.memprofilable
    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        # run predict on val dataset for now
        # TODO: clean this up and allow for passing arbitrary data
        outputs = self(**batch)
        _, logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        # TODO: condition this on a metric being configured
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        rank_zero_info(metric_dict)
