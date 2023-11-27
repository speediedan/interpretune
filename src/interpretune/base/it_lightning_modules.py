from typing import Any, Optional
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer
import finetuning_scheduler as fts

from interpretune.utils.import_utils import instantiate_class
from interpretune.utils.logging import rank_zero_info
from interpretune.base.it_module import BaseITModule, ITHookedModule
from interpretune.base.it_datamodule import ITDataModule


class ITLightningDataModule(ITDataModule, pl.LightningDataModule):
    ...

class ITLightningModule(BaseITModule, pl.LightningModule):
    """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation model
    on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """

    def setup(self, stage: str) -> None:
        if self.it_cfg.zero_shot_cfg.enabled:
            tokenizer, zs_cfg = self.trainer.datamodule.tokenizer, self.it_cfg.zero_shot_cfg
            zs_cfg.entailment_mapping_indices = torch.tensor(tokenizer.convert_tokens_to_ids(zs_cfg.entailment_mapping))

    def forward(self, **inputs: Any) -> STEP_OUTPUT:
        return self.model(**inputs)

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        if self.it_cfg.debug_lm_cfg.enabled and self.it_cfg.debug_lm_cfg.record_memory_history:
            torch.cuda.memory._dump_snapshot(f"/tmp/{self.init_hparams['experiment_id']}_start_train_step_{self.global_step}.pickle")
        loss = self(**batch)[0]
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.it_cfg.debug_lm_cfg.enabled and self.it_cfg.debug_lm_cfg.record_memory_history:
            torch.cuda.memory._dump_snapshot(f"/tmp/{self.init_hparams['experiment_id']}_start_train_epoch_{self.current_epoch}.pickle")
        assert self.logger is not None
        if self.finetuningscheduler_callback:
            self.logger.log_metrics(
                metrics={"finetuning_schedule_depth": float(self.finetuningscheduler_callback.curr_depth)},
                step=self.global_step,
            )

    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.it_cfg.zero_shot_cfg.enabled:
            self.zero_shot_test_step(batch, batch_idx)
        else:
            self.default_test_step(batch, batch_idx)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      pad_token_id=self.trainer.datamodule.tokenizer.pad_token_id,
                                      **self.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__)
        stacked_scores = torch.stack([out for out in outputs['scores']], dim=0).cpu()
        assert self.it_cfg.zero_shot_cfg.entailment_mapping_indices is not None
        answer_logits = torch.index_select(stacked_scores, -1, self.it_cfg.zero_shot_cfg.entailment_mapping_indices)
        per_example_answers, _ = torch.max(answer_logits, dim=0)
        preds = torch.argmax(per_example_answers, axis=1)  # type: ignore[call-arg]
        labels = batch["labels"]
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        # run predict on val dataset for now
        outputs = self(**batch)
        test_loss, logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("predict_loss", test_loss, prog_bar=True, sync_dist=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

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
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        rank_zero_info(metric_dict)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # With FTS >= 2.0, ``FinetuningScheduler`` simplifies initial optimizer configuration by ensuring the optimizer
        # configured here will optimize the parameters (and only those parameters) scheduled to be optimized in phase 0
        # of the current fine-tuning schedule. This auto-configuration can be disabled if desired by setting
        # ``enforce_phase0_params`` to ``False``.
        self.model.enable_input_require_grads()
        optimizer = instantiate_class(args=self.model.parameters(), init=self.init_hparams['optimizer_init'])
        scheduler = {
            "scheduler": instantiate_class(args=optimizer, init=self.init_hparams['lr_scheduler_init']),
            **self.init_hparams['pl_lrs_cfg'],
        }
        return [optimizer], [scheduler]

class ITHookedLightningModule(ITHookedModule, ITLightningModule):

    @property
    def finetuningscheduler_callback(self) -> Optional[fts.FinetuningScheduler]:  # type: ignore
        fts_callback = [c for c in self.trainer.callbacks if isinstance(c, fts.FinetuningScheduler)]  # type: ignore
        return fts_callback[0] if fts_callback else None

    def setup(self, stage: str) -> None:
        if self.it_cfg.tlens_from_pretrained_cfg.enabled:
            self._convert_hf_to_hooked()
        if self.it_cfg.zero_shot_cfg.enabled:
            tokenizer, zs_cfg = self.trainer.datamodule.tokenizer, self.it_cfg.zero_shot_cfg
            zs_cfg.entailment_mapping_indices = torch.tensor(tokenizer.convert_tokens_to_ids(zs_cfg.entailment_mapping))
        self.dump_base = Path(self.trainer.model._trainer.log_dir)

    def _convert_hf_to_hooked(self) -> HookedTransformer:
        self.model = HookedTransformer.from_pretrained(hf_model=self.model, tokenizer=self.trainer.datamodule.tokenizer,
                                                  **self.it_cfg.tlens_from_pretrained_cfg.__dict__)
