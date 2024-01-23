from typing import Optional, List

import torch
from torch.nn import CrossEntropyLoss
import evaluate
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.config.module import ITConfig
from interpretune.base.mixins.core import ProfilerHooksMixin
from it_examples.experiments.rte_boolq.datamodules import DEFAULT_TASK, TASK_NUM_LABELS, INVALID_TASK_MSG
from interpretune.utils.types import STEP_OUTPUT
from interpretune.utils.logging import rank_zero_warn, rank_zero_info

class RTEBoolqClassificationHeadSteps:
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

    # TODO: test overriding default test_step
    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      pad_token_id=self.datamodule.tokenizer.pad_token_id,
                                      **self.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__)
        stacked_scores = torch.stack([out for out in outputs['scores']], dim=0).cpu()
        assert self.it_cfg.entailment_mapping_indices is not None
        answer_logits = torch.index_select(stacked_scores, -1, self.it_cfg.entailment_mapping_indices)
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
        # TODO: condition this on a metric being configured
        #self.log("predict_loss", test_loss, prog_bar=True, sync_dist=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

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


class RTEBoolqLMHeadSteps:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # when using TransformerLens, we need to manually calculate our loss from logit output
        self.loss_fn = CrossEntropyLoss()

    def labels_to_ids(self, labels: List[str]) -> List[int]:
        return torch.take(self.it_cfg.entailment_mapping_indices, labels)

    def logits_and_labels(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        label_ids = self.labels_to_ids(batch.pop("labels"))
        logits = self(**batch)
        return torch.squeeze(logits[:, -1, :], dim=1), label_ids

    @ProfilerHooksMixin.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        # TODO: need to be explicit about the compatibility constraints/contract
        answer_logits, labels = self.logits_and_labels(batch, batch_idx)
        loss = self.loss_fn(answer_logits, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @ProfilerHooksMixin.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        answer_logits, labels = self.logits_and_labels(batch, batch_idx)
        val_loss = self.loss_fn(answer_logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        # TODO: condition this on a metric being configured and calculate per_example_answers for metric input
        # like with zero_shot_test_step
        #metric_dict = self.metric.compute(predictions=answer_logits, references=labels)
        #metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               #metric_dict.items()))
        #self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self.model.generate(input=batch['input'],
                                      #pad_token_id=self.datamodule.tokenizer.pad_token_id,
                                      **self.it_cfg.lm_generation_cfg.__dict__)
        #stacked_scores = torch.stack([out for out in outputs.logits], dim=0).cpu()
        stacked_scores = outputs.logits.cpu()
        assert self.it_cfg.entailment_mapping_indices is not None
        answer_logits = torch.index_select(stacked_scores, -1, self.it_cfg.entailment_mapping_indices)
        per_example_answers, _ = torch.max(answer_logits, dim=1)
        preds = torch.argmax(per_example_answers, axis=1)  # type: ignore[call-arg]
        #labels = batch["labels"]
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        # run predict on val dataset for now
        batch.pop("labels")
        outputs = self(**batch)
        # TODO: switch to zero shot instead of this default sequenceclassification head approach
        logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            logits.squeeze()
        #labels = batch["labels"]
        # TODO: move TL examples to use zeroshot instead of default test step
        # TODO: condition this on a metric being configured
        #self.log("predict_loss", test_loss, prog_bar=True, sync_dist=True)
        # metric_dict = self.metric.compute(predictions=preds, references=labels)
        # metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
        #                        metric_dict.items()))
        # self.log_dict(metric_dict, prog_bar=True, sync_dist=True)


class RTEBoolqModuleMixin:

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._init_entailment_mapping()

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        if it_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(it_cfg.task_name + INVALID_TASK_MSG)
            it_cfg.task_name = DEFAULT_TASK
        it_cfg.num_labels = 0 if it_cfg.zero_shot_cfg.enabled else TASK_NUM_LABELS[it_cfg.task_name]
        return it_cfg

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self.init_hparams['experiment_id'])

    def _init_entailment_mapping(self) -> None:
        ent_cfg, tokenizer = self.it_cfg, self.datamodule.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(ent_cfg.entailment_mapping)
        device = self.device if isinstance(self.device, torch.device) else self.output_device
        ent_cfg.entailment_mapping_indices = torch.tensor(token_ids).to(device)


class RTEBoolqClassificationModule(RTEBoolqClassificationHeadSteps, RTEBoolqModuleMixin, torch.nn.Module):
    ...


class RTEBoolqLMHeadModule(RTEBoolqLMHeadSteps, RTEBoolqModuleMixin, torch.nn.Module):
    ...
