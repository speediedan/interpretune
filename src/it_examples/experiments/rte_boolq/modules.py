from typing import Optional, List, Dict
import torch
from torch.nn import CrossEntropyLoss
import evaluate
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.config.module import ITConfig
from interpretune.base.components.mixins import ProfilerHooksMixin
from it_examples.experiments.rte_boolq.datamodules import DEFAULT_TASK, TASK_NUM_LABELS, INVALID_TASK_MSG
from interpretune.utils.types import STEP_OUTPUT
from interpretune.utils.logging import rank_zero_warn


class RTEBoolqSteps:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # when using TransformerLens, we need to manually calculate our loss from logit output
        self.loss_fn = CrossEntropyLoss()

    def labels_to_ids(self, labels: List[str]) -> List[int]:
        return torch.take(self.it_cfg.entailment_mapping_indices, labels), labels

    def logits_and_labels(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        label_ids, labels = self.labels_to_ids(batch.pop("labels"))
        logits = self(**batch)
        # TODO: add another layer of abstraction here to handle different model output types? Tradeoffs to consider...
        if not isinstance(logits, torch.Tensor):
            logits = logits.logits
            assert isinstance(logits, torch.Tensor), f"Expected logits to be a torch.Tensor but got {type(logits)}"
        return torch.squeeze(logits[:, -1, :], dim=1), label_ids, labels

    @ProfilerHooksMixin.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        # TODO: need to be explicit about the compatibility constraints/contract
        # TODO: note that this example uses zero_shot_cfg and lm_head except for the test_step where we demo how to
        # use the ZeroShotMixin to run inference with or without a zero_shot_cfg enabled as well as with different heads
        # (e.g., seqclassification or LM head in this case)
        answer_logits, labels, _ = self.logits_and_labels(batch, batch_idx)
        loss = self.loss_fn(answer_logits, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @ProfilerHooksMixin.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        answer_logits, labels, orig_labels = self.logits_and_labels(batch, batch_idx)
        val_loss = self.loss_fn(answer_logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.collect_answers(answer_logits, orig_labels)

    @ProfilerHooksMixin.memprofilable
    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.it_cfg.zero_shot_cfg.enabled:
            self.zero_shot_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        else:
            self.default_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self.it_generate(batch,
                                   pad_token_id=self.datamodule.tokenizer.pad_token_id,
                                   **self.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__)
        self.collect_answers(outputs.logits, labels)

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self(**batch)
        self.collect_answers(outputs.logits, labels)

    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self(**batch)
        return self.collect_answers(outputs, labels, mode='return')

    def collect_answers(self, logits: torch.Tensor | tuple, labels: torch.Tensor, mode: str = 'log') -> Optional[Dict]:
        logits = self.standardize_logits(logits)
        per_example_answers, _ = torch.max(logits, dim=-2)
        preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        # TODO: check if this type casting is still required for lightning torchmetrics, bug should be fixed now...
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        if mode == 'log':
            self.log_dict(metric_dict, prog_bar=True, sync_dist=True)
        else:
            return metric_dict

    def standardize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # to support zero_shot/non-zero_shot configs and LM/SeqClassification heads we adhere to the following logits
        # logical shape invariant: [batch size, positions to consider, answers to consider]
        if isinstance(logits, tuple):
            logits = torch.stack([out for out in logits], dim=1)
        logits = logits.to(device=self.device)
        if logits.ndim == 2:  # if answer logits have already been squeezed
            logits = logits.unsqueeze(1)
        if getattr(self.model, 'lm_head', None):
            logits = torch.index_select(logits, -1, self.it_cfg.entailment_mapping_indices)
            if not self.it_cfg.zero_shot_cfg.enabled:
                logits = logits[:, -1:, :]
        return logits


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

    def load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self._it_state._init_hparams['experiment_id'])

    def _init_entailment_mapping(self) -> None:
        ent_cfg, tokenizer = self.it_cfg, self.datamodule.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(ent_cfg.entailment_mapping)
        device = self.device if isinstance(self.device, torch.device) else self.output_device
        ent_cfg.entailment_mapping_indices = torch.tensor(token_ids).to(device)



class RTEBoolqModule(RTEBoolqSteps, RTEBoolqModuleMixin, torch.nn.Module):
    ...
