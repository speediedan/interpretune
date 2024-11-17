import os
from typing import Any, Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from pprint import pformat
import logging
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import evaluate
import datasets
from transformers import PreTrainedTokenizerBase
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.adapters.transformer_lens import ITLensConfig
from interpretune.adapters.sae_lens import SAELensConfig
from interpretune.base.config.datamodule import PromptConfig, ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.base.config.mixins import ZeroShotClassificationConfig, BaseGenerationConfig, HFGenerationConfig
from interpretune.base.components.mixins import ProfilerHooksMixin
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.types import STEP_OUTPUT
from interpretune.utils.tokenization import _sanitize_input_name


log = logging.getLogger(__name__)

TASK_TEXT_FIELD_MAP = {"rte": ("premise", "hypothesis"), "boolq": ("passage", "question")}
TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"
INVALID_TASK_MSG = f" is an invalid task_name. Proceeding with the default task: {DEFAULT_TASK!r}"


@dataclass(kw_only=True)
class RTEBoolqEntailmentMapping:
    entailment_mapping: Tuple = ("Yes", "No")  # RTE style, invert mapping for BoolQ
    entailment_mapping_indices: Optional[torch.Tensor] = None


@dataclass(kw_only=True)
class RTEBoolqZeroShotClassificationConfig(RTEBoolqEntailmentMapping, ZeroShotClassificationConfig):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=HFGenerationConfig)

    def __repr__(self):
        return f"Zero-Shot Classification Config: {os.linesep}{pformat(self.__dict__)}"


@dataclass(kw_only=True)
class RTEBoolqPromptConfig(PromptConfig):
    ctx_question_join: str = 'Does the previous passage imply that '
    question_suffix: str = '? Answer with only one word, either Yes or No.'
    cust_task_prompt: Optional[Dict[str, Any]] = None

# add our custom model attributes
@dataclass(kw_only=True)
class RTEBoolqConfig(RTEBoolqEntailmentMapping, ITConfig):
    ...


@dataclass(kw_only=True)
class RTEBoolqTLConfig(RTEBoolqEntailmentMapping, ITLensConfig):
    ...

@dataclass(kw_only=True)
class RTEBoolqSLConfig(RTEBoolqEntailmentMapping, SAELensConfig):
    ...


class RTEBoolqDataModule(ITDataModule):
    def __init__(self, itdm_cfg: ITDataModuleConfig) -> None:
        if itdm_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(itdm_cfg.task_name + INVALID_TASK_MSG)
            itdm_cfg.task_name = DEFAULT_TASK
        itdm_cfg.text_fields = TASK_TEXT_FIELD_MAP[itdm_cfg.task_name]
        super().__init__(itdm_cfg=itdm_cfg)

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        # N.B. prepare_data is called in a single process (rank 0, either per node or globally) so do not use it to
        # assign state (e.g. self.x=y)
        # note for raw pytorch we require a target_model
        # HF Datasets' transformation cache fingerprinting algo necessitates construction of these partials as the hash
        # is generated using function args, dataset file, mapping args: https://bit.ly/HF_Datasets_fingerprint_algo)
        tokenization_func = partial(
            self.encode_for_rteboolq,
            tokenizer=self.tokenizer,
            text_fields=self.itdm_cfg.text_fields,
            prompt_cfg=self.itdm_cfg.prompt_cfg,
            template_fn=self.itdm_cfg.prompt_cfg.model_chat_template_fn,
            #template_fn=self.model_chat_template_fn,
            tokenization_pattern=self.itdm_cfg.cust_tokenization_pattern,
        )
        dataset = datasets.load_dataset("super_glue", self.itdm_cfg.task_name, trust_remote_code=True)
        for split in dataset.keys():
            dataset[split] = dataset[split].map(tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
            dataset[split] = self._remove_unused_columns(dataset[split], target_model)
        dataset.save_to_disk(self.itdm_cfg.dataset_path)

    def dataloader_factory(self, split: str, use_train_batch_size: bool = False) -> DataLoader:
        dataloader_kwargs = {"dataset": self.dataset[split], "collate_fn":self.data_collator,
                             **self.itdm_cfg.dataloader_kwargs}
        dataloader_kwargs['batch_size'] = self.itdm_cfg.train_batch_size if use_train_batch_size else \
            self.itdm_cfg.eval_batch_size
        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split='train', use_train_batch_size=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split='validation')

    def test_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split='validation')

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split='validation')

    # @staticmethod
    # def model_chat_template_fn(task_prompt: str, prompt_cfg: Optional[PromptConfig] = None,
    #                            tokenization_pattern: Optional[str] = None) -> str:
    #     return task_prompt.strip()

    #TODO: relax PreTrainedTokenizerBase to the protocol that is actually required
    @staticmethod
    def encode_for_rteboolq(example_batch: LazyDict, tokenizer: PreTrainedTokenizerBase, text_fields: List[str],
                            prompt_cfg: PromptConfig, template_fn: Callable,
                            tokenization_pattern: Optional[str] = None) -> BatchEncoding:
        example_batch['sequences'] = []
        # TODO: use promptsource instead of this manual approach after tinkering
        for field1, field2 in zip(example_batch[text_fields[0]],
                                  example_batch[text_fields[1]]):
            if prompt_cfg.cust_task_prompt:
                task_prompt = (prompt_cfg.cust_task_prompt['context'] + "\n" +
                               field1 + "\n" +
                               prompt_cfg.cust_task_prompt['question'] + "\n" +
                               field2)
            else:
                task_prompt = (field1 + prompt_cfg.ctx_question_join + field2 \
                               + prompt_cfg.question_suffix)
            # sequence = template_fn(prompt_cfg=prompt_cfg, task_prompt=task_prompt,
            #                        tokenization_pattern=tokenization_pattern)
            sequence = template_fn(task_prompt=task_prompt, tokenization_pattern=tokenization_pattern)
            example_batch['sequences'].append(sequence)
        features = tokenizer.batch_encode_plus(example_batch["sequences"], padding="longest",
                                               padding_side=tokenizer.padding_side)
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        features = _sanitize_input_name(tokenizer.model_input_names, features)
        return features

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
        outputs = self.it_generate(batch, **self.it_cfg.zero_shot_cfg.lm_generation_cfg.generate_kwargs)
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
        if logits.shape[-1] != self.it_cfg.num_labels:
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
