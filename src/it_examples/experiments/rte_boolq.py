import os
from typing import Any, Dict, Optional, Tuple, List, Callable, Generator
from dataclasses import dataclass, field
from pprint import pformat
import logging
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import evaluate
import datasets
from transformers import PreTrainedTokenizerBase
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding

import interpretune as it
from interpretune import (
    ITDataModule,  # type: ignore[attr-defined]  # complex import hook pattern
    MemProfilerHooks,  # type: ignore[attr-defined]  # complex import hook pattern
    AnalysisBatch,  # type: ignore[attr-defined]  # complex import hook pattern
    ITLensConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    SAELensConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    PromptConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    ITDataModuleConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    ITConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    GenerativeClassificationConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    BaseGenerationConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    HFGenerationConfig,  # type: ignore[attr-defined]  # complex import hook pattern
    rank_zero_warn,  # type: ignore[attr-defined]  # complex import hook pattern
    sanitize_input_name,  # type: ignore[attr-defined]  # complex import hook pattern
    STEP_OUTPUT,  # type: ignore[attr-defined]  # complex import hook pattern
)
# from interpretune.config import (ITLensConfig, SAELensConfig, PromptConfig, ITDataModuleConfig, ITConfig,
#                                  GenerativeClassificationConfig, BaseGenerationConfig, HFGenerationConfig)
# from interpretune.base import MemProfilerHooks, ITDataModule
# from interpretune.utils import rank_zero_warn, sanitize_input_name
# from interpretune.protocol import STEP_OUTPUT


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
class RTEBoolqGenerativeClassificationConfig(RTEBoolqEntailmentMapping, GenerativeClassificationConfig):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=HFGenerationConfig)

    def __repr__(self):
        return f"Generative Classification Config: {os.linesep}{pformat(self.__dict__)}"


@dataclass(kw_only=True)
class RTEBoolqPromptConfig(PromptConfig):
    ctx_question_join: str = "Does the previous passage imply that "
    question_suffix: str = "? Answer with only one word, either Yes or No."
    cust_task_prompt: Optional[Dict[str, Any]] = None


# add our custom model attributes
@dataclass(kw_only=True)
class RTEBoolqConfig(RTEBoolqEntailmentMapping, ITConfig): ...


@dataclass(kw_only=True)
class RTEBoolqTLConfig(RTEBoolqEntailmentMapping, ITLensConfig): ...


@dataclass(kw_only=True)
class RTEBoolqSLConfig(RTEBoolqEntailmentMapping, SAELensConfig): ...


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
        # NOTE [HF Datasets Transformation Caching]:
        # HF Datasets' transformation cache fingerprinting algo necessitates construction of these partials as the hash
        # is generated using function args, dataset file, mapping args: https://bit.ly/HF_Datasets_fingerprint_algo)
        tokenization_func = partial(
            self.encode_for_rteboolq,
            tokenizer=self.tokenizer,
            text_fields=self.itdm_cfg.text_fields,
            prompt_cfg=self.itdm_cfg.prompt_cfg,
            template_fn=self.itdm_cfg.prompt_cfg.model_chat_template_fn,
            tokenization_pattern=self.itdm_cfg.cust_tokenization_pattern,
        )
        dataset = datasets.load_dataset("aps/super_glue", self.itdm_cfg.task_name)
        for split in dataset.keys():
            dataset[split] = dataset[split].map(tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
            dataset[split] = self._remove_unused_columns(dataset[split], target_model)

        save_path = Path(self.itdm_cfg.dataset_path)
        dataset.save_to_disk(save_path)

    def dataloader_factory(self, split: str, use_train_batch_size: bool = False) -> DataLoader:
        dataloader_kwargs = {
            "dataset": self.dataset[split],
            "collate_fn": self.data_collator,
            **self.itdm_cfg.dataloader_kwargs,
        }
        dataloader_kwargs["batch_size"] = (
            self.itdm_cfg.train_batch_size if use_train_batch_size else self.itdm_cfg.eval_batch_size
        )
        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="train", use_train_batch_size=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="validation")

    def test_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="validation")

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="validation")

    # TODO: relax PreTrainedTokenizerBase to the protocol that is actually required
    @staticmethod
    def encode_for_rteboolq(
        example_batch: LazyDict,
        tokenizer: PreTrainedTokenizerBase,
        text_fields: List[str],
        prompt_cfg: PromptConfig,
        template_fn: Callable,
        tokenization_pattern: Optional[str] = None,
    ) -> BatchEncoding:
        example_batch["sequences"] = []
        # TODO: use promptsource instead of this manual approach after tinkering
        for field1, field2 in zip(example_batch[text_fields[0]], example_batch[text_fields[1]]):
            if prompt_cfg.cust_task_prompt:
                task_prompt = (
                    prompt_cfg.cust_task_prompt["context"]
                    + "\n"
                    + field1
                    + "\n"
                    + prompt_cfg.cust_task_prompt["question"]
                    + "\n"
                    + field2
                )
            else:
                task_prompt = field1 + prompt_cfg.ctx_question_join + field2 + prompt_cfg.question_suffix
            sequence = template_fn(task_prompt=task_prompt, tokenization_pattern=tokenization_pattern)
            example_batch["sequences"].append(sequence)
        features = tokenizer.batch_encode_plus(
            example_batch["sequences"], padding="longest", padding_side=tokenizer.padding_side
        )
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        features = sanitize_input_name(tokenizer.model_input_names, features)
        return features


class RTEBoolqSteps:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # when using TransformerLens, we need to manually calculate our loss from logit output
        self.loss_fn = CrossEntropyLoss()

    @MemProfilerHooks.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        # TODO: need to be explicit about the compatibility constraints/contract
        # TODO: note that this example uses generative_step_cfg and lm_head except for the test_step where we demo how
        # to use the GenerativeMixin to run inference with or without a generative_step_cfg enabled as well as with
        # different heads (e.g., seqclassification or LM head in this case)
        answer_logits, labels, _ = self.logits_and_labels(batch, batch_idx)
        loss = self.loss_fn(answer_logits, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @MemProfilerHooks.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        answer_logits, labels, orig_labels = self.logits_and_labels(batch, batch_idx)
        val_loss = self.loss_fn(answer_logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.collect_answers(answer_logits, orig_labels)

    @MemProfilerHooks.memprofilable
    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.it_cfg.generative_step_cfg.enabled:
            self.generative_classification_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        else:
            self.default_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def generative_classification_test_step(
        self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0
    ) -> Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self.it_generate(batch, **self.it_cfg.generative_step_cfg.lm_generation_cfg.generate_kwargs)
        self.collect_answers(outputs.logits, labels)

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self(**batch)
        self.collect_answers(outputs.logits, labels)

    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self(**batch)
        return self.collect_answers(outputs, labels, mode="return")

    def analysis_step(
        self,
        batch: BatchEncoding,
        batch_idx: int,
        dataloader_idx: int = 0,
        analysis_batch: Optional[AnalysisBatch] = None,
    ) -> Generator[STEP_OUTPUT, None, None]:
        """Run analysis operations on a batch and yield results."""
        # Demo mixing model methods and native IT analysis ops
        label_ids, orig_labels = self.labels_to_ids(batch.pop("labels"))
        analysis_batch = AnalysisBatch(label_ids=label_ids, orig_labels=orig_labels)
        op_kwargs = {"module": self, "batch": batch, "batch_idx": batch_idx}
        analysis_batch = it.model_cache_forward(analysis_batch=analysis_batch, **op_kwargs)
        analysis_batch = it.logit_diffs_cache(analysis_batch=analysis_batch, **op_kwargs)
        analysis_batch = it.sae_correct_acts(analysis_batch=analysis_batch, **op_kwargs)

        # note, there is an equivalent existing composite op for the decomposed version above:
        # analysis_batch = it.logit_diffs_sae(**op_kwargs)
        yield from self.analysis_cfg.save_batch(analysis_batch, batch, tokenizer=self.datamodule.tokenizer)


class RTEBoolqModuleMixin:
    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._init_entailment_mapping()

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        if it_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(it_cfg.task_name + INVALID_TASK_MSG)
            it_cfg.task_name = DEFAULT_TASK
        it_cfg.num_labels = 0 if it_cfg.generative_step_cfg.enabled else TASK_NUM_LABELS[it_cfg.task_name]
        return it_cfg

    def load_metric(self) -> None:
        self.metric = evaluate.load(
            "super_glue", self.it_cfg.task_name, experiment_id=self._it_state._init_hparams["experiment_id"]
        )

    # we override the default labels_to_ids method to demo using our module-specific attributes/logic
    def labels_to_ids(self, labels: List[str]) -> List[int]:
        return torch.take(self.it_cfg.entailment_mapping_indices, labels), labels

    # We override the default standardize_logits method to demo using custom attributes etc.
    def standardize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # to support genclassif/non-genclassif configs and LM/SeqClassification heads we adhere to the following logits
        # logical shape invariant: [batch size, positions to consider, answers to consider]
        if isinstance(logits, tuple):
            logits = torch.stack([out for out in logits], dim=1)
        logits = logits.to(device=self.device)
        if logits.ndim == 2:  # if answer logits have already been squeezed
            logits = logits.unsqueeze(1)
        if logits.shape[-1] != self.it_cfg.num_labels:
            logits = torch.index_select(logits, -1, self.it_cfg.entailment_mapping_indices)
            if not self.it_cfg.generative_step_cfg.enabled:
                logits = logits[:, -1:, :]
        return logits

    def _init_entailment_mapping(self) -> None:
        ent_cfg, tokenizer = self.it_cfg, self.datamodule.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(ent_cfg.entailment_mapping)
        device = self.device if isinstance(self.device, torch.device) else self.output_device
        ent_cfg.entailment_mapping_indices = torch.tensor(token_ids).to(device)


class RTEBoolqModule(RTEBoolqSteps, RTEBoolqModuleMixin, torch.nn.Module): ...
