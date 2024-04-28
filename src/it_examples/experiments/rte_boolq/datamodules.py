from typing import Optional, Any
import logging

import torch
from torch.utils.data import DataLoader
import datasets
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.tokenization import _sanitize_input_name

log = logging.getLogger(__name__)


TASK_TEXT_FIELD_MAP = {"rte": ("premise", "hypothesis"), "boolq": ("passage", "question")}
TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"
INVALID_TASK_MSG = f" is an invalid task_name. Proceeding with the default task: {DEFAULT_TASK!r}"

class RTEBoolqDataModule(ITDataModule):
    def __init__(self, itdm_cfg: ITDataModuleConfig) -> None:
        if itdm_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(itdm_cfg.task_name + INVALID_TASK_MSG)
            itdm_cfg.task_name = DEFAULT_TASK
        itdm_cfg.text_fields = TASK_TEXT_FIELD_MAP[itdm_cfg.task_name]
        super().__init__(itdm_cfg=itdm_cfg)

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign
        # state (e.g. self.x=y)
        # note for raw pytorch we require a target_model
        dataset = datasets.load_dataset("super_glue", self.itdm_cfg.task_name, trust_remote_code=True)
        for split in dataset.keys():
            dataset[split] = dataset[split].map(self.tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
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


class GPT2RTEBoolqDataModule(RTEBoolqDataModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenization_func = self._tokenize_for_gpt2

    def _tokenize_for_gpt2(self, example_batch: LazyDict) -> BatchEncoding:
        example_batch['sequences'] = []
        assert example_batch is not None
        assert self.itdm_cfg.text_fields is not None
        assert self.itdm_cfg.prompt_cfg is not None
        # TODO: use promptsource instead of this manual approach after tinkering
        for field1, field2 in zip(example_batch[self.itdm_cfg.text_fields[0]],
                                  example_batch[self.itdm_cfg.text_fields[1]]):
            if self.itdm_cfg.prompt_cfg.cust_task_prompt:
                task_prompt = (self.itdm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" + field1 + "\n\n" +
                               self.itdm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" + field2)
            else:
                task_prompt = (field1 + self.itdm_cfg.prompt_cfg.ctx_question_join + field2 \
                               + self.itdm_cfg.prompt_cfg.question_suffix)
            sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        features = _sanitize_input_name(self.tokenizer.model_input_names, features)
        return features


class Llama2RTEBoolqDataModule(RTEBoolqDataModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenization_func = self._tokenize_for_llama2

    def _tokenize_for_llama2(self, example_batch: LazyDict) -> BatchEncoding:
        example_batch['sequences'] = []
        assert example_batch is not None
        assert self.itdm_cfg.text_fields is not None
        assert self.itdm_cfg.prompt_cfg is not None
        # TODO: use promptsource instead of this manual approach after tinkering
        for field1, field2 in zip(example_batch[self.itdm_cfg.text_fields[0]],
                                  example_batch[self.itdm_cfg.text_fields[1]]):
            if self.itdm_cfg.prompt_cfg.cust_task_prompt:
                task_prompt = (self.itdm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" +
                               field1 + "\n" +
                               self.itdm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" +
                               field2)
            else:
                task_prompt = (field1 + self.itdm_cfg.prompt_cfg.ctx_question_join + field2 \
                               + self.itdm_cfg.prompt_cfg.question_suffix)
            if self.itdm_cfg.cust_tokenization_pattern == "llama2-chat":
                sequence = self.itdm_cfg.prompt_cfg.SYS_PREFIX + \
                    f"{task_prompt.strip()} {self.itdm_cfg.prompt_cfg.E_INST}"
            else:
                sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer.batch_encode_plus(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        return features
