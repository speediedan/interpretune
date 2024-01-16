from typing import Optional
import logging

import torch
from torch.utils.data import DataLoader
import datasets

from interpretune.config.datamodule import ITDataModuleConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.logging import rank_zero_warn

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
        # note for raw pytorch we require a target_model (vs getting it from the trainer in the lightning version)
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
        return self.dataloader_factory()
