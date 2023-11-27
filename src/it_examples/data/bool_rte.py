from typing import Optional
import logging

import torch
from torch.utils.data import DataLoader
import datasets

from interpretune.base.base_datamodule import ITDataModule

log = logging.getLogger(__name__)


class BoolRTEDataModule(ITDataModule):

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign
        # state (e.g. self.x=y)
        # note for raw pytorch we require a target_model (vs getting it from the trainer in the lightning version)
        dataset = datasets.load_dataset("super_glue", self.itdm_cfg.task_name)
        for split in dataset.keys():
            if split == 'validation' or not self.itdm_cfg.prepare_validation_set_only:
                dataset[split] = dataset[split].map(self.tokenization_func, batched=True)
                dataset[split] = self._remove_unused_columns(dataset[split], target_model)
        if self.itdm_cfg.prepare_validation_set_only:
            for split in ['test', 'train']:
                del dataset[split]
        dataset.save_to_disk(self.itdm_cfg.dataset_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], batch_size=self.itdm_cfg.train_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validation"], batch_size=self.itdm_cfg.eval_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validation"], batch_size=self.itdm_cfg.eval_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validation"], batch_size=self.itdm_cfg.eval_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)
