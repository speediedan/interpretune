import os
import inspect
from typing import Optional, Callable
from abc import ABC

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader
from fts_examples import _HF_AVAILABLE

from interpretune.base.config_classes import ITDataModuleConfig
from interpretune.utils.cli import _import_class

if _HF_AVAILABLE:
    import datasets
    from datasets import Dataset
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase


class DataModuleInitMixin(ABC):

    def setup(self, stage: str) -> None:
        """Setup our dataset splits for training/validation."""

        self.dataset = datasets.load_from_disk(self.itdm_cfg.dataset_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], batch_size=self.hparams.itdm_cfg.train_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validation"], batch_size=self.hparams.itdm_cfg.eval_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validation"], batch_size=self.hparams.itdm_cfg.eval_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["validation"], batch_size=self.hparams.itdm_cfg.eval_batch_size,
                          collate_fn=self.data_collator, **self.itdm_cfg.dataloader_kwargs)

    def configure_tokenizer(self) -> PreTrainedTokenizerBase:
        access_token = os.environ[self.itdm_cfg.os_env_model_auth_key.upper()] if self.itdm_cfg.os_env_model_auth_key \
              else None
        if self.itdm_cfg.local_fast_tokenizer_path:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(self.itdm_cfg.local_fast_tokenizer_path,
                                                                **self.itdm_cfg.tokenizer_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.itdm_cfg.model_name_or_path, token=access_token, **self.itdm_cfg.tokenizer_kwargs
            )
            _ = tokenizer.add_special_tokens(self.itdm_cfg.special_tokens_dict)
        if self.itdm_cfg.tokenizer_id_overrides:
            for k, v in self.itdm_cfg.tokenizer_id_overrides.items():
                setattr(tokenizer, k, v)
        return tokenizer

    # adapted from HF native trainer
    def _set_signature_columns_if_needed(self, target_model: torch.nn.Module) -> None:
        if len(self.itdm_cfg.signature_columns) == 0:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(target_model.forward)
            self.itdm_cfg.signature_columns = list(signature.parameters.keys())

    # adapted from HF native trainer
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None) -> Dataset:
            if not self.itdm_cfg.remove_unused_columns:
                return dataset
            target_model = self.trainer.model.model if not self.itdm_cfg.defer_model_init else None
            self._set_signature_columns_if_needed(target_model)
            ignored_columns = list(set(dataset.column_names) - set(self.itdm_cfg.signature_columns))
            if len(ignored_columns) > 0:
                dset_description = "" if description is None else f"in the {description} set"
                target_name = f"`{target_model.__class__.__name__}.forward`"
                rank_zero_info(
                    f"The following columns {dset_description} don't have a corresponding argument in {target_name} and"
                    f" have been ignored: {', '.join(ignored_columns)}. If {', '.join(ignored_columns)} are not"
                    f" expected by {target_name}, you can safely ignore this message."
                )
            return dataset.remove_columns(ignored_columns)


class ITDataModule(DataModuleInitMixin, pl.LightningDataModule):
    """A :class:`~pytorch_lighting.core.LightningDataModule` for using either the RTE or BoolQ `SuperGLUE Hugging
    Face datasets <https://huggingface.co/datasets/super_glue#data-instances>`_."""

    tokenization_func: Callable

    def __init__(
        self,
        itdm_cfg: ITDataModuleConfig,
    ):
        r"""Initialize the ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face
        datasets.

        Args:
            itdm_cfg (:class:`~ITDataModuleConfig`): Configuration for the datamodule.
        """
        super().__init__()
        self.itdm_cfg = itdm_cfg
        if self.itdm_cfg.enable_datasets_cache:  # explicitly toggle datasets caching
            datasets.enable_caching()
        else:
            datasets.disable_caching()
        self.save_hyperparameters()
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.hparams.itdm_cfg.tokenizers_parallelism else "false"
        self.tokenizer = self.configure_tokenizer()
        collator_kwargs = self.itdm_cfg.data_collator_cfg.get('collator_kwargs', None) or {}
        collator_class = _import_class(self.itdm_cfg.data_collator_cfg["collator_class"])
        self.data_collator = collator_class(self.tokenizer, **collator_kwargs)

    def prepare_data(self) -> None:
        """Load the SuperGLUE dataset."""
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign
        # state (e.g. self.x=y)
        dataset = datasets.load_dataset("super_glue", self.hparams.itdm_cfg.task_name)
        for split in dataset.keys():
            if split == 'validation' or not self.itdm_cfg.prepare_validation_set_only:
                dataset[split] = dataset[split].map(self.tokenization_func, batched=True)
                dataset[split] = self._remove_unused_columns(dataset[split])
        if self.itdm_cfg.prepare_validation_set_only:
            for split in ['test', 'train']:
                del dataset[split]
        dataset.save_to_disk(self.itdm_cfg.dataset_path)
