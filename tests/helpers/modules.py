import os
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import torch
import datasets
import evaluate
from torch.testing import assert_close
from torch.utils.data import DataLoader
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.base.config_classes import ITDataModuleConfig, ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import ITModule, ITLightningModule
from interpretune.utils.logging import rank_zero_only, get_filesystem
from it_examples.experiments.rte_boolq.core import RTEBoolqModuleMixin
from it_examples.data.rte_bool import RTEBoolqDataModule

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import LightningDataModule


class TestITDataModule(ITDataModule):

    def __init__(self, itdm_cfg: ITDataModuleConfig, force_prepare_data: bool = False) -> None:
        super().__init__(itdm_cfg=itdm_cfg)
        self.force_prepare_data = force_prepare_data
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
        features["labels"] = example_batch["label"]
        return features

    def sample_dataset_state(self) -> Tuple:
        # note that this only validates the loaded dataset/tokenizer, the dataloaders are not tested in this method
        # so one may still need to inspect downstream variables (e.g. the dataloader kwargs) and the batch actually
        # passed to the model in a given test/step to verify that the tested model inputs align with the expected
        # deterministic dataset state defined in `tests.helpers.cfg_aliases.test_dataset_state`
        sample_state = []
        for split in self.dataset.keys():
            # as a content heuristic, inspect the id of a given index (3) for the first 5 rows of each dataset split
            sample_state.extend([t[3] for t in self.dataset[split]['input_ids'][0:5]])
        return (self.tokenizer.__class__.__name__, self.itdm_cfg.task_name, sample_state)

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        dataset_path = Path(self.itdm_cfg.dataset_path)
        # rebuild the test dataset if it does not exist in the test environment
        if not dataset_path.exists() or self.force_prepare_data:
            # regen the 'pytest_rte' subset of rte for testing
            dataset = datasets.load_dataset("super_glue", 'rte')
            for split in dataset.keys():
                dataset[split] = dataset[split].select(range(10))
                dataset[split] = dataset[split].map(self.tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
                dataset[split] = self._remove_unused_columns(dataset[split])
            dataset.save_to_disk(dataset_path)

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


class TestITDataModuleFullDataset(RTEBoolqDataModule, TestITDataModule):
    def __init__(self, itdm_cfg: ITDataModuleConfig, force_prepare_data: bool = False) -> None:
        itdm_cfg.task_name = 'rte'
        TestITDataModule.__init__(self, itdm_cfg=itdm_cfg, force_prepare_data=force_prepare_data)

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        """Load the SuperGLUE dataset."""
        # Note that the current config for the full dataset uses default batching (1000), so the len of the
        # `input_ids` will vary a few percent (e.g. between 258 and 304) contributing to differences in certain
        # profiling metrics. Note that ensuring the use of deterministic algorithims via the `make_deterministic`
        # method is much more important to reducing this variance than the batching approach (though constraining to
        # deterministic algorithms will almost always increase resource requirements on some dimension)
        self.itdm_cfg.dataset_path = Path(self.itdm_cfg.dataset_path).parent / 'rte'
        if not self.itdm_cfg.dataset_path.exists() or self.force_prepare_data:
            RTEBoolqDataModule.prepare_data(self, target_model=target_model)


class BaseTestModule:
    def __init__(self, it_cfg: ITConfig, expected_exact: Optional[Dict] = None, expected_close: Optional[Dict] = None,
                expected_memstats: Optional[Dict] = None, tolerance_map: Optional[Dict] = None,
                test_alias: Optional[str] = None, state_log_dir: Optional[str] = None, *args, **kwargs) -> None:
        super().__init__(it_cfg=it_cfg)
        self.expected_memstats = expected_memstats
        self.expected_exact = expected_exact
        self.expected_close = expected_close
        self.state_log_dir = state_log_dir
        self.test_alias = test_alias
        self.tolerance_map = tolerance_map or {}
        self.epoch_losses = {}
        self.dev_expected_exact = {}
        self.dev_expected_close = {}

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        it_cfg.num_labels = 0 if it_cfg.zero_shot_cfg.enabled else 2
        return it_cfg

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", 'rte', experiment_id=self.init_hparams['experiment_id'])

    def _init_entailment_mapping(self) -> None:
        if self.it_cfg.zero_shot_cfg.enabled:
            tokenizer, zs_cfg = self.datamodule.tokenizer, self.it_cfg.zero_shot_cfg
            zs_cfg.entailment_mapping_indices = torch.tensor(tokenizer.convert_tokens_to_ids(zs_cfg.entailment_mapping))

    # def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
    #     super().test_step(batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def _epoch_end_validation(self, *args, **kwargs) -> None:
        state_key = self.current_epoch
        current_exact = {'device_type': self.model.device.type, 'precision': self.model.dtype,
                         'dataset_state': self.datamodule.sample_dataset_state()}
        current_close = {}
        if self.epoch_losses and self.epoch_losses.get(state_key, None):
            current_close.update({'loss': self.epoch_losses[state_key]})
        self.inspect_or_assert(current_exact, current_close, state_key)

    def on_test_epoch_end(self, *args, **kwargs):
        self._epoch_end_validation(*args, **kwargs)

    def on_train_epoch_start(self, *args, **kwargs):
        pass  # TODO: planning to add some on epoch start validation

    def on_train_epoch_end(self, *args, **kwargs):
        self._epoch_end_validation(*args, **kwargs)

    def on_session_end(self) -> Optional[Any]:
        super().on_session_end()
        if self.it_cfg.memprofiler_cfg and self.expected_memstats:
            self._validate_memory_stats()
        if self.state_log_dir:
            self.log_dev_state()

    def inspect_or_assert(self, current_exact, current_close, state_key) -> None:
        if not self.state_log_dir:
            if self.expected_exact and self.expected_exact.get(state_key, None):
                for exp_k, exp_v in self.expected_exact[state_key].items():
                    assert current_exact[exp_k] == exp_v
            if self.expected_close and self.expected_close.get(state_key, None):
                for exp_k, exp_v in self.expected_close[state_key].items():
                    rtol, atol = self.tolerance_map.get(exp_k, (0, 0))
                    assert_close(actual=current_close[exp_k], expected=exp_v, rtol=rtol, atol=atol)
        else:
            self.dev_expected_exact[state_key] = current_exact
            self.dev_expected_close[state_key] = current_close

    @rank_zero_only
    def log_dev_state(self) -> None:
        dump_path = Path(self.state_log_dir)
        state_log = dump_path / "dev_state_log.yaml"
        fs = get_filesystem(state_log)
        with fs.open(state_log, "w", newline="") as fp:
            fp.write(f"State log for test `{self.test_alias}`:{os.linesep}")
            for dev_d in [self.dev_expected_exact, self.dev_expected_close]:
                fp.write(os.linesep)
                for k, v in dev_d.items():  # control formatting precisely to allow copy/paste expected output
                    fp.write(f"{' ' * 8}{k}: {v},{os.linesep}")

    def _validate_memory_stats(self) -> None:
        for act, exp in zip(self.expected_memstats[1], self.expected_memstats[2]):
            if not self.state_log_dir:
                rtol, atol = self.tolerance_map.get(act, (0, 0))
                assert_close(actual=self.memprofiler.memory_stats[self.expected_memstats[0]][act], expected=exp,
                             rtol=rtol, atol=atol)
            else:
                self.dev_expected_close[act] = self.memprofiler.memory_stats[self.expected_memstats[0]][act]


class TestITModule(BaseTestModule, RTEBoolqModuleMixin, ITModule):
    ...


if _LIGHTNING_AVAILABLE:
    class TestITLightningDataModule(TestITDataModule, LightningDataModule):
        ...
    class TestITLightningDataModuleFullDataset(TestITDataModuleFullDataset, LightningDataModule):
        ...
    class TestITLightningModule(BaseTestModule, RTEBoolqModuleMixin, ITLightningModule):
        def on_train_epoch_end(self, *args, **kwargs):
            self.epoch_losses[self.current_epoch] = self.trainer.callback_metrics['train_loss'].item()
            super().on_train_epoch_end(*args, **kwargs)
else:
    TestITLightningDataModule = object
    TestITLightningDataModuleFullDataset = object
    TestITLightningModule = object
