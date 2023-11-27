import os
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, Optional, List, Union
from abc import ABC
from functools import reduce
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer

from interpretune.utils.import_utils import _import_class, instantiate_class, _BNB_AVAILABLE
from interpretune.base.config_classes import ITConfig
from interpretune.base.it_datamodule import ITDataModule
from interpretune.utils.debug import DebugGenerationMixin
from interpretune.utils.logging import rank_zero_info, collect_env_info
from interpretune.utils.types import STEP_OUTPUT, OptimizerLRScheduler


class BaseITModule(ABC):

    #TODO: move fts callback property to this class once fts supports raw pytorch with plugin
    # @property
    # def finetuningscheduler_callback(self) -> Optional[fts.FinetuningScheduler]:  # type: ignore
    #     fts_callback = [c for c in self.trainer.callbacks if isinstance(c, fts.FinetuningScheduler)]  # type: ignore
    #     return fts_callback[0] if fts_callback else None

    def __init__(
        self,  # note we explicitly include args shared with our data module
        it_cfg: ITConfig,
    ):
        """In this example, this :class:`~lightning.pytorch.core.module.LightningModule` is initialized by composing
        the ./config/fts_defaults.yaml default configuration with various scheduled fine-tuning yaml configurations
        via the :class:`~lightning.pytorch.cli.LightningCLI` but it can be used like any other
        :class:`~lightning.pytorch.core.module.LightningModule` as well.

        Args:
            it_cfg (ITConfig): Configuration for this
                :class:`~lightning.pytorch.core.module.LightningModule`.
        """
        super().__init__()
        self.it_cfg = it_cfg
        if self.it_cfg.debug_lm_cfg.enabled:  # conditionally load debugging extension
            self.lm_debug = DebugGenerationMixin()
            self.lm_debug.connect_lmdebug(self)
        self._model_init()

    def _model_init(self) -> None:
        self.model = self._auto_model_init()
        self.model.config.update(self.it_cfg.model_cfg)  # apply model config overrides
        self.model.config.use_cache = self.it_cfg.use_model_cache
        self._configure_gradient_checkpointing()
        # TODO: disentagle use of bnb and lora configs in the future, create single peft config perhaps
        if all((_BNB_AVAILABLE, self.it_cfg.bitsandbytesconfig, self.it_cfg.lora_cfg)):
            self._configure_peft()
        self._log_hyperparameters()
        self._load_metric()
        if self.it_cfg.debug_lm_cfg.enabled and self.it_cfg.debug_lm_cfg.record_memory_history:
            torch.cuda.memory._record_memory_history()

    def _load_metric(self) -> None:
        """Optionally load a metric at the end of model initialization."""

    def _log_hyperparameters(self) -> None:
        self.init_hparams = {
            "optimizer_init": self.it_cfg.optimizer_init,
            "lr_scheduler_init": self.it_cfg.lr_scheduler_init,
            "pl_lrs_cfg": self.it_cfg.pl_lrs_cfg,
            "dynamic_module_cfg": self.it_cfg.dynamic_module_cfg,
            "quantization_cfg": self.it_cfg.lora_cfg,
            "auto_model_cfg": self.it_cfg.auto_model_cfg, # TODO: cleanup/consolidate saving configs/dedup
            "model_config": self._make_config_serializable(self.model.config,
                                                        ['quantization_config.bnb_4bit_compute_dtype',
                                                            'torch_dtype', '_pre_quantization_dtype']),
            "model_name_or_path": self.it_cfg.model_name_or_path,
            "task_name": self.it_cfg.task_name,
            "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.it_cfg.experiment_tag}",
            }
        self.init_hparams["env_info"] = collect_env_info() if self.it_cfg.log_env_details else None
        # TODO: add save_hyperparameters/basic logging func for raw pytorch
        # (override w/ lightning version where appropriate)
        #self.save_hyperparameters(self.init_hparams)

    def _configure_gradient_checkpointing(self) -> None:
        if self.it_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _configure_peft(self) -> None:
        if self.it_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, LoraConfig(**self.it_cfg.lora_cfg))

    def _cust_token_cfg(self, model: torch.nn.Module) -> Any:
        if self.it_cfg.tokenizer_id_overrides:
            for k, v in self.it_cfg.tokenizer_id_overrides.items():
                setattr(model.config, k, v)

    def _auto_model_init(self) -> Any:
        quantization_config = self._configure_quantization()
        additional_from_pretrained_kwargs = {"pretrained_model_name_or_path": self.it_cfg.model_name_or_path,
                                             "quantization_config": quantization_config,
                                             "torch_dtype": self.it_cfg.torch_dtype}
        self.it_cfg.from_pretrained_cfg.update(additional_from_pretrained_kwargs)
        access_token = os.environ[self.it_cfg.os_env_model_auth_key.upper()] if self.it_cfg.os_env_model_auth_key \
              else None
        cust_config = self._gen_cust_config(access_token)
        cust_config.num_labels = self.it_cfg.num_labels
        model = self._configured_model_init(cust_config, access_token)
        self._cust_token_cfg(model)
        model = self._maybe_resize_token_embeddings(model)
        return model

    def _maybe_resize_token_embeddings(self, model: torch.nn.Module) -> None:
        vocab_size = getattr(model.base_model, 'vocab_size', None) or model.config.vocab_size
        model.base_model.resize_token_embeddings(vocab_size + len(self.it_cfg.tokenizer_id_overrides))
        return model

    def _configure_quantization(self) -> Optional[Any]:
        if self.it_cfg.bitsandbytesconfig and _BNB_AVAILABLE:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(**self.it_cfg.bitsandbytesconfig)
        else:
            quantization_config = None
        return quantization_config

    def _gen_cust_config(self, access_token: Optional[str] = None) -> PretrainedConfig:
        if self.it_cfg.auto_model_cfg:
            self.it_cfg.model_class = _import_class(self.it_cfg.auto_model_cfg["model_head"])
            cust_config = AutoConfig.from_pretrained(**self.it_cfg.from_pretrained_cfg, token=access_token)
        elif self.it_cfg.dynamic_module_cfg:
            config_class = get_class_from_dynamic_module(self.it_cfg.dynamic_module_cfg['config_class'],
                                                         self.it_cfg.model_name_or_path)
            self.it_cfg.model_class = get_class_from_dynamic_module(self.it_cfg.dynamic_module_cfg['model_class'],
                                                                    self.it_cfg.model_name_or_path)
            cust_config = config_class.from_pretrained(self.it_cfg.model_name_or_path)
        else:
            if self.it_cfg.defer_model_init:
                rank_zero_info("`defer_model_init` not currently supported without `auto_model_cfg` or "
                               "`dynamic_module_cfg`. Proceeding with model init.")
            cust_config = AutoConfig.from_pretrained(**self.it_cfg.from_pretrained_cfg, token=access_token,
                                                     local_files_only=False)
        return cust_config

    def _configured_model_init(self, cust_config: PretrainedConfig, access_token: Optional[str] = None) \
        -> torch.nn.Module:
        head_configured = self.it_cfg.auto_model_cfg or self.it_cfg.dynamic_module_cfg
        if not self.it_cfg.defer_model_init:
            model = self.it_cfg.model_class.from_pretrained(**self.it_cfg.from_pretrained_cfg, config=cust_config,
                                                            token=access_token) \
                if head_configured else \
                    AutoModelForSequenceClassification.from_pretrained(**self.it_cfg.from_pretrained_cfg,
                                                                       config=cust_config, token=access_token)
        else: # defer model materialization (e.g., to `configure_model` hook)
            with torch.device("meta"):
                model = self.it_cfg.model_class(config=cust_config, token=access_token)
        return model

    @staticmethod
    def _make_config_serializable(config_to_clean: Any, target_keys: Union[str, List]) -> Dict:
        serial_cfg = deepcopy(config_to_clean)
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        for k in target_keys:
            fqn_l = k.split(".")
            try:
                setattr(reduce(getattr, fqn_l[:-1], serial_cfg), fqn_l[-1], repr(reduce(getattr, fqn_l, serial_cfg)))
            except AttributeError as ae:
                rank_zero_info("Attempted to clean a key that was not present, continuing without cleaning that key: "
                               f"{ae}")
        return serial_cfg

class ITModule(BaseITModule):
    """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation model
    on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """

    def __init__(
        self,  # note we explicitly include args shared with our data module
        it_cfg: ITConfig,
    ):
        """In this example, this :class:`~lightning.pytorch.core.module.LightningModule` is initialized by composing
        the ./config/fts_defaults.yaml default configuration with various scheduled fine-tuning yaml configurations
        via the :class:`~lightning.pytorch.cli.LightningCLI` but it can be used like any other
        :class:`~lightning.pytorch.core.module.LightningModule` as well.

        Args:
            it_cfg (ITConfig): Configuration for this
                :class:`~lightning.pytorch.core.module.LightningModule`.
        """
        super().__init__()
        self.it_cfg = it_cfg
        if self.it_cfg.debug_lm_cfg.enabled:  # conditionally load debugging extension
            self.lm_debug = DebugGenerationMixin()
            self.lm_debug.connect_lmdebug(self)
        self._model_init()

    def setup(self, stage: str) -> None:
        # TODO: move this setup to relevant task/model-specific modules
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

class ITHookedModule(BaseITModule):
    def __init__(
        self,  # note we explicitly include args shared with our data module
        it_cfg: ITConfig,
    ):
        """In this example, this :class:`~lightning.pytorch.core.module.LightningModule` is initialized by composing
        the ./config/fts_defaults.yaml default configuration with various scheduled fine-tuning yaml configurations
        via the :class:`~lightning.pytorch.cli.LightningCLI` but it can be used like any other
        :class:`~lightning.pytorch.core.module.LightningModule` as well.

        Args:
            it_cfg (ITConfig): Configuration for this
                :class:`~lightning.pytorch.core.module.LightningModule`.
        """
        super().__init__(it_cfg=it_cfg)
        self.datamodule = None
        self.dump_base = None

    def setup(self, datamodule: ITDataModule) -> None:
        self.datamodule = datamodule
        if self.it_cfg.tlens_from_pretrained_cfg.enabled:
            self._convert_hf_to_hooked()
        if self.it_cfg.zero_shot_cfg.enabled:
            tokenizer, zs_cfg = self.datamodule.tokenizer, self.it_cfg.zero_shot_cfg
            zs_cfg.entailment_mapping_indices = torch.tensor(tokenizer.convert_tokens_to_ids(zs_cfg.entailment_mapping))
        self.dump_base = Path("/tmp/")
        self.dump_path = self.dump_base / self.init_hparams['experiment_id']
        self.dump_path.mkdir(exist_ok=True, parents=True)

    def _configured_model_init(self, cust_config: PretrainedConfig, access_token: Optional[str] = None) \
        -> torch.nn.Module:
        # usually makes sense to init the hooketransfomer (empty) and pretrained HF model weights on cpu
        # versus moving them both to GPU (may make sense to explore meta device usage for model definition
        # in the future, only materlizing parameter by parameter during loading from pretrained weights
        # to eliminate need for two copies in memory)
        model = self.it_cfg.model_class.from_pretrained(**self.it_cfg.from_pretrained_cfg, config=cust_config,
                                                        token=access_token)
        # perhaps explore initializing on the meta device and then materializing as needed layer by layer during
        # loading/processing into hookedtransformer
        # with torch.device("meta"):
        #     model = self.it_cfg.model_class(config=cust_config)  # token=access_token)
        return model

    def _convert_hf_to_hooked(self) -> HookedTransformer:
        self.model = HookedTransformer.from_pretrained(hf_model=self.model, tokenizer=self.datamodule.tokenizer,
                                                  **self.it_cfg.tlens_from_pretrained_cfg.__dict__)

    def _log_hyperparameters(self) -> None:
        self.it_cfg.lora_cfg = None
        self.it_cfg.bitsandbytesconfig = None
        # TODO: refactor the captured config here to only add tlens_from_pretrained, other added in superclass
        self.init_hparams = {
            "optimizer_init": self.it_cfg.optimizer_init,
            "lr_scheduler_init": self.it_cfg.lr_scheduler_init,
            "pl_lrs_cfg": self.it_cfg.pl_lrs_cfg,
            "tlens_from_pretrained_cfg": self._make_config_serializable(self.it_cfg.tlens_from_pretrained_cfg,
                                                                        ['device']),
            "dynamic_module_cfg": self.it_cfg.dynamic_module_cfg,
            "quantization_cfg": self.it_cfg.lora_cfg,
            "auto_model_cfg": self.it_cfg.auto_model_cfg, # TODO: cleanup/consolidate saving configs/dedup
            "model_config": self._make_config_serializable(self.model.config,
                                                        ['quantization_config.bnb_4bit_compute_dtype',
                                                            'torch_dtype', '_pre_quantization_dtype']),
            "model_name_or_path": self.it_cfg.model_name_or_path,
            "task_name": self.it_cfg.task_name,
            "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.it_cfg.experiment_tag}",
            }
        self.init_hparams["env_info"] = collect_env_info() if self.it_cfg.log_env_details else None
        # TODO: add save_hyperparameters/basic logging func for raw pytorch
        # (override w/ lightning version where appropriate)
        #self.save_hyperparameters(self.init_hparams)

    def _maybe_resize_token_embeddings(self, model: torch.nn.Module) -> None:
        # embedding resizing not currently supported by ITHookedModule
        return model

    def _configure_gradient_checkpointing(self) -> None:
        # gradient checkpointing not currently supported by ITHookedModule
        pass

    def _configure_peft(self) -> None:
        # peft not currently supported by ITHookedModule
        pass
