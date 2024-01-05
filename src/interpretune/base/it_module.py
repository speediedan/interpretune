import os
import warnings
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, Optional, List, Union
from abc import ABC
from functools import reduce

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer

from interpretune.utils.import_utils import _import_class, instantiate_class, _BNB_AVAILABLE
from interpretune.base.config_classes import ITConfig
from interpretune.base.it_datamodule import ITDataModule
from interpretune.base.it_hooks import (OptimizerSchedulerInitMixin, CoreHelperAttributeMixin,
                                        CORE_TO_LIGHTNING_ATTRS_MAP)
from interpretune.base.debug import DebugGenerationMixin, MemProfilerMixin, ProfilerHooksMixin
from interpretune.utils.logging import rank_zero_info, rank_zero_warn, collect_env_info, rank_zero_debug
from interpretune.utils.types import STEP_OUTPUT, OptimizerLRScheduler

# TODO: add core helper log/log_dict methods for core context usage
for warnf in [".*For Lightning compatibility, this noop .*",]:
    warnings.filterwarnings("once", warnf)


class BaseITModule(ABC, OptimizerSchedulerInitMixin, ProfilerHooksMixin, torch.nn.Module):

    #TODO: move fts callback property to this class once fts supports raw pytorch with plugin
    # @property
    # def finetuningscheduler_callback(self) -> Optional[fts.FinetuningScheduler]:  # type: ignore
    #     fts_callback = [c for c in self.trainer.callbacks if isinstance(c, fts.FinetuningScheduler)]  # type: ignore
    #     return fts_callback[0] if fts_callback else None

    def __init__(
        self,
        it_cfg: ITConfig,
        *args,
        **kwargs
    ):
        """In this example, this :class:`~lightning.pytorch.core.module.LightningModule` is initialized by composing
        the ./config/fts_defaults.yaml default configuration with various scheduled fine-tuning yaml configurations
        via the :class:`~lightning.pytorch.cli.LightningCLI` but it can be used like any other
        :class:`~lightning.pytorch.core.module.LightningModule` as well.

        Args:
            it_cfg (ITConfig): Configuration for this
                :class:`~lightning.pytorch.core.module.LightningModule`.
        """
        # See NOTE [Interpretune Dataclass-Oriented Configuration]
        super().__init__(*args, **kwargs)
        # datamodule handle
        self._datamodule = None
        # root device (sometimes used if not handled by Lightning)
        self._device = None
        self._session_complete = False
        # optional optimizer and lr scheduler handles if initialized via core IT module `configure_optimizers` hook
        self.it_optimizers, self.it_lr_scheduler_configs, self.it_lr_schedulers = None, None, None
        self.it_cfg = self._before_it_cfg_init(it_cfg=it_cfg)
        if self.it_cfg.memprofiler_cfg.enabled:  # conditionally load simple memory profiling extension
            self.memprofiler = MemProfilerMixin()
            self.memprofiler.connect(self)
        if self.it_cfg.debug_lm_cfg.enabled:  # conditionally load debugging extension
            self.lm_debug = DebugGenerationMixin()
            self.lm_debug.connect(self)
        self._model_init()

    def _core_or_lightning(self, c2l_map_key: str):
        c2l = CORE_TO_LIGHTNING_ATTRS_MAP[c2l_map_key]
        try:
            attr_val = getattr(self, c2l_map_key, None) or reduce(getattr, c2l[0].split("."), self)
        except AttributeError as ae:
            rank_zero_debug(f"{c2l[2]}: {ae}")
            attr_val = c2l[1]
        return attr_val

    @property
    def core_log_dir(self) -> Optional[str | os.PathLike]:
        return self._core_or_lightning(c2l_map_key="_log_dir")

    @property
    def datamodule(self) -> Optional[ITDataModule]:
        return self._core_or_lightning(c2l_map_key="_datamodule")

    @property
    def session_complete(self) -> bool:
        return self._session_complete

    @property
    def cuda_allocator_history(self) -> bool:
        return self.it_cfg.memprofiler_cfg.enabled and self.it_cfg.memprofiler_cfg.cuda_allocator_history

    @property
    def torch_dtype(self) -> Optional[Union[torch.dtype, 'str']]:
        try:
            if dtype := getattr(self.it_cfg, "_torch_dtype", None):
                return dtype
            if getattr(self, 'model', None):
                dtype = getattr(self.model, "_torch_dtype", None) or getattr(self.model, "dtype", None)
        except AttributeError:
            dtype = None
        return dtype

    def _hook_output_handler(self, hook_name: str, output: Any) -> None:
        if hook_name == "configure_optimizers":
            self._it_init_optimizers_and_schedulers(output)
        elif hook_name == "on_train_epoch_start":
            pass  # TODO: remove if decided that no need to connect output of this hook
        else:
            rank_zero_warn(f"Output received for hook `{hook_name}` which is not yet supported.")

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        """Optionally modify configuration before it_cfg is initialized."""
        return it_cfg

    def _model_init(self) -> None:
        if self.cuda_allocator_history:
            torch.cuda.memory._record_memory_history()
        self.model = self._auto_model_init()
        self.model.config.update(self.it_cfg.model_cfg)  # apply model config overrides
        self.model.config.use_cache = self.it_cfg.use_model_cache
        self._configure_gradient_checkpointing()
        # TODO: disentagle use of bnb and lora configs in the future, create single peft config perhaps
        if all((_BNB_AVAILABLE, self.it_cfg.bitsandbytesconfig, self.it_cfg.lora_cfg)):
            self._configure_peft()
        self._capture_hyperparameters()
        self._load_metric()

    def _load_metric(self) -> None:
        """Optionally load a metric at the end of model initialization."""

    def _init_dirs_and_hooks(self) -> None:
        self._create_experiment_dir()
        if self.cuda_allocator_history:
            self.memprofiler.init_cuda_snapshots_dir()
        # TODO: add save_hyperparameters/basic logging func for raw pytorch
        # (override w/ lightning version where appropriate)
        #self.save_hyperparameters(self.init_hparams)

    def _create_experiment_dir(self) -> None:
        # we only want to create the core experiment-specific dir for non-lightning modules
        if getattr(self, '_log_dir', None):
            self._log_dir = self._log_dir / self.init_hparams['experiment_id']
            self._log_dir.mkdir(exist_ok=True, parents=True)

    def _capture_hyperparameters(self) -> None:
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

    def _configure_gradient_checkpointing(self) -> None:
        if self.it_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _configure_peft(self) -> None:
        if self.it_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, LoraConfig(**self.it_cfg.lora_cfg))

    def _set_input_require_grads(self) -> None:
        if self.it_cfg.enable_input_require_grads:
            self.model.enable_input_require_grads()

    def _cust_token_cfg(self, model: torch.nn.Module) -> Any:
        if self.it_cfg.tokenizer_id_overrides:
            for k, v in self.it_cfg.tokenizer_id_overrides.items():
                setattr(model.config, k, v)

    def _auto_model_init(self) -> Any:
        quantization_config = self._configure_quantization()
        additional_from_pretrained_kwargs = {"pretrained_model_name_or_path": self.it_cfg.model_name_or_path,
                                             "quantization_config": quantization_config,
                                             #"torch_dtype": self.it_cfg.torch_dtype,
                                             "torch_dtype": self.torch_dtype,
                                             }
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
        if self.it_cfg.from_pretrained_cfg.get('return_unused_kwargs', False):
            return cust_config[0]
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

    def setup(self, *args: Any, **kwargs: Any) -> None:
        self._init_dirs_and_hooks()

    def configure_optimizers(self) -> Optional[OptimizerLRScheduler]:
        """Optional because it is not mandatory in the context of core IT modules (required for Lightning
        modules)."""
        # With FTS >= 2.0, ``FinetuningScheduler`` simplifies initial optimizer configuration by ensuring the optimizer
        # configured here will optimize the parameters (and only those parameters) scheduled to be optimized in phase 0
        # of the current fine-tuning schedule. This auto-configuration can be disabled if desired by setting
        # ``enforce_phase0_params`` to ``False``.
        self._set_input_require_grads()
        optimizer, scheduler = None, None
        if self.it_cfg.optimizer_init:  # in case this hook is manually invoked by the user
            optimizer = instantiate_class(args=self.model.parameters(), init=self.it_cfg.optimizer_init)
        if self.it_cfg.lr_scheduler_init:
            scheduler = {
                "scheduler": instantiate_class(args=optimizer, init=self.it_cfg.lr_scheduler_init),
                **self.it_cfg.pl_lrs_cfg,
            }
        return [optimizer], [scheduler]

    def on_session_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if getattr(self, 'memprofiler', None):
            self.memprofiler.dump_memory_stats()
        self._session_complete = True

    # N.B. we call `on_session_end` at the end of train, test and predict session types only. This is because Lightning
    # calls both `on_train_end` and `on_validation_end` with most training sessions when running both a fit and
    # evaluation loop as is usually the case) but only `on_test_end` with the test stage.
    # The `on_run_end` hook for Lightning is roughly analogous to Interpretune's `on_session_end` hook but is not
    # a hook Lightning makes available to users.
    def on_train_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def on_validation_end(self) -> Optional[Any]:
        pass

    def on_test_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def on_predict_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if not self.session_complete:
            self.on_session_end()

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        pass


class ITModule(CoreHelperAttributeMixin, BaseITModule):
    """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation model
    on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """
    # def __init__(self, it_cfg: ITConfig, *args: Any, **kwargs: Any) -> None:
    #     # for core/non-lightning modules, we configure a _log_dir rather than relying on the trainer to do so
    #     self._log_dir = Path(it_cfg['log_dir'] or tempfile.gettempdir())
    #     super().__init__(it_cfg, *args, **kwargs)

    def setup(self, datamodule: ITDataModule, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._datamodule = datamodule

    def forward(self, **inputs: Any) -> STEP_OUTPUT:
        return self.model(**inputs, **self.it_cfg.cust_fwd_kwargs)

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
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    @ProfilerHooksMixin.memprofilable
    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.it_cfg.zero_shot_cfg.enabled:
            self.zero_shot_test_step(batch, batch_idx)
        else:
            self.default_test_step(batch, batch_idx)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      pad_token_id=self.datamodule.tokenizer.pad_token_id,
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
        #self.log("predict_loss", test_loss, prog_bar=True, sync_dist=True)
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


class BaseITHookedModule(BaseITModule):
    # def __init__(
    #     self,
    #     it_cfg: ITConfig,
    # ):
    #     """In this example, this :class:`~lightning.pytorch.core.module.LightningModule` is initialized by composing
    #     the ./config/fts_defaults.yaml default configuration with various scheduled fine-tuning yaml configurations
    #     via the :class:`~lightning.pytorch.cli.LightningCLI` but it can be used like any other
    #     :class:`~lightning.pytorch.core.module.LightningModule` as well.

    #     Args:
    #         it_cfg (ITConfig): Configuration for this
    #             :class:`~lightning.pytorch.core.module.LightningModule`.
    #     """
    #     # See NOTE [Interpretune Dataclass-Oriented Configuration]
    #     super().__init__(it_cfg=it_cfg)
    #     #self.dump_base = None

    def setup(self, datamodule: ITDataModule) -> None:
        self._datamodule = datamodule
        if self.it_cfg.tlens_from_pretrained_cfg.enabled:
            self._convert_hf_to_hooked()
        # self.dump_base = Path("/tmp/")
        # self.dump_path = self.dump_base / self.init_hparams['experiment_id']
        # self.dump_path.mkdir(exist_ok=True, parents=True)

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

    def _set_input_require_grads(self) -> None:
        # not currently supported by ITHookedModule
        rank_zero_warn("Setting input require grads not currently supported by ITHookedModule.")

    def _configure_gradient_checkpointing(self) -> None:
        # gradient checkpointing not currently supported by ITHookedModule
        pass

    def _configure_peft(self) -> None:
        # peft not currently supported by ITHookedModule
        pass

class ITHookedModule(CoreHelperAttributeMixin, BaseITHookedModule):
    ...
