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

from interpretune.config.module import ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.utils.import_utils import _import_class, _BNB_AVAILABLE, _LIGHTNING_AVAILABLE
from interpretune.base.hooks import BaseITHooks
from interpretune.base.mixins.core import (OptimizerSchedulerInitMixin, CoreHelperAttributeMixin, ProfilerHooksMixin,
                                           CORE_TO_LIGHTNING_ATTRS_MAP)
from interpretune.base.mixins.zero_shot_classification import ZeroShotStepMixin
from interpretune.analysis.debug_generation import DebugGeneration
from interpretune.analysis.memprofiler import MemProfiler
from interpretune.utils.logging import rank_zero_info, rank_zero_warn, collect_env_info, rank_zero_debug


# TODO: add core helper log/log_dict methods for core context usage
for warnf in [".*For Lightning compatibility, this noop .*",]:
    warnings.filterwarnings("once", warnf)


class BaseITModule(ABC, BaseITHooks, OptimizerSchedulerInitMixin, ProfilerHooksMixin, ZeroShotStepMixin,
                   torch.nn.Module):

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
            self.memprofiler = MemProfiler()
            self.memprofiler.connect(self)
        if self.it_cfg.debug_lm_cfg.enabled:  # conditionally load debugging extension
            self.lm_debug = DebugGeneration()
            self.lm_debug.connect(self)
        self.init_hparams = {}
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
        # subclasses may have provided their own hparams so we update rather than override
        self.init_hparams.update({
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
            })
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
        max_override = max(self.it_cfg.tokenizer_id_overrides.values())
        if max_override >= vocab_size:
            model.base_model.resize_token_embeddings(max_override)
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

    def on_session_end(self) -> Optional[Any]:
        """Optionally execute some post-interpretune session (train, test, iterative exploration) steps."""
        if getattr(self, 'memprofiler', None):
            self.memprofiler.dump_memory_stats()
        self._session_complete = True


class ITModule(CoreHelperAttributeMixin, BaseITModule):
    """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation model
    on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """
    ...


if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import LightningModule

    class ITLightningModule(BaseITModule, LightningModule):
        """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation
        model on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
        implementations of a given model and the `SuperGLUE Hugging Face dataset.

        <https://huggingface.co/datasets/super_glue#data-instances>`_.
        """
        def on_train_start(self) -> None:
            # ensure model is in training mode (e.g. needed for some edge cases w/ skipped sanity checking)
            self.model.train()
            return super().on_train_start()
else:
    ITLightningModule = object
