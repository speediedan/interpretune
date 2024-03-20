import os
import inspect
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
from functools import wraps

import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.analysis.debug_generation import DebugGeneration
from interpretune.analysis.memprofiler import MemProfiler
from interpretune.utils.types import  STEP_OUTPUT
from interpretune.utils.logging import rank_zero_warn, rank_zero_info
from interpretune.base.config.mixins import ITExtension
from interpretune.base.config.module import ITConfig, HFFromPretrainedConfig, ITState
from interpretune.utils.import_utils import _import_class, _BNB_AVAILABLE
from interpretune.utils.exceptions import MisconfigurationException


DEFINED_EXTENSIONS = (ITExtension("debug_lm", DebugGeneration), ITExtension("memprofiler", MemProfiler))


class ITStateMixin:
    def __init__(self, *args, **kwargs) -> None:
        # TODO: explore whether there is an initialization reorganization that can avoid this
        # some class compositions may need to initialize internal state before this __init__ is invoked, hence we also
        # make it available as a staticmethod
        ITStateMixin._init_internal_state(self)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _init_internal_state(obj: Any) -> None:
        if not obj.__dict__.get('_it_state', None):
            obj._it_state = ITState()


class ITExtensionsMixin:

    SUPPORTED_EXTENSIONS: Dict[str, ITExtension] = {}
    DEFAULT_EXTENSIONS = (ITExtension("debug_lm", "interpretune.analysis.debug_generation.DebugGeneration"),
                          ITExtension("memprofiler", "interpretune.analysis.memprofiler.MemProfiler"))

    def _detect_extensions(self) -> Dict[str, ITExtension]:
        # TODO: update custom extensions to be read/added from it_cfg once interface stabilizes
        for it_ext in self.DEFAULT_EXTENSIONS:
            try:
                ext_class = _import_class(it_ext.ext_fqn)
            except (ImportError, AttributeError) as e:
                err_msg = f"Unable to import and resolve specified extension from {it_ext.ext_fqn}: {e}"
                raise MisconfigurationException(err_msg)
            self.SUPPORTED_EXTENSIONS[it_ext.ext_attr] = ext_class

    def _connect_extensions(self):
        self._detect_extensions()
        for ext_name, ext_class in self.SUPPORTED_EXTENSIONS.items():
            if getattr(self.it_cfg, f'{ext_name}_cfg').enabled:
                self._it_state._extensions[ext_name] = ext_class()
                getattr(self, ext_name).connect(self)
            else:
                self._it_state._extensions[ext_name] = None

    def __getattr__(self, name: str) -> Any:
        # we make extension handles available as direct root module attributes for convenience
        # filter only the supported extension attributes, ensuring `_it_state`` has been initialized
        if self.SUPPORTED_EXTENSIONS.get(name) and (ext_attrs := self.__dict__.get('_it_state', None)) is not None:
            return ext_attrs._extensions[name]
        return super().__getattr__(name)


class ProfilerHooksMixin:

    @contextmanager
    @staticmethod
    def memprofile_ctx(memprofiler, phase: str, epoch_idx: Optional[int] = None, step_idx: Optional[int] = None):
        try:
            memprofiler.snap(phase=phase, epoch_idx=epoch_idx, step_idx=step_idx, step_ctx="start")
            yield
        finally:
            memprofiler.snap(phase=phase, epoch_idx=epoch_idx, step_idx=step_idx, step_ctx="end", reset_mem_hooks=True)

    @staticmethod
    def memprofilable(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.memprofiler:
                return func(self, *args, **kwargs)
            phase = func.__name__
            # for increased generality, we derive a profile `step_idx` based on a profiler snap counter rather than
            # parsing `args` if a `batch_idx` kwarg isn't found
            step_idx = kwargs.get("batch_idx", None)
            with ProfilerHooksMixin.memprofile_ctx(self.memprofiler, phase=phase, step_idx=step_idx):
                if self.memprofiler.memprofiler_cfg.enable_saved_tensors_hooks and \
                    self.memprofiler._enabled[(phase, 'start')]:
                    with torch.autograd.graph.saved_tensors_hooks(*self.memprofiler._saved_tensors_funcs):
                        return func(self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)
        return wrapper

class ZeroShotStepMixin:

    _gen_sig_keys: Optional[List] = None

    @property
    def gen_sig_keys(self) -> List:
        if not self._gen_sig_keys:
            generate_signature = inspect.signature(self.model.generate)
            self._gen_sig_keys = list(generate_signature.parameters.keys())
        return self._gen_sig_keys

    def map_gen_inputs(self, batch) -> Dict[str, Any]:
        # since we're abstracting the same zero shot classification logic to be used with different frameworks, models
        # and datasets we use a mapping function to provide only data inputs a given generate function supports (for
        # frameworks that don't handle variadic kwargs). This currently requires the user provides
        # compatible models and dataset.
        # TODO: consider adding further upstream configuration validation that warns the user if the provided dataset
        # and step logic are incompatible
        return {bk: batch[bk] for bk in list(batch.data) if bk in self.gen_sig_keys}

    def map_gen_kwargs(self, kwargs: Dict) -> Dict[str, Any]:
        # we use a mapping function to provide only generate kwargs a given generate function supports (for
        # frameworks that don't support variadic kwargs).
        return {k: v for k, v in kwargs.items() if k in self.gen_sig_keys}

    def it_generate(self, batch: BatchEncoding, **kwargs) -> Any:
        try:
            if 'kwargs' in self.gen_sig_keys:
                outputs = self.model.generate(**batch, **kwargs)
            else:
                outputs = self.model.generate(**self.map_gen_inputs(batch), **self.map_gen_kwargs(kwargs))
        except AttributeError as ae:
            gen_dataset_info_msg = (
                f"The following keys were found in the provided data batch: {os.sep} {list(batch.data)}). The current"
                f" generate method ({self.model.generate.__func__}) accepts: {os.sep} {[self._gen_sig_keys]}."
            )
            rank_zero_info(gen_dataset_info_msg)
            raise AttributeError(f"Received the following error msg: {ae}")
        return outputs

    @ProfilerHooksMixin.memprofilable
    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.it_cfg.zero_shot_cfg.enabled:
            self.zero_shot_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        else:
            self.default_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        rank_zero_warn("`zero_shot_test_step` must be implemented to be used with the Interpretune.")

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        rank_zero_warn("`default_test_step` must be implemented to be used with the Interpretune.")


class HFFromPretrainedMixin:
    """" Barebones interface to setup optimizers and schedulers for manual optimization with core IT modules."""

    # proper initialization of these variables should be done in the child class
    it_cfg: ITConfig
    model: torch.nn.Module


    @property
    def hf_cfg(self) -> Optional[HFFromPretrainedConfig]:
        return self.it_cfg.hf_from_pretrained_cfg

    def hf_pretrained_model_init(self) -> None:
        access_token = os.environ[self.it_cfg.os_env_model_auth_key.upper()] if self.it_cfg.os_env_model_auth_key \
            else None
        quantization_config = self._hf_configure_quantization()
        self._update_hf_pretrained_cfg(quantization_config)
        cust_config = self._hf_gen_cust_config(access_token)
        self.model = self.hf_configured_model_init(cust_config, access_token)
        self._hf_cust_token_cfg()
        self._hf_maybe_resize_token_embeddings()
        self._hf_post_init_cfg()

    # TODO: move this and other hooks that may be overridden for non-HF contexts into a separate class
    def set_input_require_grads(self) -> None:
        if self.hf_cfg and self.hf_cfg.enable_input_require_grads:
            self.model.enable_input_require_grads()

    def _hf_configure_quantization(self) -> Optional[Any]:
        if self.hf_cfg.bitsandbytesconfig and _BNB_AVAILABLE:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(**self.hf_cfg.bitsandbytesconfig)
        else:
            quantization_config = None
        return quantization_config

    def _update_hf_pretrained_cfg(self, quantization_config: Optional[Dict[str, Any]]) -> None:
        additional_from_pretrained_kwargs = {"pretrained_model_name_or_path": self.it_cfg.model_name_or_path,
                                            "quantization_config": quantization_config,
                                            #"torch_dtype": self.it_cfg.torch_dtype,
                                            "torch_dtype": self.torch_dtype,
                                            }
        self.hf_cfg.pretrained_kwargs.update(additional_from_pretrained_kwargs)

    def _hf_gen_cust_config(self, access_token: Optional[str] = None) -> PretrainedConfig:
        if self.hf_cfg.model_head:
            self.it_cfg.model_class = _import_class(self.hf_cfg.model_head)
            cust_config = AutoConfig.from_pretrained(**self.hf_cfg.pretrained_kwargs, token=access_token)
        elif self.hf_cfg.dynamic_module_cfg:
            config_class = get_class_from_dynamic_module(self.hf_cfg.dynamic_module_cfg['config_class'],
                                                         self.it_cfg.model_name_or_path)
            self.it_cfg.model_class = get_class_from_dynamic_module(self.hf_cfg.dynamic_module_cfg['model_class'],
                                                                    self.it_cfg.model_name_or_path)
            cust_config = config_class.from_pretrained(self.it_cfg.model_name_or_path)
        else:
            if self.it_cfg.defer_model_init:
                rank_zero_info("`defer_model_init` not currently supported without `model_head` or "
                               "`dynamic_module_cfg`. Proceeding with model init.")
            cust_config = AutoConfig.from_pretrained(**self.hf_cfg.pretrained_kwargs, token=access_token,
                                                     local_files_only=False)
        if self.hf_cfg.pretrained_kwargs.get('return_unused_kwargs', False):
            return cust_config[0]
        return cust_config

    def hf_configured_model_init(self, cust_config: PretrainedConfig, access_token: Optional[str] = None) \
        -> torch.nn.Module:
        cust_config.num_labels = self.it_cfg.num_labels
        head_configured = self.hf_cfg.model_head or self.hf_cfg.dynamic_module_cfg
        # TODO: parameterize the default model head when one not provided in pretrained config
        if not self.it_cfg.defer_model_init:
            if not head_configured:
                self.it_cfg.model_class = _import_class(self.it_cfg.hf_from_pretrained_cfg.default_head)
            model = self.it_cfg.model_class.from_pretrained(**self.hf_cfg.pretrained_kwargs, config=cust_config,
                                                            token=access_token)
        else: # defer model materialization (e.g., to `configure_model` hook)
            with torch.device("meta"):
                model = self.it_cfg.model_class(config=cust_config, token=access_token)
        return model

    def _hf_cust_token_cfg(self) -> None:
        if self.it_cfg.tokenizer_id_overrides:
            for k, v in self.it_cfg.tokenizer_id_overrides.items():
                setattr(self.model.config, k, v)

    def _hf_maybe_resize_token_embeddings(self) -> None:
        vocab_size = getattr(self.model.base_model, 'vocab_size', None) or self.model.config.vocab_size
        max_override = max(self.it_cfg.tokenizer_id_overrides.values())
        if max_override >= vocab_size:
            self.model.base_model.resize_token_embeddings(max_override)

    def _hf_post_init_cfg(self) -> None:
        self.model.config.update(self.it_cfg.model_cfg)  # apply post-init model config overrides
        self.model.config.use_cache = self.hf_cfg.use_model_cache
        self._configure_gradient_checkpointing()
        # TODO: disentagle use of bnb and lora configs in the future, create single peft config perhaps
        if all((_BNB_AVAILABLE, self.hf_cfg.bitsandbytesconfig, self.hf_cfg.lora_cfg)):
            self._configure_peft()

    def _configure_gradient_checkpointing(self) -> None:
        if self.hf_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()

    def _configure_peft(self) -> None:
        if self.hf_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, LoraConfig(**self.hf_cfg.lora_cfg))


class CoreMixins(ITStateMixin, ITExtensionsMixin, HFFromPretrainedMixin, ZeroShotStepMixin, ProfilerHooksMixin):
    ...
