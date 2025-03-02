from __future__ import annotations
import os
import inspect
from typing import Any, TYPE_CHECKING
from contextlib import contextmanager
from functools import wraps

import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.tokenization_utils_base import BatchEncoding

import interpretune as it
from interpretune.utils import rank_zero_warn, _import_class, _BNB_AVAILABLE
from interpretune.config import (HFFromPretrainedConfig, HFGenerationConfig, BaseGenerationConfig, ITConfig, ITState,
                                 ITExtensionsConfigMixin)

if TYPE_CHECKING:
    from interpretune.protocol import AnalysisCfgProtocol


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


class ProfilerHooksMixin:

    @contextmanager
    @staticmethod
    def memprofile_ctx(memprofiler, phase: str, epoch_idx: int | None = None, step_idx: int | None = None):
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



class AnalysisStepMixin:
    @property
    def analysis_cfg(self) -> AnalysisCfgProtocol:
        if not hasattr(self.it_cfg, 'analysis_cfg') or self.it_cfg.analysis_cfg is None:
            raise AttributeError("Analysis configuration has not been set.")
        return self.it_cfg.analysis_cfg

    @analysis_cfg.setter
    def analysis_cfg(self, cfg: AnalysisCfgProtocol) -> None:
        self.it_cfg.analysis_cfg = cfg

    def on_analysis_start(self) -> Any | None:
        """Optionally execute some post-interpretune session steps if the session is not complete."""
        if self.analysis_cfg.op == it.logit_diffs_attr_grad:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)


    def on_analysis_epoch_end(self) -> Any | None:
        pass
        # TODO: maybe reintroduce logic here if we decide to keep per-epoch versions or perform other caching
        # Create a shallow copy from the current analysis cache
        #cache_copy = self.analysis_cfg.analysis_store
        #self._analysis_stores.append(cache_copy)
        # TODO: we don't want to reset the analysis store but rather ensure we flush the current epoch
        #self.analysis_cfg.reset_analysis_store()  # Prepare a new instance for the next epoch, preserving save_cfg

    def on_analysis_end(self) -> Any | None:
        """Optionally execute some post-interpretune session steps if the session is not complete."""
        # reset internal cache list (TODO: maybe keep this around and reset only on session start?)
        # TODO: we can avoid this analysis_stores reset if we make dataset per-epoch subsplits
        # self._analysis_stores = []  # uncomment if we re-enable the reset of the analysis stores
        if self.analysis_cfg.op == it.logit_diffs_attr_grad:
            torch.set_grad_enabled(False)
        if not self.session_complete:
            self.on_session_end()

class GenerativeStepMixin:
    # Often used for n-shot classification, those contexts are only a subset of generative classification use cases

    _gen_sig_keys: list | None = None
    GEN_PREPARES_INPUTS_SIGS: tuple = ("_prepare_model_inputs",)

    @property
    def generation_cfg(self) -> BaseGenerationConfig | None:
        return self.it_cfg.generative_step_cfg.lm_generation_cfg

    @property
    def gen_sig_keys(self) -> list:
        if not self._gen_sig_keys:
            generate_signature = inspect.signature(self.model.generate)
            self._gen_sig_keys = list(generate_signature.parameters.keys())
        return self._gen_sig_keys

    def map_gen_inputs(self, batch) -> dict[str, Any]:
        # since we're abstracting the same generative classification logic to be used with different frameworks, models
        # and datasets we use a mapping function to provide only data inputs a given generate function supports (for
        # frameworks that don't handle variadic kwargs). This currently requires the user provides
        # compatible models and dataset.
        # TODO: consider adding further upstream configuration validation that warns the user if the provided dataset
        # and step logic are incompatible
        return {bk: batch[bk] for bk in list(batch.data) if bk in self.gen_sig_keys}

    def map_gen_kwargs(self, kwargs: dict) -> dict[str, Any]:
        # we use a mapping function to provide only generate kwargs a given generate function supports (for
        # frameworks that don't support variadic kwargs).
        return {k: v for k, v in kwargs.items() if k in self.gen_sig_keys}

    # TODO: move this property to _it_state?
    @property
    def _should_inspect_inputs(self) -> bool:
        # whether to filter inputs to include only those directly supported by the model's generate function
        return self.it_cfg.generative_step_cfg.input_inspection_enabled and not self._generate_prepares_inputs()

    def _generate_prepares_inputs(self) -> bool:
        # match sentinal methods indicating that a given model's generate function prepares inputs
        return any(hasattr(self.model, prep_method) for prep_method in self.GEN_PREPARES_INPUTS_SIGS)

    def it_generate(self, batch: BatchEncoding | torch.Tensor, **kwargs) -> Any:
        try:
            # variadic kwargs not supported by generate so inspect kwargs and use only those supported
            if 'kwargs' not in self.gen_sig_keys:
                kwargs = self.map_gen_kwargs(kwargs)
            if isinstance(batch, torch.Tensor):
                outputs = self.model.generate(batch, **kwargs)
            else:
                if self._should_inspect_inputs:
                    batch = self.map_gen_inputs(batch)
                outputs = self.model.generate(**batch, **kwargs)
        except (TypeError, AttributeError, ValueError) as ge:
            # TODO: consider further inspecting the possible generation errors encountered here to help narrow the
            # problem space for the user
            gen_dataset_info_msg = (
                f"The following keys were found in the provided data batch: {os.sep} {list(batch.data)}). The current"
                f" generate method ({self.model.generate}) accepts: {os.sep} {[self._gen_sig_keys]}."
            )
            rank_zero_warn(gen_dataset_info_msg)
            raise Exception(f"{gen_dataset_info_msg} Received the following error msg: {ge}")
        return outputs



class HFFromPretrainedMixin:
    """" Barebones interface to setup optimizers and schedulers for manual optimization with core IT modules."""

    # proper initialization of these variables should be done in the child class
    it_cfg: ITConfig
    model: torch.nn.Module

    @property
    def hf_cfg(self) -> HFFromPretrainedConfig | None:
        return self.it_cfg.hf_from_pretrained_cfg

    def hf_pretrained_model_init(self) -> None:
        access_token = os.environ[self.it_cfg.os_env_model_auth_key.upper()] if self.it_cfg.os_env_model_auth_key \
            else None
        quantization_config = self._hf_configure_quantization()
        self._update_hf_pretrained_cfg(quantization_config)
        cust_config, _ = self._hf_gen_cust_config(access_token)
        self.model = self.hf_configured_model_init(cust_config, access_token)
        self._hf_cust_token_cfg()
        self._hf_maybe_resize_token_embeddings()
        self._hf_post_init_cfg()

    # TODO: move this and other hooks that may be overridden for non-HF contexts into a separate class
    def set_input_require_grads(self) -> None:
        if self.hf_cfg and self.hf_cfg.enable_input_require_grads:
            self.model.enable_input_require_grads()

    def _hf_configure_quantization(self) -> Any | None:
        if self.hf_cfg.bitsandbytesconfig and _BNB_AVAILABLE:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(**self.hf_cfg.bitsandbytesconfig)
        else:
            quantization_config = None
        return quantization_config

    def _update_hf_pretrained_cfg(self, quantization_config: dict[str, Any] | None = None) -> None:
        additional_from_pretrained_kwargs = {"pretrained_model_name_or_path": self.it_cfg.model_name_or_path,
                                            "quantization_config": quantization_config,
                                            "torch_dtype": self.torch_dtype,
                                            }
        self.hf_cfg.pretrained_kwargs.update(additional_from_pretrained_kwargs)

    def _hf_gen_cust_config(self, access_token: str | None = None) -> tuple[PretrainedConfig, dict]:
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
                rank_zero_warn("`defer_model_init` not currently supported without `model_head` or "
                               "`dynamic_module_cfg`. Proceeding with model init.")
            cust_config = AutoConfig.from_pretrained(**self.hf_cfg.pretrained_kwargs, token=access_token,
                                                     local_files_only=False)
        unused_kwargs = {}
        if self.hf_cfg.pretrained_kwargs.pop('return_unused_kwargs', False):
            cust_config, unused_kwargs = cust_config
        cust_config.update(self.it_cfg.model_cfg)  # apply pre-init model config overrides
        return cust_config, unused_kwargs

    def hf_configured_model_init(self, cust_config: PretrainedConfig, access_token: str | None = None) \
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
                model = self.it_cfg.model_class(config=cust_config)
        return model

    def _hf_cust_token_cfg(self) -> None:
        if self.it_cfg.tokenizer_id_overrides:
            for k, v in self.it_cfg.tokenizer_id_overrides.items():
                setattr(self.model.config, k, v)

    def _hf_maybe_resize_token_embeddings(self) -> None:
        if not self.it_cfg.tokenizer_id_overrides:
            return
        vocab_size = getattr(self.model.base_model, 'vocab_size', None) or self.model.config.vocab_size
        max_override_id = max(self.it_cfg.tokenizer_id_overrides.values())
        if max_override_id >= vocab_size:
            new_num_tokens = max_override_id + 1
            self.model.base_model.resize_token_embeddings(new_num_tokens)

    def _hf_post_init_cfg(self) -> None:
        self.model.config.update(self.it_cfg.model_cfg)  # apply post-init model config overrides
        # TODO: right now, a default lm_generation_cfg is always loaded even when not needed which is wasteful
        #       this checks for the presence of `lm_generation_cfg` so it still works when lm_generation_cfg defaults
        #       to None
        # since some generation config depends on post-init model updates, we defer generation-related
        # model.config and model.generation_config until post-init
        # (key assumption: generation arguments do not affect model init)
        if self.generation_cfg and isinstance(self.generation_cfg, HFGenerationConfig):
            self.model.config.update(self.generation_cfg.model_config)
            if getattr(self.model, "generation_config", None):
                for k, v in self.generation_cfg.model_config.items():
                    setattr(self.model.generation_config, k, v)
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


class BaseITMixins(ITStateMixin, ITExtensionsConfigMixin, HFFromPretrainedMixin, AnalysisStepMixin, GenerativeStepMixin,
                   ProfilerHooksMixin):
    ...
