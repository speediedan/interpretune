from __future__ import annotations
import os
import inspect
from typing import Any, TYPE_CHECKING, List, Optional, Dict
from contextlib import contextmanager
from functools import wraps


import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.tokenization_utils_base import BatchEncoding

import interpretune as it
from interpretune.utils import rank_zero_warn, _import_class, _BNB_AVAILABLE
from interpretune.config import (
    HFFromPretrainedConfig,
    HFGenerationConfig,
    BaseGenerationConfig,
    ITConfig,
    ITState,
    ITExtensionsConfigMixin,
)

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
        if not obj.__dict__.get("_it_state", None):
            obj._it_state = ITState()


class MemProfilerHooks:
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
            with MemProfilerHooks.memprofile_ctx(self.memprofiler, phase=phase, step_idx=step_idx):
                if (
                    self.memprofiler.memprofiler_cfg.enable_saved_tensors_hooks
                    and self.memprofiler._enabled[(phase, "start")]
                ):
                    with torch.autograd.graph.saved_tensors_hooks(*self.memprofiler._saved_tensors_funcs):
                        return func(self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)

        return wrapper


class AnalysisStepMixin:
    @property
    def analysis_cfg(self) -> Optional[AnalysisCfgProtocol]:
        if not hasattr(self.it_cfg, "analysis_cfg") or self.it_cfg.analysis_cfg is None:  # type: ignore[attr-defined]  # mixin provides it_cfg
            rank_zero_warn("Analysis configuration has not been set.")
            return
        return self.it_cfg.analysis_cfg  # type: ignore[attr-defined]  # mixin provides it_cfg

    @analysis_cfg.setter
    def analysis_cfg(self, cfg: AnalysisCfgProtocol) -> None:
        self.it_cfg.analysis_cfg = cfg  # type: ignore[attr-defined]  # mixin provides it_cfg

    def on_analysis_start(self) -> Any | None:
        """Optionally execute some post-interpretune session steps if the session is not complete."""
        # TODO: we plan to avoid op-specific conditioning of this behavior, should be functionally specified in config,
        #       we should also narrow the scope if possible to a context manager around the relevant ops themselves
        # Preserve caller's grad state and restore later in on_analysis_end. Tests expect we do not leak
        # a global torch grad_enabled state change across tests.
        try:
            self._prev_grad_enabled = torch.is_grad_enabled()
        except Exception:
            # defensive: if something goes wrong, fallback to default True
            self._prev_grad_enabled = True

        if self.analysis_cfg is not None and self.analysis_cfg.op == it.logit_diffs_attr_grad:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

    def on_analysis_epoch_end(self) -> Any | None:
        pass
        # TODO: maybe reintroduce logic here if we decide to keep per-epoch versions or perform other caching
        # Create a shallow copy from the current analysis cache
        # cache_copy = self.analysis_cfg.analysis_store
        # self._analysis_stores.append(cache_copy)
        # TODO: we don't want to reset the analysis store but rather ensure we flush the current epoch
        # self.analysis_cfg.reset_analysis_store()  # Prepare a new instance for the next epoch, preserving save_cfg

    def on_analysis_end(self) -> Any | None:
        """Optionally execute some post-interpretune session steps if the session is not complete."""
        # reset internal cache list (TODO: maybe keep this around and reset only on session start?)
        # TODO: we can avoid this analysis_stores reset if we make dataset per-epoch subsplits
        # TODO: flip back to the default if we disabled grad in on_analysis_start, again, this is terrible and should
        # be handled more narrowly and functionally rather than op conditioned. Likely a context manager in the runner.
        # self._analysis_stores = []  # uncomment if we re-enable the reset of the analysis stores
        # Restore previous grad state if we saved one. This avoids leaking a disabled grad state
        # to other tests or parts of the process.
        prev = getattr(self, "_prev_grad_enabled", None)
        if prev is not None:
            torch.set_grad_enabled(prev)
            try:
                delattr(self, "_prev_grad_enabled")
            except Exception:
                pass
        if not self.session_complete:  # type: ignore[attr-defined]  # mixin provides session_complete
            self.on_session_end()  # type: ignore[attr-defined]  # mixin provides on_session_end

    def model_sig_keys(self, target_method: str) -> list:
        return [param.name for param in inspect.signature(getattr(self.model, target_method)).parameters.values()]  # type: ignore[attr-defined]  # mixin provides model

    def auto_prune_batch(self, batch: BatchEncoding, target_method: str) -> dict[str, Any]:
        # since we're abstracting the same generative classification logic to be used with different frameworks, models
        # and datasets we use a mapping function to provide only data inputs a given generate function supports (for
        # frameworks that don't handle variadic kwargs). This currently requires the user provides
        # compatible models and dataset.
        # TODO: consider adding further upstream configuration validation that warns the user if the provided dataset
        # and step logic are incompatible
        # TODO: handle regular dicts in addition to BatchEncoding?
        return {bk: batch[bk] for bk in list(batch.data) if bk in self.model_sig_keys(target_method)}  # type: ignore[attr-defined]  # mixin provides model_sig_keys


class GenerativeStepMixin:
    # Often used for n-shot classification, those contexts are only a subset of generative classification use cases

    _gen_sig_keys: list | None = None
    GEN_PREPARES_INPUTS_SIGS: tuple = ("_prepare_model_inputs",)

    @property
    def generation_cfg(self) -> BaseGenerationConfig | None:
        return self.it_cfg.generative_step_cfg.lm_generation_cfg  # type: ignore[attr-defined]  # mixin provides it_cfg

    @property
    def gen_sig_keys(self) -> list:
        if not self._gen_sig_keys:
            generate_signature = inspect.signature(self.model.generate)  # type: ignore[attr-defined]  # mixin provides model
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
        return self.it_cfg.generative_step_cfg.input_inspection_enabled and not self._generate_prepares_inputs()  # type: ignore[attr-defined]  # mixin provides it_cfg

    def _generate_prepares_inputs(self) -> bool:
        # match sentinal methods indicating that a given model's generate function prepares inputs
        return any(hasattr(self.model, prep_method) for prep_method in self.GEN_PREPARES_INPUTS_SIGS)  # type: ignore[attr-defined]  # mixin provides model

    def it_generate(self, batch: BatchEncoding | torch.Tensor, **kwargs) -> Any:
        try:
            # variadic kwargs not supported by generate so inspect kwargs and use only those supported
            if "kwargs" not in self.gen_sig_keys:
                kwargs = self.map_gen_kwargs(kwargs)
            if isinstance(batch, torch.Tensor):
                outputs = self.model.generate(batch, **kwargs)  # type: ignore[attr-defined]  # mixin provides model
            else:
                if self._should_inspect_inputs:
                    batch = self.map_gen_inputs(batch)  # type: ignore[assignment]  # BatchEncoding can be dict[str, Any] for generation
                outputs = self.model.generate(**batch, **kwargs)  # type: ignore[attr-defined]  # mixin provides model
        except (TypeError, AttributeError, ValueError) as ge:
            # TODO: consider further inspecting the possible generation errors encountered here to help narrow the
            # problem space for the user
            gen_dataset_info_msg = (
                f"The following keys were found in the provided data batch: {os.sep} {list(batch.data)}). The current"
                f" generate method ({self.model.generate}) accepts: {os.sep} {[self._gen_sig_keys]}."  # type: ignore[attr-defined]  # mixin provides model
            )
            rank_zero_warn(gen_dataset_info_msg)
            raise Exception(f"{gen_dataset_info_msg} Received the following error msg: {ge}")
        return outputs


class ClassificationMixin:
    # Default classification helper methods

    def init_classification_mapping(self) -> None:
        it_cfg, tokenizer = self.it_cfg, self.datamodule.tokenizer  # type: ignore[attr-defined]  # mixin provides it_cfg and datamodule
        token_ids = tokenizer.convert_tokens_to_ids(it_cfg.classification_mapping)
        device = self.device if isinstance(self.device, torch.device) else self.output_device  # type: ignore[attr-defined]  # mixin provides device/output_device
        it_cfg.classification_mapping_indices = torch.tensor(token_ids, device=device)

    def standardize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # to support genclassif/non-genclassif configs and LM/SeqClassification heads we adhere to the following logits
        # logical shape invariant: [batch size, positions to consider, answers to consider]
        if isinstance(logits, tuple):
            logits = torch.stack([out for out in logits], dim=1)
        logits = logits.to(device=self.device)  # type: ignore[attr-defined]  # mixin provides device
        if logits.ndim == 2:  # if answer logits have already been squeezed
            logits = logits.unsqueeze(1)
        if logits.shape[-1] != self.it_cfg.num_labels:  # type: ignore[attr-defined]  # mixin provides it_cfg
            # Only use custom mapping if generative_step_cfg is enabled and indices are set
            if (mapping_indices := self.it_cfg.classification_mapping_indices) is not None:  # type: ignore[attr-defined]  # mixin provides it_cfg
                map_indices = mapping_indices
            else:
                raise ValueError("The logits shape does not match the expected number of labels.")

            logits = torch.index_select(logits, -1, map_indices)
            # for non-generative (standard classification), keep only the last position
            if not self.it_cfg.generative_step_cfg.enabled:  # type: ignore[attr-defined]  # mixin provides it_cfg
                logits = logits[:, -1:, :]
        return logits

    def labels_to_ids(self, labels: List[str]) -> tuple[torch.Tensor, List[str]]:
        return torch.take(self.it_cfg.classification_mapping_indices, labels), labels  # type: ignore[attr-defined]  # mixin provides it_cfg

    def logits_and_labels(self, batch: BatchEncoding, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
        label_ids, labels = self.labels_to_ids(batch.pop("labels"))
        logits = self(**batch)  # type: ignore[misc]  # mixin provides __call__ through composition
        # TODO: add another layer of abstraction here to handle different model output types? Tradeoffs to consider...
        if not isinstance(logits, torch.Tensor):
            logits = logits.logits
            assert isinstance(logits, torch.Tensor), f"Expected logits to be a torch.Tensor but got {type(logits)}"
        return torch.squeeze(logits[:, -1, :], dim=1), label_ids, labels

    def collect_answers(self, logits: torch.Tensor | tuple, labels: torch.Tensor, mode: str = "log") -> Optional[Dict]:
        logits = self.standardize_logits(logits)  # type: ignore[arg-type]  # standardize_logits handles tuple case
        per_example_answers, _ = torch.max(logits, dim=-2)
        preds = torch.argmax(per_example_answers, axis=-1)  # type: ignore[call-arg]
        metric_dict = self.metric.compute(predictions=preds, references=labels)  # type: ignore[attr-defined]  # mixin provides metric
        # TODO: check if this type casting is still required for lightning torchmetrics, bug should be fixed now...
        metric_dict = dict(
            map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)), metric_dict.items())  # type: ignore[attr-defined]  # mixin provides device
        )
        if mode == "log":
            self.log_dict(metric_dict, prog_bar=True, sync_dist=True)  # type: ignore[attr-defined]  # mixin provides log_dict
        else:
            return metric_dict


class HFFromPretrainedMixin:
    """ " Barebones interface to setup optimizers and schedulers for manual optimization with core IT modules."""

    # proper initialization of these variables should be done in the child class
    it_cfg: ITConfig
    model: torch.nn.Module

    @property
    def hf_cfg(self) -> HFFromPretrainedConfig | None:
        return self.it_cfg.hf_from_pretrained_cfg  # type: ignore[attr-defined]  # mixin provides it_cfg

    def hf_pretrained_model_init(self) -> None:
        access_token = (
            os.environ[self.it_cfg.os_env_model_auth_key.upper()] if self.it_cfg.os_env_model_auth_key else None  # type: ignore[attr-defined]  # mixin provides it_cfg
        )
        quantization_config = self._hf_configure_quantization()
        self._update_hf_pretrained_cfg(quantization_config)
        cust_config, _ = self._hf_gen_cust_config(access_token)
        self.model = self.hf_configured_model_init(cust_config, access_token)  # type: ignore[attr-defined]  # mixin provides model and hf_configured_model_init
        self._hf_cust_token_cfg()
        self._hf_maybe_resize_token_embeddings()
        self._hf_post_init_cfg()

    # TODO: move this and other hooks that may be overridden for non-HF contexts into a separate class
    def set_input_require_grads(self) -> None:
        if self.hf_cfg and self.hf_cfg.enable_input_require_grads:
            self.model.enable_input_require_grads()  # type: ignore[attr-defined]  # mixin provides model

    def _hf_configure_quantization(self) -> Any | None:
        if self.hf_cfg and self.hf_cfg.bitsandbytesconfig and _BNB_AVAILABLE:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(**self.hf_cfg.bitsandbytesconfig)
        else:
            quantization_config = None
        return quantization_config

    def _update_hf_pretrained_cfg(self, quantization_config: dict[str, Any] | None = None) -> None:
        additional_from_pretrained_kwargs = {
            "pretrained_model_name_or_path": self.it_cfg.model_name_or_path,  # type: ignore[attr-defined]  # mixin provides it_cfg
            "quantization_config": quantization_config,
            "torch_dtype": self.torch_dtype,  # type: ignore[attr-defined]  # mixin provides torch_dtype
        }
        assert self.hf_cfg is not None, "hf_cfg should be set when calling _update_hf_pretrained_cfg"
        self.hf_cfg.pretrained_kwargs.update(additional_from_pretrained_kwargs)

    def _hf_gen_cust_config(self, access_token: str | None = None) -> tuple[PretrainedConfig, dict]:
        assert self.hf_cfg is not None, "hf_cfg should be set when calling _hf_gen_cust_config"
        if self.hf_cfg.model_head:
            self.it_cfg.model_class = _import_class(self.hf_cfg.model_head)  # type: ignore[attr-defined]  # mixin provides it_cfg
            cust_config = AutoConfig.from_pretrained(**self.hf_cfg.pretrained_kwargs, token=access_token)
        elif self.hf_cfg.dynamic_module_cfg:
            config_class = get_class_from_dynamic_module(
                self.hf_cfg.dynamic_module_cfg["config_class"],
                self.it_cfg.model_name_or_path,  # type: ignore[attr-defined]  # mixin provides it_cfg
            )
            self.it_cfg.model_class = get_class_from_dynamic_module(  # type: ignore[attr-defined]  # mixin provides it_cfg
                self.hf_cfg.dynamic_module_cfg["model_class"],
                self.it_cfg.model_name_or_path,  # type: ignore[attr-defined]  # mixin provides it_cfg
            )
            cust_config = config_class.from_pretrained(self.it_cfg.model_name_or_path)  # type: ignore[attr-defined]  # mixin provides it_cfg
        else:
            if self.it_cfg.defer_model_init:  # type: ignore[attr-defined]  # mixin provides it_cfg
                rank_zero_warn(
                    "`defer_model_init` not currently supported without `model_head` or "
                    "`dynamic_module_cfg`. Proceeding with model init."
                )
            cust_config = AutoConfig.from_pretrained(
                **self.hf_cfg.pretrained_kwargs, token=access_token, local_files_only=False
            )
        unused_kwargs: dict = {}
        if self.hf_cfg.pretrained_kwargs.pop("return_unused_kwargs", False):
            cust_config, unused_kwargs = cust_config  # type: ignore[misc]  # HF API can return tuple when return_unused_kwargs=True
        assert isinstance(cust_config, PretrainedConfig), "Config should be PretrainedConfig instance"
        cust_config.update(
            self.it_cfg.model_cfg
        )  # apply pre-init model config overrides  # type: ignore[attr-defined]  # mixin provides it_cfg
        return cust_config, unused_kwargs

    def hf_configured_model_init(
        self, cust_config: PretrainedConfig, access_token: str | None = None
    ) -> torch.nn.Module:
        cust_config.num_labels = self.it_cfg.num_labels  # type: ignore[attr-defined]  # mixin provides it_cfg
        assert self.hf_cfg is not None, "hf_cfg should be set when calling hf_configured_model_init"
        head_configured = self.hf_cfg.model_head or self.hf_cfg.dynamic_module_cfg
        # TODO: parameterize the default model head when one not provided in pretrained config
        if not self.it_cfg.defer_model_init:  # type: ignore[attr-defined]  # mixin provides it_cfg
            if not head_configured:
                self.it_cfg.model_class = _import_class(self.it_cfg.hf_from_pretrained_cfg.default_head)  # type: ignore[attr-defined]  # mixin provides it_cfg
            model = self.it_cfg.model_class.from_pretrained(  # type: ignore[attr-defined]  # mixin provides it_cfg
                **self.hf_cfg.pretrained_kwargs, config=cust_config, token=access_token
            )
        else:  # defer model materialization (e.g., to `configure_model` hook)
            with torch.device("meta"):
                model = self.it_cfg.model_class(config=cust_config)  # type: ignore[attr-defined]  # mixin provides it_cfg
        return model

    def _hf_cust_token_cfg(self) -> None:
        if self.it_cfg.tokenizer_id_overrides:  # type: ignore[attr-defined]  # mixin provides it_cfg
            for k, v in self.it_cfg.tokenizer_id_overrides.items():  # type: ignore[attr-defined]  # mixin provides it_cfg
                setattr(self.model.config, k, v)  # type: ignore[attr-defined]  # mixin provides model

    def _hf_maybe_resize_token_embeddings(self) -> None:
        if not self.it_cfg.tokenizer_id_overrides:  # type: ignore[attr-defined]  # mixin provides it_cfg
            return
        vocab_size = getattr(self.model.base_model, "vocab_size", None) or self.model.config.vocab_size  # type: ignore[attr-defined]  # mixin provides model
        max_override_id = max(self.it_cfg.tokenizer_id_overrides.values())  # type: ignore[attr-defined]  # mixin provides it_cfg
        if max_override_id >= vocab_size:
            new_num_tokens = max_override_id + 1
            self.model.base_model.resize_token_embeddings(new_num_tokens)  # type: ignore[attr-defined]  # mixin provides model

    def _hf_post_init_cfg(self) -> None:
        self.model.config.update(self.it_cfg.model_cfg)  # type: ignore[attr-defined]  # mixin provides model and it_cfg, HF config has update method
        # TODO: right now, a default lm_generation_cfg is always loaded even when not needed which is wasteful
        #       this checks for the presence of `lm_generation_cfg` so it still works when lm_generation_cfg defaults
        #       to None
        # since some generation config depends on post-init model updates, we defer generation-related
        # model.config and model.generation_config until post-init
        # (key assumption: generation arguments do not affect model init)
        if self.generation_cfg and isinstance(self.generation_cfg, HFGenerationConfig):  # type: ignore[attr-defined]  # mixin provides generation_cfg
            self.model.config.update(self.generation_cfg.model_config)  # type: ignore[attr-defined]  # mixin provides model and generation_cfg
            if getattr(self.model, "generation_config", None):  # type: ignore[attr-defined]  # mixin provides model
                for k, v in self.generation_cfg.model_config.items():  # type: ignore[attr-defined]  # mixin provides generation_cfg
                    setattr(self.model.generation_config, k, v)  # type: ignore[attr-defined]  # mixin provides model
        assert self.hf_cfg is not None, "hf_cfg should be set when calling _hf_post_init_cfg"
        self.model.config.use_cache = self.hf_cfg.use_model_cache  # type: ignore[attr-defined]  # mixin provides model
        self._configure_gradient_checkpointing()
        # TODO: disentagle use of bnb and lora configs in the future, create single peft config perhaps
        if all((_BNB_AVAILABLE, self.hf_cfg.bitsandbytesconfig, self.hf_cfg.lora_cfg)):
            self._configure_peft()

    def _configure_gradient_checkpointing(self) -> None:
        assert self.hf_cfg is not None, "hf_cfg should be set when calling _configure_gradient_checkpointing"
        if self.hf_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()  # type: ignore[attr-defined]  # mixin provides model

    def _configure_peft(self) -> None:
        assert self.hf_cfg is not None, "hf_cfg should be set when calling _configure_peft"
        if self.hf_cfg.activation_checkpointing:
            self.model.gradient_checkpointing_enable()  # type: ignore[attr-defined]  # mixin provides model
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

        self.model = prepare_model_for_kbit_training(self.model)  # type: ignore[attr-defined]  # mixin provides model
        self.model = get_peft_model(self.model, LoraConfig(**self.hf_cfg.lora_cfg))  # type: ignore[attr-defined]  # mixin provides model


class BaseITMixins(
    ITStateMixin,
    ITExtensionsConfigMixin,
    HFFromPretrainedMixin,
    ClassificationMixin,
    AnalysisStepMixin,
    GenerativeStepMixin,
    MemProfilerHooks,
): ...
