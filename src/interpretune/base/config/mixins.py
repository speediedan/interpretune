# see https://peps.python.org/pep-0749, no longer needed when 3.13 reaches EOL
from __future__ import annotations
import os
from typing import Any, NamedTuple
from collections.abc import Callable
from dataclasses import dataclass, field
from pprint import pformat

import torch
from transformers.generation.configuration_utils import GenerationConfig

from interpretune.base.analysis import (AnalysisCache, ablate_sae_latent, boolean_logits_to_avg_logit_diff,
                                        resolve_names_filter, _make_simple_cache_hook)
from interpretune.base.contract.analysis import NamesFilter, AnalysisCacheProtocol
from interpretune.base.config.shared import ITSerializableCfg, AnalysisMode
from interpretune.utils.logging import rank_zero_warn
from interpretune.utils.import_utils import _resolve_torch_dtype


class ITExtension(NamedTuple):
    ext_attr: str
    ext_cls_fqn: str
    ext_cfg_fqn: str

@dataclass(kw_only=True)
class BaseGenerationConfig(ITSerializableCfg):
    # kwargs passed directly to the model.generate method
    generate_kwargs: dict = field(default_factory=dict)

@dataclass(kw_only=True)
class CoreGenerationConfig(BaseGenerationConfig):
    max_new_tokens: int = 5  # nb maxing logits over multiple tokens (n<=5) will yield a very slight perf gain versus 1
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0
    # TODO: test these additions below
    return_dict_in_generate: bool | None = True
    output_logits: bool | None = True

    def __post_init__(self):
        # TODO: consider finding a more elegant abstraction that allows providing both model.config based and direct to
        #       generate method kwargs for assorted generate contexts
        #       currently, HF uses model.config based and potentially generate_kwargs, TL uses only generate_kwargs
        for k, v in self.__dict__.items():
            if k != "generate_kwargs":
                self.generate_kwargs[k] = v

@dataclass(kw_only=True)
class HFGenerationConfig(BaseGenerationConfig):
    # generation kwargs to be added to the HF model config (which in turn override the model.generation_config)
    model_config: dict = field(default_factory=dict)
    default_overrides: dict = field(default_factory=lambda: {"return_dict_in_generate": True, "output_logits": True})

    def __post_init__(self):
        valid_hf_keys = [k for k in GenerationConfig().__dict__.keys()  if not k.startswith("_")]
        # we defer to HF's default generation config for all supported `GenerationConfig` settings except for attributes
        # specified in the default or provided (`default_overrides`) override config
        # TODO: add warnings for invalid keys rather than silently ignoring
        for k, v in self.model_config.items():
            if k in valid_hf_keys:
                self.model_config[k] = v
        for k, v in self.default_overrides.items():
            if k not in self.model_config.keys() and k in valid_hf_keys:
                self.model_config[k] = v

@dataclass(kw_only=True)
class GenerativeClassificationConfig(ITSerializableCfg):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=HFGenerationConfig)
    # for generate methods that don't also perform data preparation, filter out inputs that the model's generate
    # function does not support
    input_inspection_enabled: bool = True

    def __repr__(self):
        return f"Generative Classification Config: {os.linesep}{pformat(self.__dict__)}"

@dataclass(kw_only=True)
class HFFromPretrainedConfig(ITSerializableCfg):
    """
    HFFromPretrainedConfig: Configuration for loading a pretrained model from Huggingface along with configuration
    options contingent on the HF pretrained model type.
    """
    pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    dynamic_module_cfg: dict[str, Any] = field(default_factory=dict)
    use_model_cache: bool | None = False
    model_head: str = ''
    lora_cfg: dict[str, Any] = field(default_factory=dict)
    bitsandbytesconfig: dict[str, Any] = field(default_factory=dict)
    activation_checkpointing: bool = False
    # Whether to enable gradients for the input embeddings. Useful for finetuning adapter weights w/ a frozen model.
    enable_input_require_grads: bool = True
    default_head: str = "transformers.AutoModelForCausalLM"

    def __post_init__(self):
        if self.pretrained_kwargs.get('token', None):
            del self.pretrained_kwargs['token']

    def _torch_dtype_serde(self) -> torch.dtype | None:
        if self.pretrained_kwargs and self.pretrained_kwargs.get('torch_dtype', None):
            if resolved_dtype := _resolve_torch_dtype(self.pretrained_kwargs['torch_dtype']):
                del self.pretrained_kwargs['torch_dtype']
                return resolved_dtype
            else:
                rank_zero_warn(f"The provided `torch_dtype` {self.pretrained_kwargs.pop('torch_dtype')} could not"
                               " be resolved, attempting to proceed with `torch_dtype` unset.")

@dataclass(kw_only=True)
class AnalysisCfg(ITSerializableCfg):
    analysis_cache: AnalysisCacheProtocol = field(default_factory=AnalysisCache)
    mode: AnalysisMode = AnalysisMode.clean_no_sae
    ablate_latent_fn: Callable = ablate_sae_latent
    logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff
    fwd_hooks: list[tuple] = field(default_factory=list)
    bwd_hooks: list[tuple] = field(default_factory=list)
    cache_dict: dict = field(default_factory=dict)
    names_filter: NamesFilter | None = None
    base_logit_diffs: list = field(default_factory=list)
    alive_latents: list[dict] = field(default_factory=list)
    answer_indices: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = AnalysisMode(self.mode)
        self.names_filter = resolve_names_filter(self.names_filter)
        if not self.fwd_hooks and not self.bwd_hooks:
            self.fwd_hooks, self.bwd_hooks = self.check_add_default_hooks(
                mode=self.mode,
                names_filter=self.names_filter,
                cache_dict=self.cache_dict,
            )
        if self.logit_diff_fn is None:
            self.logit_diff_fn = boolean_logits_to_avg_logit_diff
        if self.ablate_latent_fn is None:
            self.ablate_latent_fn = ablate_sae_latent

    def reset_analysis_cache(self) -> None:
        # Prepare a new cache for the next epoch preserving save_cfg (for multi-epoch AnalysisSessionCfg instances)
        current_analysis_cls = self.analysis_cache.__class__
        current_save_cfg_cls = self.analysis_cache.save_cfg.__class__
        current_save_cfg = {k: v for k, v in self.analysis_cache.save_cfg.__dict__.items() if k != 'analysis_cache'}
        self.analysis_cache = current_analysis_cls(save_cfg=current_save_cfg_cls(**current_save_cfg))
        assert id(self.analysis_cache) == id(self.analysis_cache.save_cfg.analysis_cache)

    def check_add_default_hooks(
            self, mode: AnalysisMode, names_filter: NamesFilter | None,
            cache_dict: dict | None
            ) -> tuple[list[tuple], list[tuple]]:
        """Construct forward and backward hooks based on analysis mode."""
        fwd_hooks, bwd_hooks = [], []

        if mode == AnalysisMode.clean_no_sae:
            return fwd_hooks, bwd_hooks
        if names_filter is None:
            raise ValueError("names_filter required for non-clean modes")
        if mode == AnalysisMode.attr_patching:
            fwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict))
            )
            bwd_hooks.append(
                (names_filter, _make_simple_cache_hook(cache_dict=cache_dict, is_backward=True))
            )
        return fwd_hooks, bwd_hooks
