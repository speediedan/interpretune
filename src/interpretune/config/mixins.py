# see https://peps.python.org/pep-0749, no longer needed when 3.13 reaches EOL
from __future__ import annotations
import os
from typing import Any
from dataclasses import dataclass, field
from pprint import pformat

import torch
from transformers.generation.configuration_utils import GenerationConfig

from interpretune.config import ITSerializableCfg
from interpretune.utils import MisconfigurationException, rank_zero_warn, _resolve_dtype


@dataclass(kw_only=True)
class BaseGenerationConfig(ITSerializableCfg):
    """Base for generation configuration: kwargs passed straight through to ``model.generate``."""

    # kwargs passed directly to the model.generate method
    generate_kwargs: dict = field(default_factory=dict)


# TODO: we should be able to standardize on the HF GenerationConfig interface and remove CoreGenerationConfig once
#       (if) TL migrates away from HookedTransformer.generate method to using the HF generate interface
@dataclass(kw_only=True)
class CoreGenerationConfig(BaseGenerationConfig):
    """Generation settings passed as explicit kwargs, for backends without an HF generation config.

    Used for ``HookedTransformer``, whose ``generate`` does not read an HF ``GenerationConfig``. HF dict
    flags are deliberately left unset so callers state them rather than inheriting a silent default.
    """

    max_new_tokens: int = 5  # nb maxing logits over multiple tokens (n<=5) will yield a very slight perf gain versus 1
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0
    # We intentionally leave HF dict flags unset by default. Callers or configs should explicitly
    # set these flags if they expect a ModelOutput return value from `.generate()`.
    return_dict_in_generate: bool | None = None
    output_logits: bool | None = None

    def __post_init__(self):
        # TODO: consider finding a more elegant abstraction that allows providing both model.config based and direct to
        #       generate method kwargs for assorted generate contexts
        #       currently, HF uses model.config based and potentially generate_kwargs, TL uses only generate_kwargs
        for k, v in self.__dict__.items():
            if k != "generate_kwargs":
                self.generate_kwargs[k] = v


@dataclass(kw_only=True)
class HFGenerationConfig(BaseGenerationConfig):
    """Generation settings applied to the HF model's own ``generation_config``.

    Prefer this for HF-backed models: the settings live where the model looks for them, rather than
    being re-passed at each call site.
    """

    # generation kwargs to be added to the HF model config (which in turn override the model.generation_config)
    model_config: dict = field(default_factory=dict)
    default_overrides: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        valid_hf_keys = [k for k in GenerationConfig().__dict__.keys() if not k.startswith("_")]
        # we defer to HF's default generation config for all supported `GenerationConfig` settings except for attributes
        # specified in the default or provided (`default_overrides`) override config
        # TODO: add warnings for invalid keys rather than silently ignoring
        for k, v in self.model_config.items():
            if k in valid_hf_keys:
                self.model_config[k] = v
        for k, v in self.default_overrides.items():
            if k not in self.model_config.keys() and k in valid_hf_keys:
                self.model_config[k] = v

    def __getattr__(self, name: str):
        """Expose model_config entries as direct attributes for API compatibility with CoreGenerationConfig."""
        mc = self.__dict__.get("model_config", {})
        if name in mc:
            return mc[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


@dataclass(kw_only=True)
class GenerativeClassificationConfig(ITSerializableCfg):
    """Configuration for classification performed by generating tokens rather than reading logits."""

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
    model_head: str = ""
    lora_cfg: dict[str, Any] = field(default_factory=dict)
    bitsandbytesconfig: dict[str, Any] = field(default_factory=dict)
    activation_checkpointing: bool = False
    # Whether to enable gradients for the input embeddings. Useful for finetuning adapter weights w/ a frozen model.
    enable_input_require_grads: bool = True
    default_head: str = "transformers.AutoModelForCausalLM"
    # Build the model on the meta device in every process, for a data-parallel placement to shard before it holds any
    # weights. The process a launcher made rank 0 (or one with no launcher) also loads the checkpoint on the CPU, and
    # the placement loads it into the sharded model from there (`interpretune.runners.distributed`), so no rank's
    # accelerator holds the whole model and only rank 0's host does. A model with no placement to load it stays on meta.
    meta_init: bool = False
    # How a `meta_init` model is loaded once sharded. "rank0": rank 0 loads the whole checkpoint on its CPU and the
    # placement scatters it, which works for any checkpoint `from_pretrained` reads. "dcp": every rank reads only its
    # own shards of the checkpoint's safetensors files with distributed checkpointing, so no host holds the whole model;
    # it handles key renames but refuses a model whose checkpoint needs transformers' weight converters.
    meta_init_load: str = "rank0"

    def __post_init__(self):
        if self.pretrained_kwargs.get("token", None):
            del self.pretrained_kwargs["token"]
        if self.meta_init_load not in ("rank0", "dcp"):
            raise MisconfigurationException(f"`meta_init_load` must be 'rank0' or 'dcp', not {self.meta_init_load!r}.")
        if self.meta_init_load != "rank0" and not self.meta_init:
            raise MisconfigurationException(
                f"`meta_init_load: {self.meta_init_load}` chooses how a `meta_init` model is loaded, but `meta_init`"
                " is not set."
            )
        if self.meta_init and self.bitsandbytesconfig:
            raise MisconfigurationException(
                "`meta_init` builds an unquantized model to shard and load; it cannot be combined with"
                " `bitsandbytesconfig`, whose quantized parameters FSDP2 does not shard."
            )
        if self.meta_init_load == "dcp" and self.lora_cfg:
            raise MisconfigurationException(
                "`meta_init_load: dcp` reads every value from the checkpoint's files, which hold no LoRA adapters, so"
                " it cannot initialize `lora_cfg`'s. Use `meta_init_load: rank0`, which wraps rank 0's copy with the"
                " same adapters and loads them with the base."
            )

    def _dtype_serde(self) -> torch.dtype | None:
        """Resolve and remove a 'dtype' entry from pretrained_kwargs if present."""
        if self.pretrained_kwargs and self.pretrained_kwargs.get("dtype", None):
            if resolved_dtype := _resolve_dtype(self.pretrained_kwargs["dtype"]):
                del self.pretrained_kwargs["dtype"]
                return resolved_dtype
            else:
                rank_zero_warn(
                    f"The provided `dtype` {self.pretrained_kwargs.pop('dtype')} could not"
                    " be resolved, attempting to proceed with `dtype` unset."
                )
