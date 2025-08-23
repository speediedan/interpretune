from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass
from functools import reduce

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformer_lens import HookedTransformerConfig
from transformer_lens.utils import get_device as tl_get_device

from interpretune.config import ITConfig, HFFromPretrainedConfig, CoreGenerationConfig, ITSerializableCfg
from interpretune.utils import _resolve_torch_dtype, tl_invalid_dmap, rank_zero_warn, MisconfigurationException

################################################################################
# Transformer Lens Configuration Encapsulation
################################################################################


@dataclass(kw_only=True)
class ITLensSharedConfig(ITSerializableCfg):
    """Transformer Lens configuration shared across both `from_pretrained` and config based instantiation modes."""

    move_to_device: Optional[bool] = True
    default_padding_side: Optional[Literal["left", "right"]] = "right"


# TODO: open a PR to have TL `from_pretrained` config encapsulated in a dataclass for improved external compatibility
@dataclass(kw_only=True)
class ITLensFromPretrainedConfig(ITLensSharedConfig):
    model_name: str = "gpt2-small"
    fold_ln: Optional[bool] = True
    center_writing_weights: Optional[bool] = True
    center_unembed: Optional[bool] = True
    refactor_factored_attn_matrices: Optional[bool] = False
    checkpoint_index: Optional[int] = None
    checkpoint_value: Optional[int] = None
    # for pretrained cfg, IT handles the HF model instantiation via model_name or_path
    hf_model: Optional[AutoModelForCausalLM | str] = None
    # currently only support serializing str for device due to omegaconf container dumping limitations
    device: Optional[str] = None
    n_devices: Optional[int] = 1
    # IT handles the tokenizer instantiation via either tokenizer, tokenizer_name or model_name_or_path
    tokenizer: Optional[PreTrainedTokenizerBase] = None  # for pretrained cfg, IT instantiates the tokenizer
    fold_value_biases: Optional[bool] = True
    default_prepend_bos: Optional[bool] = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.device is None:  # align with TL default device resolution
            self.device = tl_get_device()


# rather than use from_pretrained_no_processing wrapper, we can specify the simplified config defaults we want directly
@dataclass(kw_only=True)
class ITLensFromPretrainedNoProcessingConfig(ITLensFromPretrainedConfig):
    fold_ln: bool | None = False
    center_writing_weights: bool | None = False
    center_unembed: bool | None = False
    refactor_factored_attn_matrices: bool | None = False
    fold_value_biases: bool | None = False
    dtype: str = "float32"
    default_prepend_bos: bool | None = True


@dataclass(kw_only=True)
class ITLensCustomConfig(ITLensSharedConfig):
    cfg: HookedTransformerConfig | Dict[str, Any]

    # IT handles the tokenizer instantiation via either tokenizer, tokenizer_name or model_name_or_path
    # tokenizer: Optional[PreTrainedTokenizerBase] = None
    def __post_init__(self) -> None:
        if not isinstance(self.cfg, HookedTransformerConfig):
            # ensure the user provided a valid dtype (should be handled by HookedTransformerConfig ideally)
            if self.cfg.get("dtype", None) and not isinstance(self.cfg["dtype"], torch.dtype):
                self.cfg["dtype"] = _resolve_torch_dtype(self.cfg["dtype"])
            self.cfg = HookedTransformerConfig.from_dict(self.cfg)


@dataclass(kw_only=True)
class ITLensConfig(ITConfig):
    """Dataclass to encapsulate the ITModuleinternal state."""

    tl_cfg: ITLensFromPretrainedConfig | ITLensCustomConfig

    def __post_init__(self) -> None:
        if not self.tl_cfg:
            raise MisconfigurationException(
                "Either a `ITLensFromPretrainedConfig` or `ITLensCustomConfig` must be"
                " provided to initialize a HookedTransformer and use Transformer Lens."
            )
        # internal variable used to bootstrap model initialization mode (we may need to override hf_from_pretrained_cfg)
        self._load_from_pretrained = False if isinstance(self.tl_cfg, ITLensCustomConfig) else True
        if not self._load_from_pretrained:
            self._disable_pretrained_model_mode()  # after this, hf_from_pretrained_cfg exists only if used
            self._torch_dtype = _resolve_torch_dtype(self.tl_cfg.cfg.dtype)
        else:
            # TL from pretrained currently requires a hf_from_pretrained_cfg, create one if it's not already configured
            if not self.hf_from_pretrained_cfg:
                self.hf_from_pretrained_cfg = HFFromPretrainedConfig()
            elif not isinstance(self.hf_from_pretrained_cfg, HFFromPretrainedConfig):
                try:
                    self.hf_from_pretrained_cfg = HFFromPretrainedConfig(**self.hf_from_pretrained_cfg)
                except Exception as e:
                    raise MisconfigurationException(
                        f"Failed to initialize `HFFromPretrainedConfig` from provided"
                        f" `hf_from_pretrained_cfg`. `hf_from_pretrained_cfg` should be "
                        " either a `HFFromPretrainedConfig` or a dict convertible to one."
                        f" Error: {e}"
                    )
            self._sync_pretrained_cfg()
        self._translate_tl_config()
        super().__post_init__()

    def _map_tl_fallback(self, target_key: str, tl_cfg_key: str):
        qual_sub_key = tl_cfg_key.split(".")
        fallback_val = reduce(getattr, qual_sub_key, self)
        if fallback_val not in [None, "custom"]:
            if getattr(self, target_key, None) is not None:
                hf_override_msg = f"Since `{target_key}` was provided, `{tl_cfg_key}` will be ignored."
            else:
                hf_override_msg = (
                    f"Since `{target_key} was not provided, the value provided for `{tl_cfg_key}` will"
                    f" be used for `{target_key}`."
                )
                setattr(self, target_key, fallback_val)
            setattr(reduce(getattr, qual_sub_key[:-1], self), qual_sub_key[-1], None)
            rank_zero_warn(
                f"Interpretune manages the HF model instantiation via `model_name_or_path`. {hf_override_msg}"
            )

    def _translate_tl_config(self):
        # TODO: driving this fallback mapping from a dict
        if self._load_from_pretrained:
            self._map_tl_fallback(target_key="model_name_or_path", tl_cfg_key="tl_cfg.hf_model")
            self._map_tl_fallback(target_key="tokenizer", tl_cfg_key="tl_cfg.tokenizer")
            # self._prune_converted_keys()
        else:
            self._map_tl_fallback(target_key="model_name_or_path", tl_cfg_key="tl_cfg.cfg.model_name")
            self._map_tl_fallback(target_key="tokenizer_name", tl_cfg_key="tl_cfg.cfg.tokenizer_name")

    # def _prune_converted_keys(self):
    #     if self._load_from_pretrained:
    #         for attr in ['tokenizer', 'hf_model']:
    #             if attr in type(self.tl_cfg).__dict__:
    #                 if getattr(self.tl_cfg, attr) is not None:
    #                     expected_none = f"{getattr(self.tl_cfg, attr)} should have been translated and set to `None`."
    #                     assert getattr(self.tl_cfg, attr) is None, expected_none
    #                 delattr(self.tl_cfg, attr)

    def _disable_pretrained_model_mode(self):
        ignored_attrs = []
        for attr in ["hf_from_pretrained_cfg", "defer_model_init"]:
            if getattr(self, attr):
                ignored_attrs.append(attr)
                setattr(self, attr, None)
        if len(ignored_attrs) > 0:
            rank_zero_warn(
                "Since an `ITLensCustomConfig` has been provided, the following list of set `ITConfig`"
                f" attributes will be ignored: {ignored_attrs}."
            )

    def _sync_pretrained_cfg(self):
        if self.hf_from_pretrained_cfg:
            self._check_supported_device_map()
            if hf_dtype := self.hf_from_pretrained_cfg.pretrained_kwargs.get("torch_dtype", None):
                hf_dtype = _resolve_torch_dtype(hf_dtype)
            tl_dtype = _resolve_torch_dtype(self.tl_cfg.dtype)
            self._sync_hf_tl_dtypes(hf_dtype, tl_dtype)

    def _check_supported_device_map(self):
        if self.hf_from_pretrained_cfg is None or self.hf_from_pretrained_cfg.pretrained_kwargs is None:
            return
        device_map = self.hf_from_pretrained_cfg.pretrained_kwargs.get("device_map", None)
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            rank_zero_warn(tl_invalid_dmap)
            self.hf_from_pretrained_cfg.pretrained_kwargs["device_map"] = "cpu"

    def _sync_hf_tl_dtypes(self, hf_dtype, tl_dtype):
        if self.hf_from_pretrained_cfg is None:
            return

        if self.hf_from_pretrained_cfg.pretrained_kwargs is None:
            self.hf_from_pretrained_cfg.pretrained_kwargs = {}

        if hf_dtype and tl_dtype:
            if hf_dtype != tl_dtype:  # if both are provided, TL dtype takes precedence
                rank_zero_warn(
                    f"HF `from_pretrained` dtype {hf_dtype} does not match TL dtype {tl_dtype}."
                    f" Setting both to the specified TL dtype {tl_dtype}."
                )
                self.hf_from_pretrained_cfg.pretrained_kwargs["torch_dtype"] = tl_dtype
        else:
            rank_zero_warn(
                "HF `from_pretrained` dtype was not provided. Setting `from_pretrained` dtype to match"
                f" specified TL dtype: {tl_dtype}."
            )
            self.hf_from_pretrained_cfg.pretrained_kwargs["torch_dtype"] = tl_dtype


@dataclass(kw_only=True)
class TLensGenerationConfig(CoreGenerationConfig):
    stop_at_eos: bool = True
    eos_token_id: Optional[int] = None
    freq_penalty: float = 0.0
    use_past_kv_cache: bool = True
    prepend_bos: Optional[bool] = None
    padding_side: Optional[Literal["left", "right"]] = None
    return_type: Optional[str] = "input"
    output_logits: Optional[bool] = True  # TODO: consider setting this back to None after testing
    verbose: bool = True
