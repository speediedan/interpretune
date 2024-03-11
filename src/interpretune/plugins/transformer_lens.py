import os
from typing import Optional, Literal, List, Dict, Any
from dataclasses import dataclass
from functools import reduce

import torch
from transformers import PretrainedConfig, AutoModelForCausalLM, PreTrainedTokenizerBase
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utilities.devices import get_device_for_block_index

from interpretune.base.config.module import ITConfig,HFFromPretrainedConfig
from interpretune.base.modules import BaseITModule, ITLightningModule
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE, _resolve_torch_dtype
from interpretune.base.components.core import CoreHelperAttributes
from interpretune.base.config.mixins import BaseGenerationConfig
from interpretune.utils.logging import rank_zero_warn, rank_zero_info
from interpretune.base.config.shared import ITSerializableCfg
from interpretune.utils.warnings import tl_invalid_dmap
from interpretune.utils.patched_tlens_generate import generate as patched_generate


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

@dataclass(kw_only=True)
class ITLensCustomConfig(ITLensSharedConfig):
    cfg: HookedTransformerConfig | Dict[str, Any]
    # IT handles the tokenizer instantiation via either tokenizer, tokenizer_name or model_name_or_path
    # tokenizer: Optional[PreTrainedTokenizerBase] = None
    def __post_init__(self) -> None:
        if not isinstance(self.cfg, HookedTransformerConfig):
            self.cfg = HookedTransformerConfig.from_dict(self.cfg)

@dataclass(kw_only=True)
class ITLensConfig(ITConfig):
    """Dataclass to encapsulate the ITModuleinternal state."""
    tl_cfg: ITLensFromPretrainedConfig | ITLensCustomConfig

    def __post_init__(self) -> None:
        if not self.tl_cfg:
            raise ValueError("Either a `ITLensFromPretrainedConfig` or `ITLensCustomConfig` must be provided to"
                             " initialize a HookedTransformer and use Transformer Lens.")
        # internal variable used to bootstrap model initialization mode (we may need to override hf_from_pretrained_cfg)
        self._load_from_pretrained = False if isinstance(self.tl_cfg, ITLensCustomConfig) else True
        if not self._load_from_pretrained:
            self._disable_pretrained_model_mode()  # after this, hf_from_pretrained_cfg exists only if used
            self._torch_dtype = _resolve_torch_dtype(self.tl_cfg.cfg.dtype)
        else:
            # TL from pretrained currently requires a hf_from_pretrained_cfg, create one if it's not already configured
            if not self.hf_from_pretrained_cfg:
                self.hf_from_pretrained_cfg = HFFromPretrainedConfig()
            self._sync_pretrained_cfg()
        self._translate_tl_config()
        super().__post_init__()

    def _map_tl_fallback(self, target_key: str, tl_cfg_key: str):
        qual_sub_key = tl_cfg_key.split(".")
        fallback_val = reduce(getattr, qual_sub_key, self)
        if fallback_val not in [None, 'custom']:
            if getattr(self, target_key, None) is not None:
                hf_override_msg = f"Since `{target_key}` was provided, `{tl_cfg_key}` will be ignored."
            else:
                hf_override_msg = (f"Since `{target_key} was not provided, the value provided for `{tl_cfg_key}` will"
                                   f" be used for `{target_key}`.")
                setattr(self, target_key, fallback_val)
            setattr(reduce(getattr, qual_sub_key[:-1], self), qual_sub_key[-1], None)
            rank_zero_warn("Interpretune manages the HF model instantiation via `model_name_or_path`."
                            f" {hf_override_msg}")

    def _translate_tl_config(self):
        if self._load_from_pretrained:
            self._map_tl_fallback(target_key='model_name_or_path', tl_cfg_key='tl_cfg.hf_model')
            self._map_tl_fallback(target_key='tokenizer', tl_cfg_key='tl_cfg.tokenizer')
            self._prune_converted_keys()
        else:
            self._map_tl_fallback(target_key='model_name_or_path', tl_cfg_key='tl_cfg.cfg.model_name')
            self._map_tl_fallback(target_key='tokenizer_name', tl_cfg_key='tl_cfg.cfg.tokenizer_name')

    def _prune_converted_keys(self):
        if self._load_from_pretrained:
            for attr in ['tokenizer', 'hf_model']:
                if hasattr(self.tl_cfg, attr):
                    expected_none = f"{getattr(self.tl_cfg, attr)} should have been translated and set to `None`."
                    assert getattr(self.tl_cfg, attr) is None, expected_none
                    delattr(self.tl_cfg, attr)

    def _disable_pretrained_model_mode(self):
        ignored_attrs = []
        for attr in ['hf_from_pretrained_cfg', 'defer_model_init']:
            if getattr(self, attr):
                ignored_attrs.append(attr)
                setattr(self, attr, None)
        if len(ignored_attrs) > 0:
            rank_zero_warn("Since an `ITLensCustomConfig` has been provided, the following list of set `ITConfig`"
                           f" attributes will be ignored: {ignored_attrs}.")

    def _sync_pretrained_cfg(self):
        if self.hf_from_pretrained_cfg:
            self._check_supported_device_map()
            if hf_dtype := self.hf_from_pretrained_cfg.pretrained_kwargs.get('torch_dtype', None):
                hf_dtype = _resolve_torch_dtype(hf_dtype)
            tl_dtype = _resolve_torch_dtype(self.tl_cfg.dtype)
            self._sync_hf_tl_dtypes(hf_dtype, tl_dtype)

    def _check_supported_device_map(self):
      device_map = self.hf_from_pretrained_cfg.pretrained_kwargs.get('device_map', None)
      if isinstance(device_map, dict) and len(device_map.keys()) > 1:
        rank_zero_warn(tl_invalid_dmap)
        self.hf_from_pretrained_cfg.pretrained_kwargs['device_map'] = 'cpu'

    def _sync_hf_tl_dtypes(self, hf_dtype, tl_dtype):
        if hf_dtype and tl_dtype:
            if hf_dtype != tl_dtype:  # if both are provided, TL dtype takes precedence
                rank_zero_warn(f"HF `from_pretrained` dtype {hf_dtype} does not match TL dtype {tl_dtype}."
                                f" Setting both to the specified TL dtype {tl_dtype}.")
                self.hf_from_pretrained_cfg.pretrained_kwargs['torch_dtype'] = hf_dtype
        else:
            rank_zero_warn("HF `from_pretrained` dtype was not provided. Setting `from_pretrained` dtype to match"
                           f" specified TL dtype: {tl_dtype}.")
            self.hf_from_pretrained_cfg.pretrained_kwargs['torch_dtype'] = tl_dtype


@dataclass(kw_only=True)
class TLensGenerationConfig(BaseGenerationConfig):
    stop_at_eos: bool = True
    eos_token_id: Optional[int] = None
    freq_penalty: float = 0.0
    use_past_kv_cache: bool = True
    prepend_bos: Optional[bool] = None
    padding_side: Optional[Literal["left", "right"]] = None
    return_type: Optional[str] = "input"
    output_logits: Optional[bool] = True  # TODO: consider setting this back to None after testing
    verbose: bool = True

################################################################################
# Hooks and Mixins to support Transformer Lens in both Core/Lightning Contexts
################################################################################

class BaseITLensModuleHooks:
    """" LightningModule hooks implemented by the BaseITLensModule (used by both core and Lightning TransformerLens
    modules)"""

    # proper initialization of these variables should be done in the child class
    it_cfg: List[ITConfig]

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        if self.it_cfg.hf_from_pretrained_cfg:
            self._convert_hf_to_tl()


class TLensAttributeMixin:
    @property
    def tl_cfg(self) -> Optional[HookedTransformerConfig]:
        return self._core_or_lightning(c2l_map_key="_tl_cfg")

    @property
    def device(self) -> Optional[torch.device]:
        try:
            device = getattr(self, "_device", None) or getattr(self.tl_cfg, "device", None) or \
                reduce(getattr, "model.device".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a device reference (has it been set yet?): {ae}")
            device = None
        return device

    def get_tl_device(self, block_index: int) -> Optional[torch.device]:
        try:
            device = get_device_for_block_index(block_index, self.tl_cfg)
        except AttributeError as ae:
            rank_zero_warn(f"Problem determining appropriate device for block {block_index} from TransformerLens"
                           f" config. Received: {ae}")
            device = None
        return device

    @property
    def output_device(self) -> Optional[torch.device]:
        return self.get_tl_device(self.model.cfg.n_layers - 1)

    @property
    def input_device(self) -> Optional[torch.device]:
        return self.get_tl_device(0)


class BaseITLensModule(BaseITLensModuleHooks, BaseITModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        HookedTransformer.generate = patched_generate
        self.loss_fn = None

    def hf_pretrained_model_init(self) -> None:
        # for TL, only a subset of the HF pretrained init flow used since the model is replaced with a HookedTransformer
        access_token = os.environ[self.it_cfg.os_env_model_auth_key.upper()] if self.it_cfg.os_env_model_auth_key \
            else None
        quantization_config = super()._hf_configure_quantization()
        super()._update_hf_pretrained_cfg(quantization_config)
        cust_config = super()._hf_gen_cust_config(access_token)
        self.model = self._hf_configured_model_init(cust_config, access_token)

    def _hf_configured_model_init(self, cust_config: PretrainedConfig | ITLensCustomConfig,
                                  access_token: Optional[str] = None) -> torch.nn.Module:
        # usually makes sense to init the HookedTransfomer (empty) and pretrained HF model weights on cpu
        # versus moving them both to GPU (may make sense to explore meta device usage for model definition
        # in the future, only materializing parameter by parameter during loading from pretrained weights
        # to eliminate need for two copies in memory)
        # TODO: add warning that TransformerLens only specifying a single device via device  is supported
        # (though the model will automatically be moved to multiple devices if n_devices > 1)
        cust_config.num_labels = self.it_cfg.num_labels
        if (dmap := self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs.get('device_map', None)) != 'cpu':
            rank_zero_warn('Overriding `device_map` passed to TransformerLens to transform pretrained weights on'
                        f' cpu prior to moving the model to target device: {dmap}')
            self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs['device_map'] = "cpu"
        model = self.it_cfg.model_class.from_pretrained(**self.it_cfg.hf_from_pretrained_cfg.pretrained_kwargs,
                                                        config=cust_config, token=access_token)
        # perhaps explore initializing on the meta device and then materializing as needed layer by layer during
        # loading/processing into hookedtransformer
        # with torch.device("meta"):
        #     model = self.it_cfg.model_class(config=cust_config)  # token=access_token)
        return model

    def custom_model_init(self):
        # TODO: suppress messages from tl about no tokenizer here, we're deferring the tokenizer attach until setup
        self.model = HookedTransformer(tokenizer=self.it_cfg.tokenizer, **self.it_cfg.tl_cfg.__dict__)

    def _convert_hf_to_tl(self) -> None:
        # TODO: decide whether to pass remaining hf_from_pretrained_cfg args to HookedTransformer
        # (other than torch_dtype which should already have been processed and removed, device_map should also be
        # removed before passing to HookedTransformer)
        self.model = HookedTransformer.from_pretrained(hf_model=self.model, tokenizer=self.datamodule.tokenizer,
                                                       **self.it_cfg.tl_cfg.__dict__)

    def _capture_hyperparameters(self) -> None:
        # override unsupported from pretrained options
        if self.hf_cfg:
            self.hf_cfg.lora_cfg = None
            self.hf_cfg.bitsandbytesconfig = None
        # TODO: refactor the captured config here to only add tl_from_pretrained, other added in superclass
        # TODO: serialize tl_config
        if self.it_cfg.hf_from_pretrained_cfg:
            self.init_hparams = {"tl_cfg": self._make_config_serializable(self.it_cfg.tl_cfg, ['device', 'dtype']),}
        else:
            self.it_cfg.tl_cfg.cfg = self._make_config_serializable(self.it_cfg.tl_cfg.cfg, ['device', 'dtype'])
            self.init_hparams = {"tl_cfg": self.it_cfg.tl_cfg}
        super()._capture_hyperparameters()

    def set_input_require_grads(self) -> None:
        # not currently supported by ITLensModule
        rank_zero_info("Setting input require grads not currently supported by ITLensModule.")


################################################################################
# Transformer Lens Core/Lightning Module Composition
################################################################################

class ITLensModule(TLensAttributeMixin, CoreHelperAttributes, BaseITLensModule):
    ...

if _LIGHTNING_AVAILABLE:
    class ITLensLightningModule(TLensAttributeMixin, BaseITLensModule, ITLightningModule):
        ...
else:
    ITLensLightningModule = object
