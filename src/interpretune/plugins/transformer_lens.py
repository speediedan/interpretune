from typing import Optional, Literal, List
from dataclasses import dataclass, field
from functools import reduce

import torch
from transformers import PretrainedConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utilities.devices import get_device_for_block_index

from interpretune.base.config.module import ITConfig
from interpretune.base.modules import BaseITModule, ITLightningModule
from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.base.mixins.core import CoreHelperAttributeMixin
from interpretune.base.mixins.zero_shot_classification import BaseGenerationConfig
from interpretune.utils.logging import rank_zero_warn, rank_zero_info
from interpretune.base.config.shared import ITSerializableCfg
from interpretune.utils.warnings import tl_invalid_dmap
from interpretune.utils.patched_tlens_generate import generate as patched_generate

################################################################################
# Transformer Lens Configuration Encapsulation
################################################################################

@dataclass(kw_only=True)
class ITLensFromPretrainedConfig(ITSerializableCfg):
    enabled: bool = False
    model_name: str = "gpt2-small"
    fold_ln: Optional[bool] = True
    center_writing_weights: Optional[bool] = True
    center_unembed: Optional[bool] = True
    refactor_factored_attn_matrices: Optional[bool] = False
    checkpoint_index: Optional[int] = None
    checkpoint_value: Optional[int] = None
    # only supporting str for device for now due to omegaconf container dumping limitations
    device: Optional[str] = None
    n_devices: Optional[int] = 1
    move_to_device: Optional[bool] = True
    fold_value_biases: Optional[bool] = True
    default_prepend_bos: Optional[bool] = True
    default_padding_side: Optional[Literal["left", "right"]] = "right"
    dtype: Optional[str] = "float32"

@dataclass(kw_only=True)
class ITLensConfig(ITConfig):
    """Dataclass to encapsulate the ITModuleinternal state."""
    # TODO: support only creation of HookedTransformer with pretrained method for now, later support direct creation
    tl_from_pretrained_cfg: ITLensFromPretrainedConfig = field(default_factory=lambda: ITLensFromPretrainedConfig())
    def __post_init__(self) -> None:
        device_map = self.from_pretrained_cfg.get('device_map', None)
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            rank_zero_warn(tl_invalid_dmap)
            self.from_pretrained_cfg['device_map'] = 'cpu'

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
        if self.it_cfg.tl_from_pretrained_cfg.enabled:
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
        self.loss_fn = None

    def _configured_model_init(self, cust_config: PretrainedConfig, access_token: Optional[str] = None) \
        -> torch.nn.Module:
        # usually makes sense to init the HookedTransfomer (empty) and pretrained HF model weights on cpu
        # versus moving them both to GPU (may make sense to explore meta device usage for model definition
        # in the future, only materializing parameter by parameter during loading from pretrained weights
        # to eliminate need for two copies in memory)
        # TODO: add warning that TransformerLens only specifying a single device via device map
        # (though the model will automatically be moved to multiple devices if n_devices > 1)
        if (dmap := self.it_cfg.from_pretrained_cfg.get('device_map', None)) != 'cpu':
            rank_zero_warn('Overriding TransformerLens `from_pretrained_cfg.device_map` to transform pretrained '
                            f'weights on cpu prior to moving the model to target device: {dmap}')
            self.it_cfg.from_pretrained_cfg['device_map'] = "cpu"
        model = self.it_cfg.model_class.from_pretrained(**self.it_cfg.from_pretrained_cfg, config=cust_config,
                                                        token=access_token)
        # perhaps explore initializing on the meta device and then materializing as needed layer by layer during
        # loading/processing into hookedtransformer
        # with torch.device("meta"):
        #     model = self.it_cfg.model_class(config=cust_config)  # token=access_token)
        return model

    def _convert_hf_to_tl(self) -> HookedTransformer:
        HookedTransformer.generate = patched_generate
        self.model = HookedTransformer.from_pretrained(hf_model=self.model, tokenizer=self.datamodule.tokenizer,
                                                       **self.it_cfg.tl_from_pretrained_cfg.__dict__)

    def _capture_hyperparameters(self) -> None:
        self.it_cfg.lora_cfg = None
        self.it_cfg.bitsandbytesconfig = None
        # TODO: refactor the captured config here to only add tl_from_pretrained, other added in superclass
        self.init_hparams = {
            "tl_from_pretrained_cfg": self._make_config_serializable(self.it_cfg.tl_from_pretrained_cfg,
                                                                        ['device']),
            }
        super()._capture_hyperparameters()


    def _maybe_resize_token_embeddings(self, model: torch.nn.Module) -> None:
        # embedding resizing not currently supported by ITLensModule
        return model

    def _set_input_require_grads(self) -> None:
        # not currently supported by ITLensModule
        rank_zero_info("Setting input require grads not currently supported by ITLensModule.")

    def _configure_gradient_checkpointing(self) -> None:
        # gradient checkpointing not currently supported by ITLensModule
        pass

    def _configure_peft(self) -> None:
        # peft not currently supported by ITLensModule
        pass


################################################################################
# Transformer Lens Core/Lightning Module Composition
################################################################################

class ITLensModule(TLensAttributeMixin, CoreHelperAttributeMixin, BaseITLensModule):
    ...

if _LIGHTNING_AVAILABLE:
    class ITLensLightningModule(BaseITLensModule, ITLightningModule):
        ...
else:
    ITLensLightningModule = object
