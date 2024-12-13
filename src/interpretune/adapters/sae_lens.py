from typing import Optional, Dict, Any, List, Sequence, TypeAlias
from dataclasses import dataclass, field
from functools import reduce
from copy import deepcopy
from typing_extensions import override

from sae_lens.sae import SAE, SAEConfig
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens.utils import get_device as tl_get_device

from interpretune.adapters.registration import CompositionRegistry
from interpretune.adapters.lightning import LightningDataModule, LightningModule, LightningAdapter
from interpretune.adapters.transformer_lens import ITLensConfig, BaseITLensModule, TLensAttributeMixin
from interpretune.base.config.shared import ITSerializableCfg, Adapter
from interpretune.base.components.core import CoreHelperAttributes
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import BaseITModule
from interpretune.utils.data_movement import move_data_to_device
from interpretune.utils.logging import rank_zero_warn, rank_zero_info
from interpretune.utils.patched_tlens_generate import generate as patched_generate
from interpretune.utils.exceptions import MisconfigurationException


################################################################################
# SAE Lens Configuration Encapsulation
################################################################################

@dataclass(kw_only=True)
class InstantiatedSAE:
    handle: SAE
    original_cfg: Dict[str, Any] = field(default_factory=dict)
    sparsity: Dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class SAELensFromPretrainedConfig(ITSerializableCfg):
    release: str
    sae_id: str
    device: Optional[str] = None
    dtype: Optional[str] = None

    def __post_init__(self) -> None:
        if self.device is None:  # align with TL default device resolution
            self.device = str(tl_get_device())

@dataclass(kw_only=True)
class SAELensCustomConfig(ITSerializableCfg):
    cfg: SAEConfig | Dict[str, Any]
    # TODO: may add additional custom behavior handling attributes here
    def __post_init__(self) -> None:
        if not isinstance(self.cfg, SAEConfig):
            # TODO: add a PR to SAELens to allow SAEConfig to ref torch dtype and device objects instead of str repr
            # ensure the user provided a valid dtype (should be handled by SAEConfig ideally)
            # if self.cfg.get('dtype', None) and not isinstance(self.cfg['dtype'], torch.dtype):
            #     self.cfg['dtype'] = _resolve_torch_dtype(self.cfg['dtype'])
            self.cfg = SAEConfig.from_dict(self.cfg)

SAECfgType: TypeAlias = SAELensFromPretrainedConfig | SAELensCustomConfig

@dataclass(kw_only=True)
class SAELensConfig(ITLensConfig):
    sae_cfgs: SAECfgType | Sequence[SAECfgType]
    add_saes_on_init: bool = False  # TODO: may push this down to SAE config level instead of setting for all saes
    # use_error_term: bool = False  # TODO: add support for use_error_term with on_init stateful SAEs

    @property
    def normalized_sae_cfg_refs(self) -> List[str]:
        normalized_names = []
        for sae_cfg in self.sae_cfgs:
            if isinstance(sae_cfg, SAELensFromPretrainedConfig):
                normalized_names.append(sae_cfg.sae_id)
            elif isinstance(sae_cfg, SAELensCustomConfig):
                normalized_names.append(sae_cfg.cfg.hook_name)
        return normalized_names

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sae_cfgs:
            raise MisconfigurationException("At least one `SAELensFromPretrainedConfig` or `SAELensCustomConfig` must "
                                            " be provided to initialize a HookedSAETransformer and use SAE Lens.")
        if isinstance(self.sae_cfgs, (SAELensFromPretrainedConfig, SAELensCustomConfig)):
            self.sae_cfgs = [self.sae_cfgs]
        self._sync_sl_tl_device_cfg()

    def _sync_sl_tl_device_cfg(self):
        tl_device = self.tl_cfg.cfg.device if hasattr(self.tl_cfg, 'cfg') else self.tl_cfg.device
        for sae_cfg in self.sae_cfgs:
            if hasattr(sae_cfg, 'cfg'):
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg.cfg, tl_device=str(tl_device))
            else:
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg, tl_device=str(tl_device))

    def _sync_sl_tl_default_device(self, sae_cfg_obj: SAECfgType, tl_device):
        if sae_cfg_obj.device and tl_device:
            if sae_cfg_obj.device != tl_device:  # if both are provided, TL dtype takes precedence
                rank_zero_warn(f"This SAEConfig's device type ('{sae_cfg_obj.device}') does not match the configured TL"
                               f" device ('{tl_device}'). Setting the device type for this SAE to match the specified"
                               f" TL device type ('{tl_device}').")
                setattr(sae_cfg_obj, 'device', tl_device)
        else:  # only SAEConfigs can be provided without a device
            rank_zero_warn("An SAEConfig device type was not provided. Setting the device type to match"
                           f" the currently specified TL device type: '{tl_device}'.")
            setattr(sae_cfg_obj, 'device', tl_device)


################################################################################
# Mixins to support SAE Lens in different adapter contexts
################################################################################

class SAELensAttributeMixin(TLensAttributeMixin):
    @property
    def sae_cfgs(self) -> Optional[SAEConfig]:
        try:
            # TODO: probably will need to add a separate sae_cfg property here as well that points to the configured
            #       SAEConfig
            cfg = reduce(getattr, "it_cfg.sae_cfgs".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a `SAEConfig` reference (has it been set yet?): {ae}")
            cfg = None
        return cfg

    @property
    def sae_handles(self) -> List[SAE]:
        return [sae.handle for sae in self.saes]


class BaseSAELensModule(BaseITLensModule):
    def __init__(self, *args, **kwargs):
        # using cooperative inheritance, so initialize attributes that may be required in base init methods
        self.saes: List[InstantiatedSAE] = []
        super().__init__(*args, **kwargs)
        HookedSAETransformer.generate = patched_generate

    def _convert_hf_to_tl(self) -> None:
        # if datamodule is not attached yet, attempt to retrieve tokenizer handle directly from provided it_cfg
        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer
        hf_preconversion_config = deepcopy(self.model.config)  # capture original hf config before conversion
        self.model = HookedSAETransformer.from_pretrained(hf_model=self.model, tokenizer=tokenizer_handle,
                                                          **self.it_cfg.tl_cfg.__dict__)
        self.model.config = hf_preconversion_config
        self.instantiate_saes()

    def instantiate_saes(self) -> None:
        for sae_cfg in self.it_cfg.sae_cfgs:
            assert isinstance(sae_cfg, (SAELensFromPretrainedConfig, SAELensCustomConfig))
            original_cfg, sparsity = None, None
            if isinstance(sae_cfg, SAELensFromPretrainedConfig):
                handle, original_cfg, sparsity = SAE.from_pretrained(**sae_cfg.__dict__)
            else:
                handle = SAE(cfg=sae_cfg.cfg)
            self.saes.append(added_sae := InstantiatedSAE(handle=handle, original_cfg=original_cfg, sparsity=sparsity))
            if self.it_cfg.add_saes_on_init:
                self.model.add_sae(added_sae.handle)

    def tl_config_model_init(self) -> None:
        self.model = HookedSAETransformer(tokenizer=self.it_cfg.tokenizer, **self.it_cfg.tl_cfg.__dict__)
        self.instantiate_saes()

    def _capture_hyperparameters(self) -> None:
        self._it_state._init_hparams = {"sae_cfgs": deepcopy(self.it_cfg.sae_cfgs)}
        super()._capture_hyperparameters()

    def set_input_require_grads(self) -> None:
        # not currently supported by ITLensModule
        rank_zero_info("Setting input require grads not currently supported by BASESAELensModule.")


################################################################################
# SAE Lens Module Composition
################################################################################

class SAELensAdapter(SAELensAttributeMixin):

    @classmethod
    @override
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        adapter_ctx_registry.register(Adapter.sae_lens, component_key = "datamodule",
                                        adapter_combination=(Adapter.core, Adapter.sae_lens),
                                        composition_classes=(ITDataModule,),
                                        description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(Adapter.sae_lens, component_key = "datamodule",
                                        adapter_combination=(Adapter.lightning, Adapter.sae_lens),
                                        composition_classes=(ITDataModule, LightningDataModule),
                                        description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(Adapter.sae_lens, component_key = "module",
                                        adapter_combination=(Adapter.core, Adapter.sae_lens),
                                        composition_classes=(SAELensModule,),
                                        description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(Adapter.sae_lens, component_key = "module_cfg",
                                        adapter_combination=(Adapter.core, Adapter.sae_lens),
                                        composition_classes=(SAELensConfig,),
                                        description="SAE Lens configuration that can be composed with core and l...",
        )
        adapter_ctx_registry.register(Adapter.sae_lens, component_key = "module",
                                        adapter_combination=(Adapter.lightning, Adapter.sae_lens),
                                        composition_classes=(SAELensAttributeMixin, BaseSAELensModule, LightningAdapter,
                                                             BaseITModule, LightningModule),
                                        description="SAE Lens adapter that can be composed with core and l...",
        )
        adapter_ctx_registry.register(Adapter.sae_lens, component_key = "module_cfg",
                                        adapter_combination=(Adapter.lightning, Adapter.sae_lens),
                                        composition_classes=(SAELensConfig,),
                                        description="SAE Lens adapter that can be composed with core and l...",
        )

    def batch_to_device(self, batch) -> BatchEncoding:
        move_data_to_device(batch, self.input_device)
        return batch

class SAELensModule(SAELensAdapter, CoreHelperAttributes, BaseSAELensModule):
    ...
