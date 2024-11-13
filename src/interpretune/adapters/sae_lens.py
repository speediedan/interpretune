from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import reduce
from copy import deepcopy
from typing_extensions import override

from sae_lens.sae import SAE, SAEConfig
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.adapters.registration import CompositionRegistry, Adapter
from interpretune.adapters.lightning import LightningDataModule, LightningModule, LightningAdapter
from interpretune.adapters.transformer_lens import ITLensConfig, BaseITLensModule, TLensAttributeMixin
from interpretune.base.config.shared import ITSerializableCfg
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
class SAELensFromPretrainedConfig(ITSerializableCfg):
    release: str
    sae_id: str
    device: str = "cpu"
    dtype: Optional[str] = None


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

@dataclass(kw_only=True)
class SAELensConfig(ITLensConfig):
    # TODO: enable adding a list of saes (pretrained or custom) to a HookedSAETransformer after single case verification
    sae_cfg: SAELensFromPretrainedConfig | SAELensCustomConfig
    # use_error_term: bool = False  # TODO: add support for use_error_term

    def __post_init__(self) -> None:
        if not self.sae_cfg:
            raise MisconfigurationException("Either a `SAELensFromPretrainedConfig` or `SAELensCustomConfig` must be"
                                            " provided to initialize a HookedSAETransformer and use SAE Lens.")
        super().__post_init__()


################################################################################
# Mixins to support SAE Lens in different adapter contexts
################################################################################

class SAELensAttributeMixin(TLensAttributeMixin):
    @property
    def sae_cfg(self) -> Optional[SAEConfig]:
        try:
            # TODO: probably will need to add a separate sae_cfg property here as well that points to the configured
            #       SAEConfig
            cfg = reduce(getattr, "it_cfg.sae_cfg".split("."), self)
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a `SAEConfig` reference (has it been set yet?): {ae}")
            cfg = None
        return cfg


class BaseSAELensModule(BaseITLensModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_cfg_dict = None
        self.sparsity = None
        HookedSAETransformer.generate = patched_generate

    def _convert_hf_to_tl(self) -> None:
        # if datamodule is not attached yet, attempt to retrieve tokenizer handle directly from provided it_cfg
        tokenizer_handle = self.datamodule.tokenizer if self.datamodule else self.it_cfg.tokenizer
        hf_preconversion_config = deepcopy(self.model.config)  # capture original hf config before conversion
        self.model = HookedSAETransformer.from_pretrained(hf_model=self.model, tokenizer=tokenizer_handle,
                                                          **self.it_cfg.tl_cfg.__dict__)
        self.model.config = hf_preconversion_config
        self.sae, self.original_cfg_dict, self.sparsity = SAE.from_pretrained(**self.sae_cfg.__dict__)
        self.model.add_sae(self.sae)

    def tl_config_model_init(self) -> None:
        self.model = HookedSAETransformer(tokenizer=self.it_cfg.tokenizer, **self.it_cfg.tl_cfg.__dict__)
        self.sae = SAE(cfg=self.sae_cfg.cfg)
        self.model.add_sae(self.sae)

    def _capture_hyperparameters(self) -> None:
        self._it_state._init_hparams = {"sae_cfg": deepcopy(self.it_cfg.sae_cfg)}
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
        adapter_ctx_registry.register(Adapter.sae_lens, component_key = "module",
                                        adapter_combination=(Adapter.lightning, Adapter.sae_lens),
                                        composition_classes=(SAELensAttributeMixin, BaseSAELensModule, LightningAdapter,
                                                             BaseITModule, LightningModule),
                                        description="SAE Lens adapter that can be composed with core and l...",
        )

    def batch_to_device(self, batch) -> BatchEncoding:
        move_data_to_device(batch, self.input_device)
        return batch

class SAELensModule(SAELensAdapter, CoreHelperAttributes, BaseSAELensModule):
    ...
