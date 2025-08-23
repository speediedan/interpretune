from typing import Any, TypeAlias
from collections.abc import Sequence
from dataclasses import dataclass

from sae_lens.saes.sae import SAEConfig
from sae_lens.saes.standard_sae import StandardSAEConfig
from transformer_lens.utils import get_device as tl_get_device

from interpretune.config import ITLensConfig, ITSerializableCfg
from interpretune.utils import rank_zero_warn, MisconfigurationException

################################################################################
# SAE Lens Configuration Encapsulation
################################################################################


@dataclass(kw_only=True)
class SAELensFromPretrainedConfig(ITSerializableCfg):
    release: str
    sae_id: str
    device: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.device is None:  # align with TL default device resolution
            self.device = str(tl_get_device())


@dataclass(kw_only=True)
class SAELensCustomConfig(ITSerializableCfg):
    cfg: SAEConfig | dict[str, Any]

    # TODO: may add additional custom behavior handling attributes here
    def __post_init__(self) -> None:
        if not isinstance(self.cfg, SAEConfig):
            # TODO: add a PR to SAELens to allow SAEConfig to ref torch dtype and device objects instead of str repr
            # ensure the user provided a valid dtype (should be handled by SAEConfig ideally)
            # if self.cfg.get('dtype', None) and not isinstance(self.cfg['dtype'], torch.dtype):
            #     self.cfg['dtype'] = _resolve_torch_dtype(self.cfg['dtype'])
            # TODO: enable configuration/introspection of custom SAE subclasses here
            self.cfg = StandardSAEConfig.from_dict(self.cfg)


SAECfgType: TypeAlias = SAELensFromPretrainedConfig | SAELensCustomConfig


@dataclass(kw_only=True)
class SAELensConfig(ITLensConfig):
    sae_cfgs: SAECfgType | Sequence[SAECfgType]
    add_saes_on_init: bool = False  # TODO: may push this down to SAE config level instead of setting for all saes
    # use_error_term: bool = False  # TODO: add support for use_error_term with on_init stateful SAEs

    @property
    def normalized_sae_cfg_refs(self) -> list[str]:
        normalized_names = []
        for sae_cfg in self.sae_cfgs:
            if isinstance(sae_cfg, SAELensFromPretrainedConfig):
                normalized_names.append(sae_cfg.sae_id)
            elif isinstance(sae_cfg, SAELensCustomConfig):
                normalized_names.append(sae_cfg.cfg.metadata.hook_name)
        return normalized_names

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sae_cfgs:
            raise MisconfigurationException(
                "At least one `SAELensFromPretrainedConfig` or `SAELensCustomConfig` must be provided to "
                "initialize a HookedSAETransformer and use SAE Lens."
            )
        if isinstance(self.sae_cfgs, (SAELensFromPretrainedConfig, SAELensCustomConfig)):
            self.sae_cfgs = [self.sae_cfgs]
        self._sync_sl_tl_device_cfg()

    def _sync_sl_tl_device_cfg(self):
        tl_device = self.tl_cfg.cfg.device if hasattr(self.tl_cfg, "cfg") else self.tl_cfg.device
        for sae_cfg in self.sae_cfgs:
            if hasattr(sae_cfg, "cfg"):
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg.cfg, tl_device=str(tl_device))
            else:
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg, tl_device=str(tl_device))

    def _sync_sl_tl_default_device(self, sae_cfg_obj: SAECfgType, tl_device):
        if sae_cfg_obj.device and tl_device:
            if sae_cfg_obj.device != tl_device:
                rank_zero_warn(
                    f"This SAEConfig's device type ('{sae_cfg_obj.device}') does not match the configured TL device "
                    f"('{tl_device}'). Setting the device type for this SAE to match the specified TL device "
                    f"('{tl_device}')."
                )
            setattr(sae_cfg_obj, "device", tl_device)
        else:
            rank_zero_warn(
                "An SAEConfig device type was not provided. Setting the device type to match the currently specified "
                f"TL device type: '{tl_device}'."
            )
            setattr(sae_cfg_obj, "device", tl_device)
