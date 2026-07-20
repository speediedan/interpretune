# N.B. we need to avoid annotations import here due to jsonargparse validation issues that emerge when it is used
# (PEP 563 deferred annotations cause all annotations to become strings, and jsonargparse's
# evaluate_postponed_annotations fails globally when get_type_hints encounters AnalysisCfgProtocol
# in the ITConfig hierarchy, leaving required fields like sae_cfgs unresolved)
# from __future__ import annotations
from typing import Any, TypeAlias, TYPE_CHECKING
from collections.abc import Sequence
from dataclasses import dataclass

from sae_lens.saes.sae import SAEConfig
from sae_lens.saes.standard_sae import StandardSAEConfig
from transformer_lens.utilities.devices import get_device as tl_get_device
from transformer_lens.config import HookedTransformerConfig

from interpretune.config import (
    ITConfig,
    ITSerializableCfg,
    ITLensCfgTypes,
    ITLensCustomConfig,
    ITLensFromPretrainedConfig,
)
from interpretune.config.transformer_lens import ITLensBridgeConfig, TLConfigInitMixin
from interpretune.utils import rank_zero_warn, MisconfigurationException, _resolve_dtype

if TYPE_CHECKING:
    from interpretune.config.nnsight import NNsightConfig

# Valid backend identifiers
_VALID_SAE_BACKENDS = ("transformerlens", "nnsight")

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
            #     self.cfg['dtype'] = _resolve_dtype(self.cfg['dtype'])
            # TODO: enable configuration/introspection of custom SAE subclasses here
            self.cfg = StandardSAEConfig.from_dict(self.cfg)


SAECfgType: TypeAlias = SAELensFromPretrainedConfig | SAELensCustomConfig


@dataclass(kw_only=True)
class SAELensConfig(ITConfig, TLConfigInitMixin):
    """Configuration for SAE Lens adapter.

    Supports both TransformerLens and NNsight backends via the ``backend`` field.
    When ``backend="transformerlens"`` (default), ``tl_cfg`` must be provided.
    When ``backend="nnsight"``, ``nnsight_cfg`` must be provided instead.

    Inherits from :class:`ITConfig` (not ``ITLensConfig``) so that it is backend-agnostic
    at the type level.  TL-specific initialization logic is provided by
    :class:`TLConfigInitMixin`, which is shared with ``ITLensConfig``.

    The ``use_bridge`` field is only meaningful when ``backend="transformerlens"``
    and controls whether a SAETransformerBridge (True) or HookedSAETransformer (False) is used.
    """

    # Backend selection
    backend: str = "transformerlens"
    use_bridge: bool = True

    # TL backend configuration (required when backend="transformerlens")
    tl_cfg: ITLensFromPretrainedConfig | ITLensCustomConfig | ITLensBridgeConfig | None = None

    # NNsight backend configuration (required when backend="nnsight")
    nnsight_cfg: "NNsightConfig | None" = None

    # SAE-specific fields
    # NOTE: TypeAlias inlined here so jsonargparse can resolve the type in CLI contexts
    # (PEP 563 deferred annotations + TypeAlias causes fail_untyped failures).
    sae_cfgs: (
        SAELensFromPretrainedConfig | SAELensCustomConfig | Sequence[SAELensFromPretrainedConfig | SAELensCustomConfig]
    )
    add_saes_on_init: bool = False  # TODO: may push this down to SAE config level instead of setting for all saes
    # use_error_term: bool = False  # TODO: add support for use_error_term with on_init stateful SAEs

    @property
    def normalized_sae_cfg_refs(self) -> list[str]:
        normalized_names = []
        # Handle both single config and list of configs
        if isinstance(self.sae_cfgs, (SAELensFromPretrainedConfig, SAELensCustomConfig)):
            sae_cfgs = [self.sae_cfgs]
        else:
            sae_cfgs = self.sae_cfgs
        for sae_cfg in sae_cfgs:
            if isinstance(sae_cfg, SAELensFromPretrainedConfig):
                normalized_names.append(sae_cfg.sae_id)
            elif isinstance(sae_cfg, SAELensCustomConfig):
                assert isinstance(sae_cfg.cfg, SAEConfig)
                normalized_names.append(sae_cfg.cfg.metadata.hook_name)
        return normalized_names

    def __post_init__(self) -> None:
        # Validate backend
        if self.backend not in _VALID_SAE_BACKENDS:
            raise ValueError(f"Invalid backend '{self.backend}'. Must be one of {_VALID_SAE_BACKENDS}")
        if self.backend != "transformerlens" and self.use_bridge:
            rank_zero_warn(
                "use_bridge=True is only meaningful when backend='transformerlens'. This setting will be ignored."
            )

        # Validate and normalize sae_cfgs (backend-agnostic)
        if not self.sae_cfgs:
            raise MisconfigurationException(
                "At least one `SAELensFromPretrainedConfig` or `SAELensCustomConfig` must be provided to "
                "initialize SAE Lens."
            )
        if isinstance(self.sae_cfgs, (SAELensFromPretrainedConfig, SAELensCustomConfig)):
            self.sae_cfgs = [self.sae_cfgs]

        # Backend-specific initialization
        if self.backend == "transformerlens":
            self._init_tl_backend()
        else:
            self._init_nnsight_backend()

    def _init_tl_backend(self) -> None:
        """Initialize TransformerLens backend configuration."""
        if not self.tl_cfg:
            raise MisconfigurationException(
                "A valid tl_cfg (ITLensFromPretrainedConfig, ITLensCustomConfig, or ITLensBridgeConfig) must be "
                "provided when backend='transformerlens'."
            )

        # Warn if use_bridge=True but tl_cfg is not ITLensBridgeConfig (likely misconfiguration)
        if (
            self.use_bridge
            and isinstance(self.tl_cfg, ITLensFromPretrainedConfig)
            and not isinstance(self.tl_cfg, ITLensBridgeConfig)
        ):
            rank_zero_warn(
                "use_bridge=True but tl_cfg is an ITLensFromPretrainedConfig (HookedTransformer config), "
                "not an ITLensBridgeConfig. This will initialize a HookedTransformer, not a TransformerBridge. "
                "To use TransformerBridge, set tl_cfg to ITLensBridgeConfig. "
                "To silence this warning, set use_bridge=False explicitly."
            )

        # Warn if nnsight_cfg is set but not used
        if self.nnsight_cfg is not None:
            rank_zero_warn("nnsight_cfg is set but backend is 'transformerlens'. This setting will be ignored.")

        # Initialize TL config state (validation, pretrained sync, config translation)
        # via TLConfigInitMixin — shared with ITLensConfig
        self._init_tl_cfg_state()

        # Call ITConfig.__post_init__ for base config setup (dtype resolution etc.)
        super().__post_init__()

        # Sync SAE device config with TL device
        self._sync_sl_tl_device_cfg()

    def _init_nnsight_backend(self) -> None:
        """Initialize NNsight backend configuration."""
        from interpretune.config.nnsight import NNsightConfig

        if not self.nnsight_cfg:
            raise MisconfigurationException(
                "A valid nnsight_cfg (NNsightConfig) must be provided when backend='nnsight'."
            )

        if not isinstance(self.nnsight_cfg, NNsightConfig):
            try:
                self.nnsight_cfg = NNsightConfig(**self.nnsight_cfg)
            except Exception as e:
                raise MisconfigurationException(
                    f"Failed to initialize NNsightConfig from provided nnsight_cfg. "
                    f"nnsight_cfg should be either a NNsightConfig instance or a dict "
                    f"convertible to one. Error: {e}"
                )

        # Warn if tl_cfg is set but not used
        if self.tl_cfg is not None:
            rank_zero_warn("tl_cfg is set but backend is 'nnsight'. This setting will be ignored.")

        # Sync model_name_or_path with nnsight_cfg.model_name
        self._sync_nnsight_model_name()

        # Set dtype from nnsight_cfg if available
        assert self.nnsight_cfg is not None  # narrowing for type checker; validated above
        if self.nnsight_cfg.resolved_dtype is not None:
            self._dtype = _resolve_dtype(self.nnsight_cfg.resolved_dtype)

        # Call ITConfig.__post_init__ for base config setup (dtype resolution etc.)
        super().__post_init__()

        # Sync SAE device config with NNsight model device
        self._sync_sl_nnsight_device_cfg()

    def _sync_sl_nnsight_device_cfg(self) -> None:
        """Sync SAE config devices with the NNsight model's target device.

        Mirrors :meth:`_sync_sl_tl_device_cfg` for the NNsight backend.
        ``SAELensFromPretrainedConfig.__post_init__`` calls ``tl_get_device()`` which may
        resolve to CUDA even when the NNsight model is on CPU.  This method corrects
        the SAE configs to match the NNsight device.
        """
        assert self.nnsight_cfg is not None
        device_map = self.nnsight_cfg.device_map
        # Resolve a simple target device string from the NNsight device_map
        if isinstance(device_map, str) and device_map not in ("auto", "balanced", "sequential"):
            target_device = device_map  # e.g. "cpu", "cuda", "cuda:0"
        else:
            # For auto / dict / None, fall back to auto-detection
            target_device = str(tl_get_device())

        if isinstance(self.sae_cfgs, (SAELensFromPretrainedConfig, SAELensCustomConfig)):
            sae_cfgs: Sequence[SAECfgType] = [self.sae_cfgs]
        else:
            sae_cfgs = self.sae_cfgs
        for sae_cfg in sae_cfgs:
            if hasattr(sae_cfg, "cfg"):
                assert isinstance(sae_cfg, SAELensCustomConfig)
                assert isinstance(sae_cfg.cfg, SAEConfig)
                setattr(sae_cfg.cfg, "device", target_device)
            else:
                assert isinstance(sae_cfg, SAELensFromPretrainedConfig)
                setattr(sae_cfg, "device", target_device)

    def _sync_nnsight_model_name(self) -> None:
        """Synchronize model_name_or_path with nnsight_cfg.model_name."""
        assert self.nnsight_cfg is not None  # validated in _init_nnsight_backend
        it_model = self.model_name_or_path
        ns_model = self.nnsight_cfg.model_name

        if not it_model and not ns_model:
            raise MisconfigurationException("Either model_name_or_path or nnsight_cfg.model_name must be provided.")

        if not it_model and ns_model:
            self.model_name_or_path = ns_model
        elif it_model and not ns_model:
            self.nnsight_cfg.model_name = it_model
        elif it_model != ns_model:
            rank_zero_warn(
                f"model_name_or_path ('{it_model}') differs from nnsight_cfg.model_name ('{ns_model}'). "
                f"Using model_name_or_path. Set nnsight_cfg.model_name=None to silence this warning."
            )
            self.nnsight_cfg.model_name = it_model

    def _sync_sl_tl_device_cfg(self):
        assert isinstance(self.tl_cfg, ITLensCfgTypes)
        if hasattr(self.tl_cfg, "cfg"):  # TODO: consider reverting this to ternary assignment w/ type check directives
            assert isinstance(self.tl_cfg, ITLensCustomConfig)
            assert isinstance(self.tl_cfg.cfg, HookedTransformerConfig)
            tl_device = self.tl_cfg.cfg.device
        elif isinstance(self.tl_cfg, ITLensBridgeConfig):
            tl_device = self.tl_cfg.device
        else:
            assert isinstance(self.tl_cfg, ITLensFromPretrainedConfig)
            tl_device = self.tl_cfg.device
        # Handle both single config and list of configs
        if isinstance(self.sae_cfgs, (SAELensFromPretrainedConfig, SAELensCustomConfig)):
            sae_cfgs = [self.sae_cfgs]
        else:
            sae_cfgs = self.sae_cfgs
        for sae_cfg in sae_cfgs:
            if hasattr(sae_cfg, "cfg"):
                assert isinstance(sae_cfg, SAELensCustomConfig)
                assert isinstance(sae_cfg.cfg, SAEConfig)
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg.cfg, tl_device=str(tl_device))
            else:
                assert isinstance(sae_cfg, SAELensFromPretrainedConfig)
                self._sync_sl_tl_default_device(sae_cfg_obj=sae_cfg, tl_device=str(tl_device))

    def _sync_sl_tl_default_device(self, sae_cfg_obj: SAELensFromPretrainedConfig | SAEConfig, tl_device):
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
