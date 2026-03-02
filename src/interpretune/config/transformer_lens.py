from typing import Literal, Any, TypeAlias
from dataclasses import dataclass
from functools import reduce

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformer_lens.config import HookedTransformerConfig
from transformer_lens.utilities import get_device as tl_get_device

from interpretune.config import ITConfig, HFFromPretrainedConfig, CoreGenerationConfig, ITSerializableCfg
from interpretune.utils import _resolve_dtype, tl_invalid_dmap, rank_zero_warn, MisconfigurationException

################################################################################
# TransformerLens Configuration Encapsulation
################################################################################


@dataclass(kw_only=True)
class ITLensSharedConfig(ITSerializableCfg):
    """TransformerLens configuration shared across both `from_pretrained` and config based instantiation modes."""

    move_to_device: bool | None = True
    default_padding_side: Literal["left", "right"] | None = "right"
    use_bridge: bool | None = True  # Use TransformerBridge (v3) by default, set False for legacy HookedTransformer


@dataclass(kw_only=True)
class ITLensBridgeConfig(ITLensSharedConfig):
    """TransformerBridge-specific configuration for TransformerLens v3 integration.

    This config provides explicit control over TransformerBridge initialization and
    compatibility mode settings. Use this config when you need fine-grained control
    over how the bridge is configured (e.g., enabling/disabling weight processing).

    Args:
        model_name: The model name for TransformerBridge (passed to TransformerBridgeConfig).
        transformer_bridge_config_overrides: Optional dict of kwargs to pass to/override
            in the TransformerBridgeConfig constructor. Use this to set device, dtype,
            or any other TransformerBridgeConfig fields.
        enable_compatibility_mode: Whether to call enable_compatibility_mode() on the
            TransformerBridge after instantiation. Default: False.
        enable_compatibility_mode_kwargs: Optional dict of kwargs for enable_compatibility_mode().
            Supported kwargs:
            - disable_warnings: bool (default False)
            - no_processing: bool (default False) - disables ALL weight processing
            - fold_ln: bool (default True) - fold LayerNorm weights
            - center_writing_weights: bool (default True)
            - center_unembed: bool (default True)
            - fold_value_biases: bool (default True)
            - refactor_factored_attn_matrices: bool (default False)

    Example:
        # Basic bridge config (no processing, current default behavior)
        config = ITLensBridgeConfig(model_name="gpt2-small")

        # Enable weight processing via compatibility mode
        config = ITLensBridgeConfig(
            model_name="gpt2-small",
            enable_compatibility_mode=True,
            enable_compatibility_mode_kwargs={"fold_ln": True, "fold_value_biases": True}
        )

        # Use no_processing (analogous to ITLensFromPretrainedNoProcessingConfig behavior)
        config = ITLensBridgeConfig(
            model_name="gpt2-small",
            enable_compatibility_mode=True,
            enable_compatibility_mode_kwargs={"no_processing": True}
        )
    """

    # The model name/path for TransformerBridge - IT handles HF model instantiation via model_name_or_path
    model_name: str = "gpt2-small"
    # Optional kwargs to pass to TransformerBridgeConfig constructor
    transformer_bridge_config_overrides: dict[str, Any] | None = None
    # Whether to call enable_compatibility_mode on the bridge after instantiation
    # N.B.: See transformer_lens/model_bridge/bridge.py for details, among other things, this mode:
    # 1. Breaks weight tying between embed and unembed to allow separate unembed centering
    # 2. Extracts q/k/v from joint qkv matrices for compatibility with HookedTransformer parameterizations
    enable_compatibility_mode: bool = False
    # Optional kwargs for enable_compatibility_mode()
    enable_compatibility_mode_kwargs: dict[str, Any] | None = None
    # Bridge config defaults to using bridge
    use_bridge: bool | None = True
    # Device is commonly set, so we provide a top-level field for convenience
    device: str | None = None
    # Dtype is commonly set, so we provide a top-level field for convenience
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.device is None:  # align with TL default device resolution
            self.device = tl_get_device()  # type: ignore
        # Validate that enable_compatibility_mode_kwargs are only set when enable_compatibility_mode is True
        if self.enable_compatibility_mode_kwargs and not self.enable_compatibility_mode:
            rank_zero_warn(
                "enable_compatibility_mode_kwargs was provided but enable_compatibility_mode is False. "
                "The kwargs will be ignored. Set enable_compatibility_mode=True to use them."
            )


# TODO: open a PR to have TL `from_pretrained` config encapsulated in a dataclass for improved external compatibility
@dataclass(kw_only=True)
class ITLensFromPretrainedConfig(ITLensSharedConfig):
    model_name: str = "gpt2-small"
    fold_ln: bool | None = True
    center_writing_weights: bool | None = True
    center_unembed: bool | None = True
    refactor_factored_attn_matrices: bool | None = False
    checkpoint_index: int | None = None
    checkpoint_value: int | None = None
    # for pretrained cfg, IT handles the HF model instantiation via model_name or_path
    hf_model: AutoModelForCausalLM | str | None = None
    # currently only annotating with str due to omegaconf container dumping limitations wrt torch.device
    device: str | None = None
    n_devices: int | None = 1
    # IT handles the tokenizer instantiation via either tokenizer, tokenizer_name or model_name_or_path
    tokenizer: PreTrainedTokenizerBase | None = None  # for pretrained cfg, IT instantiates the tokenizer
    fold_value_biases: bool | None = True
    default_prepend_bos: bool | None = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.device is None:  # align with TL default device resolution
            self.device = tl_get_device()  # type: ignore


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
    """Custom TL config for creating a HookedTransformer from a TL config.

    NOTE: TransformerBridge is not supported with config-only initialization.
    Set `use_bridge=False` (default) or interpretune will force the value to False and warn.
    """

    cfg: HookedTransformerConfig | dict[str, Any]
    # When using a custom config, default to legacy HookedTransformer behavior to prevent
    # misconfiguration. If the user explicitly sets `use_bridge=True`, Interpretune will
    # warn and force it to False in `ITLensConfig.__post_init__`.
    use_bridge: bool | None = False

    # IT handles the tokenizer instantiation via either tokenizer, tokenizer_name or model_name_or_path
    # tokenizer: PreTrainedTokenizerBase | None = None
    def __post_init__(self) -> None:
        if not isinstance(self.cfg, HookedTransformerConfig):
            # ensure the user provided a valid dtype (should be handled by HookedTransformerConfig ideally)
            if self.cfg.get("dtype", None) and not isinstance(self.cfg["dtype"], torch.dtype):
                self.cfg["dtype"] = _resolve_dtype(self.cfg["dtype"])
            self.cfg = HookedTransformerConfig.from_dict(self.cfg)


ITLensCfg: TypeAlias = ITLensFromPretrainedConfig | ITLensCustomConfig | ITLensBridgeConfig  # for static typing
ITLensCfgTypes: tuple[type, type, type] = (
    ITLensFromPretrainedConfig,
    ITLensCustomConfig,
    ITLensBridgeConfig,
)  # for runtime checks


class TLConfigInitMixin:
    """Mixin providing TransformerLens config initialization logic.

    Shared by :class:`ITLensConfig` (pure TL config) and :class:`SAELensConfig` (multi-backend config
    that delegates to TL when ``backend="transformerlens"``).  Separating these helpers avoids code
    duplication without requiring SAELensConfig to inherit from ITLensConfig.

    Note: This mixin always co-inherits with :class:`ITConfig`, which provides
    ``hf_from_pretrained_cfg``, ``model_name_or_path``, ``tokenizer_kwargs``, etc.
    Only attributes unique to TL initialization are declared here.
    """

    # Attributes unique to TL initialization (not provided by ITConfig).
    # ``tl_cfg`` is typed as ``Any`` to avoid invariance conflicts — consuming
    # dataclasses narrow the type (non-optional on ITLensConfig, optional on
    # SAELensConfig).
    tl_cfg: Any
    _load_from_pretrained: bool
    _dtype: torch.dtype | None

    # ------------------------------------------------------------------
    # Core TL config state initialization (called by __post_init__ or
    # backend-specific init methods on consuming classes).
    # ------------------------------------------------------------------

    def _init_tl_cfg_state(self) -> None:
        """Validate ``tl_cfg`` and initialize TL-specific state.

        This method encapsulates the logic that was previously in ``ITLensConfig.__post_init__``
        (minus the final ``super().__post_init__()`` call, which remains the caller's responsibility
        so each consuming class can control its own MRO chain).
        """
        if not self.tl_cfg:
            raise MisconfigurationException(
                "A valid tl_cfg (ITLensFromPretrainedConfig, ITLensCustomConfig, or ITLensBridgeConfig) must be"
                " provided to initialize a HookedTransformer/TransformerBridge and use TransformerLens."
            )
        # internal variable used to bootstrap model initialization mode (we may need to override hf_from_pretrained_cfg)
        # ITLensBridgeConfig is a pretrained mode config (like ITLensFromPretrainedConfig)
        self._load_from_pretrained = not isinstance(self.tl_cfg, ITLensCustomConfig)
        if not self._load_from_pretrained:
            # If a custom config was provided, TransformerBridge (v3) cannot be used because it requires an HF model.
            # Default to legacy HookedTransformer (use_bridge=False) for custom configs. If the user explicitly
            # set `use_bridge=True`, warn and force it to False so the session doesn't fail unexpectedly.
            if getattr(self.tl_cfg, "use_bridge", False):
                rank_zero_warn(
                    "ITLensCustomConfig does not support TransformerBridge (use_bridge=True); "
                    "forcing `use_bridge=False` and falling back to HookedTransformer."
                )
                # Make sure downstream logic sees the intended value
                self.tl_cfg.use_bridge = False
            self._disable_pretrained_model_mode()  # after this, hf_from_pretrained_cfg exists only if used
            assert isinstance(self.tl_cfg, ITLensCustomConfig)
            assert isinstance(self.tl_cfg.cfg, HookedTransformerConfig)
            self._dtype = _resolve_dtype(self.tl_cfg.cfg.dtype)
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

    # ------------------------------------------------------------------
    # TL config helper methods
    # ------------------------------------------------------------------

    def _map_tl_fallback(self, target_key: str, tl_cfg_key: str):
        qual_sub_key = tl_cfg_key.split(".")
        fallback_val = reduce(getattr, qual_sub_key, self)
        if fallback_val not in [None, "custom"]:
            existing_val = getattr(self, target_key, None)
            # Treat both None and empty string as "not provided"
            if existing_val is not None and existing_val != "":
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
            # Only ITLensFromPretrainedConfig and variants have an `hf_model` field that maps to model_name_or_path,
            # ITLensBridgeConfig natively uses `model_name` for HF model instantiation (TL registry aliases for models
            # have been deprecated in favor of HF model names/paths for bridge mode).
            if hasattr(self.tl_cfg, "hf_model"):
                self._map_tl_fallback(target_key="model_name_or_path", tl_cfg_key="tl_cfg.hf_model")
            # Only ITLensFromPretrainedConfig has tokenizer field; ITLensBridgeConfig doesn't have it
            if hasattr(self.tl_cfg, "tokenizer"):
                self._map_tl_fallback(target_key="tokenizer", tl_cfg_key="tl_cfg.tokenizer")
        else:
            self._map_tl_fallback(target_key="model_name_or_path", tl_cfg_key="tl_cfg.cfg.model_name")
            self._map_tl_fallback(target_key="tokenizer_name", tl_cfg_key="tl_cfg.cfg.tokenizer_name")

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
            if hf_dtype := self.hf_from_pretrained_cfg.pretrained_kwargs.get("dtype", None):
                hf_dtype = _resolve_dtype(hf_dtype)
            # Both ITLensFromPretrainedConfig and ITLensBridgeConfig have dtype attribute
            assert isinstance(self.tl_cfg, (ITLensFromPretrainedConfig, ITLensBridgeConfig))
            tl_dtype = _resolve_dtype(self.tl_cfg.dtype)
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
                self.hf_from_pretrained_cfg.pretrained_kwargs["dtype"] = tl_dtype
        else:
            rank_zero_warn(
                "HF `from_pretrained` dtype was not provided. Setting `from_pretrained` dtype to match"
                f" specified TL dtype: {tl_dtype}."
            )
            self.hf_from_pretrained_cfg.pretrained_kwargs["dtype"] = tl_dtype


@dataclass(kw_only=True)
class ITLensConfig(ITConfig, TLConfigInitMixin):
    """Dataclass to encapsulate the ITModule internal state."""

    tl_cfg: ITLensFromPretrainedConfig | ITLensCustomConfig | ITLensBridgeConfig

    def __post_init__(self) -> None:
        self._init_tl_cfg_state()
        super().__post_init__()


# TODO: we should be able to standardize on the HF GenerationConfig interface and remove TLensGenerationConfig once
#       once (if) TL migrates away from HookedTransformer.generate method to using the HF generate interface
@dataclass(kw_only=True)
class TLensGenerationConfig(CoreGenerationConfig):
    stop_at_eos: bool = True
    eos_token_id: int | None = None
    freq_penalty: float = 0.0
    use_past_kv_cache: bool = True
    prepend_bos: bool | None = None
    padding_side: Literal["left", "right"] | None = None
    return_type: str | None = "input"
    output_logits: bool | None = None
    verbose: bool = True
