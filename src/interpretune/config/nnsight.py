"""NNsight configuration for interpretune integration.

This module provides configuration dataclasses for NNsight model integration within interpretune's adapter composition
system.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

import torch

from interpretune.config.shared import ITSerializableCfg
from interpretune.config.module import ITConfig
from interpretune.utils import rank_zero_warn, MisconfigurationException


@dataclass(kw_only=True)
class NNsightConfig(ITSerializableCfg):
    """Configuration for NNsight model integration.

    NNsight wraps HuggingFace models directly without weight conversion, providing
    tracing-based activation access and intervention capabilities.

    This config provides IT-specific settings that control how NNsight's LanguageModel
    is initialized and configured. Most kwargs are passed through to the underlying
    HuggingFace `from_pretrained` call.

    Attributes:
        model_name: The model name or path for loading (e.g., "openai-community/gpt2").
            If None, uses the base model name from ITConfig.model_name_or_path.
        device_map: Device mapping strategy.
            Options: "auto", "cpu", "cuda:0", or dict mapping.
        torch_dtype: Data type for model weights (e.g., "float32", "float16", "bfloat16").
        dispatch: Whether to load model immediately (True) or defer loading (False).
            Default True for typical usage.
        tokenizer_kwargs: Additional kwargs passed to tokenizer initialization.
        trust_remote_code: Whether to trust remote code for custom model architectures.
        attn_implementation: Attention implementation to use (e.g., "flash_attention_2").

    Example:
        Basic GPT-2 configuration:

        >>> config = NNsightConfig(model_name="openai-community/gpt2")

        Larger model with precision settings:

        >>> config = NNsightConfig(
        ...     model_name="meta-llama/Llama-3.1-8B",
        ...     device_map="auto",
        ...     torch_dtype="bfloat16",
        ...     trust_remote_code=True,
        ... )
    """

    # Model loading settings
    model_name: str | None = "openai-community/gpt2"
    """Model name or path for loading.

    If None, uses ITConfig.model_name_or_path.
    """

    device_map: str | dict[str, Any] | None = None
    """Device mapping strategy for HuggingFace Accelerate.

    Common options:
        - "auto": Automatically distribute across available devices
        - "cpu": Force CPU execution
        - "cuda:0": Specific CUDA device
        - dict: Custom layer-to-device mapping
    """

    torch_dtype: str | torch.dtype | None = "float32"
    """Data type for model weights.

    Accepts string ("float32", "bfloat16") or torch.dtype.
    """

    dispatch: bool = True
    """Whether to load model immediately (True) or defer (False).

    When True (default), the model is loaded into memory immediately upon initialization. When False, model loading is
    deferred until first use (useful for lazy initialization).
    """

    # Tokenizer settings
    tokenizer_kwargs: dict[str, Any] | None = None
    """Additional kwargs passed to AutoTokenizer.from_pretrained().

    Example:
        >>> config = NNsightConfig(
        ...     model_name="gpt2",
        ...     tokenizer_kwargs={"padding_side": "left", "add_bos_token": True}
        ... )
    """

    # HuggingFace model loading settings (passed through to from_pretrained)
    trust_remote_code: bool = False
    """Whether to trust remote code for custom model architectures.

    Required for some models that use custom code not in the transformers library.
    """

    attn_implementation: str | None = None
    """Attention implementation to use.

    Options include:
        - None: Use default implementation
        - "flash_attention_2": Use Flash Attention 2 (requires compatible hardware)
        - "sdpa": Use PyTorch's scaled dot product attention
    """

    # NNsight-specific runtime settings
    default_padding_side: Literal["left", "right"] | None = "left"
    """Default padding side for tokenization.

    NNsight defaults to left padding.
    """

    # Remote execution settings (NDIF)
    remote: bool = False
    """Whether to use NNsight's remote execution via NDIF.

    When True, model execution happens on NDIF servers rather than locally. Requires NDIF_API_KEY environment variable
    to be set.
    """

    api_key: str | None = None
    """API key for NDIF remote execution.

    If not provided, will be read from NDIF_API_KEY environment variable. Only used when remote=True.
    """

    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization."""
        # Normalize torch_dtype if string provided
        if isinstance(self.torch_dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float64": torch.float64,
            }
            if self.torch_dtype in dtype_map:
                self._resolved_dtype = dtype_map[self.torch_dtype]
            else:
                rank_zero_warn(f"Unknown dtype string '{self.torch_dtype}', will pass as-is to NNsight")
                self._resolved_dtype = self.torch_dtype
        else:
            self._resolved_dtype = self.torch_dtype

        # Initialize tokenizer_kwargs with defaults if not provided
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {}

        # Apply default padding side to tokenizer kwargs if not explicitly set
        if self.default_padding_side and "padding_side" not in self.tokenizer_kwargs:
            self.tokenizer_kwargs["padding_side"] = self.default_padding_side

    @property
    def resolved_dtype(self) -> torch.dtype | str | None:
        """Get the resolved torch dtype."""
        return getattr(self, "_resolved_dtype", self.torch_dtype)

    def get_nnsight_kwargs(self) -> dict[str, Any]:
        """Generate kwargs dict for NNsight LanguageModel initialization.

        Returns:
            Dictionary of kwargs to pass to LanguageModel constructor.
        """
        kwargs: dict[str, Any] = {"dispatch": self.dispatch}

        if self.device_map is not None:
            kwargs["device_map"] = self.device_map

        if self.resolved_dtype is not None:
            kwargs["torch_dtype"] = self.resolved_dtype

        if self.trust_remote_code:
            kwargs["trust_remote_code"] = self.trust_remote_code

        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        if self.tokenizer_kwargs:
            kwargs["tokenizer_kwargs"] = self.tokenizer_kwargs

        # Remote execution settings
        if self.remote:
            kwargs["remote"] = True
            # API key is handled separately in adapter initialization
            # to allow for environment variable fallback

        return kwargs


# Type aliases for static typing
NNsightCfg = NNsightConfig  # Alias for consistency with other adapters (e.g., ITLensCfg)
NNsightCfgTypes: tuple[type] = (NNsightConfig,)  # For runtime checks


@dataclass(kw_only=True)
class ITNNsightConfig(ITConfig):
    """Dataclass to encapsulate ITModule configuration for NNsight models.

    This class extends ITConfig to provide NNsight-specific model initialization
    and configuration handling. Similar to ITLensConfig for TransformerLens,
    ITNNsightConfig manages the integration between Interpretune's module system
    and NNsight's LanguageModel wrapper.

    NNsight wraps HuggingFace models directly without weight conversion, providing
    tracing-based activation access for analysis and intervention.

    Attributes:
        nnsight_cfg: Configuration for NNsight model initialization.

    Example:
        Basic GPT-2 configuration:

        >>> config = ITNNsightConfig(
        ...     model_name_or_path="openai-community/gpt2",
        ...     nnsight_cfg=NNsightConfig(model_name="openai-community/gpt2"),
        ... )

        Using registry-based configuration::

            from interpretune import ITSession
            from interpretune.protocol import Adapter
            session = ITSession(model_src_key="gpt2", adapter_ctx=(Adapter.core, Adapter.nnsight))
    """

    nnsight_cfg: NNsightConfig

    def __post_init__(self) -> None:
        """Validate and configure NNsight-specific settings."""
        if not self.nnsight_cfg:
            raise MisconfigurationException(
                "A valid nnsight_cfg (NNsightConfig) must be provided to initialize an NNsight "
                "LanguageModel and use the NNsight adapter."
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

        # Note: We don't require hf_from_pretrained_cfg for NNsight because:
        # 1. NNsight handles HF model loading internally via LanguageModel(model_name, **kwargs)
        # 2. dtype is resolved from nnsight_cfg.resolved_dtype (set below)
        # 3. The base ITConfig.__post_init__() only uses hf_from_pretrained_cfg for dtype resolution
        #    which we handle explicitly here

        # Sync model_name_or_path with nnsight_cfg.model_name if not explicitly set
        self._sync_nnsight_model_name()

        # Set dtype from nnsight_cfg if available (must be before super().__post_init__())
        if self.nnsight_cfg.resolved_dtype is not None:
            self._dtype = self.nnsight_cfg.resolved_dtype

        super().__post_init__()

    def _sync_nnsight_model_name(self) -> None:
        """Synchronize model_name_or_path with nnsight_cfg.model_name.

        If model_name_or_path is not set, use nnsight_cfg.model_name. If nnsight_cfg.model_name is not set, use
        model_name_or_path. Warns if both are set to different values.
        """
        it_model = self.model_name_or_path
        ns_model = self.nnsight_cfg.model_name

        if not it_model and not ns_model:
            raise MisconfigurationException("Either model_name_or_path or nnsight_cfg.model_name must be provided.")

        if not it_model and ns_model:
            # Use nnsight_cfg.model_name for ITConfig
            self.model_name_or_path = ns_model
        elif it_model and not ns_model:
            # Use model_name_or_path for nnsight_cfg
            self.nnsight_cfg.model_name = it_model
        elif it_model != ns_model:
            # Both set but different - warn and use model_name_or_path
            rank_zero_warn(
                f"model_name_or_path ('{it_model}') differs from nnsight_cfg.model_name ('{ns_model}'). "
                f"Using model_name_or_path. Set nnsight_cfg.model_name=None to silence this warning."
            )
            self.nnsight_cfg.model_name = it_model
