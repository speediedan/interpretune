"""NNsight adapter for interpretune integration.

This module provides the NNsight adapter that enables interpretune to use NNsight's
LanguageModel for model wrapping, tracing, and intervention capabilities.

NNsight wraps HuggingFace models directly without weight conversion, providing:
- Tracing-based activation access via context managers
- In-place activation modification
- Generation with intervention support
- Native HuggingFace ecosystem compatibility
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from copy import deepcopy
import os

import torch
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.adapters import CompositionRegistry, LightningDataModule, LightningModule, LightningAdapter
from interpretune.base import CoreHelperAttributes, ITDataModule, BaseITModule
from interpretune.utils import move_data_to_device, rank_zero_warn, rank_zero_info, rank_zero_debug
from interpretune.protocol import Adapter

if TYPE_CHECKING:
    from nnsight import LanguageModel
    from interpretune.config.nnsight import NNsightConfig, ITNNsightConfig


################################################################################
# Mixins to support NNsight in different adapter contexts
################################################################################


class NNsightAttributeMixin:
    """Mixin providing property access for NNsight-specific attributes.

    Provides consistent attribute access patterns for NNsight models similar to TLensAttributeMixin for TransformerLens
    models.
    """

    it_cfg: ITNNsightConfig  # Provided by mixing class
    model: LanguageModel | None  # Provided by mixing class

    @property
    def nnsight_cfg(self) -> NNsightConfig | None:
        """Get NNsight configuration from ITConfig."""
        if hasattr(self.it_cfg, "nnsight_cfg"):
            return self.it_cfg.nnsight_cfg
        return None

    @property
    def nnsight_model(self) -> LanguageModel | None:
        """Get the underlying NNsight LanguageModel."""
        if hasattr(self, "model") and self.model is not None:
            from nnsight import LanguageModel

            if isinstance(self.model, LanguageModel):
                return self.model
        return None

    @property
    def device(self) -> torch.device | None:
        """Get the device from NNsight model or IT state.

        NNsight uses HuggingFace's device_map, so we try to infer the device from the underlying model's parameters.
        """
        device: torch.device | None = None
        try:
            # First check IT state
            device = getattr(self._it_state, "_device", None)  # type: ignore[attr-defined]
            if device is not None:
                return device

            # Try to get from model parameters
            if self.nnsight_model is not None:
                # NNsight's _model attribute holds the actual PyTorch model
                underlying_model = getattr(self.nnsight_model, "_model", None)
                if underlying_model is not None:
                    try:
                        # Get device from first parameter
                        param = next(underlying_model.parameters())
                        device = param.device
                    except StopIteration:
                        pass
        except AttributeError as ae:
            rank_zero_warn(f"Could not find a device reference (has it been set yet?): {ae}")
            device = None
        return device

    @device.setter
    def device(self, value: str | torch.device | None) -> None:
        if value is not None and not isinstance(value, torch.device):
            value = torch.device(value)
        self._it_state._device = value  # type: ignore[attr-defined]

    @property
    def output_device(self) -> torch.device | None:
        """Get the output device (same as device for NNsight)."""
        return self.device

    @property
    def input_device(self) -> torch.device | None:
        """Get the input device (same as device for NNsight)."""
        return self.device

    def batch_to_device(self, batch: BatchEncoding) -> BatchEncoding:
        """Move a batch to the NNsight input device if one is available.

        Implemented on the NNsight mixin so that any composition that exposes NNsight properties
        (e.g., `input_device`) can reuse consistent behavior.
        """
        device = self.input_device
        if device is not None:
            move_data_to_device(batch, device)
        return batch


class BaseNNsightModule(BaseITModule):
    """Base module for NNsight integration.

    Provides NNsight model initialization and configuration handling. NNsight wraps HuggingFace models directly without
    weight conversion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = None

    def auto_model_init(self) -> None:
        """Initialize model using NNsight LanguageModel.

        NNsight wraps HuggingFace models directly, so we use its initialization path rather than a separate HF -> TL
        conversion.
        """
        if self.it_cfg.nnsight_cfg is not None:
            self._init_nnsight_model()
        else:
            rank_zero_warn("No nnsight_cfg found, falling back to base model_init")
            self.model_init()

    def _init_nnsight_model(self) -> None:
        """Initialize the NNsight LanguageModel.

        Uses NNsight's LanguageModel which handles:
        - HuggingFace model loading
        - Tokenizer initialization
        - Device placement
        """
        from nnsight import LanguageModel

        nnsight_cfg = self.it_cfg.nnsight_cfg
        if nnsight_cfg is None:
            raise ValueError("nnsight_cfg must be set for NNsight model initialization")

        # Determine model name - prefer nnsight_cfg.model_name, fall back to it_cfg.model_name_or_path
        model_name = nnsight_cfg.model_name or self.it_cfg.model_name_or_path
        if model_name is None:
            raise ValueError("model_name must be specified in nnsight_cfg or model_name_or_path in it_cfg")

        rank_zero_info(f"Initializing NNsight LanguageModel with model: {model_name}")

        # Get NNsight initialization kwargs
        nnsight_kwargs = nnsight_cfg.get_nnsight_kwargs()

        # Handle authentication token for model access
        if self.it_cfg.os_env_model_auth_key:
            access_token = os.environ.get(self.it_cfg.os_env_model_auth_key.upper())
            if access_token:
                nnsight_kwargs["token"] = access_token
                rank_zero_debug("Using authentication token from environment for NNsight model")

        # Handle NDIF API key for remote execution
        if nnsight_cfg.remote:
            api_key = nnsight_cfg.api_key or os.environ.get("NDIF_API_KEY")
            if api_key:
                nnsight_kwargs["api_key"] = api_key
                rank_zero_debug("Using NDIF_API_KEY for remote execution")
            else:
                rank_zero_warn(
                    "NNsight remote execution enabled but no API key found. "
                    "Set NDIF_API_KEY environment variable or provide api_key in nnsight_cfg."
                )

        # Initialize NNsight LanguageModel
        rank_zero_debug(f"NNsight kwargs: {nnsight_kwargs}")
        self.model = LanguageModel(model_name, **nnsight_kwargs)

        # Store reference to tokenizer if not already set
        if self.it_cfg.tokenizer is None and hasattr(self.model, "tokenizer"):
            self.it_cfg.tokenizer = self.model.tokenizer

        rank_zero_info("NNsight LanguageModel initialized successfully")

    def _capture_hyperparameters(self) -> None:
        """Capture hyperparameters for logging.

        Serializes:
        1. NNsight configuration
        2. Model configuration (from underlying HF model)
        """
        # Add NNsight config to hyperparameters
        if self.it_cfg.nnsight_cfg is not None:
            self._it_state._init_hparams.update({"nnsight_cfg": deepcopy(self.it_cfg.nnsight_cfg)})

        # Try to capture underlying HF config
        if self.nnsight_model is not None:
            try:
                underlying_model = getattr(self.nnsight_model, "_model", None)
                if underlying_model is not None and hasattr(underlying_model, "config"):
                    hf_config = deepcopy(underlying_model.config)
                    self._it_state._init_hparams.update({"hf_model_config": hf_config})
            except Exception as e:
                rank_zero_warn(f"Could not capture HF model config: {e}")

        # Call superclass to capture base hyperparameters
        super()._capture_hyperparameters()

    def set_input_require_grads(self) -> None:
        """Set input gradients for NNsight models."""
        # NNsight handles gradient requirements through its tracing mechanism
        rank_zero_info("Input gradient requirements handled by NNsight tracing mechanism.")


################################################################################
# NNsight Module Composition
################################################################################


class NNsightAdapter(NNsightAttributeMixin):
    """NNsight adapter for registration and composition."""

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        """Register NNsight adapter combinations.

        Registers:
        - (core, nnsight): Basic NNsight support
        - (lightning, nnsight): NNsight with Lightning integration
        """
        # ======================================================================
        # Core + NNsight registrations
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.nnsight,
            component_key="datamodule",
            adapter_combination=(Adapter.core, Adapter.nnsight),  # type: ignore[arg-type]
            composition_classes=(ITDataModule,),
            description="NNsight adapter that can be composed with core for basic NNsight model support.",
        )
        adapter_ctx_registry.register(
            Adapter.nnsight,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.nnsight),  # type: ignore[arg-type]
            composition_classes=(NNsightModule,),
            description="NNsight adapter that can be composed with core for basic NNsight model support.",
        )

        # ======================================================================
        # Lightning + NNsight registrations
        # ======================================================================
        adapter_ctx_registry.register(
            Adapter.nnsight,
            component_key="datamodule",
            adapter_combination=(Adapter.lightning, Adapter.nnsight),  # type: ignore[arg-type]
            composition_classes=(ITDataModule, LightningDataModule),
            description="NNsight adapter that can be composed with Lightning for NNsight model support.",
        )
        adapter_ctx_registry.register(
            Adapter.nnsight,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.nnsight),  # type: ignore[arg-type]
            composition_classes=(
                NNsightAttributeMixin,
                BaseNNsightModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
            description="NNsight adapter that can be composed with Lightning for NNsight model support.",
        )

    def batch_to_device(self, batch: BatchEncoding) -> BatchEncoding:
        """Move batch to device."""
        device = self.input_device
        if device is not None:
            move_data_to_device(batch, device)
        return batch


class NNsightModule(NNsightAdapter, CoreHelperAttributes, BaseNNsightModule):  # type: ignore[misc]
    """NNsight module combining adapter, core helpers, and base functionality.

    This is the composed module class for (core, nnsight) adapter combination.
    """

    ...
