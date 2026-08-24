from __future__ import annotations
from typing import TYPE_CHECKING

from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base import CoreHelperAttributes, BaseITModule, ITDataModule
from interpretune.utils import to_device
from interpretune.protocol import Adapter

if TYPE_CHECKING:
    from interpretune.adapters import CompositionRegistry


class CoreAdapter(CoreHelperAttributes):
    """The framework-free adapter: native PyTorch, no external training framework."""

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        """Register the core-only datamodule and module compositions."""
        adapter_ctx_registry.register(
            Adapter.core,
            component_key="datamodule",
            adapter_combination=(Adapter.core,),
            composition_classes=(ITDataModule,),
            description="core adapter to be used with native PyTorch",
        )
        adapter_ctx_registry.register(
            Adapter.core,
            component_key="module",
            adapter_combination=(Adapter.core,),
            composition_classes=(ITModule,),
            description="core adapter to be used with native PyTorch",
        )

    def batch_to_device(self, batch: BatchEncoding) -> BatchEncoding:
        """Move a batch to this module's device, in place, and return it."""
        # TODO: switch to move_data_to_device
        # move_data_to_device(batch, self.device)
        to_device(self.device, batch)
        return batch


class ITModule(CoreAdapter, BaseITModule):
    """The core-adapter module composition: :class:`CoreAdapter` over ``BaseITModule``."""
