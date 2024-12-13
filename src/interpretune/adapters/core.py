from typing_extensions import override

from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.modules import BaseITModule
from interpretune.base.datamodules import ITDataModule
from interpretune.base.config.shared import Adapter
from interpretune.adapters.registration import CompositionRegistry
from interpretune.base.components.core import CoreHelperAttributes
from interpretune.utils.data_movement import to_device


class CoreAdapter(CoreHelperAttributes):
    @classmethod
    @override
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        adapter_ctx_registry.register(
                Adapter.core,
                component_key = "datamodule",
                adapter_combination=(Adapter.core,),
                composition_classes=(ITDataModule,),
                description="core adapter to be used with native PyTorch",
            )
        adapter_ctx_registry.register(
                Adapter.core,
                component_key = "module",
                adapter_combination=(Adapter.core,),
                composition_classes=(ITModule,),
                description="core adapter to be used with native PyTorch",
            )

    def batch_to_device(self, batch) -> BatchEncoding:
        # TODO: switch to move_data_to_device
        #move_data_to_device(batch, self.device)
        to_device(self.device, batch)
        return batch

class ITModule(CoreAdapter, BaseITModule):
    ...
