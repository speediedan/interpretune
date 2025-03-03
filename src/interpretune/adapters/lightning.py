from typing_extensions import override

from interpretune.base import ITDataModule, BaseITModule
from interpretune.utils import  _LIGHTNING_AVAILABLE
from interpretune.protocol import Adapter
from interpretune.adapters import CompositionRegistry


if _LIGHTNING_AVAILABLE:
    from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
    from lightning.pytorch import LightningDataModule, LightningModule
    class LightningAdapter:
        CORE_TO_FRAMEWORK_ATTRS_MAP = {
            "_it_lr_scheduler_configs": ("trainer.strategy.lr_scheduler_configs", None,
                                        "No lr_scheduler_configs have been set."),
            "_it_optimizers": ("trainer.optimizers", None, "No optimizers have been set yet."),
            "_log_dir": ("trainer.model._trainer.log_dir", None, "No log_dir has been set yet."),
            "_datamodule": ("trainer.datamodule", None,
                            "Could not find datamodule reference (has it been attached yet?)"),
            "_current_epoch": ("trainer.current_epoch", 0, ""),
            "_global_step": ("trainer.global_step", 0, ""),
        }

        PROPERTY_COMPOSITION = {
            # property composition is by default only enabled for `device` if Lightning is available and only effective
            # if Lightning actively being used.
             "device": {"enabled": True, "target": _DeviceDtypeModuleMixin,
                        "dispatch": _DeviceDtypeModuleMixin.device}
                        }

        def on_train_start(self) -> None:
            # ensure model is in training mode (e.g. needed for some edge cases w/ skipped sanity checking)
            self.model.train()
            return super().on_train_start()

        @classmethod
        @override
        def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
            adapter_ctx_registry.register(Adapter.lightning, component_key = "datamodule",
                                          adapter_combination=(Adapter.lightning,),
                                          composition_classes=(ITDataModule, LightningDataModule),
                                          description="lightning adapter to be used with lightning",
            )
            adapter_ctx_registry.register(Adapter.lightning, component_key = "module",
                                          adapter_combination=(Adapter.lightning,),
                                          composition_classes=(LightningAdapter, BaseITModule, LightningModule),
                                          description="lightning adapter to be used with lightning",
            )
else:
    LightningDataModule = object
    LightningModule = object
    LightningAdapter = object
