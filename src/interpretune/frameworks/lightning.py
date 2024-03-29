from interpretune.utils.import_utils import  _LIGHTNING_AVAILABLE
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import BaseITModule


# TODO: dynamically generate ITLightningDataModule and ITLightningModule classes (avoid the model train mode with
# LightningModule) with session-based approach now that it's available (should be able to get rid of this module)
if _LIGHTNING_AVAILABLE:
    from lightning.pytorch import LightningDataModule, LightningModule

    class ITLightningDataModule(ITDataModule, LightningDataModule):
        ...

    class ITLightningModule(BaseITModule, LightningModule):
        def on_train_start(self) -> None:
            # ensure model is in training mode (e.g. needed for some edge cases w/ skipped sanity checking)
            self.model.train()
            return super().on_train_start()

else:
    ITLightningDataModule = object
    ITLightningModule = object
