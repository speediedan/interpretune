import lightning.pytorch as pl

from interpretune.base.it_module import BaseITModule, BaseITLensModule
from interpretune.base.it_datamodule import ITDataModule

#TODO: consider moving these definitions to it_module.py, conditioning definitions on lightning availability
class ITLightningDataModule(ITDataModule, pl.LightningDataModule):
    ...

class ITLightningModule(BaseITModule, pl.LightningModule):
    """A :class:`~lightning.pytorch.core.module.LightningModule` that can be used to fine-tune a foundation model
    on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """

    def on_train_start(self) -> None:
        self.model.train()  # ensure model is in training mode
        return super().on_train_start()


class ITHookedLightningModule(BaseITLensModule, ITLightningModule):
    ...
