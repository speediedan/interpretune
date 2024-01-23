from typing import  Protocol, runtime_checkable, Union, TypeAlias, Optional, NamedTuple, TypeVar

import torch

from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import BaseITModule
from interpretune.utils.types import STEP_OUTPUT, OptimizerLRScheduler


@runtime_checkable
class DataPrepable(Protocol):
    """Minimum requirement for an Interpretunable DataModule is to have a prepare_data method and a valid
    datamodule config."""

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None:
        ...

@runtime_checkable
class DataLoadable(Protocol):
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        ...
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        ...
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        ...
    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        ...

@runtime_checkable
class DataModuleInitable(DataPrepable, DataLoadable, Protocol):
    ...

# N.B. runtime protocol validation will check for attribute presence but not validate the type. With this
# protocol-based approach we're providing rudimentary functional checks while erroring on the side of flexibility
@runtime_checkable
class InterpretunableDataModule(DataModuleInitable, Protocol):
    itdm_cfg: ITDataModuleConfig

    def setup(self, *args, **kwargs) -> None:
        ...

@runtime_checkable
class TrainSteppable(Protocol):
    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...

@runtime_checkable
class ValidationSteppable(Protocol):
    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...

@runtime_checkable
class TestSteppable(Protocol):
    def test_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...

@runtime_checkable
class PredictSteppable(Protocol):
    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...

@runtime_checkable
class InterpretunableModule(TrainSteppable, ValidationSteppable, TestSteppable, PredictSteppable, Protocol):
    it_cfg: ITConfig

    def setup(self, *args, **kwargs) -> None:
        ...

    def configure_optimizers(self) -> Optional[OptimizerLRScheduler]:
        ...

ModuleSteppable: TypeAlias = TrainSteppable | ValidationSteppable | TestSteppable | PredictSteppable

T_dm = TypeVar('T_dm', bound=ITDataModule, covariant=True)
T_m = TypeVar('T_m', bound=BaseITModule, covariant=True)


class NamedWrapper:
    def __str__(self):
        return f"{self.__class__.__name__}({self._orig_module_name})"
    def __repr__(self):
        return str(self)

InterpretunableType: TypeAlias = Union[InterpretunableDataModule, InterpretunableModule]

class InterpretunableTuple(NamedTuple):
    datamodule: Optional[InterpretunableDataModule] = None
    module: Optional[InterpretunableModule] = None
