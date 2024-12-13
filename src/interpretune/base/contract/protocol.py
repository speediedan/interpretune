from typing import  Protocol, runtime_checkable, Union, TypeAlias, Optional, NamedTuple

import torch

from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.utils.types import STEP_OUTPUT, OptimizerLRScheduler, gen_protocol_variants


@runtime_checkable
class DataPrepable(Protocol):
    """Minimum requirement for an Interpretunable DataModule is to have a prepare_data method and a valid
    datamodule config."""

    def prepare_data(self, target_model: Optional[torch.nn.Module] = None) -> None: ...

@runtime_checkable
class TrainLoadable(DataPrepable, Protocol):
    def train_dataloader(self) -> torch.utils.data.DataLoader: ...

@runtime_checkable
class ValLoadable(DataPrepable, Protocol):
    def val_dataloader(self) -> torch.utils.data.DataLoader: ...

@runtime_checkable
class TestLoadable(DataPrepable, Protocol):
    def test_dataloader(self) -> torch.utils.data.DataLoader: ...

@runtime_checkable
class PredictLoadable(DataPrepable, Protocol):
    def predict_dataloader(self) -> torch.utils.data.DataLoader: ...

@runtime_checkable
class DataModuleInvariants(Protocol):
    itdm_cfg: ITDataModuleConfig

    def setup(self, *args, **kwargs) -> None: ...

@runtime_checkable
class TrainSteppable(Protocol):
    def training_step(self, *args, **kwargs) -> STEP_OUTPUT: ...

@runtime_checkable
class ValidationSteppable(Protocol):
    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT: ...

@runtime_checkable
class TestSteppable(Protocol):
    def test_step(self, *args, **kwargs) -> STEP_OUTPUT: ...

@runtime_checkable
class PredictSteppable(Protocol):
    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT: ...

@runtime_checkable
class ModuleInvariants(Protocol):
    it_cfg: ITConfig

    def setup(self, *args, **kwargs) -> None: ...

    def configure_optimizers(self) -> Optional[OptimizerLRScheduler]: ...

# N.B. runtime protocol validation will check for attribute presence but not validate signatures etc. With this
# protocol-based approach we're providing rudimentary functional checks while erroring on the side of flexibility
ModuleSteppable: TypeAlias = TrainSteppable | ValidationSteppable | TestSteppable | PredictSteppable
DataModuleInitable: TypeAlias = TrainLoadable | ValLoadable | TestLoadable | PredictLoadable

# We generate valid datamodule/module protocol variants by composing their respective base protocols with the set of
# valid subprotocols over which `any` semantics apply.
ITDataModuleProtocol: TypeAlias = gen_protocol_variants(DataModuleInitable, DataModuleInvariants)  # type: ignore
ITModuleProtocol: TypeAlias = gen_protocol_variants(ModuleSteppable, ModuleInvariants)   # type: ignore
# TODO: ensure protocol variants are explicitly documented and possibly add a section describing the approach to
#       supported protocol variant generation. Also add an issue tracker for this approach to solicit ideas for a
#       cleaner/more pythonic approach. As Python structural subtyping features are still evolving, if a cleaner and
#       more pythonic approach isn't available now, one will hopefully be available in the near future.
InterpretunableType: TypeAlias = Union[ITDataModuleProtocol, ITModuleProtocol]

class InterpretunableTuple(NamedTuple):
    datamodule: Optional[ITDataModuleProtocol] = None
    module: Optional[ITModuleProtocol] = None
