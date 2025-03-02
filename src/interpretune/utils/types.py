from dataclasses import dataclass
from os import PathLike
from types import UnionType
import inspect
from enum import auto, Enum, EnumMeta
from typing import (
    Any,
    Optional,
    TypeVar,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
    TypeAlias,
    _ProtocolMeta,
    get_args
)
from collections.abc import Mapping, Callable, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import NotRequired, Required
from jsonargparse import Namespace


################################################################################
# Interpretune helper types
################################################################################
StrOrPath: TypeAlias = Union[str, PathLike]

################################################################################
# Interpretune Enhanced Enums
################################################################################

class AutoStrEnum(Enum):
    def _generate_next_value_(name, _start, _count, _last_values) -> str:  # type: ignore
        return name


# NOTE [Interpretability Adapters]:
# `TransformerLens` is the first supported interpretability adapter but support for other adapters is expected
class Adapter(AutoStrEnum):
    # CORE: The provided module and datamodule will be prepared for use with core PyTorch. The default
    #       trainer, a custom trainer or no trainer all can be used in combination with any supported and specified
    #       adapter.
    core = auto()
    # LIGHTNING: The provided module and datamodule will be prepared for use with the Lightning trainer and any
    #            supported and specified adapter.
    lightning = auto()
    # TRANSFORMER_LENS: The provided module and datamodule will be prepared for use with the TransformerLens adapter in
    #                  in combination with any supported and specified adapter.
    transformer_lens = auto()
    # SAE_LENS: The provided module and datamodule will be prepared for use with the SAELens adapter in
    #                  in combination with any supported and specified adapter.
    sae_lens = auto()

    def __lt__(self, other: 'Adapter') -> bool:
        return self.value < other.value

# DerivedEnumMeta is a custom metaclass that adds enum members from an input set.
class DerivedEnumMeta(EnumMeta):  # change EnumMeta alias to EnumType when 3.11 python is minimum
    _derived_enum_internal = ("_input_set", "_transform")

    @classmethod
    def __prepare__(metacls, clsname, bases, **kwargs):
        enum_dict = super().__prepare__(clsname, bases, **kwargs)
        enum_dict._ignore = list(metacls._derived_enum_internal)
        return enum_dict

    def __new__(mcls, clsname, bases, classdict):
        # Pop and remove temporary keys so they don't become enum members.
        # While we could use '_ignore_', as we need these variables anyway, we go ahead and proactively pop them here
        input_set = classdict.pop("_input_set", None)
        transform = classdict.pop("_transform", None)
        if input_set is not None:
            if transform and callable(transform):
                input_set = {transform(x) for x in input_set}
            for member in sorted(input_set):
                if member not in classdict:
                    classdict[member] = auto()
        return super().__new__(mcls, clsname, bases, classdict)

class SetDerivedEnum(AutoStrEnum, metaclass=DerivedEnumMeta):
    ...

################################################################################
# Core Enums
################################################################################

# TODO: consider switching these to a data structure that natively allows for more flexible DRY composition
#       (currently preferring a custom enum for IDE autocompletion etc.)

CORE_PHASES = frozenset(['train', 'validation', 'test', 'predict'])
EXT_PHASES = frozenset(['analysis'])
ALL_PHASES = CORE_PHASES.union(EXT_PHASES)

class CorePhases(SetDerivedEnum):
    # The _input_set class attribute is used by DerivedEnumMeta to generate enum members.
    _input_set = CORE_PHASES

class CoreSteps(SetDerivedEnum):
    _input_set = CORE_PHASES
    _transform = lambda x: f"{x}_step"

class AllPhases(SetDerivedEnum):
    _input_set = ALL_PHASES

class AllSteps(SetDerivedEnum):
    _input_set = ALL_PHASES
    _transform = lambda x: f"{x}_step"

################################################################################
# Framework Compatibility helper types
# originally inspired by https://bit.ly/lightning_types definitions
################################################################################

_DictKey = TypeVar("_DictKey")

@runtime_checkable
class Steppable(Protocol):
    """To structurally type ``optimizer.step()``"""

    # Inferred from `torch.optim.optimizer.pyi`
    def step(self, closure: Callable[[], float] | None = ...) -> float | None: ...

@runtime_checkable
class Optimizable(Steppable, Protocol):
    """To structurally type ``optimizer``"""

    param_groups: list[dict[Any, Any]]
    defaults: dict[Any, Any]
    state: dict[Any, Any]

    def state_dict(self) -> dict[str, dict[Any, Any]]: ...

    def load_state_dict(self, state_dict: dict[str, dict[Any, Any]]) -> None: ...


@runtime_checkable
class _Stateful(Protocol[_DictKey]):
    """This class is used to detect if an object is stateful using `isinstance(obj, _Stateful)`."""

    def state_dict(self) -> dict[_DictKey, Any]: ...

    def load_state_dict(self, state_dict: dict[_DictKey, Any]) -> None: ...


@runtime_checkable
class LRScheduler(_Stateful[str], Protocol):
    optimizer: Optimizer
    base_lrs: list[float]

    def __init__(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> None: ...

    def step(self, epoch: int | None = None) -> None: ...


# Inferred from `torch.optim.lr_scheduler.pyi`
# Missing attributes were added to improve typing
@runtime_checkable
class ReduceLROnPlateau(_Stateful[str], Protocol):
    in_cooldown: bool
    optimizer: Optimizer

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = ...,
        factor: float = ...,
        patience: int = ...,
        verbose: bool = ...,
        threshold: float = ...,
        threshold_mode: str = ...,
        cooldown: int = ...,
        min_lr: float = ...,
        eps: float = ...,
    ) -> None: ...

    def step(self, metrics: float | int | Tensor, epoch: int | None = None) -> None: ...

STEP_OUTPUT = Optional[Union[Tensor, Mapping[str, Any]]]

LRSchedulerTypeUnion = Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]

@dataclass
class LRSchedulerConfig:
    scheduler: torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau
    # no custom name
    name: str | None = None
    # after epoch is over
    interval: str = "epoch"
    # every epoch/batch
    frequency: int = 1
    # most often not ReduceLROnPlateau scheduler
    reduce_on_plateau: bool = False
    # value to monitor for ReduceLROnPlateau
    monitor: str | None = None
    # enforce that the monitor exists for ReduceLROnPlateau
    strict: bool = True


class LRSchedulerConfigType(TypedDict, total=False):
    scheduler: Required[LRSchedulerTypeUnion]
    name: str | None
    interval: str
    frequency: int
    reduce_on_plateau: bool
    monitor: str | None
    scrict: bool


class OptimizerLRSchedulerConfig(TypedDict):
    optimizer: Optimizer
    lr_scheduler: NotRequired[LRSchedulerTypeUnion | LRSchedulerConfigType]


OptimizerLRScheduler = Optional[
    Union[
        Optimizer,
        Sequence[Optimizer],
        tuple[Sequence[Optimizer], Sequence[Union[LRSchedulerTypeUnion, LRSchedulerConfig]]],
        OptimizerLRSchedulerConfig,
    ]
]

ArgsType = Optional[Union[list[str], dict[str, Any], Namespace]]


def gen_protocol_variants(supported_sub_protocols: UnionType,
                          base_protocols: _ProtocolMeta | tuple[_ProtocolMeta]) -> UnionType:
    protocol_components = []
    if not isinstance(base_protocols, tuple):
        base_protocols = (base_protocols,)
    for sub_proto in get_args(supported_sub_protocols):
        supported_cls = _ProtocolMeta(f'Built{sub_proto.__name__}', (*base_protocols, sub_proto, Protocol), {})
        supported_cls._is_runtime_protocol = True
        supported_cls.__module__ = inspect.getmodule(inspect.currentframe().f_back).__name__
        protocol_components.append(supported_cls)
    gen_type_alias = " | ".join([f"protocol_components[{i}]" for i in range(len(protocol_components))])
    return eval(gen_type_alias)

AnyDataClass = TypeVar('AnyDataClass', bound=dataclass)
