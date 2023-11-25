from dataclasses import dataclass
from typing import (
    Any,
    List,
    Mapping,
    Dict,
    Optional,
    TypeVar,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    runtime_checkable,
)
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import NotRequired, Required
from jsonargparse import Namespace

_DictKey = TypeVar("_DictKey")

@runtime_checkable
class _Stateful(Protocol[_DictKey]):
    """This class is used to detect if an object is stateful using `isinstance(obj, _Stateful)`."""

    def state_dict(self) -> Dict[_DictKey, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[_DictKey, Any]) -> None:
        ...


@runtime_checkable
class LRScheduler(_Stateful[str], Protocol):
    optimizer: Optimizer
    base_lrs: List[float]

    def __init__(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> None:
        ...

    def step(self, epoch: Optional[int] = None) -> None:
        ...


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
    ) -> None:
        ...

    def step(self, metrics: Union[float, int, Tensor], epoch: Optional[int] = None) -> None:
        ...


STEP_OUTPUT = Optional[Union[Tensor, Mapping[str, Any]]]

LRSchedulerTypeTuple = (torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
LRSchedulerTypeUnion = Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]
LRSchedulerType = Union[Type[torch.optim.lr_scheduler.LRScheduler], Type[torch.optim.lr_scheduler.ReduceLROnPlateau]]
LRSchedulerPLType = Union[LRScheduler, ReduceLROnPlateau]


@dataclass
class LRSchedulerConfig:
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, ReduceLROnPlateau]
    # no custom name
    name: Optional[str] = None
    # after epoch is over
    interval: str = "epoch"
    # every epoch/batch
    frequency: int = 1
    # most often not ReduceLROnPlateau scheduler
    reduce_on_plateau: bool = False
    # value to monitor for ReduceLROnPlateau
    monitor: Optional[str] = None
    # enforce that the monitor exists for ReduceLROnPlateau
    strict: bool = True


class LRSchedulerConfigType(TypedDict, total=False):
    scheduler: Required[LRSchedulerTypeUnion]
    name: Optional[str]
    interval: str
    frequency: int
    reduce_on_plateau: bool
    monitor: Optional[str]
    scrict: bool


class OptimizerLRSchedulerConfig(TypedDict):
    optimizer: Optimizer
    lr_scheduler: NotRequired[Union[LRSchedulerTypeUnion, LRSchedulerConfigType]]


OptimizerLRScheduler = Optional[
    Union[
        Optimizer,
        Sequence[Optimizer],
        Tuple[Sequence[Optimizer], Sequence[Union[LRSchedulerTypeUnion, LRSchedulerConfig]]],
        OptimizerLRSchedulerConfig,
    ]
]

ArgsType = Optional[Union[List[str], Dict[str, Any], Namespace]]
