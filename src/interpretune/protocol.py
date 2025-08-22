from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import (
    Protocol,
    runtime_checkable,
    Union,
    TypeAlias,
    NamedTuple,
    TYPE_CHECKING,
    Callable,
    Optional,
    Any,
    Sequence,
    _ProtocolMeta,
    get_args,
    Mapping,
    TypedDict,
    TypeVar,
)
from os import PathLike
from types import UnionType
from enum import auto, Enum, EnumMeta
from dataclasses import dataclass
import inspect

import torch
from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import NotRequired, Required
from jsonargparse import Namespace
from transformers import BatchEncoding, PreTrainedTokenizerBase
from sae_lens.config import HfDataset

if TYPE_CHECKING:
    from interpretune.config import ITDataModuleConfig, ITConfig


################################################################################
# Interpretune helper types
################################################################################

# TODO: remove this type in favor using PathLike alone now that the type resolution issue should be fixed
StrOrPath: TypeAlias = Union[str, PathLike]

################################################################################
# Interpretune Enhanced Enums
################################################################################


class AutoStrEnum(Enum):
    def _generate_next_value_(name, _start, _count, _last_values) -> str:  # type: ignore
        return name


# NOTE [Interpretability Adapters]:
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
    # CIRCUIT_TRACER: The provided module and datamodule will be prepared for use with the Circuit Tracer adapter in
    #                  in combination with any supported and specified adapter.
    circuit_tracer = auto()

    def __lt__(self, other: "Adapter") -> bool:
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


class SetDerivedEnum(AutoStrEnum, metaclass=DerivedEnumMeta): ...


################################################################################
# Core Enums
################################################################################

# TODO: consider switching these to a data structure that natively allows for more flexible DRY composition
#       (currently preferring a custom enum for IDE autocompletion etc.)

CORE_PHASES = frozenset(["train", "validation", "test", "predict"])
EXT_PHASES = frozenset(["analysis"])
ALL_PHASES = CORE_PHASES.union(EXT_PHASES)


class CorePhases(SetDerivedEnum):
    # The _input_set class attribute is used by DerivedEnumMeta to generate enum members.
    _input_set = CORE_PHASES


class CoreSteps(SetDerivedEnum):
    _input_set = CORE_PHASES
    _transform = lambda x: "training_step" if x == "train" else f"{x}_step"


class AllPhases(SetDerivedEnum):
    _input_set = ALL_PHASES


class AllSteps(SetDerivedEnum):
    _input_set = ALL_PHASES
    _transform = lambda x: "training_step" if x == "train" else f"{x}_step"


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

AnyDataClass = TypeVar("AnyDataClass", bound=dataclass)


################################################################################
# Core Protocols
################################################################################


@runtime_checkable
class DataPrepable(Protocol):
    """Minimum requirement for an Interpretunable DataModule is to have a prepare_data method and a valid
    datamodule config."""

    def prepare_data(self, target_model: torch.nn.Module | None = None) -> None: ...


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
    itdm_cfg: "ITDataModuleConfig"

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
    it_cfg: "ITConfig"

    def setup(self, *args, **kwargs) -> None: ...

    def configure_optimizers(self) -> OptimizerLRScheduler | None: ...


# N.B. runtime protocol validation will check for attribute presence but not validate signatures etc. With this
# protocol-based approach we're providing rudimentary functional checks while erroring on the side of flexibility
ModuleSteppable: TypeAlias = TrainSteppable | ValidationSteppable | TestSteppable | PredictSteppable
DataModuleInitable: TypeAlias = TrainLoadable | ValLoadable | TestLoadable | PredictLoadable


def gen_protocol_variants(
    supported_sub_protocols: UnionType, base_protocols: _ProtocolMeta | tuple[_ProtocolMeta]
) -> UnionType:
    protocol_components = []
    if not isinstance(base_protocols, tuple):
        base_protocols = (base_protocols,)
    for sub_proto in get_args(supported_sub_protocols):
        supported_cls = _ProtocolMeta(f"Built{sub_proto.__name__}", (*base_protocols, sub_proto, Protocol), {})
        supported_cls._is_runtime_protocol = True
        supported_cls.__module__ = inspect.getmodule(inspect.currentframe().f_back).__name__
        protocol_components.append(supported_cls)
    gen_type_alias = " | ".join([f"protocol_components[{i}]" for i in range(len(protocol_components))])
    return eval(gen_type_alias)


# We generate valid datamodule/module protocol variants by composing their respective base protocols with the set of
# valid subprotocols over which `any` semantics apply.
ITDataModuleProtocol: TypeAlias = gen_protocol_variants(DataModuleInitable, DataModuleInvariants)  # type: ignore
ITModuleProtocol: TypeAlias = gen_protocol_variants(ModuleSteppable, ModuleInvariants)  # type: ignore
# TODO: ensure protocol variants are explicitly documented and possibly add a section describing the approach to
#       supported protocol variant generation. Also add an issue tracker for this approach to solicit ideas for a
#       cleaner/more pythonic approach. As Python structural subtyping features are still evolving, if a cleaner and
#       more pythonic approach isn't available now, one will hopefully be available in the near future.
InterpretunableType: TypeAlias = Union[ITDataModuleProtocol, ITModuleProtocol]


class InterpretunableTuple(NamedTuple):
    datamodule: ITDataModuleProtocol | None = None
    module: ITModuleProtocol | None = None


################################################################################
# Analysis Protocols
################################################################################

NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str], str]]


class SAEFqn(NamedTuple):
    release: str
    sae_id: str


class AnalysisOpProtocol(Protocol):
    """Protocol defining required interface for analysis operations."""

    name: str
    description: str
    output_schema: dict
    input_schema: Optional[dict]

    def save_batch(
        self,
        analysis_batch: BaseAnalysisBatchProtocol,
        batch: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase | None = None,
        save_prompts: bool = False,
        save_tokens: bool = False,
        decode_kwargs: Optional[dict] = None,
    ) -> BaseAnalysisBatchProtocol: ...


class SAEDictProtocol(Protocol):
    """Protocol for SAE analysis dictionary operations."""

    def shapes(self) -> dict[str, torch.Size | list[torch.Size]]: ...
    def batch_join(
        self, across_saes: bool = False, join_fn: Callable = torch.cat
    ) -> SAEDictProtocol | list[torch.Tensor]: ...
    def apply_op_by_sae(self, operation: Callable | str, *args, **kwargs) -> SAEDictProtocol: ...


class AnalysisStoreProtocol(Protocol):
    """Protocol verifying core analysis store functionality."""

    dataset: HfDataset
    streaming: bool
    cache_dir: str | None
    save_dir: StrOrPath
    stack_batches: bool
    split: str
    op_output_dataset_path: str | None

    def by_sae(self, field_name: str, stack_latents: bool = True) -> SAEDictProtocol: ...
    def __getattr__(self, name: str) -> list: ...
    def reset_dataset(self) -> None: ...


class AnalysisCfgProtocol(Protocol):
    """Protocol verifying core analysis configuration functionality."""

    output_store: AnalysisStoreProtocol
    input_store: AnalysisStoreProtocol
    op: AnalysisOpProtocol
    fwd_hooks: list[tuple]
    bwd_hooks: list[tuple]
    cache_dict: dict
    names_filter: NamesFilter | None
    # Save configuration fields
    save_prompts: bool
    save_tokens: bool
    decode_kwargs: dict

    def check_add_default_hooks(
        self, op: AnalysisOpProtocol, names_filter: str | Callable | None, cache_dict: dict | None
    ) -> tuple[list[tuple], list[tuple]]: ...


class SAEAnalysisProtocol(Protocol):
    """Protocol for SAE analysis components requiring a subset of SAEAnalysisMixin methods."""

    def construct_names_filter(
        self, target_layers: list[int], sae_hook_match_fn: Callable[[str, list[int] | None], bool]
    ) -> NamesFilter: ...


class ActivationCacheProtocol(Protocol):
    """Core activation cache protocol."""

    cache_dict: dict[str, torch.Tensor]
    has_batch_dim: bool
    has_embed: bool
    has_pos_embed: bool

    def __getitem__(self, key: str | tuple) -> torch.Tensor: ...
    def stack_activation(
        self, activation_name: str, layer: int = -1, sublayer_type: str | None = None
    ) -> torch.Tensor: ...


class BaseAnalysisBatchProtocol(Protocol):
    """Base protocol defining methods all analysis batches should implement.

    Subclasses should define which dataset columns will have attribute-based access enabled for associated AnalysisStore
    objects.
    """

    def update(self, **kwargs) -> None: ...
    def to_cpu(self) -> None: ...


class DefaultAnalysisBatchProtocol(BaseAnalysisBatchProtocol):
    """Default analysis batch protocol defining which dataset columns should have attribute-based access enabled
    for AnalysisStore objects. Subclasses can extend this protocol (or the base one) to add additional attributes
    or change existing attributes as needed.

    Attributes:
        logit_diffs (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Per batch logit differences with shape [batch_size]
        answer_logits (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Model output logits with shape [batch_size, 1, num_classes]
        loss (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Loss values with shape [batch_size]
        label_ids (Optional[torch.Tensor]):
            Input labels translated to token ids with shape [batch_size] (if labels provided & translation is needed)
        orig_labels (Optional[torch.Tensor]):
            Ground truth unmodified labels with shape [batch_size]
        preds (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Model predictions with shape [batch_size]
        cache (Optional[ActivationCacheProtocol]):
            Forward pass activation cache
        grad_cache (Optional[ActivationCacheProtocol]):
            Backward pass gradient cache
        answer_indices (Optional[torch.Tensor]):
            Indices of answers with shape [batch_size]
        alive_latents (Optional[dict[str, list[int]]]):
            Active latent indices per SAE hook
        correct_activations (Optional[dict[str, torch.Tensor]]):
            SAE activations after corrections with shape [batch_size, d_sae] for each SAE
        attribution_values (Optional[dict[str, torch.Tensor]]):
            Attribution values per SAE hook
        tokens (Optional[torch.Tensor]):
            Input token IDs
        prompts (Optional[list[str]]):
            Text prompts
    """

    logit_diffs: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    answer_logits: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    loss: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    preds: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    label_ids: Optional[torch.Tensor]
    orig_labels: Optional[torch.Tensor]
    cache: Optional[ActivationCacheProtocol]
    grad_cache: Optional[ActivationCacheProtocol]
    answer_indices: Optional[torch.Tensor]
    alive_latents: Optional[dict[str, list[int]]]
    correct_activations: Optional[dict[str, torch.Tensor]]
    attribution_values: Optional[dict[str, torch.Tensor]]
    tokens: Optional[torch.Tensor]
    prompts: Optional[list[str]]


class CircuitAnalysisBatchProtocol(DefaultAnalysisBatchProtocol):
    """Circuit analysis batch protocol defining additional attributes for circuit tracer operations.

    Extends the default protocol with circuit tracing specific attributes.

    Attributes:
        attribution_graphs (Optional[list]):
            Generated attribution graphs for each prompt in the batch
        graph_metadata (Optional[list[dict]]):
            Metadata for each generated graph including parameters used
        graph_paths (Optional[list[str]]):
            File paths where graphs are saved (if saved)
        circuit_prompts (Optional[list[str]]):
            Prompts used for circuit attribution (may differ from input prompts)
    """

    attribution_graphs: Optional[list]
    graph_metadata: Optional[list[dict]]
    graph_paths: Optional[list[str]]
    circuit_prompts: Optional[list[str]]
