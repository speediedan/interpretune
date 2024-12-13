from typing import Any, Dict, Optional, List, Tuple, TypeVar, Type, Set, Sequence, TypeAlias
from dataclasses import dataclass, field, fields, make_dataclass
import inspect
import logging
import os
import sys
from enum import auto, Enum

import yaml
from transformers import PreTrainedTokenizerBase

from interpretune.utils.logging import rank_zero_warn, rank_zero_debug
from interpretune.utils.types import AnyDataClass

log = logging.getLogger(__name__)

################################################################################
# Core Enums
################################################################################

class AutoStrEnum(Enum):
    def _generate_next_value_(name, start, count, last_values) -> str:  # type: ignore
        return name

class CorePhase(AutoStrEnum):
    train = auto()
    validation = auto()
    test = auto()
    predict = auto()

class CoreSteps(AutoStrEnum):
    training_step = auto()
    validation_step = auto()
    test_step = auto()
    predict_step = auto()

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

AdapterSeq: TypeAlias = Sequence[Adapter | str] | Adapter | str

def adapter_seq_to_list(adapter_seq: AdapterSeq):
    if isinstance(adapter_seq, str):
        adapter_seq = [adapter_seq]
    if not isinstance(adapter_seq, list):
        adapter_seq = list(adapter_seq)
    return [Adapter[adp] if isinstance(adp, str) else adp for adp in adapter_seq]

################################################################################
# Auto Composition Target Resolution
################################################################################

class ComposedCfgWrapper:

    def __repr__(self) -> str:
        orig_module = getattr(self, "_orig_module_cfg_name",
                              "Original module config attribute not set, instantiation incomplete.")
        composed_classes = getattr(self, "_composed_classes", "N/A")
        enriched_mod_str = f"Original module cfg: {orig_module} {os.linesep}"
        enriched_mod_str += f"Now {self.__class__.__name__}, a composition of: {os.linesep}  - "
        composed_mod_lines = [c.__name__ for c in composed_classes] if not isinstance(composed_classes, str) else "N/A"
        enriched_mod_str += f'{os.linesep}  - '.join(composed_mod_lines) + f'{os.linesep}'
        return enriched_mod_str + super().__repr__()

#TODO: add custom constructors and representers for core IT object types
@dataclass(kw_only=True)
class ITSerializableCfg(yaml.YAMLObject):
    ...

@dataclass(kw_only=True)
class AutoCompConfig(ITSerializableCfg):
    module_cfg_name: str
    module_cfg_mixin: List[AnyDataClass] | AnyDataClass
    target_adapters: Optional[AdapterSeq] = None

    def __post_init__(self):
        if not isinstance(self.module_cfg_mixin, list):
            self.module_cfg_mixin = [self.module_cfg_mixin]
        if self.target_adapters is not None:
            self.target_adapters = adapter_seq_to_list(self.target_adapters)

@dataclass(kw_only=True)
class AutoCompConf(ITSerializableCfg):
    auto_comp_cfg: Optional[AutoCompConfig] = None

    def __new__(cls, **kwargs):
        if kwargs.get('auto_comp_cfg', None) is not None:
            built_class = AutoCompConf.compose_cfg_dataclass(cls, kwargs)
            return super().__new__(built_class)
        else:
            return super().__new__(cls)

    @staticmethod
    def compose_cfg_dataclass(cls, kwargs):
        setattr(kwargs['auto_comp_cfg'], '_orig_cfg_cls', cls)
        auto_comp_cfg = kwargs.pop('auto_comp_cfg')
        assert getattr(auto_comp_cfg, '_orig_cfg_cls', None) is not None, "`auto_comp_cfg` missing `_orig_cfg_cls`"
        composition_classes = resolve_composition_classes(auto_comp_cfg, kwargs)
        if not composition_classes:
            return cls
        built_class = make_dataclass(auto_comp_cfg.module_cfg_name, kwargs, bases=composition_classes, kw_only=True)
        built_class = type(auto_comp_cfg.module_cfg_name, (ComposedCfgWrapper, built_class), {})
        built_class.__module__ = 'interpretune'
        built_class._orig_module_cfg_name = auto_comp_cfg._orig_cfg_cls.__qualname__
        built_class._composed_classes = composition_classes
        return built_class

def collect_exhaustive_attr_set(target_type: Type) -> Set[str]:
    target_type_attrs = {attr for attr in dir(target_type) if not attr.startswith('__')}
    parent_attrs = set()
    for parent_cls in inspect.getmro(target_type)[1:]:
        parent_attrs.update(attr for attr in dir(parent_cls) if not attr.startswith('__'))
    dataclass_fields = {field.name for field in fields(target_type)} \
        if hasattr(target_type, '__dataclass_fields__') else set()
    all_attrs = target_type_attrs.union(parent_attrs).union(dataclass_fields)
    return all_attrs

def candidate_subclass_attrs(kwargs: Dict, target_type: Type) -> Dict:
    """Finds the keys in kwargs that are not attributes of target_type."""
    all_attrs = collect_exhaustive_attr_set(target_type)
    return {key: value for key, value in kwargs.items() if key not in all_attrs and not key.startswith('__')}

T = TypeVar("T")

def find_adapter_subclasses(T: Type, target_adapters: Optional[AdapterSeq] = None) -> Dict[str, Type[T]]:
    """Searches the current global namespace underneath `interpretune.adapters` for subclasses of `T` and returns
    them.

    If target_adapters is provided, only considers subclasses from the specified adapters.
    """
    subclasses, superclasses = {}, {}
    adapter_space = adapter_seq_to_list(target_adapters) if target_adapters is not None \
        else Adapter.__members__.values()
    candidate_modules = {val: (f'interpretune.adapters.{val.name}',
                               sys.modules[f'interpretune.adapters.{val.name}']) for val in adapter_space}
    for adapter, (module_fqn, module) in candidate_modules.items():
        for _, member in inspect.getmembers(module, inspect.isclass):
            if member.__module__ != module_fqn:
                continue
            if issubclass(member, T) and member is not T:
                subclasses[adapter] = member
            elif issubclass(T, member):
                superclasses[adapter] = member
    return subclasses, superclasses

def search_candidate_subclass_attrs(candidate_modules: Dict[Adapter, Type],
                                    kwargs_not_in_target_type: Dict) -> Optional[Tuple[Type]]:
    valid_subclasses = []
    min_extra_attrs = float('inf')
    for _, module_class in candidate_modules.items():
        module_attrs = collect_exhaustive_attr_set(module_class)
        # find candidate subclasses with all required attributes and a minimum number of extra attributes
        if all(attr in module_attrs for attr in kwargs_not_in_target_type):
            extra_attrs = len(module_attrs) - len(kwargs_not_in_target_type)
            if extra_attrs < min_extra_attrs:
                min_extra_attrs = extra_attrs
                valid_subclasses = [module_class]
            elif extra_attrs == min_extra_attrs:
                valid_subclasses.append(module_class)

    if not valid_subclasses:
        return
    return (valid_subclasses[0],)  # Return the first valid subclass (they all have the same number of extra attributes)

def check_non_subclasses(target_class: Type, candidate_classes: List[Type]) -> Optional[Tuple[Type]]:
    unfullfilled_subclasses = []
    for cls in candidate_classes:
        if not issubclass(target_class, cls):
            unfullfilled_subclasses.append(cls)
    if unfullfilled_subclasses:
        return tuple(unfullfilled_subclasses)
    return

def issue_noncomposition_feedback(auto_comp_cfg, superclasses, subclasses):
    is_ready = f"already supports all of the provided kwargs, is already a subclass of {auto_comp_cfg.module_cfg_mixin}"
    base_message = (f"No auto-composition needed for {auto_comp_cfg._orig_cfg_cls} as it {is_ready}")
    if not auto_comp_cfg.target_adapters:
        rank_zero_debug(f"{base_message} and no `target_adapters` were provided.")
    elif superclasses:
        rank_zero_debug(f"{base_message} and already is a subclass of a class in `target_adapters`.")
    elif not subclasses:
        rank_zero_warn("No candidate classes in the specified `target_adapters` were found to further compose with."
                       f"Since {auto_comp_cfg._orig_cfg_cls} {is_ready}, instantiating without auto-composition.")

def issue_incomplete_composition_feedback(auto_comp_cfg, kwargs_not_in_target_type, nonsubcls_mixins):
    no_auto_prefix = (f"Could not find an auto-composition for {auto_comp_cfg._orig_cfg_cls} that supports all of"
                        f" the following kwargs: {kwargs_not_in_target_type}.")
    if nonsubcls_mixins:
        rank_zero_warn(f"{no_auto_prefix} Trying instantiation while composing with {nonsubcls_mixins}.")
        return (auto_comp_cfg._orig_cfg_cls,) + nonsubcls_mixins
    else:
        rank_zero_warn(f"{no_auto_prefix} As {auto_comp_cfg._orig_cfg_cls} is already a subclass of "
                    f"{auto_comp_cfg.module_cfg_mixin}, trying instantiation without further composition.")
        return

def resolve_composition_classes(auto_comp_cfg: AutoCompConfig, kwargs: Dict) -> Optional[Tuple[Type]]:
    adapter_composition_classes = None
    subclasses, superclasses = find_adapter_subclasses(auto_comp_cfg._orig_cfg_cls, auto_comp_cfg.target_adapters)
    kwargs_not_in_target_type = candidate_subclass_attrs(kwargs, auto_comp_cfg._orig_cfg_cls)
    nonsubcls_mixins = check_non_subclasses(auto_comp_cfg._orig_cfg_cls, auto_comp_cfg.module_cfg_mixin)
    adapter_composition_classes = search_candidate_subclass_attrs(subclasses, kwargs_not_in_target_type)
    match bool(kwargs_not_in_target_type), bool(nonsubcls_mixins), bool(adapter_composition_classes):
        case (False, False, _):
            issue_noncomposition_feedback(auto_comp_cfg, superclasses, subclasses)
            return
        case (False, True, _):
            rank_zero_debug(f"{auto_comp_cfg._orig_cfg_cls} already supports all of the provided kwargs but needs to "
                            f"be composed with {nonsubcls_mixins}.")
            return (auto_comp_cfg._orig_cfg_cls,) + nonsubcls_mixins
        case (True, _, False):
            return issue_incomplete_composition_feedback(auto_comp_cfg, kwargs_not_in_target_type, nonsubcls_mixins)
        case (_, False, True):
            return adapter_composition_classes
        case (_, True, True):
            return adapter_composition_classes + nonsubcls_mixins

################################################################################
# Core Shared Configuration for Datamodules and Modules
################################################################################

@dataclass(kw_only=True)
class ITSharedConfig(ITSerializableCfg):
    model_name_or_path: str = ''
    task_name: str = ''
    tokenizer_name: Optional[str] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    os_env_model_auth_key: Optional[str] = None
    tokenizer_id_overrides: Optional[Dict] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    defer_model_init: Optional[bool] = False

    def _validate_on_session_cfg_init(self):
        # deferred validation for attributes that my be set via shared datamodule/module config
        if self.defer_model_init:
            assert self.signature_columns is not None, ("`signature_columns` must be specified if `defer_model_init` "
                                                        "is set to True")
