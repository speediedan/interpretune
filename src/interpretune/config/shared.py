from typing import Any, TypeVar, TypeAlias, Sequence
from dataclasses import dataclass, field, fields, make_dataclass
import inspect
import logging
import os
import sys
from pathlib import PosixPath, WindowsPath

import yaml
from transformers import PreTrainedTokenizerBase

from interpretune.utils import ITInstantiationFeedbackWarning, rank_zero_warn, rank_zero_debug
from interpretune.protocol import Adapter


log = logging.getLogger(__name__)

# DEFAULT auto-composition search TEMPLATES, formatted with an adapter name.
#
# Each adapter now owns one package holding its module composition and its config (#401), so
# auto-composition looks inside that package rather than at two parallel namespaces keyed by adapter
# name. Templates rather than base paths because the interesting modules are no longer all the same
# depth, and because a template states the convention it depends on instead of implying it.
#
# A hub-delivered adapter is NOT reachable this way: its module is executed from a cache under a
# revision-scoped synthetic name, so no import path can be derived from the adapter name. Discovery for
# those goes through the registry the component's entrypoint writes to, which is why this list stays
# BUNDLED-only rather than growing a "search everything" mode that would still miss them.
# NOT the bare package: `inspect.getmembers` below calls `dir()` and then `getattr` for every name, and
# these packages export lazily, so scanning the package would RESOLVE every export and import each
# adapter's framework as a side effect of composing a config. The submodules define the classes anyway
# (the `member.__module__` guard already discards anything merely re-exported), so the package entry
# would contribute nothing while costing heavy imports.
AUTOCOMP_SEARCH_TEMPLATES = [
    "interpretune.adapters.{adapter}.config",
    "interpretune.adapters.{adapter}.adapter",
]

AdapterSeq: TypeAlias = Sequence[Adapter | str] | Adapter | str


def adapter_seq_to_list(adapter_seq: AdapterSeq):
    """Normalize an adapter sequence to a list, accepting a bare string as a one-element sequence."""
    if isinstance(adapter_seq, str):
        adapter_seq = [adapter_seq]
    elif isinstance(adapter_seq, Adapter):
        # Handle single Adapter enum
        adapter_seq = [adapter_seq]
    elif not isinstance(adapter_seq, list):
        # Handle Sequence types
        adapter_seq = list(adapter_seq)
    return [Adapter[adp] if isinstance(adp, str) else adp for adp in adapter_seq]


################################################################################
# Auto Composition Target Resolution
################################################################################


class ComposedCfgWrapper:
    """Base of auto-composed config classes, giving them a repr naming what they were composed from.

    Auto-composition synthesizes a dataclass at runtime, so the default repr would name a class the user
    never wrote. The counterpart of :class:`~interpretune.session.NamedWrapper` on the config side.
    """

    def __repr__(self) -> str:
        orig_module = getattr(
            self, "_orig_module_cfg_name", "Original module config attribute not set, instantiation incomplete."
        )
        composed_classes = getattr(self, "_composed_classes", "N/A")
        enriched_mod_str = f"Original module cfg: {orig_module} {os.linesep}"
        enriched_mod_str += f"Now {self.__class__.__name__}, a composition of: {os.linesep}  - "
        composed_mod_lines = [c.__name__ for c in composed_classes] if not isinstance(composed_classes, str) else "N/A"
        enriched_mod_str += f"{os.linesep}  - ".join(composed_mod_lines) + f"{os.linesep}"
        return enriched_mod_str + super().__repr__()


# TODO: add custom constructors and representers for core IT object types
@dataclass(kw_only=True)
class ITSerializableCfg(yaml.YAMLObject):
    """Base class for serializable Interpretune configs.

    Automatically registers subclasses and Path types as safe globals for PyTorch checkpoint loading.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register all ITSerializableCfg subclasses as safe for pickle deserialization
        # This is required when loading checkpoints with weights_only=True
        # Also register Path types to allow Path objects in serialized configs
        try:
            import torch.serialization

            # Register the config class and both platform-specific Path types
            torch.serialization.add_safe_globals([cls, PosixPath, WindowsPath])
        except (ImportError, AttributeError):
            # torch.serialization.add_safe_globals not available in older PyTorch versions
            pass


@dataclass(kw_only=True)
class AutoCompConfig(ITSerializableCfg):
    """Declares an auto-composition: the synthesized class's name, its mixins, and the adapters to search."""

    module_cfg_name: str
    module_cfg_mixin: list[Any] | Any
    target_adapters: AdapterSeq | None = None
    _orig_cfg_cls: type | None = None

    def __post_init__(self):
        if not isinstance(self.module_cfg_mixin, list):
            self.module_cfg_mixin = [self.module_cfg_mixin]
        if self.target_adapters is not None:
            self.target_adapters = adapter_seq_to_list(self.target_adapters)


@dataclass(kw_only=True)
class AutoCompConf(ITSerializableCfg):
    """Base for configs that may auto-compose themselves at construction time.

    When ``auto_comp_cfg`` is supplied, ``__new__`` synthesizes a subclass carrying the extra fields
    before instantiation -- so a config can accept adapter-specific kwargs it does not itself declare.
    """

    auto_comp_cfg: AutoCompConfig | None = None

    def __new__(cls, **kwargs):
        """Synthesize and instantiate a composed subclass when ``auto_comp_cfg`` is present, else the class
        itself."""
        if kwargs.get("auto_comp_cfg", None) is not None:
            built_class = AutoCompConf.compose_cfg_dataclass(cls, kwargs)
            return super().__new__(built_class)
        else:
            return super().__new__(cls)

    @staticmethod
    def compose_cfg_dataclass(target_cls, kwargs):
        """Build a dataclass composing ``target_cls`` with the resolved adapter config classes.

        Returns ``target_cls`` unchanged when no composition is needed, so the caller can always use the
        result without checking whether anything was synthesized.
        """
        setattr(kwargs["auto_comp_cfg"], "_orig_cfg_cls", target_cls)
        auto_comp_cfg = kwargs.pop("auto_comp_cfg")
        assert getattr(auto_comp_cfg, "_orig_cfg_cls", None) is not None, "`auto_comp_cfg` missing `_orig_cfg_cls`"
        composition_classes = resolve_composition_classes(auto_comp_cfg, kwargs)
        if not composition_classes:
            return target_cls
        built_class = make_dataclass(auto_comp_cfg.module_cfg_name, kwargs, bases=composition_classes, kw_only=True)
        built_class = type(auto_comp_cfg.module_cfg_name, (ComposedCfgWrapper, built_class), {})
        built_class.__module__ = "interpretune"
        built_class._orig_module_cfg_name = auto_comp_cfg._orig_cfg_cls.__qualname__
        built_class._composed_classes = composition_classes
        return built_class


def collect_exhaustive_attr_set(target_type: type) -> set[str]:
    """All public attribute names on a type, including inherited ones."""
    target_type_attrs = {attr for attr in dir(target_type) if not attr.startswith("__")}
    parent_attrs = set()
    for parent_cls in inspect.getmro(target_type)[1:]:
        parent_attrs.update(attr for attr in dir(parent_cls) if not attr.startswith("__"))
    dataclass_fields = (
        {field.name for field in fields(target_type)} if hasattr(target_type, "__dataclass_fields__") else set()
    )
    all_attrs = target_type_attrs.union(parent_attrs).union(dataclass_fields)
    return all_attrs


def candidate_subclass_attrs(kwargs: dict, target_type: type) -> dict:
    """Finds the keys in kwargs that are not attributes of target_type."""
    all_attrs = collect_exhaustive_attr_set(target_type)
    return {key: value for key, value in kwargs.items() if key not in all_attrs and not key.startswith("__")}


T = TypeVar("T")


def find_adapter_subclasses(
    target_type: type, target_adapters: AdapterSeq | None = None
) -> tuple[dict[Adapter, type], dict[Adapter, type]]:
    """Searches `interpretune.adapters` and `interpretune.config` for subclasses of `target_type` and returns them.

    If target_adapters is provided, only considers subclasses from the specified adapters.
    """
    subclasses, superclasses = {}, {}
    adapter_space = (
        adapter_seq_to_list(target_adapters) if target_adapters is not None else Adapter.__members__.values()
    )
    # Search each adapter's package and the submodules that define its composition and config classes.
    # Only modules ALREADY imported are considered, which is what keeps this pass from importing heavy
    # frameworks as a side effect of resolving a config.
    for template in AUTOCOMP_SEARCH_TEMPLATES:
        candidate_modules = {}
        for val in adapter_space:
            module_path = template.format(adapter=val.name)
            if module_path in sys.modules:
                candidate_modules[val] = (module_path, sys.modules[module_path])
        for adapter, (module_fqn, module) in candidate_modules.items():
            for _, member in inspect.getmembers(module, inspect.isclass):
                if member.__module__ != module_fqn:
                    continue
                if issubclass(member, target_type) and member is not target_type:
                    subclasses[adapter] = member
                elif issubclass(target_type, member):
                    superclasses[adapter] = member
    return subclasses, superclasses


def search_candidate_subclass_attrs(
    candidate_modules: dict[Adapter, type], kwargs_not_in_target_type: dict
) -> tuple[type, ...] | None:
    """Find the candidate classes covering the unmatched kwargs, preferring the least over-broad.

    Minimizing EXTRA attributes matters: several adapter configs may accept the given kwargs, and picking
    the narrowest avoids silently composing in a surface the caller never asked for.
    """
    valid_subclasses = []
    min_extra_attrs = float("inf")
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


def check_non_subclasses(target_class: type, candidate_classes: list[type]) -> tuple[type, ...] | None:
    """Return the candidates ``target_class`` does NOT already subclass, or None when it subclasses all."""
    unfullfilled_subclasses = []
    for cls in candidate_classes:
        if not issubclass(target_class, cls):
            unfullfilled_subclasses.append(cls)
    if unfullfilled_subclasses:
        return tuple(unfullfilled_subclasses)
    return


def issue_noncomposition_feedback(auto_comp_cfg, superclasses, subclasses):
    """Explain why no composition happened, at a severity matching whether the caller likely expected one.

    Debug when the config already satisfies everything asked of it; a warning when ``target_adapters``
    were given but nothing composable was found -- that case is usually a misconfiguration.
    """
    is_ready = f"already supports all of the provided kwargs, is already a subclass of {auto_comp_cfg.module_cfg_mixin}"
    base_message = f"No auto-composition needed for {auto_comp_cfg._orig_cfg_cls} as it {is_ready}"
    if not auto_comp_cfg.target_adapters:
        rank_zero_debug(f"{base_message} and no `target_adapters` were provided.")
    elif superclasses:
        rank_zero_debug(f"{base_message} and already is a subclass of a class in `target_adapters`.")
    elif not subclasses:
        rank_zero_warn(
            "No candidate classes in the specified `target_adapters` were found to further compose with."
            f"Since {auto_comp_cfg._orig_cfg_cls} {is_ready}, instantiating without auto-composition.",
            category=ITInstantiationFeedbackWarning,
        )


def issue_incomplete_composition_feedback(
    auto_comp_cfg: AutoCompConfig, kwargs_not_in_target_type: dict, nonsubcls_mixins: tuple[type, ...] | None
):
    """Warn that no composition covers all supplied kwargs, naming which kwargs went unmatched.

    Instantiation still proceeds with a partial composition where one exists, so the warning is the only signal that a
    kwarg may be silently unsupported -- it names them rather than saying composition failed.
    """
    no_auto_prefix = (
        f"Could not find an auto-composition for {auto_comp_cfg._orig_cfg_cls} that supports all of"
        f" the following kwargs: {kwargs_not_in_target_type}."
    )
    if nonsubcls_mixins:
        rank_zero_warn(
            f"{no_auto_prefix} Trying instantiation while composing with {nonsubcls_mixins}.",
            category=ITInstantiationFeedbackWarning,
        )
        assert auto_comp_cfg._orig_cfg_cls is not None
        return (auto_comp_cfg._orig_cfg_cls,) + nonsubcls_mixins
    else:
        rank_zero_warn(
            f"{no_auto_prefix} As {auto_comp_cfg._orig_cfg_cls} is already a subclass of "
            f"{auto_comp_cfg.module_cfg_mixin}, trying instantiation without further composition.",
            category=ITInstantiationFeedbackWarning,
        )
        return


def resolve_composition_classes(auto_comp_cfg: AutoCompConfig, kwargs: dict) -> tuple[type, ...] | None:
    """Decide which classes to compose for this config, or None when none are needed.

    Emits the feedback above rather than failing silently, since "nothing composed" is indistinguishable from
    "composition not needed" at the call site.
    """
    adapter_composition_classes = None
    assert auto_comp_cfg._orig_cfg_cls is not None
    subclasses, superclasses = find_adapter_subclasses(auto_comp_cfg._orig_cfg_cls, auto_comp_cfg.target_adapters)
    kwargs_not_in_target_type = candidate_subclass_attrs(kwargs, auto_comp_cfg._orig_cfg_cls)
    # Ensure module_cfg_mixin is a list of types
    mixin_list = (
        auto_comp_cfg.module_cfg_mixin
        if isinstance(auto_comp_cfg.module_cfg_mixin, list)
        else [auto_comp_cfg.module_cfg_mixin]
    )
    nonsubcls_mixins = check_non_subclasses(auto_comp_cfg._orig_cfg_cls, mixin_list)
    adapter_composition_classes = search_candidate_subclass_attrs(subclasses, kwargs_not_in_target_type)
    match bool(kwargs_not_in_target_type), bool(nonsubcls_mixins), bool(adapter_composition_classes):
        case (False, False, _):
            issue_noncomposition_feedback(auto_comp_cfg, superclasses, subclasses)
            return
        case (False, True, _):
            rank_zero_debug(
                f"{auto_comp_cfg._orig_cfg_cls} already supports all of the provided kwargs but needs to "
                f"be composed with {nonsubcls_mixins}."
            )
            if nonsubcls_mixins is None:
                return (auto_comp_cfg._orig_cfg_cls,)
            return (auto_comp_cfg._orig_cfg_cls,) + nonsubcls_mixins
        case (True, _, False):
            return issue_incomplete_composition_feedback(auto_comp_cfg, kwargs_not_in_target_type, nonsubcls_mixins)
        case (_, False, True):
            return adapter_composition_classes
        case (_, True, True):
            if nonsubcls_mixins is None or adapter_composition_classes is None:
                return adapter_composition_classes or nonsubcls_mixins
            return adapter_composition_classes + nonsubcls_mixins


################################################################################
# Core Shared Configuration for Datamodules and Modules
################################################################################


@dataclass(kw_only=True)
class ITSharedConfig(ITSerializableCfg):
    """Configuration shared by the datamodule and module halves: model, task, and tokenizer identity.

    These are the fields a ``shared_config`` block populates on both sides through the one merge site,
    so the two halves cannot disagree about which model or tokenizer a session is using.
    """

    model_name_or_path: str = ""
    task_name: str = ""
    tokenizer_name: str | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    os_env_model_auth_key: str | None = None
    tokenizer_id_overrides: dict | None = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    defer_model_init: bool | None = False

    def _validate_on_session_cfg_init(self):
        # deferred validation for attributes that my be set via shared datamodule/module config
        # type-checker directive used here since our ITSessionConfig is dynamically applying datamodule/module config
        if self.defer_model_init:
            assert self.signature_columns is not None, (  # pyright: ignore[reportAttributeAccessIssue]
                "`signature_columns` must be specified if `defer_model_init` is set to True"
            )
