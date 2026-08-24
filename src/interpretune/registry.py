from typing import (
    Any,
    Dict,
    Tuple,
    Type,
    Set,
    Sequence,
    NamedTuple,
    List,
    Callable,
    Protocol,
    runtime_checkable,
)
import warnings
import threading
from functools import partial

from typing_extensions import override
from pprint import pformat
from pathlib import Path
from copy import deepcopy
from tabulate import tabulate
from enum import Enum

from interpretune.utils import ITInstantiationFeedbackWarning, rank_zero_debug, rank_zero_warn, instantiate_class
from interpretune.config import ITDataModuleConfig, ITConfig
from interpretune.base import ITDataModule
from interpretune.adapters import ITModule
from interpretune.protocol import Adapter, ModuleSteppable, DataModuleInitable
from interpretune.adapter_registry import ADAPTER_REGISTRY

import yaml

DEFAULT_DATAMODULE = ITDataModule
DEFAULT_MODULE = ITModule
DEFAULT_MODULE_REGISTRY_PATH = Path(__file__).parent / "module_registry.yaml"


class RegKeyType(Enum):
    STRING = ""
    TUPLE = tuple()
    COMBO = tuple()


class RegisteredCfg(NamedTuple):
    datamodule_cfg: ITDataModuleConfig
    module_cfg: ITConfig
    datamodule_cls: Type[DataModuleInitable] = DEFAULT_DATAMODULE  # type: ignore[assignment]
    module_cls: Type[ModuleSteppable] = DEFAULT_MODULE  # type: ignore[assignment]


class RegisteredDataModuleCfg(NamedTuple):
    """The datamodule-only half of the two-path contract (#128).

    ``RegisteredCfg`` binds datamodule and module inseparably, which is right for task components. A
    standalone datamodule entry hydrates to this instead: no module coupling, so a datamodule can be
    addressed, fetched, and instantiated on its own (``ITSessionConfig`` accepts a pre-built
    datamodule, and datamodules instantiate before modules -- the tokenizer handshake).
    """

    datamodule_cfg: ITDataModuleConfig
    datamodule_cls: Type[DataModuleInitable] = DEFAULT_DATAMODULE  # type: ignore[assignment]


@runtime_checkable
class RegKeyQueryable(Protocol):
    model_src_key: str
    model_cfg_key: str
    adapter_ctx: Tuple


class ModuleRegistry(dict):  # type: ignore[type-arg]
    def register(
        self,
        model_src_key: str,
        model_cfg_key: str,
        adapter_combinations: tuple[Adapter] | tuple[tuple[Adapter]],
        reg_key: str,
        registered_cfg: RegisteredCfg,
        cfg_dict: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> None:
        """Registers valid component + adapter compositions mapped to composition keys with required metadata.

        Args:
            model_src_key: model source key (e.g. ``gpt2``)
            model_cfg_key: task/configuration key (e.g. ``rte``)
            adapter_combinations: tuple(s) identifying the valid adapter composition(s)
            reg_key: The canonical key of the test/example module.
            registered_cfg: the hydrated :class:`RegisteredCfg` for this entry
            description: composition description
            cfg_dict: optionally save original configuration dictionary
        """
        supported_composition: dict[str | Adapter | tuple[Adapter | str], Any] = {}
        supported_composition[reg_key] = registered_cfg
        supported_composition["description"] = description if description is not None else ""
        supported_composition["cfg_dict"] = cfg_dict
        self[reg_key] = supported_composition
        for a_combo in adapter_combinations:
            a_combo = (a_combo,) if not isinstance(a_combo, tuple) else a_combo
            composition_key = (model_src_key, model_cfg_key, self.canonicalize_composition(a_combo))
            supported_composition[composition_key] = registered_cfg  # type: ignore[assignment]
            self[composition_key] = supported_composition  # type: ignore[assignment]

    def canonicalize_composition(self, adapter_ctx: Sequence[Adapter]) -> Tuple:
        return tuple(sorted(list(adapter_ctx), key=lambda a: a.value))

    def available_keys(self, key_type: RegKeyType | str = "string") -> None:
        if isinstance(key_type, str):
            key_type = RegKeyType[key_type.upper()]
        print(self.available_keys_feedback(key_type.value))

    def available_keys_feedback(self, target_key: str | Tuple) -> str:
        assert isinstance(target_key, (str, tuple)), "`target_key` must be either a str or a tuple"
        # Collect entries as (displayable_key, description) and sort by the displayable key
        entries: list[tuple[str, str]] = []
        for key in self.keys():
            if not isinstance(key, type(target_key)):
                continue
            desc = self[key].get("description", "")
            # Convert key to a stable, human-readable string for sorting/display
            if isinstance(key, tuple):
                # Represent tuple keys in a compact, stable way
                key_str = "(" + ", ".join(map(str, key)) + ")"
            else:
                key_str = str(key)
            entries.append((key_str, desc))

        # Sort entries deterministically by the key string
        entries.sort(key=lambda it: it[0])

        if isinstance(target_key, str):
            return tabulate(entries, headers=["Key", "Description"])
        else:
            return tabulate(entries, headers=["(Model Src, Task Name, Adapter Ctx)", "Description"])

    def composition_keys(self) -> Set:
        return {key for key in self.keys() if isinstance(key, tuple)}

    @override
    def get(self, target: Tuple | str | RegKeyQueryable, default: Any = None) -> Any:
        if not isinstance(target, (tuple, str)):
            assert isinstance(target, RegKeyQueryable), (
                f"Non-string/non-tuple keys must be `RegKeyQueryable` (i.e. an object "
                "with at least these 3 attributes: `model_src_key`, `model_cfg_key`, `adapter_ctx`): but got "
                f"{type(target)}."
            )
            target = (target.model_src_key, target.model_cfg_key, target.adapter_ctx)
        try:
            if target in self:
                supported_composition = self[target]
                return supported_composition[target]
            else:
                raise KeyError
        except KeyError:
            if default is not None:
                return default
            # Get a nicely formatted, sorted table of available keys for the same key type
            available_keys_str = self.available_keys_feedback(target)
            err_msg = (
                f"A module registered with `{target}` was not found in the registry."
                "\nAvailable valid modules:\n"
                f"{available_keys_str}"
            )
            raise KeyError(err_msg)

    def remove(self, composition_key: tuple[Adapter | str]) -> None:
        """Removes the registered adapter composition by name."""
        del self[composition_key]

    def available_compositions(self, adapter_filter: Sequence[Adapter] | Adapter | None = None) -> Set:
        """Returns a list of registered compositions, optionally filtering by an adapter or sequence of
        adapters."""
        if adapter_filter is not None:
            adapter_filter = ADAPTER_REGISTRY.resolve_adapter_filter(adapter_filter)
            return {key for key in self.composition_keys() for subkey in key[2] if subkey in adapter_filter}
        return set(self.composition_keys())

    def __str__(self) -> str:
        return f"Registered Modules: {pformat(self.keys())}"


MODULE_REGISTRY = ModuleRegistry()


def instantiate_and_register(
    reg_key: str,
    rv: dict[str, Any],
    datamodule_cls: Type[DataModuleInitable] | str | None = None,
    module_cls: Type[ModuleSteppable] | str | None = None,
    target_registry: ModuleRegistry = MODULE_REGISTRY,
    itdm_cfg_defaults_fn: Callable | None = None,
    it_cfg_defaults_fn: Callable | None = None,
) -> None:
    cfg_dict = deepcopy(rv)
    reg_info, shared_cfg, registered_cfg = rv["reg_info"], rv["shared_config"], rv["registered_cfg"]
    reg_info["adapter_combinations"] = resolve_adapter_combinations(reg_info["adapter_combinations"])
    datamodule_cfg, module_cfg, datamodule_cls, module_cls = instantiate_or_import(
        registered_cfg, shared_cfg, itdm_cfg_defaults_fn, it_cfg_defaults_fn, datamodule_cls, module_cls
    )
    registered_cfg = RegisteredCfg(
        datamodule_cfg=datamodule_cfg,
        module_cfg=module_cfg,
        datamodule_cls=datamodule_cls,  # type: ignore[arg-type]
        module_cls=module_cls,  # type: ignore[arg-type]
    )
    target_registry.register(**reg_info, reg_key=reg_key, registered_cfg=registered_cfg, cfg_dict=cfg_dict)


def instantiate_or_import(
    registered_cfg, shared_cfg, itdm_cfg_defaults_fn, it_cfg_defaults_fn, datamodule_cls, module_cls
):
    datamodule_cfg = itdm_cfg_factory(registered_cfg["datamodule_cfg"], shared_cfg, defaults_func=itdm_cfg_defaults_fn)
    module_cfg = it_cfg_factory(registered_cfg["module_cfg"], shared_cfg, defaults_func=it_cfg_defaults_fn)
    if datamodule_cls_path := registered_cfg.get("datamodule_cls", None):
        datamodule_cls = instantiate_class(init=datamodule_cls_path, import_only=True)
    if module_cls_path := registered_cfg.get("module_cls", None):
        module_cls = instantiate_class(init=module_cls_path, import_only=True)
    return datamodule_cfg, module_cfg, datamodule_cls, module_cls


def gen_module_registry(
    yaml_reg_path: Path = DEFAULT_MODULE_REGISTRY_PATH, register_func: Callable = instantiate_and_register
) -> None:
    with open(yaml_reg_path, encoding="utf-8") as file:
        # Load the YAML file content
        data = yaml.safe_load(file)
        if not data:
            rank_zero_debug("No modules found to auto-register.")
            return
        # Bulk hydration instantiates EVERY registered entry, so per-entry config-normalization
        # feedback (tokenizer_name fallbacks, auto-composition notices, ...) would spam callers who
        # only requested a single entry (e.g. the first `MODULE_EXAMPLE_REGISTRY.get(...)` in a
        # notebook). Suppress exactly that categorized feedback here — direct config instantiation
        # outside bulk hydration still surfaces it, and all other warning categories (deprecations,
        # registration failures below) remain visible.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ITInstantiationFeedbackWarning)
            for reg_key, rv in data.items():
                try:
                    register_func(reg_key, rv)
                except Exception as e:  # we don't want to fail on a single example registration for any reason
                    rank_zero_warn(f"Failed to register module: {reg_key}. Exception: {e}")
                    continue
                rank_zero_debug(f"Registered module: {reg_key}")


def resolve_adapter_combinations(adapter_combinations: Sequence):
    registered_combinations = []
    for adps in adapter_combinations:
        if isinstance(adps, str):
            adps = (adps,)
        resolved_adapters = []
        for adp in adps:
            if not isinstance(adp, Adapter):
                adp = Adapter[adp]
            resolved_adapters.append(adp)
        registered_combinations.append(tuple(resolved_adapters))
    return tuple(registered_combinations)


def _declarative_field_names(cls: type) -> Set:
    """Field names whose DECLARED dataclass type says a plain dict IS the value (one-grammar recursion rule).

    ``class_path`` dicts instantiate recursively EXCEPT where the target field's declared type admits a plain
    dict — those fields are declarative (e.g. ``optimizer_init``: a directive instantiated later, with model
    params, at ``configure_optimizers`` time). Type-driven only, never a name list; ``Optional``/``Union``
    members are unwrapped; ``Any``-typed fields KEEP the instantiate behavior (the skip applies only where the
    type explicitly says dict), matching how jsonargparse already treated declared types.
    """
    import dataclasses
    import typing

    import re

    if not dataclasses.is_dataclass(cls):
        return set()
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        # unresolvable hints (TYPE_CHECKING-only forward refs, synthesized classes): the declared type
        # is still authoritative — read it textually from the stringized annotations instead
        hints = None
    names = set()
    dict_ann = re.compile(r"(?:^|[\[\s|,])(?:dict|Dict)(?:\[|$|[\s\]|,])")
    for f in dataclasses.fields(cls):
        declared = hints.get(f.name) if hints is not None else f.type
        if isinstance(declared, str):
            # genuinely stringized annotation that could not be resolved: read the declared type textually
            if dict_ann.search(declared):
                names.add(f.name)
            continue
        if _admits_plain_dict(declared):
            names.add(f.name)
    return names


def _admits_plain_dict(declared) -> bool:
    """True when the declared type is dict (possibly parameterized) or a Union admitting one.

    ``get_args`` on ``dict[str, Any]`` yields the TYPE PARAMETERS, not union members — only Unions
    unwrap; a parameterized dict is itself the answer.
    """
    import types
    import typing

    origin = typing.get_origin(declared) or declared
    if origin is dict:
        return True
    if origin in (typing.Union, types.UnionType):
        return any(_admits_plain_dict(member) for member in typing.get_args(declared))
    return False


def instantiate_nested(c: Dict | List, skip_keys: Set | None = None):
    skip_keys = skip_keys or set()
    if isinstance(c, dict) and "compose_ref" in c:
        # cross-repo prompt-config composition (design §11.5): one grammar, one extension point
        from interpretune.hub.promptconfigs import instantiate_prompt_cfg_node

        return instantiate_prompt_cfg_node(c)
    if isinstance(c, dict):
        child_skip: Set = set()
        if "class_path" in c:
            # resolve the TARGET class first so its declared field types govern which init_args
            # children are declarative dicts (skipped) vs nested directives (recursed)
            try:
                child_skip = _declarative_field_names(
                    instantiate_class({"class_path": c["class_path"]}, import_only=True)
                )
            except Exception:
                child_skip = set()
        for k, v in c.items():  # recursively instantiate nested directives
            if k in skip_keys:
                continue
            if isinstance(v, (dict, List)):
                c[k] = instantiate_nested(v, skip_keys=child_skip if k == "init_args" else None)
    elif isinstance(c, List):
        for i, v in enumerate(c):
            c[i] = instantiate_nested(c[i])
    if "class_path" in c:  # if the dict directly contains a class_path key
        c = instantiate_class(c, import_only=c.pop("import_only", False))  # type: ignore[arg-type]  # with instantiating the class
    return c


def apply_defaults(cfg: ITConfig | ITDataModuleConfig, defaults: Dict, force_override: bool = False):
    for k, v in defaults.items():
        if not getattr(cfg, k, None) or force_override:
            setattr(cfg, k, v)


def itdm_cfg_factory(cfg: Dict, shared_config: Dict, defaults_func: Callable | None = None):
    prompt_cfg = cfg.get("prompt_cfg", {})
    # instantiate supported class_path refs (compose_ref EXTENDS class_path instantiation: the
    # class_path task schema composes with a hub-referenced model-prompt definition, design §11.5)
    # TODO: add path for specifying custom datamodule_cfg subclass when necessary
    if "compose_ref" in prompt_cfg:
        from interpretune.hub.promptconfigs import instantiate_prompt_cfg_node

        cfg["prompt_cfg"] = instantiate_prompt_cfg_node(prompt_cfg)
    elif "class_path" in prompt_cfg:
        cfg["prompt_cfg"] = instantiate_class(prompt_cfg)
    instantiated_cfg = ITDataModuleConfig(**shared_config, **cfg)
    if defaults_func:
        defaults_func(instantiated_cfg)
    return instantiated_cfg


def it_cfg_factory(cfg: Dict, shared_config: Dict | None = None, defaults_func: Callable | None = None):
    if "class_path" in cfg:
        cfg["init_args"] = cfg["init_args"] | shared_config if "init_args" in cfg else shared_config
        instantiated_cfg = instantiate_nested(cfg)
    else:
        instantiated_cfg = ITConfig(**cfg)
    if defaults_func:
        defaults_func(instantiated_cfg)
    return instantiated_cfg


#######################################
# Register Module Configs
#######################################

gen_module_registry()


class ModuleHydrator:
    """Per-key hydration of component-tree configuration files into a :class:`ModuleRegistry`.

    Generic over the component root and the registration callable (hub design v3 §11.2 centralization):
    ``register_func_factory(registry)`` returns the callable applied to each ``(key, body)``. Manifest
    parsing and key parity live in ``interpretune.hub.manifest``; nothing here is example-specific.
    """

    def __init__(self, registry_root: Path | None = None, register_func_factory: Callable | None = None):
        self.registry_root = registry_root
        self._register_func_factory = register_func_factory
        self._index: dict[str, Path] | None = None
        self._hydrated: set[str] = set()
        self._registry = None
        self._lock = threading.RLock()

    @property
    def index(self) -> dict[str, Path]:
        """Key → configuration-file map assembled from the (small) component manifests; no entry construction."""
        from interpretune.hub.manifest import iter_component_manifests

        if self._index is None:
            with self._lock:
                if self._index is None:
                    index: dict[str, Path] = {}
                    for component_dir, manifest in iter_component_manifests(self.registry_root):
                        for key, rel in (manifest.get("module", {}).get("configs") or {}).items():
                            index[key] = component_dir / rel
                    self._index = index
        return self._index

    @property
    def registry(self) -> "ModuleRegistry":
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    self._registry = ModuleRegistry()
        return self._registry

    def _register_func(self):
        factory = self._register_func_factory or (
            lambda registry: partial(instantiate_and_register, target_registry=registry)
        )
        return factory(self.registry)

    def hydrate(self, key: str) -> bool:
        """Construct and register exactly one entry.

        Returns False if the key is not in any manifest.
        """
        if key in self._hydrated:
            return True
        if key not in self.index:
            return False
        with self._lock:
            if key in self._hydrated:
                return True
            from interpretune.hub.manifest import load_config_file

            parity_key, body = load_config_file(self.index[key], expected_key=key)
            self._register_func()(parity_key, body)
            self._hydrated.add(key)
        return True

    def hydrate_all(self) -> None:
        """Hydrate every indexed entry.

        Bulk hydration suppresses per-entry config-normalization feedback (`ITInstantiationFeedbackWarning`) just
        as the pre-decomposition bulk loader did — callers listing or tuple-resolving the registry did not ask for
        any single entry's feedback. Single-key `hydrate()` stays verbose: per-key construction is exactly the
        "directly instantiating a config" case where feedback should surface (interpretune#236).
        """
        from interpretune.utils import ITInstantiationFeedbackWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ITInstantiationFeedbackWarning)
            for key in self.index:
                self.hydrate(key)


class LazyModuleRegistry:
    """Mapping-style facade over per-key lazy hydration; ``builder`` swaps in an eager registry (tests)."""

    def __init__(
        self,
        builder: Callable[[], "ModuleRegistry"] | None = None,
        registry_root: Path | None = None,
        register_func_factory: Callable | None = None,
    ):
        self._builder = builder
        self._built = None
        self._hydrator = None if builder is not None else ModuleHydrator(registry_root, register_func_factory)
        self._lock = threading.RLock()

    @property
    def registry(self) -> "ModuleRegistry":
        """The underlying registry.

        Builder mode constructs eagerly; hydrator mode does NOT hydrate here.
        """
        if self._builder is not None:
            if self._built is None:
                with self._lock:
                    if self._built is None:
                        self._built = self._builder()
            return self._built
        assert self._hydrator is not None
        return self._hydrator.registry

    def _resolve(self, key) -> None:
        """Hydrate what a lookup needs: one entry for a known string key, everything for tuple/protocol keys."""
        if self._hydrator is None:
            return
        if isinstance(key, str):
            if not self._hydrator.hydrate(key):
                # unknown string key: hydrate all so the raised KeyError lists every available entry
                self._hydrator.hydrate_all()
        else:
            self._hydrator.hydrate_all()

    def get(self, target: Tuple | str | Any, default: Any = None) -> Any:
        self._resolve(target)
        return self.registry.get(target, default)

    def register(self, *args, **kwargs):
        return self.registry.register(*args, **kwargs)

    def __getitem__(self, key):
        self._resolve(key)
        return self.registry[key]

    def __setitem__(self, key, value):
        self.registry[key] = value

    def __contains__(self, key):
        if self._hydrator is not None and isinstance(key, str) and key in self._hydrator.index:
            return True
        self._resolve(key)
        return key in self.registry

    def _full_registry(self) -> "ModuleRegistry":
        if self._hydrator is not None:
            self._hydrator.hydrate_all()
        return self.registry

    def keys(self):
        return self._full_registry().keys()

    def values(self):
        return self._full_registry().values()

    def items(self):
        return self._full_registry().items()

    def __len__(self):
        return len(self._full_registry())

    def __str__(self):
        return str(self._full_registry())

    def __repr__(self):
        return repr(self._full_registry())

    # Forward other common methods
    def available_keys(self, *args, **kwargs):
        return self._full_registry().available_keys(*args, **kwargs)

    def available_keys_feedback(self, *args, **kwargs):
        return self._full_registry().available_keys_feedback(*args, **kwargs)

    def available_compositions(self, *args, **kwargs):
        return self._full_registry().available_compositions(*args, **kwargs)

    def remove(self, *args, **kwargs):
        return self.registry.remove(*args, **kwargs)
