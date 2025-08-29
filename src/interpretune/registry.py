from typing import (
    Optional,
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
from typing_extensions import override
from pprint import pformat
from pathlib import Path
from copy import deepcopy
from tabulate import tabulate
from enum import Enum

from interpretune.utils import rank_zero_debug, rank_zero_warn, instantiate_class
from interpretune.config import ITDataModuleConfig, ITConfig
from interpretune.base import ITDataModule
from interpretune.adapters import ITModule
from interpretune.protocol import AllPhases
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


@runtime_checkable
class RegKeyQueryable(Protocol):
    model_src_key: str
    model_key: str
    phase: str
    adapter_ctx: Tuple


class ModuleRegistry(dict):  # type: ignore[type-arg]
    def register(
        self,
        phase: str,
        model_src_key: str,
        task_name: str,
        adapter_combinations: Tuple[Adapter] | Tuple[Tuple[Adapter]],
        reg_key: str,
        registered_cfg: RegisteredCfg,
        cfg_dict: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Registers valid component + adapter compositions mapped to composition keys with required metadata.

        Args:
            lead_adapter: The adapter registering this set of valid compositions (e.g. LightningAdapter)
            phase:
            model_src_key:
            task_name:
            adapter_combination: tuple identifying the valid adapter composition
            reg_key: The canonical key of the test/example module.
            registered_cfg: Tuple[Callable],
            description : composition description
            cfg_dict: optionally save original configuration dictionary
        """
        supported_composition: Dict[str | Adapter | Tuple[Adapter | str], Any] = {}
        supported_composition[reg_key] = registered_cfg
        supported_composition["description"] = description if description is not None else ""
        supported_composition["cfg_dict"] = cfg_dict
        self[reg_key] = supported_composition
        for a_combo in adapter_combinations:
            a_combo = (a_combo,) if not isinstance(a_combo, tuple) else a_combo
            composition_key = (model_src_key, task_name, phase, self.canonicalize_composition(a_combo))
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
        entries: List[Tuple[str, str]] = []
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
            return tabulate(entries, headers=["(Model Src, Task Name, Phase, Adapter Ctx)", "Description"])

    def composition_keys(self) -> Set:
        return {key for key in self.keys() if isinstance(key, tuple)}

    @override
    def get(self, target: Tuple | str | RegKeyQueryable, default: Any = None) -> Any:
        if not isinstance(target, (tuple, str)):
            assert isinstance(target, RegKeyQueryable), (
                f"Non-string/non-tuple keys must be `RegKeyQueryable` (i.e. an object "
                "with at least these 4 attributes: `model_src_key`, `model_key`, `phase`, `adapter_ctx`): but got "
                f"{type(target)}."
            )
            # TODO: change "model_key" references to "task_key" to avoid confusion
            target = (target.model_src_key, target.model_key, target.phase, target.adapter_ctx)
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

    def remove(self, composition_key: Tuple[Adapter | str]) -> None:
        """Removes the registered adapter composition by name."""
        del self[composition_key]

    def available_compositions(self, adapter_filter: Optional[Sequence[Adapter] | Adapter] = None) -> Set:
        """Returns a list of registered compositions, optionally filtering by an adapter or sequence of
        adapters."""
        if adapter_filter is not None:
            adapter_filter = ADAPTER_REGISTRY.resolve_adapter_filter(adapter_filter)
            return {key for key in self.composition_keys() for subkey in key[3] if subkey in adapter_filter}
        return set(self.composition_keys())

    def __str__(self) -> str:
        return f"Registered Modules: {pformat(self.keys())}"


MODULE_REGISTRY = ModuleRegistry()


def instantiate_and_register(
    reg_key: str,
    rv: Dict[str, Any],
    datamodule_cls: Optional[Type[DataModuleInitable] | str] = None,
    module_cls: Optional[Type[ModuleSteppable] | str] = None,
    target_registry: ModuleRegistry = MODULE_REGISTRY,
    itdm_cfg_defaults_fn: Optional[Callable] = None,
    it_cfg_defaults_fn: Optional[Callable] = None,
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
    for supported_p in AllPhases:
        reg_info["phase"] = supported_p.value
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


def instantiate_nested(c: Dict | List):
    if isinstance(c, dict):
        for k, v in c.items():  # recursively instantiate nested directives
            if isinstance(v, (dict, List)):
                c[k] = instantiate_nested(v)
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


def itdm_cfg_factory(cfg: Dict, shared_config: Dict, defaults_func: Optional[Callable] = None):
    prompt_cfg = cfg.get("prompt_cfg", {})
    # instantiate supported class_path refs
    # TODO: add path for specifying custom datamodule_cfg subclass when necessary
    if "class_path" in prompt_cfg:
        cfg["prompt_cfg"] = instantiate_class(prompt_cfg)
    instantiated_cfg = ITDataModuleConfig(**shared_config, **cfg)
    if defaults_func:
        defaults_func(instantiated_cfg)
    return instantiated_cfg


def it_cfg_factory(cfg: Dict, shared_config: Optional[Dict] = None, defaults_func: Optional[Callable] = None):
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
