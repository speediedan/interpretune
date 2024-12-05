from typing import Optional, Any, Dict, Tuple, Type, Set, Sequence, NamedTuple, List
from typing_extensions import override
from pprint import pformat
from pathlib import Path
from copy import deepcopy

from interpretune.utils.logging import rank_zero_debug, rank_zero_warn
from interpretune.utils.import_utils import instantiate_class
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import ITConfig
from interpretune.adapters.registration import Adapter, CompositionRegistry
from interpretune.base.contract.protocol import ModuleSteppable, DataModuleInitable
from tests.modules import (TestITDataModule, TestITModule)
from tests.base_defaults import BaseCfg

import yaml

from interpretune.base.components.cli import IT_BASE

# We define and 'register' basic example module configs here which are used in both example and parity acceptance tests.

DEFAULT_TEST_DATAMODULE = TestITDataModule
DEFAULT_TEST_MODULE = TestITModule

class RegisteredExampleCfg(NamedTuple):
    datamodule_cfg: ITDataModuleConfig
    module_cfg: ITConfig
    datamodule_cls: Type[DataModuleInitable] = DEFAULT_TEST_DATAMODULE
    module_cls: Type[ModuleSteppable] = DEFAULT_TEST_MODULE

class ModuleExampleRegistry(dict):
    def register(
        self,
        phase: str,
        model_src_key: str,
        task_name: str,
        adapter_combinations: Tuple[Adapter] | Tuple[Tuple[Adapter]],
        reg_key: str,
        registered_example_cfg: RegisteredExampleCfg,
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
            registered_example_cfg: Tuple[Callable],
            description : composition description
            cfg_dict: optionally save original configuration dictionary
        """
        supported_composition: Dict[str | Adapter | Tuple[Adapter | str], Tuple[Dict]] = {}
        supported_composition[reg_key] = registered_example_cfg
        supported_composition["description"] = description if description is not None else ""
        supported_composition["cfg_dict"] = cfg_dict
        self[reg_key] = supported_composition
        for a_combo in adapter_combinations:
            a_combo = (a_combo,) if not isinstance(a_combo, tuple) else a_combo
            composition_key = (model_src_key, task_name, phase, self.canonicalize_composition(a_combo))
            supported_composition[composition_key] = registered_example_cfg
            self[composition_key] = supported_composition

    def canonicalize_composition(self, adapter_ctx: Sequence[Adapter]) -> Tuple:
        return tuple(sorted(list(adapter_ctx), key=lambda a: a.value))

    def available_keys_feedback(self, target_key: str | Tuple) -> Set:
        assert isinstance(target_key, (str, tuple)), "`target_key` must be either a str or a tuple"
        return {key for key in self.keys() if isinstance(key, type(target_key))}

    def composition_keys(self) -> Set:
        return {key for key in self.keys() if isinstance(key, tuple)}

    @override
    def get(self, target: Tuple| BaseCfg | str) -> Any:
        if not isinstance(target, (tuple, str)):
            assert isinstance(target, BaseCfg), "`composition_key` must be either a tuple or a `BaseCfg`"
            # TODO: change "model_key" references to "task_key" to avoid confusion
            target = (target.model_src_key, target.model_key, target.phase,
                               target.adapter_ctx)
        if target in self:
            supported_composition = self[target]
            return supported_composition[target]

        available_keys_set = None
        if available_keys_set := self.available_keys_feedback(target):
            available_keys_set = pformat(sorted(available_keys_set))
        err_msg = (f"A test/example module registered with `{target}` was not found in the registry."
                   "\nAvailable valid modules:\n"
                   f"{available_keys_set}")
        raise KeyError(err_msg)

    def remove(self, composition_key: Tuple[Adapter | str]) -> None:
        """Removes the registered adapter composition by name."""
        del self[composition_key]

    def available_compositions(self, adapter_filter: Optional[Sequence[Adapter]| Adapter] = None) -> Set:
        """Returns a list of registered compositions, optionally filtering by an adapter or sequence of
        adapters."""
        if adapter_filter is not None:
            adapter_filter = CompositionRegistry.resolve_adapter_filter(adapter_filter)
            return {key for key in self.composition_keys() for subkey in key[3] if subkey in adapter_filter}
        return set(self.composition_keys())

    def __str__(self) -> str:
        return f"Registered Module Example Compositions: {pformat(self.keys())}"

MODULE_EXAMPLE_REGISTRY = ModuleExampleRegistry()

def gen_module_example_registry() -> None:
    yaml_reg_path = Path(IT_BASE) / "example_module_registry.yaml"
    with open(yaml_reg_path) as file:
      # Load the YAML file content
      data = yaml.safe_load(file)
      for reg_key, rv in data.items():
        try:
            instantiate_and_register(reg_key, rv)
        except Exception as e:  # we don't want to fail on a single example registration for any reason
            rank_zero_warn(f"Failed to register module example: {reg_key}. Exception: {e}")
            continue
        rank_zero_debug(f"Registered module example: {reg_key}")

def instantiate_and_register(reg_key: str, rv: Dict[str, Any]) -> None:
    datamodule_cls, module_cls = DEFAULT_TEST_DATAMODULE, DEFAULT_TEST_MODULE
    cfg_dict = deepcopy(rv)
    reg_info, shared_cfg, registered_example_cfg = rv['reg_info'], rv['shared_config'], rv['registered_example_cfg']
    reg_info['adapter_combinations'] = resolve_adapter_combinations(reg_info['adapter_combinations'])
    example_datamodule_cfg = itdm_cfg_factory(registered_example_cfg['datamodule_cfg'], shared_cfg)
    example_module_cfg = it_cfg_factory(registered_example_cfg['module_cfg'], shared_cfg)
    if datamodule_cls_path := registered_example_cfg.get('datamodule_cls', None):
        datamodule_cls = instantiate_class(init=datamodule_cls_path, import_only=True)
    registered_example = RegisteredExampleCfg(datamodule_cfg=example_datamodule_cfg, module_cfg=example_module_cfg,
                                              datamodule_cls=datamodule_cls, module_cls=module_cls)
    for supported_p in reg_info.get('supported_phases', ("train", "test", "predict")):
        reg_info['phase'] = supported_p
        MODULE_EXAMPLE_REGISTRY.register(**reg_info, reg_key=reg_key, registered_example_cfg=registered_example,
                                         cfg_dict=cfg_dict)

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
    if 'class_path' in c:  # if the dict directly contains a class_path key
        c = instantiate_class(c)  # with instantiating the class
    return c

def apply_example_defaults(cfg: ITConfig| ITDataModuleConfig, example_defaults: Dict, force_override: bool = False):
    for k, v in example_defaults.items():
        if not getattr(cfg, k) or force_override:
            setattr(cfg, k, v)

def itdm_cfg_factory(cfg: Dict, shared_config: Dict):
    prompt_cfg = cfg.get('prompt_cfg', {})
    # instantiate supported class_path refs
    # TODO: add path for specifying custom datamodule_cfg subclass when necessary
    if 'class_path' in prompt_cfg:
        cfg['prompt_cfg'] = instantiate_class(prompt_cfg)
    instantiated_cfg = ITDataModuleConfig(**shared_config, **cfg)
    apply_example_defaults(instantiated_cfg, example_datamodule_defaults) # update instantiated_cfg w/ example defaults
    return instantiated_cfg

def it_cfg_factory(cfg: Dict, shared_config: Dict):
    if 'class_path' in cfg:
        cfg['init_args'] = cfg['init_args'] | shared_config if 'init_args' in cfg else shared_config
        instantiated_cfg = instantiate_nested(cfg)
    else:
        instantiated_cfg = ITConfig(**cfg)
    apply_example_defaults(instantiated_cfg, example_itmodule_defaults) # update instantiated_cfg with example defaults
    return instantiated_cfg

##################################
# Core example config aliases
##################################

default_experiment_tag = 'test_itmodule'
example_datamodule_defaults = dict(prepare_data_map_cfg={"batched": True})
example_itmodule_defaults = dict(
    optimizer_init={"class_path": "torch.optim.AdamW",
                    "init_args": {"weight_decay": 1.0e-06, "eps": 1.0e-07, "lr": 3.0e-05}},
    lr_scheduler_init={"class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                       "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-06}})

#######################################
# Register Test/Example Module Configs
#######################################

gen_module_example_registry()
