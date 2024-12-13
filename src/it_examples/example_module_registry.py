from pathlib import Path
from functools import partial

from interpretune.base.registry import ModuleRegistry, gen_module_registry, instantiate_and_register, apply_defaults
from tests.modules import (TestITDataModule, TestITModule)

from interpretune.base.components.cli import IT_BASE

# We define and 'register' basic example module configs here which are used in both example and parity acceptance tests.

DEFAULT_TEST_DATAMODULE = TestITDataModule
DEFAULT_TEST_MODULE = TestITModule
DEFAULT_MODULE_EXAMPLE_REGISTRY_PATH = Path(IT_BASE) / "example_module_registry.yaml"
MODULE_EXAMPLE_REGISTRY = ModuleRegistry()

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

itdm_cfg_defaults = partial(apply_defaults, defaults=example_datamodule_defaults)
it_cfg_defaults = partial(apply_defaults, defaults=example_itmodule_defaults)

example_instantiate_and_register = partial(instantiate_and_register, datamodule_cls=DEFAULT_TEST_DATAMODULE,
                                          module_cls=DEFAULT_TEST_MODULE, target_registry=MODULE_EXAMPLE_REGISTRY,
                                          itdm_cfg_defaults_fn=itdm_cfg_defaults,
                                          it_cfg_defaults_fn=it_cfg_defaults)

gen_module_registry(yaml_reg_path=DEFAULT_MODULE_EXAMPLE_REGISTRY_PATH, register_func=example_instantiate_and_register)
