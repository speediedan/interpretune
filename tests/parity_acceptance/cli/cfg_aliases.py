from copy import deepcopy
from typing import Optional
from enum import auto
from dataclasses import dataclass

from interpretune.base.config.shared import AutoStrEnum
from tests.configuration import get_nested, set_nested
from tests.parity_acceptance.plugins.transformer_lens.cfg_aliases import (test_tl_datamodule_kwargs,
                                                                          test_tl_gpt2_shared_config,
                                                                          test_tl_cust_config_it_module_kwargs)
from tests.parity_acceptance.base.cfg_aliases import (test_optimizer_init, test_lr_scheduler_init,
                                                      core_cust_shared_config, test_core_cust_it_module_kwargs,
                                                      core_cust_datamodule_kwargs)
from tests.utils.warns import CORE_CTX_WARNS, LIGHTING_CTX_WARNS, TL_CTX_WARNS, TL_LIGHTNING_CTX_WARNS
from interpretune.base.contract.session import Framework, Plugin

from tests.configuration import BaseAugTest
from tests.utils.warns import EXPECTED_WARNS, HF_EXPECTED_WARNS, TL_EXPECTED_WARNS


class CLI_TESTS(AutoStrEnum):
    core_tl_test = auto()
    core_tl_norun = auto()
    core_optim_train = auto()
    l_tl_test = auto()
    l_tl_norun = auto()
    l_optim_fit = auto()

# tests currently use only a single experiment and custom model but use a variety of configurations
CLI_EXP_MODEL = ("cust_test", "cust")

EXAMPLE_WARNS = EXPECTED_WARNS + HF_EXPECTED_WARNS + TL_EXPECTED_WARNS

CLI_EXPECTED_WARNS = {
    # framework_cli, plugin_ctx
    (Framework.core, None): CORE_CTX_WARNS,
    (Framework.lightning, None): LIGHTING_CTX_WARNS,
    (Framework.lightning, Plugin.transformer_lens): TL_LIGHTNING_CTX_WARNS,
    (Framework.core, Plugin.transformer_lens): TL_CTX_WARNS
}

RUN_FN = "run_experiment.py"

@dataclass(kw_only=True)
class CLICfg:
    framework_cli: Framework = Framework.core
    run: Optional[str] = None
    compose_cfg: bool = False
    plugin_ctx: Optional[Plugin] = None
    debug_mode: bool = False


TEST_CONFIGS_CLI = (
    BaseAugTest(alias=CLI_TESTS.core_tl_test.value, cfg=CLICfg(compose_cfg=True, run="test", debug_mode=True,
                                                           plugin_ctx=Plugin.transformer_lens)),
    BaseAugTest(alias=CLI_TESTS.core_tl_norun.value, cfg=CLICfg(plugin_ctx=Plugin.transformer_lens),
            expected={'hasattr': 'tl_config_model_init'}),
    BaseAugTest(alias=CLI_TESTS.core_optim_train.value, cfg=CLICfg(compose_cfg=True, run="train", debug_mode=True)),
    BaseAugTest(alias=CLI_TESTS.l_tl_test.value, cfg=CLICfg(framework_cli=Framework.lightning, run="test",
                                                        plugin_ctx=Plugin.transformer_lens), marks="lightning"),
    BaseAugTest(alias=CLI_TESTS.l_tl_norun.value, cfg=CLICfg(framework_cli=Framework.lightning,
                                                         plugin_ctx=Plugin.transformer_lens), marks="l_optional",
                                                         expected={'hasattr': 'tl_config_model_init'}),
    BaseAugTest(alias=CLI_TESTS.l_optim_fit.value, cfg=CLICfg(compose_cfg=True, run="fit", debug_mode=True,
                                                          framework_cli=Framework.lightning)),
)

EXPECTED_RESULTS_CLI = {cfg.alias: cfg.expected for cfg in TEST_CONFIGS_CLI}


########################################################################################################################
# Composable CLI config aliases
# We use these configuration definitions to dynamically build composable CLI test configuration files that we write
# to a session-scoped test directory and subsequently read during relevant CLI tests to validate expected CLI
# configuration composition
########################################################################################################################
# TODO: Use more granular composable aliases for these configs to improve efficiency

cli_cfgs = {}
cli_cfgs["exp_cfgs"] = {}

# note we use the same datamodule and module cls for all test contexts
default_session_cfg = {
    "datamodule_cls": {"class_path": "tests.modules.TestITDataModule"},
    "datamodule_cfg": {"class_path": "interpretune.base.config.datamodule.ITDataModuleConfig"},
    "module_cls": {"class_path": "tests.modules.TestITModule"},
    "module_cfg": {"class_path": "interpretune.base.config.module.ITConfig"}
}

seed_cfg = {"seed_everything": 42}

################################################################################
# global core framework config file
################################################################################

default_trainer_kwargs = {
    "limit_train_batches": 1,
    "limit_val_batches": 1,
    "limit_test_batches": 1,
    "max_steps": -1,
    "max_epochs": 1,
}

cli_cfgs["global_core"] = {**seed_cfg,
    "session_cfg": {"framework_ctx": "core", **default_session_cfg},
    "trainer_cfg": {**default_trainer_kwargs},
    }

################################################################################
# global Lightning framework config file
################################################################################

cli_cfgs["global_lightning"] = {**seed_cfg,
    "session_cfg": {"framework_ctx": "lightning", **default_session_cfg},
    "trainer": {
        **default_trainer_kwargs,
        "accelerator": "gpu",
        "strategy": "auto",
        "logger": {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {"save_dir": "lightning_logs", "name": "core_default"},
        },
    },
}


################################################################################
# global debug config file
################################################################################

cli_cfgs["global_debug"] = set_nested("session_cfg.module_cfg.init_args")

get_nested(cli_cfgs["global_debug"], "session_cfg.module_cfg.init_args")["debug_lm_cfg"] = {
    "class_path": "interpretune.analysis.debug_generation.DebugLMConfig",
    "init_args": {"enabled": True, "raw_debug_sequences": ["How many days in a week?", "How old is Barack Obama?",],},
}

################################################################################
# global transformer_lens plugin config file
################################################################################

del test_tl_datamodule_kwargs['text_fields']  # no need to update this default and it causes serialization errors
## override with appropriate CLI class_path config since we're serializing and loading from yaml
test_tl_datamodule_kwargs["prompt_cfg"] = {"class_path":
                                           "it_examples.experiments.rte_boolq.config.RTEBoolqPromptConfig"}

cli_cfgs["global_tl"] = {}
cli_cfgs["global_tl"]["session_cfg"] = {
    "plugin_ctx": "transformer_lens", "datamodule_cfg": {"init_args": {**test_tl_gpt2_shared_config,
                                                                       **test_tl_datamodule_kwargs}},
}

################################################################################
# model-level transformer_lens plugin config file (only using one model for now)
################################################################################

test_tl_cust_config_it_module_kwargs_zero_shot = deepcopy(test_tl_cust_config_it_module_kwargs)
# override zero_shot_cfg with serializable CLI args
test_tl_cust_config_it_module_kwargs_zero_shot["zero_shot_cfg"] = {
    "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqZeroShotClassificationConfig",
    "init_args": {"enabled": True,
                  "lm_generation_cfg": {
                      "class_path": "interpretune.plugins.transformer_lens.TLensGenerationConfig",
                      "init_args": {"max_new_tokens": 1}
                    }
    },
}
# add necessary class_path/init_args for provided custom config
test_tl_cust_config_it_module_kwargs_zero_shot['tl_cfg'] = {
    'class_path': 'interpretune.plugins.transformer_lens.ITLensCustomConfig',
    'init_args': {'cfg': test_tl_cust_config_it_module_kwargs_zero_shot['tl_cfg']['cfg']}
}

cli_cfgs["model_tl_cfg"] = {}
cli_cfgs["model_tl_cfg"]["session_cfg"] = {
    "module_cfg": {
        "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqTLConfig",
        "init_args": { **test_tl_cust_config_it_module_kwargs_zero_shot}
    }
}


################################################################################
# model-level core framework config file
################################################################################

test_core_cust_it_module_kwargs_zero_shot = deepcopy(test_core_cust_it_module_kwargs)
test_core_cust_it_module_kwargs_zero_shot["zero_shot_cfg"] = {
    "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqZeroShotClassificationConfig",
    "init_args": {"enabled": True, "lm_generation_cfg": {
        "class_path": "tests.parity_acceptance.base.cfg_aliases.ToyGenCfg",
        "init_args": {"max_new_tokens": 2}},
        },
}
core_cust_datamodule_kwargs_prompt_cfg = deepcopy(core_cust_datamodule_kwargs)
## override with appropriate CLI class_path config since we're serializing and loading from yaml
core_cust_datamodule_kwargs_prompt_cfg["prompt_cfg"] = {"class_path":
                                             "it_examples.experiments.rte_boolq.config.RTEBoolqPromptConfig"}

del core_cust_datamodule_kwargs_prompt_cfg['text_fields']  # don't need to change default, causes serialization errors

# create a deepcopy for likely future manipulation
core_cust_shared_config_cli = deepcopy(core_cust_shared_config)

cli_cfgs["model_cfgs"] = {}
cli_cfgs["model_cfgs"]["session_cfg"] = {
    "datamodule_cfg": {"init_args": {**core_cust_shared_config_cli, **core_cust_datamodule_kwargs_prompt_cfg}},
    "module_cfg": {
        "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqConfig",
        "init_args": {**test_core_cust_it_module_kwargs_zero_shot},
    },
}

test_core_cus_it_model_cfg_cuda_bf16 = deepcopy(test_core_cust_it_module_kwargs['model_cfg'])
test_core_cus_it_model_cfg_cuda_bf16.update({"dtype": "bfloat16", "device": "cuda"})

################################################################################
# core framework training with no transformer_lens plugin context
################################################################################

cli_cfgs["exp_cfgs"][CLI_TESTS.core_optim_train] = set_nested("session_cfg.module_cfg")
cli_cfgs["exp_cfgs"][CLI_TESTS.core_optim_train]["session_cfg"]["module_cfg"]["init_args"] = {
    "experiment_tag": CLI_TESTS.core_optim_train.value,
    **test_lr_scheduler_init, **test_optimizer_init,
}

################################################################################
# Lightning framework testing with transformer_lens plugin context w/ cuda, bf16
################################################################################
test_tl_cust_config_it_module_kwargs_cuda = deepcopy(test_tl_cust_config_it_module_kwargs)
test_tl_cust_config_it_module_kwargs_cuda["tl_cfg"]["cfg"].update({"device": "cuda"})

test_tl_cust_config_it_module_kwargs_cuda_bf16 = deepcopy(test_tl_cust_config_it_module_kwargs_cuda)
test_tl_cust_config_it_module_kwargs_cuda_bf16["tl_cfg"]["cfg"].update({"dtype": "bfloat16"})

cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_test] = {
    "session_cfg": { "module_cfg": {"init_args": {
        "experiment_tag": CLI_TESTS.l_tl_test.value,
        "tl_cfg": {**test_tl_cust_config_it_module_kwargs_cuda_bf16["tl_cfg"]},
        }
    }},
    "trainer": {
        "precision": "bf16-true",
        "accumulate_grad_batches": 1,
        "accelerator": "gpu",
        "strategy": "auto",
        "num_sanity_val_steps": 0,
        "devices": 1,
        "logger": {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {"name": CLI_TESTS.l_tl_test.value},
        },
    },
}

################################################################################
# lightning framework fit with no transformer_lens plugin context, cuda bf16
################################################################################

cli_cfgs["exp_cfgs"][CLI_TESTS.l_optim_fit] = set_nested("session_cfg.module_cfg")
cli_cfgs["exp_cfgs"][CLI_TESTS.l_optim_fit]["session_cfg"]["module_cfg"]["init_args"] = {
    "experiment_tag": CLI_TESTS.l_optim_fit.value,
    **test_lr_scheduler_init, **test_optimizer_init,
}
cli_cfgs["exp_cfgs"][CLI_TESTS.l_optim_fit]["trainer"] = deepcopy(cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_test]["trainer"])
cli_cfgs["exp_cfgs"][CLI_TESTS.l_optim_fit]["trainer"]["logger"]["init_args"]["name"] = CLI_TESTS.l_optim_fit.value
get_nested(cli_cfgs["exp_cfgs"][CLI_TESTS.l_optim_fit], "session_cfg.module_cfg.init_args")["model_cfg"] = \
    test_core_cus_it_model_cfg_cuda_bf16

################################################################################
# core framework test with transformer_lens plugin context, cuda float32
################################################################################

test_tl_cust_config_it_module_kwargs_cuda_float32 = deepcopy(test_tl_cust_config_it_module_kwargs_cuda)
test_tl_cust_config_it_module_kwargs_cuda_float32["tl_cfg"]["cfg"].update({"dtype": "float32"})
cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_test] = set_nested("session_cfg.module_cfg")
cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_test]["session_cfg"]["module_cfg"]["init_args"] = {
    "experiment_tag": CLI_TESTS.core_tl_test.value,
    "tl_cfg": {**test_tl_cust_config_it_module_kwargs_cuda_float32["tl_cfg"]},
}

################################################################################
# core framework test with transformer_lens plugin, return cli w/o run
################################################################################

# TODO: if more of these aliased tests are needed, use a factory function to gen the config
# `core_tl_norun` is the same as `core_tl_test` but validating core CLI instantiation-only mode
cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_norun] = deepcopy(cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_test])
cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_norun]["session_cfg"]["module_cfg"]["init_args"]["experiment_tag"] \
    = CLI_TESTS.core_tl_norun.value

################################################################################
# lightning framework test with transformer_lens plugin, return cli w/o run
################################################################################

# similarly `l_tl_norun` is the same as `l_tl_test` but validating the lightning CLI instantiation-only mode
cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_norun] = deepcopy(cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_test])
cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_norun]["session_cfg"]["module_cfg"]["init_args"]["experiment_tag"] \
    = CLI_TESTS.l_tl_norun.value
