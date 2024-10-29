import os
from copy import deepcopy
from enum import auto
from pathlib import Path

from interpretune.adapters.registration import Adapter
from interpretune.base.config.module import HFFromPretrainedConfig
from interpretune.base.config.shared import AutoStrEnum
from interpretune.extensions.memprofiler import MemProfilerCfg, MemProfilerSchedule
from it_examples.example_module_registry import (core_cust_shared_config, test_tl_datamodule_cfg, test_optimizer_init,
                                       test_lr_scheduler_init, test_tl_signature_columns)
from base_defaults import default_prof_bs
from tests.modules import TestFTS
from tests.utils import get_nested, set_nested


# TODO: add model-specific mapping and mapping functions here to dedup some of the shared explicit mapping logic here
################################################################################
# Adapter Context cfg aliases
################################################################################
w_lit = {"adapter_ctx": (Adapter.lightning,)}
w_l_tl = {"adapter_ctx": (Adapter.lightning, Adapter.transformer_lens)}

################################################################################
# Device and Precision cfg aliases
################################################################################
cuda = {"device_type": "cuda"}
bf16 = {"precision": "bf16"}
cuda_bf16 = {**cuda, **bf16}
cuda_bf16_l = {**cuda, **bf16, **w_lit}

################################################################################
# Extension cfg aliases
################################################################################
enable_memprofiler_kwargs = {"enabled": True, "cuda_allocator_history": True}
bs_override = {'train_batch_size': default_prof_bs, 'eval_batch_size': default_prof_bs}
memprofiler_cfg = MemProfilerCfg(**enable_memprofiler_kwargs)
no_savedt_memprofiler_cfg = MemProfilerCfg(**enable_memprofiler_kwargs, enable_saved_tensors_hooks=False)
warm_maxstep_memprof_cfg = MemProfilerCfg(**enable_memprofiler_kwargs,
                                      **{"schedule": MemProfilerSchedule(warmup_steps=2, max_step=4)})
nowarm_maxstep_memprof_cfg = MemProfilerCfg(**enable_memprofiler_kwargs,
                                                 **{"schedule": MemProfilerSchedule(max_step=4)})
nowarm_maxstep_hk_memprof_cfg = MemProfilerCfg(retain_hooks_for_funcs=["training_step"], **enable_memprofiler_kwargs,
                                                 **{"schedule": MemProfilerSchedule(max_step=4)})
memprof_steps = {"limit_train_batches": 5, "limit_val_batches": 3}
bs1_memprof_steps = {"dm_override_cfg": bs_override, **memprof_steps}
test_bs1_mem = {"phase": "test", "dm_override_cfg": bs_override, "memprofiler_cfg": memprofiler_cfg}
test_bs1_mem_nosavedt = {**test_bs1_mem, "memprofiler_cfg": no_savedt_memprofiler_cfg}
bs1_warm_mem = {**bs1_memprof_steps,  "memprofiler_cfg": warm_maxstep_memprof_cfg}
bs1_nowarm_mem = {**bs1_memprof_steps, "memprofiler_cfg": nowarm_maxstep_memprof_cfg}
bs1_nowarm_hk_mem = {**bs1_memprof_steps, "memprofiler_cfg": nowarm_maxstep_hk_memprof_cfg}

################################################################################
# Feature-specific cfg aliases
################################################################################
act_ckpt = {
    "hf_from_pretrained_cfg": HFFromPretrainedConfig(
        pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"}, model_head="transformers.GPT2LMHeadModel",
        activation_checkpointing=True)}
cuda_act = {**cuda, **act_ckpt}

##################################
# FTS config aliases
##################################

default_test_fts_kwargs = {"max_depth": -1}
l_gpt2_explicit_sched = {"fts_schedule_key": ("l_gpt2", "basic_explicit")}
l_tl_gpt2_explicit_sched = {"fts_schedule_key": ("l_tl_gpt2", "basic_explicit")}
l_ctx = {"adapter_ctx": (Adapter.lightning,)}
default_fts_cfg = {"callback_cfgs": {TestFTS: default_test_fts_kwargs}}
l_gpt2_fts = {**default_fts_cfg, **l_gpt2_explicit_sched, **l_ctx}
l_tl_gpt2_fts = {**default_fts_cfg, **l_tl_gpt2_explicit_sched}


########################################################################################################################
# Composable CLI config aliases
# We use these configuration definitions to dynamically build composable CLI test configuration files that we write
# to a session-scoped test directory and subsequently read during relevant CLI tests to validate expected CLI
# configuration composition
########################################################################################################################
# TODO: Use more granular composable aliases for these configs to improve efficiency
# tests currently use only a single experiment and custom model but use a variety of configurations
CLI_EXP = "cust_test"
RUN_FN = "run_experiment.py"
IT_HOME = Path(os.environ.get("IT_HOME", Path(__file__).parent.parent.parent / "src" / "interpretune"))

class CLI_TESTS(AutoStrEnum):
    core_tl_test = auto()
    core_tl_test_noharness = auto()
    core_tl_norun = auto()
    core_optim_train = auto()
    l_tl_test = auto()
    l_tl_norun = auto()
    l_tl_norun_noharness = auto()
    l_optim_fit = auto()

################################################################################
# CLI config definitions
################################################################################

### nested cfg aliases

mod_cfg = "session_cfg.module_cfg"
datamod_cfg = "session_cfg.datamodule_cfg"
mod_initargs = f"{mod_cfg}.init_args"
tl_cfg_initargs = f"{mod_initargs}.tl_cfg.init_args"


parity_cli_cfgs = {}
parity_cli_cfgs["exp_cfgs"] = {}

default_trainer_kwargs = {
    "limit_train_batches": 1,
    "limit_val_batches": 1,
    "limit_test_batches": 1,
    "max_steps": -1,
    "max_epochs": 1,
}

default_seed_cfg = {"seed_everything": 42}

# note we use the same datamodule and module cls for all test contexts
default_session_cfg = {
    "datamodule_cls": "tests.modules.TestITDataModule",
    "datamodule_cfg": {"class_path": "interpretune.base.config.datamodule.ITDataModuleConfig"},
    "module_cls": "tests.modules.TestITModule",
    "module_cfg": {"class_path": "interpretune.base.config.module.ITConfig"}
}

test_core_cust_it_module_kwargs_zero_shot = {
    "model_cfg": {"device": "cpu", "dtype": "float32", "model_args": {"max_seq_len": 200}},
    "zero_shot_cfg": {"class_path": "it_examples.experiments.rte_boolq.RTEBoolqZeroShotClassificationConfig",
                      "init_args": {"enabled": True, "lm_generation_cfg": {
                          "class_path": "tests.utils.ToyGenCfg",
                          "init_args": {"max_new_tokens": 2}},
                       },
    }
}

## override with appropriate CLI class_path config since we're serializing and loading from yaml
core_cust_datamodule_cfg_prompt_cfg = { "prompt_cfg":
                                       {"class_path": "it_examples.experiments.rte_boolq.RTEBoolqPromptConfig"}}

# create a deepcopy for likely future manipulation
core_cust_shared_config_cli = deepcopy(core_cust_shared_config)
test_core_cust_it_model_cfg_cuda_bf16 = {"device": "cuda", "dtype": "bfloat16", "model_args": {"max_seq_len": 200}}

# override zero_shot_cfg with serializable CLI args
test_tl_cust_config_it_module_kwargs_zero_shot = {
    "zero_shot_cfg": {
        "class_path": "it_examples.experiments.rte_boolq.RTEBoolqZeroShotClassificationConfig",
        "init_args": {"enabled": True,
                      "lm_generation_cfg": {
                          "class_path": "interpretune.adapters.transformer_lens.TLensGenerationConfig",
                          "init_args": {"max_new_tokens": 1}
                    }
                }
    },
    "tl_cfg": {
        'class_path': 'interpretune.adapters.transformer_lens.ITLensCustomConfig',
        'init_args': {'cfg': {"n_layers":1, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
                              "act_fn":'relu', "tokenizer_name": 'gpt2'}
        }
    }
}


base_tl_cust_model_cfg = {}
base_tl_cust_model_cfg["session_cfg"] = {
    "module_cfg": {
        "class_path": "it_examples.experiments.rte_boolq.RTEBoolqTLConfig",
        "init_args": { **test_tl_cust_config_it_module_kwargs_zero_shot}
    }
}

base_lightning_trainer_cfg = {
    "trainer": {
        **default_trainer_kwargs,
        "devices": 1,
        "accelerator": "gpu",
        "strategy": "auto",
        "logger": {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {"save_dir": "lightning_logs", "name": "core_default"},
        },
    },
}

base_cust_rte_cfg = {}
base_cust_rte_cfg["session_cfg"] = {
    "datamodule_cfg": {"init_args": {**core_cust_shared_config_cli, **core_cust_datamodule_cfg_prompt_cfg}},
    "module_cfg": {
        "class_path": "it_examples.experiments.rte_boolq.RTEBoolqConfig",
        "init_args": {**test_core_cust_it_module_kwargs_zero_shot},
    },
}

test_tl_datamodule_cfg.text_fields = None
## override with appropriate CLI class_path config since we're serializing and loading from yaml
test_tl_datamodule_cfg.prompt_cfg = {"class_path": "it_examples.experiments.rte_boolq.RTEBoolqPromptConfig"}

################################################################################
# global default cfg file
################################################################################

default_cfg = {** default_seed_cfg,
    "session_cfg": {**default_session_cfg},
    }

# note while we define global defaults, we explicitly set defaults in tests rather than inherit them clarity
parity_cli_cfgs["global_defaults"] = default_cfg

################################################################################
# global debug config file
################################################################################

parity_cli_cfgs["global_debug"] = set_nested("session_cfg.module_cfg.init_args")

get_nested(parity_cli_cfgs["global_debug"], "session_cfg.module_cfg.init_args")["debug_lm_cfg"] = {
    "class_path": "interpretune.extensions.debug_generation.DebugLMConfig",
    "init_args": {"enabled": True, "raw_debug_sequences": ["How many days in a week?", "How old is Barack Obama?",],},
}

################################################################################
# core adapter training with no transformer_lens adapter context
################################################################################

core_optim_train = deepcopy(default_cfg)
core_optim_train["session_cfg"].update({"module_cfg": deepcopy(base_cust_rte_cfg["session_cfg"]["module_cfg"]),
                         "datamodule_cfg": deepcopy(base_cust_rte_cfg["session_cfg"]["datamodule_cfg"])})
core_optim_train["trainer_cfg"] = deepcopy(default_trainer_kwargs)
get_nested(core_optim_train, mod_cfg)["init_args"] = {"experiment_tag": CLI_TESTS.core_optim_train.value,
                                                      **test_core_cust_it_module_kwargs_zero_shot,
                                                      **test_lr_scheduler_init, **test_optimizer_init}
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.core_optim_train] = core_optim_train

################################################################################
# core adapter test with transformer_lens adapter context, cuda float32
################################################################################

core_tl_test = deepcopy(default_cfg)
core_tl_test["trainer_cfg"] = deepcopy(default_trainer_kwargs)
core_tl_test["session_cfg"]["adapter_ctx"] = ["core", "transformer_lens"]
tl_task_sig_columns = {'task_name': 'pytest_rte_tl', 'signature_columns': test_tl_signature_columns}
get_nested(core_tl_test, datamod_cfg)['init_args'] = {
    **core_cust_shared_config_cli, **core_cust_datamodule_cfg_prompt_cfg, **tl_task_sig_columns}
get_nested(core_tl_test, datamod_cfg)['init_args']["tokenizer_kwargs"].update({"model_input_names":
                                                                               ['input', 'attention_mask']})
get_nested(core_tl_test, "session_cfg")["module_cfg"] = deepcopy(base_tl_cust_model_cfg["session_cfg"]["module_cfg"])
get_nested(core_tl_test, mod_initargs)["model_cfg"] = {"device": "cpu", "dtype": "float32",
                                                       "model_args": {"max_seq_len": 200}}
get_nested(core_tl_test, f"{mod_initargs}.tl_cfg.init_args")["cfg"].update({"dtype": "float32", "device": "cuda"})
get_nested(core_tl_test, mod_initargs)["experiment_tag"] = CLI_TESTS.core_tl_test.value
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_test] = core_tl_test

################################################################################
# core adapter test, transformer_lens adapter context, cuda float32, no harness
################################################################################

# identical config to `core_tl_test` but used without the experiment harness/bootstrap args
core_tl_test_noharness = deepcopy(core_tl_test)
get_nested(core_tl_test_noharness, mod_initargs)["experiment_tag"] = CLI_TESTS.core_tl_test_noharness.value
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_test_noharness] = core_tl_test_noharness

################################################################################
# Lightning adapter testing with transformer_lens adapter context w/ cuda, bf16
################################################################################

l_tl_test = deepcopy(core_tl_test)
l_tl_test.pop("trainer_cfg")
get_nested(l_tl_test, "session_cfg")["adapter_ctx"] = ["lightning", "transformer_lens"]
get_nested(l_tl_test, tl_cfg_initargs)["cfg"].update({'device': 'cuda', 'dtype': 'bfloat16'})
get_nested(l_tl_test, mod_initargs)["experiment_tag"] = CLI_TESTS.l_tl_test.value
l_tl_test.update(deepcopy(base_lightning_trainer_cfg))
l_tl_test["trainer"].update({"precision": "bf16-true", "accumulate_grad_batches": 1, "num_sanity_val_steps": 0})
l_tl_test["trainer"]["logger"]["init_args"]["name"] = CLI_TESTS.l_tl_test.value
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_test] = l_tl_test

################################################################################
# lightning adapter fit with no transformer_lens adapter context, cuda bf16
################################################################################

l_optim_fit = deepcopy(core_optim_train)
l_optim_fit.pop("trainer_cfg")
get_nested(l_optim_fit, "session_cfg")["adapter_ctx"] = ["lightning"]
l_optim_fit["trainer"] = deepcopy(l_tl_test["trainer"])
l_optim_fit["trainer"]["logger"]["init_args"]["name"] = CLI_TESTS.l_optim_fit.value
get_nested(l_optim_fit, mod_initargs)["model_cfg"].update({'device': 'cuda', 'dtype': 'bfloat16'})
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.l_optim_fit] = l_optim_fit

################################################################################
# core adapter test with transformer_lens adapter, return cli w/o run
################################################################################

# TODO: if more of these aliased tests are needed, use a factory function to gen the config
# `core_tl_norun` is the same as `core_tl_test` but validating core CLI instantiation-only mode
core_tl_norun = deepcopy(core_tl_test)
get_nested(core_tl_norun, mod_initargs)["experiment_tag"] = CLI_TESTS.core_tl_norun.value
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.core_tl_norun] = core_tl_norun

################################################################################
# lightning adapter test with transformer_lens, return cli w/o run
################################################################################

# similarly `l_tl_norun` is the same as `l_tl_test` but validating the lightning CLI instantiation-only mode
l_tl_norun = deepcopy(l_tl_test)
get_nested(l_tl_norun, mod_initargs)["experiment_tag"] = CLI_TESTS.l_tl_norun.value
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_norun] = l_tl_norun

################################################################################
# lightning adapter test with transformer_lens, return cli w/o run, no harness
################################################################################

# identical config to `l_tl_norun` but used without the experiment harness/bootstrap args
l_tl_norun_noharness = deepcopy(l_tl_norun)
get_nested(l_tl_norun_noharness, mod_initargs)["experiment_tag"] = CLI_TESTS.l_tl_norun_noharness.value
parity_cli_cfgs["exp_cfgs"][CLI_TESTS.l_tl_norun_noharness] = l_tl_norun_noharness
