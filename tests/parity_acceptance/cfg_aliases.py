import os
from collections import ChainMap
from copy import deepcopy
from enum import auto
from pathlib import Path

from it_examples.experiments.rte_boolq.config import (RTEBoolqPromptConfig, Llama2PromptConfig,
                                                      RTEBoolqZeroShotClassificationConfig)
from interpretune.adapters.registration import Adapter
from interpretune.adapters.transformer_lens import TLensGenerationConfig
from interpretune.base.config.mixins import HFGenerationConfig
from interpretune.base.config.module import HFFromPretrainedConfig
from interpretune.base.config.shared import AutoStrEnum
from interpretune.extensions.memprofiler import MemProfilerCfg, MemProfilerSchedule
from tests.global_defaults import default_prof_bs, default_test_bs
from tests.modules import TestFTS
from tests.utils import ToyGenCfg, get_nested, set_nested


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

##################################
# Core config aliases
##################################

tokenizer_base_kwargs = {"add_bos_token": True, "local_files_only": False}
model_input_names_hf_pretrained = {"model_input_names": ['input_ids', 'attention_mask']}
model_input_names_cust = {"model_input_names": ['tokens']}

# TODO: refactor to init dataclasses earlier and update them directly with a deferred init flag?
core_hf_pretrained_tokenizer_kwargs = {"tokenizer_kwargs": {**model_input_names_hf_pretrained, "padding_side": "left",
                                                            **tokenizer_base_kwargs}}
core_cust_tokenizer_kwargs = {"tokenizer_kwargs": {**model_input_names_cust, "padding_side": "left",
                                                   **tokenizer_base_kwargs}}

gpt2_token_overrides = {"tokenizer_id_overrides": {"pad_token_id": 50256}}
llama2_token_overrides = {"tokenizer_id_overrides": {"pad_token_id": 32000}}

llama2_special_tokens_dict = {"special_tokens_dict": {"pad_token": "<PAD>"}}

gpt2_shared_config =  {"model_name_or_path": "gpt2", **gpt2_token_overrides}
cust_shared_config = {"tokenizer_name": "gpt2", **gpt2_token_overrides}
llama2_shared_config =  {"model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
                         "os_env_model_auth_key": "LLAMA2_AUTH_KEY",
                         **llama2_token_overrides}


core_gpt2_shared_config = {"task_name": "pytest_rte_hf", **core_hf_pretrained_tokenizer_kwargs, **gpt2_shared_config}
core_cust_shared_config = {"task_name": "pytest_rte_pt", **core_cust_tokenizer_kwargs, **cust_shared_config}
core_llama2_shared_config = {"task_name": "pytest_rte_hf", **core_hf_pretrained_tokenizer_kwargs,
                             **llama2_shared_config}


core_pretrained_signature_columns = ['input_ids', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds',
                                     'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
core_cust_signature_columns = ['tokens', 'labels']
core_datamodule_kwargs = {"enable_datasets_cache": True, "prepare_data_map_cfg": {"batched": True},
                               "text_fields": ("premise", "hypothesis"),  "train_batch_size": default_test_bs,
                               "eval_batch_size": default_test_bs}
core_pretrained_datamodule_kwargs = {"signature_columns": core_pretrained_signature_columns, **core_datamodule_kwargs}

core_gpt2_datamodule_kwargs = {"prompt_cfg": RTEBoolqPromptConfig(), **core_pretrained_datamodule_kwargs}
core_llama2_datamodule_kwargs = {"prompt_cfg": Llama2PromptConfig(), "cust_tokenization_pattern": "llama2-chat",
                                 **llama2_special_tokens_dict, **core_pretrained_datamodule_kwargs}
core_cust_datamodule_kwargs = {"prompt_cfg": RTEBoolqPromptConfig(), "signature_columns": core_cust_signature_columns,
                              **core_datamodule_kwargs}

test_optimizer_init = {"optimizer_init": {"class_path": "torch.optim.AdamW",
                              "init_args": {"weight_decay": 1.0e-06, "eps": 1.0e-07, "lr": 3.0e-05}}}

test_lr_scheduler_init = {"lr_scheduler_init": {"class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                              "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-06}}}

test_optimizer_scheduler_init = ChainMap(test_optimizer_init, test_lr_scheduler_init)
base_it_module_kwargs = {"experiment_tag": "test_itmodule", "cust_fwd_kwargs": {}}

base_hf_from_pretrained_kwargs = {"device_map": "cpu", "torch_dtype": "float32"}
no_tie_word_embeddings = {"tie_word_embeddings": False}
no_tie_hf_from_pretrained_kwargs = {**base_hf_from_pretrained_kwargs, **no_tie_word_embeddings}

llama2_lora_cfg = {"lora_cfg": {"r": 8, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.05,
                                "bias": "none", "task_type": "CAUSAL_LM"}}

gpt2_hf_from_pretrained_kwargs = {"pretrained_kwargs": base_hf_from_pretrained_kwargs,
                                  "model_head":"transformers.GPT2LMHeadModel"}
no_tie_gpt2_hf_from_pretrained_kwargs = {"pretrained_kwargs": no_tie_hf_from_pretrained_kwargs,
                                  "model_head":"transformers.GPT2LMHeadModel"}

enable_act_checkpointing = {"activation_checkpointing": True}
gpt2_hf_from_pretrained_cfg = HFFromPretrainedConfig(**gpt2_hf_from_pretrained_kwargs)
no_tie_gpt2_hf_from_pretrained_cfg = HFFromPretrainedConfig(**no_tie_gpt2_hf_from_pretrained_kwargs)
gpt2_hf_from_pretrained_act_ckpt_cfg = HFFromPretrainedConfig(**gpt2_hf_from_pretrained_kwargs,
                                                              **enable_act_checkpointing)
llama2_hf_from_pretrained_cfg = HFFromPretrainedConfig(pretrained_kwargs=base_hf_from_pretrained_kwargs,
                                                     model_head="transformers.LlamaForCausalLM", **llama2_lora_cfg)

core_hf_zero_shot_cfg = RTEBoolqZeroShotClassificationConfig(
    enabled=True, lm_generation_cfg=HFGenerationConfig(model_config={"max_new_tokens": 3}))

test_core_gpt2_it_module_kwargs = {"zero_shot_cfg": core_hf_zero_shot_cfg,
                                   "hf_from_pretrained_cfg": gpt2_hf_from_pretrained_cfg, **base_it_module_kwargs}
test_core_llama2_it_module_kwargs = {"zero_shot_cfg": core_hf_zero_shot_cfg,
                                   "hf_from_pretrained_cfg": llama2_hf_from_pretrained_cfg, **base_it_module_kwargs}

base_cust_model_cfg_kwargs = {"max_seq_len": 200}
base_core_cust_config = {"device": "cpu", "dtype": "float32", "model_args": base_cust_model_cfg_kwargs}

core_cust_zero_shot_cfg = RTEBoolqZeroShotClassificationConfig(enabled=True,
                                                               lm_generation_cfg=ToyGenCfg(max_new_tokens=2))
test_core_cust_it_module_kwargs = {"zero_shot_cfg": core_cust_zero_shot_cfg, "model_cfg": base_core_cust_config,
                                   **base_it_module_kwargs}

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

test_core_gpt2_it_module_base = ChainMap(core_gpt2_shared_config, test_core_gpt2_it_module_kwargs)
test_core_gpt2_it_module_optim = ChainMap(test_core_gpt2_it_module_base, test_optimizer_scheduler_init)
test_core_llama2_it_module_base = ChainMap(core_llama2_shared_config, test_core_llama2_it_module_kwargs)
test_core_llama2_it_module_optim = ChainMap(test_core_llama2_it_module_base, test_optimizer_scheduler_init)
test_core_cust_it_module_base = ChainMap(core_cust_shared_config, test_core_cust_it_module_kwargs)
test_core_cust_it_module_optim = ChainMap(test_core_cust_it_module_base, test_optimizer_scheduler_init)

################################################################################
# Extension cfg aliases
################################################################################
memprof_steps = {"limit_train_batches": 5, "limit_val_batches": 3}
bs1_memprof_steps = {"dm_override_cfg": bs_override, **memprof_steps}
debug_hidden = {"cust_fwd_kwargs": {"output_hidden_states": True}}
test_bs1_mem = {"phase": "test", "dm_override_cfg": bs_override, "memprofiler_cfg": memprofiler_cfg}
test_bs1_mem_nosavedt = {**test_bs1_mem, "memprofiler_cfg": no_savedt_memprofiler_cfg}
bs1_warm_mem = {**bs1_memprof_steps,  "memprofiler_cfg": warm_maxstep_memprof_cfg}
bs1_nowarm_mem = {**bs1_memprof_steps, "memprofiler_cfg": nowarm_maxstep_memprof_cfg}
bs1_nowarm_hk_mem = {**bs1_memprof_steps, "memprofiler_cfg": nowarm_maxstep_hk_memprof_cfg}

################################################################################
# Feature-specific cfg aliases
################################################################################
act_ckpt = {"hf_from_pretrained_cfg": gpt2_hf_from_pretrained_act_ckpt_cfg}
cuda_act = {**cuda, **act_ckpt}

##################################
# TL config aliases
##################################

tl_model_input_names = {"model_input_names": ['input', 'attention_mask']}
test_tl_tokenizer_kwargs = {"tokenizer_kwargs": {**tl_model_input_names, **tokenizer_base_kwargs}}
test_tl_gpt2_shared_config = {"task_name": "pytest_rte_tl", **test_tl_tokenizer_kwargs, **gpt2_shared_config}
test_tl_signature_columns = ['input', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'labels',
                            'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
test_tl_datamodule_kwargs = {"prompt_cfg": RTEBoolqPromptConfig(), "signature_columns": test_tl_signature_columns,
                          "enable_datasets_cache": False, "prepare_data_map_cfg": {"batched": True},
                          "text_fields": ("premise", "hypothesis"),  "train_batch_size": default_test_bs,
                          "eval_batch_size": default_test_bs}
# TODO: add zero shot testing separately
tl_zero_shot_cfg = RTEBoolqZeroShotClassificationConfig(enabled=True,
                                                        lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1))

test_tl_from_pretrained_config = {}  # currently using default from pretrained config
test_tl_cust_config = {"n_layers":1, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200, "act_fn":'relu',
            "tokenizer_name": 'gpt2'}

test_tl_gpt2_cfg = HFFromPretrainedConfig(pretrained_kwargs=base_hf_from_pretrained_kwargs,
                                                     model_head="transformers.GPT2LMHeadModel")
test_tl_gpt2_it_module_kwargs = {"zero_shot_cfg": tl_zero_shot_cfg, "tl_cfg": test_tl_from_pretrained_config,
                                             "hf_from_pretrained_cfg": test_tl_gpt2_cfg,
                                             **base_it_module_kwargs}
test_tl_cust_config_it_module_kwargs = {"zero_shot_cfg": tl_zero_shot_cfg, "tl_cfg": {"cfg": test_tl_cust_config},
                                        **base_it_module_kwargs}
test_tl_gpt2_it_module_base = ChainMap(test_tl_gpt2_shared_config, test_tl_gpt2_it_module_kwargs)
test_tl_gpt2_it_module_optim = ChainMap(test_tl_gpt2_it_module_base, test_optimizer_scheduler_init)
test_tl_cust_it_module_base = ChainMap(test_tl_gpt2_shared_config, test_tl_cust_config_it_module_kwargs)
test_tl_cust_it_module_optim = ChainMap(test_tl_cust_it_module_base, test_optimizer_scheduler_init)

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

test_core_cust_it_module_kwargs_zero_shot = deepcopy(test_core_cust_it_module_kwargs)
test_core_cust_it_module_kwargs_zero_shot["zero_shot_cfg"] = {
    "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqZeroShotClassificationConfig",
    "init_args": {"enabled": True, "lm_generation_cfg": {
        "class_path": "tests.utils.ToyGenCfg",
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
test_core_cust_it_model_cfg_cuda_bf16 = deepcopy(test_core_cust_it_module_kwargs['model_cfg'])
test_core_cust_it_model_cfg_cuda_bf16.update({"dtype": "bfloat16", "device": "cuda"})

test_tl_cust_config_it_module_kwargs_zero_shot = deepcopy(test_tl_cust_config_it_module_kwargs)
# override zero_shot_cfg with serializable CLI args
test_tl_cust_config_it_module_kwargs_zero_shot["zero_shot_cfg"] = {
    "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqZeroShotClassificationConfig",
    "init_args": {"enabled": True,
                  "lm_generation_cfg": {
                      "class_path": "interpretune.adapters.transformer_lens.TLensGenerationConfig",
                      "init_args": {"max_new_tokens": 1}
                    }
    },
}
# add necessary class_path/init_args for provided custom config
test_tl_cust_config_it_module_kwargs_zero_shot['tl_cfg'] = {
    'class_path': 'interpretune.adapters.transformer_lens.ITLensCustomConfig',
    'init_args': {'cfg': test_tl_cust_config_it_module_kwargs_zero_shot['tl_cfg']['cfg']}
}

base_tl_cust_model_cfg = {}
base_tl_cust_model_cfg["session_cfg"] = {
    "module_cfg": {
        "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqTLConfig",
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
    "datamodule_cfg": {"init_args": {**core_cust_shared_config_cli, **core_cust_datamodule_kwargs_prompt_cfg}},
    "module_cfg": {
        "class_path": "it_examples.experiments.rte_boolq.config.RTEBoolqConfig",
        "init_args": {**test_core_cust_it_module_kwargs_zero_shot},
    },
}

test_tl_datamodule_kwargs.pop('text_fields', None)  # no need to update this default and it causes serialization errors
## override with appropriate CLI class_path config since we're serializing and loading from yaml
test_tl_datamodule_kwargs["prompt_cfg"] = {"class_path":
                                           "it_examples.experiments.rte_boolq.config.RTEBoolqPromptConfig"}

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
    **core_cust_shared_config_cli, **core_cust_datamodule_kwargs_prompt_cfg, **tl_task_sig_columns}
get_nested(core_tl_test, datamod_cfg)['init_args']["tokenizer_kwargs"].update(tl_model_input_names)
get_nested(core_tl_test, "session_cfg")["module_cfg"] = deepcopy(base_tl_cust_model_cfg["session_cfg"]["module_cfg"])
get_nested(core_tl_test, mod_initargs)["model_cfg"] = base_core_cust_config
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
