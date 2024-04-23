from typing import NamedTuple
from collections import ChainMap
from dataclasses import dataclass

from it_examples.experiments.rte_boolq.config import (RTEBoolqPromptConfig, Llama2PromptConfig,
                                                      RTEBoolqZeroShotClassificationConfig)
from interpretune.base.config.mixins import CoreGenerationConfig, HFGenerationConfig
from interpretune.base.config.module import HFFromPretrainedConfig
from interpretune.base.contract.session import Framework
from interpretune.analysis.memprofiler import MemProfilerCfg, MemProfilerSchedule

@dataclass(kw_only=True)
class ToyGenCfg(CoreGenerationConfig):
    output_logits: bool = True
    verbose: bool = True

# TODO: add model-specific mapping and mapping functions here to dedup some of the shared explicit mapping logic here

default_test_bs = 2
default_prof_bs = 1
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

llama2_lora_cfg = {"lora_cfg": {"r": 8, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.05,
                                "bias": "none", "task_type": "CAUSAL_LM"}}

gpt2_hf_from_pretrained_kwargs = {"pretrained_kwargs": base_hf_from_pretrained_kwargs,
                                  "model_head":"transformers.GPT2LMHeadModel"}
enable_activation_checkpointing = {"activation_checkpointing": True}
gpt2_hf_from_pretrained_cfg = HFFromPretrainedConfig(**gpt2_hf_from_pretrained_kwargs)
gpt2_hf_from_pretrained_act_ckpt_cfg = HFFromPretrainedConfig(**gpt2_hf_from_pretrained_kwargs,
                                                              **enable_activation_checkpointing)
llama2_hf_from_pretrained_cfg = HFFromPretrainedConfig(pretrained_kwargs=base_hf_from_pretrained_kwargs,
                                                     model_head="transformers.LlamaForCausalLM", **llama2_lora_cfg)

core_hf_zero_shot_cfg = RTEBoolqZeroShotClassificationConfig(
    enabled=True, lm_generation_cfg=HFGenerationConfig(kwargs={"max_new_tokens": 3}))

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

########################################################################################################################
# NOTE [Test Dataset Fingerprint]
# A simple fingerprint of the (deterministic) test dataset used to generate the current incarnation of expected results.
# Useful for validating that the test dataset has not changed wrt the test dataset used to generate the reference
# results. A few things to note:
#   - The dataloader kwargs are not currently part of these fingerprint so if the loss of a given test diverges
#      from expectation, one may still need to verify shuffling of the fingerprinted dataset etc. has not been
#      introduced and compare the examples actually passed to the model in a given test/step to the ids below before
#      subsequently assessing other sources of indeterminism that could be the source of the loss change.
#   - One should see `tests.tools.core.modules.TestITDataModule.sample_dataset_state()` for the indices used to generate
#      this fingerprint
#   - The fingerprinted dataset below is not shuffled or sorted with the current dataloader configurations
#   - All current expected loss results were generated with [train|eval]_batch_size = 2
#   - All current memory profile results were generated with [train|eval]_batch_size = 1
NUM_SAMPLE_ROWS = 5
SAMPLE_POSITION = 3
test_datasets = ("rte", "pytest_rte_hf", "pytest_rte_pt", "pytest_rte_tl")
rte_fields = ("premise", "hypothesis")
TEST_TASK_NUM_LABELS = {k: 2 for k in test_datasets}
TEST_TASK_TEXT_FIELD_MAP = {k: rte_fields for k in test_datasets}
# note that we also sample the 'test' split after 'train' and 'validation' though we aren't yet using it
deterministic_token_ids = [5674, 24140, 373, 666, 2233, 303, 783, 783, 2055, 319, 373, 910, 17074, 284, 6108]
expected_first_fwd_ids = {"no_sample": ([],),
                          "train": (deterministic_token_ids[:default_test_bs],),
                          "train_prof": (deterministic_token_ids[:default_prof_bs],),
                          "test": (deterministic_token_ids[NUM_SAMPLE_ROWS:(NUM_SAMPLE_ROWS+default_test_bs)],),
                          "test_prof": (deterministic_token_ids[NUM_SAMPLE_ROWS:(NUM_SAMPLE_ROWS+default_prof_bs)],)}
gpt2_dataset_state = ('GPT2TokenizerFast', deterministic_token_ids)
llama2_dataset_state = ('LlamaTokenizerFast', [])
test_dataset_state_core_gpt2 = ('pytest_rte_hf',) + gpt2_dataset_state
test_dataset_state_core_llama2 = ('pytest_rte_hf',) + llama2_dataset_state
test_dataset_state_core_cust = ('pytest_rte_pt',) + gpt2_dataset_state

# TODO: add current dataloader kwargs to the fingerprint above? May be an excessively rigid check. Consider associating
# a fingerprint of salient config with each specific expected scalar test result in the future. At present, that
# approach seems like overkill given the current codebase.
########################################################################################################################
class MemProfResult(NamedTuple):
    # encapsulate memprofiler result defaults
    # we default to step:
    #   - 3 for cuda in the train phase
    #   - 0 for all other tests (e.g. hook-based (by default, cpu/rss-based) and all test phase assessment)
    # all tests currently default to test epoch 0 to minimize TTS
    epoch = 0
    rank = 0
    default_step = 0
    cuda_train_step = 3
    cuda_mem_keys = ('allocated_bytes.all.current', 'allocated_bytes.all.peak', 'reserved_bytes.all.peak','npp_diff')
    cpu_mem_keys = {"test": ('rss_diff',), "train": ('rss_diff', 'npp_diff'),}
    test_key = f'{rank}.test_step.{epoch}.{default_step}.end'
    train_keys = {"cuda": f'{rank}.training_step.{epoch}.{cuda_train_step}.end',
                  "cpu": f'{rank}.training_step.{epoch}.{default_step}.end'}



# composable cfg aliases
w_lit = {"framework_ctx": Framework.lightning}
cuda = {"device_type": "cuda"}
act_ckpt = {"hf_from_pretrained_cfg": gpt2_hf_from_pretrained_act_ckpt_cfg}
cuda_act = {**cuda, **act_ckpt}
bf16 = {"precision": "bf16"}
cuda_bf16 = {**cuda, **bf16}
cuda_bf16_l = {**cuda, **bf16, **w_lit}
memprof_steps = {"limit_train_batches": 5, "limit_val_batches": 3}
bs1_memprof_steps = {"dm_override_cfg": bs_override, **memprof_steps}
debug_hidden = {"cust_fwd_kwargs": {"output_hidden_states": True}}
test_bs1_mem = {"phase": "test", "dm_override_cfg": bs_override, "memprofiler_cfg": memprofiler_cfg}
test_bs1_mem_nosavedt = {**test_bs1_mem, "memprofiler_cfg": no_savedt_memprofiler_cfg}
bs1_warm_mem = {**bs1_memprof_steps,  "memprofiler_cfg": warm_maxstep_memprof_cfg}
bs1_nowarm_mem = {**bs1_memprof_steps, "memprofiler_cfg": nowarm_maxstep_memprof_cfg}
bs1_nowarm_hk_mem = {**bs1_memprof_steps, "memprofiler_cfg": nowarm_maxstep_hk_memprof_cfg}
