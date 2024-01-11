from typing import NamedTuple
from collections import ChainMap

from interpretune.base.config_classes import (MemProfilerCfg, MemProfilerSchedule, )
from it_examples.experiments.rte_boolq.core import RTEBoolqPromptConfig

# TODO: refactor to dict-based configs to decomposed yaml strings, files or init dataclasses earlier? may make sense
test_tokenizer_kwargs = {"tokenizer_kwargs": {"add_bos_token": True, "local_files_only": False, "padding_side": "right",
                         "model_input_names": ['input_ids', 'attention_mask']}}

test_shared_config = {
    "task_name": "pytest_rte",
    "model_name_or_path": "gpt2",
    "tokenizer_id_overrides": {"pad_token_id": 50256},
    "tokenizer_kwargs": test_tokenizer_kwargs,
}

test_signature_columns= ['input_ids', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'labels',
                         'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']

test_datamodule_kwargs = {"prompt_cfg": RTEBoolqPromptConfig(), "signature_columns": test_signature_columns,
                          "enable_datasets_cache": True, "prepare_data_map_cfg": {"batched": True},
                          "text_fields": ("premise", "hypothesis"),  "train_batch_size": 2, "eval_batch_size": 2}

test_optimizer_init = {"optimizer_init": {"class_path": "torch.optim.AdamW",
                              "init_args": {"weight_decay": 1.0e-06, "eps": 1.0e-07, "lr": 3.0e-05}}}

test_lr_scheduler_init = {"lr_scheduler_init": {"class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                              "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-06}}}

test_optimizer_scheduler_init = ChainMap(test_optimizer_init, test_lr_scheduler_init)

test_it_module_kwargs = {"use_model_cache": False, "cust_fwd_kwargs": {}, "from_pretrained_cfg":
                         {"device_map": "cpu", "torch_dtype": "float32"}, "experiment_tag": "test_itmodule",
                         "auto_model_cfg": {"model_head": "transformers.GPT2ForSequenceClassification"},}

enable_memprofiler_kwargs = {"enabled": True, "cuda_allocator_history": True}
bs1_override = {'train_batch_size': 1, 'eval_batch_size': 1}
memprofiler_cfg = MemProfilerCfg(**enable_memprofiler_kwargs)
no_savedt_memprofiler_cfg = MemProfilerCfg(**enable_memprofiler_kwargs, enable_saved_tensors_hooks=False)
warm_maxstep_memprof_cfg = MemProfilerCfg(**enable_memprofiler_kwargs,
                                      **{"schedule": MemProfilerSchedule(warmup_steps=2, max_step=4)})
nowarm_maxstep_memprof_cfg = MemProfilerCfg(**enable_memprofiler_kwargs,
                                                 **{"schedule": MemProfilerSchedule(max_step=4)})
nowarm_maxstep_hk_memprof_cfg = MemProfilerCfg(retain_hooks_for_funcs=["training_step"], **enable_memprofiler_kwargs,
                                                 **{"schedule": MemProfilerSchedule(max_step=4)})

test_it_module_base = ChainMap(test_shared_config, test_it_module_kwargs)
test_it_module_optim = ChainMap(test_it_module_base, test_optimizer_scheduler_init)

########################################################################################################################
# NOTE [Test Dataset Fingerprint]
# A simple fingerprint of the (deterministic) test dataset used to generate the current incarnation of expected results.
# Useful for validating that the test dataset has not changed wrt the test dataset used to generate the reference
# results. A few things to note:
#   - The dataloader kwargs are not currently part of these fingerprint so if the loss of a given test diverges
#      from expectation, one may still need to verify shuffling of the fingerprinted dataset etc. has not been
#      introduced and compare the examples actually passed to the model in a given test/step to the ids below before
#      subsequently assessing other sources of indeterminism that could be the source of the loss change.
#   - One should see `tests.helpers.modules.TestITDataModule.sample_dataset_state()` for the indices used to generate
#      this fingerprint
#   - The fingerprinted dataset below is not shuffled or sorted with the current dataloader configurations
#   - All current expected loss results were generated with [train|eval]_batch_size = 2
#   - All current memory profile results were generated with [train|eval]_batch_size = 1
deterministic_token_ids = [5674, 24140, 373, 666, 2233, 303, 783, 783, 2055, 319, 373, 910, 17074, 284, 6108]
test_dataset_state = ('GPT2TokenizerFast', 'pytest_rte', deterministic_token_ids)
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


# runif components
cuda_mark = {'min_cuda_gpus': 1}
bf16_cuda_mark = {'bf16_cuda': True}
profiling_mark = {'profiling': True}
lightning_mark = {"lightning": True}
bitsandbytes_mark = {"bitsandbytes": True}
skip_win_mark = {'skip_windows': True}
slow_mark = {'slow': True}

# RunIf aliases
RUNIF_ALIASES = {
    "lightning": lightning_mark,
    "bitsandbytes": bitsandbytes_mark,
    "prof": profiling_mark,
    "prof_l": {**profiling_mark, **lightning_mark},
    "cuda": cuda_mark,
    "cuda_l": {**cuda_mark, **lightning_mark},
    "cuda_prof": {**cuda_mark, **profiling_mark},
    "cuda_prof_l": {**cuda_mark, **profiling_mark, **lightning_mark},
    "bf16_cuda": bf16_cuda_mark,
    "bf16_cuda_l": {**bf16_cuda_mark, **lightning_mark},
    "bf16_cuda_prof": {**bf16_cuda_mark, **profiling_mark},
    "bf16_cuda_prof_l": {**bf16_cuda_mark, **profiling_mark, **lightning_mark},
    "skip_win_slow": {**skip_win_mark, **slow_mark},
}

# composable cfg aliases
w_lit = {"lightning": True}
cuda = {"device_type": "cuda"}
cuda_act = {**cuda, "act_ckpt": True}
bf16 = {"precision": "bf16"}
cuda_bf16 = {**cuda, **bf16}
cuda_bf16_l = {**cuda, **bf16, **w_lit}
memprof_steps = {"train_steps": 5, "val_steps": 3}
bs1_memprof_steps = {"dm_override_cfg": bs1_override, **memprof_steps}
debug_hidden = {"cust_fwd_kwargs": {"output_hidden_states": True}}
test_bs1_mem = {"loop_type": "test", "dm_override_cfg": bs1_override, "memprofiling_cfg": memprofiler_cfg}
test_bs1_mem_nosavedt = {**test_bs1_mem, "memprofiling_cfg": no_savedt_memprofiler_cfg}
bs1_warm_mem = {**bs1_memprof_steps,  "memprofiling_cfg": warm_maxstep_memprof_cfg}
bs1_nowarm_mem = {**bs1_memprof_steps, "memprofiling_cfg": nowarm_maxstep_memprof_cfg}
bs1_nowarm_hk_mem = {**bs1_memprof_steps, "memprofiling_cfg": nowarm_maxstep_hk_memprof_cfg}
