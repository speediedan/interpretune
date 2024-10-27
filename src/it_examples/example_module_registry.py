from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from it_examples.experiments.rte_boolq import (RTEBoolqPromptConfig, RTEBoolqDataModule, RTEBoolqConfig,
                                               RTEBoolqTLConfig, RTEBoolqZeroShotClassificationConfig)
from interpretune.adapters.transformer_lens import (TLensGenerationConfig, ITLensFromPretrainedConfig,
                                                    ITLensCustomConfig)
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.mixins import HFGenerationConfig, CoreGenerationConfig
from interpretune.base.config.module import HFFromPretrainedConfig
from interpretune.adapters.registration import Adapter


default_example_bs = 2

# We define and 'register' basic example module configs here which are used in both example and parity acceptance tests.

##################################
# Core config aliases
##################################

core_pretrained_signature_columns = ['input_ids', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds',
                                     'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
default_tokenizer_kwargs = {"add_bos_token": True, "local_files_only": False,  "padding_side": "left",
                            "model_input_names": ["input_ids", "attention_mask"]}
test_optimizer_init = {"optimizer_init": {"class_path": "torch.optim.AdamW",
                              "init_args": {"weight_decay": 1.0e-06, "eps": 1.0e-07, "lr": 3.0e-05}}}
test_lr_scheduler_init = {"lr_scheduler_init": {"class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
                              "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-06}}}
test_optimizer_scheduler_init = {**test_optimizer_init, **test_lr_scheduler_init}
base_it_module_kwargs = {"experiment_tag": "test_itmodule", "cust_fwd_kwargs": {}}

####################################
# Test/Example Module Configs
####################################

####################################
# GPT2
####################################

core_gpt2_shared_config = dict(task_name="pytest_rte_hf", tokenizer_kwargs=default_tokenizer_kwargs,
                               model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})

core_gpt2_datamodule_cfg = ITDataModuleConfig(**core_gpt2_shared_config, prompt_cfg=RTEBoolqPromptConfig(),
                                               signature_columns=core_pretrained_signature_columns,
                                               prepare_data_map_cfg={"batched": True},
                                               text_fields=("premise", "hypothesis"),
                                               enable_datasets_cache=True,
                                               train_batch_size=default_example_bs, eval_batch_size=default_example_bs)
test_core_gpt2_it_module_base = RTEBoolqConfig(**base_it_module_kwargs, **core_gpt2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=HFGenerationConfig(model_config={"max_new_tokens": 3})),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                                                  model_head="transformers.GPT2LMHeadModel"))
test_core_gpt2_it_module_optim = deepcopy(test_core_gpt2_it_module_base)
test_core_gpt2_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

####################################
# Llama3
####################################

@dataclass(kw_only=True)
class Llama3PromptConfig:
    # see https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md for more details
    sys_prompt: str = ("You are a helpful assistant.")
    B_TEXT: str = "<|begin_of_text|>"
    E_TEXT: str = "<|end_of_text|>"
    B_HEADER: str = "<|start_header_id|>"
    E_HEADER: str = "<|end_header_id|>"
    E_TURN: str = "<|eot_id|>"
    # tool tags, see https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
    # for tool prompt format details
    TOOL_TAG: str = "<|python_tag|>"
    E_TOOL_MSG: str = "<|eom_id|>"
    SYS_ROLE: str = "system"
    USER_ROLE: str = "user"
    ASSISTANT_ROLE: str = "assistant"
    TOOL_ROLE: str = "ipython"

    def __post_init__(self) -> None:
        self.SYS_ROLE_HEADER = self.B_HEADER + self.SYS_ROLE + self.E_HEADER
        self.USER_ROLE_HEADER = self.B_HEADER + self.USER_ROLE + self.E_HEADER
        self.ASSISTANT_ROLE_HEADER = self.B_HEADER + self.ASSISTANT_ROLE + self.E_HEADER
        self.SYS_ROLE_START = self.B_TEXT + self.SYS_ROLE_HEADER + "\n" + self.sys_prompt + self.E_TURN + \
            self.USER_ROLE_HEADER + "\n"
        self.USER_ROLE_END = self.E_TURN + self.ASSISTANT_ROLE_HEADER + "\n"

@dataclass(kw_only=True)
class RTEBoolqLlama3PromptConfig(Llama3PromptConfig, RTEBoolqPromptConfig):
    ...

class LlamaRTEBoolqDataModule(RTEBoolqDataModule):

    def cust_tokenization_pattern(self, task_prompt: str, tokenization_pattern: Optional[str] = None) -> str:
        if tokenization_pattern == "llama3-chat":
            sequence = self.itdm_cfg.prompt_cfg.SYS_ROLE_START + \
            f"{task_prompt.strip()} {self.itdm_cfg.prompt_cfg.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence

# NOTE: this configuration is for testing, for finetuning, llama3 should be changed to right padding
llama3_cust_tokenizer_kwargs = {"model_input_names": ["input_ids", "attention_mask"],
                                "padding_side": "left", "add_bos_token": False, "local_files_only": False}

core_llama3_shared_config = dict(task_name="pytest_rte_hf", tokenizer_kwargs=llama3_cust_tokenizer_kwargs,
                               model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
                               os_env_model_auth_key="HF_GATED_PUBLIC_REPO_AUTH_KEY",
                               tokenizer_id_overrides={"pad_token_id": 128004})

core_llama3_datamodule_cfg = ITDataModuleConfig(**core_llama3_shared_config, prompt_cfg=RTEBoolqLlama3PromptConfig(),
                                               signature_columns=core_pretrained_signature_columns,
                                               cust_tokenization_pattern="llama3-chat",
                                               special_tokens_dict={"pad_token": "<|finetune_right_pad_id|>"},
                                               prepare_data_map_cfg={"batched": True},
                                               text_fields=("premise", "hypothesis"),
                                               enable_datasets_cache=False,
                                               train_batch_size=default_example_bs, eval_batch_size=default_example_bs)

test_core_llama3_it_module_base = RTEBoolqConfig(**base_it_module_kwargs, **core_llama3_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(enabled=True,
                                                       lm_generation_cfg=HFGenerationConfig(
                                                           model_config={"max_new_tokens": 3})),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(
        pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
        model_head="transformers.LlamaForCausalLM",
        lora_cfg={"r": 8, "lora_alpha": 32, "bias": "none", "target_modules": ["q_proj", "v_proj"],
                  "lora_dropout": 0.05, "task_type": "CAUSAL_LM"})
)
test_core_llama3_it_module_optim = deepcopy(test_core_llama3_it_module_base)
test_core_llama3_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

##################################
# Cust Model
##################################

core_cust_tokenizer_kwargs = {"model_input_names": ['tokens'], "padding_side": "left", "add_bos_token": True,
                              "local_files_only": False}

core_cust_shared_config = dict(task_name="pytest_rte_pt", tokenizer_kwargs=core_cust_tokenizer_kwargs,
                               tokenizer_name="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})

core_cust_datamodule_cfg = ITDataModuleConfig(**core_cust_shared_config, prompt_cfg=RTEBoolqPromptConfig(),
                                               signature_columns=['tokens', 'labels'],
                                               prepare_data_map_cfg={"batched": True},
                                               text_fields=("premise", "hypothesis"),
                                               enable_datasets_cache=True,
                                               train_batch_size=default_example_bs, eval_batch_size=default_example_bs)

test_core_cust_it_module_base = RTEBoolqConfig(**base_it_module_kwargs, **core_cust_shared_config,
    model_cfg={"device": "cpu", "dtype": "float32", "model_args": {"max_seq_len": 200}},
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=CoreGenerationConfig(max_new_tokens=2, output_logits=True)),
)

test_core_cust_it_module_optim = deepcopy(test_core_cust_it_module_base)
test_core_cust_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

##################################
# TL GPT2
##################################

tl_tokenizer_kwargs = deepcopy(default_tokenizer_kwargs) | {"model_input_names": ['input', 'attention_mask']}
test_tl_signature_columns = core_pretrained_signature_columns.copy()
test_tl_signature_columns[0] = 'input'

test_tl_gpt2_shared_config = dict(task_name="pytest_rte_tl", tokenizer_kwargs=tl_tokenizer_kwargs,
                                  model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})

test_tl_datamodule_cfg = ITDataModuleConfig(**test_tl_gpt2_shared_config, prompt_cfg=RTEBoolqPromptConfig(),
                                               signature_columns=test_tl_signature_columns,
                                               prepare_data_map_cfg={"batched": True},
                                               text_fields=("premise", "hypothesis"),
                                               enable_datasets_cache=False,
                                               train_batch_size=default_example_bs, eval_batch_size=default_example_bs)

test_tl_gpt2_it_module_base = RTEBoolqTLConfig(**base_it_module_kwargs, **test_tl_gpt2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)),
    tl_cfg=ITLensFromPretrainedConfig(),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                                                  model_head="transformers.GPT2LMHeadModel"))
test_tl_gpt2_it_module_optim = deepcopy(test_tl_gpt2_it_module_base)
test_tl_gpt2_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

##################################
# TL Cust
##################################

test_tl_cust_config = {"n_layers":1, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
                       "act_fn":'relu', "tokenizer_name": 'gpt2'}

test_tl_cust_it_module_base = RTEBoolqTLConfig(**base_it_module_kwargs, **test_tl_gpt2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(enabled=True,
                                                       lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)),
    tl_cfg=ITLensCustomConfig(cfg=test_tl_cust_config)
)
test_tl_cust_it_module_optim = deepcopy(test_tl_cust_it_module_base)
test_tl_cust_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

TEST_DATAMODULE_BASE_CONFIGS = {
    # TODO: make this dict a more robust registry if the number of tested models profilerates
    # TODO: pull module/datamodule configs from model-keyed test config dict (fake lightweight registry)
    # (dm_adapter_ctx, model_src_key)
    (Adapter.core, "gpt2"): core_gpt2_datamodule_cfg,
    (Adapter.core, "llama3"): core_llama3_datamodule_cfg,
    (Adapter.core, "cust"): core_cust_datamodule_cfg,
    (Adapter.transformer_lens, "any"): test_tl_datamodule_cfg,
}

TEST_MODULE_BASE_CONFIGS = {
    # (phase, adapter_mod_cfg_key, model_src_key)
    ("test", None, "gpt2"): test_core_gpt2_it_module_base,
    ("train", None, "gpt2"): test_core_gpt2_it_module_optim,
    ("test", None, "llama3"): test_core_llama3_it_module_base,
    ("train", None, "llama3"): test_core_llama3_it_module_optim,
    ("predict", None, "cust"): test_core_cust_it_module_base,
    ("test", None, "cust"): test_core_cust_it_module_base,
    ("train", None, "cust"): test_core_cust_it_module_optim,
    ("test", Adapter.transformer_lens, "gpt2"): test_tl_gpt2_it_module_base,
    ("train", Adapter.transformer_lens, "gpt2"): test_tl_gpt2_it_module_optim,
    ("test", Adapter.transformer_lens, "cust"): test_tl_cust_it_module_base,
    ("train", Adapter.transformer_lens, "cust"): test_tl_cust_it_module_optim,
}
