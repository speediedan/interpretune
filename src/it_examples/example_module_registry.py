from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, Type, Set, Iterable, NamedTuple
from typing_extensions import override
from pprint import pformat

from it_examples.experiments.rte_boolq import (RTEBoolqPromptConfig, RTEBoolqDataModule, RTEBoolqConfig,
                                               RTEBoolqSLConfig, RTEBoolqTLConfig, RTEBoolqZeroShotClassificationConfig)
from interpretune.adapters.transformer_lens import (TLensGenerationConfig, ITLensFromPretrainedConfig,
                                                    ITLensFromPretrainedNoProcessingConfig,
                                                    ITLensCustomConfig)
from interpretune.adapters.sae_lens import SAELensFromPretrainedConfig, SAELensCustomConfig
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.mixins import HFGenerationConfig, CoreGenerationConfig
from interpretune.base.config.module import HFFromPretrainedConfig, ITConfig
from interpretune.adapters.registration import Adapter
from interpretune.base.contract.protocol import ModuleSteppable, DataModuleInitable
from tests.modules import (TestITDataModule, TestITModule, BaseTestDataModule, SimpleDatasetStateMixin)
from tests.base_defaults import BaseCfg, default_test_bs


# We define and 'register' basic example module configs here which are used in both example and parity acceptance tests.

class RegisteredExampleCfg(NamedTuple):
    datamodule_cfg: ITDataModuleConfig
    module_cfg: ITConfig
    datamodule_cls: Type[DataModuleInitable] = TestITDataModule
    module_cls: Type[ModuleSteppable] = TestITModule


class ModuleExampleRegistry(dict):
    def register(
        self,
        phase: str,
        model_src_key: str,
        task_name: str,
        adapter_combinations: Tuple[Adapter] | Tuple[Tuple[Adapter]],
        registered_example_cfg: RegisteredExampleCfg,
        description: Optional[str] = None,
    ) -> None:
        """Registers valid component + adapter compositions mapped to composition keys with required metadata.

        Args:
            lead_adapter: The adapter registering this set of valid compositions (e.g. LightningAdapter)
            phase:
            model_src_key:
            task_name:
            adapter_combination: tuple identifying the valid adapter composition
            registered_example_cfg: Tuple[Callable],
            description : composition description
        """
        supported_composition: Dict[str | Adapter | Tuple[Adapter | str], Tuple[Dict]] = {}
        for a_combo in adapter_combinations:
            a_combo = (a_combo,) if not isinstance(a_combo, tuple) else a_combo
            composition_key = (model_src_key, task_name, phase, self.canonicalize_composition(a_combo))
            supported_composition[composition_key] = registered_example_cfg
            supported_composition["description"] = description if description is not None else ""
            self[composition_key] = supported_composition

    def canonicalize_composition(self, adapter_ctx: Iterable[Adapter]) -> Tuple:
        return tuple(sorted(list(adapter_ctx), key=lambda a: a.value))

    @override
    def get(self, composition_key: Tuple| BaseCfg ) -> Any:
        if not isinstance(composition_key, tuple):
            assert isinstance(composition_key, BaseCfg), "`composition_key` must be either a tuple or a `BaseCfg`"
            # TODO: change "model_key" references to "task_key" to avoid confusion
            composition_key = (composition_key.model_src_key, composition_key.model_key, composition_key.phase,
                               composition_key.adapter_ctx)
        if composition_key in self:
            supported_composition = self[composition_key]
            return supported_composition[composition_key]

        available_keys = pformat(sorted(self.keys())) if self.keys() else "none"
        err_msg = (f"The composition key `{composition_key}` was not found in the registry."
                   f" Available valid module example compositions: {available_keys}")
        raise KeyError(err_msg)

    def remove(self, composition_key: Tuple[Adapter | str]) -> None:
        """Removes the registered adapter composition by name."""
        del self[composition_key]

    def available_compositions(self, adapter_filter: Optional[Iterable[Adapter]| Adapter] = None) -> Set:
        """Returns a list of registered adapters, optionally filtering by an adapter or iterable of adapters."""
        if adapter_filter is not None:
            if isinstance(adapter_filter, Adapter):
                adapter_filter = (adapter_filter,)
            return {key for key in self.keys() for subkey in key if subkey in adapter_filter}
        return set(self.keys())

    def __str__(self) -> str:
        return f"Registered Module Example Compositions: {pformat(self.keys())}"

MODULE_EXAMPLE_REGISTRY = ModuleExampleRegistry()

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
                                               train_batch_size=default_test_bs, eval_batch_size=default_test_bs)
test_core_gpt2_it_module_base = RTEBoolqConfig(**base_it_module_kwargs, **core_gpt2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=HFGenerationConfig(model_config={"max_new_tokens": 3})),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                                                  model_head="transformers.GPT2LMHeadModel"))
test_core_gpt2_it_module_optim = deepcopy(test_core_gpt2_it_module_base)
test_core_gpt2_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="gpt2", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_gpt2_datamodule_cfg,
                                     module_cfg=test_core_gpt2_it_module_base),
                                 description="Basic example, GPT2 with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="gpt2", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_gpt2_datamodule_cfg,
                                     module_cfg=test_core_gpt2_it_module_optim),
                                 description="Basic example, GPT2 with supported adapter compositions")

####################################
# Gemma2
####################################
# TODO: add option for using HF `tokenizer.apply_chat_template` api?
#       We usually want full control so lower priority atm, but will likely be valuable in the future.

@dataclass(kw_only=True)
class Gemma2PromptConfig:
    # see https://huggingface.co/google/gemma-2-2b-it for more details
    B_TURN: str = "<start_of_turn>"
    E_TURN: str = "<end_of_turn>"
    USER_ROLE: str = "user"
    ASSISTANT_ROLE: str = "model"

    def __post_init__(self) -> None:
        self.USER_ROLE_START = self.B_TURN + self.USER_ROLE + "\n"
        self.USER_ROLE_END = self.E_TURN + self.B_TURN + self.ASSISTANT_ROLE + "\n"

@dataclass(kw_only=True)
class RTEBoolqGemma2PromptConfig(Gemma2PromptConfig, RTEBoolqPromptConfig):
    ...

class Gemma2RTEBoolqDataModule(RTEBoolqDataModule):

    @staticmethod
    def model_chat_template_fn(task_prompt: str, prompt_cfg: Optional[Gemma2PromptConfig] = None,
                                  tokenization_pattern: Optional[str] = None) -> str:
        if tokenization_pattern == "gemma2-chat":
            sequence = prompt_cfg.USER_ROLE_START + f"{task_prompt.strip()} {prompt_cfg.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence

class Gemma2TestITDataModule(SimpleDatasetStateMixin, BaseTestDataModule, Gemma2RTEBoolqDataModule):
    ...

gemma2_tokenizer_kwargs = {"model_input_names": ["input_ids", "attention_mask"],
                                "padding_side": "left", "add_bos_token": True, "add_eos_token": False,
                                "local_files_only": False}

core_gemma2_shared_config = dict(task_name="pytest_rte_hf", tokenizer_kwargs=gemma2_tokenizer_kwargs,
                                      model_name_or_path="google/gemma-2-2b-it")

core_gemma2_datamodule_cfg = ITDataModuleConfig(**core_gemma2_shared_config,
                                                     prompt_cfg=RTEBoolqGemma2PromptConfig(),
                                                     signature_columns=core_pretrained_signature_columns,
                                                     prepare_data_map_cfg={"batched": True},
                                                     text_fields=("premise", "hypothesis"),
                                                     enable_datasets_cache=True,
                                                     train_batch_size=default_test_bs,
                                                     eval_batch_size=default_test_bs)
test_core_gemma2_it_module_base = RTEBoolqConfig(**base_it_module_kwargs, **core_gemma2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=HFGenerationConfig(model_config={"max_new_tokens": 3})),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                                                  model_head="transformers.Gemma2ForCausalLM"))
test_core_gemma2_it_module_optim = deepcopy(test_core_gemma2_it_module_base)
test_core_gemma2_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="gemma2", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_gemma2_datamodule_cfg,
                                     module_cfg=test_core_gemma2_it_module_base,
                                     datamodule_cls=Gemma2TestITDataModule),
                                 description="Basic example, Gemma2 with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="gemma2", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_gemma2_datamodule_cfg,
                                     module_cfg=test_core_gemma2_it_module_optim,
                                     datamodule_cls=Gemma2TestITDataModule),
                                 description="Basic example, Gemma2 with supported adapter compositions")


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

    @staticmethod
    def model_chat_template_fn(task_prompt: str, prompt_cfg: Optional[Llama3PromptConfig] = None,
                                  tokenization_pattern: Optional[str] = None) -> str:
        if tokenization_pattern == "llama3-chat":
            sequence = prompt_cfg.SYS_ROLE_START + f"{task_prompt.strip()} {prompt_cfg.USER_ROLE_END}"
        else:
            sequence = task_prompt.strip()
        return sequence

class LlamaTestITDataModule(SimpleDatasetStateMixin, BaseTestDataModule, LlamaRTEBoolqDataModule):
    ...

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
                                               train_batch_size=default_test_bs, eval_batch_size=default_test_bs)

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

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="llama3", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_llama3_datamodule_cfg,
                                     module_cfg=test_core_llama3_it_module_base,
                                     datamodule_cls=LlamaTestITDataModule),
                                 description="Basic example, Llama3 with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="llama3", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_llama3_datamodule_cfg,
                                     module_cfg=test_core_llama3_it_module_optim,
                                     datamodule_cls=LlamaTestITDataModule),
                                 description="Basic example, Llama3 with supported adapter compositions")

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
                                               train_batch_size=default_test_bs, eval_batch_size=default_test_bs)

test_core_cust_it_module_base = RTEBoolqConfig(**base_it_module_kwargs, **core_cust_shared_config,
    model_cfg={"device": "cpu", "dtype": "float32", "model_args": {"max_seq_len": 200}},
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=CoreGenerationConfig(max_new_tokens=2, output_logits=True)),
)

test_core_cust_it_module_optim = deepcopy(test_core_cust_it_module_base)
test_core_cust_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="cust", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_cust_datamodule_cfg,
                                     module_cfg=test_core_cust_it_module_base),
                                 description="Basic example, Custom Transformer with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="predict", model_src_key="cust", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_cust_datamodule_cfg,
                                     module_cfg=test_core_cust_it_module_base),
                                 description="Basic example, Custom Transformer with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="cust", task_name="rte",
                                 adapter_combinations=((Adapter.core,), (Adapter.lightning,)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=core_cust_datamodule_cfg,
                                     module_cfg=test_core_cust_it_module_optim),
                                 description="Basic example, Custom Transformer with supported adapter compositions")

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
                                               train_batch_size=default_test_bs, eval_batch_size=default_test_bs)

test_tl_gpt2_it_module_base = RTEBoolqTLConfig(**base_it_module_kwargs, **test_tl_gpt2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)),
    tl_cfg=ITLensFromPretrainedConfig(),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                                                  model_head="transformers.GPT2LMHeadModel"))
test_tl_gpt2_it_module_optim = deepcopy(test_tl_gpt2_it_module_base)
test_tl_gpt2_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="gpt2", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.transformer_lens),
                                                       (Adapter.lightning, Adapter.transformer_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_tl_datamodule_cfg,
                                     module_cfg=test_tl_gpt2_it_module_base),
                                 description="Basic TL example, GPT2 with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="gpt2", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.transformer_lens),
                                                       (Adapter.lightning, Adapter.transformer_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_tl_datamodule_cfg,
                                     module_cfg=test_tl_gpt2_it_module_optim),
                                 description="Basic TL example, GPT2 with supported adapter compositions")

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

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="cust", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.transformer_lens),
                                                       (Adapter.lightning, Adapter.transformer_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_tl_datamodule_cfg,
                                     module_cfg=test_tl_cust_it_module_base),
                                 description="Basic TL example, Custom Transformer with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="cust", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.transformer_lens),
                                                       (Adapter.lightning, Adapter.transformer_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_tl_datamodule_cfg,
                                     module_cfg=test_tl_cust_it_module_optim),
                                 description="Basic TL example, Custom Transformer with supported adapter compositions")

##################################
# SL GPT2
##################################

sl_tokenizer_kwargs = deepcopy(default_tokenizer_kwargs) | {"model_input_names": ['input', 'attention_mask']}
test_sl_signature_columns = core_pretrained_signature_columns.copy()
test_sl_signature_columns[0] = 'input'

test_sl_gpt2_shared_config = dict(task_name="pytest_rte_tl", tokenizer_kwargs=sl_tokenizer_kwargs,
                                  model_name_or_path="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})

test_sl_gpt2_datamodule_cfg = ITDataModuleConfig(**test_sl_gpt2_shared_config, prompt_cfg=RTEBoolqPromptConfig(),
                                               signature_columns=test_sl_signature_columns,
                                               prepare_data_map_cfg={"batched": True},
                                               text_fields=("premise", "hypothesis"),
                                               enable_datasets_cache=False,
                                               train_batch_size=default_test_bs, eval_batch_size=default_test_bs)

test_sl_gpt2_it_module_base = RTEBoolqSLConfig(**base_it_module_kwargs, **test_sl_gpt2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        enabled=True, lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)),
    tl_cfg=ITLensFromPretrainedNoProcessingConfig(),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                                                  model_head="transformers.GPT2LMHeadModel"),
    sae_cfg=SAELensFromPretrainedConfig(release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre", device="cpu"))
test_sl_gpt2_it_module_optim = deepcopy(test_sl_gpt2_it_module_base)
test_sl_gpt2_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="gpt2", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.sae_lens),
                                                       (Adapter.lightning, Adapter.sae_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_sl_gpt2_datamodule_cfg,
                                     module_cfg=test_sl_gpt2_it_module_base),
                                 description="Basic SL example, GPT2 with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="gpt2", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.sae_lens),
                                                       (Adapter.lightning, Adapter.sae_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_sl_gpt2_datamodule_cfg,
                                     module_cfg=test_sl_gpt2_it_module_optim),
                                 description="Basic SL example, GPT2 with supported adapter compositions")

##################################
# SL Gemma2
##################################
# TODO: maybe replace gpt2 example with gemma2 once no longer needed as a ref

tl_gemma2_tokenizer_kwargs = deepcopy(gemma2_tokenizer_kwargs) | {"model_input_names": ['input', 'attention_mask']}

tl_gemma2_shared_config = dict(task_name="pytest_rte_tl", tokenizer_kwargs=tl_gemma2_tokenizer_kwargs,
                                 model_name_or_path="google/gemma-2-2b-it")

test_tl_gemma2_datamodule_cfg = ITDataModuleConfig(**tl_gemma2_shared_config,
                                                     prompt_cfg=RTEBoolqPromptConfig(),
                                                     signature_columns=core_pretrained_signature_columns,
                                                     prepare_data_map_cfg={"batched": True},
                                                     text_fields=("premise", "hypothesis"),
                                                     enable_datasets_cache=True,
                                                     train_batch_size=default_test_bs,
                                                     eval_batch_size=default_test_bs)

test_sl_gemma2_it_module_base = RTEBoolqSLConfig(**base_it_module_kwargs, **tl_gemma2_shared_config,
    zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(
        # TODO: address possible need for use_past_kv_cache=False
        enabled=True, lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)),
    tl_cfg=ITLensFromPretrainedNoProcessingConfig(),
    hf_from_pretrained_cfg=HFFromPretrainedConfig(pretrained_kwargs={"device_map": "cpu", "torch_dtype": "float32"},
                                                  model_head="transformers.Gemma2ForCausalLM"),
    sae_cfg=SAELensFromPretrainedConfig(release="gemma-scope-2b-pt-res-canonical",
                                        sae_id="layer_25/width_16k/canonical", device="cuda"))
test_sl_gemma2_it_module_optim = deepcopy(test_sl_gpt2_it_module_base)
test_sl_gemma2_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="gemma2", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.sae_lens),
                                                       (Adapter.lightning, Adapter.sae_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_tl_gemma2_datamodule_cfg,
                                     module_cfg=test_sl_gemma2_it_module_base,
                                     datamodule_cls=Gemma2TestITDataModule),
                                 description="SL example, gemma2 with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="gemma2", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.sae_lens),
                                                       (Adapter.lightning, Adapter.sae_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_tl_gemma2_datamodule_cfg,
                                     module_cfg=test_sl_gemma2_it_module_optim,
                                     datamodule_cls=Gemma2TestITDataModule),
                                 description="SL example, gemma2 with supported adapter compositions")

##################################
# SL Cust
##################################

test_sl_cust_shared_config = dict(task_name="pytest_rte_tl", tokenizer_kwargs=sl_tokenizer_kwargs,
                                  tokenizer_name="gpt2", tokenizer_id_overrides={"pad_token_id": 50256})

test_sl_cust_datamodule_cfg = ITDataModuleConfig(**test_sl_cust_shared_config, prompt_cfg=RTEBoolqPromptConfig(),
                                               signature_columns=test_sl_signature_columns,
                                               prepare_data_map_cfg={"batched": True},
                                               text_fields=("premise", "hypothesis"),
                                               enable_datasets_cache=False,
                                               train_batch_size=default_test_bs, eval_batch_size=default_test_bs)

test_tl_cust_config = {"n_layers":1, "d_mlp": 10, "d_model":10, "d_head":5, "n_heads":2, "n_ctx":200,
                       "act_fn":'relu', "tokenizer_name": 'gpt2'}

test_sae_cust_config = dict(
        architecture="standard",
        d_in=10,
        d_sae=10 * 2,
        dtype="float32",
        device="cpu",
        model_name="cust",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",
        prepend_bos=True,
        context_size=200,
        dataset_path="test",
        dataset_trust_remote_code=True,
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        sae_lens_training_version=None,
        normalize_activations="none",
    )

test_sl_cust_it_module_base = RTEBoolqSLConfig(**base_it_module_kwargs, **test_sl_cust_shared_config,
           zero_shot_cfg=RTEBoolqZeroShotClassificationConfig(enabled=True,
                                                       lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1)),
           tl_cfg=ITLensCustomConfig(cfg=test_tl_cust_config),
           sae_cfg=SAELensCustomConfig(cfg=test_sae_cust_config))
test_sl_cust_it_module_optim = deepcopy(test_sl_cust_it_module_base)
test_sl_cust_it_module_optim.__dict__.update(test_optimizer_scheduler_init)

MODULE_EXAMPLE_REGISTRY.register(phase="test", model_src_key="cust", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.sae_lens),
                                                       (Adapter.lightning, Adapter.sae_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_sl_cust_datamodule_cfg,
                                     module_cfg=test_sl_cust_it_module_base),
                                 description=("SL example, Custom Transformer with pretrained SAE from custom config"
                                              "with supported adapter compositions"))
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="cust", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.sae_lens),
                                                       (Adapter.lightning, Adapter.sae_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_sl_cust_datamodule_cfg,
                                     module_cfg=test_sl_cust_it_module_optim),
                                 description=("SL example, Custom Transformer with pretrained SAE from custom config"
                                              "with supported adapter compositions"))
