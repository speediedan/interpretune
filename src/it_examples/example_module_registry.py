from copy import deepcopy
from typing import Optional, Any, Dict, Tuple, Type, Set, Iterable, NamedTuple
from typing_extensions import override
from pprint import pformat
from pathlib import Path

from interpretune.utils.logging import rank_zero_debug, rank_zero_warn
from interpretune.utils.import_utils import instantiate_class
from it_examples.experiments.rte_boolq import (RTEBoolqPromptConfig, RTEBoolqSLConfig, RTEBoolqTLConfig,
                                               RTEBoolqZeroShotClassificationConfig)
from interpretune.adapters.transformer_lens import (TLensGenerationConfig, ITLensFromPretrainedConfig,
                                                    ITLensFromPretrainedNoProcessingConfig,
                                                    ITLensCustomConfig)
from interpretune.adapters.sae_lens import SAELensFromPretrainedConfig, SAELensCustomConfig
from interpretune.base.config.datamodule import ITDataModuleConfig
from interpretune.base.config.module import HFFromPretrainedConfig, ITConfig
from interpretune.adapters.registration import Adapter
from interpretune.base.contract.protocol import ModuleSteppable, DataModuleInitable
from tests.modules import (TestITDataModule, TestITModule, NoFingerprintTestITDataModule)
from tests.base_defaults import BaseCfg, default_test_bs

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

def instantiate_and_register(reg_info: Dict[str, Any], rv: Dict[str, Any]) -> None:
    datamodule_cls, module_cls = DEFAULT_TEST_DATAMODULE, DEFAULT_TEST_MODULE
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
        MODULE_EXAMPLE_REGISTRY.register(**reg_info, registered_example_cfg=registered_example)

def resolve_adapter_combinations(adapter_combinations: Iterable):
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

def instantiate_nested(d: Dict):
    for k, v in d.items():  # recursively instantiate nested directives
        if isinstance(v, dict):
            d[k] = instantiate_nested(v)
    if 'class_path' in d:  # if the dict directly contains a class_path key
        d = instantiate_class(d)  # with instantiating the class
    return d

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

gen_module_example_registry()

####################################
# Gemma2
####################################
# TODO: add option for using HF `tokenizer.apply_chat_template` api?
#       We usually want full control so lower priority atm, but will likely be valuable in the future.

gemma2_tokenizer_kwargs = {"model_input_names": ["input_ids", "attention_mask"],
                                "padding_side": "left", "add_bos_token": True, "add_eos_token": False,
                                "local_files_only": False}

# TODO: NEXT below example registrations remain to be converted yaml-driven configs

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
                                     datamodule_cls=NoFingerprintTestITDataModule),
                                 description="SL example, gemma2 with supported adapter compositions")
MODULE_EXAMPLE_REGISTRY.register(phase="train", model_src_key="gemma2", task_name="rte",
                                 adapter_combinations=((Adapter.core, Adapter.sae_lens),
                                                       (Adapter.lightning, Adapter.sae_lens)),
                                 registered_example_cfg=RegisteredExampleCfg(
                                     datamodule_cfg=test_tl_gemma2_datamodule_cfg,
                                     module_cfg=test_sl_gemma2_it_module_optim,
                                     datamodule_cls=NoFingerprintTestITDataModule),
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
