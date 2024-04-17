from collections import ChainMap

from interpretune.base.config.module import HFFromPretrainedConfig
from it_examples.experiments.rte_boolq.config import RTEBoolqPromptConfig, RTEBoolqZeroShotClassificationConfig
from tests.parity_acceptance.base.cfg_aliases import (
    tokenizer_base_kwargs, gpt2_shared_config, default_test_bs, gpt2_dataset_state, base_it_module_kwargs,
    test_optimizer_scheduler_init, base_hf_from_pretrained_kwargs)
from interpretune.base.contract.session import Plugin
from interpretune.plugins.transformer_lens import TLensGenerationConfig

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

########################################################################################################################
# See NOTE [Test Dataset Fingerprint]
########################################################################################################################
test_dataset_state_tl = ('pytest_rte_tl',) + gpt2_dataset_state

w_tl = {"plugin_ctx": Plugin.transformer_lens}
