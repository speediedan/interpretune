from collections import ChainMap

from it_examples.experiments.rte_boolq.config import RTEBoolqPromptConfig
from tests.base.cfg_aliases import (tokenizer_base_kwargs, base_shared_config, default_test_bs, shared_dataset_state,
                                    base_it_module_kwargs, test_optimizer_scheduler_init)

tl_model_input_names = {"model_input_names": ['input', 'attention_mask']}
test_tl_tokenizer_kwargs = {"tokenizer_kwargs": {**tl_model_input_names, **tokenizer_base_kwargs}}
test_tl_shared_config = {"task_name": "pytest_rte_tl", **test_tl_tokenizer_kwargs, **base_shared_config}
test_tl_signature_columns= ['input', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds', 'labels',
                         'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
test_tl_datamodule_kwargs = {"prompt_cfg": RTEBoolqPromptConfig(), "signature_columns": test_tl_signature_columns,
                          "enable_datasets_cache": False, "prepare_data_map_cfg": {"batched": True},
                          "text_fields": ("premise", "hypothesis"),  "train_batch_size": default_test_bs,
                          "eval_batch_size": default_test_bs}
# TODO: add zero shot testing separately
# tl_zero_shot_cfg = RTEBoolqZeroShotClassificationConfig(enabled=True, lm_generation_cfg=TLensGenerationConfig())
# test_tl_it_module_kwargs = {"tl_from_pretrained_cfg": {"enabled": True}, "zero_shot_cfg": tl_zero_shot_cfg,
#                             "auto_model_cfg": {"model_head": "transformers.GPT2LMHeadModel"}, **base_it_module_kwargs}
#tl_entailment_cfg = RTEBoolqZeroShotClassificationConfig(enabled=True, lm_generation_cfg=TLensGenerationConfig())
test_tl_it_module_kwargs = {"tl_from_pretrained_cfg": {"enabled": True},
                            "auto_model_cfg": {"model_head": "transformers.GPT2LMHeadModel"}, **base_it_module_kwargs}
test_tl_it_module_base = ChainMap(test_tl_shared_config, test_tl_it_module_kwargs)
test_tl_it_module_optim = ChainMap(test_tl_it_module_base, test_optimizer_scheduler_init)

########################################################################################################################
# See NOTE [Test Dataset Fingerprint]
########################################################################################################################
test_dataset_state_tl = ('pytest_rte_tl',) + shared_dataset_state

w_tl = {"plugin": "transformerlens"}
