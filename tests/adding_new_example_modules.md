# Adding new test/example modules to Interpretune

0. Add api key if necessary, usually can just use the default HF public gated repo key
   `~/repos/interpretune/.env`

1. Most changes are currently to `example_module_registry` (may simplfy this and package them into a deferred init dataclass)
   `~/repos/interpretune/src/it_examples/example_module_registry.py`

   - cust prompt config, example datamodule subclass w/ tokenization func, datamodule config, module config
     ```python
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

         def __init__(self, *args: Any, **kwargs: Any) -> None:
             super().__init__(*args, **kwargs)

             # HF Datasets' transformation cache fingerprinting algo necessitates construction of these partials as the hash
             # is generated using function args, dataset file, mapping args: https://bit.ly/HF_Datasets_fingerprint_algo)
             self.tokenization_func = partial(self._tokenize_for_llama,
                                              tokenization_pattern=self.itdm_cfg.cust_tokenization_pattern)

         def _tokenize_for_llama(self, example_batch: LazyDict, tokenization_pattern: Optional[str] = None) -> BatchEncoding:
             example_batch['sequences'] = []
             assert example_batch is not None
             assert self.itdm_cfg.text_fields is not None
             assert self.itdm_cfg.prompt_cfg is not None
             # TODO: use promptsource instead of this manual approach after tinkering
             for field1, field2 in zip(example_batch[self.itdm_cfg.text_fields[0]],
                                       example_batch[self.itdm_cfg.text_fields[1]]):
                 if self.itdm_cfg.prompt_cfg.cust_task_prompt:
                     task_prompt = (self.itdm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" +
                                    field1 + "\n" +
                                    self.itdm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" +
                                    field2)
                 else:
                     task_prompt = (field1 + self.itdm_cfg.prompt_cfg.ctx_question_join + field2 \
                                    + self.itdm_cfg.prompt_cfg.question_suffix)
                 if tokenization_pattern == "llama3-chat":
                     sequence = self.itdm_cfg.prompt_cfg.SYS_ROLE_START + \
                         f"{task_prompt.strip()} {self.itdm_cfg.prompt_cfg.USER_ROLE_END}"
                 else:
                     sequence = task_prompt.strip()
                 example_batch['sequences'].append(sequence)
             features = self.tokenizer.batch_encode_plus(example_batch["sequences"], padding="longest")
             features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
             return features

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
     ```
   - add new module variants to test module registry
     ```python
     TEST_DATAMODULE_BASE_CONFIGS = {
         # (dm_adapter_ctx, model_src_key)
         ...
         (Adapter.core, "llama3"): (core_llama3_shared_config, core_llama3_datamodule_kwargs),

     TEST_MODULE_BASE_CONFIGS = {
         # (phase, adapter_mod_cfg_key, model_src_key)
         ...
         ("test", None, "llama3"): test_core_llama3_it_module_base,
         ("train", None, "llama3"): test_core_llama3_it_module_optim,
     ```

1. For modules we're not currently activating dataset sample fingerprinting for, we need to override the `<ModelType><ExpType>>DataModule` with a `SimpleDatasetStateMixin` which samples only the tokenizer and task_name (may make sense to invert this so that fingerprinting is disabled rather than enabled by default)
   `~/repos/interpretune/tests/modules.py`

   ```python
   class LlamaTestITDataModule(SimpleDatasetStateMixin, BaseTestDataModule, LlamaRTEBoolqDataModule):
       ...
   ```

1. We always verify the tokenizer, (for datasets that have fingerprint testing enabled, we need to update that enum as well here)
   `~/repos/interpretune/tests/results.py`

   ```python
   llama_dataset_state = ('LlamaTokenizerFast', [])
   ...
   test_dataset_state_core_llama = (TestDatasetKey.pytest_rte_hf,) +  llama_dataset_state

   ```

1. Usually add a model debugging cfg for unit testing
   `~/repos/interpretune/tests/unit/cfg_aliases.py`

   ```python
   @dataclass(kw_only=True)
   class LightningLlama3DebugCfg(BaseCfg):
   ```

1. Usually want to add a test fixture associated with a new model cfg
   `~/repos/interpretune/tests/conftest.py`

   ```python
       "l_llama3_debug": FixtureCfg(test_cfg=LightningLlama3DebugCfg),
   ```

1. Add the supported `DebugGeneration` chat format (TODO: maybe abstract this)
   `~/repos/interpretune/src/interpretune/extensions/debug_generation.py`

   ```python

       def chat_debug_sequences(self, format = 'llama3', sequences: Optional[List] = None) -> List:
       ...
           sequences = sequences or self.phandle.it_cfg.debug_lm_cfg.raw_debug_sequences
           match format:
               case 'llama3':
                   return [self.phandle.datamodule.itdm_cfg.prompt_cfg.SYS_ROLE_START + \
                           f"{ex.strip()} {self.phandle.datamodule.itdm_cfg.prompt_cfg.USER_ROLE_END}" \
                               for ex in sequences]
   ```

1. Update examples if desired, potentially adding a new model directory
   `~/repos/interpretune/src/it_examples/config/experiments/rte_boolq/lightning_rte_3b_qlora_zero_shot_test_only.yaml`

### Debugging guidance

- note when adding new model tests, you may need to toggle `force_prepare_data` while debugging transformation caching
  `~/repos/interpretune/tests/configuration.py`
