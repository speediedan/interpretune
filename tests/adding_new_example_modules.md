# Adding new test/example modules to Interpretune

0. New test/example modules are registered in`example_module_registry.yaml`
   `~/repos/interpretune/src/it_examples/example_module_registry.yaml`

   ```yaml
    gpt2.rte.transformer_lens:
    reg_info:
      model_src_key: gpt2
        task_name: rte
        adapter_combinations:
          - [core, transformer_lens]
          - [lightning, transformer_lens]
        description: Basic TL example, GPT2 with supported adapter compositions
    shared_config:
        task_name: pytest_rte_tl
        model_name_or_path: gpt2
        tokenizer_id_overrides:
          pad_token_id: 50256
        tokenizer_kwargs:
          model_input_names: ['input', 'attention_mask']
          padding_side: left
          add_bos_token: false
    registered_example_cfg:
        datamodule_cfg:
          prompt_cfg:
            class_path: it_examples.experiments.rte_boolq.RTEBoolqPromptConfig
          signature_columns: ['input', 'attention_mask', 'position_ids', 'past_key_values', 'inputs_embeds',
                            'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict']
          text_fields: ["premise", "hypothesis"]
          enable_datasets_cache: True
          train_batch_size: 2
          eval_batch_size: 2
        module_cfg:
          class_path: it_examples.experiments.rte_boolq.RTEBoolqTLConfig
          init_args:
            zero_shot_cfg:
              class_path: it_examples.experiments.rte_boolq.RTEBoolqZeroShotClassificationConfig
              init_args:
                enabled: True
                lm_generation_cfg:
                  class_path: interpretune.adapters.transformer_lens.TLensGenerationConfig
                  init_args:
                    max_new_tokens: 1
            hf_from_pretrained_cfg:
              class_path: interpretune.base.config.mixins.HFFromPretrainedConfig
              init_args:
                pretrained_kwargs:
                  device_map: cpu
                  torch_dtype: float32
                model_head: transformers.GPT2LMHeadModel
            tl_cfg:
              class_path: interpretune.adapters.transformer_lens.ITLensFromPretrainedConfig
        datamodule_cls:  # if you want to enable dataset fingerprinting, override the base test datamodule as follows:
          class_path: tests.modules.FingerprintTestITDataModule
   ```

   - if the new model requires a custom prompt (most instruction tuned models will), add the relevant example model
     prompt config dataclass to `~/repos/interpretune/src/it_examples/example_prompt_configs.py`, e.g.

   ```python
   ####################################
   # Gemma2
   ####################################

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

       def model_chat_template_fn(self, task_prompt: str, tokenization_pattern: Optional[str] = None) -> str:
           if tokenization_pattern == "gemma2-chat":
               sequence = self.USER_ROLE_START + f"{task_prompt.strip()} {self.USER_ROLE_END}"
           else:
               sequence = task_prompt.strip()
           return sequence

   @dataclass(kw_only=True)
   class RTEBoolqGemma2PromptConfig(Gemma2PromptConfig, RTEBoolqPromptConfig):
       ...
   ```

   - add new module variants to test module registry
     ```python
     TEST_DATAMODULE_BASE_CONFIGS = {
         # (dm_adapter_ctx, model_src_key)
         ...
        (Adapter.core, "llama3"): core_llama3_datamodule_cfg,

     TEST_MODULE_BASE_CONFIGS = {
         # (phase, adapter_mod_cfg_key, model_src_key)
         ...
         ("test", None, "llama3"): test_core_llama3_it_module_base,
         ("train", None, "llama3"): test_core_llama3_it_module_optim,
     ```

1. OPTIONAL: Add api key if necessary, usually can just use the default HF public gated repo key
   `~/repos/interpretune/.env`

1. OPTIONAL: If any of the test/example modules for a new model will require dataset fingerprinting enabled, update
   `~/repos/interpretune/tests/results.py` with:

   - the target tokenizer (always used for fingerprinting)

   ```python
   llama_dataset_state = ('LlamaTokenizerFast', [])
   ...
   test_dataset_state_core_llama = (TestDatasetKey.pytest_rte_hf,) +  llama_dataset_state
   ```

   - if any tests will have `dstype_agnostic=False` (e.g. new profiling tests), generate and add the relevant expected dataset fingerprint samples (dataset and `forward`)

   ```python
   deterministic_token_ids = [5674, 24140, 373, 666, 2233, 303, 783, 783, 2055, 319, 373, 910, 17074, 284, 6108]
   EXPECTED_FIRST_FWD_IDS = {"no_sample": ([],),
                           "train": (deterministic_token_ids[:default_test_bs],),
                           "train_prof": (deterministic_token_ids[:default_prof_bs],),
                           "test": (deterministic_token_ids[NUM_SAMPLE_ROWS:(NUM_SAMPLE_ROWS+default_test_bs)],),
                           "test_prof": (deterministic_token_ids[NUM_SAMPLE_ROWS:(NUM_SAMPLE_ROWS+default_prof_bs)],)}
   gpt2_dataset_state = ('GPT2TokenizerFast', deterministic_token_ids)
   ```

1. OPTIONAL:Update examples if desired, potentially adding a new model directory
   `~/repos/interpretune/src/it_examples/config/experiments/rte_boolq/lightning_rte_3b_qlora_zero_shot_test_only.yaml`

### Debugging guidance

- note when adding new model tests, you may need to toggle `force_prepare_data` while debugging transformation caching
  `~/repos/interpretune/tests/configuration.py`
