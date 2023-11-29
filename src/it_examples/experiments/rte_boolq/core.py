from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field

import torch
import evaluate
from torch.testing import assert_close
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer

from interpretune.utils.types import STEP_OUTPUT
from interpretune.base.it_module import ITModule, ITHookedModule
from interpretune.utils.logging import rank_zero_warn
from interpretune.base.config_classes import ITZeroShotClassificationConfig, ITConfig, LMGenerationConfig
from it_examples.data.rte_bool import RTEBoolqDataModule, DEFAULT_TASK, TASK_NUM_LABELS, INVALID_TASK_MSG


@dataclass
class RTEBoolqZeroShotClassificationConfig(ITZeroShotClassificationConfig):
    enabled: bool = False
    lm_generation_cfg: LMGenerationConfig = field(default_factory=lambda: LMGenerationConfig())
    entailment_mapping: Tuple = ("Yes", "No")  # RTE style, invert mapping for BoolQ
    entailment_mapping_indices: Optional[torch.Tensor] = None


# TODO: use class inheritance instead with 3.10 since it is now minimum version
@dataclass
class GPT2PromptConfig:
    ctx_question_join: str = 'Does the previous passage imply that '
    question_suffix: str = '? Answer with only one word, either Yes or No.'
    cust_task_prompt: Optional[Dict[str, Any]] = None

class RTEBoolqModuleMixin:
    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        if it_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(it_cfg.task_name + INVALID_TASK_MSG)
            it_cfg.task_name = DEFAULT_TASK
        it_cfg.num_labels = 0 if it_cfg.zero_shot_cfg.enabled else TASK_NUM_LABELS[it_cfg.task_name]
        return it_cfg

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self.init_hparams['experiment_id'])


class GPT2RTEBoolqDataModule(RTEBoolqDataModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenization_func = self._tokenize_for_gpt2

    def _tokenize_for_gpt2(self, example_batch: LazyDict) -> BatchEncoding:
        example_batch['sequences'] = []
        assert example_batch is not None
        assert self.itdm_cfg.text_fields is not None
        assert self.itdm_cfg.prompt_cfg is not None
        # TODO: use promptsource instead of this manual approach after tinkering
        for field1, field2 in zip(example_batch[self.itdm_cfg.text_fields[0]],
                                  example_batch[self.itdm_cfg.text_fields[1]]):
            if self.itdm_cfg.prompt_cfg.cust_task_prompt:
                # task_prompt = (self.itdm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" +
                #                field1 + "\n\n" +
                #                self.itdm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" +
                #                field2)
                task_prompt = (self.itdm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" + field1 + "\n\n" +
                               self.itdm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" + field2)
            else:
                task_prompt = (field1 + self.itdm_cfg.prompt_cfg.ctx_question_join + field2 \
                               + self.itdm_cfg.prompt_cfg.question_suffix)
            sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        return features


class GPT2RTEBoolqITModule(RTEBoolqModuleMixin, ITModule):

    def temp_hooked_test(self, model_description_text: str) -> None:

        loss = self.hooked_ref_gpt2(model_description_text, return_type="loss")
        print("Model loss:", loss)

        print(self.hooked_ref_gpt2.to_str_tokens("gpt2"))
        print(self.hooked_ref_gpt2.to_tokens("gpt2"))
        print(self.hooked_ref_gpt2.to_string([50256, 70, 457, 17]))

        logits = self.hooked_ref_gpt2(model_description_text, return_type="logits")
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = self.hooked_ref_gpt2.to_tokens(model_description_text).squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()

        print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
        print(f"Correct words: {self.hooked_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])}")

        return num_correct/len(true_tokens), self.hooked_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])

    def hooked_gpt2_parity_test(self) -> None:
        model_description_text = """## Loading Models
        HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with
        `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer
        architecture, designed to be clean, consistent and interpretability-friendly.
        For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out,
        let's find the loss on this paragraph!"""
        # note loading hooked transformer after loading unhooked to avoid an issue where cpu model was partially
        # referring to tensors moved to the GPU (haven't debugged root cause yet)
        self.hooked_ref_gpt2: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device="cpu",
                                                                                    dtype="float32")
        hooked_acc, hooked_correct_tokens = self.temp_hooked_test(model_description_text)
        acc, correct_tokens = self.lm_debug.top1_token_accuracy_on_sample(model_description_text)
        assert_close(actual=acc.cpu(), expected=hooked_acc, rtol=0.03, atol=0)
        assert_close(actual=len(correct_tokens), expected=len(hooked_correct_tokens), rtol=0.03, atol=0)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        # uncomment to run debugging sanity check before running the main test step
        # self.hooked_gpt2_parity_test()
        # answers, full_outputs = self.lm_debug.debug_generate_batch(self.lm_debug.debug_sequences(
        #     ["Hello, I'm a large language model,", "The day after Tuesday is naturally"]),
        #                             max_new_tokens=30,
        #                             gen_config_override={"output_scores": True, "num_return_sequences": 5})
        #ppl = self.lm_debug.perplexity_on_sample()
        # self.lm_debug.debug_generate_serial(self.lm_debug.debug_sequences(), max_new_tokens=10,
        #                                     gen_config_override={"output_scores": True,
        #                                                          "num_return_sequences": 3})
        # self.lm_debug.debug_generate_serial(self.lm_debug.debug_sequences(), max_new_tokens=10)
        # self.lm_debug.debug_generate_batch(self.lm_debug.debug_sequences(),  max_new_tokens=10,
        #                                    gen_config_override={"output_scores": True})
        super().zero_shot_test_step(batch, batch_idx, dataloader_idx)


class GPT2RTEBoolqITHookedModule(RTEBoolqModuleMixin, ITHookedModule):

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self.init_hparams['experiment_id'])

    def temp_hooked_test(self, model_description_text: str) -> None:

        loss = self.hooked_ref_gpt2(model_description_text, return_type="loss")
        print("Model loss:", loss)

        print(self.hooked_ref_gpt2.to_str_tokens("gpt2"))
        print(self.hooked_ref_gpt2.to_tokens("gpt2"))
        print(self.hooked_ref_gpt2.to_string([50256, 70, 457, 17]))

        logits = self.hooked_ref_gpt2(model_description_text, return_type="logits")
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = self.hooked_ref_gpt2.to_tokens(model_description_text).squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()

        print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
        print(f"Correct words: {self.hooked_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])}")

        return num_correct/len(true_tokens), self.hooked_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])

    def hooked_gpt2_parity_test(self) -> None:
        model_description_text = """## Loading Models
        HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with
        `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer
        architecture, designed to be clean, consistent and interpretability-friendly.
        For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out,
        let's find the loss on this paragraph!"""
        # note loading hooked transformer after loading unhooked to avoid an issue where cpu model was partially
        # referring to tensors moved to the GPU (haven't debugged root cause yet)
        self.hooked_ref_gpt2: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device="cpu",
                                                                                    dtype="float32")
        hooked_acc, hooked_correct_tokens = self.temp_hooked_test(model_description_text)
        acc, correct_tokens = self.lm_debug.top1_token_accuracy_on_sample(model_description_text)
        assert_close(actual=acc.cpu(), expected=hooked_acc, rtol=0.03, atol=0)
        assert_close(actual=len(correct_tokens), expected=len(hooked_correct_tokens), rtol=0.03, atol=0)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        # uncomment to run debugging sanity check before running the main test step
        # self.hooked_gpt2_parity_test()
        # answers, full_outputs = self.lm_debug.debug_generate_batch(self.lm_debug.debug_sequences(
        #     ["Hello, I'm a large language model,", "The day after Tuesday is naturally"]),
        #                             max_new_tokens=30,
        #                             gen_config_override={"output_scores": True, "num_return_sequences": 5})
        #ppl = self.lm_debug.perplexity_on_sample()
        # self.lm_debug.debug_generate_serial(self.lm_debug.debug_sequences(), max_new_tokens=10,
        #                                     gen_config_override={"output_scores": True,
        #                                                          "num_return_sequences": 3})
        # self.lm_debug.debug_generate_serial(self.lm_debug.debug_sequences(), max_new_tokens=10)
        # self.lm_debug.debug_generate_batch(self.lm_debug.debug_sequences(),  max_new_tokens=10,
        #                                    gen_config_override={"output_scores": True})
        super().zero_shot_test_step(batch, batch_idx, dataloader_idx)


@dataclass
class Llama2PromptConfig:
    sys_prompt: str = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as"
    " possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic,"
    " dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
    " If a question does not make any sense, or is not factually coherent, explain why instead of answering something"
    " not correct. If you don't know the answer to a question, please don't share false information.")
    ctx_question_join: str = 'Does the previous passage imply that '
    question_suffix: str = '? Answer with only one word, either Yes or No.'
    cust_task_prompt: Optional[Dict[str, Any]] = None
    B_INST: str = "[INST]"
    E_INST: str = "[/INST]"
    B_SYS: str = "<<SYS>>\n"
    E_SYS: str = "\n<</SYS>>\n\n"

    def __post_init__(self) -> None:
        self.SYS_PREFIX_START = f"{self.B_INST} " + self.B_SYS
        self.SYS_PREFIX = self.SYS_PREFIX_START + self.sys_prompt + self.E_SYS


class Llama2RTEBoolqDataModule(RTEBoolqDataModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenization_func = self._tokenize_for_llama2

    def _tokenize_for_llama2(self, example_batch: LazyDict) -> BatchEncoding:
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
            if self.itdm_cfg.cust_tokenization_pattern == "llama2-chat":
                sequence = self.itdm_cfg.prompt_cfg.SYS_PREFIX + \
                    f"{task_prompt.strip()} {self.itdm_cfg.prompt_cfg.E_INST}"
            else:
                sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer.batch_encode_plus(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        return features


class Llama2RTEBoolqITModule(RTEBoolqModuleMixin, ITModule):

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self.init_hparams['experiment_id'])

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        # uncomment to run llama-specific debugging sanity check before running the main test step
        # self.lm_debug.debug_generate_serial(self.sys_inst_debug_sequences())
        # self.lm_debug.debug_generate_batch(self.sys_inst_debug_sequences())
        super().zero_shot_test_step(batch, batch_idx, dataloader_idx)

    # some Llama2-specific debug helper functions
    def sys_inst_debug_sequences(self, sequences: Optional[List] = None) -> List:
        """_summary_

        Args:
            sequences (Optional[List], optional): _description_. Defaults to None.

        Returns:
            List: _description_

        Usage:

        ```python
        # when using a llama2 chat model, you'll want to have input tokenized with sys and inst metadata
        # to do so with some reasonable default questions as a sanity check and in batch mode:
        self.lm_debug.debug_generate_batch(self.sys_inst_debug_sequences())
        # to narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.lm_debug.debug_generate_serial(self.sys_inst_debug_sequences())
        # to override the defaults (both questions and current `max_new_tokens` config)
        self.lm_debug.debug_generate_batch(self.sys_inst_debug_sequences([
            'What is the color of a cloudless sky?', 'How many days are in a year?']), max_new_tokens=25)
        ```
        """
        sequences = sequences or self.it_cfg.debug_lm_cfg.raw_debug_sequences
        return [self.trainer.datamodule.tokenizer.bos_token + \
            self.trainer.datamodule.itdm_cfg.prompt_cfg.SYS_PREFIX + \
                f"{ex.strip()} {self.trainer.datamodule.itdm_cfg.prompt_cfg.E_INST}" \
                for ex in sequences]

    def no_sys_inst_debug_sequences(self, sequences: Optional[List] = None) -> List:
        """_summary_

        Args:
            sequences (Optional[List], optional): _description_. Defaults to None.

        Returns:
            List: _description_

        Usage:
        ```python
        # one can use this method to probe non-chat fine-tuned LLAMA2 models (just the raw sequences, no SYS
        # or INST metadata)
        self.lm_debug.debug_generate_batch(self.no_sys_inst_debug_sequences(), max_new_tokens=25)
        ```
        """
        sequences = sequences or self.it_cfg.debug_lm_cfg.raw_debug_sequences
        return [f"{ex.strip()}" for ex in sequences]
