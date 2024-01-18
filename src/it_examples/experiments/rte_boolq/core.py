import os
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from pprint import pformat

import torch
import evaluate
from torch.testing import assert_close
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer

from interpretune.utils.types import STEP_OUTPUT
from interpretune.base.modules import ITModule
from interpretune.base.datamodules import ITLightningDataModule
from interpretune.base.modules import ITLightningModule
from interpretune.utils.logging import rank_zero_warn, rank_zero_info
from interpretune.base.config.module import ITConfig
from interpretune.base.mixins.core import ProfilerHooksMixin
from interpretune.base.mixins.zero_shot_classification import (ZeroShotClassificationConfig, BaseGenerationConfig,
                                                          HFGenerationConfig)
from interpretune.base.config.datamodule import PromptConfig
from it_examples.data.rte_bool import RTEBoolqDataModule, DEFAULT_TASK, TASK_NUM_LABELS, INVALID_TASK_MSG


@dataclass(kw_only=True)
class RTEBoolqEntailmentMapping:
    entailment_mapping: Tuple = ("Yes", "No")  # RTE style, invert mapping for BoolQ
    entailment_mapping_indices: Optional[torch.Tensor] = None

@dataclass(kw_only=True)
class RTEBoolqZeroShotClassificationConfig(RTEBoolqEntailmentMapping, ZeroShotClassificationConfig):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=lambda: HFGenerationConfig())

    def __repr__(self):
        return f"Zero-Shot Classification Config: {os.linesep}{pformat(self.__dict__)}"


@dataclass(kw_only=True)
class RTEBoolqPromptConfig(PromptConfig):
    ctx_question_join: str = 'Does the previous passage imply that '
    question_suffix: str = '? Answer with only one word, either Yes or No.'
    cust_task_prompt: Optional[Dict[str, Any]] = None

# add our custom model attributes
@dataclass(kw_only=True)
class RTEBoolqConfig(RTEBoolqEntailmentMapping, ITConfig):
    ...


class RTEBoolqClassificationHeadSteps:
    @ProfilerHooksMixin.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
         # TODO: decide whether to build a closure for the core training_step to enable identical
         # core/lightning module training_steps in more cases (need to be explicit about the compatibility constraints)
        outputs = self(**batch)
        loss, _other_outputs = outputs[0], outputs[1:]
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @ProfilerHooksMixin.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        # TODO: condition this on a metric being configured
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    # TODO: test overriding default test_step
    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      pad_token_id=self.datamodule.tokenizer.pad_token_id,
                                      **self.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__)
        stacked_scores = torch.stack([out for out in outputs['scores']], dim=0).cpu()
        assert self.it_cfg.entailment_mapping_indices is not None
        answer_logits = torch.index_select(stacked_scores, -1, self.it_cfg.entailment_mapping_indices)
        per_example_answers, _ = torch.max(answer_logits, dim=0)
        preds = torch.argmax(per_example_answers, axis=1)  # type: ignore[call-arg]
        labels = batch["labels"]
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        # run predict on val dataset for now
        outputs = self(**batch)
        test_loss, logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        # TODO: condition this on a metric being configured
        #self.log("predict_loss", test_loss, prog_bar=True, sync_dist=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    @ProfilerHooksMixin.memprofilable
    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        # run predict on val dataset for now
        # TODO: clean this up and allow for passing arbitrary data
        outputs = self(**batch)
        _, logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        # TODO: condition this on a metric being configured
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        rank_zero_info(metric_dict)


class RTEBoolqModuleMixin:

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._init_entailment_mapping()

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        if it_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(it_cfg.task_name + INVALID_TASK_MSG)
            it_cfg.task_name = DEFAULT_TASK
        it_cfg.num_labels = 0 if it_cfg.zero_shot_cfg.enabled else TASK_NUM_LABELS[it_cfg.task_name]
        return it_cfg

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self.init_hparams['experiment_id'])

    def _init_entailment_mapping(self) -> None:
        ent_cfg, tokenizer = self.it_cfg, self.datamodule.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(ent_cfg.entailment_mapping)
        device = self.device if isinstance(self.device, torch.device) else self.output_device
        ent_cfg.entailment_mapping_indices = torch.tensor(token_ids).to(device)


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
                task_prompt = (self.itdm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" + field1 + "\n\n" +
                               self.itdm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" + field2)
            else:
                task_prompt = (field1 + self.itdm_cfg.prompt_cfg.ctx_question_join + field2 \
                               + self.itdm_cfg.prompt_cfg.question_suffix)
            sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        if (primary_input := self.tokenizer.model_input_names[0]) != "input_ids":
            features[primary_input] = features["input_ids"]
        return features


class GPT2RTEBoolqITModule(RTEBoolqClassificationHeadSteps, RTEBoolqModuleMixin, ITModule):

    def temp_tl_test(self, model_description_text: str) -> None:

        loss = self.tl_ref_gpt2(model_description_text, return_type="loss")
        print("Model loss:", loss)

        print(self.tl_ref_gpt2.to_str_tokens("gpt2"))
        print(self.tl_ref_gpt2.to_tokens("gpt2"))
        print(self.tl_ref_gpt2.to_string([50256, 70, 457, 17]))

        logits = self.tl_ref_gpt2(model_description_text, return_type="logits")
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = self.tl_ref_gpt2.to_tokens(model_description_text).squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()

        print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
        print(f"Correct words: {self.tl_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])}")

        return num_correct/len(true_tokens), self.tl_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])

    def tl_gpt2_parity_test(self) -> None:
        model_description_text = """## Loading Models
        HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with
        `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer
        architecture, designed to be clean, consistent and interpretability-friendly.
        For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out,
        let's find the loss on this paragraph!"""
        # note loading hooked transformer after loading unhooked to avoid an issue where cpu model was partially
        # referring to tensors moved to the GPU (haven't debugged root cause yet)
        self.tl_ref_gpt2: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device="cpu",
                                                                                    dtype="float32")
        tl_acc, tl_correct_tokens = self.temp_tl_test(model_description_text)
        acc, correct_tokens = self.lm_debug.top1_token_accuracy_on_sample(model_description_text)
        assert_close(actual=acc.cpu(), expected=tl_acc, rtol=0.03, atol=0)
        assert_close(actual=len(correct_tokens), expected=len(tl_correct_tokens), rtol=0.03, atol=0)


@dataclass(kw_only=True)
class Llama2PromptConfig(RTEBoolqPromptConfig):
    sys_prompt: str = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as"
    " possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic,"
    " dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
    " If a question does not make any sense, or is not factually coherent, explain why instead of answering something"
    " not correct. If you don't know the answer to a question, please don't share false information.")
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


class Llama2RTEBoolqITModule(RTEBoolqClassificationHeadSteps, RTEBoolqModuleMixin, ITModule):

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self.init_hparams['experiment_id'])

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
        return [self.datamodule.tokenizer.bos_token + \
            self.datamodule.itdm_cfg.prompt_cfg.SYS_PREFIX + \
                f"{ex.strip()} {self.datamodule.itdm_cfg.prompt_cfg.E_INST}" \
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


#### Model/Experiment Datamodules

class GPT2RTEBoolqLightningDataModule(GPT2RTEBoolqDataModule, ITLightningDataModule):
    ...


class Llama2RTEBoolqLightningDataModule(Llama2RTEBoolqDataModule, ITLightningDataModule):
    ...


#### Model/Experiment Lightning Modules

class GPT2ITLightningModule(RTEBoolqClassificationHeadSteps, RTEBoolqModuleMixin, ITLightningModule):

    def temp_tl_test(self, model_description_text: str) -> None:

        loss = self.tl_ref_gpt2(model_description_text, return_type="loss")
        print("Model loss:", loss)

        print(self.tl_ref_gpt2.to_str_tokens("gpt2"))
        print(self.tl_ref_gpt2.to_tokens("gpt2"))
        print(self.tl_ref_gpt2.to_string([50256, 70, 457, 17]))

        logits = self.tl_ref_gpt2(model_description_text, return_type="logits")
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = self.tl_ref_gpt2.to_tokens(model_description_text).squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()

        print(f"Model accuracy: {num_correct}/{len(true_tokens)}")
        print(f"Correct words: {self.tl_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])}")

        return num_correct/len(true_tokens), self.tl_ref_gpt2.to_str_tokens(prediction[prediction == true_tokens])

    def tl_gpt2_parity_test(self) -> None:
        model_description_text = """## Loading Models
        HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with
        `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer
        architecture, designed to be clean, consistent and interpretability-friendly.
        For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out,
        let's find the loss on this paragraph!"""
        # note loading tl transformer after loading untl to avoid an issue where cpu model was partially
        # referring to tensors moved to the GPU (haven't debugged root cause yet)
        self.tl_ref_gpt2: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small", device="cpu",
                                                                                    dtype="float32")
        tl_acc, tl_correct_tokens = self.temp_tl_test(model_description_text)
        acc, correct_tokens = self.lm_debug.top1_token_accuracy_on_sample(model_description_text)
        assert_close(actual=acc.cpu(), expected=tl_acc, rtol=0.03, atol=0)
        assert_close(actual=len(correct_tokens), expected=len(tl_correct_tokens), rtol=0.03, atol=0)


class Llama2ITLightningModule(RTEBoolqClassificationHeadSteps, RTEBoolqModuleMixin, ITLightningModule):

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
        return [self.datamodule.tokenizer.bos_token + \
            self.datamodule.itdm_cfg.prompt_cfg.SYS_PREFIX + \
                f"{ex.strip()} {self.datamodule.itdm_cfg.prompt_cfg.E_INST}" \
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
