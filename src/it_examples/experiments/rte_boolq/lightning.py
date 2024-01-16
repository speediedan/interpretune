from typing import Any, Optional, List

import torch
from torch.testing import assert_close
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer

from interpretune.base.datamodules import ITLightningDataModule
from interpretune.base.modules import ITLightningModule
from interpretune.plugins.transformer_lens import ITLensLightningModule
from it_examples.experiments.rte_boolq.core import (GPT2RTEBoolqDataModule, Llama2RTEBoolqDataModule,
                                                    RTEBoolqModuleMixin)

#### Model/Experiment Datamodules

class GPT2RTEBoolqLightningDataModule(GPT2RTEBoolqDataModule, ITLightningDataModule):
    ...


class Llama2RTEBoolqLightningDataModule(Llama2RTEBoolqDataModule, ITLightningDataModule):
    ...


#### Model/Experiment Lightning Modules

class GPT2ITLightningModule(RTEBoolqModuleMixin, ITLightningModule):

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

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        # uncomment to run debugging sanity check before running the main test step
        # self.tl_gpt2_parity_test()
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


class Llama2ITLightningModule(RTEBoolqModuleMixin, ITLightningModule):

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


#### Model/Experiment TransformerLens Lightning Modules

class GPT2ITLensLightningModule(RTEBoolqModuleMixin, ITLensLightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probe_setup_only = True

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

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        # uncomment to run debugging sanity check before running the main test step
        # self.tl_gpt2_parity_test()
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

    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.probe_setup_only:
            return f"predict step entered and exited successfully for batch_idx: {batch_idx}"
        else:
            dump_path = self.core_log_dir / f'fp32_activations_logits_rte_validation_gpt2_batch_{batch_idx}.pt'
            original_logits, cache = self.model.run_with_cache(batch['input_ids'])
            logits_and_activations = {"original_logits": original_logits, "cache": cache}
            torch.save(logits_and_activations, dump_path)
            return f"saved activations and logits to: {dump_path}"
