from typing import Optional

from torch.testing import assert_close
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer

from interpretune.base.base_lightning_modules import ITLightningDataModule, ITLightningModule
from interpretune.models.core.gpt2 import GPT2BoolRTEDataModule


class GPT2BoolRTELightningDataModule(GPT2BoolRTEDataModule, ITLightningDataModule):
    ...


class GPT2ITLightningModule(ITLightningModule):

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
