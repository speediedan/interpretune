from typing import Any, Optional, List
from dataclasses import dataclass

import torch
from torch.nn import CrossEntropyLoss
import evaluate
from torch.testing import assert_close
from transformers.tokenization_utils_base import BatchEncoding
from transformer_lens import HookedTransformer

from interpretune.plugins.transformer_lens import ITLensLightningModule
from interpretune.utils.types import STEP_OUTPUT
from interpretune.plugins.transformer_lens import ITLensModule, ITLensConfig
from interpretune.base.mixins.core import ProfilerHooksMixin
from it_examples.experiments.rte_boolq.core import RTEBoolqEntailmentMapping, RTEBoolqModuleMixin


@dataclass(kw_only=True)
class RTEBoolqTLConfig(RTEBoolqEntailmentMapping, ITLensConfig):
    ...

class RTEBoolqLMHeadSteps:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # when using TransformerLens, we need to manually calculate our loss from logit output
        self.loss_fn = CrossEntropyLoss()

    def labels_to_ids(self, labels: List[str]) -> List[int]:
        return torch.take(self.it_cfg.entailment_mapping_indices, labels)

    def logits_and_labels(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        label_ids = self.labels_to_ids(batch.pop("labels"))
        logits = self(**batch)
        return torch.squeeze(logits[:, -1, :], dim=1), label_ids

    @ProfilerHooksMixin.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        # TODO: need to be explicit about the compatibility constraints/contract
        answer_logits, labels = self.logits_and_labels(batch, batch_idx)
        loss = self.loss_fn(answer_logits, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @ProfilerHooksMixin.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        answer_logits, labels = self.logits_and_labels(batch, batch_idx)
        val_loss = self.loss_fn(answer_logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        # TODO: condition this on a metric being configured and calculate per_example_answers for metric input
        # like with zero_shot_test_step
        #metric_dict = self.metric.compute(predictions=answer_logits, references=labels)
        #metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               #metric_dict.items()))
        #self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        labels = batch.pop("labels")
        outputs = self.model.generate(input=batch['input'],
                                      #pad_token_id=self.datamodule.tokenizer.pad_token_id,
                                      **self.it_cfg.lm_generation_cfg.__dict__)
        #stacked_scores = torch.stack([out for out in outputs.logits], dim=0).cpu()
        stacked_scores = outputs.logits.cpu()
        assert self.it_cfg.entailment_mapping_indices is not None
        answer_logits = torch.index_select(stacked_scores, -1, self.it_cfg.entailment_mapping_indices)
        per_example_answers, _ = torch.max(answer_logits, dim=1)
        preds = torch.argmax(per_example_answers, axis=1)  # type: ignore[call-arg]
        #labels = batch["labels"]
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        # run predict on val dataset for now
        batch.pop("labels")
        outputs = self(**batch)
        # TODO: switch to zero shot instead of this default sequenceclassification head approach
        logits = outputs[:2]
        if self.it_cfg.num_labels >= 1:
            torch.argmax(logits, axis=1)  # type: ignore[call-arg]
        elif self.it_cfg.num_labels == 1:
            logits.squeeze()
        #labels = batch["labels"]
        # TODO: move TL examples to use zeroshot instead of default test step
        # TODO: condition this on a metric being configured
        #self.log("predict_loss", test_loss, prog_bar=True, sync_dist=True)
        # metric_dict = self.metric.compute(predictions=preds, references=labels)
        # metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
        #                        metric_dict.items()))
        # self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

class GPT2RTEBoolqITLensModule(RTEBoolqLMHeadSteps, RTEBoolqModuleMixin, ITLensModule):

    def _load_metric(self) -> None:
        self.metric = evaluate.load("super_glue", self.it_cfg.task_name,
                                    experiment_id=self.init_hparams['experiment_id'])

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


#### Model/Experiment TransformerLens Lightning Modules
class GPT2ITLensLightningModule(RTEBoolqLMHeadSteps, RTEBoolqModuleMixin, ITLensLightningModule):

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

    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.probe_setup_only:
            return f"predict step entered and exited successfully for batch_idx: {batch_idx}"
        else:
            dump_path = self.core_log_dir / f'fp32_activations_logits_rte_validation_gpt2_batch_{batch_idx}.pt'
            original_logits, cache = self.model.run_with_cache(batch['input_ids'])
            logits_and_activations = {"original_logits": original_logits, "cache": cache}
            torch.save(logits_and_activations, dump_path)
            return f"saved activations and logits to: {dump_path}"
