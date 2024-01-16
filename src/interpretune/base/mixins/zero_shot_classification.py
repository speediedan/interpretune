from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers.tokenization_utils_base import BatchEncoding

from interpretune.config.shared import ITSerializableCfg
from interpretune.utils.types import  STEP_OUTPUT


@dataclass(kw_only=True)
class BaseGenerationConfig(ITSerializableCfg):
    max_new_tokens: int = 5  # nb maxing logits over multiple tokens (n<=5) will yield a very slight perf gain versus 1
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0

@dataclass(kw_only=True)
class HFGenerationConfig(BaseGenerationConfig):
    use_cache: bool = True
    repetition_penalty: float = 1.0
    output_attentions: bool = False
    output_hidden_states: bool = False
    length_penalty: float = 1.0
    output_scores: bool = True
    return_dict_in_generate: bool = True

@dataclass(kw_only=True)
class ZeroShotClassificationConfig(ITSerializableCfg):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=lambda: HFGenerationConfig())

class ZeroShotStepMixin:
        # TODO: make memprofilable by default directly? (already usually wrapped by test_step)
    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      pad_token_id=self.datamodule.tokenizer.pad_token_id,
                                      **self.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__)
        stacked_scores = torch.stack([out for out in outputs['scores']], dim=0).cpu()
        assert self.it_cfg.zero_shot_cfg.entailment_mapping_indices is not None
        answer_logits = torch.index_select(stacked_scores, -1, self.it_cfg.zero_shot_cfg.entailment_mapping_indices)
        per_example_answers, _ = torch.max(answer_logits, dim=0)
        preds = torch.argmax(per_example_answers, axis=1)  # type: ignore[call-arg]
        labels = batch["labels"]
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        metric_dict = dict(map(lambda x: (x[0], torch.tensor(x[1], device=self.device).to(torch.float32)),
                               metric_dict.items()))
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    # TODO: make memprofilable by default directly? (already usually wrapped by test_step)
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
