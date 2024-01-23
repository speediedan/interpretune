from typing import Optional
from dataclasses import dataclass, field

from transformers.tokenization_utils_base import BatchEncoding

from interpretune.base.config.shared import ITSerializableCfg
from interpretune.base.mixins.core import ProfilerHooksMixin
from interpretune.utils.types import  STEP_OUTPUT
from interpretune.utils.logging import rank_zero_warn


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

    @ProfilerHooksMixin.memprofilable
    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        if self.it_cfg.zero_shot_cfg.enabled:
            self.zero_shot_test_step(batch, batch_idx)
        else:
            self.default_test_step(batch, batch_idx)

    def zero_shot_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> \
        Optional[STEP_OUTPUT]:
        rank_zero_warn("`zero_shot_test_step` must be implemented to be used with the Interpretune.")

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> Optional[STEP_OUTPUT]:
        rank_zero_warn("`default_test_step` must be implemented to be used with the Interpretune.")
