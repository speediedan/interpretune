from typing import Any, Dict, Optional
from dataclasses import dataclass

from fts_examples import _HF_AVAILABLE
from reasonable_interpretation.base.base_datamodule import RIDataModule

if _HF_AVAILABLE:
    from datasets.arrow_dataset import LazyDict
    from transformers.tokenization_utils_base import BatchEncoding


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


class Llama2RIDataModule(RIDataModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenization_func = self._tokenize_for_llama2

    def _tokenize_for_llama2(self, example_batch: LazyDict) -> BatchEncoding:
        example_batch['sequences'] = []
        # TODO: use promptsource instead of this manual approach after tinkering
        for field1, field2 in zip(example_batch[self.ridm_cfg.text_fields[0]], 
                                  example_batch[self.ridm_cfg.text_fields[1]]):
            if self.ridm_cfg.prompt_cfg.cust_task_prompt:
                task_prompt = (self.ridm_cfg.prompt_cfg.cust_task_prompt['context'] + "\n" +
                               field1 + "\n" +
                               self.ridm_cfg.prompt_cfg.cust_task_prompt['question'] + "\n" + 
                               field2)
            else:
                task_prompt = (field1 + self.ridm_cfg.prompt_cfg.ctx_question_join + field2 \
                               + self.ridm_cfg.prompt_cfg.question_suffix)
            if self.ridm_cfg.cust_tokenization_pattern == "llama2-chat":
                sequence = self.ridm_cfg.prompt_cfg.SYS_PREFIX + f"{task_prompt.strip()} {self.ridm_cfg.prompt_cfg.E_INST}"
            else:
                sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer.batch_encode_plus(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        return features
