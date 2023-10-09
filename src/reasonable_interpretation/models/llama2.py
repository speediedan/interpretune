from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from fts_examples import _HF_AVAILABLE
from reasonable_interpretation.base.base_datamodule import RIDataModule
from reasonable_interpretation.base.base_module import RIModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

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
        assert example_batch is not None
        assert self.ridm_cfg.text_fields is not None
        assert self.ridm_cfg.prompt_cfg is not None
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
                sequence = self.ridm_cfg.prompt_cfg.SYS_PREFIX + \
                    f"{task_prompt.strip()} {self.ridm_cfg.prompt_cfg.E_INST}"
            else:
                sequence = task_prompt.strip()
            example_batch['sequences'].append(sequence)
        features = self.tokenizer.batch_encode_plus(example_batch["sequences"], padding="longest")
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        return features


class Llama2RIModule(RIModule):

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
        sequences = sequences or self.ri_cfg.debug_lm_cfg.raw_debug_sequences
        return [self.trainer.datamodule.tokenizer.bos_token + \
            self.trainer.datamodule.ridm_cfg.prompt_cfg.SYS_PREFIX + \
                f"{ex.strip()} {self.trainer.datamodule.ridm_cfg.prompt_cfg.E_INST}" \
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
        sequences = sequences or self.ri_cfg.debug_lm_cfg.raw_debug_sequences
        return [f"{ex.strip()}" for ex in sequences]
