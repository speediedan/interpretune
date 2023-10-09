from typing import Optional, List, Any
from abc import ABC

import torch


class DebugGenerationMixin(ABC):
    """Give user-provided callbacks with the ability to connect to another user-provided callback.

    This resolution logic is provided in order to avoid callback-dependent trainer attributes (e.g.
    trainer.finetuningscheduler_callback)
    """
    # TODO:
    # - may make sense to add some additional debugging methods that parse and analyze all of the generated outputs
    #   including `output_attentions` and `output_hidden_states` etc.

    def __init__(
        self,
    ) -> None:
        """Arguments."""
        super().__init__()
        self.phandle = None

    def connect_lmdebug(self, obj_ref: Any) -> None:
        self.phandle = obj_ref

    def sys_inst_debug_sequences(self, sequences: Optional[List] = None) -> List:
        sequences = sequences or self.phandle.ri_cfg.debug_lm_cfg.raw_debug_sequences
        return [self.phandle.trainer.datamodule.tokenizer.bos_token + \
            self.phandle.trainer.datamodule.ridm_cfg.prompt_cfg.SYS_PREFIX + \
                f"{ex.strip()} {self.phandle.trainer.datamodule.ridm_cfg.prompt_cfg.E_INST}" \
                for ex in sequences]

    def no_sys_inst_debug_sequences(self, sequences: Optional[List] = None) -> List:
        sequences = sequences or self.phandle.ri_cfg.debug_lm_cfg.raw_debug_sequences
        return [f"{ex.strip()}" for ex in sequences]

    def _debug_generate(self, inputs: List|torch.Tensor, max_new_tokens_override: Optional[int] = None) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_
            max_new_tokens_override (Optional[int], optional): _description_. Defaults to None.

        Usage:

        ```python
        # when using a llama2 chat model, you'll want to have input tokenized with sys and inst metadata
        # to do so with some reasonable default questions as a sanity check and in batch mode:
        self.lm_debug.debug_generate_batch(self.lm_debug.sys_inst_debug_sequences())
        # to  narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.lm_debug.debug_generate_serial(self.lm_debug.sys_inst_debug_sequences())
        # to override the defaults (both questions and current `max_new_tokens` config)
        self.lm_debug.debug_generate_batch(self.lm_debug.sys_inst_debug_sequences([
            'What is the color of a cloudless sky?', 'How many days are in a year?']), max_new_tokens=25)
        # similarly, one can use this method to probe non-chat fine-tuned LLAMA2 models (just the raw sequences, no SYS
        # or INST metadata)
        self.lm_debug.debug_generate_batch(self.lm_debug.no_sys_inst_debug_sequences(), max_new_tokens=25)
        ```
        """
        # note we're not using a context manager here, keeping our new override for subsequent debugging convenience
        if max_new_tokens_override:
            self.phandle.ri_cfg.zero_shot_cfg.lm_generation_cfg.max_new_tokens = max_new_tokens_override
        return self.phandle.model.generate(input_ids=inputs,
                                           pad_token_id=self.phandle.trainer.datamodule.tokenizer.pad_token_id,
                                           **self.phandle.ri_cfg.zero_shot_cfg.lm_generation_cfg.__dict__)

    def debug_generate_batch(self, sequences: List, max_new_tokens: Optional[int] = None) -> List:
        fake_input_ids = self.phandle.trainer.datamodule.tokenizer.batch_encode_plus(sequences)
        fake_input_ids = self.phandle.trainer.datamodule.data_collator(fake_input_ids)
        fake_input_ids = fake_input_ids.to("cuda:0")
        outputs = self._debug_generate(fake_input_ids['input_ids'], max_new_tokens)
        answers = self.phandle.trainer.datamodule.tokenizer.batch_decode(outputs['sequences'],
                                                                         skip_special_tokens=False,
                                                                         clean_up_tokenization_spaces=True)
        return answers

    def debug_generate_serial(self, sequences: List, max_new_tokens: Optional[int] = None) -> List:
        answers = []
        for seq in sequences:
            fake_input_ids = self.phandle.trainer.datamodule.tokenizer.encode(seq)
            fake_input_ids = torch.tensor(fake_input_ids).to("cuda:0")
            fake_input_ids = fake_input_ids.unsqueeze(0)
            output = self._debug_generate(fake_input_ids, max_new_tokens)
            answers.append(self.phandle.trainer.datamodule.tokenizer.decode(output['sequences'].squeeze(),
                                                                            skip_special_tokens=False,
                                                                            clean_up_tokenization_spaces=True))
        return answers
