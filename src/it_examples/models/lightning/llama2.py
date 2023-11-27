from typing import Optional, List

from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers.tokenization_utils_base import BatchEncoding
import evaluate

from interpretune.base.base_lightning_modules import ITLightningModule, ITLightningDataModule
from it_examples.models.core.llama2 import Llama2BoolRTEDataModule


class Llama2BoolRTELightningDataModule(Llama2BoolRTEDataModule, ITLightningDataModule):
    ...


class Llama2ITLightningModule(ITLightningModule):

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
