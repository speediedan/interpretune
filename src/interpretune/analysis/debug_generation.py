from typing import Optional, List, Any, Dict, Tuple, Union
from dataclasses import dataclass, field

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np

from interpretune.config.shared import ITSerializableCfg


@dataclass(kw_only=True)
class DebugLMConfig(ITSerializableCfg):
    enabled: bool = False
    debug_raw_preds: np.ndarray = field(default_factory=lambda: np.array([]))
    debug_raw_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    debug_raw_sequences: Optional[List[str]] = None
    raw_debug_sequences: List = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.raw_debug_sequences) == 0 and self.enabled:
            self.raw_debug_sequences = ['What is the color of a banana?', 'List the first 5 letters in the alphabet.',
                                        'How many days in a week?', 'How old is Barack Obama?']


class DebugGeneration:
    """Give user-provided callbacks with the ability to connect to another user-provided callback.

    This resolution logic is provided in order to avoid callback-dependent trainer attributes (e.g.
    trainer.finetuningscheduler_callback)
    """
    # TODO:
    # - after basic functionality set, get device from config instead of hardcoding
    # - may make sense to add some additional debugging methods that parse and analyze all of the generated outputs
    #   including `output_attentions` and `output_hidden_states` etc.

    def __init__(
        self,
    ) -> None:
        """Arguments."""
        super().__init__()
        self.phandle = None

    def connect(self, obj_ref: Any) -> None:
        self.phandle = obj_ref

    def debug_sequences(self, sequences: Optional[Union[List, str]] = None) -> List:
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
        sequences = sequences or self.phandle.it_cfg.debug_lm_cfg.raw_debug_sequences
        if not isinstance(sequences, list):
            sequences = [sequences]
        return [f"{ex.strip()}" for ex in sequences]

    def _debug_generate(self, inputs: List|torch.Tensor, max_new_tokens_override: Optional[int] = None,
                        gen_config_override: Optional[Dict] = None) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_
            max_new_tokens_override (Optional[int], optional): _description_. Defaults to None.
            gen_config_override (Optional[Dict], optional): _description_. Defaults to None.

        Usage:

        ```python
        self.lm_debug.debug_generate_batch(['my sequence potentially with chat specific tags', 'another sequence'])
        # to narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.lm_debug.debug_generate_serial(['my sequence potentially with chat specific tags', 'another sequence'])
        # to override the defaults (both questions and current `max_new_tokens` config)
        self.lm_debug.debug_generate_batch(self.lm_debug.sys_inst_debug_sequences([
            'What is the color of a cloudless sky?', 'How many days are in a year?']), max_new_tokens=25)
        ```
        """
        # note we're not using a context manager here, keeping our new override for subsequent debugging convenience
        if max_new_tokens_override:
            self.phandle.it_cfg.zero_shot_cfg.lm_generation_cfg.max_new_tokens = max_new_tokens_override
        gen_config_dict = self.phandle.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__
        if gen_config_override:
            gen_config_dict.update(gen_config_override)
        return self.phandle.model.generate(input_ids=inputs,
                                           pad_token_id=self.phandle.trainer.datamodule.tokenizer.pad_token_id,
                                           **gen_config_dict)

    def perplexity_on_sample(self, corpus: Optional[Dataset] = None) -> float:
        if not corpus:
            corpus = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # max_length_sample_index = np.argmax(np.array([len(i) for i in corpus["text"]]))
        encoded_corpus = self.phandle.trainer.datamodule.tokenizer("\n\n".join(corpus["text"]), return_tensors="pt")
        return self.naive_perplexity(encoded_corpus)

    def top1_token_accuracy_on_sample(self, sample: str) -> Tuple[float, List[str]]:
        sample_input_ids = self.phandle.trainer.datamodule.tokenizer.encode(sample)
        sample_input_ids = torch.tensor(sample_input_ids).to("cuda:0")
        sample_input_ids = sample_input_ids.unsqueeze(0)
        with torch.no_grad():
            output = self.phandle.trainer.model(input_ids=sample_input_ids)
        logits = output['logits']
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = sample_input_ids.squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()
        correct_tokens = self.phandle.trainer.datamodule.tokenizer.batch_decode(prediction[prediction == true_tokens])
        return num_correct/len(true_tokens), correct_tokens

    def naive_perplexity(self, encoded_corpus, stride: int = 512) -> float:
        max_length = self.phandle.trainer.model.model.config.n_positions
        stride = stride
        seq_len = encoded_corpus.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encoded_corpus.input_ids[:, begin_loc:end_loc].to("cuda:0")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.phandle.trainer.model(input_ids=input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl

    def debug_generate_batch(self, sequences: List, max_new_tokens: Optional[int] = None,
                             gen_config_override: Optional[Dict] = None) -> Tuple[List, List]:
        fake_input_ids = self.phandle.trainer.datamodule.tokenizer.batch_encode_plus(sequences)
        fake_input_ids = self.phandle.trainer.datamodule.data_collator(fake_input_ids)
        fake_input_ids = fake_input_ids.to("cuda:0")
        outputs = self._debug_generate(fake_input_ids['input_ids'], max_new_tokens, gen_config_override)
        answers = self.phandle.trainer.datamodule.tokenizer.batch_decode(outputs['sequences'],
                                                                         skip_special_tokens=False,
                                                                         clean_up_tokenization_spaces=True)
        return answers, outputs

    def debug_generate_serial(self, sequences: List, max_new_tokens: Optional[int] = None,
                              gen_config_override: Optional[Dict] = None) -> Tuple[List, List]:
        answers = []
        full_outputs = []
        for seq in sequences:
            fake_input_ids = self.phandle.trainer.datamodule.tokenizer.encode(seq)
            fake_input_ids = torch.tensor(fake_input_ids).to("cuda:0")
            fake_input_ids = fake_input_ids.unsqueeze(0)
            output = self._debug_generate(fake_input_ids, max_new_tokens, gen_config_override)
            sequences = output['sequences'].unbind()
            for seq in sequences:  # in case num_return_sequences > 1
                answers.append(self.phandle.trainer.datamodule.tokenizer.decode(seq,
                                                                                skip_special_tokens=False,
                                                                                clean_up_tokenization_spaces=True))
            full_outputs.append(output)
        return answers, full_outputs
