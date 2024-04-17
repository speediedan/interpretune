from typing import Optional, List, Any, Dict, Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy

import torch
from datasets import load_dataset, Dataset
import numpy as np
from torch.nn import CrossEntropyLoss

from interpretune.base.config.shared import ITSerializableCfg
from interpretune.utils.tokenization import _sanitize_input_name


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
    # - note availability of HF tokenizer methods are assumed for the moment, need to add to contract
    # - may make sense to add some additional debugging methods that parse and analyze all of the generated outputs
    #   including `output_attentions` and `output_hidden_states` etc.
    DEFAULT_OUTPUT_ATTRS = ("sequences", "tokens")
    DEFAULT_MODEL_CONFIG_ATTRS = ("cfg", "config")
    DEFAULT_DECODE_KWARGS = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)

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
        # one can use this method to probe non-chat fine-tuned models (just the raw sequences, no SYS
        # or INST metadata)
        self.debug_lm.debug_generate_batch(self.debug_lm.debug_sequences('My single custom sequence'),
        gen_config_override={"max_new_tokens": 25})
        ```
        """
        sequences = sequences or self.phandle.it_cfg.debug_lm_cfg.raw_debug_sequences
        if not isinstance(sequences, list):
            sequences = [sequences]
        return [f"{ex.strip()}" for ex in sequences]

    # TODO: add check to validate compatible model to gracefully handle use with incompatible models
    # debug helper function for models that use sys and inst metadata (e.g. llama2)
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
        self.debug_lm.debug_generate_batch(self.sys_inst_debug_sequences())
        # to narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.debug_lm.debug_generate_serial(self.sys_inst_debug_sequences())
        # to override the defaults (both questions and current `max_new_tokens` config)
        self.debug_lm.debug_generate_batch(self.sys_inst_debug_sequences([
            'What is the color of a cloudless sky?', 'How many days are in a year?']),
            gen_config_override={"max_new_tokens": 25})
        ```
        """
        sequences = sequences or self.it_cfg.debug_lm_cfg.raw_debug_sequences
        return [self.phandle.datamodule.tokenizer.bos_token + \
            self.phandle.datamodule.itdm_cfg.prompt_cfg.SYS_PREFIX + \
                f"{ex.strip()} {self.phandle.datamodule.itdm_cfg.prompt_cfg.E_INST}" \
                for ex in sequences]

    def _debug_generate(self, inputs: List|torch.Tensor, gen_config_override: Optional[Dict] = None) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_
            gen_config_override (Optional[Dict], optional): _description_. Defaults to None.

        Usage:

        ```python
        self.debug_lm.debug_generate_batch(['my sequence potentially with chat specific tags', 'another sequence'])
        # to narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.debug_lm.debug_generate_serial(['my sequence potentially with chat specific tags', 'another sequence'])
        # to override the defaults (both questions and current generation config with a different `max_new_tokens`)
        self.debug_lm.debug_generate_batch(self.debug_lm.sys_inst_debug_sequences([
            'What is the color of a cloudless sky?', 'How many days are in a year?']),
            gen_config_override={"max_new_tokens": 25})
        ```
        """
        # note we're not using a context manager here, keeping our new override for subsequent debugging convenience
        gen_config_dict = self.phandle.it_cfg.zero_shot_cfg.lm_generation_cfg.__dict__
        if gen_config_override:
            gen_config_dict.update(gen_config_override)
        return self.phandle.it_generate(inputs,
                                          pad_token_id=self.phandle.datamodule.tokenizer.pad_token_id,
                                          **gen_config_dict)

    def perplexity_on_sample(self, corpus: Optional[Dataset|Dict] = None, stride: Optional[int] = None,
                             limit_chars: Optional[int] = None) -> float:
        if not corpus:
            corpus = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        corpus_raw = "\n\n".join(corpus["text"])
        corpus_max_idx = limit_chars or len(corpus_raw)
        encoded_corpus = self.phandle.datamodule.tokenizer(corpus_raw[:corpus_max_idx], return_tensors="pt")
        encoded_corpus = _sanitize_input_name(self.model_input_names, encoded_corpus)
        return self.naive_perplexity(encoded_corpus, stride=stride)

    def top1_token_accuracy_on_sample(self, sample: str) -> Tuple[float, List[str]]:
        sample_input_ids = self.phandle.datamodule.tokenizer.encode(sample)
        sample_input_ids = torch.tensor(sample_input_ids).to(self.phandle.device)
        sample_input_ids = sample_input_ids.unsqueeze(0)
        with torch.no_grad():
            logits = self.phandle.model(sample_input_ids)
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = sample_input_ids.squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()
        correct_tokens = self.phandle.datamodule.tokenizer.batch_decode(prediction[prediction == true_tokens])
        return num_correct/len(true_tokens), correct_tokens

    def naive_perplexity(self, encoded_corpus, stride: int = 512) -> float:
        max_length = self.phandle.datamodule.tokenizer.model_max_length
        corpus_ids = getattr(encoded_corpus, self.model_input_names[0])
        seq_len = corpus_ids.size(1)
        stride = min(stride, seq_len)
        nlls = []
        prev_end_loc = 0
        loss_fn = CrossEntropyLoss()
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            inputs = corpus_ids[:, begin_loc:end_loc].to(self.phandle.device)
            target_ids = inputs.clone()
            target_ids[:, :-trg_len] = -100
            with torch.inference_mode():
                output_logits = self.phandle.model.forward(inputs)
                shift_logits = output_logits[..., :-1, :].contiguous()
                shift_target_ids = target_ids[..., 1:].contiguous()
                preds = shift_logits.view(-1, shift_logits.size(-1))
                labels = shift_target_ids.view(-1)
                neg_log_likelihood = loss_fn(preds, labels)
            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl

    def sanitize_gen_output(self, outputs: Any, gen_output_attr: Optional[str] = None,
                              decode_cfg_override: Optional[Dict] = None) -> Tuple[Any, Dict]:
        decode_target = self.sanitize_model_output(outputs, gen_output_attr)
        decode_kwargs = deepcopy(self.DEFAULT_DECODE_KWARGS)
        if decode_cfg_override:
            decode_kwargs.update(decode_cfg_override)
        return decode_target, decode_kwargs

    def sanitize_model_output(self, output: Any, gen_output_attr: Optional[str] = None) -> List[str]:
        if gen_output_attr:
            return getattr(output, gen_output_attr)
        for output_attr in self.DEFAULT_OUTPUT_ATTRS:
            if hasattr(type(output), output_attr):
                return getattr(output, output_attr)
        raise ValueError(f"No compatible default output attribute found for type: {type(output)}, if the"
                            " generate method attached to your model is not returning a supported output attribute"
                            f" ({self.DEFAULT_OUTPUT_ATTRS}) please provide a manual `gen_output_attr` argument to this"
                            " debug_generate method.")

    @property
    def model_input_names(self) -> List[str]:
        return self.phandle.datamodule.tokenizer.model_input_names

    def debug_generate_batch(self, sequences: List,
                             gen_output_attr: Optional[str] = None,
                             gen_config_override: Optional[Dict] = None,
                             decode_cfg_override: Optional[Dict] = None) -> Tuple[List, List]:
        test_input_ids = self.phandle.datamodule.tokenizer.batch_encode_plus(sequences)
        test_input_ids = _sanitize_input_name(self.model_input_names, test_input_ids)
        test_input_ids = self.phandle.datamodule.data_collator(test_input_ids)
        test_input_ids = test_input_ids.to(self.phandle.device)
        outputs = self._debug_generate(test_input_ids[self.model_input_names[0]], gen_config_override)
        decode_target, decode_kwargs = self.sanitize_gen_output(outputs, gen_output_attr, decode_cfg_override)
        answers = self.phandle.datamodule.tokenizer.batch_decode(decode_target, **decode_kwargs)
        return answers, outputs

    def debug_generate_serial(self, sequences: List, gen_output_attr: Optional[str] = None,
                              gen_config_override: Optional[Dict] = None,
                              decode_cfg_override: Optional[Dict] = None) -> Tuple[List, List]:
        answers = []
        full_outputs = []
        for seq in sequences:
            test_input_ids = self.phandle.datamodule.tokenizer.encode(seq)
            test_input_ids = torch.tensor(test_input_ids).to(self.phandle.device)
            test_input_ids = test_input_ids.unsqueeze(0)
            output = self._debug_generate(test_input_ids, gen_config_override)
            decode_target, decode_kwargs = self.sanitize_gen_output(output, gen_output_attr, decode_cfg_override)
            sequences = decode_target.unbind()
            decode_kwargs = deepcopy(self.DEFAULT_DECODE_KWARGS)
            if decode_cfg_override:
                decode_kwargs.update(decode_cfg_override)
            for seq in sequences:  # in case num_return_sequences > 1
                answers.append(self.phandle.datamodule.tokenizer.decode(seq, **decode_kwargs))
            full_outputs.append(output)
        return answers, full_outputs
