from typing import Optional, List, Any, Dict, Tuple, Union, cast
from dataclasses import dataclass, field
from copy import deepcopy

import torch
from datasets import load_dataset, Dataset
import numpy as np
from torch.nn import CrossEntropyLoss

from interpretune.config import ITSerializableCfg
from interpretune.utils import rank_zero_warn, sanitize_input_name, DEFAULT_DECODE_KWARGS
from interpretune.protocol import ITModuleGenDebuggable


@dataclass(kw_only=True)
class DebugLMConfig(ITSerializableCfg):
    enabled: bool = False
    debug_raw_preds: Optional[np.ndarray] = None  # field(default_factory=lambda: np.array([]))
    debug_raw_labels: Optional[np.ndarray] = None  # field(default_factory=lambda: np.array([]))
    debug_raw_sequences: Optional[List[str]] = None
    raw_debug_sequences: List = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.raw_debug_sequences) == 0 and self.enabled:
            self.raw_debug_sequences = [
                "What is the color of a banana?",
                "List the first 5 letters in the alphabet.",
                "How many days in a week?",
                "How old is Barack Obama?",
            ]


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
    phandle: Optional[ITModuleGenDebuggable]

    def __init__(
        self,
    ) -> None:
        """Arguments."""
        super().__init__()
        self.phandle = None

    def connect(self, obj_ref: ITModuleGenDebuggable) -> None:
        self.phandle = obj_ref

    def _check_phandle(self) -> ITModuleGenDebuggable:
        """Helper: raise a RuntimeError if phandle isn't connected.

        This centralizes the check so all call sites have consistent behavior.
        """
        if self.phandle is None:
            raise RuntimeError("Extension not connected to module - call connect() first")
        # Help static type checkers: cast to the protocol we've defined
        assert isinstance(self.phandle, ITModuleGenDebuggable)
        return cast(ITModuleGenDebuggable, self.phandle)

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
        if sequences is None:
            ph = self._check_phandle()
            sequences = ph.it_cfg.debug_lm_cfg.raw_debug_sequences or []
        if not isinstance(sequences, list):
            sequences = [sequences]
        return [f"{ex.strip()}" for ex in sequences]

    def chat_debug_sequences(self, sequences: Optional[List] = None, format: Optional[str] = None) -> List:
        """_summary_

        Args:
            sequences (Optional[List], optional): _description_. Defaults to None.

        Returns:
            List: _description_

        Usage:

        ```python
        # for example, using the llama3 chat format, you want to have input tokenized with sys and inst metadata
        # to do so with some reasonable default questions as a sanity check and in batch mode:
        self.debug_lm.debug_generate_batch(self.debug_lm.chat_debug_sequences()
        # to narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.debug_lm.debug_generate_serial(self.debug_lm.chat_debug_sequences())
        # to override the defaults (both questions and current `max_new_tokens` config)
        # you can also specify a specific model variant pattern for a given prompt config, e.g. `format="llama3-chat"`
        self.debug_lm.debug_generate_batch(self.debug_lm.chat_debug_sequences(format='llama3-chat', sequences=[
            'What is the color of a cloudless sky?', 'How many days are in a year?']),
            gen_config_override={"max_new_tokens": 25})
        ```
        """
        try:
            ph = self._check_phandle()
            if sequences is None:
                sequences = ph.it_cfg.debug_lm_cfg.raw_debug_sequences
            if format is None:
                format = ph.datamodule.itdm_cfg.cust_tokenization_pattern
            assert isinstance(sequences, list)
            return [ph.datamodule.itdm_cfg.prompt_cfg.model_chat_template_fn(ex, format) for ex in sequences]
        except Exception as e:
            rank_zero_warn(
                f"Failed to generate chat debug sequences. Exception: {e}. "
                "Returning the stripped sequences but without the corresponding chat format metadata."
            )
            sequences = sequences or []  # sequences could still be None at Exception
            return [f"{ex.strip()}" for ex in sequences]

    def _debug_generate(
        self,
        inputs: List | torch.Tensor,
        gen_kwargs_override: Optional[Dict] = None,
        gen_config_override: Optional[Dict] = None,
    ) -> Any:
        """_summary_

        Args:
            inputs (_type_): _description_
            gen_kwargs_override (Optional[Dict], optional): _description_. Defaults to None.

        Usage:

        ```python
        self.debug_lm.debug_generate_batch(['my sequence potentially with chat specific tags', 'another sequence'])
        # to narrow the problem space, using serial inference (non-batch mode) for a list of strings can be useful
        self.debug_lm.debug_generate_serial(['my sequence potentially with chat specific tags', 'another sequence'])
        # to override the defaults (both questions and current generation config with a different `max_new_tokens`)
        self.debug_lm.debug_generate_batch(self.debug_lm.chat_debug_sequences([
            'What is the color of a cloudless sky?', 'How many days are in a year?']),
            gen_config_override={"max_new_tokens": 25})
        ```
        """
        ph = self._check_phandle()
        # note we're not using a context manager here, keeping our new override for subsequent debugging convenience
        gen_kwargs = ph.it_cfg.generative_step_cfg.lm_generation_cfg.generate_kwargs
        if gen_kwargs_override:
            gen_kwargs.update(gen_kwargs_override)
        if gen_config_override and getattr(ph.model, "generation_config", None):
            for k, v in gen_config_override.items():
                setattr(ph.model.generation_config, k, v)
        return ph.it_generate(inputs, **gen_kwargs)

    def perplexity_on_sample(
        self,
        corpus: Optional[Dataset | Dict] = None,
        stride: Optional[int] = None,
        limit_chars: Optional[int] = None,
    ) -> torch.Tensor:
        ph = self._check_phandle()

        if not corpus:
            corpus_default = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            assert isinstance(corpus_default, Dataset)
            corpus = corpus_default
        corpus_raw = "\n\n".join(corpus["text"])
        corpus_max_idx = limit_chars or len(corpus_raw)
        encoded_corpus = ph.datamodule.tokenizer(corpus_raw[:corpus_max_idx], return_tensors="pt")
        encoded_corpus = sanitize_input_name(self.model_input_names, encoded_corpus)
        perplexity_kwargs = {"stride": stride} if stride else {}
        return self.naive_perplexity(encoded_corpus, **perplexity_kwargs)

    def top1_token_accuracy_on_sample(self, sample: str) -> Tuple[float, List[str]]:
        ph = self._check_phandle()

        sample_input_ids = ph.datamodule.tokenizer.encode(sample)
        sample_input_ids = torch.tensor(sample_input_ids).to(ph.device)
        sample_input_ids = sample_input_ids.unsqueeze(0)
        with torch.no_grad():
            logits = ph.model(sample_input_ids)
        prediction = logits.argmax(dim=-1).squeeze()[:-1]
        true_tokens = sample_input_ids.squeeze()[1:]
        num_correct = (prediction == true_tokens).sum()
        correct_tokens = ph.datamodule.tokenizer.batch_decode(prediction[prediction == true_tokens])
        return num_correct / len(true_tokens), correct_tokens

    def naive_perplexity(self, encoded_corpus, stride: int = 512) -> torch.Tensor:
        ph = self._check_phandle()

        max_length = ph.datamodule.tokenizer.model_max_length
        corpus_ids = getattr(encoded_corpus, self.model_input_names[0])
        seq_len = corpus_ids.size(1)
        stride = min(stride, seq_len)
        nlls = []
        prev_end_loc = 0
        loss_fn = CrossEntropyLoss()
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            inputs = corpus_ids[:, begin_loc:end_loc].to(ph.device)
            target_ids = inputs.clone()
            target_ids[:, :-trg_len] = -100
            with torch.inference_mode():
                output_logits = ph.model.forward(inputs)
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

    def sanitize_gen_output(
        self, outputs: Any, gen_output_attr: Optional[str] = None, decode_cfg_override: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        decode_target = self.sanitize_model_output(outputs, gen_output_attr)
        decode_kwargs = deepcopy(DEFAULT_DECODE_KWARGS)
        if decode_cfg_override:
            decode_kwargs.update(decode_cfg_override)
        return decode_target, decode_kwargs

    def sanitize_model_output(self, output: Any, gen_output_attr: Optional[str] = None) -> List[str]:
        if gen_output_attr:
            return getattr(output, gen_output_attr)
        for output_attr in self.DEFAULT_OUTPUT_ATTRS:
            if hasattr(output, output_attr):
                return getattr(output, output_attr)
        raise ValueError(
            f"No compatible default output attribute found for type: {type(output)}, if the"
            " generate method attached to your model is not returning a supported output attribute"
            f" ({self.DEFAULT_OUTPUT_ATTRS}) please provide a manual `gen_output_attr` argument to this"
            " debug_generate method."
        )

    @property
    def model_input_names(self) -> List[str]:
        ph = self._check_phandle()
        return ph.datamodule.tokenizer.model_input_names

    def debug_generate_batch(
        self,
        sequences: List,
        gen_output_attr: Optional[str] = None,
        gen_config_override: Optional[Dict] = None,
        gen_kwargs_override: Optional[Dict] = None,
        decode_cfg_override: Optional[Dict] = None,
    ) -> Tuple[List, List]:
        ph = self._check_phandle()

        test_input_ids = ph.datamodule.tokenizer.batch_encode_plus(sequences)
        test_input_ids = sanitize_input_name(self.model_input_names, test_input_ids)
        test_input_ids = ph.datamodule.data_collator(test_input_ids)
        test_input_ids = test_input_ids.to(ph.device)
        outputs = self._debug_generate(
            inputs=test_input_ids[self.model_input_names[0]],
            gen_config_override=gen_config_override,
            gen_kwargs_override=gen_kwargs_override,
        )
        decode_target, decode_kwargs = self.sanitize_gen_output(outputs, gen_output_attr, decode_cfg_override)
        answers = ph.datamodule.tokenizer.batch_decode(decode_target, **decode_kwargs)
        return answers, outputs

    def debug_generate_serial(
        self,
        sequences: List,
        gen_output_attr: Optional[str] = None,
        gen_config_override: Optional[Dict] = None,
        gen_kwargs_override: Optional[Dict] = None,
        decode_cfg_override: Optional[Dict] = None,
    ) -> Tuple[List, List]:
        ph = self._check_phandle()

        answers = []
        full_outputs = []
        for seq in sequences:
            test_input_ids = ph.datamodule.tokenizer.encode(seq)
            test_input_ids = torch.tensor(test_input_ids).to(ph.device)
            test_input_ids = test_input_ids.unsqueeze(0)
            output = self._debug_generate(
                inputs=test_input_ids, gen_config_override=gen_config_override, gen_kwargs_override=gen_kwargs_override
            )
            decode_target, decode_kwargs = self.sanitize_gen_output(output, gen_output_attr, decode_cfg_override)
            sequences = decode_target.unbind()
            decode_kwargs = deepcopy(DEFAULT_DECODE_KWARGS)
            if decode_cfg_override:
                decode_kwargs.update(decode_cfg_override)
            for seq in sequences:  # in case num_return_sequences > 1
                answers.append(ph.datamodule.tokenizer.decode(seq, **decode_kwargs))
            full_outputs.append(output)
        return answers, full_outputs
