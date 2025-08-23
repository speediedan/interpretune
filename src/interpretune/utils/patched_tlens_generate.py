"""Hooked Transformer.

The Hooked Transformer is the core part of TransformerLens.

In common PyTorch model implementations (e.g. ones from HuggingFace) it's fairly easy to extract model weights, but much
harder to extract activations. TransformerLens aims to simplify this task by attaching hooks to every notable activation
within the model. This enables the inspection and/or alteration of activations in individual components like attention
heads and MLP layers, facilitating a deeper understanding of the internal workings of transformers like GPT-2.
"""

from typing import NamedTuple, Optional, Union

import torch
import tqdm.auto as tqdm
from jaxtyping import Float, Int
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
from transformer_lens.utils import USE_DEFAULT_VALUE

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class Output(NamedTuple):
    """Output Named Tuple.

    Named tuple object for if we want to output both logits and loss.
    """

    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss


class SampledOutput(NamedTuple):
    """Sampled Output Named Tuple.

    Named tuple object for if we want to output both logits and tokens.
    """

    tokens: Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]
    logits: Float[torch.Tensor, "batch pos d_vocab"]


@torch.inference_mode()
def generate(
    self,
    input: Union[str, Float[torch.Tensor, "batch pos"]] = "",
    max_new_tokens: int = 10,
    stop_at_eos: bool = True,
    eos_token_id: Optional[int] = None,
    do_sample: bool = True,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    freq_penalty: float = 0.0,
    use_past_kv_cache: bool = True,
    prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
    padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
    output_logits: Optional[bool] = None,
    return_type: Optional[str] = "input",
    verbose: bool = True,
    # **kwargs,  # TODO: consider allowing unsupported kwargs to pass through
) -> Union[SampledOutput, Int[torch.Tensor, "batch pos_plus_new_tokens"], str]:
    """Sample Tokens from the Model.

    Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

    To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
    (by producing an EOT token), we keep running the model on the entire batch, but throw away
    the output for a finished sequence and just keep adding EOTs to pad.

    This supports entering a single string, but not a list of strings - if the strings don't
    tokenize to exactly the same length, this gets messy. If that functionality is needed,
    convert them to a batch of tokens and input that instead.

    Args:
        input (Union[str, Int[torch.Tensor, "batch pos"])]): Either a batch of tokens ([batch,
            pos]) or a text string (this will be converted to a batch of tokens with batch size
            1).
        max_new_tokens (int): Maximum number of tokens to generate.
        stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
        eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
            of sentence. If None, use the tokenizer's eos_token_id - required if using
            stop_at_eos. It's also possible to provide a list of token IDs (not just the
            eos_token_id), in which case the generation will stop when any of them are output
            (useful e.g. for stable_lm).
        do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
            greedy search (take the max logit each time).
        top_k (int): Number of tokens to sample from. If None, sample from all tokens.
        top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
            we take the top tokens with cumulative probability >= top_p.
        temperature (float): Temperature for sampling. Higher values will make the model more
            random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
            sampling from a uniform distribution).
        freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
            tokens. Higher values will make the model more random.
        use_past_kv_cache (bool): If True, create and use cache to speed up generation.
        prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
            the BOS token to the input (applicable when input is a string). Defaults to None,
            implying usage of self.cfg.default_prepend_bos (default is True unless specified
            otherwise). Pass True or False to override the default.
        padding_side (Union[Literal["left", "right"], None], optional): Overrides
            self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
            strings of different lengths.
        return_type (Optional[str]): The type of the output to return - either a string (str),
            a tensor of tokens (tensor) or whatever the format of the input was (input).
            # TODO: update the output_logits description
        output_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        verbose (bool): If True, show tqdm progress bars for generation.

    Returns:
        outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
            (by default returns same type as input).
    """
    # port a subset of HF generate/sample's (generation/utils.py) functionality to to generate()
    # TODO: add permalink to relevant code ported
    out_logits = () if output_logits else None
    with utils.LocallyOverridenDefaults(self, prepend_bos=prepend_bos, padding_side=padding_side):
        if type(input) == str:  # noqa: E721
            # If text, convert to tokens (batch_size=1)
            assert self.tokenizer is not None, "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input

        if return_type == "input":
            if type(input) == str:  # noqa: E721
                return_type = "str"
            else:
                return_type = "tensor"

        assert isinstance(tokens, torch.Tensor)
        batch_size, ctx_length = tokens.shape
        device = devices.get_device_for_block_index(0, self.cfg)
        tokens = tokens.to(device)
        if use_past_kv_cache:
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(self.cfg, self.cfg.device, batch_size)
        else:
            past_kv_cache = None

        stop_tokens = []
        eos_token_for_padding = 0
        if stop_at_eos:
            tokenizer_has_eos_token = self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            if eos_token_id is None:
                assert tokenizer_has_eos_token, (
                    "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
                )

                eos_token_id = self.tokenizer.eos_token_id

            if isinstance(eos_token_id, int):
                stop_tokens = [eos_token_id]
                eos_token_for_padding = eos_token_id
            else:
                # eos_token_id is a Sequence (e.g. list or tuple)
                stop_tokens = eos_token_id
                eos_token_for_padding = self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]

        # An array to track which sequences in the batch have finished.
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

        # Currently nothing in HookedTransformer changes with eval, but this is here in case
        # that changes in the future.
        self.eval()
        for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
            # While generating, we keep generating logits, throw away all but the final logits,
            # and then use those logits to sample from the distribution We keep adding the
            # sampled tokens to the end of tokens.
            if use_past_kv_cache:
                # We just take the final tokens, as a [batch, 1] tensor
                if index > 0:
                    logits = self.forward(
                        tokens[:, -1:],
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_cache=past_kv_cache,
                    )
                else:
                    logits = self.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        past_kv_cache=past_kv_cache,
                    )
            else:
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                # the cache.
                logits = self.forward(
                    tokens,
                    return_type="logits",
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                )
            final_logits = logits[:, -1, :]
            if output_logits:
                out_logits += (final_logits,)

            if do_sample:
                sampled_tokens = utils.sample_logits(
                    final_logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    freq_penalty=freq_penalty,
                    tokens=tokens,
                ).to(devices.get_device_for_block_index(0, self.cfg))
            else:
                sampled_tokens = final_logits.argmax(-1).to(devices.get_device_for_block_index(0, self.cfg))

            if stop_at_eos:
                # For all unfinished sequences, add on the next token. If a sequence was
                # finished, throw away the generated token and add eos_token_for_padding
                # instead.
                sampled_tokens[finished_sequences] = eos_token_for_padding
                finished_sequences.logical_or_(torch.isin(sampled_tokens, torch.tensor(stop_tokens).to(device)))

            tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

            if stop_at_eos and finished_sequences.all():
                break

        if return_type == "str":
            if self.cfg.default_prepend_bos:
                # If we prepended a BOS token, remove it when returning output.
                tokens = self.tokenizer.decode(tokens[0, 1:])
            else:
                tokens = self.tokenizer.decode(tokens[0])
        if output_logits:
            return SampledOutput(tokens, torch.stack(out_logits, dim=1))
        else:
            return tokens
