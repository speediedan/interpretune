"""A tiny Gemma 2 circuit-tracer replacement model with random transcoders, for CPU tests of graph and intervention
semantics that need a real model but not a trained one."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

TOY_PROMPT = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])


def tiny_gemma2_replacement_model(n_layers: int = 3, seed: int = 670):
    """An ``n_layers``-deep, 8-wide Gemma 2 replacement model; builds and runs on CPU in seconds."""
    from circuit_tracer import ReplacementModel
    from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
    from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
    from circuit_tracer.transcoder.activation_functions import JumpReLU
    from transformers import Gemma2Config

    cfg = Gemma2Config(
        architectures=["Gemma2ForCausalLM"],  # circuit-tracer's nnsight mapping keys on the architecture
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=16,
        query_pre_attn_scalar=4,
        final_logit_softcapping=None,
        torch_dtype="float32",
    )
    cfg._name_or_path = "openai-community/gpt2"  # the tokenizer from_config loads; warmed for offline CI
    torch.manual_seed(seed)
    transcoders = {
        layer: SingleLayerTranscoder(cfg.hidden_size, cfg.hidden_size * 4, JumpReLU(torch.tensor(0.0), 0.1), layer)
        for layer in range(n_layers)
    }
    for transcoder in transcoders.values():
        for param in transcoder.parameters():
            torch.nn.init.uniform_(param, a=-1, b=1)
    transcoder_set = TranscoderSet(transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
    model = ReplacementModel.from_config(cfg, transcoder_set, backend="nnsight")
    for param in model.parameters():
        torch.nn.init.uniform_(param, a=-1, b=1)
    for transcoder in model.transcoders[0]:  # type: ignore[index]
        torch.nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)
    assert isinstance(model, NNSightReplacementModel)
    return model


@contextmanager
def only_token_zero_special(model) -> Iterator[None]:
    """The gpt2 tokenizer's special ids fall outside the toy vocabulary; treat id 0 as the only special token."""
    tokenizer_class = type(model.tokenizer)
    original = tokenizer_class.all_special_ids
    tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore[assignment]
    try:
        yield
    finally:
        tokenizer_class.all_special_ids = original  # type: ignore[assignment]
