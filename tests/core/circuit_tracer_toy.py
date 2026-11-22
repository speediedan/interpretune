"""A tiny Gemma 2 circuit-tracer replacement model with random transcoders, for CPU tests of graph and intervention
semantics that need a real model but not a trained one."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

TOY_PROMPT = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])


class BaseModelBackendStandIn:
    """The model backend a circuit-tracer composition carries beneath its analysis backend, for a fake module that
    exercises only the analysis backend: it declares the surfaces a full model backend provides (``GRADIENTS``,
    ``LATENT_MODELS``) and implements nothing, so a test that reached it would fail rather than pass on a stand-in."""

    def __init__(self) -> None:
        from interpretune.analysis.backends import Capability, LatentModelSupport

        self.capabilities = frozenset({Capability.GRADIENTS, Capability.LATENT_MODELS})
        self.requires = frozenset()
        self.latent_model_support = LatentModelSupport()


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
    for transcoder in transcoders.values():
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


def tiny_gemma2_backend_twins(n_layers: int = 2, seed: int = 933):
    """The same tiny Gemma 2 and transcoders on circuit-tracer's nnsight and interp_engine backends, as ``(nnsight,
    interp_engine)``, for checking that a technique gives one answer whichever backend carries it.

    The vocabulary is the tokenizer's, because interp_engine probes each model at load with its tokenizer's BOS token,
    which the 16-token toy vocabulary above cannot embed.
    """
    import copy

    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
    from circuit_tracer.transcoder.activation_functions import JumpReLU
    from transformers import Gemma2Config

    cfg = Gemma2Config(
        architectures=["Gemma2ForCausalLM"],
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=50257,
        query_pre_attn_scalar=4,
        final_logit_softcapping=None,
        torch_dtype="float32",
    )
    cfg._name_or_path = "openai-community/gpt2"
    torch.manual_seed(seed)
    transcoders = {
        layer: SingleLayerTranscoder(cfg.hidden_size, cfg.hidden_size * 4, JumpReLU(torch.tensor(0.0), 0.1), layer)
        for layer in range(n_layers)
    }
    for transcoder in transcoders.values():
        for param in transcoder.parameters():
            torch.nn.init.uniform_(param, a=-1, b=1)
        torch.nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)
    transcoder_set = TranscoderSet(transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
    nnsight = ReplacementModel.from_config(copy.deepcopy(cfg), copy.deepcopy(transcoder_set), backend="nnsight")
    interp_engine = ReplacementModel.from_config(copy.deepcopy(cfg), transcoder_set, backend="interp_engine")
    for param in interp_engine.hf_model.parameters():
        torch.nn.init.uniform_(param, a=-1, b=1)
    # strict: every causal-LM weight is copied. nnsight registers its transcoders under the causal LM, and they are
    # already identical copies of interp_engine's.
    own_transcoders = {k: v for k, v in nnsight._module.state_dict().items() if k.startswith("transcoders.")}
    nnsight._module.load_state_dict({**own_transcoders, **interp_engine.hf_model.state_dict()})
    # one precision on both sides, or the comparison measures rounding rather than the backends
    dtypes = {
        next(nnsight._module.parameters()).dtype,
        next(nnsight.transcoders._module.parameters()).dtype,
        next(interp_engine.hf_model.parameters()).dtype,
        next(interp_engine.transcoders.parameters()).dtype,
    }
    assert dtypes == {torch.float32}, dtypes
    return nnsight, interp_engine


def tiny_gemma2_tl_twin(nnsight_model):
    """Circuit-tracer's TransformerLens replacement model on the same weights and transcoders as ``nnsight_model``.

    The bridge wraps its own copy of the causal LM, built from the nnsight model's config and loaded with every weight
    of it, so the two backends answer one model.
    """
    import copy

    from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import AutoModelForCausalLM

    config = copy.deepcopy(nnsight_model._module.config)
    config._attn_implementation = "eager"
    hf_model = AutoModelForCausalLM.from_config(config).eval()
    weights = {k: v for k, v in nnsight_model._module.state_dict().items() if not k.startswith("transcoders.")}
    hf_model.load_state_dict(weights)  # strict: every causal-LM weight is copied
    bridge = TransformerBridge.boot_transformers(
        config._name_or_path, hf_model=hf_model, tokenizer=nnsight_model.tokenizer, device="cpu", dtype=torch.float32
    )
    return TransformerLensReplacementModel.from_bridge(bridge, copy.deepcopy(nnsight_model.transcoders._module))
