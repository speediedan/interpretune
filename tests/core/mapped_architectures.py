"""The architectures Interpretune supports, and a tiny meta-device model of each to bind its component map to.

The maps themselves are venngram's, read from its registry (and, for ``Qwen3_5ForCausalLM`` alone, from the fixture
``tests/conftest.py`` registers until the registry serves that class). A supported architecture is one the suite checks
end to end: its map binds to the transformers class here, and agrees with TransformerLens' adapter for it.
"""

from __future__ import annotations

import pytest

#: Whether the final norm scales by ``(1 + weight)``, per supported architecture. Its keys ARE the supported set: an
#: architecture enters by gaining an entry here and a builder in :func:`tiny_model`.
EXPECTED_RMSNORM_OFFSET = {
    "GPT2LMHeadModel": False,
    "LlamaForCausalLM": False,
    "Gemma2ForCausalLM": True,
    "Gemma3ForCausalLM": True,
    "Gemma3ForConditionalGeneration": True,
    "Qwen3_5ForCausalLM": True,
    "Qwen3_5ForConditionalGeneration": True,
}

SUPPORTED_ARCHITECTURES = tuple(sorted(EXPECTED_RMSNORM_OFFSET))

#: Supported architectures the registry at the CI warm generation does not serve yet, each with where its map comes
#: from meanwhile. Only until the registry serves the class (#1026).
NOT_YET_IN_THE_REGISTRY = {
    "Qwen3_5ForCausalLM": "tests/fixtures/component_maps/qwen3_5_text.yaml, registered by tests/conftest.py",
}

#: Qwen3.5's full-attention layers at the default `layer_types` of the 32-layer tiny model (Qwen3.5-4B's own pattern).
QWEN3_5_FULL_ATTENTION = (3, 7, 11, 15, 19, 23, 27, 31)


def tiny_model(architecture: str):
    """The transformers class for a supported architecture, instantiated tiny on the meta device."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    small = dict(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=2, vocab_size=64)
    qwen3_5_text = dict(
        num_hidden_layers=32,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        **small,
    )
    builders = {
        "GPT2LMHeadModel": lambda: tr.GPT2LMHeadModel(tr.GPT2Config(n_layer=3, n_embd=16, n_head=2, vocab_size=64)),
        "LlamaForCausalLM": lambda: tr.LlamaForCausalLM(tr.LlamaConfig(num_hidden_layers=3, **small)),
        "Gemma2ForCausalLM": lambda: tr.Gemma2ForCausalLM(tr.Gemma2Config(num_hidden_layers=3, head_dim=8, **small)),
        "Gemma3ForCausalLM": lambda: tr.Gemma3ForCausalLM(
            tr.Gemma3TextConfig(num_hidden_layers=3, head_dim=8, **small)
        ),
        "Gemma3ForConditionalGeneration": lambda: tr.Gemma3ForConditionalGeneration(
            tr.Gemma3Config(
                text_config=dict(num_hidden_layers=2, head_dim=8, **small),
                vision_config=dict(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    image_size=16,
                    patch_size=8,
                ),
            )
        ),
        # 32 layers so the default `layer_types` is Qwen3.5-4B's own (full attention at 3, 7, ..., 31): the maps'
        # optional attention rows are bound by probing these layers
        "Qwen3_5ForCausalLM": lambda: tr.Qwen3_5ForCausalLM(tr.Qwen3_5TextConfig(**qwen3_5_text)),
        "Qwen3_5ForConditionalGeneration": lambda: tr.Qwen3_5ForConditionalGeneration(
            tr.Qwen3_5Config(
                text_config=qwen3_5_text,
                vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16),
            )
        ),
    }
    if architecture not in builders:
        pytest.fail(
            f"{architecture} is supported but has no tiny builder here: entering the supported set means adding one, "
            "so its map is bound to its class on every run (docs/activation_point_vocabulary.md, 'Entering and "
            "leaving the supported set')."
        )
    with torch.device("meta"):
        return builders[architecture]()


def bound_map(architecture: str):
    """The architecture's component map bound to its tiny model."""
    from interpretune.analysis.points import bind_component_map, component_map_for

    return bind_component_map(component_map_for(architecture), tiny_model(architecture))
