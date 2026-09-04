"""The derived resolver reproduces the retired hand-written tables row for row, except where they were wrong.

The tables in the old ``hook_mapping.py`` were the five-architecture, 555-line encoding that #444 replaced with
component-map documents. This pins that the replacement is behaviour-preserving for every row that was right,
and enumerates the rows that changed, each with its reason, so a future edit to a map cannot silently move a
row that the old table had correct.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.backends.hook_mapping import HookNameResolver

OLD_TABLE: dict[str, dict[str, tuple[str, str]]] = {
    "GPT2LMHeadModel": {
        "attn.hook_out": ("transformer.h.{layer}.attn", "output"),
        "attn.hook_z": ("transformer.h.{layer}.attn.c_proj", "input"),
        "attn.o.hook_in": ("transformer.h.{layer}.attn.c_proj", "input"),
        "hook_attn_out": ("transformer.h.{layer}.attn", "output"),
        "hook_in": ("transformer.h.{layer}", "input"),
        "hook_mlp_out": ("transformer.h.{layer}.mlp", "output"),
        "hook_out": ("transformer.h.{layer}", "output"),
        "hook_resid_mid": ("transformer.h.{layer}.ln_2", "input"),
        "hook_resid_post": ("transformer.h.{layer}", "output"),
        "hook_resid_pre": ("transformer.h.{layer}", "input"),
        "mlp.hook_out": ("transformer.h.{layer}.mlp", "output"),
        "mlp.hook_pre": ("transformer.h.{layer}.mlp", "input"),
        "unembed.hook_in": ("lm_head", "input"),
    },
    "LlamaForCausalLM": {
        "attn.hook_out": ("model.layers.{layer}.self_attn", "output"),
        "attn.hook_z": ("model.layers.{layer}.self_attn.o_proj", "input"),
        "attn.o.hook_in": ("model.layers.{layer}.self_attn.o_proj", "input"),
        "hook_attn_out": ("model.layers.{layer}.self_attn", "output"),
        "hook_in": ("model.layers.{layer}", "input"),
        "hook_mlp_in": ("model.layers.{layer}.post_attention_layernorm", "output"),
        "hook_mlp_out": ("model.layers.{layer}.mlp", "output"),
        "hook_out": ("model.layers.{layer}", "output"),
        "hook_resid_mid": ("model.layers.{layer}.post_attention_layernorm", "input"),
        "hook_resid_post": ("model.layers.{layer}", "output"),
        "hook_resid_pre": ("model.layers.{layer}", "input"),
        "mlp.hook_in": ("model.layers.{layer}.post_attention_layernorm", "output"),
        "mlp.hook_out": ("model.layers.{layer}.mlp", "output"),
        "mlp.hook_pre": ("model.layers.{layer}.mlp", "input"),
        "unembed.hook_in": ("lm_head", "input"),
    },
    "Gemma2ForCausalLM": {
        "attn.hook_out": ("model.layers.{layer}.self_attn", "output"),
        "attn.hook_z": ("model.layers.{layer}.self_attn.o_proj", "input"),
        "attn.o.hook_in": ("model.layers.{layer}.self_attn.o_proj", "input"),
        "hook_attn_out": ("model.layers.{layer}.self_attn", "output"),
        "hook_in": ("model.layers.{layer}", "input"),
        "hook_mlp_out": ("model.layers.{layer}.post_feedforward_layernorm", "output"),
        "hook_out": ("model.layers.{layer}", "output"),
        "hook_resid_mid": ("model.layers.{layer}.pre_feedforward_layernorm", "input"),
        "hook_resid_post": ("model.layers.{layer}", "output"),
        "hook_resid_pre": ("model.layers.{layer}", "input"),
        "ln2.hook_out": ("model.layers.{layer}.pre_feedforward_layernorm", "output"),
        "mlp.hook_in": ("model.layers.{layer}.pre_feedforward_layernorm", "output"),
        "unembed.hook_in": ("lm_head", "input"),
    },
    "Gemma3ForCausalLM": {
        "attn.hook_out": ("model.layers.{layer}.self_attn", "output"),
        "attn.hook_z": ("model.layers.{layer}.self_attn.o_proj", "input"),
        "attn.o.hook_in": ("model.layers.{layer}.self_attn.o_proj", "input"),
        "hook_attn_out": ("model.layers.{layer}.self_attn", "output"),
        "hook_in": ("model.layers.{layer}", "input"),
        "hook_mlp_out": ("model.layers.{layer}.post_feedforward_layernorm", "output"),
        "hook_out": ("model.layers.{layer}", "output"),
        "hook_resid_mid": ("model.layers.{layer}.pre_feedforward_layernorm", "input"),
        "hook_resid_post": ("model.layers.{layer}", "output"),
        "hook_resid_pre": ("model.layers.{layer}", "input"),
        "ln2.hook_out": ("model.layers.{layer}.pre_feedforward_layernorm", "output"),
        "mlp.hook_in": ("model.layers.{layer}.pre_feedforward_layernorm", "output"),
        "unembed.hook_in": ("lm_head", "input"),
    },
    "Gemma3ForConditionalGeneration": {
        "attn.hook_out": ("model.language_model.layers.{layer}.self_attn", "output"),
        "attn.hook_z": ("model.language_model.layers.{layer}.self_attn.o_proj", "input"),
        "attn.o.hook_in": ("model.language_model.layers.{layer}.self_attn.o_proj", "input"),
        "hook_attn_out": ("model.language_model.layers.{layer}.self_attn", "output"),
        "hook_in": ("model.language_model.layers.{layer}", "input"),
        "hook_mlp_out": ("model.language_model.layers.{layer}.post_feedforward_layernorm", "output"),
        "hook_out": ("model.language_model.layers.{layer}", "output"),
        "hook_resid_mid": ("model.language_model.layers.{layer}.pre_feedforward_layernorm", "input"),
        "hook_resid_post": ("model.language_model.layers.{layer}", "output"),
        "hook_resid_pre": ("model.language_model.layers.{layer}", "input"),
        "ln2.hook_out": ("model.language_model.layers.{layer}.pre_feedforward_layernorm", "output"),
        "mlp.hook_in": ("model.language_model.layers.{layer}.pre_feedforward_layernorm", "output"),
        "unembed.hook_in": ("lm_head", "input"),
    },
}

#: Rows the vocabulary deliberately resolves differently, with the reason. Everything else must match exactly.
CHANGED: dict[tuple[str, str], str] = {
    ("GPT2LMHeadModel", "mlp.hook_pre"): "TL defines hook_pre as the up-projection's OUTPUT (mlp.in.hook_out)",
    ("LlamaForCausalLM", "mlp.hook_pre"): "TL defines hook_pre as the up-projection's OUTPUT (mlp.in.hook_out)",
    (
        "LlamaForCausalLM",
        "hook_mlp_in",
    ): "refused: the old row silently substituted the post-norm tensor for a pre-norm name",
    ("Gemma2ForCausalLM", "hook_attn_out"): "TL applies ln1_post BEFORE hook_attn_out on sandwich-norm blocks",
    ("Gemma3ForCausalLM", "hook_attn_out"): "TL applies ln1_post BEFORE hook_attn_out on sandwich-norm blocks",
    (
        "Gemma3ForConditionalGeneration",
        "hook_attn_out",
    ): "TL applies ln1_post BEFORE hook_attn_out on sandwich-norm blocks",
}


@pytest.mark.parametrize("arch", sorted(OLD_TABLE))
def test_every_unchanged_row_is_reproduced(arch):
    resolver = HookNameResolver(arch)
    mismatches = []
    for base, (path, io) in OLD_TABLE[arch].items():
        if (arch, base) in CHANGED:
            continue
        new = resolver._mapping.hook_mappings.get(base)
        if new is None or (new.envoy_path, new.io_type) != (path, io):
            mismatches.append((base, (path, io), None if new is None else (new.envoy_path, new.io_type)))
    assert not mismatches, mismatches


@pytest.mark.parametrize(("arch", "base"), sorted(CHANGED))
def test_every_changed_row_really_changed(arch, base):
    """The exceptions list must not outlive its reasons: a row listed as changed must differ from the old table."""
    resolver = HookNameResolver(arch)
    old = OLD_TABLE[arch][base]
    new = resolver._mapping.hook_mappings.get(base)
    if new is None:
        with pytest.raises(ValueError):
            resolver.resolve(f"blocks.0.{base}")
    else:
        assert (new.envoy_path, new.io_type) != old, f"{arch}/{base} is listed as changed but matches the old table"
