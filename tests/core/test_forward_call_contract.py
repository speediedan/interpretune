"""The direct-forward input check agrees with what each wrapper's forward does (``forward_call_mismatch``).

A Trainer loop calls the model with the batch the datamodule yields, so the check is only worth having where its verdict
is the call's. Each case here runs the real call for every primary input key a configuration may declare, and requires a
refusal exactly where the call fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from interpretune.utils.tokenization import forward_call_mismatch, verifiable_forward_inputs
from tests.core.circuit_tracer_toy import TOY_PROMPT, tiny_gemma2_replacement_model

INPUT_KEYS = ("input_ids", "input", "tokens")
NATIVE_BRIDGE_CFG = dict(d_model=16, d_head=4, n_heads=4, n_layers=1, n_ctx=8, d_vocab=32, act_fn="relu", seed=0)


def _hf_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    return GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=32, n_positions=16, bos_token_id=0, eos_token_id=0)
    ).eval()


def _nnsight_over_hf():
    from nnsight import NNsight

    return NNsight(_hf_model())


def _peft():
    peft = pytest.importorskip("peft")
    cfg = peft.LoraConfig(target_modules=["c_attn"], r=2, fan_in_fan_out=True, task_type="CAUSAL_LM")
    return peft.get_peft_model(_hf_model(), cfg)


def _nnsight_over_peft():
    from nnsight import NNsight

    return NNsight(_peft())


def _circuit_tracer_nnsight():
    pytest.importorskip("circuit_tracer")
    return tiny_gemma2_replacement_model(n_layers=1)


def _bridge():
    from transformer_lens.model_bridge.bridge import TransformerBridge

    return TransformerBridge.boot_native(dict(NATIVE_BRIDGE_CFG))


def _sae_bridge():
    """As the SAE Lens adapter makes one: a bridge with its class replaced."""
    from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

    model = _bridge()
    model.__class__ = SAETransformerBridge
    model._acts_to_saes = {}
    model._transcoder_output_hooks = {}
    return model


def _runs(model, key: str) -> bool:
    try:
        with torch.no_grad():
            out = model(**{key: TOY_PROMPT[None]})
    except (TypeError, ValueError):
        return False
    logits = getattr(out, "logits", out)
    assert logits.shape[:2] == (1, len(TOY_PROMPT))
    return True


@pytest.mark.parametrize(
    "build, takes",
    [
        pytest.param(_hf_model, "input_ids", id="hf"),
        pytest.param(_peft, "input_ids", id="peft"),
        pytest.param(_nnsight_over_hf, "input_ids", id="nnsight"),
        pytest.param(_nnsight_over_peft, "input_ids", id="nnsight-peft"),
        pytest.param(_circuit_tracer_nnsight, "input_ids", id="circuit_tracer-nnsight"),
        pytest.param(_bridge, "input", id="bridge"),
        pytest.param(_sae_bridge, "input", id="sae-bridge"),
    ],
)
def test_the_check_refuses_exactly_the_keys_the_forward_fails_on(build, takes):
    model = build()
    runs = {key: _runs(model, key) for key in INPUT_KEYS}
    assert [key for key, ok in runs.items() if ok] == [takes], runs
    refused = {key: forward_call_mismatch(model, {key: TOY_PROMPT[None]}) is not None for key in INPUT_KEYS}
    assert refused == {key: not ok for key, ok in runs.items()}


def test_the_refusal_names_the_key_to_declare_for_a_circuit_tracer_model():
    pytest.importorskip("circuit_tracer")
    problem = forward_call_mismatch(_circuit_tracer_nnsight(), {"input": TOY_PROMPT[None]})
    assert problem is not None
    assert "NNSightReplacementModel.forward takes ['input_ids'" in problem
    assert "model_input_names: ['input_ids', 'attention_mask']" in problem


def test_an_nnsight_subclass_that_calls_its_own_way_is_not_vouched_for():
    """Its own ``__call__`` may rename inputs, so the wrapped module's signature says nothing about it."""
    from nnsight import NNsight

    class Renames(NNsight):
        def __call__(self, *args, **kwargs):
            kwargs["input_ids"] = kwargs.pop("input")
            return super().__call__(*args, **kwargs)

    model = Renames(_hf_model())
    assert _runs(model, "input")
    assert verifiable_forward_inputs(model) is None
    assert forward_call_mismatch(model, {"input": TOY_PROMPT[None]}) is None


def test_an_nnsight_wrapper_over_an_unknown_module_is_not_vouched_for():
    from nnsight import NNsight

    assert verifiable_forward_inputs(NNsight(torch.nn.Linear(2, 2))) is None


# Declarations: the key a configuration's tokenizer names must be one its composition's forward takes.

REGISTRY = Path(__file__).parents[1] / "test_module_registry.yaml"


def _hf_called_entries() -> list[tuple[str, dict]]:
    """Registry entries whose direct forward is a Hugging Face model's: nnsight compositions, and circuit-tracer
    ones, whose replacement model is nnsight's unless a TransformerLens composition builds it."""
    entries = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return [
        (key, entry)
        for key, entry in entries.items()
        if {"nnsight", "circuit_tracer"} & set(entry.get("composition") or ())
        and "transformer_lens" not in entry["composition"]
    ]


def test_registry_entries_called_through_a_hugging_face_forward_declare_its_key_and_the_mask():
    entries = _hf_called_entries()
    assert len(entries) >= 6, [key for key, _ in entries]
    wrong = {}
    for key, entry in entries:
        declared = entry["shared_config"]["tokenizer_kwargs"]["model_input_names"]
        columns = entry["registered_cfg"]["datamodule_cfg"]["signature_columns"]
        # left padding without the mask in the batch runs a direct forward over the padding as if it were text
        if declared[0] != "input_ids" or "attention_mask" not in declared or not set(declared) <= set(columns):
            wrong[key] = (declared, columns)
    assert not wrong


def test_the_conformance_suites_adapter_free_batches_survive_a_direct_forward():
    import interpretune as it
    from interpretune.testing.conformance.inputs import ConformanceInputs

    inputs = ConformanceInputs()
    try:
        session = it.ITSession(inputs.session_cfg(("core",)))
        it.it_init(session.module, session.datamodule)
        batch = next(iter(session.datamodule.test_dataloader()))
        batch = {k: v.to(session.module.device) for k, v in batch.items() if k != "labels"}
        assert "attention_mask" in batch
        with torch.no_grad():
            out = session.module(**batch)
        assert out.logits.shape[:2] == batch["input_ids"].shape
    finally:
        inputs.cleanup()
