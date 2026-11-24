"""Mounting a pretrained PEFT adapter on a session's model (interpretune.utils.peft_mount)."""

from __future__ import annotations

import builtins

import pytest
import torch

from interpretune.utils import peft_mount as pm

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


def _tiny(name: str = "org/tiny", seed: int = 0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    # `from_pretrained` sets both; PEFT records `name_or_path` as the adapter's base when it saves one
    model.config._name_or_path = model.name_or_path = name
    return model


def _save_lora(path, base_name: str = "org/tiny"):
    """A LoRA with non-zero weights on ``base_name``, saved where ``load_adapter`` can read it."""
    donor = _tiny(base_name, seed=1)
    lora = peft.LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    peft.get_peft_model(donor, lora).save_pretrained(str(path))
    return str(path)


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


@pytest.fixture
def ids():
    return torch.tensor([[1, 5, 9, 13, 2]])


def test_mounted_adapter_is_off_outside_the_block(tmp_path, ids):
    model = _tiny()
    bare = _logits(model, ids)
    pm.mount_adapter(model, _save_lora(tmp_path / "lens"), name="lens", base="org/tiny")
    assert torch.equal(_logits(model, ids), bare), "a mounted adapter must leave the model as it was"
    with pm.adapter_enabled(model, "lens") as hf:
        assert hf is model
        assert not torch.equal(_logits(model, ids), bare), "the adapter must change the read when switched on"
    assert torch.equal(_logits(model, ids), bare)
    with pytest.raises(RuntimeError, match="inside"):
        with pm.adapter_enabled(model, "lens"):
            raise RuntimeError("inside")
    assert torch.equal(_logits(model, ids), bare), "an error in the block must still switch the adapter off"


def test_session_on_another_base_is_refused(tmp_path):
    model = _tiny("org/other")
    with pytest.raises(ValueError, match="is for 'org/tiny', but the session's model is 'org/other'"):
        pm.mount_adapter(model, _save_lora(tmp_path / "lens"), name="lens", base="org/tiny")


def test_adapter_declaring_another_base_is_refused(tmp_path):
    model = _tiny()
    with pytest.raises(ValueError, match="declares the base 'org/other'"):
        pm.mount_adapter(model, _save_lora(tmp_path / "lens", base_name="org/other"), name="lens", base="org/tiny")
    assert pm.mounted_adapters(model) == {}


def test_remount_same_source_is_a_noop_and_a_different_one_is_refused(tmp_path):
    model = _tiny()
    first = pm.mount_adapter(model, _save_lora(tmp_path / "a"), name="lens", base="org/tiny")
    assert pm.mount_adapter(model, str(tmp_path / "a"), name="lens", base="org/tiny") == first
    with pytest.raises(ValueError, match="already mounted as 'lens'"):
        pm.mount_adapter(model, _save_lora(tmp_path / "b"), name="lens", base="org/tiny")


def test_enabling_an_unmounted_adapter_is_refused():
    with pytest.raises(KeyError, match="no adapter is mounted as 'lens'; mounted: none"):
        with pm.adapter_enabled(_tiny(), "lens"):
            pass


def test_missing_peft_is_refused_by_name(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def no_peft(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("No module named 'peft'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_peft)
    with pytest.raises(ImportError, match="needs `peft`"):
        pm.mount_adapter(_tiny(), str(tmp_path), name="lens", base="org/tiny")


def test_unsupported_models_are_refused():
    class TransformerBridge(torch.nn.Module):
        def load_adapter(self, *a, **k):  # pragma: no cover - refused before any call
            raise AssertionError

    with pytest.raises(TypeError, match="not on a TransformerLens TransformerBridge"):
        pm.hf_model_of(TransformerBridge())
    with pytest.raises(TypeError, match="has no PEFT adapter support"):
        pm.hf_model_of(torch.nn.Linear(2, 2))


def test_nnsight_wrapper_resolves_to_its_module():
    model = _tiny()

    class Wrapper:
        def __init__(self, module):
            self._module = module

    assert pm.hf_model_of(Wrapper(model)) is model


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B"),
        ("/cache/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd/", "Qwen/Qwen3.6-27B"),
        ("/some/local/dir/", "/some/local/dir"),
    ],
)
def test_base_id(ref, expected):
    assert pm.base_id(ref) == expected
