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


def _save_lora(path, base_name: str = "org/tiny", seed: int = 1):
    """A LoRA with non-zero weights on ``base_name``, saved where ``load_adapter`` can read it."""
    donor = _tiny(base_name, seed=seed)
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


def test_an_adapter_whose_tensors_do_not_land_is_refused_and_removed(tmp_path, ids):
    # PEFT gives every configured target fresh weights, so without the load report this adapter would mount and read
    # as a random one: planted by renaming its q_proj tensors to a module the model does not have
    from safetensors.torch import load_file, save_file

    path = tmp_path / "lens"
    _save_lora(path)
    sd = load_file(str(path / "adapter_model.safetensors"))
    save_file({k.replace("q_proj", "nowhere_proj"): v for k, v in sd.items()}, str(path / "adapter_model.safetensors"))
    model = _tiny()
    bare = _logits(model, ids)
    with pytest.raises(ValueError, match="does not fit the session's model: 4 of its tensors match no parameter"):
        pm.mount_adapter(model, str(path), name="lens", base="org/tiny")
    assert pm.mounted_adapters(model) == {}
    assert "lens" not in (getattr(model, "peft_config", None) or {})
    assert torch.equal(_logits(model, ids), bare)


def test_two_mounted_adapters_keep_their_own_weights_and_do_not_nest(tmp_path, ids):
    model = _tiny()
    bare = _logits(model, ids)
    pm.mount_adapter(model, _save_lora(tmp_path / "a"), name="a", base="org/tiny")
    with pm.adapter_enabled(model, "a"):
        read_a = _logits(model, ids)
        with pytest.raises(RuntimeError, match="'a' is on; mount adapters outside its block"):
            pm.mount_adapter(model, _save_lora(tmp_path / "b", seed=2), name="b", base="org/tiny")
    pm.mount_adapter(model, str(tmp_path / "b"), name="b", base="org/tiny")
    with pm.adapter_enabled(model, "b"):
        assert not torch.equal(_logits(model, ids), read_a), "the two adapters must read differently"
    with pm.adapter_enabled(model, "a"):
        # ending an inner block would switch this one off, so it is refused and this one stays on
        with pytest.raises(RuntimeError, match="'a' is already on; blocks do not nest"):
            with pm.adapter_enabled(model, "b"):
                pass
        assert torch.equal(_logits(model, ids), read_a)
    assert torch.equal(_logits(model, ids), bare)


def test_a_session_that_runs_its_own_adapter_is_refused_and_unchanged(tmp_path, ids):
    lens = _save_lora(tmp_path / "lens")
    native = _tiny()
    native.load_adapter(_save_lora(tmp_path / "own", seed=2), adapter_name="own")
    own = _logits(native, ids)
    with pytest.raises(ValueError, match="already carries adapters that were not mounted here: \\['own'\\]"):
        pm.mount_adapter(native, lens, name="lens", base="org/tiny")
    assert torch.equal(_logits(native, ids), own), "a refused mount must leave the session's own adapter on"

    tuned = peft.get_peft_model(_tiny(), peft.LoraConfig(r=4, target_modules=["q_proj"], init_lora_weights=False))
    own = _logits(tuned, ids)
    with pytest.raises(TypeError, match="already runs an adapter of its own"):
        pm.mount_adapter(tuned, lens, name="lens", base="org/tiny")
    assert torch.equal(_logits(tuned, ids), own)


@pytest.mark.parametrize("stored", [torch.float32, torch.bfloat16])
def test_a_lower_precision_session_reads_the_adapter_at_its_own_precision(tmp_path, ids, stored):
    from safetensors.torch import load_file, save_file

    path = tmp_path / "lens"
    weights = load_file(_save_lora(path) + "/adapter_model.safetensors")
    save_file({k: v.to(stored) for k, v in weights.items()}, str(path / "adapter_model.safetensors"))
    reference = peft.PeftModel.from_pretrained(_tiny().to(torch.bfloat16), str(path))
    model = _tiny().to(torch.bfloat16)
    mount = pm.mount_adapter(model, str(path), name="lens", base="org/tiny")
    assert mount.dtype is torch.float32
    with pm.adapter_enabled(model, "lens"):
        assert torch.equal(_logits(model, ids), _logits(reference, ids))
    assert {p.dtype for n, p in model.named_parameters() if ".lens." not in n} == {torch.bfloat16}

    # rounded to the session's dtype, the same adapter reads differently: what the default keeps out
    rounded = _tiny().to(torch.bfloat16)
    pm.mount_adapter(rounded, str(path), name="lens", base="org/tiny", dtype=torch.bfloat16)
    with pm.adapter_enabled(rounded, "lens"):
        assert not torch.equal(_logits(rounded, ids), _logits(reference, ids))
    with pytest.raises(ValueError, match="already mounted as 'lens'"):
        pm.mount_adapter(rounded, str(path), name="lens", base="org/tiny")


def test_an_adapter_that_did_not_load_at_its_dtype_is_refused_and_removed(tmp_path, ids, monkeypatch):
    model = _tiny().to(torch.bfloat16)
    bare = _logits(model, ids)
    load = type(model).load_adapter
    # a loader that ignores the plan
    monkeypatch.setattr(type(model), "load_adapter", lambda self, *a, dtype_plan=None, **k: load(self, *a, **k))
    with pytest.raises(RuntimeError, match=r"was to be held in torch.float32, but loaded as \['torch.bfloat16'\]"):
        pm.mount_adapter(model, _save_lora(tmp_path / "lens"), name="lens", base="org/tiny")
    assert pm.mounted_adapters(model) == {}
    assert torch.equal(_logits(model, ids), bare)


def test_an_adapter_added_after_the_mount_is_refused_at_the_read(tmp_path, ids):
    model = _tiny()
    pm.mount_adapter(model, _save_lora(tmp_path / "lens"), name="lens", base="org/tiny")
    model.load_adapter(_save_lora(tmp_path / "own", seed=2), adapter_name="own")
    own = _logits(model, ids)
    with pytest.raises(ValueError, match="not mounted here: \\['own'\\]"):
        with pm.adapter_enabled(model, "lens"):
            pass
    assert torch.equal(_logits(model, ids), own)


def test_enabling_an_unmounted_adapter_is_refused():
    with pytest.raises(KeyError, match="no adapter is mounted as 'lens'; mounted: none"):
        with pm.adapter_enabled(_tiny(), "lens"):
            pass


def test_missing_peft_is_refused(monkeypatch, tmp_path):
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
