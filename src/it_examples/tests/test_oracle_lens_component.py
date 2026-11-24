"""CPU tests for the oracle lens ops component (``src/it_examples/examples/oracle_lens``).

They run on a tiny random model of Qwen3.6-27B's architecture, the real tokenizer (the injection marker depends on it)
and fake adapters in the released checkpoint's layout, so what they cover is the mechanics: the marker, the injection,
the capture, the adapter being on only for the read, the reconstructor fit, and every contract term the checkpoints fix
being refused. Each check has a planted failure beside it, because a check that cannot fail proves nothing. Whether the
lens reads well is measured on the real weights (interpretune#1202, step 1).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

COMPONENT = pathlib.Path(__file__).resolve().parents[1] / "examples" / "oracle_lens"
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


def _runtime():
    spec = importlib.util.spec_from_file_location("oracle_lens_runtime", COMPONENT / "oracle_lens_runtime.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oracle_lens_runtime"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _runtime()


def _ops():
    spec = importlib.util.spec_from_file_location("oracle_lens_ops", COMPONENT / "oracle_lens_ops.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tok():
    return transformers.AutoTokenizer.from_pretrained(R.BASE_ID, revision=R.BASE_REV)


def _tiny_cfg(layers: int):
    cfg = transformers.AutoConfig.from_pretrained(R.BASE_ID, revision=R.BASE_REV).text_config
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = layers, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.layer_types = cfg.layer_types[:layers]
    return cfg


@pytest.fixture(scope="module")
def text_cfg():
    """Deep enough to capture at layer 20, the lowest of the trained band."""
    return _tiny_cfg(24)


@pytest.fixture(scope="module")
def shallow_cfg():
    """For what never indexes a layer: the read takes the layer into its prompt text only."""
    return _tiny_cfg(4)


def _model(text_cfg, seed: int = 0):
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(text_cfg).eval()
    model.config._name_or_path = model.name_or_path = R.BASE_ID
    return model


def _save_lora(path: pathlib.Path, text_cfg, seed: int = 1) -> dict:
    """A non-zero LoRA on the full-attention projections, saved in the released layout; returns its tensors."""
    from safetensors.torch import load_file

    lora = peft.LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    peft.get_peft_model(_model(text_cfg, seed), lora).save_pretrained(str(path))
    return load_file(str(path / "adapter_model.safetensors"))


@pytest.fixture(scope="module")
def root(tmp_path_factory, text_cfg):
    """A checkpoint directory in the released layout: one lens adapter, the AR (keys as released) and its files."""
    from safetensors.torch import save_file

    root = tmp_path_factory.mktemp("olens")
    _save_lora(root / R.CONTRACT["space"]["dir"], text_cfg)
    ar_dir = root / "ar_ptag_pooled" / "lora"
    sd = _save_lora(ar_dir, text_cfg, seed=2)
    released = {
        k.replace("base_model.model.model.layers.", "base_model.model.layers.").replace(
            ".self_attn.", "._orig_mod.self_attn."
        ): v
        for k, v in sd.items()
    }
    save_file(released, str(ar_dir / "adapter_model.safetensors"))
    d = text_cfg.hidden_size
    torch.manual_seed(3)
    torch.save(
        {
            "layers": [20],
            "tag_ids": torch.tensor([[5, 6]]),
            "head": {"linear.weight": torch.randn(d, d), "linear.bias": torch.zeros(d)},
        },
        root / "ar_ptag_pooled" / "heads.pt",
    )
    (root / "whitener" / "W").mkdir(parents=True)
    save_file(
        {"mu": torch.zeros(d), "w": torch.eye(d)},
        str(root / "whitener" / "W" / "whitening_iolens_pooled_W_L20.safetensors"),
    )
    return root


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids[None]).logits


def test_manifest_routes_only_the_op_yaml():
    from interpretune.hub.opcollections import resolve_cached_op_files

    files = resolve_cached_op_files(COMPONENT, "oracle_lens@local")
    assert [f.name for f in files] == ["oracle_lens_ops.yaml"]


def test_the_published_tree_carries_what_the_op_imports(tmp_path):
    from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher
    from interpretune.hub.publish import build_component_tree

    out = tmp_path / "build"
    build_component_tree(COMPONENT, out)
    staged = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"oracle_lens_ops.yaml", "oracle_lens_ops.py", "oracle_lens_runtime.py"} <= staged
    dispatcher = AnalysisOpDispatcher(yaml_paths=[out], enable_hub_ops=False)
    dispatcher._cache_manager.cache_dir = tmp_path / "cache"
    dispatcher.load_definitions()
    assert "oracle_lens_read" in dispatcher._op_definitions


@pytest.mark.parametrize("kind", sorted(R.PROMPTS))
@pytest.mark.parametrize("wrap", [False, True])
def test_the_marker_is_one_token_appearing_once(tok, kind, wrap):
    p = R.render_prompt(tok, kind, 44, wrap)
    assert p.ids.tolist().count(p.char_id) == 1 and int(p.ids[p.slot]) == p.char_id
    assert len(tok(tok.decode([p.char_id]), add_special_tokens=False).input_ids) == 1


def test_a_prompt_where_no_marker_appears_once_is_refused(tok, monkeypatch):
    # planted: every candidate appears twice, so none qualifies
    monkeypatch.setitem(R.PROMPTS, "twice", "layer {layer}: {char} and {char}")
    with pytest.raises(ValueError, match="no character in U\\+3200..U\\+33FF"):
        R.render_prompt(tok, "twice", 44)


def test_contract_terms_are_refused():
    with pytest.raises(ValueError, match="outside the oracle lens's trained band"):
        R.check_layer(22)
    with pytest.raises(ValueError, match="unknown oracle lens adapter 'lensB'"):
        R.check_adapter("lensB")
    with pytest.raises(ValueError, match="unknown oracle lens prompt 'concepts'"):
        R.prompt_text("concepts", 44, "x")
    with pytest.raises(ValueError, match="unknown oracle lens transform 'norm'"):
        R.transform(torch.ones(1, 4), "norm", 1.0)


def test_transforms():
    v = R.transform(torch.full((2, 8), 5.0), "unit", 16000.0)
    assert torch.allclose(v.norm(dim=-1), torch.full((2,), 16000.0), rtol=1e-4)
    assert torch.equal(R.transform(torch.ones(1, 8), "scaled", 2.0), torch.full((1, 8), 2.0))


def test_injecting_the_markers_own_embedding_reproduces_the_prompt(tok, shallow_cfg):
    model = _model(shallow_cfg)
    p = R.render_prompt(tok, "concepts_raw", 44)
    own = model.get_input_embeddings().weight[p.char_id].detach()
    with torch.no_grad():
        plain = model(input_ids=p.ids[None]).logits[0, -1]
        e, m = R.inject(model, p, own[None])
        same = model(inputs_embeds=e, attention_mask=m).logits[0, -1]
        e2, m2 = R.inject(model, p, own[None] * 3)  # planted: another vector must change the read
        other = model(inputs_embeds=e2, attention_mask=m2).logits[0, -1]
    assert torch.allclose(same, plain, atol=1e-5)
    assert not torch.allclose(other, plain, atol=1e-5)


def test_capture_is_the_block_output_with_the_adapter_off(tok, text_cfg, root, tmp_path):
    model = _model(text_cfg)
    ids = torch.tensor(tok("The capital of France is Paris.").input_ids)
    before = R.capture(model, ids, (20, 22))
    bare = _logits(model, ids)
    R.mount_lens(model, "space", root, tmp_path)
    after = R.capture(model, ids, (20, 22))
    assert torch.equal(after[20], before[20]) and torch.equal(after[22], before[22])
    assert torch.equal(_logits(model, ids), bare), "a mounted lens must leave the session's model as it was"
    with torch.no_grad():
        hs = model(input_ids=ids[None], output_hidden_states=True).hidden_states
    assert torch.allclose(after[20], hs[21][0].float(), atol=1e-5), "block 20's output is hidden_states[21]"
    assert not torch.allclose(after[20], hs[20][0].float(), atol=1e-5), "planted off-by-one must be caught"


def test_read_is_seeded_and_leaves_the_adapter_off(tok, shallow_cfg, tmp_path):
    root = tmp_path / "olens"
    _save_lora(root / R.CONTRACT["space"]["dir"], shallow_cfg)
    model = _model(shallow_cfg)
    R.mount_lens(model, "space", root, tmp_path)
    ids = torch.tensor(tok("The capital of France is Paris.").input_ids)
    bare = _logits(model, ids)
    h = torch.randn(2, shallow_cfg.hidden_size)
    read = lambda seed: R.read(model, tok, "space", 44, h, samples=2, seed=seed, max_new_tokens=5)  # noqa: E731
    first, again, other = read(1), read(1), read(2)
    assert first == again and first != other
    assert len(first) == 2 and all(len(r) == 2 for r in first)
    assert torch.equal(_logits(model, ids), bare)


def test_a_read_without_a_mounted_lens_is_refused(tok, shallow_cfg):
    with pytest.raises(KeyError, match="no adapter is mounted as 'oracle_lens_space'"):
        R.read(_model(shallow_cfg), tok, "space", 44, torch.randn(1, shallow_cfg.hidden_size), max_new_tokens=2)


def test_an_adapter_whose_tensors_match_nothing_is_refused(shallow_cfg, tmp_path):
    from safetensors.torch import load_file, save_file

    root = tmp_path / "bad"
    sub = root / R.CONTRACT["space"]["dir"]
    _save_lora(sub, shallow_cfg)
    sd = load_file(str(sub / "adapter_model.safetensors"))
    save_file({k.replace("q_proj", "nowhere_proj"): v for k, v in sd.items()}, str(sub / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="does not fit the session's model: [0-9]+ of its tensors match no parameter"):
        R.mount_lens(_model(shallow_cfg), "space", root, tmp_path / "scratch")


def test_the_ar_keys_are_remapped_and_a_collision_is_refused(root, tmp_path):
    from safetensors.torch import load_file, save_file

    out = R._remap_ar(root / "ar_ptag_pooled" / "lora", tmp_path / "scratch")
    keys = list(load_file(str(out / "adapter_model.safetensors")))
    assert keys and all(k.startswith("base_model.model.model.layers.") and "_orig_mod" not in k for k in keys)
    bad = tmp_path / "collide"
    bad.mkdir()
    (bad / "adapter_config.json").write_text("{}")
    save_file(
        {"base_model.model.layers.0._orig_mod.x": torch.zeros(1), "base_model.model.model.layers.0.x": torch.ones(1)},
        str(bad / "adapter_model.safetensors"),
    )
    with pytest.raises(ValueError, match="two AR keys map to"):
        R._remap_ar(bad, tmp_path / "scratch")


def test_each_ar_checkpoint_gets_its_own_remapped_copy(root, tmp_path):
    # planted: two checkpoints sharing a scratch directory; keyed by anything but the file, the second would be given
    # the first one's copy
    import shutil

    from safetensors.torch import load_file, save_file

    src, scratch = root / "ar_ptag_pooled" / "lora", tmp_path / "scratch"
    first = R._remap_ar(src, scratch)
    assert R._remap_ar(src, scratch) == first
    other = tmp_path / "other"
    shutil.copytree(src, other)
    sd = {k: v + 1 for k, v in load_file(str(src / "adapter_model.safetensors")).items()}
    save_file(sd, str(other / "adapter_model.safetensors"))
    second = R._remap_ar(other, scratch)
    a, b = (load_file(str(d / "adapter_model.safetensors")) for d in (first, second))
    assert second != first and all(torch.equal(b[k], a[k] + 1) for k in a)
    assert sorted(p.name for p in scratch.iterdir()) == sorted([first.name, second.name]), "no partial copies are left"


def test_bullets():
    assert R.bullets("intro\n- rhyme\n  - night \n-nope\n- \n") == ["rhyme", "night"]


def test_joint_fve():
    y = torch.tensor([3.0, 4.0, 0.0])
    fve, c = R.joint_fve(torch.eye(3)[:2], y)
    assert fve == pytest.approx(1.0) and torch.allclose(c, torch.tensor([3.0, 4.0]))
    fve, c = R.joint_fve(
        -torch.eye(3)[:2], y
    )  # planted: only negative coefficients would help, and they are not allowed
    assert fve == pytest.approx(0.0, abs=1e-6) and torch.all(c == 0)
    with pytest.raises(ValueError, match="zero norm"):
        R.joint_fve(torch.eye(3), torch.zeros(3))


class _Module:
    def __init__(self, model, tok):
        self.model, self.tokenizer = model, tok


def test_the_op_reads_each_prompt_at_its_position_and_fits_the_bullets(tok, text_cfg, root, monkeypatch, tmp_path):
    ops = _ops()
    monkeypatch.setattr(ops, "_scratch", lambda revision: tmp_path / revision)
    # the random model writes no bullets, so the read is replaced to exercise the fit; the read itself is tested above
    seen = []

    def fake_read(model, tok_, name, layer, h, **sampling):
        seen.append((name, layer, h.clone(), sampling["samples"]))
        return [["- alpha\n- beta"] * sampling["samples"]]

    monkeypatch.setattr(ops.R, "read", fake_read)
    model = _model(text_cfg)
    module = _Module(model, tok)
    enc = tok(["The capital of France is Paris.", "Roses are red"], padding=True, return_tensors="pt")
    out = ops.oracle_lens_read_impl(
        module,
        {},
        {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask},
        0,
        oracle_lens_root=str(root),
        oracle_lens_layers=[20],
        oracle_lens_positions=[-1, -2],
        oracle_lens_ar=True,
        oracle_lens_samples=2,
    )
    assert out["oracle_lens_bullets"] == [[[["alpha", "beta"]] * 2]] * 2
    assert len(out["oracle_lens_fve"][0][0]) == 2 and all(0 <= f <= 1 for f in out["oracle_lens_fve"][0][0])
    assert all(len(c) == 2 and min(c) >= 0 for c in out["oracle_lens_coefs"][1][0])
    lens = [int(m.sum()) for m in enc.attention_mask]
    assert out["oracle_lens_positions"].tolist() == [lens[0] - 1, lens[1] - 2]
    first_ids = enc.input_ids[0][enc.attention_mask[0].bool()]
    assert torch.equal(seen[0][2][0], R.capture(model, first_ids, (20,))[20][lens[0] - 1])
    assert '"adapter": "space"' in out["oracle_lens_sampling"]


def test_the_op_refuses_a_position_outside_the_prompt(tok, text_cfg, root, monkeypatch, tmp_path):
    ops = _ops()
    monkeypatch.setattr(ops, "_scratch", lambda revision: tmp_path / revision)
    enc = tok(["Paris"], return_tensors="pt")
    with pytest.raises(ValueError, match="position -40 is outside its"):
        ops.oracle_lens_read_impl(
            _Module(_model(text_cfg), tok),
            {},
            {"input_ids": enc.input_ids},
            0,
            oracle_lens_root=str(root),
            oracle_lens_layers=[20],
            oracle_lens_positions=[-40],
        )
