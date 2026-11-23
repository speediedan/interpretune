"""CPU tests for the looped lens ops component (``src/it_examples/examples/looped_lens``).

They cover what can be checked without Inkling 2 Mini's weights: the manifest routes exactly the op YAML, the
dispatcher discovers ``looped_lens_read`` from the component directory, the shipped books are well-formed lens books,
the list filters behave as the scored lists did, and every readout or setting the runtime cannot serve is refused by
name rather than read some other way.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch
import yaml

COMPONENT = pathlib.Path(__file__).resolve().parents[1] / "examples" / "looped_lens"


def _runtime():
    spec = importlib.util.spec_from_file_location("looped_lens_runtime", COMPONENT / "looped_lens_runtime.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("looped_lens_runtime", mod)
    spec.loader.exec_module(mod)
    return mod


def _ops():
    _runtime()
    spec = importlib.util.spec_from_file_location("looped_lens_ops", COMPONENT / "looped_lens_ops.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_op_reads_the_model_under_an_nnsight_wrapper():
    # an envoy forwards attributes it lacks to the wrapped module, so a wrong attribute name reads as "no wrapper"
    # and the local runtime would hook the envoy rather than the model
    import nnsight

    ops = _ops()
    lin = torch.nn.Linear(2, 2)
    assert ops._hf_model(nnsight.NNsight(lin)) is lin
    assert ops._hf_model(lin) is lin


def test_manifest_routes_only_the_op_yaml():
    from interpretune.hub.opcollections import resolve_cached_op_files

    files = resolve_cached_op_files(COMPONENT, "looped_lens@local")
    assert [f.name for f in files] == ["looped_lens_ops.yaml"]


def test_the_published_tree_carries_what_the_op_imports(tmp_path):
    # a publish copies only what the manifest declares, so the op module, the runtime it imports and the books it reads
    # must be declared, and the staged tree must load the op on its own
    from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher
    from interpretune.hub.publish import build_component_tree

    out = tmp_path / "build"
    build_component_tree(COMPONENT, out)
    staged = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    expected = {"looped_lens_ops.yaml", "looped_lens_ops.py", "looped_lens_runtime.py", "model_specs.yaml"}
    expected |= {f"books/{p.name}" for p in (COMPONENT / "books").glob("*.yaml")}
    assert expected <= staged, sorted(expected - staged)
    dispatcher = AnalysisOpDispatcher(yaml_paths=[out], enable_hub_ops=False)
    dispatcher._cache_manager.cache_dir = tmp_path / "cache"
    dispatcher.load_definitions()
    assert "looped_lens_read" in dispatcher._op_definitions


def test_the_model_specs_build_inkling_for_its_task():
    from interpretune.harness.sessions import build_session_body, resolve_model_spec

    for variant, remote in (("mini_ndif", True), ("mini", False)):
        spec = resolve_model_spec("inkling2", variant, model_specs=COMPONENT / "model_specs.yaml")
        nnsight_cfg = build_session_body(spec, force_device="cpu")["module_cfg"]["init_args"]["nnsight_cfg"]
        assert nnsight_cfg["init_args"]["task"] == "image-text-to-text"
        assert nnsight_cfg["init_args"].get("remote", False) is remote


def test_dispatcher_discovers_the_op(tmp_path):
    from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

    dispatcher = AnalysisOpDispatcher(yaml_paths=[COMPONENT], enable_hub_ops=False)
    dispatcher._cache_manager.cache_dir = tmp_path
    dispatcher.load_definitions()
    assert "looped_lens_read" in dispatcher._op_definitions
    assert "looped_lens" not in dispatcher._op_definitions or dispatcher._op_definitions["looped_lens"] is not None


@pytest.mark.parametrize("book", sorted((COMPONENT / "books").glob("*.yaml")), ids=lambda p: p.stem)
def test_books_are_lens_books(book):
    R = _runtime()
    doc = yaml.safe_load(book.read_text())
    assert doc["applies_to"]["architectures"] == ["Inkling2LoopForConditionalGeneration"]
    lens = doc["lens"]
    assert lens["variant"] == "plain" and set(lens["filters"]) <= R.FILTERS
    for r in lens["readouts"]:
        assert r["readout"] in R.BOOK_READOUTS
        assert r["set"] in doc["sets"]


def test_lists_apply_the_filters_in_order():
    R = _runtime()
    vocab = ["\n", " Paris", "!", " paris", " France", " capital", " Lyon"]
    top = torch.tensor([[0, 1, 2, 3, 4, 5, 6, -1]])
    assert R.build_lists(top, vocab, "The capital of France is", 3, set()) == [["\n", " Paris", "!"]]
    assert R.build_lists(top, vocab, "The capital of France is", 3, {"content"}) == [[" Paris", " paris", " France"]]
    assert R.build_lists(top, vocab, "The capital of France is", 3, {"content", "echo"}) == [
        [" Paris", " paris", " Lyon"]
    ]


def _site(label: str, readout: str, layer: int, p: int, points=("state.active",)) -> dict:
    return {
        "label": label,
        "readout": readout,
        "layer": layer,
        "pass": p,
        "reads": {
            pt: {
                "module": f"model.language_model.layers.{layer}",
                "call_index": p - 1,
                "hook": "output",
                "path": [],
                "step": layer,
            }
            for pt in points
        },
    }


@pytest.mark.parametrize(
    "site, match",
    [
        (_site("L9p2:coda", "coda", 9, 2), "needs a core visit at layers 12 to 17"),
        (_site("L11p2:moe_in", "moe_in", 11, 2, ("message.gated.mlp",)), "MoE input on a message visit"),
        (_site("L15p3:sae", "sae_dir", 15, 3), "has no remote path"),
    ],
)
def test_remote_refuses_what_it_cannot_serve(site, match):
    R = _runtime()
    with pytest.raises(ValueError, match=match):
        R.check_remote({"spec": {"K": 3}, "sites": [site]})


def test_remote_accepts_pass_three_coda_visits():
    R = _runtime()
    R.check_remote(
        {"spec": {"K": 3}, "sites": [_site("L15p3:coda", "coda", 15, 3), _site("L19:active", "active", 19, 1)]}
    )


def test_unimplemented_variant_and_filter_are_refused():
    R = _runtime()
    with pytest.raises(ValueError, match="not implemented"):
        R.check_spec({"variant": "ablate", "filters": []})
    with pytest.raises(ValueError, match="not implemented"):
        R.check_spec({"variant": "plain", "filters": ["stopwords"]})


def test_a_book_without_a_map_is_refused():
    R = _runtime()
    with pytest.raises(ValueError, match="needs venngram's Inkling 2 Mini map"):
        R.load_table(COMPONENT / "books" / "inkling2_mini_icv_nocoda_k10_n6.yaml")


def test_a_venngram_without_execution_maps_is_refused(monkeypatch, tmp_path):
    # a venngram that reads the book but predates execution maps loads the map without one; the book's visits must be
    # refused by name rather than resolved against the weight-layer map (the pinned venngram refuses the book itself)
    import venngram.book
    import venngram.maps

    class _Book:
        @classmethod
        def from_file(cls, path):
            return cls()

        def check_applies(self, **kwargs):
            return None

        def sets(self):
            return {}

    R = _runtime()
    book = sorted((COMPONENT / "books").glob("*.yaml"))[0]
    map_path = tmp_path / "inkling2_mini.yaml"
    map_path.write_text("architecture: Inkling2LoopForConditionalGeneration\n")
    monkeypatch.setattr(venngram.book, "AddressBook", _Book)
    monkeypatch.setattr(venngram.maps, "load_component_map", lambda path: object())
    with pytest.raises(ValueError, match="venngram read no execution map"):
        R.resolve_book(book, map_path)
