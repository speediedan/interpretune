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
        assert r.get("position", "read") in R.POSITIONS


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


class _Tok:
    """A stand-in tokenizer: one character per token, ``U`` the user-turn marker."""

    def convert_tokens_to_ids(self, token):
        return ord("U") if token == "<|message_user|>" else -1

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_question_end_reads_the_last_question_mark_of_the_user_turn():
    R = _runtime()
    tok = _Tok()
    ids = [ord(c) for c in "S?xU ab? cd? e.M"]
    read = len(ids) - 1
    assert R.question_end(ids, read, tok) == 11  # the second "?" of the user turn, not the system turn's
    plain = [ord(c) for c in "what? no marker"]
    assert R.question_end(plain, len(plain) - 1, tok) == len(plain) - 1
    no_mark = [ord(c) for c in "SUabc.M"]
    assert R.question_end(no_mark, len(no_mark) - 1, tok) == len(no_mark) - 1


def test_site_positions_follow_each_site():
    R = _runtime()
    table = {"sites": [{"label": "a", "position": "read"}, {"label": "b", "position": "question_end"}, {"label": "c"}]}
    ids = [ord(c) for c in "Uab?c.M"]
    assert R.site_positions(table, ids, len(ids) - 1, _Tok()) == {"a": 6, "b": 3, "c": 6}


def test_a_book_position_outside_the_runtimes_is_refused(monkeypatch, tmp_path):
    import venngram.book
    import venngram.maps

    class _Book:
        @classmethod
        def from_file(cls, path):
            return cls()

        def check_applies(self, **kwargs):
            return None

        def sets(self):
            return {"s": "point:state.active & layers:19"}

    class _Exec:
        def resolve(self, address, sets=None):
            raise AssertionError("the position is checked before any address resolves")

    class _Map:
        execution = _Exec()

    R = _runtime()
    book = tmp_path / "b.yaml"
    book.write_text(
        "lens: {model: m, K: 3, variant: plain, filters: [], readouts: [{set: s, readout: plain, position: answer}]}\n"
    )
    map_path = tmp_path / "inkling2_mini.yaml"
    map_path.write_text("architecture: Inkling2LoopForConditionalGeneration\n")
    monkeypatch.setattr(venngram.book, "AddressBook", _Book)
    monkeypatch.setattr(venngram.maps, "load_component_map", lambda path: _Map())
    with pytest.raises(ValueError, match="readout position 'answer' is not one this lens reads"):
        R.resolve_book(book, map_path)


# ---------------------------------------------------------------------------------------------------------------------
# the lens output as an AnalysisStore, and the store as the challenge's readouts
# ---------------------------------------------------------------------------------------------------------------------
LABELS = ["L15p3:coda", "L19:active", "L18:moe_in"]
TOKENS = [[["Egypt", "Cairo"], ["Giza"], []], [["Paris", "France"], ["Lyon", "Nice"], ["Rhone"]]]


def _op_batch(prompt_tokens, first: int):
    from interpretune.analysis.ops.base import AnalysisBatch

    n = len(prompt_tokens)
    ab = AnalysisBatch()
    ab.update(
        looped_lens_labels=list(LABELS),
        looped_lens_tokens=prompt_tokens,
        looped_lens_top_ids=torch.arange(n * 3 * 100).reshape(n, 3, 100) + first,
        looped_lens_positions=torch.arange(n),
        looped_lens_book="books/example.yaml",
    )
    return ab


@pytest.fixture
def lens_store(tmp_path):
    from types import SimpleNamespace

    from interpretune.analysis.core import analysis_store_from_batches
    from interpretune.analysis.ops.base import AnalysisOp
    from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

    dispatcher = AnalysisOpDispatcher(yaml_paths=[COMPONENT], enable_hub_ops=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    dispatcher._cache_manager.cache_dir = cache
    dispatcher.load_definitions()
    op = dispatcher.get_op("looped_lens_read")
    assert isinstance(op, AnalysisOp)
    tok = SimpleNamespace(vocab_size=200058, model_max_length=4096)
    module = SimpleNamespace(
        datamodule=SimpleNamespace(itdm_cfg=SimpleNamespace(eval_batch_size=1)),
        it_cfg=SimpleNamespace(),
        model=SimpleNamespace(tokenizer=tok),
        analysis_cfg=None,
    )
    # one prompt, then two in one call: rows of different prompt counts
    return analysis_store_from_batches(module, [_op_batch(TOKENS[:1], 0), _op_batch(TOKENS, 1000)], op=op)


def test_the_lens_output_survives_the_store(lens_store):
    # a list of strings declared as `object` is stored as its repr; the columns must come back as the lists they were
    row = lens_store.dataset[1]
    assert row["looped_lens_labels"] == LABELS
    assert [[list(s) for s in p] for p in row["looped_lens_tokens"]] == TOKENS
    raw = lens_store.dataset.with_format(None)[0]["looped_lens_tokens"]
    assert not isinstance(raw, str), raw


def test_the_store_writes_the_challenge_readouts(lens_store, tmp_path):
    R = _runtime()
    ids = ["q0", "q1", "q2"]
    expect_first = [
        {"layer": 39, "tokens": ["Egypt", "Cairo"]},
        {"layer": 42, "tokens": []},
        {"layer": 43, "tokens": ["Giza"]},
    ]
    got = R.challenge_readouts(lens_store, ids)
    assert [g["id"] for g in got] == ids and got[0]["readouts"] == expect_first
    assert [r["layer"] for r in got[2]["readouts"]] == [39, 42, 43]
    assert got[2]["readouts"][0]["tokens"] == ["Paris", "France"]
    # the same rows read back from disk, and streamed, write the same file
    from datasets import load_dataset

    from interpretune.analysis.core import AnalysisStore

    path = tmp_path / "rows.parquet"
    lens_store.dataset.to_parquet(str(path))
    fmt = {"col_cfg": lens_store.it_format_kwargs["col_cfg"]}
    again = AnalysisStore(dataset=load_dataset("parquet", data_files=str(path), split="train"), it_format_kwargs=fmt)
    assert R.challenge_readouts(again, ids) == got
    stream = load_dataset("parquet", data_files=str(path), split="train", streaming=True)
    assert R.challenge_readouts(AnalysisStore(dataset=stream, it_format_kwargs=fmt), ids) == got


def test_the_visit_of_a_label_is_the_challenge_layer():
    R = _runtime()
    assert [
        R.visit_index(x) for x in ("L0", "L6p1:coda", "L14p3:coda", "L15p3", "L17p3", "L18:moe_in", "L23:active")
    ] == [0, 6, 38, 39, 41, 42, 47]
    for bad in ("X3", "L15:coda", "L6p1:coda".replace("p1", ""), "L3p1", "L24", "L18p2"):
        with pytest.raises(ValueError):
            R.visit_index(bad)


def test_readouts_refuse_what_the_challenge_would_not_take(lens_store):
    R = _runtime()
    with pytest.raises(ValueError, match="more prompts"):
        R.challenge_readouts(lens_store, ["only-one"])
    with pytest.raises(ValueError, match="prompts for"):
        R.challenge_readouts(lens_store, ["q0", "q1", "q2", "q3"])
    row = {"looped_lens_labels": ["L15p3:coda", "L15p3:active"], "looped_lens_tokens": [[["a"], ["b"]]]}
    with pytest.raises(ValueError, match="at most"):
        R.challenge_readouts([row], ["q0"])
