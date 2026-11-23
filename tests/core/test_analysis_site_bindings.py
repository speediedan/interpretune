"""Reading an AnalysisStore against another model: which declared site bindings carry over, and which are refused."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from interpretune.analysis.ops.base import ColCfg
from interpretune.analysis.site_bindings import (
    SiteBinding,
    SiteBindingError,
    StoreResolutionError,
    check_bindings,
    declares_bindings,
    parse_site,
)
from interpretune.hub.subject import StoreSubject

GEMMA2 = "Gemma2ForCausalLM"
TRIPLE = [
    {"binds": "layer", "slot": 0, "point": "hook_mlp_out"},
    {"binds": "tokens", "slot": 1},
    {"binds": "latent_feature", "slot": 2},
]
DICTIONARY = {"kind": "circuit_tracer", "transcoder_set": "gemma"}


def _subject(**overrides) -> StoreSubject:
    fields = {
        "model_id": "google/gemma-2-2b",
        "model_revision": "c5ebcd40",
        "architecture": GEMMA2,
        "tokenizer_id": "google/gemma-2-2b",
        "dictionaries": (DICTIONARY,),
        "n_layers": 26,
    }
    return StoreSubject(**{**fields, **overrides})


def _cfg(site=None, **kwargs) -> dict:
    return ColCfg(datasets_dtype="int64", site=site, **kwargs).to_dict()


def _check(cfg, *, source=None, target=None, columns=("col",), keys=None):
    check_bindings(
        {"col": cfg} if "col" not in (cfg or {}) else cfg,
        columns,
        keys or {},
        source or _subject(),
        target or _subject(),
    )


class TestDeclaration:
    def test_a_site_is_parsed_to_bindings_and_survives_json(self):
        cfg = ColCfg(datasets_dtype="int64", array_shape=(None, 3), site=TRIPLE)
        assert cfg.site[0] == SiteBinding("layer", 0, "hook_mlp_out") and len(cfg.site) == 3
        again = ColCfg.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert again == cfg and hash(again) == hash(cfg)
        assert hash(cfg) != hash(ColCfg(datasets_dtype="int64", array_shape=(None, 3)))

    def test_a_mapping_is_one_binding_and_none_is_none(self):
        assert parse_site({"binds": "tokens"}) == (SiteBinding("tokens"),)
        assert parse_site(None) is None
        assert ColCfg(datasets_dtype="int64").site is None

    @pytest.mark.parametrize(
        ("site", "match"),
        [
            ({"binds": "layers"}, "not a binding kind"),
            ({"slot": 0}, "takes `binds`"),
            ({"binds": "layer", "at": 3}, "takes `binds`, `slot` and `point`"),
            ({"binds": "tokens", "point": "hook_mlp_out"}, "belongs to a `layer` binding"),
            ({"binds": "layer", "slot": -1}, "non-negative"),
            ([], "`site` is empty"),
            ("layer", "a site binding is a mapping"),
        ],
    )
    def test_a_malformed_site_is_refused_by_name(self, site, match):
        with pytest.raises(SiteBindingError, match=match):
            parse_site(site)

    def test_an_entry_without_the_site_key_predates_bindings(self):
        current = {"col": _cfg(TRIPLE), "plain": _cfg()}
        assert declares_bindings(current)
        for entry in current.values():
            assert "site" in entry
        older = {"col": {k: v for k, v in _cfg().items() if k != "site"}, "plain": _cfg()}
        assert not declares_bindings(older)  # one column without the key is enough: it cannot be vouched for
        assert not declares_bindings({}) and not declares_bindings(None)


class TestLayerBindings:
    def test_the_same_depth_carries_over(self):
        _check(_cfg({"binds": "layer"}))

    def test_another_depth_is_refused_naming_both(self):
        with pytest.raises(StoreResolutionError, match="has 26 layers and the target 18"):
            _check(_cfg({"binds": "layer"}), target=_subject(n_layers=18))

    @pytest.mark.parametrize(("side", "source_n", "target_n"), [("store", None, 26), ("target", 26, None)])
    def test_an_unrecorded_depth_is_refused_naming_the_side(self, side, source_n, target_n):
        with pytest.raises(StoreResolutionError, match=f"the {side}'s layer count is not recorded"):
            _check(_cfg({"binds": "layer"}), source=_subject(n_layers=source_n), target=_subject(n_layers=target_n))

    def test_the_point_must_resolve_on_the_target(self):
        _check(_cfg({"binds": "layer", "point": "hook_mlp_out"}))
        with pytest.raises(StoreResolutionError, match="has no component"):
            # a component Gemma2's map has no row for
            _check(_cfg({"binds": "layer", "point": "cross_attn.hook_out"}))
        with pytest.raises(StoreResolutionError, match="no component map is registered for NoSuchArch"):
            _check(_cfg({"binds": "layer", "point": "hook_mlp_out"}), target=_subject(architecture="NoSuchArch"))


class TestTokenAndDictionaryBindings:
    def test_tokens_need_the_same_tokenizer(self):
        _check(_cfg({"binds": "tokens"}))
        with pytest.raises(StoreResolutionError, match="was tokenized with 'google/gemma-2-2b' and the target uses"):
            _check(_cfg({"binds": "tokens"}), target=_subject(tokenizer_id="gpt2"))
        with pytest.raises(StoreResolutionError, match="the target's tokenizer is not recorded"):
            _check(_cfg({"binds": "tokens"}), target=_subject(tokenizer_id=None))

    def test_features_need_the_stores_dictionaries_on_the_target(self):
        _check(_cfg({"binds": "latent_feature"}), target=_subject(model_id="google/gemma-2-2b-it"))  # not model-bound
        with pytest.raises(StoreResolutionError, match="lacks the store's dictionaries"):
            _check(_cfg({"binds": "latent_feature"}), target=_subject(dictionaries=()))
        with pytest.raises(StoreResolutionError, match="records no latent dictionaries"):
            _check(_cfg({"binds": "latent_feature"}), source=_subject(dictionaries=()))

    def test_a_triple_reports_every_reason_once(self):
        with pytest.raises(StoreResolutionError) as excinfo:
            _check(_cfg(TRIPLE), target=_subject(n_layers=18, tokenizer_id="gpt2", dictionaries=()))
        message = str(excinfo.value)
        assert "(layer)" in message and "(tokens)" in message and "(latent_feature)" in message
        assert message.count("has 26 layers and the target 18") == 1


class TestFrameBindings:
    def test_values_in_an_activation_space_need_the_same_model_and_revision(self):
        _check(_cfg({"binds": "frame"}))
        with pytest.raises(StoreResolutionError, match="in google/gemma-2-2b's activation space and the target is"):
            _check(_cfg({"binds": "frame"}), target=_subject(model_id="google/gemma-2-2b-it"))
        for revision in (None, "other"):
            with pytest.raises(StoreResolutionError, match="not shown to be the same"):
                _check(_cfg({"binds": "frame"}), target=_subject(model_revision=revision))


class TestHookKeyBindings:
    KEY = "blocks.9.hook_resid_pre.hook_sae_acts_post"

    def test_a_key_both_maps_resolve_carries_over(self):
        _check(_cfg({"binds": "hook_key"}), keys={"col": (self.KEY,)})

    def test_a_key_the_target_cannot_resolve_is_refused_naming_the_key(self):
        with pytest.raises(StoreResolutionError, match="blocks.9.hook_resid_pre"):
            _check(_cfg({"binds": "hook_key"}), keys={"col": (self.KEY,)}, target=_subject(architecture="NoSuchArch"))

    def test_a_key_outside_the_vocabulary_is_refused(self):
        with pytest.raises(StoreResolutionError, match="cannot be addressed"):
            _check(_cfg({"binds": "hook_key"}), keys={"col": ("model.layers.9",)})

    def test_hook_keys_need_the_same_depth(self):
        with pytest.raises(StoreResolutionError, match="has 26 layers and the target 12"):
            _check(_cfg({"binds": "hook_key"}), keys={"col": (self.KEY,)}, target=_subject(n_layers=12))


def test_a_column_the_store_records_no_configuration_for_is_refused():
    with pytest.raises(StoreResolutionError, match="records no column configuration for it"):
        check_bindings({"col": _cfg()}, ("col", "extra"), {}, _subject(), _subject())


def test_a_column_that_names_no_site_carries_over_to_any_target():
    _check(_cfg(None), target=_subject(model_id="other/model", n_layers=2, tokenizer_id=None, dictionaries=()))


def _store(col_cfg: dict | None = None, dataset=None):
    """A store of graph triples and token ids, as an op writes one on the subject's model."""
    from datasets import Array2D, Dataset, Features, Sequence, Value

    from interpretune.analysis import AnalysisStore

    if col_cfg is None:
        col_cfg = {
            "active_features": _cfg(TRIPLE, array_shape=(None, 3), sequence_type=False),
            "input_tokens": _cfg({"binds": "tokens"}),
            "loss": ColCfg(datasets_dtype="float32").to_dict(),
        }
    if dataset is None:
        features = Features(
            {
                "active_features": Array2D(shape=(None, 3), dtype="int64"),
                "input_tokens": Sequence(Value("int64")),
                "loss": Sequence(Value("float32")),
            }
        )
        dataset = Dataset.from_dict(
            {"active_features": [[[0, 1, 2]]], "input_tokens": [[1, 2, 3]], "loss": [[0.5]]}, features=features
        )
    store = AnalysisStore(dataset=dataset, it_format_kwargs={"col_cfg": col_cfg})
    store.subject = _subject()
    return store


class TestResolveOn:
    def test_a_compatible_target_gives_a_view_of_the_same_rows(self):
        store = _store()
        target = _subject(model_id="google/gemma-2-2b-it")
        view = store.resolve_on(target)
        assert view.resolved_on == target and store.resolved_on is None
        assert view.subject == store.subject, "the subject stays what the columns were computed on"
        assert [t.tolist() for t in view["input_tokens"]] == [t.tolist() for t in store["input_tokens"]]

    def test_resolution_reads_no_rows(self, monkeypatch):
        """Refused and accepted alike: a store whose rows cannot be read still resolves."""
        store = _store()

        def unreadable(*args, **kwargs):
            raise AssertionError("resolve_on read rows")

        monkeypatch.setattr(type(store.dataset), "__getitem__", unreadable)
        monkeypatch.setattr(type(store.dataset), "__iter__", unreadable)
        with pytest.raises(AssertionError, match="read rows"):  # the control: a read would fail
            store["input_tokens"]
        store.resolve_on(_subject())
        with pytest.raises(StoreResolutionError):
            store.resolve_on(_subject(n_layers=2))

    def test_every_refusal_is_named_and_the_remedy_is_select_columns(self):
        store = _store()
        with pytest.raises(StoreResolutionError) as excinfo:
            store.resolve_on(_subject(n_layers=2, dictionaries=()))
        message = str(excinfo.value)
        assert "'active_features' (layer)" in message and "'active_features' (latent_feature)" in message
        assert "'input_tokens'" not in message  # the tokenizer is the same
        assert "select_columns" in message
        store.select_columns(["input_tokens", "loss"])
        assert store.resolve_on(_subject(n_layers=2, dictionaries=())).resolved_on is not None

    def test_a_streamed_store_resolves_from_its_features(self):
        from datasets import Array2D, Dataset, Features, IterableDataset, Sequence, Value

        features = Features(
            {"active_features": Array2D(shape=(None, 3), dtype="int64"), "input_tokens": Sequence(Value("int64"))}
        )
        rows = Dataset.from_dict(
            {"active_features": [[[0, 1, 2]]], "input_tokens": [[1, 2]]}, features=features
        ).to_iterable_dataset()
        assert isinstance(rows, IterableDataset)
        store = _store(
            {"active_features": _cfg(TRIPLE, array_shape=(None, 3), sequence_type=False), "input_tokens": _cfg()},
            dataset=rows,
        )
        assert store.resolve_on(_subject()).stream is store.stream
        with pytest.raises(StoreResolutionError, match=r"\(layer\)"):
            store.resolve_on(_subject(n_layers=2))

    def test_a_store_written_before_bindings_existed_is_refused(self):
        older = {"active_features": {k: v for k, v in _cfg().items() if k != "site"}}
        with pytest.raises(StoreResolutionError, match="declares no site bindings"):
            _store(older).resolve_on(_subject())

    def test_a_store_without_a_subject_or_column_configurations_is_refused(self):
        store = _store()
        store.subject = None
        with pytest.raises(StoreResolutionError, match="states no subject"):
            store.resolve_on(_subject())
        store = _store()
        store.it_format_kwargs = None
        with pytest.raises(StoreResolutionError, match="declares no site bindings"):
            store.resolve_on(_subject())

    def test_a_module_target_is_read_as_its_subject(self):
        config = SimpleNamespace(
            to_dict=dict, architectures=[GEMMA2], _commit_hash="c5ebcd40", _name_or_path="google/gemma-2-2b"
        )
        config.num_hidden_layers = 26
        module = SimpleNamespace(
            model=SimpleNamespace(config=config, _module=None),
            it_cfg=SimpleNamespace(
                model_name_or_path="google/gemma-2-2b",
                sae_cfgs=None,
                circuit_tracer_cfg=SimpleNamespace(transcoder_set="gemma"),
            ),
            datamodule=SimpleNamespace(tokenizer=SimpleNamespace(name_or_path="google/gemma-2-2b")),
        )
        assert _store().resolve_on(module).resolved_on.n_layers == 26
        config.num_hidden_layers = 2
        with pytest.raises(StoreResolutionError, match="has 26 layers and the target 2"):
            _store().resolve_on(module)


class TestThroughTheEnvelope:
    def test_bindings_are_published_with_the_col_cfg_and_resolve_after_a_round_trip(self):
        from interpretune.hub.artifacts import build_analysis_store_envelope

        envelope = json.loads(json.dumps(build_analysis_store_envelope(_store())))
        assert envelope["schema"] == 3
        assert envelope["subject"]["model"]["n_layers"] == 26
        store = _store(envelope["interpretune"]["col_cfg"])
        store.subject = StoreSubject.from_dict(envelope["subject"])
        assert store.resolve_on(_subject()).resolved_on is not None
        with pytest.raises(StoreResolutionError, match="has 26 layers and the target 2"):
            store.resolve_on(_subject(n_layers=2))

    def test_the_frozen_schema_3_envelope_carries_bindings_and_depth(self):
        path = Path(__file__).parent / "fixtures" / "it_artifact_schema3.json"
        envelope = json.loads(path.read_text(encoding="utf-8"))
        assert declares_bindings(envelope["interpretune"]["col_cfg"])
        assert StoreSubject.from_dict(envelope["subject"]).n_layers == 26


class TestBundledOpDeclarations:
    """The bundled ops' output columns that name a site declare it, so the next store can be read across models."""

    @staticmethod
    def _schemas():
        from interpretune.analysis import DISPATCHER

        for name in sorted(DISPATCHER.registered_ops):
            op = DISPATCHER.get_op(name)
            yield name, op.output_schema

    def test_columns_keyed_by_hook_declare_a_hook_key(self):
        for op_name, schema in self._schemas():
            for column, cfg in schema.items():
                if cfg.per_latent_model_hook and not cfg.intermediate_only:
                    kinds = {b.binds for b in cfg.site or ()}
                    assert kinds == {"hook_key", "latent_feature"}, (op_name, column)

    def test_declared_columns_and_kinds(self):
        """The inventory of #1008, as declarations: each named column declares what it binds."""
        expected = {
            "active_features": {"layer", "tokens", "latent_feature"},
            "node_feature_ids": {"layer", "tokens", "latent_feature"},
            "top_feature_ids": {"layer", "tokens", "latent_feature"},
            "jlens_layer": {"layer"},
            "head_loading_layers": {"layer"},
            "intervention_layers": {"layer"},
            "input_tokens": {"tokens"},
            "jlens_positions": {"tokens"},
            "intervention_feature_ids": {"latent_feature"},
            "concept_direction": {"frame"},
        }
        seen: dict[str, set[str]] = {}
        for _, schema in self._schemas():
            for column, cfg in schema.items():
                if column in expected:
                    seen.setdefault(column, set()).update(b.binds for b in cfg.site or ())
        assert seen == expected

    def test_a_triples_layer_slot_names_its_point(self):
        schema = dict(self._schemas())["graph_node_influence"]
        assert schema["node_feature_ids"].site[0] == SiteBinding("layer", 0, "hook_mlp_out")
