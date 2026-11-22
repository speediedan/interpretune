"""Site bindings declared per column, carried by a store, and checked against another model by ``resolve_on``."""

from __future__ import annotations

import dataclasses

import pytest

from interpretune.analysis.ops.base import ColCfg, SiteBinding
from interpretune.analysis.site_resolution import StoreResolutionError
from interpretune.hub.subject import StoreSubject

SOURCE = StoreSubject(
    model_id="someorg/base",
    architecture="GPT2LMHeadModel",
    tokenizer_id="someorg/tok",
    dictionaries=({"kind": "circuit_tracer", "transcoder_set": "someorg/tc"},),
    n_layers=12,
)
TUNED = dataclasses.replace(SOURCE, model_id="someorg/tuned")
TRIPLE = [
    {"binds": "layer", "slot": 0, "point": "transcoder.out"},
    {"binds": "tokens", "slot": 1},
    {"binds": "latent_feature", "slot": 2},
]


class TestDeclarations:
    def test_a_site_reads_from_its_serialized_form_and_round_trips(self):
        cfg = ColCfg(datasets_dtype="int64", array_shape=(None, 3), sequence_type=False, site=TRIPLE)
        assert cfg.site == (
            SiteBinding("layer", 0, "transcoder.out"),
            SiteBinding("tokens", 1),
            SiteBinding("latent_feature", 2),
        )
        assert cfg.to_dict()["site"] == TRIPLE
        assert ColCfg.from_dict(cfg.to_dict()) == cfg
        assert hash(ColCfg.from_dict(cfg.to_dict())) == hash(cfg)

    @pytest.mark.parametrize("flag", ["per_latent", "per_latent_model_hook"])
    def test_a_per_hook_column_binds_hook_keys_without_declaring_it(self, flag):
        assert ColCfg(datasets_dtype="float32", **{flag: True}).site == (SiteBinding("hook_key"),)

    def test_a_column_with_no_site_declares_none(self):
        assert ColCfg(datasets_dtype="float32").site is None
        assert ColCfg(datasets_dtype="float32").to_dict()["site"] is None

    @pytest.mark.parametrize(
        "site, match",
        [
            ([{"binds": "depth"}], "not one of"),
            ([{"binds": "layer", "slot": -1}], "non-negative"),
            ([{"binds": "layer", "where": 1}], "unknown keys"),
            (["layer"], "a mapping with `binds`"),
        ],
    )
    def test_a_malformed_site_is_refused(self, site, match):
        with pytest.raises(ValueError, match=match):
            ColCfg(datasets_dtype="int64", site=site)

    @pytest.mark.parametrize(
        "op, column, kinds",
        [
            ("compute_attribution_graph", "active_features", ["layer", "tokens", "latent_feature"]),
            ("compute_attribution_graph", "input_tokens", ["tokens"]),
            ("jlens_read", "jlens_layer", ["layer"]),
            ("jlens_read", "jlens_positions", ["tokens"]),
        ],
    )
    def test_bundled_ops_declare_their_site_bound_columns(self, op, column, kinds):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        site = DISPATCHER.get_op(op).output_schema[column].site
        assert [b.binds for b in site] == kinds

    def test_the_subject_records_depth_and_reads_it_back(self):
        record = SOURCE.to_dict()
        assert record["n_layers"] == 12
        assert StoreSubject.from_dict(record) == SOURCE
        # a version-2 subject written before depth was recorded reads as unknown depth
        assert StoreSubject.from_dict({k: v for k, v in record.items() if k != "n_layers"}).n_layers is None


def _store(col_cfg: dict, data: dict, features, subject=SOURCE):
    from datasets import Dataset

    from interpretune.analysis import AnalysisStore

    store = AnalysisStore(
        dataset=Dataset.from_dict(data, features=features),
        split="validation",
        it_format_kwargs={"col_cfg": {name: cfg.to_dict() for name, cfg in col_cfg.items()}},
    )
    store.subject = subject
    return store


@pytest.fixture()
def graph_store():
    """Triples, tokens and a layer column, as an attribution op writes them."""
    from datasets import Array2D, Features, Sequence, Value

    col_cfg = {
        "active_features": ColCfg(datasets_dtype="int64", array_shape=(None, 3), sequence_type=False, site=TRIPLE),
        "input_tokens": ColCfg(datasets_dtype="int64", site=[{"binds": "tokens"}]),
        "score": ColCfg(datasets_dtype="float32", sequence_type=False),
    }
    features = Features(
        {
            "active_features": Array2D(shape=(None, 3), dtype="int64"),
            "input_tokens": Sequence(Value("int64")),
            "score": Value("float32"),
        }
    )
    data = {"active_features": [[[3, 1, 7]]], "input_tokens": [[5, 6, 7]], "score": [0.5]}
    return _store(col_cfg, data, features)


@pytest.fixture()
def hook_store():
    """A per-hook column whose keys are the hooks it was read at, as the SAE ops write them."""
    from datasets import Features, Sequence, Value

    def make(keys):
        col_cfg = {"alive_latents": ColCfg(datasets_dtype="int64", per_latent_model_hook=True)}
        features = Features({"alive_latents": {k: Sequence(Value("int64")) for k in keys}})
        return _store(col_cfg, {"alive_latents": [{k: [1, 2] for k in keys}]}, features)

    return make


class TestResolveOn:
    def test_a_store_carries_over_to_a_model_sharing_every_binding(self, graph_store):
        resolved = graph_store.resolve_on(TUNED)
        assert resolved.resolved_on == TUNED
        assert resolved.dataset is graph_store.dataset  # values carry over unchanged; nothing is copied
        assert graph_store.resolved_on is None

    @pytest.mark.parametrize(
        "change, columns, reason",
        [
            ({"tokenizer_id": "other/tok"}, ["active_features", "input_tokens"], "same tokenizer"),
            (
                {"dictionaries": ({"kind": "circuit_tracer", "transcoder_set": "other/tc"},)},
                ["active_features"],
                "same dictionaries",
            ),
            ({"n_layers": 24}, ["active_features"], "cross-depth use is refused"),
            ({"n_layers": None}, ["active_features"], "same depth"),
        ],
    )
    def test_a_binding_that_does_not_carry_over_is_refused_naming_its_column(
        self, graph_store, change, columns, reason
    ):
        with pytest.raises(StoreResolutionError, match=reason) as err:
            graph_store.resolve_on(dataclasses.replace(TUNED, **change))
        failing = {line.strip().split(":")[0] for line in str(err.value).splitlines()[1:]}
        assert failing == set(columns)

    def test_an_unknown_source_tokenizer_never_matches(self, graph_store):
        graph_store.subject = dataclasses.replace(SOURCE, tokenizer_id=None)
        with pytest.raises(StoreResolutionError, match="same tokenizer"):
            graph_store.resolve_on(dataclasses.replace(TUNED, tokenizer_id=None))

    def test_a_store_declaring_no_bindings_is_refused(self, graph_store):
        cfgs = graph_store.it_format_kwargs["col_cfg"]
        for cfg in cfgs.values():
            cfg["site"] = None
        with pytest.raises(StoreResolutionError, match="declares no site bindings.*Re-push"):
            graph_store.resolve_on(TUNED)

    def test_a_store_without_a_subject_is_refused(self, graph_store):
        graph_store.subject = None
        with pytest.raises(StoreResolutionError, match="states no subject"):
            graph_store.resolve_on(TUNED)

    def test_hook_keys_that_resolve_on_the_target_carry_over(self, hook_store):
        store = hook_store(["blocks.3.hook_resid_post", "blocks.3.hook_mlp_out.hook_sae_acts_post"])
        assert store.resolve_on(TUNED).resolved_on == TUNED

    def test_a_hook_key_naming_no_point_is_refused(self, hook_store):
        store = hook_store(["blocks.3.hook_resid_post", "blocks.3.not_a_point"])
        with pytest.raises(StoreResolutionError, match=r"alive_latents: hook key 'blocks.3.not_a_point' names no"):
            store.resolve_on(TUNED)

    def test_hook_keys_on_a_target_without_a_component_map_are_refused(self, hook_store):
        store = hook_store(["blocks.3.hook_resid_post"])
        with pytest.raises(StoreResolutionError, match="has no registered component map"):
            store.resolve_on(dataclasses.replace(TUNED, architecture="UnmappedForCausalLM"))

    def test_hook_keys_at_another_depth_are_refused(self, hook_store):
        store = hook_store(["blocks.3.hook_resid_post"])
        with pytest.raises(StoreResolutionError, match="cross-depth use is refused"):
            store.resolve_on(dataclasses.replace(TUNED, n_layers=6))


def test_site_bindings_travel_in_the_envelope_and_reattach_on_pull(graph_store):
    """A pushed envelope carries the declarations; the col_cfg a pull re-attaches rebuilds them."""
    from interpretune.hub.artifacts import build_analysis_store_envelope

    envelope = build_analysis_store_envelope(graph_store)
    assert envelope["interpretune"]["col_cfg"]["active_features"]["site"] == TRIPLE
    assert envelope["subject"]["n_layers"] == 12
    rebuilt = ColCfg.from_dict(envelope["interpretune"]["col_cfg"]["active_features"])
    assert [b.binds for b in rebuilt.site] == ["layer", "tokens", "latent_feature"]
