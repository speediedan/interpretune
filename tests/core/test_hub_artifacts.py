"""#124 MVP: artifact envelope, push/pull/describe, and formatter re-attach round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretune.hub.artifacts import (
    ANALYSIS_STORE_KIND,
    ARTIFACT_SCHEMA_MIN_READABLE,
    ARTIFACT_SCHEMA_VERSION,
    IT_ARTIFACT_ENVELOPE,
    ArtifactEnvelopeError,
    build_analysis_store_envelope,
    content_fingerprint,
    describe_analysis_store,
    validate_artifact_envelope,
)


@pytest.fixture()
def small_store():
    """A minimal AnalysisStore with graph-shaped primitive columns and ColCfg format kwargs."""
    from datasets import Array2D, Dataset, Features, Sequence, Value

    from interpretune.analysis import AnalysisStore
    from interpretune.analysis.ops.base import ColCfg

    features = Features(
        {
            "input_string": Value("string"),
            "input_tokens": Sequence(Value("int64")),
            "active_features": Array2D(shape=(None, 3), dtype="int64"),
            "graph_cfg_json": Value("string"),
        }
    )
    dataset = Dataset.from_dict(
        {
            "input_string": ["a premise", "b premise"],
            "input_tokens": [[1, 2, 3], [4, 5]],
            "active_features": [[[0, 1, 2], [1, 2, 3]], [[2, 3, 4]]],
            "graph_cfg_json": ['{"k": 1}', '{"k": 2}'],
        },
        features=features,
    )
    # the formatter contract: col_cfg is passed SERIALIZED (OpSchemaExt runs ColCfg.from_dict)
    col_cfg = {
        "active_features": ColCfg(datasets_dtype="int64", array_shape=(None, 3), sequence_type=False).to_dict(),
        "input_tokens": ColCfg(datasets_dtype="int64").to_dict(),
    }
    return AnalysisStore(dataset=dataset, split="validation", it_format_kwargs={"col_cfg": col_cfg})


class TestEnvelope:
    def test_envelope_shape_and_colcfg_roundtrip(self, small_store):
        from interpretune.analysis.ops.base import ColCfg

        env = build_analysis_store_envelope(small_store)
        validate_artifact_envelope(env)
        assert env["schema"] == ARTIFACT_SCHEMA_VERSION and env["artifact_kind"] == ANALYSIS_STORE_KIND
        assert env["identity"]["store_id"] and env["identity"]["created_utc"]
        assert env["artifacts"]["num_rows"] == 2
        assert env["artifacts"]["interchange"]["format"] == "parquet"
        # col_cfg survives a JSON round-trip including the array_shape list->tuple boundary
        wire = json.loads(json.dumps(env))
        restored = ColCfg.from_dict(wire["interpretune"]["col_cfg"]["active_features"])
        assert restored.array_shape == (None, 3) and restored.sequence_type is False

    def test_identity_preserved_provenance_refreshed(self, small_store):
        env1 = build_analysis_store_envelope(small_store)
        env2 = build_analysis_store_envelope(small_store, identity=env1["identity"], provenance={"note": "re-push"})
        assert env2["identity"] == env1["identity"]  # never rewritten
        assert env2["provenance"]["note"] == "re-push"

    def test_content_fingerprint_tracks_content(self, small_store):
        from datasets import Dataset

        from interpretune.analysis import AnalysisStore

        fp1 = content_fingerprint(small_store)
        assert fp1 == content_fingerprint(small_store)  # deterministic
        changed = AnalysisStore(dataset=Dataset.from_dict({"input_string": ["different"]}), split="validation")
        assert content_fingerprint(changed) != fp1

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ({}, "not an integer version"),
            ({"schema": 1, "artifact_kind": "sorcery"}, "artifact_kind"),
            ({"schema": 1, "artifact_kind": ANALYSIS_STORE_KIND, "identity": {}}, "store_id"),
            (
                {"schema": 1, "artifact_kind": ANALYSIS_STORE_KIND, "identity": {"store_id": "x"}},
                "interpretune",
            ),
        ],
        ids=["no-schema", "bad-kind", "no-store-id", "no-col-cfg-block"],
    )
    def test_envelope_validation_failure_modes(self, mutation, match):
        with pytest.raises(ArtifactEnvelopeError, match=match):
            validate_artifact_envelope(mutation)


class TestSchemaVersionPolicy:
    """#257: hub artifacts outlive the code that wrote them, so the reader accepts a WINDOW."""

    @staticmethod
    def _envelope(schema):
        return {
            "schema": schema,
            "artifact_kind": ANALYSIS_STORE_KIND,
            "identity": {"store_id": "x"},
            "interpretune": {"col_cfg": {}},
        }

    def test_current_schema_reads(self):
        validate_artifact_envelope(self._envelope(ARTIFACT_SCHEMA_VERSION))

    def test_newer_schema_refused_pointing_at_an_upgrade(self):
        """A reader that cannot know what a newer schema means must not guess."""
        with pytest.raises(ArtifactEnvelopeError, match="written by a newer Interpretune"):
            validate_artifact_envelope(self._envelope(ARTIFACT_SCHEMA_VERSION + 1))

    def test_older_than_floor_refused_pointing_at_a_republish(self):
        with pytest.raises(ArtifactEnvelopeError, match="older than the minimum readable schema"):
            validate_artifact_envelope(self._envelope(ARTIFACT_SCHEMA_MIN_READABLE - 1))

    @pytest.mark.parametrize("schema", ["1", 1.0, True, None], ids=["str", "float", "bool", "none"])
    def test_non_integer_schema_refused(self, schema):
        """`True` is an int subclass — an envelope declaring it is malformed, not version 1."""
        with pytest.raises(ArtifactEnvelopeError, match="not an integer version"):
            validate_artifact_envelope(self._envelope(schema))

    def test_floor_does_not_exceed_current(self):
        assert ARTIFACT_SCHEMA_MIN_READABLE <= ARTIFACT_SCHEMA_VERSION

    def test_unknown_keys_tolerated_so_additive_fields_need_no_bump(self):
        """The tolerance IS the policy: additive optional fields ship without a schema bump."""
        envelope = self._envelope(ARTIFACT_SCHEMA_VERSION)
        envelope["a_field_from_a_later_release"] = {"anything": [1, 2, 3]}
        envelope["identity"]["future_identity_field"] = "tolerated"
        validate_artifact_envelope(envelope)

    def test_frozen_schema1_envelope_still_reads(self):
        """A literal v1 envelope pinned in-tree: what actually stops a silent mandatory-field change.

        Every other test builds its envelope with the current writer, so writer and reader drift
        together and neither notices. This one cannot drift.
        """
        fixture = Path(__file__).parent / "fixtures" / "it_artifact_schema1.json"
        envelope = validate_artifact_envelope(json.loads(fixture.read_text(encoding="utf-8")), source=str(fixture))
        assert envelope["schema"] == 1
        # the fields a reader must still find after any future in-window schema work
        assert envelope["identity"]["store_id"] and envelope["artifacts"]["split"] == "validation"
        assert envelope["interpretune"]["analysis_backend"] == "circuit_tracer"
        assert envelope["interpretune"]["col_cfg"]["active_features"]["array_shape"] == [None, 3]


class TestWriterSideGuard:
    """#257: refuse to PUBLISH an envelope this build could not itself read back."""

    def test_build_returns_a_validated_envelope(self, small_store):
        env = build_analysis_store_envelope(small_store)
        assert validate_artifact_envelope(env) is env

    def test_malformed_preserved_identity_refused_before_upload(self, small_store):
        """The re-push path feeds an EXISTING envelope's identity back in; a bad one must not ship."""
        with pytest.raises(ArtifactEnvelopeError, match="store_id"):
            build_analysis_store_envelope(small_store, identity={"created_utc": "2026-08-15T00:00:00+00:00"})


class TestLocalArtifactRoundTrip:
    """Push/pull semantics against a LOCAL HF-cache-layout snapshot (socket-blocked, no hub)."""

    @staticmethod
    def _materialize_local_artifact(store, repo_id: str, cache: Path) -> str:
        """Bridge-style: write the parquet + envelope into the artifacts cache in HF layout."""
        import hashlib

        env = build_analysis_store_envelope(store)
        repo_dir = cache / f"datasets--{repo_id.replace('/', '--')}"
        payload = json.dumps(env, sort_keys=True).encode()
        revision = f"local{hashlib.sha256(payload).hexdigest()[:35]}"
        snapshot = repo_dir / "snapshots" / revision / "data"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.mkdir(parents=True, exist_ok=True)
        store.dataset.to_parquet(str(snapshot / "validation-00000-of-00001.parquet"))
        (snapshot.parent / IT_ARTIFACT_ENVELOPE).write_text(json.dumps(env, indent=1), encoding="utf-8")
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
        return revision

    def test_describe_is_cache_only_and_names_fetch(self, small_store, tmp_path, monkeypatch):
        import socket

        cache = tmp_path / "artifacts"
        with pytest.raises(KeyError, match="pull_analysis_store"):
            describe_analysis_store("someorg/absent", cache_dir=cache)
        self._materialize_local_artifact(small_store, "someorg/store", cache)
        monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        env = describe_analysis_store("someorg/store", cache_dir=cache)
        assert env["artifact_kind"] == ANALYSIS_STORE_KIND

    def test_parquet_roundtrip_reattaches_formatter(self, small_store, tmp_path, monkeypatch):
        """Arrow -> parquet -> load -> AnalysisStore with col_cfg re-attach: tensors come back."""
        import socket

        import torch
        from datasets import load_dataset

        from interpretune.analysis import AnalysisStore

        cache = tmp_path / "artifacts"
        self._materialize_local_artifact(small_store, "someorg/store", cache)
        monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        env = describe_analysis_store("someorg/store", cache_dir=cache)
        repo_dir = cache / "datasets--someorg--store"
        snapshot = repo_dir / "snapshots" / (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
        dataset = load_dataset(str(snapshot / "data"), split="validation")  # split inferred from filename
        store = AnalysisStore(
            dataset=dataset, split="validation", it_format_kwargs={"col_cfg": env["interpretune"]["col_cfg"]}
        )
        row = store.dataset[0]
        assert isinstance(row["active_features"], torch.Tensor)
        assert row["active_features"].shape == (2, 3)
        assert row["input_string"] == "a premise"  # non-tensor survives untouched


class TestGraphHydrationRoundTrip:
    """The gate's core (USER amendment): a REAL graph reproduces from the artifact, no pipeline re-run."""

    def test_graph_store_roundtrips_and_hydrates_via_named_backend(self, tmp_path, monkeypatch):
        import socket

        import torch
        from datasets import Dataset, load_dataset

        pytest.importorskip("circuit_tracer")
        from circuit_tracer.graph import Graph

        from interpretune.analysis import AnalysisStore
        from interpretune.analysis.core import schema_to_features
        from interpretune.analysis.ops.dispatcher import DISPATCHER
        from tests.core.test_analysis_backend_graph_serialization import (
            _FakeModule,
            _graph_batch_from_graph,
            _make_graph,
        )

        graph = _make_graph()
        decomposed = _graph_batch_from_graph(graph)
        module = _FakeModule(graph=graph)
        dataset = Dataset.from_dict(
            {
                "input_string": [decomposed.input_string],
                "adjacency_matrix": [decomposed.adjacency_matrix.tolist()],
                "active_features": [decomposed.active_features.tolist()],
                "selected_features": [decomposed.selected_features.tolist()],
                "activation_values": [decomposed.activation_values.tolist()],
                "logit_target_ids": [decomposed.logit_target_ids.tolist()],
                "logit_target_tokens": [decomposed.logit_target_tokens],
                "logit_probabilities": [decomposed.logit_probabilities.tolist()],
                "input_tokens": [decomposed.input_tokens.tolist()],
                "graph_cfg_json": [decomposed.graph_cfg_json],
                "graph_scan_json": [decomposed.graph_scan_json],
                "graph_vocab_size": [decomposed.graph_vocab_size],
                "graph_metadata": [decomposed.graph_metadata],
            },
            features=schema_to_features(module, schema=DISPATCHER.get_op("compute_attribution_graph").output_schema),
        )
        store = AnalysisStore(
            dataset=dataset, split="validation", it_format_kwargs={"analysis_backend": "circuit_tracer"}
        )

        # envelope carries the backend NAME (instances are not wire-format)
        env = build_analysis_store_envelope(store)
        assert env["interpretune"]["analysis_backend"] == "circuit_tracer"

        # parquet round-trip in HF cache layout, then hydrate CACHE-ONLY with sockets blocked
        cache = tmp_path / "artifacts"
        TestLocalArtifactRoundTrip._materialize_local_artifact(store, "someorg/graph-store", cache)
        monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        env = describe_analysis_store("someorg/graph-store", cache_dir=cache)
        repo_dir = cache / "datasets--someorg--graph-store"
        snapshot = repo_dir / "snapshots" / (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
        dataset = load_dataset(str(snapshot / "data"), split="validation")
        from interpretune.analysis.backends import resolve_analysis_backend

        pulled = AnalysisStore(
            dataset=dataset,
            split="validation",
            it_format_kwargs={"analysis_backend": resolve_analysis_backend(env["interpretune"]["analysis_backend"])},
        )
        restored = pulled[0]["attribution_graph"]
        assert isinstance(restored, Graph)
        assert torch.equal(restored.adjacency_matrix, graph.adjacency_matrix)
        assert torch.equal(restored.active_features, graph.active_features)
        assert restored.input_string == graph.input_string

    def test_store_from_batches_to_envelope_to_hydration(self, tmp_path, monkeypatch):
        """The notebook's exact seam: in-memory pipeline results -> store -> artifact -> hydrated Graph."""
        import socket

        import torch
        from datasets import load_dataset

        pytest.importorskip("circuit_tracer")
        from circuit_tracer.graph import Graph

        from interpretune.analysis import AnalysisStore, analysis_store_from_batches
        from interpretune.analysis.ops.dispatcher import DISPATCHER
        from tests.core.test_analysis_backend_graph_serialization import (
            _FakeModule,
            _graph_batch_from_graph,
            _make_graph,
        )

        graph = _make_graph()
        decomposed = _graph_batch_from_graph(graph)
        module = _FakeModule(graph=graph)
        store = analysis_store_from_batches(
            module,
            [decomposed],
            op=DISPATCHER.get_op("compute_attribution_graph"),
            analysis_backend="circuit_tracer",
        )
        env = build_analysis_store_envelope(store)
        assert env["interpretune"]["analysis_backend"] == "circuit_tracer"  # instance mapped back to its name
        cache = tmp_path / "artifacts"
        TestLocalArtifactRoundTrip._materialize_local_artifact(store, "someorg/nb-seam", cache)
        monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        env = describe_analysis_store("someorg/nb-seam", cache_dir=cache)
        repo_dir = cache / "datasets--someorg--nb-seam"
        snapshot = repo_dir / "snapshots" / (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
        from interpretune.analysis.backends import resolve_analysis_backend

        pulled = AnalysisStore(
            dataset=load_dataset(str(snapshot / "data"), split="validation"),
            split="validation",
            it_format_kwargs={"analysis_backend": resolve_analysis_backend(env["interpretune"]["analysis_backend"])},
        )
        restored = pulled[0]["attribution_graph"]
        assert isinstance(restored, Graph)
        assert torch.equal(restored.adjacency_matrix, graph.adjacency_matrix)

    def test_unresolvable_backend_names_requirement(self):
        from interpretune.analysis.backends import resolve_analysis_backend

        with pytest.raises(KeyError, match="No analysis backend registered"):
            resolve_analysis_backend("no_such_backend")

    def test_unregistered_backend_instance_refused_at_envelope(self, small_store):
        class RogueBackend:
            pass

        small_store.it_format_kwargs = {"analysis_backend": RogueBackend()}
        with pytest.raises(ArtifactEnvelopeError, match="register_analysis_backend"):
            build_analysis_store_envelope(small_store)


def test_artifact_surface_resolves_in_fresh_process():
    """Lesson 5: the surfaces the round-trip notebook will teach must work in a clean process."""
    import subprocess
    import sys

    code = (
        "import interpretune as it; "
        "assert callable(it.hub.push_analysis_store) and callable(it.hub.pull_analysis_store) "
        "and callable(it.hub.describe_analysis_store)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]


class TestWriteTimeOpProvenance:
    """#284: op-collection provenance is RECORDED at write time, never reconstructed at push time.

    The reconstruction path these tests exist to foreclose is not merely inaccurate, it is undetectably so:
    `op_precedence` is session-mutable and re-read from the environment on every access, and the store keeps
    no record of whether a column came from a bare or a fully-qualified name -- which is the only thing that
    would distinguish a safely-derivable case from an unsafe one.
    """

    @staticmethod
    def _hub_def(name: str, **overrides):
        from interpretune.analysis.ops.compiler.cache_manager import OpDef
        from interpretune.analysis.ops.base import OpSchema

        kwargs = dict(
            name=name,
            description="hub op",
            implementation="module.fn",
            input_schema=OpSchema({}),
            output_schema=OpSchema({}),
            source="hub:testuser.repo",
            collection_name="testrepo",
            collection_version="2.1.0",
        )
        kwargs.update(overrides)
        return OpDef(**kwargs)

    def test_bundled_op_records_collection_without_a_revision(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        (record,) = DISPATCHER.op_provenance(DISPATCHER.get_op("labels_to_ids"))
        assert record.source == "bundled"
        assert record.resolved_name == "labels_to_ids"
        # a bundled op has no fetched revision; the key is OMITTED rather than emitted as null so a reader
        # cannot mistake "nothing to fetch" for "lookup failed"
        assert record.revision is None
        assert "revision" not in record.to_dict()

    def test_composite_records_every_constituent(self):
        """A composition can mix collections, so one record per constituent -- a single record would lie."""
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        composite = DISPATCHER.get_op("logit_diffs_base")
        records = DISPATCHER.op_provenance(composite)
        assert [r.resolved_name for r in records] == [op.name for op in composite.composition]

    @pytest.mark.parametrize("op", [None, "unregistered"])
    def test_nothing_to_record_reads_as_absence_not_as_bundled(self, op):
        """The failure mode this guards is a fabricated `bundled`, which is worse than recording nothing."""
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        if op == "unregistered":
            op = type("Unregistered", (), {"name": "definitely_not_a_registered_op"})()
        assert DISPATCHER.op_provenance(op) == ()

    def test_hub_op_records_collection_identity_and_revision(self, monkeypatch):
        from interpretune.analysis.ops import dispatcher as dispatcher_mod
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        dispatcher = AnalysisOpDispatcher()
        dispatcher._op_definitions = {"testuser.repo.hub_op": self._hub_def("testuser.repo.hub_op")}
        dispatcher._loaded = True
        monkeypatch.setattr(dispatcher_mod, "_cached_op_revision", lambda source: "abc123def456")

        (record,) = dispatcher.op_provenance(type("Op", (), {"name": "testuser.repo.hub_op"})())
        assert record.source == "hub:testuser.repo"
        assert (record.collection, record.version) == ("testrepo", "2.1.0")
        assert record.revision == "abc123def456"

    def test_bare_and_qualified_requests_are_distinguishable_after_the_fact(self, monkeypatch):
        """The discriminator whose ABSENCE makes push-time derivation unsound must survive into the record.

        `_preferred_name` re-ranks bare names only, so a qualified name is immune to precedence while a bare
        one is not. Recording the name as written is what lets a reader tell which case a column was.
        """
        from interpretune.analysis.ops import dispatcher as dispatcher_mod
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        dispatcher = AnalysisOpDispatcher()
        hub_def = self._hub_def("testuser.repo.hub_op")
        # bare-name aliasing registers a second key pointing at the SAME OpDef
        dispatcher._op_definitions = {"testuser.repo.hub_op": hub_def, "hub_op": hub_def}
        dispatcher._loaded = True
        monkeypatch.setattr(dispatcher_mod, "_cached_op_revision", lambda source: None)

        by_bare = dispatcher.op_provenance(type("Op", (), {"name": "hub_op"})())[0]
        by_qualified = dispatcher.op_provenance(type("Op", (), {"name": "testuser.repo.hub_op"})())[0]

        assert by_bare.requested_name == "hub_op"
        assert by_qualified.requested_name == "testuser.repo.hub_op"
        # both reach the same definition; only the REQUEST distinguishes them
        assert by_bare.resolved_name == by_qualified.resolved_name == "testuser.repo.hub_op"

    def test_envelope_omits_the_key_entirely_when_the_store_is_unstamped(self, small_store):
        assert small_store.op_provenance == ()
        assert "op_collections" not in build_analysis_store_envelope(small_store)["provenance"]

    def test_envelope_reads_the_stamp_rather_than_a_caller_supplied_dict(self, small_store):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        small_store.op_provenance = DISPATCHER.op_provenance(DISPATCHER.get_op("labels_to_ids"))
        recorded = build_analysis_store_envelope(small_store)["provenance"]["op_collections"]
        assert recorded == [
            {
                "requested_name": "labels_to_ids",
                "resolved_name": "labels_to_ids",
                "source": "bundled",
                "collection": "core",
                "version": "0.1.0",
            }
        ]

    def test_caller_supplied_provenance_still_wins(self, small_store):
        """A store loaded from disk carries no stamp, so the explicit escape hatch must keep working."""
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        small_store.op_provenance = DISPATCHER.op_provenance(DISPATCHER.get_op("labels_to_ids"))
        override = [{"requested_name": "explicit", "resolved_name": "explicit", "source": "local"}]
        envelope = build_analysis_store_envelope(small_store, provenance={"op_collections": override})
        assert envelope["provenance"]["op_collections"] == override

    def test_stamped_envelope_still_validates(self, small_store):
        """Additive key: it must not push the envelope outside what this build can read back."""
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        small_store.op_provenance = DISPATCHER.op_provenance(DISPATCHER.get_op("logit_diffs_base"))
        envelope = build_analysis_store_envelope(small_store)
        assert validate_artifact_envelope(envelope, source="<test>") == envelope
        assert envelope["schema"] == ARTIFACT_SCHEMA_VERSION

    def test_apply_stamps_the_store_it_creates(self):
        """Pins the WIRING, not just the helper: a resolver nothing calls records nothing."""
        from unittest.mock import MagicMock

        from interpretune.analysis.ops.dispatcher import DISPATCHER
        from interpretune.config.analysis import AnalysisCfg

        cfg = AnalysisCfg(target_op=DISPATCHER.get_op("logit_diffs_base"))
        module = MagicMock()
        module.analysis_cfg = cfg
        cfg.apply(module)

        assert [r.resolved_name for r in cfg.output_store.op_provenance] == [
            "labels_to_ids",
            "model_fwd",
            "logit_diffs",
        ]

    def test_apply_stamps_a_caller_supplied_store_too(self):
        """The op writes that store either way, so it is that store's provenance."""
        from unittest.mock import MagicMock

        from interpretune.analysis import AnalysisStore
        from interpretune.analysis.ops.dispatcher import DISPATCHER
        from interpretune.config.analysis import AnalysisCfg

        supplied = AnalysisStore()
        assert supplied.op_provenance == ()
        cfg = AnalysisCfg(target_op=DISPATCHER.get_op("labels_to_ids"), output_store=supplied)
        module = MagicMock()
        module.analysis_cfg = cfg
        cfg.apply(module)

        assert [r.resolved_name for r in supplied.op_provenance] == ["labels_to_ids"]

    def test_recording_provenance_never_fails_a_run(self, recwarn):
        """Provenance is descriptive; a resolver failure must warn, not abort the analysis."""
        from unittest.mock import patch

        from interpretune.analysis import AnalysisStore
        from interpretune.analysis.ops.dispatcher import DISPATCHER
        from interpretune.config.analysis import AnalysisCfg

        cfg = AnalysisCfg(target_op=DISPATCHER.get_op("labels_to_ids"))
        store = AnalysisStore()
        with patch.object(type(DISPATCHER), "op_provenance", side_effect=RuntimeError("boom")):
            cfg._stamp_op_provenance(store)

        assert store.op_provenance == ()
        assert any("Could not record op provenance" in str(w.message) for w in recwarn.list)

    def test_local_source_records_a_label_not_an_address(self):
        """`source` is a CATEGORY.

        Local entries stay distinguishable by collection name but carry no revision and no locator -- documented as a
        limit rather than papered over.
        """
        from interpretune.analysis.ops.base import OpSchema
        from interpretune.analysis.ops.compiler.cache_manager import OpDef
        from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

        def local_def(name, collection):
            return OpDef(
                name=name,
                description="",
                implementation="m.f",
                input_schema=OpSchema({}),
                output_schema=OpSchema({}),
                source="local",
                collection_name=collection,
                collection_version="0.1.0",
            )

        dispatcher = AnalysisOpDispatcher()
        dispatcher._op_definitions = {"a_op": local_def("a_op", "coll_a"), "b_op": local_def("b_op", "coll_b")}
        dispatcher._loaded = True

        records = [dispatcher.op_provenance(type("Op", (), {"name": n})())[0] for n in ("a_op", "b_op")]
        assert {r.source for r in records} == {"local"}
        # distinguishable from each other...
        assert [r.collection for r in records] == ["coll_a", "coll_b"]
        # ...but carrying no revision, so the key is omitted rather than emitted as null
        assert all(r.revision is None and "revision" not in r.to_dict() for r in records)
