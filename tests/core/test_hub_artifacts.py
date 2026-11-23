"""Artifact envelope, push/pull/describe, and formatter re-attach round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretune.hub.artifacts import (
    ANALYSIS_STORE_KIND,
    ARTIFACT_SCHEMA_MIN_READABLE,
    ARTIFACT_SCHEMA_VERSION,
    IT_ARTIFACT_ENVELOPE,
    SUBJECT_REQUIRED_FROM_SCHEMA,
    ArtifactEnvelopeError,
    build_analysis_store_envelope,
    content_fingerprint,
    describe_analysis_store,
    validate_artifact_envelope,
)


def _small_store_subject():
    from interpretune.hub.subject import StoreSubject

    return StoreSubject(
        model_id="google/gemma-2-2b",
        model_revision="c5ebcd40d208330abc697524c919956e692655cf",
        architecture="Gemma2ForCausalLM",
        tokenizer_id="google/gemma-2-2b",
        component_map={"architecture": "Gemma2ForCausalLM", "source": "gemma2.yaml", "schema_version": 1},
        dictionaries=({"kind": "circuit_tracer", "transcoder_set": "gemma"},),
    )


SMALL_STORE_SUBJECT = _small_store_subject()


def _fixture(version: int) -> dict:
    """A frozen envelope, as the writer of that schema version produced it."""
    return json.loads((Path(__file__).parent / "fixtures" / f"it_artifact_schema{version}.json").read_text("utf-8"))


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
    store = AnalysisStore(dataset=dataset, split="validation", it_format_kwargs={"col_cfg": col_cfg})
    store.subject = SMALL_STORE_SUBJECT  # as `AnalysisCfg.apply` stamps a store an op writes on a module
    return store


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
    """Hub artifacts outlive the code that wrote them, so the reader accepts a WINDOW."""

    @staticmethod
    def _envelope(schema):
        envelope = {
            "schema": schema,
            "artifact_kind": ANALYSIS_STORE_KIND,
            "identity": {"store_id": "x"},
            "interpretune": {"col_cfg": {}},
        }
        if isinstance(schema, int) and schema >= SUBJECT_REQUIRED_FROM_SCHEMA:
            envelope["subject"] = {"model": {"id": "gpt2"}}
        return envelope

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

    @pytest.mark.parametrize("version", range(ARTIFACT_SCHEMA_MIN_READABLE, ARTIFACT_SCHEMA_VERSION + 1))
    def test_every_readable_version_has_a_frozen_fixture(self, version):
        """Make #283's deferred work fire at the bump instead of relying on someone reading the issue.

        #283 says the per-version fixtures and reader-behavior matrix should land IN the change that bumps
        `ARTIFACT_SCHEMA_VERSION`, "not after it". Nothing enforced that. The sibling test below pins
        schema 1 by name, so a bump to 2 leaves it passing untouched -- schema 1 is still inside the
        window -- while no schema-2 fixture is ever created and the deferred work silently does not happen.

        Deriving the cases from the window closes it structurally rather than by acknowledgement: bumping
        adds a case that fails until its fixture exists, and unlike a pinned literal it cannot be satisfied
        by editing a number. Every version a reader must accept therefore has one artifact that cannot
        drift with the writer.

        Today the window is [1, 1] and this is a single passing case. That is the point -- it costs nothing
        now and cannot be forgotten later.
        """
        fixture = Path(__file__).parent / "fixtures" / f"it_artifact_schema{version}.json"
        assert fixture.is_file(), (
            f"schema {version} is inside the readable window [{ARTIFACT_SCHEMA_MIN_READABLE}, "
            f"{ARTIFACT_SCHEMA_VERSION}] but has no frozen fixture at tests/core/fixtures/{fixture.name}. "
            "Freeze one PRODUCED BY THE WRITER at that version (not hand-written -- a hand-written envelope "
            "tests the validator against a string, which is the drift the fixture exists to prevent), and "
            "do the rest of the deferred work in the same change. See interpretune#283."
        )
        envelope = validate_artifact_envelope(json.loads(fixture.read_text(encoding="utf-8")), source=str(fixture))
        assert envelope["schema"] == version

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
        assert "subject" not in envelope  # version 1 predates the subject: read without one

    def test_frozen_schema2_envelope_reads_with_its_subject(self):
        """The first envelope version that says what a store was computed on, pinned as the writer produced it."""
        from interpretune.hub.subject import StoreSubject

        envelope = validate_artifact_envelope(_fixture(2), source="it_artifact_schema2.json")
        subject = StoreSubject.from_dict(envelope["subject"])
        assert subject.model_id == "google/gemma-2-2b" and subject.architecture == "Gemma2ForCausalLM"
        assert subject.dictionaries == ({"kind": "circuit_tracer", "transcoder_set": "gemma"},)

    def test_a_schema2_envelope_without_a_subject_is_refused(self):
        envelope = _fixture(2)
        del envelope["subject"]
        with pytest.raises(ArtifactEnvelopeError, match="must name the model as `model.id`"):
            validate_artifact_envelope(envelope, source="me/store")


class TestReaderWindowMatrix:
    """The three reader paths the window makes reachable, each against an envelope a writer produced (#283).

    A reader one version behind, and a reader whose floor was raised, are expressed by narrowing this build's bounds, so
    every case reads a frozen artifact rather than a hand-written envelope.
    """

    def test_a_reader_one_version_behind_refuses_the_current_envelope_as_newer(self, monkeypatch):
        monkeypatch.setattr("interpretune.hub.artifacts.ARTIFACT_SCHEMA_VERSION", 1)
        with pytest.raises(ArtifactEnvelopeError, match="written by a newer Interpretune"):
            validate_artifact_envelope(_fixture(2), source="it_artifact_schema2.json")

    def test_an_older_envelope_inside_the_window_reads(self):
        assert ARTIFACT_SCHEMA_MIN_READABLE <= 1 < ARTIFACT_SCHEMA_VERSION
        assert validate_artifact_envelope(_fixture(1), source="it_artifact_schema1.json")["schema"] == 1

    def test_raising_the_floor_refuses_the_older_envelope_pointing_at_a_republish(self, monkeypatch):
        monkeypatch.setattr("interpretune.hub.artifacts.ARTIFACT_SCHEMA_MIN_READABLE", 2)
        with pytest.raises(ArtifactEnvelopeError, match="older than the minimum readable schema"):
            validate_artifact_envelope(_fixture(1), source="it_artifact_schema1.json")


class TestWriterSideGuard:
    """Refuse to PUBLISH an envelope this build could not itself read back."""

    def test_build_returns_a_validated_envelope(self, small_store):
        env = build_analysis_store_envelope(small_store)
        assert validate_artifact_envelope(env) is env

    def test_a_store_without_a_subject_is_refused_naming_the_fix(self, small_store):
        small_store.subject = None
        with pytest.raises(ArtifactEnvelopeError, match=r"records no subject.*subject=StoreSubject"):
            build_analysis_store_envelope(small_store)

    def test_an_explicit_subject_overrides_the_stamped_one(self, small_store):
        from interpretune.hub.subject import StoreSubject

        envelope = build_analysis_store_envelope(small_store, subject=StoreSubject(model_id="gpt2"))
        assert envelope["subject"]["model"] == {"id": "gpt2", "revision": None}

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

    def test_pull_restores_the_subject_onto_the_store(self, small_store, tmp_path, monkeypatch):
        """The subject travels with the store: a pulled store says what it was computed on."""
        from interpretune.hub.artifacts import pull_analysis_store

        cache = tmp_path / "artifacts"
        revision = self._materialize_local_artifact(small_store, "someorg/store", cache)
        snapshot = cache / "datasets--someorg--store" / "snapshots" / revision
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(snapshot / IT_ARTIFACT_ENVELOPE))
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snapshot / "data"))
        pulled = pull_analysis_store("someorg/store", cache_dir=cache)
        assert pulled.subject == SMALL_STORE_SUBJECT

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
                "graph_attribution_regime_json": [decomposed.graph_attribution_regime_json],
                "graph_metadata": [decomposed.graph_metadata],
            },
            features=schema_to_features(module, schema=DISPATCHER.get_op("compute_attribution_graph").output_schema),
        )
        store = AnalysisStore(
            dataset=dataset, split="validation", it_format_kwargs={"analysis_backend": "circuit_tracer"}
        )
        store.subject = SMALL_STORE_SUBJECT  # assembled outside an op, so its subject is stated

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
        store.subject = SMALL_STORE_SUBJECT  # assembled from batches, so its subject is stated
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
    """Op-collection provenance is RECORDED at write time, never reconstructed at push time.

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


class TestProtocolClsThreading:
    """An op's declared protocol reaches the stores it produces, and survives a hub round trip.

    The literal ask -- "pass protocol_cls to the AnalysisStore constructor" -- was already satisfied; the parameter has
    existed since before it was actionable. The real gaps were that nothing PASSED it, and that a pushed store lost its
    protocol on pull because the envelope did not record it.
    """

    @staticmethod
    def _custom_protocol():
        from interpretune.protocol import DefaultAnalysisBatchProtocol

        class CustomBatchProtocol(DefaultAnalysisBatchProtocol):
            pass

        return CustomBatchProtocol

    def test_declaring_op_hands_its_protocol_to_the_store_it_builds(self):
        from unittest.mock import MagicMock

        from interpretune.analysis.ops.dispatcher import DISPATCHER
        from interpretune.config.analysis import AnalysisCfg

        protocol = self._custom_protocol()
        op = DISPATCHER.get_op("labels_to_ids")
        op.protocol_cls = protocol  # as the dispatcher resolver sets it
        try:
            cfg = AnalysisCfg(target_op=op)
            module = MagicMock()
            module.analysis_cfg = cfg
            cfg.apply(module)
            assert cfg.output_store._protocol_cls is protocol
        finally:
            op.protocol_cls = None  # DISPATCHER caches op instances; do not leak into other tests

    def test_undeclared_op_leaves_the_default_in_place(self):
        """Passing None would override the constructor default with nothing."""
        from unittest.mock import MagicMock

        from interpretune.analysis.ops.dispatcher import DISPATCHER
        from interpretune.config.analysis import AnalysisCfg
        from interpretune.protocol import DefaultAnalysisBatchProtocol

        cfg = AnalysisCfg(target_op=DISPATCHER.get_op("get_answer_indices"))
        module = MagicMock()
        module.analysis_cfg = cfg
        cfg.apply(module)
        assert cfg.output_store._protocol_cls is DefaultAnalysisBatchProtocol

    def test_envelope_omits_the_key_for_the_default(self, small_store):
        """An envelope should only carry the key when it says something."""
        assert "protocol_cls" not in build_analysis_store_envelope(small_store)["interpretune"]

    def test_envelope_records_a_custom_protocol_as_a_portable_path(self, small_store):
        protocol = self._custom_protocol()
        small_store._protocol_cls = protocol
        recorded = build_analysis_store_envelope(small_store)["interpretune"]["protocol_cls"]
        assert recorded == f"{protocol.__module__}.{protocol.__qualname__}"

    def test_recorded_protocol_survives_the_round_trip(self, small_store):
        """The gap this issue is really about: a pushed store previously came back with the DEFAULT."""
        from interpretune.hub.artifacts import _resolve_protocol_path
        from interpretune.protocol import DefaultAnalysisBatchProtocol

        small_store._protocol_cls = DefaultAnalysisBatchProtocol
        # record it explicitly so the path is exercised even though it equals the default class
        recorded = "interpretune.protocol.DefaultAnalysisBatchProtocol"
        assert _resolve_protocol_path(recorded) is DefaultAnalysisBatchProtocol

    @pytest.mark.parametrize(
        "path, match",
        [
            ("no_such_module.NotReal", "outside interpretune and the hub components"),
            ("builtins.str", "outside interpretune and the hub components"),
            ("it_hub_components.someone__repo.abc123.Proto", "outside interpretune and the hub components"),
            ("interpretune.no_such_module.Proto", "does not import"),
            ("interpretune.hub.artifacts.ArtifactEnvelopeError", "not a BaseAnalysisBatchProtocol subclass"),
        ],
    )
    def test_an_untrusted_or_unresolvable_protocol_is_refused_by_name(self, path, match):
        """Importing the path runs code, and serving the default instead would silently change the store's
        surface."""
        from interpretune.hub.artifacts import ArtifactEnvelopeError, _resolve_protocol_path

        with pytest.raises(ArtifactEnvelopeError, match=match):
            _resolve_protocol_path(path, source="me/store")

    def test_a_protocol_from_a_loaded_hub_component_resolves_without_an_import(self, monkeypatch):
        """A hub component module is in ``sys.modules`` only once a trusted load executed it."""
        import sys
        import types

        from interpretune.hub.artifacts import _resolve_protocol_path

        protocol = self._custom_protocol()
        name = "it_hub_components.someone__repo.abc123"
        module = types.ModuleType(name)
        module.HubProto = protocol
        monkeypatch.setitem(sys.modules, name, module)
        assert _resolve_protocol_path(f"{name}.HubProto") is protocol


class TestProtocolFieldsInheritance:
    """A store's attribute surface is the protocol's fields over its MRO, not one class's ``__annotations__``."""

    @staticmethod
    def _store(protocol):
        from datasets import Dataset

        from interpretune.analysis import AnalysisStore

        ds = Dataset.from_dict({"prompts": ["a b", "c d"], "preds": [1, 0], "input_string": ["a", "b"]})
        return AnalysisStore(dataset=ds, split="validation", protocol_cls=protocol)

    def test_a_subclass_of_the_default_keeps_the_inherited_fields(self):
        from interpretune.protocol import CircuitAnalysisBatchProtocol

        # the control: these names are inherited, so a reader of the class's own annotations drops them
        assert {"prompts", "preds"}.isdisjoint(CircuitAnalysisBatchProtocol.__annotations__)
        store = self._store(CircuitAnalysisBatchProtocol)
        assert list(store.prompts) == ["a b", "c d"]
        assert store.preds is not None and list(store.input_string) == ["a", "b"]
        assert store.attribution_graphs is None  # declared, not held

    def test_a_subclass_of_the_base_keeps_prompts_and_tokens(self):
        """The base declares ``prompts`` and ``tokens`` so that families need not redeclare them."""
        import torch

        from interpretune.protocol import BaseAnalysisBatchProtocol

        class OwnFields(BaseAnalysisBatchProtocol):
            preds: torch.Tensor | None

        store = self._store(OwnFields)
        assert list(store.prompts) == ["a b", "c d"] and store.tokens is None
        with pytest.raises(AttributeError):
            store.input_string  # held, but not declared

    def test_batch_protocol_fields_collects_over_the_mro(self):
        from interpretune.protocol import (
            CircuitAnalysisBatchProtocol,
            DefaultAnalysisBatchProtocol,
            batch_protocol_fields,
        )

        fields = batch_protocol_fields(CircuitAnalysisBatchProtocol)
        assert set(batch_protocol_fields(DefaultAnalysisBatchProtocol)) < fields
        assert set(CircuitAnalysisBatchProtocol.__annotations__) < fields


class TestSchemaOnlyRead:
    """``pull_analysis_store(..., protocol="schema")`` reads a store whose protocol the reader has not trusted."""

    THIRD_PARTY = "someone_elses_ops.definitions.TheirBatchProtocol"

    @staticmethod
    def _custom_protocol():
        from interpretune.protocol import CircuitAnalysisBatchProtocol

        class TheirBatchProtocol(CircuitAnalysisBatchProtocol):
            their_score: list | None

        return TheirBatchProtocol

    def _publish(self, store, tmp_path, monkeypatch, *, drop_fields: bool = False):
        """Materialize ``store`` as a cached artifact whose envelope names a module this process cannot have
        trusted, and point the hub downloads at it."""
        store._protocol_cls = self._custom_protocol()
        cache = tmp_path / "artifacts"
        revision = TestLocalArtifactRoundTrip._materialize_local_artifact(store, "someone/store", cache)
        snapshot = cache / "datasets--someone--store" / "snapshots" / revision
        env_path = snapshot / IT_ARTIFACT_ENVELOPE
        env = json.loads(env_path.read_text(encoding="utf-8"))
        env["interpretune"]["protocol_cls"] = self.THIRD_PARTY
        if drop_fields:
            del env["interpretune"]["protocol_fields"]  # an envelope written before the field existed
        env_path.write_text(json.dumps(env), encoding="utf-8")
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(env_path))
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snapshot / "data"))
        return cache, env

    def test_the_envelope_records_the_fields_inherited_ones_included(self, small_store):
        small_store._protocol_cls = self._custom_protocol()
        fields = build_analysis_store_envelope(small_store)["interpretune"]["protocol_fields"]
        assert {"their_score", "attribution_graphs", "preds", "prompts"} <= set(fields)
        assert fields == sorted(fields)

    def test_the_default_mode_still_refuses_and_names_the_schema_read(self, small_store, tmp_path, monkeypatch):
        from interpretune.hub.artifacts import ArtifactEnvelopeError, pull_analysis_store

        cache, _ = self._publish(small_store, tmp_path, monkeypatch)
        with pytest.raises(ArtifactEnvelopeError, match="protocol='schema'"):
            pull_analysis_store("someone/store", cache_dir=cache)

    def test_a_schema_read_imports_nothing_and_keeps_the_surface(self, small_store, tmp_path, monkeypatch):
        import sys

        import torch

        from interpretune.hub.artifacts import pull_analysis_store

        cache, env = self._publish(small_store, tmp_path, monkeypatch)
        store = pull_analysis_store("someone/store", cache_dir=cache, protocol="schema")
        assert "someone_elses_ops" not in sys.modules
        proto = store._protocol_cls
        assert proto.recorded_protocol == self.THIRD_PARTY and proto.fields_from == "envelope"
        # the same attribute access a resolved pull gives, a declared column the store lacks included
        assert list(store.input_string) == ["a premise", "b premise"]
        assert store.their_score is None and store.attribution_graphs is None
        # columns decode through col_cfg exactly as before
        assert isinstance(store["active_features"][0], torch.Tensor)
        # re-publishing records the original protocol, not the stand-in
        republished = build_analysis_store_envelope(store)["interpretune"]
        assert republished["protocol_cls"] == self.THIRD_PARTY
        assert republished["protocol_fields"] == env["interpretune"]["protocol_fields"]

    def test_an_envelope_without_fields_gets_the_stores_columns(self, small_store, tmp_path, monkeypatch):
        from interpretune.hub.artifacts import pull_analysis_store

        cache, _ = self._publish(small_store, tmp_path, monkeypatch, drop_fields=True)
        store = pull_analysis_store("someone/store", cache_dir=cache, protocol="schema")
        assert store._protocol_cls.fields_from == "columns"
        assert list(store.input_string) == ["a premise", "b premise"]
        with pytest.raises(AttributeError):
            store.their_score  # declared by the protocol, but nothing recorded says so

    def test_an_unknown_protocol_mode_is_refused(self):
        from interpretune.hub.artifacts import pull_analysis_store

        with pytest.raises(ValueError, match="not a protocol mode"):
            pull_analysis_store("someone/store", protocol="trust")  # pyright: ignore[reportArgumentType]


class TestPushPublishChecks:
    """``push_analysis_store`` against a mocked Hub: name validation, the skip record, and the post-upload read-
    back."""

    PARQUET = "data/validation-00000-of-00001.parquet"

    @pytest.fixture()
    def hub(self, small_store, tmp_path, monkeypatch):
        """A mocked ``HfApi`` plus an ``HfFileSystem`` that serves the store's own parquet as the published
        copy."""
        from unittest.mock import MagicMock

        local = tmp_path / "published.parquet"
        small_store.dataset.to_parquet(str(local))
        opened: list[str] = []

        class _FS:
            def __init__(self, token=None):
                pass

            def size(self, path):
                return local.stat().st_size

            def open(self, path, mode="rb"):
                opened.append(path)
                return local.open(mode)

        api = MagicMock()
        api.hf_hub_download.side_effect = FileNotFoundError("no envelope yet")
        api.list_repo_files.return_value = [self.PARQUET, "README.md"]
        api.repo_info.return_value.sha = "abc123"
        monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: api)
        monkeypatch.setattr("huggingface_hub.HfFileSystem", _FS)
        monkeypatch.setattr(type(small_store.dataset), "push_to_hub", lambda self, *a, **k: None)
        return api, opened, _FS

    @staticmethod
    def _uploaded_envelope(api) -> dict:
        (call,) = [c for c in api.upload_file.call_args_list if c.kwargs["path_in_repo"] == IT_ARTIFACT_ENVELOPE]
        return json.loads(call.kwargs["path_or_fileobj"].getvalue())

    def test_reads_back_the_published_parquet_at_the_pushed_revision(self, small_store, hub):
        from interpretune.hub.artifacts import push_analysis_store

        api, opened, _ = hub
        assert push_analysis_store(small_store, "me/store") == "abc123"
        assert opened == [f"hf://datasets/me/store@abc123/{self.PARQUET}"]
        assert "publish_checks_skipped" not in self._uploaded_envelope(api)["provenance"]

    def test_skip_is_recorded_in_the_envelope(self, small_store, hub):
        from interpretune.hub.artifacts import push_analysis_store

        api, opened, _ = hub
        push_analysis_store(small_store, "me/store", skip_checks={"range-read"})
        assert opened == []
        assert self._uploaded_envelope(api)["provenance"]["publish_checks_skipped"] == ["range-read"]

    def test_unknown_check_name_is_refused_before_any_network_call(self, small_store, hub):
        from interpretune.hub.artifacts import push_analysis_store
        from interpretune.utils.publish_checks import UnknownPublishCheckError

        api, _, _ = hub
        # page-index is a dashboard-corpus check: push_to_hub page-indexes the store's parquet itself
        with pytest.raises(UnknownPublishCheckError, match="page-index"):
            push_analysis_store(small_store, "me/store", skip_checks={"page-index"})
        api.create_repo.assert_not_called()

    def test_a_store_without_a_subject_is_refused_before_any_network_call(self, small_store, hub):
        from interpretune.hub.artifacts import push_analysis_store

        api, _, _ = hub
        small_store.subject = None
        with pytest.raises(ArtifactEnvelopeError, match="records no subject"):
            push_analysis_store(small_store, "me/store")
        api.create_repo.assert_not_called()

    def test_the_published_envelope_carries_the_subject(self, small_store, hub):
        from interpretune.hub.artifacts import push_analysis_store

        api, _, _ = hub
        push_analysis_store(small_store, "me/store")
        assert self._uploaded_envelope(api)["subject"] == SMALL_STORE_SUBJECT.to_dict()

    def test_unreadable_published_copy_raises_after_the_upload(self, small_store, hub, monkeypatch):
        import io

        from interpretune.hub.artifacts import push_analysis_store
        from interpretune.utils.publish_checks import PublishedArtifactCheckError

        api, _, fs = hub

        class _Rewritten(fs):
            def open(self, path, mode="rb"):
                return io.BytesIO(b"PAR1garbage")

        monkeypatch.setattr("huggingface_hub.HfFileSystem", _Rewritten)
        with pytest.raises(PublishedArtifactCheckError, match="was published"):
            push_analysis_store(small_store, "me/store")
        assert api.upload_file.call_count == 2  # envelope and card are already up: the artifact is live


class TestPublishedRowGroups:
    """Row groups are sized for streamed reads, through the one setting ``Dataset.push_to_hub`` reads."""

    def test_the_target_reaches_the_writer_batch_push_to_hub_computes(self, small_store, tmp_path):
        """``push_to_hub`` sizes row groups from ``datasets.config.MAX_ROW_GROUP_SIZE`` at call time; if a
        ``datasets`` release stops reading it there, the target silently stops reaching the files and this
        fails."""
        import pyarrow.parquet as pq
        from datasets import Dataset
        from datasets.arrow_writer import get_writer_batch_size_from_data_size

        from interpretune.hub.artifacts import _row_group_target

        dataset = Dataset.from_dict({"values": [[float(i)] * 256 for i in range(64)]})
        nbytes = dataset._estimate_nbytes()
        default_rows = get_writer_batch_size_from_data_size(len(dataset), nbytes)
        with _row_group_target(nbytes // 8):
            rows = get_writer_batch_size_from_data_size(len(dataset), nbytes)
        assert default_rows >= len(dataset) and rows == len(dataset) // 8
        path = tmp_path / "grouped.parquet"
        dataset.to_parquet(str(path), batch_size=rows)
        assert pq.ParquetFile(path).metadata.num_row_groups == 8

    def test_the_target_is_restored_after_the_block_even_when_it_raises(self):
        import datasets.config

        from interpretune.hub.artifacts import _row_group_target

        before = datasets.config.MAX_ROW_GROUP_SIZE
        with pytest.raises(RuntimeError, match="push failed"):
            with _row_group_target(1024):
                assert datasets.config.MAX_ROW_GROUP_SIZE == 1024
                raise RuntimeError("push failed")
        assert datasets.config.MAX_ROW_GROUP_SIZE == before

    @pytest.mark.parametrize("size, match", [("lots", "is not a size"), (0, "must be a positive"), (-5, "positive")])
    def test_an_unusable_size_is_refused_by_name(self, size, match):
        from interpretune.hub.artifacts import _row_group_bytes

        with pytest.raises(ValueError, match=match):
            _row_group_bytes(size)

    def test_sizes_parse_as_datasets_parses_them(self):
        from interpretune.hub.artifacts import ANALYSIS_STORE_ROW_GROUP_SIZE, _row_group_bytes

        assert _row_group_bytes(ANALYSIS_STORE_ROW_GROUP_SIZE) == 16 * 10**6
        assert _row_group_bytes(4096) == 4096


class TestPushRowGroupTarget:
    """The push applies the target while ``push_to_hub`` runs and records it; refusals land before any network
    call."""

    PARQUET = TestPushPublishChecks.PARQUET
    hub = TestPushPublishChecks.hub
    _uploaded_envelope = staticmethod(TestPushPublishChecks._uploaded_envelope)

    def test_push_to_hub_runs_under_the_target_and_the_envelope_records_it(self, small_store, hub, monkeypatch):
        import datasets.config

        from interpretune.hub.artifacts import push_analysis_store

        api, _, _ = hub
        seen = []
        monkeypatch.setattr(
            type(small_store.dataset),
            "push_to_hub",
            lambda self, *a, **k: seen.append(datasets.config.MAX_ROW_GROUP_SIZE),
        )
        before = datasets.config.MAX_ROW_GROUP_SIZE
        push_analysis_store(small_store, "me/store", row_group_size="4MB")
        assert seen == [4 * 10**6]
        assert datasets.config.MAX_ROW_GROUP_SIZE == before
        interchange = self._uploaded_envelope(api)["artifacts"]["interchange"]
        assert interchange["row_group_target_bytes"] == 4 * 10**6
        assert interchange["files"] == [self.PARQUET]

    def test_an_unusable_size_is_refused_before_any_network_call(self, small_store, hub):
        from interpretune.hub.artifacts import push_analysis_store

        api, _, _ = hub
        with pytest.raises(ValueError, match="is not a size"):
            push_analysis_store(small_store, "me/store", row_group_size="lots")
        api.create_repo.assert_not_called()

    def test_a_streamed_store_is_refused_before_any_network_call(self, small_store, hub):
        from interpretune.analysis import AnalysisStore
        from interpretune.hub.artifacts import push_analysis_store

        api, _, _ = hub
        streamed = AnalysisStore(dataset=small_store.dataset.with_format(None).to_iterable_dataset())
        streamed.subject = SMALL_STORE_SUBJECT
        with pytest.raises(ArtifactEnvelopeError, match="streamed AnalysisStore"):
            push_analysis_store(streamed, "me/store")
        api.create_repo.assert_not_called()
