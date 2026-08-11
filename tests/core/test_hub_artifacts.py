"""#124 MVP: artifact envelope, push/pull/describe, and formatter re-attach round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpretune.hub.artifacts import (
    ANALYSIS_STORE_KIND,
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
            ({}, "version header"),
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
