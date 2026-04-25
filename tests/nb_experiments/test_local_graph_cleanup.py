from __future__ import annotations

import json
from pathlib import Path

from tests.nb_experiments import local_graph_cleanup


def _write_graph(path: Path, *, slug: str, model_id: str = "gemma-3-1b-it") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": {"slug": slug, "scan": model_id}}),
        encoding="utf-8",
    )


def test_discover_local_graph_artifacts_filters_by_slug_prefix(tmp_path: Path) -> None:
    _write_graph(tmp_path / "graph_artifacts" / "embed" / "keep-one.json", slug="keep-one")
    _write_graph(tmp_path / "graph_artifacts" / "store" / "drop-one.json", slug="drop-one")
    (tmp_path / "graph_artifacts" / "embed" / "invalid.json").write_text("{}", encoding="utf-8")

    artifacts = local_graph_cleanup.discover_local_graph_artifacts(tmp_path, slug_prefix="keep")

    assert [artifact.slug for artifact in artifacts] == ["keep-one"]


def test_cleanup_local_graph_artifacts_deletes_rows_and_files(monkeypatch, tmp_path: Path) -> None:
    first_graph = tmp_path / "graph_artifacts" / "embed" / "first.json"
    second_graph = tmp_path / "graph_artifacts" / "store" / "second.json"
    _write_graph(first_graph, slug="keep-first")
    _write_graph(second_graph, slug="keep-second")

    executed: list[tuple[str, object]] = []

    class _FakeCursor:
        def execute(self, query: str, params) -> None:
            executed.append((query, params))

        def fetchall(self):
            if executed and executed[-1][0].startswith('SELECT slug, "modelId", url'):
                return [
                    (
                        "keep-first",
                        "gemma-3-1b-it",
                        "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/user/keep-first.json",
                    ),
                    (
                        "keep-second",
                        "gemma-3-1b-it",
                        "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/user/keep-second.json",
                    ),
                ]
            return [("keep-first",), ("keep-second",)]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

    class _FakeConnection:
        def __init__(self) -> None:
            self.committed = False

        def cursor(self):
            return _FakeCursor()

        def commit(self) -> None:
            self.committed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

    fake_connection = _FakeConnection()
    monkeypatch.setattr(local_graph_cleanup.psycopg, "connect", lambda db_url: fake_connection)

    summary = local_graph_cleanup.cleanup_local_graph_artifacts(
        tmp_path,
        local_db_url="postgresql://localhost/neuronpedia",
        slug_prefix="keep",
    )

    assert fake_connection.committed is True
    assert summary.deleted_slugs == ("keep-first", "keep-second")
    assert summary.deleted_remote_slugs == ()
    assert all(not path.exists() for path in (first_graph, second_graph))
    assert len(executed) == 3


def test_cleanup_local_graph_artifacts_deletes_remote_payloads_via_local_webapp(monkeypatch, tmp_path: Path) -> None:
    graph_path = tmp_path / "graph_artifacts" / "embed" / "remote.json"
    _write_graph(graph_path, slug="keep-remote")

    executed: list[tuple[str, object]] = []
    remote_calls: list[tuple[str, str, str]] = []

    class _FakeCursor:
        def execute(self, query: str, params) -> None:
            executed.append((query, params))

        def fetchall(self):
            if executed and executed[-1][0].startswith('SELECT slug, "modelId", url'):
                return [
                    (
                        "keep-remote",
                        "gemma-3-1b-it",
                        "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/user/keep-remote.json",
                    )
                ]
            if executed and executed[-1][0].startswith('DELETE FROM public."GraphMetadata"'):
                return []
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

    class _FakeConnection:
        def __init__(self) -> None:
            self.committed = False

        def cursor(self):
            return _FakeCursor()

        def commit(self) -> None:
            self.committed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

    monkeypatch.setattr(local_graph_cleanup.psycopg, "connect", lambda db_url: _FakeConnection())
    monkeypatch.setattr(
        local_graph_cleanup,
        "_delete_remote_graph_payload",
        lambda *, model_id, slug, local_webapp_url, local_api_key: remote_calls.append(
            (model_id, slug, f"{local_webapp_url}|{local_api_key}")
        ),
    )

    summary = local_graph_cleanup.cleanup_local_graph_artifacts(
        tmp_path,
        local_db_url="postgresql://localhost/neuronpedia",
        local_webapp_url="http://localhost:3000",
        local_api_key="dev-key",
    )

    assert summary.deleted_slugs == ("keep-remote",)
    assert summary.deleted_remote_slugs == ("keep-remote",)
    assert summary.skipped_remote_slugs == ()
    assert remote_calls == [("gemma-3-1b-it", "keep-remote", "http://localhost:3000|dev-key")]


def test_cleanup_local_graph_artifacts_dry_run_keeps_files(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph_artifacts" / "embed" / "dry-run.json"
    _write_graph(graph_path, slug="dry-run")

    summary = local_graph_cleanup.cleanup_local_graph_artifacts(tmp_path, dry_run=True)

    assert summary.discovered_slugs == ("dry-run",)
    assert summary.deleted_slugs == ()
    assert summary.deleted_remote_slugs == ()
    assert graph_path.exists()
