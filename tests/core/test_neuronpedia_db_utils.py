from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# neuronpedia_utils is a local-checkout package (neuronpedia repo, utils/neuronpedia-utils);
# skip the whole module in envs without it (e.g. plain CI) — db_utils itself stays importable
# there via its structural fallbacks, but these tests compare against the real package.
upstream_db_import = pytest.importorskip("neuronpedia_utils.local_db_import")
db_utils = importlib.import_module("interpretune.utils.neuronpedia_db_utils")


def test_resolve_local_neuronpedia_db_url_returns_the_url_it_is_given() -> None:
    """The URL is used as given: no host or port rewriting, whatever the environment holds."""
    url = "postgresql://np_user:secret@postgres:5432/postgres"
    assert db_utils.resolve_local_neuronpedia_db_url(url, env={"POSTGRES_HOST_PORT": "5433"}) == url


def test_resolve_local_neuronpedia_db_url_takes_the_first_set_env_var() -> None:
    resolved = db_utils.resolve_local_neuronpedia_db_url(
        env={
            "POSTGRES_URL_NON_POOLING": "postgresql://np_user:secret@127.0.0.1:5432/second",
            "DATABASE_URL": "postgresql://np_user:secret@127.0.0.1:5432/third",
        }
    )
    assert resolved == "postgresql://np_user:secret@127.0.0.1:5432/second"
    resolved = db_utils.resolve_local_neuronpedia_db_url(
        env={
            "LOCAL_NEURONPEDIA_DB_URL": "postgresql://np_user:secret@127.0.0.1:5432/first",
            "POSTGRES_URL_NON_POOLING": "postgresql://np_user:secret@127.0.0.1:5432/second",
        }
    )
    assert resolved == "postgresql://np_user:secret@127.0.0.1:5432/first"


def test_resolve_local_neuronpedia_db_url_requires_candidate() -> None:
    with pytest.raises(db_utils.NeuronpediaDBError, match="Could not resolve a local Neuronpedia DB URL"):
        db_utils.resolve_local_neuronpedia_db_url(env={})


def test_summary_helpers_are_neuronpedia_importer_reexports() -> None:
    assert db_utils.summarize_neuronpedia_export_bundle is upstream_db_import.summarize_neuronpedia_export_bundle
    assert (
        db_utils.summarize_neuronpedia_export_bundle_parquet
        is upstream_db_import.summarize_neuronpedia_export_bundle_parquet
    )
    assert (
        db_utils.compare_neuronpedia_export_bundle_to_import_summary
        is upstream_db_import.compare_neuronpedia_export_bundle_to_import_summary
    )


def test_import_neuronpedia_export_bundle_local_db_resolves_url_before_neuronpedia_importer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()
    upstream_calls: list[dict[str, Any]] = []

    def _fake_import(*args: Any, **kwargs: Any) -> SimpleNamespace:
        upstream_calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(imported_row_counts={"Activation": 2})

    monkeypatch.setattr(db_utils, "_import_neuronpedia_export_bundle_local_db", _fake_import)

    summary = db_utils.import_neuronpedia_export_bundle_local_db(
        export_root,
        local_db_url="postgresql://np_user:secret@postgres:5432/postgres",
        prefer_arrow_for_tables=("Activation",),
        prefer_copy_for_tables=("Activation",),
        artifact_format_by_table={"Activation": "arrow"},
    )

    assert summary.imported_row_counts == {"Activation": 2}
    assert upstream_calls == [
        {
            "args": (export_root,),
            "kwargs": {
                "local_db_url": "postgresql://np_user:secret@postgres:5432/postgres",
                "prefer_arrow_for_tables": ("Activation",),
                "prefer_copy_for_tables": ("Activation",),
                "artifact_format_by_table": {"Activation": "arrow"},
            },
        }
    ]


def test_benchmark_neuronpedia_export_bundle_local_db_modes_resolves_url_before_neuronpedia_importer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_root = tmp_path / "bundle"
    export_root.mkdir()
    import_modes = {"activation_arrow_copy": db_utils.DEFAULT_IMPORT_MODE_CONFIGS["activation_arrow_copy"]}
    upstream_calls: list[dict[str, Any]] = []

    def _fake_benchmark(*args: Any, **kwargs: Any) -> dict[str, SimpleNamespace]:
        upstream_calls.append({"args": args, "kwargs": kwargs})
        return {"activation_arrow_copy": SimpleNamespace(imported_row_counts={"Activation": 2})}

    monkeypatch.setattr(db_utils, "_benchmark_neuronpedia_export_bundle_local_db_modes", _fake_benchmark)

    summaries = db_utils.benchmark_neuronpedia_export_bundle_local_db_modes(
        export_root,
        local_db_url="postgresql://np_user:secret@postgres:5432/postgres",
        import_modes=import_modes,
        rollback_each_mode=False,
    )

    assert summaries["activation_arrow_copy"].imported_row_counts == {"Activation": 2}
    assert upstream_calls == [
        {
            "args": (export_root,),
            "kwargs": {
                "local_db_url": "postgresql://np_user:secret@postgres:5432/postgres",
                "import_modes": import_modes,
                "rollback_each_mode": False,
            },
        }
    ]
