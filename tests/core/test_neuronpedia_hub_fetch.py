"""Tests for the dashboard-availability probe and the fetch-if-missing helper.

The probe is what the example notebooks call before doing anything else, so its two failure modes
matter more than its happy path: reporting a PARTIAL import as present (which sends a notebook on to
resolve features that do not exist), and raising on an unreachable database (which turns a
recoverable situation into a dead cell).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from interpretune.utils.neuronpedia_hub_fetch import (
    KNOWN_DASHBOARD_BUCKETS,
    CorpusPlan,
    build_import_command,
    dashboards_present,
    default_corpus_dest,
    ensure_dashboards,
    resolve_bucket_for,
)
from interpretune.utils.neuronpedia_source_conflicts import ConflictPolicy, SourceSetOccupancy

DB = "postgres://user:pw@127.0.0.1:5433/postgres"
MODEL = "gemma-3-1b-it"
SET = "gemmascope-2-transcoder-16k"


def _occupancy(sources: int, neurons: int, explanations: int = 0) -> SourceSetOccupancy:
    return SourceSetOccupancy(
        model_id=MODEL,
        source_set_id=SET,
        source_count=sources,
        neuron_count=neurons,
        explanation_count=explanations,
    )


def _patched_describe(result):
    target = "interpretune.utils.neuronpedia_hub_fetch.describe_source_set"
    if isinstance(result, Exception):
        return patch(target, side_effect=result)
    return patch(target, return_value=result)


class TestDashboardsPresent:
    def test_complete_set_is_present(self) -> None:
        with _patched_describe(_occupancy(26, 425984)):
            assert dashboards_present(DB, model_id=MODEL, source_set_id=SET, min_sources=26)

    def test_partial_import_is_not_present(self) -> None:
        """A half-finished import must not read as present.

        26 layers were requested and 12 landed; treating that as available sends the caller on to resolve features in
        layers that were never imported.
        """
        with _patched_describe(_occupancy(12, 196608)):
            assert not dashboards_present(DB, model_id=MODEL, source_set_id=SET, min_sources=26)

    def test_sources_without_neurons_is_not_present(self) -> None:
        """Source rows alone are scaffolding; a caller needs the neurons."""
        with _patched_describe(_occupancy(26, 0)):
            assert not dashboards_present(DB, model_id=MODEL, source_set_id=SET, min_sources=26)

    def test_empty_is_not_present(self) -> None:
        with _patched_describe(_occupancy(0, 0)):
            assert not dashboards_present(DB, model_id=MODEL, source_set_id=SET)

    def test_min_sources_defaults_to_any_populated_set(self) -> None:
        with _patched_describe(_occupancy(1, 16384)):
            assert dashboards_present(DB, model_id=MODEL, source_set_id=SET)

    def test_unreachable_database_reports_absent_rather_than_raising(self) -> None:
        """The subsequent import reports the connection failure properly; this must not pre-empt it."""
        with _patched_describe(OSError("connection refused")):
            assert not dashboards_present(DB, model_id=MODEL, source_set_id=SET)


class TestBucketResolution:
    def test_known_pairs_resolve(self) -> None:
        assert resolve_bucket_for(MODEL, SET) == KNOWN_DASHBOARD_BUCKETS[(MODEL, SET)]
        assert "monology" in resolve_bucket_for(MODEL, SET)
        assert "rte" in resolve_bucket_for(MODEL, f"{SET}-rte")

    def test_unknown_pair_lists_what_is_available(self) -> None:
        with pytest.raises(LookupError, match="no published corpus is registered") as excinfo:
            resolve_bucket_for("llama-3", "some-set")
        assert "gemma-3-1b-it" in str(excinfo.value), "the error must name the pairs that do work"


class TestEnsureDashboards:
    def test_present_short_circuits_without_fetching(self) -> None:
        with _patched_describe(_occupancy(26, 425984)):
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                assert ensure_dashboards(DB, model_id=MODEL, source_set_id=SET, min_sources=26) is True
        fetch.assert_not_called()

    def test_absent_triggers_a_fetch_of_the_registered_bucket(self) -> None:
        with _patched_describe(_occupancy(0, 0)):
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                assert ensure_dashboards(DB, model_id=MODEL, source_set_id=SET) is False
        assert fetch.call_args.args[0] == KNOWN_DASHBOARD_BUCKETS[(MODEL, SET)]
        assert fetch.call_args.kwargs["db_url"] == DB

    def test_explicit_bucket_overrides_the_registry(self) -> None:
        with _patched_describe(_occupancy(0, 0)):
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                ensure_dashboards(DB, model_id=MODEL, source_set_id=SET, bucket="ns/other")
        assert fetch.call_args.args[0] == "ns/other"

    def test_allow_fetch_off_raises_with_the_command_to_run(self) -> None:
        """Where an unattended multi-GiB download would be a surprise, this is an assertion instead."""
        with _patched_describe(_occupancy(0, 0)):
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                with pytest.raises(RuntimeError, match="allow_fetch is off") as excinfo:
                    ensure_dashboards(DB, model_id=MODEL, source_set_id=SET, allow_fetch=False)
        fetch.assert_not_called()
        assert "fetch_dashboards_from_hub.py" in str(excinfo.value)

    def test_partial_import_triggers_a_fetch(self) -> None:
        """The probe's completeness rule has to reach the fetch decision, not just the return value."""
        with _patched_describe(_occupancy(12, 196608)):
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                ensure_dashboards(DB, model_id=MODEL, source_set_id=SET, min_sources=26)
        fetch.assert_called_once()


class TestImportCommand:
    def _plan(self, tmp_path: Path) -> CorpusPlan:
        return CorpusPlan(
            bucket="ns/b",
            model_id=MODEL,
            source_set_id=SET,
            n_prompts=24576,
            n_tokens=128,
            layers=26,
            page_index=True,
            config=tmp_path / "cfg.yaml",
            run_dir=tmp_path / "corpora" / f"{MODEL}_{SET}",
        )

    def test_run_root_is_the_parent_of_the_corpus(self, tmp_path: Path) -> None:
        """The pipeline derives the run directory itself, so it needs the parent, not the corpus."""
        command = build_import_command(self._plan(tmp_path), db_url=DB)
        assert f"--run-root={tmp_path / 'corpora'}" in command

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"policy": ConflictPolicy.RENAME}, "--autosuffix-on-exists"),
            ({"policy": ConflictPolicy.OVERWRITE}, "--overwrite-existing"),
            ({"rename_suffix": "mine"}, "--rename-suffix=mine"),
        ],
    )
    def test_conflict_policy_is_passed_through(self, tmp_path: Path, kwargs: dict, expected: str) -> None:
        assert expected in build_import_command(self._plan(tmp_path), db_url=DB, **kwargs)

    def test_explicit_suffix_wins_over_autosuffix(self, tmp_path: Path) -> None:
        command = build_import_command(
            self._plan(tmp_path), db_url=DB, policy=ConflictPolicy.RENAME, rename_suffix="mine"
        )
        assert "--rename-suffix=mine" in command
        assert "--autosuffix-on-exists" not in command

    def test_default_policy_adds_no_conflict_flag(self, tmp_path: Path) -> None:
        command = build_import_command(self._plan(tmp_path), db_url=DB)
        assert not any("exists" in part or "rename" in part or "overwrite" in part for part in command)


class TestDefaultDest:
    def test_prefers_it_np_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IT_NP_CACHE", "/np")
        assert default_corpus_dest() == Path("/np/hub_downloads")

    def test_falls_back_to_hf_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("IT_NP_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", "/hf")
        assert default_corpus_dest() == Path("/hf/interpretune/neuronpedia/hub_downloads")
