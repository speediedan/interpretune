"""Tests for the dashboard-availability probe and the fetch-if-missing helper.

The probe is what the example notebooks call before doing anything else, so its two failure modes
matter more than its happy path: reporting a PARTIAL import as present (which sends a notebook on to
resolve features that do not exist), and raising on an unreachable database (which turns a
recoverable situation into a dead cell).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from interpretune.hub.discovery import DEFAULT_TRUSTED_PUBLISHERS, UntrustedPublisherSpecError
from interpretune.utils.neuronpedia_hub_fetch import (
    CorpusPlan,
    build_import_command,
    dashboards_present,
    default_corpus_dest,
    ensure_dashboards,
    find_dashboard_corpora,
    resolve_bucket_for,
)
from interpretune.utils.neuronpedia_source_conflicts import ConflictPolicy, SourceSetOccupancy

DB = "postgres://user:pw@127.0.0.1:5433/postgres"
MODEL = "gemma-3-1b-it"
SET = "gemmascope-2-transcoder-16k"
RECORDED = json.loads((Path(__file__).parent / "fixtures/dashboard_corpus_publishers.json").read_text(encoding="utf-8"))
PUBLISHER = DEFAULT_TRUSTED_PUBLISHERS[0]
# The buckets the hardcoded stub named before discovery replaced it: resolution must not drift from them.
MONOLOGY = "speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__monology__dashboards"
RTE = "speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__rte__dashboards"


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


def _recorded_hub(extra_buckets: list[dict] | None = None, extra_manifests: dict | None = None):
    """Serve the recorded publishers' buckets and manifests, with optional extra buckets, in place of the Hub."""
    buckets = {publisher: list(rows) for publisher, rows in RECORDED["publishers"].items()}
    for row in extra_buckets or []:
        buckets.setdefault(row["id"].split("/")[0], []).append(row)
    manifests = {**RECORDED["manifests"], **(extra_manifests or {})}
    has_manifest = {row["id"]: row["has_manifest"] for rows in buckets.values() for row in rows}
    listed: list = []

    def list_buckets(self, namespace=None, *, search=None, token=None):
        listed.append(namespace)
        return [SimpleNamespace(id=row["id"]) for row in buckets.get(namespace, [])]

    def get_bucket_paths_info(self, bucket_id, paths, *, token=None):
        return [SimpleNamespace(path=p) for p in paths if has_manifest[bucket_id]]

    return (
        patch.multiple("huggingface_hub.HfApi", list_buckets=list_buckets, get_bucket_paths_info=get_bucket_paths_info),
        patch(
            "interpretune.utils.neuronpedia_hub_fetch.read_remote_manifest",
            lambda bucket, token=None: manifests[bucket],
        ),
        listed,
    )


class TestCorpusDiscovery:
    @pytest.fixture(autouse=True)
    def _default_trust(self, monkeypatch) -> None:
        monkeypatch.delenv("IT_TRUSTED_PUBLISHERS", raising=False)

    def test_existing_pairs_resolve_to_the_buckets_the_stub_named(self) -> None:
        hub, man, _ = _recorded_hub()
        with hub, man:
            assert resolve_bucket_for(MODEL, SET) == MONOLOGY
            assert resolve_bucket_for(MODEL, f"{SET}-rte") == RTE

    def test_listings_carry_what_a_caller_chooses_by(self) -> None:
        hub, man, _ = _recorded_hub()
        with hub, man:
            listings = {c.bucket: c for c in find_dashboard_corpora(MODEL)}
        assert set(listings) == {MONOLOGY, RTE}
        assert listings[MONOLOGY].prompt_corpus == "monology/pile-uncopyrighted"
        assert listings[RTE].prompt_corpus == "aps/super_glue:rte"
        assert {c.layers for c in listings.values()} == {26} and all(
            c.publisher == PUBLISHER for c in listings.values()
        )
        assert listings[MONOLOGY].n_prompts == 24576 and listings[RTE].n_tokens == 319

    def test_unknown_pair_lists_what_the_publishers_serve(self) -> None:
        hub, man, _ = _recorded_hub()
        with hub, man, pytest.raises(LookupError, match="no corpus published by") as excinfo:
            resolve_bucket_for("llama-3", "some-set")
        assert f"{MODEL}/{SET} (monology/pile-uncopyrighted)" in str(excinfo.value)

    def test_several_corpora_for_one_source_set_need_a_prompt_corpus(self) -> None:
        """More than one prompt corpus per source set is normal, so the first match is never taken."""
        other = f"{PUBLISHER}/gemma-3-1b-it__gemmascope-2-transcoder-16k__wikitext__dashboards"
        manifest = json.loads(json.dumps(RECORDED["manifests"][MONOLOGY]))
        manifest["prompt_corpus"].update(hf_dataset_path="Salesforce/wikitext", hf_dataset_config_name="wikitext-103")
        hub, man, _ = _recorded_hub([{"id": other, "has_manifest": True}], {other: manifest})
        with hub, man:
            with pytest.raises(LookupError, match="several corpora serve.*prompt_corpus="):
                resolve_bucket_for(MODEL, SET)
            assert resolve_bucket_for(MODEL, SET, prompt_corpus="monology/pile-uncopyrighted") == MONOLOGY
            assert resolve_bucket_for(MODEL, SET, prompt_corpus="Salesforce/wikitext") == other

    def test_buckets_without_a_manifest_are_not_corpora(self) -> None:
        """A publisher's other buckets are not corpora, whatever they are named."""
        decoy = f"{PUBLISHER}/gemma-3-1b-it__gemmascope-2-transcoder-16k__decoy__dashboards"
        hub, man, _ = _recorded_hub([{"id": decoy, "has_manifest": False}])
        with hub, man:
            assert {c.bucket for c in find_dashboard_corpora()} == {MONOLOGY, RTE}

    def test_only_trusted_publishers_are_listed(self, monkeypatch) -> None:
        """The trust list is the keyword, else the environment variable, else the default, read at call time."""
        stranger = "stranger/gemma-3-1b-it__gemmascope-2-transcoder-16k__other__dashboards"
        manifest = json.loads(json.dumps(RECORDED["manifests"][MONOLOGY]))
        manifest["prompt_corpus"].update(hf_dataset_path="Salesforce/wikitext")
        hub, man, listed = _recorded_hub([{"id": stranger, "has_manifest": True}], {stranger: manifest})
        with hub, man:
            assert {c.bucket for c in find_dashboard_corpora()} == {MONOLOGY, RTE}
            assert {c.bucket for c in find_dashboard_corpora(publishers=["stranger"])} == {stranger}
            monkeypatch.setenv("IT_TRUSTED_PUBLISHERS", f"{PUBLISHER}, stranger")
            assert {c.bucket for c in find_dashboard_corpora()} == {MONOLOGY, RTE, stranger}
            assert resolve_bucket_for(MODEL, SET, prompt_corpus="Salesforce/wikitext") == stranger
        assert listed[:2] == [PUBLISHER, "stranger"]

    def test_a_repo_id_is_refused_as_a_publisher(self) -> None:
        hub, man, _ = _recorded_hub()
        with hub, man, pytest.raises(UntrustedPublisherSpecError, match="not a Hub namespace"):
            find_dashboard_corpora(publishers=[f"{PUBLISHER}/some-collection"])

    @pytest.mark.hf_live
    def test_existing_pairs_resolve_through_the_live_publisher(self) -> None:
        assert resolve_bucket_for(MODEL, SET) == MONOLOGY
        assert resolve_bucket_for(MODEL, f"{SET}-rte") == RTE


class TestEnsureDashboards:
    def test_present_short_circuits_without_fetching(self) -> None:
        with _patched_describe(_occupancy(26, 425984)):
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                assert ensure_dashboards(DB, model_id=MODEL, source_set_id=SET, min_sources=26) is True
        fetch.assert_not_called()

    def test_absent_triggers_a_fetch_of_the_discovered_bucket(self) -> None:
        coll, man, _ = _recorded_hub()
        with _patched_describe(_occupancy(0, 0)), coll, man:
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                assert ensure_dashboards(DB, model_id=MODEL, source_set_id=SET) is False
        assert fetch.call_args.args[0] == MONOLOGY
        assert fetch.call_args.kwargs["db_url"] == DB

    def test_explicit_bucket_overrides_the_registry(self) -> None:
        with _patched_describe(_occupancy(0, 0)):
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                ensure_dashboards(DB, model_id=MODEL, source_set_id=SET, bucket="ns/other")
        assert fetch.call_args.args[0] == "ns/other"

    def test_allow_fetch_off_raises_with_the_command_to_run(self) -> None:
        """Where an unattended multi-GiB download would be a surprise, this is an assertion instead."""
        coll, man, _ = _recorded_hub()
        with _patched_describe(_occupancy(0, 0)), coll, man:
            with patch("interpretune.utils.neuronpedia_hub_fetch.fetch_dashboards") as fetch:
                with pytest.raises(RuntimeError, match="allow_fetch is off") as excinfo:
                    ensure_dashboards(DB, model_id=MODEL, source_set_id=SET, allow_fetch=False)
        fetch.assert_not_called()
        assert "fetch_dashboards_from_hub.py" in str(excinfo.value)

    def test_partial_import_triggers_a_fetch(self) -> None:
        """The probe's completeness rule has to reach the fetch decision, not just the return value."""
        coll, man, _ = _recorded_hub()
        with _patched_describe(_occupancy(12, 196608)), coll, man:
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
