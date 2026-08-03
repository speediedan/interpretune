"""Tests for the columnar dashboard Hub publisher.

The page-index tests deliberately write REAL parquet files with and without ``write_page_index``
rather than mocking the footer. The whole point of that check is to detect a property of files
produced by a specific writer configuration, so a mocked footer would test our mock rather than the
behaviour that gates a multi-GiB upload.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from interpretune.utils.neuronpedia_dashboard_hub import (
    COPY_ROWS_STEM,
    DASHBOARD_STORES,
    DEFAULT_STORE,
    HubBucketStore,
    HubDatasetStore,
    download_dashboard_run,
    get_dashboard_store,
    resolve_dashboards_token,
    DashboardPublishPlan,
    MissingPageIndexError,
    PageIndexCoverage,
    build_publish_plan,
    collect_publish_files,
    format_publish_plan,
    page_index_coverage,
    publish_dashboard_run,
    publish_ignore_patterns,
    summarize_by_table,
)

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "publish_dashboards_to_hub.py"


@lru_cache(maxsize=1)
def _publish_cli() -> ModuleType:
    """Load the CLI wrapper by path -- `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("publish_dashboards_to_hub", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_parquet(path: Path, *, page_index: bool, rows: int = 256) -> Path:
    """Write a small parquet file with the page index on or off."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"feature_idx": list(range(rows)), "act": [float(i) / 3 for i in range(rows)]})
    # A small row-group size forces multiple pages, so an offset index is meaningful.
    pq.write_table(table, path, write_page_index=page_index, row_group_size=64)
    return path


def _build_run(root: Path, *, page_index: bool = False, layers: int = 2) -> Path:
    """A miniature stand-in for a real dashboard run tree."""
    for layer in range(layers):
        leaf = root / f"layer_{layer}" / "source" / "batch-0.columnar" / "feature_batch_0"
        for stem in ("activation_rows", COPY_ROWS_STEM, "feature_tables"):
            _write_parquet(leaf / f"{stem}.parquet", page_index=page_index)
        (leaf / "manifest.json").write_text('{"ok": true}', encoding="utf-8")
    (root / "run.log").write_text("host=/home/someone pid=1234\n", encoding="utf-8")
    (root / ".hidden_scratch").write_text("ignore me", encoding="utf-8")
    return root


class TestCollect:
    def test_excludes_copy_rows_and_logs_by_default(self, tmp_path: Path) -> None:
        upload, skip = collect_publish_files(_build_run(tmp_path))
        assert not any(p.stem == COPY_ROWS_STEM for p in upload)
        assert {p.name for p in skip} == {f"{COPY_ROWS_STEM}.parquet", "run.log"}

    def test_include_copy_rows_keeps_them_but_still_drops_logs(self, tmp_path: Path) -> None:
        upload, skip = collect_publish_files(_build_run(tmp_path), include_copy_rows=True)
        assert sum(p.stem == COPY_ROWS_STEM for p in upload) == 2
        # Logs are unconditional -- there is no flag that ships them.
        assert [p.name for p in skip] == ["run.log"]

    def test_hidden_files_are_neither_uploaded_nor_reported(self, tmp_path: Path) -> None:
        upload, skip = collect_publish_files(_build_run(tmp_path))
        names = {p.name for p in upload} | {p.name for p in skip}
        assert ".hidden_scratch" not in names


class TestSummarize:
    def test_parquet_groups_by_stem_others_by_name(self, tmp_path: Path) -> None:
        upload, _ = collect_publish_files(_build_run(tmp_path))
        summary = summarize_by_table(upload)
        # Two layers -> one logical table each, not four separate entries.
        assert summary["activation_rows"][1] == 2
        assert summary["feature_tables"][1] == 2
        assert summary["manifest.json"][1] == 2
        assert all(nbytes > 0 for nbytes, _ in summary.values())


class TestPageIndexCoverage:
    """The write_page_index round-trip: written on, detected on; written off, detected off."""

    def test_detects_page_index_present(self, tmp_path: Path) -> None:
        paths = [_write_parquet(tmp_path / f"a{i}.parquet", page_index=True) for i in range(3)]
        coverage = page_index_coverage(paths)
        assert coverage == PageIndexCoverage(with_index=3, checked=3)
        assert coverage.complete and not coverage.absent

    def test_detects_page_index_absent(self, tmp_path: Path) -> None:
        paths = [_write_parquet(tmp_path / f"a{i}.parquet", page_index=False) for i in range(3)]
        coverage = page_index_coverage(paths)
        assert coverage == PageIndexCoverage(with_index=0, checked=3)
        assert coverage.absent and not coverage.complete

    def test_no_parquet_files_is_unknown_not_absent(self, tmp_path: Path) -> None:
        (txt := tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        coverage = page_index_coverage([txt])
        assert coverage.checked == 0
        # Critical distinction: "unknown" must not trip the upload guard.
        assert not coverage.absent

    def test_sample_bounds_files_opened(self, tmp_path: Path) -> None:
        paths = [_write_parquet(tmp_path / f"a{i}.parquet", page_index=True) for i in range(5)]
        assert page_index_coverage(paths, sample=2) == PageIndexCoverage(with_index=2, checked=2)

    def test_unreadable_footer_does_not_raise(self, tmp_path: Path) -> None:
        good = _write_parquet(tmp_path / "good.parquet", page_index=True)
        (bad := tmp_path / "bad.parquet").write_bytes(b"not a parquet file")
        coverage = page_index_coverage([good, bad])
        assert coverage == PageIndexCoverage(with_index=1, checked=2)


class TestBuildPlan:
    def test_missing_run_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            build_publish_plan(tmp_path / "nope")

    def test_empty_run_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_publish_plan(tmp_path)

    def test_plan_reports_sizes_and_largest(self, tmp_path: Path) -> None:
        plan = build_publish_plan(_build_run(tmp_path))
        assert plan.upload_bytes > 0
        assert plan.skip_bytes > 0
        assert len(plan.copy_rows_skipped) == 2
        assert plan.largest is not None and plan.largest.suffix == ".parquet"


class TestFormatPlan:
    def test_warns_when_page_index_absent(self, tmp_path: Path) -> None:
        report = format_publish_plan(build_publish_plan(_build_run(tmp_path, page_index=False)))
        assert "WARNING: no page index" in report
        assert "regenerating the corpus" in report

    def test_no_warning_when_page_index_present(self, tmp_path: Path) -> None:
        plan = build_publish_plan(_build_run(tmp_path, page_index=True))
        report = format_publish_plan(plan)
        assert "WARNING" not in report
        # 2 layers x (activation_rows + feature_tables); copy_rows are excluded, so not 6.
        assert plan.page_index == PageIndexCoverage(with_index=4, checked=4)
        assert "page index: 4/4" in report

    def test_hint_only_shown_when_flag_would_change_outcome(self, tmp_path: Path) -> None:
        run = _build_run(tmp_path)
        assert "include_copy_rows" in format_publish_plan(build_publish_plan(run))
        # With copy-rows already included, advertising the flag would be nonsense.
        assert "include_copy_rows" not in format_publish_plan(build_publish_plan(run, include_copy_rows=True))

    def test_repo_id_shown_only_when_given(self, tmp_path: Path) -> None:
        plan = build_publish_plan(_build_run(tmp_path))
        assert "repo     : me/mine" in format_publish_plan(plan, repo_id="me/mine")
        assert "repo     :" not in format_publish_plan(plan)


class TestIgnorePatterns:
    def test_patterns_match_what_collect_skips(self, tmp_path: Path) -> None:
        """The reviewed plan and the actual upload must exclude exactly the same files.

        These are enforced by two different mechanisms -- our own walk vs the Hub client's matcher
        -- so a divergence silently uploads something the operator was told was excluded. This
        asserts against ``filter_repo_objects``, the real function ``upload_folder`` uses; matching
        with a hand-rolled fnmatch here would test our assumption rather than the client.

        Regression: ``**/*.log`` does NOT match a root-level ``run.log``, so an earlier pattern set
        reported the run log as excluded and would have uploaded it.
        """
        filter_repo_objects = pytest.importorskip("huggingface_hub.utils").filter_repo_objects
        run = _build_run(tmp_path)
        for include in (False, True):
            upload, skip = collect_publish_files(run, include_copy_rows=include)
            patterns = publish_ignore_patterns(include_copy_rows=include)
            all_rel = [p.relative_to(run).as_posix() for p in sorted(upload + skip)]
            kept = set(filter_repo_objects(all_rel, ignore_patterns=patterns))

            assert kept == {p.relative_to(run).as_posix() for p in upload}, (
                f"upload_folder would ship a different set than the plan reported (patterns={patterns})"
            )
            for path in skip:
                assert path.relative_to(run).as_posix() not in kept

    def test_root_level_log_is_actually_excluded(self, tmp_path: Path) -> None:
        """Guards the specific shape that broke: a log at the run root, not nested under a layer."""
        filter_repo_objects = pytest.importorskip("huggingface_hub.utils").filter_repo_objects
        kept = list(filter_repo_objects(["run.log", "nested/run.log"], ignore_patterns=publish_ignore_patterns()))
        assert kept == []

    def test_patterns_do_not_over_match_similar_names(self) -> None:
        """`*activation_copy_rows.parquet` would wrongly exclude unrelated files; ours must not."""
        filter_repo_objects = pytest.importorskip("huggingface_hub.utils").filter_repo_objects
        candidates = ["a/my_activation_copy_rows.parquet", "a/activation_rows.parquet"]
        assert set(filter_repo_objects(candidates, ignore_patterns=publish_ignore_patterns())) == set(candidates)

    def test_copy_rows_pattern_dropped_when_included(self) -> None:
        assert f"**/{COPY_ROWS_STEM}.parquet" in publish_ignore_patterns()
        assert f"**/{COPY_ROWS_STEM}.parquet" not in publish_ignore_patterns(include_copy_rows=True)
        # Logs stay excluded either way. Bare `*.log`, not `**/*.log` -- see publish_ignore_patterns.
        assert "*.log" in publish_ignore_patterns(include_copy_rows=True)


class TestPublish:
    def test_refuses_without_page_index_and_makes_no_network_call(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.HfApi") as api:
            with pytest.raises(MissingPageIndexError, match="no Parquet page index"):
                publish_dashboard_run(_build_run(tmp_path, page_index=False), "me/mine")
        api.assert_not_called()

    def test_override_allows_publishing_without_page_index(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.HfApi") as api:
            plan = publish_dashboard_run(_build_run(tmp_path, page_index=False), "me/mine", require_page_index=False)
        assert isinstance(plan, DashboardPublishPlan)
        api.return_value.upload_folder.assert_called_once()

    def test_publishes_when_page_index_present(self, tmp_path: Path) -> None:
        with patch("huggingface_hub.HfApi") as api:
            publish_dashboard_run(_build_run(tmp_path, page_index=True), "me/mine", private=True)

        api.return_value.create_repo.assert_called_once()
        assert api.return_value.create_repo.call_args.kwargs["repo_type"] == "dataset"
        assert api.return_value.create_repo.call_args.kwargs["private"] is True

        upload_kwargs = api.return_value.upload_folder.call_args.kwargs
        assert upload_kwargs["repo_type"] == "dataset"
        assert f"**/{COPY_ROWS_STEM}.parquet" in upload_kwargs["ignore_patterns"]

    def test_unknown_coverage_does_not_block_publish(self, tmp_path: Path) -> None:
        """No parquet at all -> coverage unknown -> must not be treated as a missing index."""
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
        with patch("huggingface_hub.HfApi") as api:
            publish_dashboard_run(tmp_path, "me/mine")
        api.return_value.upload_folder.assert_called_once()


class TestCli:
    def test_dry_run_makes_no_network_call(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        main = _publish_cli().main

        with patch("huggingface_hub.HfApi") as api:
            rc = main(["--run-root", str(_build_run(tmp_path)), "--repo-id", "me/mine", "--dry-run"])
        assert rc == 0
        api.assert_not_called()
        assert "DRY RUN" in capsys.readouterr().out

    def test_refusal_exits_two(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        main = _publish_cli().main

        with patch("huggingface_hub.HfApi"):
            rc = main(["--run-root", str(_build_run(tmp_path, page_index=False)), "--repo-id", "me/mine"])
        assert rc == 2
        assert "REFUSING TO PUBLISH" in capsys.readouterr().err

    def test_bad_run_root_exits_one(self, tmp_path: Path) -> None:
        assert _publish_cli().main(["--run-root", str(tmp_path / "nope"), "--repo-id", "me/mine"]) == 1


class TestStoreSelection:
    def test_known_backends(self) -> None:
        assert sorted(DASHBOARD_STORES) == ["bucket", "dataset"]
        assert isinstance(DASHBOARD_STORES["dataset"], HubDatasetStore)
        assert isinstance(DASHBOARD_STORES["bucket"], HubBucketStore)
        assert get_dashboard_store().name == DEFAULT_STORE
        assert get_dashboard_store("BUCKET").name == "bucket"

    def test_unknown_backend_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="unknown dashboard store"):
            get_dashboard_store("s3")


class TestBucketUri:
    def test_prefix_composition(self) -> None:
        assert HubBucketStore.uri("ns/bkt") == "hf://buckets/ns/bkt"
        assert HubBucketStore.uri("ns/bkt", "with-copy-rows") == "hf://buckets/ns/bkt/with-copy-rows"
        # Stray slashes must not produce `//`, which bucket object keys forbid outright.
        assert HubBucketStore.uri("/ns/bkt/", "/pfx/") == "hf://buckets/ns/bkt/pfx"


class TestBackendParity:
    """Both backends must honor the SAME exclusions the plan reported.

    The plan a user reviews and the bytes that land must agree regardless of transport; a backend that quietly shipped
    copy-rows or logs would invalidate the size claims the plan prints.
    """

    @pytest.mark.parametrize("store_name", ["dataset", "bucket"])
    @pytest.mark.parametrize("include_copy_rows", [False, True])
    def test_push_excludes_match_the_plan(self, tmp_path: Path, store_name: str, include_copy_rows: bool) -> None:
        run = _build_run(tmp_path, page_index=True)
        with patch("huggingface_hub.HfApi") as api:
            publish_dashboard_run(run, "ns/name", store=store_name, include_copy_rows=include_copy_rows, token="t")
        api_obj = api.return_value
        if store_name == "dataset":
            patterns = api_obj.upload_folder.call_args.kwargs["ignore_patterns"]
            api_obj.create_repo.assert_called_once()
        else:
            patterns = api_obj.sync_bucket.call_args.kwargs["exclude"]
            api_obj.create_bucket.assert_called_once()
        assert "*.log" in patterns
        assert (f"**/{COPY_ROWS_STEM}.parquet" in patterns) is not include_copy_rows

    @pytest.mark.parametrize("store_name", ["dataset", "bucket"])
    def test_page_index_guard_precedes_any_network_call(self, tmp_path: Path, store_name: str) -> None:
        """An unstreamable corpus is unstreamable wherever it lands, so the guard is backend-independent."""
        with patch("huggingface_hub.HfApi") as api:
            with pytest.raises(MissingPageIndexError):
                publish_dashboard_run(_build_run(tmp_path, page_index=False), "ns/name", store=store_name, token="t")
        api.assert_not_called()

    def test_bucket_push_targets_an_hf_bucket_uri_with_prefix_for_revision(self, tmp_path: Path) -> None:
        """Buckets have no revisions, so a revision must become a path prefix rather than be dropped."""
        with patch("huggingface_hub.HfApi") as api:
            publish_dashboard_run(
                _build_run(tmp_path, page_index=True), "ns/bkt", store="bucket", revision="with-copy-rows", token="t"
            )
        kwargs = api.return_value.sync_bucket.call_args.kwargs
        assert kwargs["dest"] == "hf://buckets/ns/bkt/with-copy-rows"
        # Additive by default: bucket deletions are immediate and permanent.
        assert kwargs.get("delete") is False

    @pytest.mark.parametrize("store_name", ["dataset", "bucket"])
    def test_download_round_trips_through_the_backend(self, tmp_path: Path, store_name: str) -> None:
        dest = tmp_path / "dl"
        with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as snap:
            snap.return_value = str(dest)
            download_dashboard_run("ns/name", dest, store=store_name, token="t")
        if store_name == "dataset":
            snap.assert_called_once()
        else:
            assert api.return_value.sync_bucket.call_args.kwargs["source"] == "hf://buckets/ns/name"


class TestTokenResolution:
    def test_process_env_wins_over_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("HF_HUB_DASHBOARDS_TOKEN=from_file\n", encoding="utf-8")
        monkeypatch.setenv("HF_HUB_DASHBOARDS_TOKEN", "from_env")
        assert resolve_dashboards_token(env_file) == "from_env"

    def test_falls_back_to_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("HF_HUB_DASHBOARDS_TOKEN=from_file\n", encoding="utf-8")
        monkeypatch.delenv("HF_HUB_DASHBOARDS_TOKEN", raising=False)
        assert resolve_dashboards_token(env_file) == "from_file"

    def test_missing_token_returns_none_rather_than_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """None keeps the ambient huggingface-cli credential working for anyone who has not adopted the scoped
        one."""
        monkeypatch.delenv("HF_HUB_DASHBOARDS_TOKEN", raising=False)
        assert resolve_dashboards_token(tmp_path / "absent.env") is None
