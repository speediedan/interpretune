from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_artifacts_module() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "dashboard_benchmark_artifacts.py"
    spec = importlib.util.spec_from_file_location("test_dashboard_benchmark_artifacts_mod", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dba = _load_artifacts_module()


def _load_suite_module() -> ModuleType:
    scripts_dir = Path(__file__).parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "run_dashboard_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("test_dashboard_benchmark_suite_mod", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


suite = _load_suite_module()


def _stage_event(stage: str, wall_s: float) -> dict:
    return {"event": "stage_timing", "fields": {"stage": stage, "wall_s": wall_s}}


def _batch_total_event(batch: int, wall_s: float) -> dict:
    return {"event": "batch_total", "fields": {"batch": batch, "wall_s": wall_s}}


def _synthetic_events() -> list[dict]:
    """Two labeled batches of stage events using the preceding-``batch_total`` assignment convention."""

    events: list[dict] = []
    # Events before the first batch_total are unlabeled and dropped (documented behavior).
    events.append(_stage_event("activation_and_encode_total", 99.0))
    events.append(_batch_total_event(0, 30.0))
    # Labeled batch 0 (excluded by warmup=1).
    events.append(_stage_event("activation_and_encode_total", 10.0))
    events.append(_stage_event("rolling_coefficient_update", 0.6))
    events.append(_stage_event("rolling_coefficient_update", 0.4))
    events.append(_batch_total_event(1, 31.0))
    # Labeled batch 1 (steady state).
    events.append(_stage_event("activation_and_encode_total", 12.0))
    events.append(_stage_event("sequence_packaging", 5.0))
    events.append(_stage_event("rolling_coefficient_update", 0.8))
    events.append(_stage_event("rolling_coefficient_update", 0.2))
    events.append(
        {
            "event": "disk_write",
            "fields": {"batch": 1, "wall_s": 0.5, "output_bytes": 2 * 1024**2},
        }
    )
    events.append(_batch_total_event(2, 32.0))
    # Labeled batch 2 (steady state).
    events.append(_stage_event("activation_and_encode_total", 14.0))
    events.append(_stage_event("sequence_packaging", 7.0))
    events.append(_stage_event("rolling_coefficient_update", 0.9))
    events.append(_stage_event("rolling_coefficient_update", 0.3))
    return events


class TestLegsForMode:
    def _args(self, mode: str, config: list[str] | None = None):
        return suite.build_parser().parse_args(["--mode", mode] + [f"--config={c}" for c in (config or [])])

    def test_threeway_mode_has_three_unswept_legs(self):
        legs = suite.legs_for_mode(self._args("threeway"), "rte")
        assert len(legs) == 3
        assert all(config_spec is None for _, _, _, config_spec in legs)

    def test_scaling_mode_defaults_to_conservative_sweep(self):
        legs = suite.legs_for_mode(self._args("scaling"), "rte")
        # 2 paths (detached excluded) x 2 default configs.
        assert len(legs) == 4
        assert {config_spec for _, _, _, config_spec in legs} == set(suite.DEFAULT_SCALING_CONFIGS["rte"])
        assert all(tag != "det" for _, _, tag, _ in legs)

    def test_scaling_mode_explicit_configs_override_defaults(self):
        legs = suite.legs_for_mode(self._args("scaling", ["2048:128"]), "monology")
        assert len(legs) == 2
        assert {config_spec for _, _, _, config_spec in legs} == {"2048:128"}

    def test_full_mode_combines_threeway_and_default_sweep(self):
        legs = suite.legs_for_mode(self._args("full"), "monology")
        threeway = [leg for leg in legs if leg[3] is None]
        swept = [leg for leg in legs if leg[3] is not None]
        assert len(threeway) == 3
        assert len(swept) == 2 * len(suite.DEFAULT_SCALING_CONFIGS["monology"])
        # Swept variant dirs must not collide with threeway variant dirs (config suffix added later).
        assert all(tag != "det" for _, _, tag, _ in swept)


class TestClassifyVariant:
    def test_canonical_detached_legacy(self):
        assert dba.classify_variant("detached-legacy-rte-pretokenized-reduced", "detached_legacy_rte") == (
            "rte",
            "detached_legacy",
        )

    def test_canonical_bare_legacy_is_in_tree_legacy(self):
        assert dba.classify_variant("legacy-monology-pretokenized-reduced", "legacy_monology") == (
            "monology",
            "current_legacy",
        )

    def test_canonical_columnar(self):
        assert dba.classify_variant("columnar-rte-pretokenized-reduced", "columnar_rte") == ("rte", "columnar_gpu")

    def test_old_phase3_legacy_is_detached(self):
        assert dba.classify_variant("phase3-legacy-rte-pretokenized-reduced", "legacy_rte") == (
            "rte",
            "detached_legacy",
        )

    def test_old_current_legacy_monology(self):
        assert dba.classify_variant("phase4-current-legacy-monology-pretokenized-reduced") == (
            "monology",
            "current_legacy",
        )

    def test_old_lazy_maps_to_columnar_gpu(self):
        assert dba.classify_variant("phase3-lazy-rte-pretokenized-reduced", "lazy_rte") == ("rte", "columnar_gpu")

    def test_variant_dir_fallback(self):
        assert dba.classify_variant("some-preset", "current_legacy_monology") == ("monology", "current_legacy")

    def test_unknown_label(self):
        assert dba.classify_variant("unrelated") == (None, None)


class TestAggregateStageTimings:
    def test_last_event_per_batch_semantics(self):
        agg = dba.aggregate_stage_timings(_synthetic_events(), summary_warmup_batches=1)
        # Steady batches are labels 1 and 2: last rolling events are 0.2 and 0.3.
        assert agg.per_batch_last_mean["rolling_coefficient_update"] == pytest.approx(0.25)

    def test_per_batch_sum_and_event_mean(self):
        agg = dba.aggregate_stage_timings(_synthetic_events(), summary_warmup_batches=1)
        # Sums: batch 1 -> 1.0, batch 2 -> 1.2; events: 0.8, 0.2, 0.9, 0.3.
        assert agg.per_batch_sum_mean["rolling_coefficient_update"] == pytest.approx(1.1)
        assert agg.per_event_mean["rolling_coefficient_update"] == pytest.approx(0.55)

    def test_warmup_exclusion(self):
        agg = dba.aggregate_stage_timings(_synthetic_events(), summary_warmup_batches=1)
        # Labeled batch 0 value (10.0) excluded; steady mean of 12.0 and 14.0.
        assert agg.per_batch_last_mean["activation_and_encode_total"] == pytest.approx(13.0)

    def test_all_batches_mode_includes_warmup(self):
        agg = dba.aggregate_stage_timings(_synthetic_events(), summary_warmup_batches=1, timing_mode="all-batches")
        assert agg.per_batch_last_mean["activation_and_encode_total"] == pytest.approx(12.0)

    def test_pre_first_batch_total_events_dropped(self):
        agg = dba.aggregate_stage_timings(_synthetic_events(), summary_warmup_batches=0)
        # The unlabeled 99.0 event must not appear in any aggregate.
        assert agg.per_batch_last_mean["activation_and_encode_total"] == pytest.approx((10.0 + 12.0 + 14.0) / 3)

    def test_batch_total_wall_and_disk_write(self):
        agg = dba.aggregate_stage_timings(_synthetic_events(), summary_warmup_batches=1)
        assert agg.batch_total_wall_s == {0: 30.0, 1: 31.0, 2: 32.0}
        assert agg.disk_write_mean_wall_s == pytest.approx(0.5)
        assert agg.disk_write_mean_output_bytes == pytest.approx(2 * 1024**2)


def _write_synthetic_leaf(
    root: Path,
    variant_dir: str,
    session: str,
    label: str,
    *,
    status: str = "target_reached",
    columnar: bool = False,
) -> Path:
    leaf = root / variant_dir / session / label
    leaf.mkdir(parents=True, exist_ok=True)
    result = {
        "label": label,
        "status": status,
        "n_features_per_batch": 512,
        "n_prompts_in_forward_pass": 128,
        "completed_batches": [0, 1, 2, 3],
        "avg_batch_seconds": 30.0,
        "throughput_features_per_min": 1024.0,
        "max_tree_rss_gib": 20.0,
        "max_host_used_gib": 40.0,
        "max_gpu_process_used_mib": 12000,
        "max_gpu_device_used_mib": 13000,
        "max_gpu_util_percent": 90,
        "run_root": str(leaf / "run_root"),
    }
    (leaf / "result.json").write_text(json.dumps(result), encoding="utf-8")
    events = _synthetic_events()
    events.append(
        {
            "event": "packaging_shape_summary",
            "fields": {"batch": 0, "feature_count": 512, "token_shape": [2488, 319], "valid_token_count": 233201},
        }
    )
    if columnar:
        events.append(
            {
                "event": "columnar_output_summary",
                "fields": {"batch": 0, "row_counts": {"0": {"activation_rows": 15654}}},
            }
        )
    (leaf / "runner_perf_events.json").write_text(json.dumps(events), encoding="utf-8")
    import_profile = {
        "mode": "columnar" if columnar else "legacy_jsonl",
        "wall_seconds": 12.0,
        "conversion_seconds": 0.0 if columnar else 6.0,
        "activation_load_seconds": None if columnar else 7.0,
        "activation_table_load_seconds": 0.7,
        "activation_import_seconds": 8.0,
        "imported_activation_rows": 63081,
        "imported_neuron_rows": 2048,
        "imported_row_counts": {"Activation": 63081, "Neuron": 2048},
    }
    (leaf / "import_stage_profile.json").write_text(json.dumps(import_profile), encoding="utf-8")
    return leaf


class TestDiscoveryAndExtraction:
    def test_find_variant_leaves_picks_latest_successful(self, tmp_path: Path):
        _write_synthetic_leaf(tmp_path, "lazy_rte", "20260101_000000", "phase3-lazy-rte-pretokenized-reduced")
        _write_synthetic_leaf(
            tmp_path, "lazy_rte", "20260102_000000", "phase3-lazy-rte-pretokenized-reduced", status="failed"
        )
        leaves = dba.find_variant_leaves(tmp_path)
        assert len(leaves) == 1
        variant_dir, leaf = leaves[0]
        assert variant_dir == "lazy_rte"
        # Latest session failed, so the earlier successful one is selected.
        assert "20260101_000000" in str(leaf)

    def test_extract_variant_fields(self, tmp_path: Path):
        leaf = _write_synthetic_leaf(
            tmp_path, "lazy_rte", "20260101_000000", "phase3-lazy-rte-pretokenized-reduced", columnar=True
        )
        variant = dba.extract_variant(leaf, variant_dir="lazy_rte")
        assert (variant.scenario, variant.path_key) == ("rte", "columnar_gpu")
        assert variant.avg_batch_seconds == pytest.approx(30.0)
        assert variant.valid_token_count == 233201
        assert variant.imported_activation_rows == 63081
        # activation_load falls back to activation_table_load_seconds when unset.
        assert variant.import_activation_load_seconds == pytest.approx(0.7)
        assert variant.activation_rows_per_batch == {0: 15654}
        # E2E: 512 features x 4 batches x 60 / (30 x 4 + 12).
        assert variant.e2e_features_per_min == pytest.approx(512 * 4 * 60 / (30 * 4 + 12))

    def test_extract_root_returns_all_variants(self, tmp_path: Path):
        _write_synthetic_leaf(tmp_path, "legacy_rte", "20260101_000000", "phase3-legacy-rte-pretokenized-reduced")
        _write_synthetic_leaf(
            tmp_path, "lazy_rte", "20260101_000000", "phase3-lazy-rte-pretokenized-reduced", columnar=True
        )
        variants = dba.extract_root(tmp_path)
        assert {v.path_key for v in variants} == {"detached_legacy", "columnar_gpu"}


def _write_batch_jsons(run_root: Path, rows_per_feature: list[list[int]]) -> None:
    batch_dir = run_root / "layer_9" / "leaf"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for batch_num, feature_rows in enumerate(rows_per_feature):
        payload = {"features": [{"activations": [{}] * count} for count in feature_rows]}
        (batch_dir / f"batch-{batch_num}.json").write_text(json.dumps(payload), encoding="utf-8")


class TestSelectParityPair:
    def _variant(self, tmp_path: Path, name: str, label: str, features: int, prompts: int):
        leaf = _write_synthetic_leaf(tmp_path, name, "20260101_000000", label)
        variant = dba.extract_variant(leaf, variant_dir=name)
        variant.n_features_per_batch = features
        variant.n_prompts_in_forward_pass = prompts
        return variant

    def test_matches_same_config_over_last_seen(self, tmp_path: Path):
        det = self._variant(tmp_path, "legacy_mon", "phase3-legacy-monology-r", 1024, 256)
        cur_base = self._variant(tmp_path, "cur_mon", "phase4-current-legacy-monology-r", 1024, 256)
        cur_swept = self._variant(tmp_path, "cur_mon_2048x256", "phase4-current-legacy-monology-r-2048x256", 2048, 256)
        pair = dba.select_parity_pair([det, cur_swept, cur_base])
        assert pair is not None
        assert pair[0] is det and pair[1] is cur_base

    def test_returns_none_without_config_match(self, tmp_path: Path):
        det = self._variant(tmp_path, "legacy_mon", "phase3-legacy-monology-r", 1024, 256)
        cur_swept = self._variant(tmp_path, "cur_mon_2048x256", "phase4-current-legacy-monology-r-2048x256", 2048, 256)
        assert dba.select_parity_pair([det, cur_swept]) is None


class TestActivationRowParity:
    def _variant_with_batches(self, tmp_path: Path, name: str, rows: list[list[int]]):
        leaf = _write_synthetic_leaf(tmp_path, name, "20260101_000000", f"phase3-legacy-rte-{name}")
        run_root = leaf / "run_root"
        _write_batch_jsons(run_root, rows)
        return dba.extract_variant(leaf, variant_dir=name)

    def test_exact_match(self, tmp_path: Path):
        det = self._variant_with_batches(tmp_path, "det", [[3, 4], [5, 6]])
        cur = self._variant_with_batches(tmp_path, "cur", [[3, 4], [5, 6]])
        parity = dba.activation_row_parity(det, cur)
        assert parity.total_feature_batches == 4
        assert parity.mismatched_feature_batches == 0
        assert parity.match_rate == pytest.approx(1.0)
        assert all(row["match"] for row in parity.per_batch)

    def test_mismatch_detected(self, tmp_path: Path):
        det = self._variant_with_batches(tmp_path, "det", [[3, 4]])
        cur = self._variant_with_batches(tmp_path, "cur", [[3, 5]])
        parity = dba.activation_row_parity(det, cur)
        assert parity.mismatched_feature_batches == 1
        assert parity.per_batch[0]["match"] is False


class TestRendering:
    @pytest.fixture()
    def variants(self, tmp_path: Path):
        leaves = [
            _write_synthetic_leaf(tmp_path, "legacy_rte", "20260101_000000", "phase3-legacy-rte-pretokenized-reduced"),
            _write_synthetic_leaf(
                tmp_path, "lazy_rte", "20260101_000000", "phase3-lazy-rte-pretokenized-reduced", columnar=True
            ),
        ]
        return [dba.extract_variant(leaf, variant_dir=leaf.parents[1].name) for leaf in leaves]

    def test_primary_table(self, variants):
        table = dba.render_primary_table(variants)
        assert "Preserved baseline" in table and "columnar_gpu" in table
        assert "63,081" in table

    def test_substage_table_includes_batch_total_row(self, variants):
        table = dba.render_substage_table(variants)
        assert "batch_total" in table
        assert "30.000" in table  # avg_batch_seconds

    @pytest.fixture()
    def monology_variants(self, tmp_path: Path):
        # the diagram intentionally renders only Monology showcase variants (current_legacy + columnar_gpu)
        leaves = [
            _write_synthetic_leaf(
                tmp_path,
                "current_legacy_monology",
                "20260101_000000",
                "phase3-current-legacy-monology-pretokenized-reduced",
            ),
            _write_synthetic_leaf(
                tmp_path, "lazy_monology", "20260101_000000", "phase3-lazy-monology-pretokenized-reduced", columnar=True
            ),
        ]
        return [dba.extract_variant(leaf, variant_dir=leaf.parents[1].name) for leaf in leaves]

    def test_mermaid_diagram_structure(self, monology_variants, variants):
        diagram = dba.render_mermaid_diagram(monology_variants, lineages={"columnar_gpu": "SD-x/SL-y/NP-z/IT-w"})
        assert diagram.startswith("flowchart TD")
        assert "monology_current_legacy" in diagram and "monology_columnar_gpu" in diagram
        assert "SD-x/SL-y/NP-z/IT-w" in diagram
        assert "columnar_artifact_write" in diagram
        assert "jsonl_gz_conversion_export" in diagram
        assert "PRETOK" in diagram
        # non-Monology variants are intentionally excluded from the showcase diagram
        rte_diagram = dba.render_mermaid_diagram(variants)
        assert "subgraph" not in rte_diagram

    def test_summary_markdown(self, variants):
        manifest = {
            "generated_at_utc": "2026-07-02T00:00:00+00:00",
            "source_root": "/tmp/example",
            "lineages": {"detached_legacy": "det-lineage", "columnar_gpu": "cur-lineage"},
            "dirty_repos": [],
        }
        summary = dba.render_summary_markdown(variants, {}, manifest=manifest, notebook_name="nb.ipynb")
        assert "# Dashboard Benchmark Summary" in summary
        assert "dashboard_benchmark_diagram.mmd" in summary
        assert "nb.ipynb" in summary
        assert "## RTE" in summary

    def test_write_extracted_data_round_trip(self, variants, tmp_path: Path):
        out = dba.write_extracted_data(tmp_path, variants, {})
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload["variants"]) == 2
        assert payload["summary_stages"] == list(dba.SUMMARY_STAGES)
