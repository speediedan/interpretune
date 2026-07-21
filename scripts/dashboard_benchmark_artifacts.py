"""Extraction and rendering library for Neuronpedia dashboard benchmark reviewer artifacts.

This module turns artifact roots produced by ``scripts/profile_neuronpedia_dashboard_generation.py`` into
reviewer-facing assets: primary/substage/import/parity markdown tables, a unified Mermaid flow diagram, and a
machine-readable ``extracted_data.json`` consumed by the parameterized dashboard profiling notebook.

Extraction follows the documented benchmark data extraction process:

- ``stage_timing`` events inherit their batch from the most recent preceding ``batch_total`` event (this labeling
  reproduces the published steady-state substage tables exactly).
- Summary-table substage values use last-event-per-batch semantics (``stages[stage][batch] = wall_s``), matching
  the documented extraction process and all previously published tables. For once-per-batch stages this is the
  per-batch value; for per-minibatch stages (``rolling_coefficient_update``) it is the final (partial) minibatch's
  wall time. Per-event means and per-batch sums are additionally captured for the notebook and diagram.
- The ``batch_total`` summary row uses the profiling harness ``avg_batch_seconds`` (detected batch intervals).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

SCENARIOS = ("rte", "monology")
PATH_KEYS = ("detached_legacy", "current_legacy", "columnar_gpu")
PATH_DISPLAY = {
    "detached_legacy": "Preserved baseline",
    "current_legacy": "legacy",
    "columnar_gpu": "columnar_gpu",
}
SCENARIO_DISPLAY = {"rte": "RTE", "monology": "Monology"}
SUMMARY_STAGES = (
    "activation_and_encode_total",
    "feature_statistics_packaging",
    "logits_histogram_packaging",
    "activation_histogram_packaging",
    "sequence_packaging",
    "rolling_coefficient_update",
)
LEGACY_CONVERSION_STAGE = "neuronpedia_conversion_and_json_serialization"
COLUMNAR_WRITE_STAGES = ("activation_row_packaging", "activation_copy_row_packaging")
ROLLING_SUBSTAGE_PREFIX = "rolling_"
DETACHED_BASELINE_LINEAGE_PREFIX = "SD-7886eaa+benchmark_patches/SL-3eea6552/NP-5a33f17"

_SUCCESS_STATUSES = {"target_reached", "exited"}


@dataclass
class SubstageAggregation:
    """Aggregated stage timings for one variant."""

    per_event_mean: dict[str, float] = field(default_factory=dict)
    per_batch_last_mean: dict[str, float] = field(default_factory=dict)
    per_batch_sum_mean: dict[str, float] = field(default_factory=dict)
    batch_total_wall_s: dict[int, float] = field(default_factory=dict)
    rolling_substage_means: dict[str, float] = field(default_factory=dict)
    disk_write_mean_wall_s: float | None = None
    disk_write_mean_output_bytes: float | None = None


@dataclass
class VariantExtract:
    """Extracted benchmark data for one scenario/path variant leaf."""

    scenario: str
    path_key: str
    label: str
    variant_dir: str
    session_dir: str
    leaf_dir: str
    status: str
    n_features_per_batch: int
    n_prompts_in_forward_pass: int
    completed_batches: list[int]
    summary_warmup_batches: int
    avg_batch_seconds: float | None
    throughput_features_per_min: float | None
    max_tree_rss_gib: float | None
    max_host_used_gib: float | None
    max_gpu_process_used_mib: int | None
    max_gpu_device_used_mib: int | None
    max_gpu_util_percent: int | None
    avg_gpu_util_percent_steady: float | None
    # Total prompt count — the §3d prompt-dimension sweep axis. Parsed from a 3-component config label
    # suffix (features x minibatch x nprompts, e.g. "...-4096x256x8192"); None for feature-axis legs.
    n_prompts_total: int | None = None
    substage_seconds: dict[str, float] = field(default_factory=dict)
    substage_per_batch_sum: dict[str, float] = field(default_factory=dict)
    batch_total_wall_s: dict[int, float] = field(default_factory=dict)
    rolling_substage_means: dict[str, float] = field(default_factory=dict)
    legacy_conversion_seconds: float | None = None
    disk_write_mean_wall_s: float | None = None
    disk_write_mean_output_bytes: float | None = None
    columnar_write_seconds: float | None = None
    import_wall_seconds: float | None = None
    import_conversion_seconds: float | None = None
    import_activation_load_seconds: float | None = None
    import_activation_import_seconds: float | None = None
    imported_activation_rows: int | None = None
    imported_neuron_rows: int | None = None
    imported_row_counts: dict[str, int] = field(default_factory=dict)
    valid_token_count: int | None = None
    token_shape: list[int] | None = None
    activation_rows_per_batch: dict[int, int] = field(default_factory=dict)
    e2e_features_per_min: float | None = None
    lineage: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ParityResult:
    """Activation row parity comparison between two legacy variants."""

    scenario: str
    per_batch: list[dict[str, Any]] = field(default_factory=list)
    total_feature_batches: int = 0
    mismatched_feature_batches: int = 0

    @property
    def match_rate(self) -> float | None:
        if not self.total_feature_batches:
            return None
        return 1.0 - self.mismatched_feature_batches / self.total_feature_batches


def classify_variant(label: str, variant_dir: str = "") -> tuple[str | None, str | None]:
    """Classify a preset label (and fallback variant dir name) into (scenario, path_key).

    Handles both naming eras: the canonical ``detached-legacy-* / legacy-* / columnar-*`` names and the retired
    ``phase3-legacy-* / phase4-current-legacy-* / phase3-lazy-*`` names (where bare ``legacy`` meant the detached
    baseline). Bare ``legacy`` without an era marker is classified as the in-tree legacy path.
    """

    haystacks = (label.lower(), variant_dir.lower())
    scenario = None
    for candidate in SCENARIOS:
        if any(candidate in h for h in haystacks):
            scenario = candidate
            break
    path_key = None
    text = f"{label.lower()} {variant_dir.lower()}"
    if "detached" in text or "phase3-legacy" in text:
        path_key = "detached_legacy"
    elif "current-legacy" in text or "current_legacy" in text:
        path_key = "current_legacy"
    elif "lazy" in text or "columnar" in text:
        path_key = "columnar_gpu"
    elif "legacy" in text:
        path_key = "current_legacy"
    return scenario, path_key


def find_variant_leaves(root: Path) -> list[tuple[str, Path]]:
    """Return (variant_dir_name, leaf_dir) pairs for the latest successful session under each variant dir.

    Layout: ``<root>/<variant>/<timestamp>/<preset_label>/result.json``. Sessions are ordered by timestamp
    directory name; the latest session containing a successful ``result.json`` leaf wins.
    """

    leaves: list[tuple[str, Path]] = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        session_dirs = sorted((p for p in variant_dir.iterdir() if p.is_dir()), reverse=True)
        for session_dir in session_dirs:
            leaf = _successful_leaf(session_dir)
            if leaf is not None:
                leaves.append((variant_dir.name, leaf))
                break
    return leaves


def _successful_leaf(session_dir: Path) -> Path | None:
    for leaf in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        result_path = leaf / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if result.get("status") in _SUCCESS_STATUSES:
            return leaf
    return None


def aggregate_stage_timings(
    events: list[dict[str, Any]],
    *,
    summary_warmup_batches: int = 1,
    timing_mode: str = "steady-state",
) -> SubstageAggregation:
    """Aggregate ``stage_timing`` events using the documented preceding-``batch_total`` batch assignment."""

    per_stage_batch_sums: dict[str, dict[int, float]] = {}
    per_stage_batch_last: dict[str, dict[int, float]] = {}
    per_stage_batch_events: dict[str, dict[int, list[float]]] = {}
    rolling_batch_events: dict[str, dict[int, list[float]]] = {}
    batch_totals: dict[int, float] = {}
    disk_write_walls: dict[int, float] = {}
    disk_write_bytes: dict[int, float] = {}
    current_batch: int | None = None

    for event in events:
        kind = event.get("event")
        fields = event.get("fields", {})
        if kind == "batch_total":
            batch = fields.get("batch")
            if batch is not None:
                current_batch = int(batch)
                wall_s = fields.get("wall_s")
                if wall_s is not None:
                    batch_totals[int(batch)] = float(wall_s)
            continue
        if kind == "disk_write":
            batch = fields.get("batch")
            if batch is not None and fields.get("wall_s") is not None:
                disk_write_walls[int(batch)] = float(fields["wall_s"])
                if fields.get("output_bytes") is not None:
                    disk_write_bytes[int(batch)] = float(fields["output_bytes"])
            continue
        if kind != "stage_timing" or current_batch is None:
            continue
        stage = fields.get("stage")
        wall_s = fields.get("wall_s")
        if not stage or wall_s is None:
            continue
        per_stage_batch_sums.setdefault(stage, {}).setdefault(current_batch, 0.0)
        per_stage_batch_sums[stage][current_batch] += float(wall_s)
        per_stage_batch_last.setdefault(stage, {})[current_batch] = float(wall_s)
        per_stage_batch_events.setdefault(stage, {}).setdefault(current_batch, []).append(float(wall_s))
        if stage.startswith(ROLLING_SUBSTAGE_PREFIX) and stage != "rolling_coefficient_update":
            rolling_batch_events.setdefault(stage, {}).setdefault(current_batch, []).append(float(wall_s))

    warmup = 0 if timing_mode == "all-batches" else summary_warmup_batches

    def _selected(batch_map: dict[int, Any]) -> list[Any]:
        return [value for batch, value in sorted(batch_map.items()) if batch >= warmup]

    aggregation = SubstageAggregation(batch_total_wall_s=batch_totals)
    for stage, batch_events in per_stage_batch_events.items():
        selected_events = [value for values in _selected(batch_events) for value in values]
        if selected_events:
            aggregation.per_event_mean[stage] = mean(selected_events)
    for stage, batch_sums in per_stage_batch_sums.items():
        selected_sums = _selected(batch_sums)
        if selected_sums:
            aggregation.per_batch_sum_mean[stage] = mean(selected_sums)
    for stage, batch_last in per_stage_batch_last.items():
        selected_last = _selected(batch_last)
        if selected_last:
            aggregation.per_batch_last_mean[stage] = mean(selected_last)
    for stage, batch_events in rolling_batch_events.items():
        selected_events = [value for values in _selected(batch_events) for value in values]
        if selected_events:
            aggregation.rolling_substage_means[stage] = sum(selected_events) / max(
                1, len([b for b in batch_events if b >= warmup])
            )
    selected_disk_walls = _selected(disk_write_walls)
    if selected_disk_walls:
        aggregation.disk_write_mean_wall_s = mean(selected_disk_walls)
    selected_disk_bytes = _selected(disk_write_bytes)
    if selected_disk_bytes:
        aggregation.disk_write_mean_output_bytes = mean(selected_disk_bytes)
    return aggregation


def _extract_columnar_rows(events: list[dict[str, Any]]) -> dict[int, int]:
    rows: dict[int, int] = {}
    for event in events:
        if event.get("event") != "columnar_output_summary":
            continue
        fields = event.get("fields", {})
        batch = fields.get("batch")
        row_counts = fields.get("row_counts") or {}
        total = sum(int(layer_counts.get("activation_rows", 0)) for layer_counts in row_counts.values())
        if batch is not None:
            rows[int(batch)] = total
    return rows


def _extract_shape_summary(events: list[dict[str, Any]]) -> tuple[int | None, list[int] | None]:
    for event in events:
        if event.get("event") == "packaging_shape_summary":
            fields = event.get("fields", {})
            token_shape = fields.get("token_shape")
            return fields.get("valid_token_count"), list(token_shape) if token_shape else None
    return None, None


def find_batch_json_dir(run_root: Path) -> Path | None:
    """Locate the directory holding legacy ``batch-*.json`` outputs under a run root."""

    for candidate in sorted(run_root.rglob("batch-0.json")):
        return candidate.parent
    return None


def legacy_activation_rows_per_batch(run_root: Path) -> dict[int, int]:
    """Sum per-feature activation row counts for each legacy ``batch-*.json`` output."""

    batch_dir = find_batch_json_dir(run_root)
    if batch_dir is None:
        return {}
    rows: dict[int, int] = {}
    for batch_path in sorted(batch_dir.glob("batch-*.json")):
        match = re.search(r"batch-(\d+)\.json$", batch_path.name)
        if match is None:
            continue
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        rows[int(match.group(1))] = sum(len(feature.get("activations", [])) for feature in payload.get("features", []))
    return rows


def extract_variant(
    leaf_dir: Path,
    *,
    variant_dir: str = "",
    summary_warmup_batches: int = 1,
    timing_mode: str = "steady-state",
    include_legacy_batch_rows: bool = False,
) -> VariantExtract:
    """Extract all benchmark data for one variant leaf directory."""

    result = json.loads((leaf_dir / "result.json").read_text(encoding="utf-8"))
    label = result.get("label", leaf_dir.name)
    scenario, path_key = classify_variant(label, variant_dir)
    events_path = leaf_dir / "runner_perf_events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
    aggregation = aggregate_stage_timings(
        events, summary_warmup_batches=summary_warmup_batches, timing_mode=timing_mode
    )

    import_profile = result.get("import_stage_profile") or {}
    import_path = leaf_dir / "import_stage_profile.json"
    if not import_profile and import_path.exists():
        import_profile = json.loads(import_path.read_text(encoding="utf-8"))

    valid_token_count, token_shape = _extract_shape_summary(events)
    substage_seconds = {
        stage: aggregation.per_batch_last_mean[stage]
        for stage in SUMMARY_STAGES
        if stage in aggregation.per_batch_last_mean
    }

    variant = VariantExtract(
        scenario=scenario or "unknown",
        path_key=path_key or "unknown",
        label=label,
        variant_dir=variant_dir,
        session_dir=str(leaf_dir.parent),
        leaf_dir=str(leaf_dir),
        status=result.get("status", "unknown"),
        n_features_per_batch=int(result.get("n_features_per_batch", 0)),
        n_prompts_in_forward_pass=int(result.get("n_prompts_in_forward_pass", 0)),
        completed_batches=list(result.get("completed_batches", [])),
        summary_warmup_batches=summary_warmup_batches,
        avg_batch_seconds=result.get("avg_batch_seconds"),
        throughput_features_per_min=result.get("throughput_features_per_min"),
        max_tree_rss_gib=result.get("max_tree_rss_gib"),
        max_host_used_gib=result.get("max_host_used_gib"),
        max_gpu_process_used_mib=result.get("max_gpu_process_used_mib"),
        max_gpu_device_used_mib=result.get("max_gpu_device_used_mib"),
        max_gpu_util_percent=result.get("max_gpu_util_percent"),
        avg_gpu_util_percent_steady=result.get("avg_gpu_util_percent_steady"),
        n_prompts_total=(
            int(result["n_prompts_total"])
            if result.get("n_prompts_total")
            else (int(_m.group(1)) if (_m := re.search(r"x\d+x(\d+)(?:-|$)", label)) else None)
        ),
        substage_seconds=substage_seconds,
        substage_per_batch_sum=aggregation.per_batch_sum_mean,
        batch_total_wall_s=aggregation.batch_total_wall_s,
        rolling_substage_means=aggregation.rolling_substage_means,
        legacy_conversion_seconds=aggregation.per_event_mean.get(LEGACY_CONVERSION_STAGE),
        disk_write_mean_wall_s=aggregation.disk_write_mean_wall_s,
        disk_write_mean_output_bytes=aggregation.disk_write_mean_output_bytes,
        columnar_write_seconds=(
            sum(aggregation.per_batch_sum_mean.get(stage, 0.0) for stage in COLUMNAR_WRITE_STAGES) or None
            if path_key == "columnar_gpu"
            else None
        ),
        import_wall_seconds=import_profile.get("wall_seconds"),
        import_conversion_seconds=import_profile.get("conversion_seconds"),
        import_activation_load_seconds=(
            import_profile.get("activation_load_seconds")
            if import_profile.get("activation_load_seconds") is not None
            else import_profile.get("activation_table_load_seconds")
        ),
        import_activation_import_seconds=import_profile.get("activation_import_seconds"),
        imported_activation_rows=import_profile.get("imported_activation_rows"),
        imported_neuron_rows=import_profile.get("imported_neuron_rows"),
        imported_row_counts=dict(import_profile.get("imported_row_counts") or {}),
        valid_token_count=valid_token_count,
        token_shape=token_shape,
    )

    if variant.path_key == "columnar_gpu":
        variant.activation_rows_per_batch = _extract_columnar_rows(events)
    elif include_legacy_batch_rows:
        run_root = Path(result.get("run_root", leaf_dir / "run_root"))
        if run_root.exists():
            variant.activation_rows_per_batch = legacy_activation_rows_per_batch(run_root)

    if (
        variant.avg_batch_seconds
        and variant.import_wall_seconds is not None
        and variant.completed_batches
        and variant.n_features_per_batch
    ):
        n_batches = len(variant.completed_batches)
        generation_total_s = variant.avg_batch_seconds * n_batches
        total_features = variant.n_features_per_batch * n_batches
        variant.e2e_features_per_min = total_features * 60.0 / (generation_total_s + variant.import_wall_seconds)

    return variant


def extract_root(
    root: Path,
    *,
    summary_warmup_batches: int = 1,
    timing_mode: str = "steady-state",
    include_legacy_batch_rows: bool = False,
) -> list[VariantExtract]:
    """Extract all variants under an artifact root."""

    return [
        extract_variant(
            leaf,
            variant_dir=variant_dir,
            summary_warmup_batches=summary_warmup_batches,
            timing_mode=timing_mode,
            include_legacy_batch_rows=include_legacy_batch_rows,
        )
        for variant_dir, leaf in find_variant_leaves(root)
    ]


def select_parity_pair(variants: Iterable[VariantExtract]) -> tuple[VariantExtract, VariantExtract] | None:
    """Return the (detached_legacy, current_legacy) pair sharing the same batch-shape config, if any.

    Scaling sweeps add multiple current-legacy variants per scenario; parity is only meaningful between runs with
    identical ``n_features_per_batch``/``n_prompts_in_forward_pass`` (feature batching changes per-batch feature
    membership, not just timing).
    """

    detached = [v for v in variants if v.path_key == "detached_legacy"]
    current = [v for v in variants if v.path_key == "current_legacy"]
    for det in detached:
        for cur in current:
            if (cur.n_features_per_batch, cur.n_prompts_in_forward_pass) == (
                det.n_features_per_batch,
                det.n_prompts_in_forward_pass,
            ):
                return det, cur
    return None


def activation_row_parity(det_variant: VariantExtract, cur_variant: VariantExtract) -> ParityResult:
    """Compare per-feature activation row counts between the detached and current legacy variants."""

    parity = ParityResult(scenario=det_variant.scenario)
    det_result = json.loads((Path(det_variant.leaf_dir) / "result.json").read_text(encoding="utf-8"))
    cur_result = json.loads((Path(cur_variant.leaf_dir) / "result.json").read_text(encoding="utf-8"))
    det_dir = find_batch_json_dir(Path(det_result["run_root"]))
    cur_dir = find_batch_json_dir(Path(cur_result["run_root"]))
    if det_dir is None or cur_dir is None:
        parity.per_batch.append({"error": "batch JSON outputs unavailable for one or both variants"})
        return parity
    for det_path in sorted(det_dir.glob("batch-*.json")):
        cur_path = cur_dir / det_path.name
        if not cur_path.exists():
            continue
        det_payload = json.loads(det_path.read_text(encoding="utf-8"))
        cur_payload = json.loads(cur_path.read_text(encoding="utf-8"))
        det_features = det_payload.get("features", [])
        cur_features = cur_payload.get("features", [])
        det_rows = sum(len(f.get("activations", [])) for f in det_features)
        cur_rows = sum(len(f.get("activations", [])) for f in cur_features)
        mismatches = sum(
            1
            for det_f, cur_f in zip(det_features, cur_features)
            if len(det_f.get("activations", [])) != len(cur_f.get("activations", []))
        )
        parity.total_feature_batches += min(len(det_features), len(cur_features))
        parity.mismatched_feature_batches += mismatches
        batch_num = int(re.search(r"batch-(\d+)\.json$", det_path.name).group(1))  # type: ignore[union-attr]
        parity.per_batch.append(
            {
                "batch": batch_num,
                "det_rows": det_rows,
                "cur_rows": cur_rows,
                "match": det_rows == cur_rows and mismatches == 0,
                "mismatched_features": mismatches,
            }
        )
    return parity


def _fmt(value: Any, spec: str = ".3f", fallback: str = "-") -> str:
    if value is None:
        return fallback
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def _ordered_paths(variants: Iterable[VariantExtract]) -> list[VariantExtract]:
    order = {key: idx for idx, key in enumerate(PATH_KEYS)}
    return sorted(variants, key=lambda v: (order.get(v.path_key, 99), v.label))


def variant_config_label(v: VariantExtract) -> str:
    """Config label for headings/legends: features x fwd-minibatch [x total-prompts for sweep legs]."""
    label = f"{v.n_features_per_batch}x{v.n_prompts_in_forward_pass}"
    if v.n_prompts_total is not None:
        label += f"x{v.n_prompts_total}"
    return label


def split_prompt_sweep(variants: Iterable[VariantExtract]) -> tuple[list[VariantExtract], list[VariantExtract]]:
    """Split variants into (main benchmark legs, n-prompt sweep legs) — sweep legs carry n_prompts_total."""
    main: list[VariantExtract] = []
    sweep: list[VariantExtract] = []
    for v in variants:
        (sweep if v.n_prompts_total is not None else main).append(v)
    return main, sweep


def variants_by_scenario(variants: Iterable[VariantExtract]) -> dict[str, list[VariantExtract]]:
    grouped: dict[str, list[VariantExtract]] = {}
    for variant in variants:
        grouped.setdefault(variant.scenario, []).append(variant)
    return {scenario: _ordered_paths(group) for scenario, group in grouped.items()}


def render_primary_table(variants: list[VariantExtract]) -> str:
    lines = [
        "| Variant | Config | Avg batch s | Features/min | Import wall s | Import load s | Import act s | "
        "Activation rows | E2E features/min |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in _ordered_paths(variants):
        lines.append(
            f"| {PATH_DISPLAY.get(v.path_key, v.path_key)} | {variant_config_label(v)} "
            f"| {_fmt(v.avg_batch_seconds, '.2f')} | {_fmt(v.throughput_features_per_min, '.1f')} "
            f"| {_fmt(v.import_wall_seconds, '.1f')} | {_fmt(v.import_activation_load_seconds, '.1f')} "
            f"| {_fmt(v.import_activation_import_seconds, '.1f')} | {_fmt(v.imported_activation_rows, ',')} "
            f"| {_fmt(v.e2e_features_per_min, '.1f')} |"
        )
    return "\n".join(lines)


def _variant_column_heading(v: VariantExtract) -> str:
    """Unambiguous per-variant column heading: path display + config label (features x minibatch
    [x total prompts]) so multi-config tables never repeat identical headings."""
    return f"{PATH_DISPLAY.get(v.path_key, v.path_key)} {variant_config_label(v)}"


def render_substage_table(variants: list[VariantExtract]) -> str:
    ordered = _ordered_paths(variants)
    header = " | ".join(f"{_variant_column_heading(v)} s" for v in ordered)
    lines = [
        f"| Stage | {header} |",
        "| --- |" + " ---: |" * len(ordered),
    ]
    for stage in SUMMARY_STAGES:
        cells = " | ".join(_fmt(v.substage_seconds.get(stage)) for v in ordered)
        lines.append(f"| {stage} | {cells} |")
    cells = " | ".join(_fmt(v.avg_batch_seconds) for v in ordered)
    lines.append(f"| batch_total | {cells} |")
    return "\n".join(lines)


def render_import_table(variants: list[VariantExtract]) -> str:
    lines = [
        "| Variant | Config | Import wall s | Conversion s | Activation load s | Activation import s | Imported rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in _ordered_paths(variants):
        lines.append(
            f"| {PATH_DISPLAY.get(v.path_key, v.path_key)} | {variant_config_label(v)} "
            f"| {_fmt(v.import_wall_seconds, '.1f')} "
            f"| {_fmt(v.import_conversion_seconds, '.1f')} | {_fmt(v.import_activation_load_seconds, '.1f')} "
            f"| {_fmt(v.import_activation_import_seconds, '.1f')} | {_fmt(v.imported_activation_rows, ',')} |"
        )
    return "\n".join(lines)


def render_parity_table(parity: ParityResult) -> str:
    lines = [
        "| Batch | Det rows | Cur rows | Match | Mismatched features |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in parity.per_batch:
        if "error" in row:
            return f"Parity unavailable: {row['error']}"
        lines.append(
            f"| {row['batch']} | {row['det_rows']} | {row['cur_rows']} | "
            f"{'MATCH' if row['match'] else 'MISMATCH'} | {row['mismatched_features']} |"
        )
    rate = parity.match_rate
    if rate is not None:
        lines.append("")
        lines.append(
            f"**{rate:.2%} per-feature match across {parity.total_feature_batches} feature-batches "
            f"({parity.mismatched_feature_batches} mismatches)**"
        )
    return "\n".join(lines)


def render_resource_table(variants: list[VariantExtract]) -> str:
    lines = [
        "| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB "
        "| Avg GPU util % (steady) | Max GPU util % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for v in _ordered_paths(variants):
        lines.append(
            f"| {PATH_DISPLAY.get(v.path_key, v.path_key)} | {variant_config_label(v)} "
            f"| {_fmt(v.max_tree_rss_gib, '.2f')} "
            f"| {_fmt(v.max_host_used_gib, '.2f')} | {_fmt(v.max_gpu_process_used_mib, ',')} "
            f"| {_fmt(v.max_gpu_device_used_mib, ',')} | {_fmt(v.avg_gpu_util_percent_steady, '.1f')} "
            f"| {_fmt(v.max_gpu_util_percent, 'd')} |"
        )
    return "\n".join(lines)


def _mermaid_node_id(scenario: str, path_key: str, stage: str) -> str:
    return f"{scenario}_{path_key}_{stage}".replace("-", "_")


# The diagram intentionally shows ONE representative variant per path (Monology scenario,
# the showcase 4096x256 config) so every node has exactly one inbound/outbound arrow and the
# timings/cardinality/throughput are unambiguous. The full multi-config data lives in the
# summary tables and the profiling notebook.
MERMAID_SCENARIO = "monology"
MERMAID_PATHS = ("current_legacy", "columnar_gpu")
MERMAID_CONFIG = (4096, 256)


def _select_mermaid_variants(variants: list[VariantExtract]) -> list[VariantExtract]:
    selected: list[VariantExtract] = []
    for path_key in MERMAID_PATHS:
        candidates = [
            v
            for v in variants
            if v.scenario == MERMAID_SCENARIO and v.path_key == path_key and v.n_prompts_total is None
        ]
        if not candidates:
            continue
        preferred = next(
            (v for v in candidates if (v.n_features_per_batch, v.n_prompts_in_forward_pass) == MERMAID_CONFIG),
            None,
        )
        selected.append(preferred if preferred is not None else max(candidates, key=lambda v: v.n_features_per_batch))
    return selected


def render_mermaid_diagram(
    variants: list[VariantExtract],
    *,
    lineages: dict[str, str] | None = None,
    pretokenization_record: dict[str, Any] | None = None,
) -> str:
    """Render the dashboard benchmark flow diagram: one legacy and one columnar_gpu subgraph
    (Monology, showcase config), each a single linear flow with per-node timings."""

    lineages = lineages or {}
    if pretokenization_record:
        pretok_runtime = f"{pretokenization_record.get('duration_s', 'TBD')} s"
        pretok_rows = pretokenization_record.get("rows")
        pretok_ctx = pretokenization_record.get("context_size")
        pretok_cardinality = f"{pretok_rows:,} prompts x {pretok_ctx} tokens" if pretok_rows else "TBD"
    else:
        pretok_runtime = "TBD"
        pretok_cardinality = "TBD"
    lines = [
        "flowchart TD",
        "    %% Regenerated by scripts/run_dashboard_benchmark_suite.py — do not edit by hand.",
        "    classDef source fill:#f9f,stroke:#333,stroke-width:2px;",
        "    classDef process fill:#bbf,stroke:#333,stroke-width:1px;",
        "    classDef output fill:#ffd,stroke:#333,stroke-width:1px;",
        "    classDef sink fill:#bfb,stroke:#333,stroke-width:2px;",
        "    classDef meta fill:#eee,stroke:#666,stroke-width:1px;",
        "",
        '    PRETOK[/"📥 pretokenization (one-time per model/dataset)<br/>'
        f'<b>Runtime:</b> {pretok_runtime}<br/><b>Cardinality:</b> {pretok_cardinality}"/]:::source',
        "",
    ]
    for variant in _select_mermaid_variants(variants):
        s, p = variant.scenario, variant.path_key
        lineage = lineages.get(p, "")
        cfg_label = variant_config_label(variant)
        title = f"{SCENARIO_DISPLAY.get(s, s)} — {PATH_DISPLAY.get(p, p)} · {cfg_label} config"
        lines.append(f'    subgraph {s}_{p}["{title}"]')
        lines.append("        direction TB")

        def node(stage: str, text: str, style: str = "process") -> str:
            node_id = _mermaid_node_id(s, p, stage)
            lines.append(f'        {node_id}["{text}"]:::{style}')
            return node_id

        # Run metadata lives in its own node (not the subgraph title) so it cannot overlap
        # the first flow node when mermaid lays the subgraph out.
        meta_text = (
            f"📈 <b>config:</b> {cfg_label} (features x fwd-minibatch)"
            + (f"<br/><b>lineage:</b> {lineage}" if lineage else "")
            + f"<br/><b>batch_total:</b> {_fmt(variant.avg_batch_seconds, '.2f')} s · "
            f"{_fmt(variant.throughput_features_per_min, '.0f')} features/min"
            f"<br/><b>peak RSS:</b> {_fmt(variant.max_tree_rss_gib, '.1f')} GiB · "
            f"<b>peak GPU:</b> {_fmt(variant.max_gpu_process_used_mib, ',')} MiB"
        )
        meta = node("meta", meta_text, "meta")

        token_note = ""
        if variant.token_shape:
            token_note = f"<br/><b>Tokens:</b> {variant.token_shape[0]}x{variant.token_shape[1]}"
        if variant.valid_token_count:
            token_note += f"<br/><b>Valid tokens:</b> {variant.valid_token_count:,}"
        tok = node("tokens", f"🧾 prompt tokens{token_note}", "source")
        lines.append(f"        {meta} ~~~ {tok}")

        act = node(
            "act",
            "activation_and_encode_total<br/><b>Steady:</b> "
            f"{_fmt(variant.substage_seconds.get('activation_and_encode_total'))} s",
        )
        roll = node(
            "roll",
            "rolling_coefficient_update<br/><b>Per-batch sum:</b> "
            f"{_fmt(variant.substage_per_batch_sum.get('rolling_coefficient_update'))} s"
            f"<br/><b>Final minibatch (table value):</b> "
            f"{_fmt(variant.substage_seconds.get('rolling_coefficient_update'))} s",
        )
        fstat = node(
            "fstat",
            "feature_statistics_packaging<br/><b>Steady:</b> "
            f"{_fmt(variant.substage_seconds.get('feature_statistics_packaging'))} s",
        )
        lhist = node(
            "lhist",
            "logits_histogram_packaging<br/><b>Steady:</b> "
            f"{_fmt(variant.substage_seconds.get('logits_histogram_packaging'))} s",
        )
        ahist = node(
            "ahist",
            "activation_histogram_packaging<br/><b>Steady:</b> "
            f"{_fmt(variant.substage_seconds.get('activation_histogram_packaging'))} s",
        )
        seq = node(
            "seq",
            f"sequence_packaging<br/><b>Steady:</b> {_fmt(variant.substage_seconds.get('sequence_packaging'))} s",
        )

        if p == "columnar_gpu":
            write_text = (
                "columnar_artifact_write<br/><b>Steady (row+copy-row packaging):</b> "
                f"{_fmt(variant.columnar_write_seconds)} s"
            )
            out = node("colwrite", write_text, "output")
        else:
            conv_text = f"conversion/JSON serialization<br/><b>Steady:</b> {_fmt(variant.legacy_conversion_seconds)} s"
            if variant.disk_write_mean_wall_s is not None:
                conv_text += f"<br/><b>disk_write:</b> {_fmt(variant.disk_write_mean_wall_s)} s"
                if variant.disk_write_mean_output_bytes:
                    conv_text += f" ({variant.disk_write_mean_output_bytes / 1024**2:.0f} MiB)"
            out = node("jsonconv", conv_text, "output")
            out2 = node(
                "jsonlgz",
                "jsonl_gz_conversion_export<br/><b>Import-side conversion:</b> "
                f"{_fmt(variant.import_conversion_seconds, '.1f')} s",
                "output",
            )

        imp = node(
            "import",
            "DB import<br/><b>Wall:</b> "
            f"{_fmt(variant.import_wall_seconds, '.1f')} s<br/><b>Activation load:</b> "
            f"{_fmt(variant.import_activation_load_seconds, '.1f')} s<br/><b>Activation import:</b> "
            f"{_fmt(variant.import_activation_import_seconds, '.1f')} s",
            "sink",
        )

        feature_edge = f"📊 {variant.n_features_per_batch} features/batch"
        lines.append(f"        {tok} -->|{feature_edge}| {act}")
        lines.append(f"        {act} -.-> {roll}")
        lines.append(f"        {act} --> {fstat} --> {lhist} --> {ahist} --> {seq}")
        rows_note = (
            f"📊 {variant.imported_activation_rows:,} activation rows"
            if variant.imported_activation_rows is not None
            else "📊 rows n/a"
        )
        lines.append(f"        {seq} -->|{rows_note}| {out}")
        if p != "columnar_gpu":
            lines.append(f"        {out} --> {out2}")
            out = out2
        e2e_note = (
            f"⚡ {variant.e2e_features_per_min:.0f} E2E features/min"
            if variant.e2e_features_per_min is not None
            else "⚡ E2E n/a"
        )
        lines.append(f"        {out} -->|{e2e_note}| {imp}")
        lines.append("    end")
        lines.append(f"    PRETOK -.-> {tok}")
        lines.append("")
    return "\n".join(lines)


def render_summary_markdown(
    variants: list[VariantExtract],
    parities: dict[str, ParityResult],
    *,
    manifest: dict[str, Any],
    notebook_name: str | None = None,
    diagram_name: str = "dashboard_benchmark_diagram.mmd",
) -> str:
    """Render the top-level reviewer summary document for one artifact package."""

    main_variants, sweep_variants = split_prompt_sweep(variants)
    grouped = variants_by_scenario(main_variants)
    generated_at = manifest.get("generated_at_utc", "")
    lines = [
        "# Dashboard Benchmark Summary",
        "",
        f"Generated: {generated_at}",
        "",
        f"Source artifact root: `{manifest.get('source_root', 'unknown')}`",
        f"Timing mode: {manifest.get('timing_mode', 'steady-state')} "
        f"(warmup batches excluded: {manifest.get('summary_warmup_batches', 1)})",
        "",
        "## Lineage",
        "",
        "| Path | Lineage | Dirty repos |",
        "| --- | --- | --- |",
    ]
    lineages = manifest.get("lineages", {})
    dirty = ", ".join(manifest.get("dirty_repos", [])) or "none"
    for path_key in PATH_KEYS:
        if path_key in lineages:
            lines.append(f"| {PATH_DISPLAY[path_key]} | `{lineages[path_key]}` | {dirty} |")
    invocation = manifest.get("invocation")
    if invocation:
        sweep = manifest.get("monology_sweep")
        lines += [
            "",
            "## Generation commands",
            "",
            "Suite invocation that produced these artifacts:",
            "",
            "```bash",
            invocation,
            "```",
        ]
        # Reference the pretokenization command of record; note whether a pretokenized prompt cache was
        # (re)used vs streamed, so the artifact is reproducible without re-running pretokenization.
        if sweep == "pretok":
            lines.append(
                "Prompts: reused the largest built pretokenized Monology `concat_<N>` cache (columnar) — NOT "
                "re-pretokenized this run. Rebuild it with the Monology command in "
                "`docs/neuronpedia_dashboard_pipeline.md` (§ Pretokenize dashboard datasets / Prompt-dimension "
                "scaling sweep sets); the legacy leg streamed the same first-N prompts."
            )
        elif sweep == "streaming":
            lines.append(
                "Prompts: streamed `monology/pile-uncopyrighted` via `load_dataset` (no pretokenized cache). "
                "For a reproducible/HF-independent baseline, build a pretokenized `concat_<N>` set per "
                "`docs/neuronpedia_dashboard_pipeline.md` and use `--monology-sweep pretok`."
            )
        else:
            lines.append(
                "Prompts: the accepted-shape presets use their pretokenized caches (built once via the "
                "commands in `docs/neuronpedia_dashboard_pipeline.md` § Pretokenize dashboard datasets); not "
                "re-pretokenized this run."
            )
    lines += [
        "",
        "## Linked Assets",
        "",
        f"- Unified flow diagram: [{diagram_name}]({diagram_name})",
    ]
    if notebook_name:
        lines.append(f"- Profiling notebook: [{notebook_name}]({notebook_name})")
        lines.append(f"- Notebook HTML export: [{Path(notebook_name).stem}.html]({Path(notebook_name).stem}.html)")
    lines.append("")
    for scenario in SCENARIOS:
        group = grouped.get(scenario)
        if not group:
            continue
        display = SCENARIO_DISPLAY.get(scenario, scenario)
        lines += [
            f"## {display}",
            "",
            "### Primary Benchmark",
            "",
            render_primary_table(group),
            "",
            "### Substage Timings (steady-state)",
            "",
            render_substage_table(group),
            "",
            "### DB Import Substage",
            "",
            render_import_table(group),
            "",
            "### Resource Peaks",
            "",
            render_resource_table(group),
            "",
        ]
        parity = parities.get(scenario)
        if parity is not None:
            lines += ["### Activation Row Parity (detached vs current legacy)", "", render_parity_table(parity), ""]
    if sweep_variants:
        sweep_layer = manifest.get("prompt_sweep_layer", manifest.get("layer", "?"))
        path_order = {key: idx for idx, key in enumerate(PATH_KEYS)}
        sweep_group = sorted(sweep_variants, key=lambda v: (path_order.get(v.path_key, 99), v.n_prompts_total or 0))
        sweep_configs = ", ".join(variant_config_label(v) for v in sweep_group)
        pretok_record = manifest.get("pretokenization_record") or {}
        lines += [
            "## N-prompt scaling (Monology)",
            "",
            f"Prompt-dimension scaling sweep on **layer {sweep_layer}** (single configurable layer; "
            "column headings carry the full config as `features x fwd-minibatch x total-prompts`). "
            f"Swept configs: {sweep_configs}. Columnar path only, run under the opt-in reduced-peak-memory flags "
            f"(`{'`, `'.join(manifest.get('prompt_sweep_mitigation_args', []) or ['see leg logs'])}`) that "
            "the 24,576-prompt point requires — outputs are bit-identical; only peak GPU "
            "memory and speed move. NOTE: a single-layer curve understates the OOM ceiling set by the "
            "densest layer (layer-density adds several GiB to the packaging working set).",
            "",
            "### Primary Benchmark (n-prompt sweep)",
            "",
            render_primary_table(sweep_group),
            "",
            "### Substage Timings (n-prompt sweep, steady-state)",
            "",
            render_substage_table(sweep_group),
            "",
            "### Resource Peaks (n-prompt sweep)",
            "",
            render_resource_table(sweep_group),
            "",
        ]
        if pretok_record:
            lines += [
                f"Pretokenization (one-time): {pretok_record.get('duration_s', '?')} s for "
                f"{_fmt(pretok_record.get('rows'), ',', '?')} prompts x "
                f"{pretok_record.get('context_size', '?')} tokens "
                f"(recorded {pretok_record.get('generated_at_utc', '?')}).",
                "",
            ]
    lines += [
        "## Regeneration",
        "",
        "One command regenerates this full artifact (all benchmark legs, the n-prompt scaling sweep, "
        "tables, diagram, and notebook):",
        "",
        "```bash",
        "python scripts/run_dashboard_benchmark_suite.py --mode full",
        "```",
        "",
        "To re-package from an existing artifact root without re-running benchmarks:",
        "",
        "```bash",
        "python scripts/run_dashboard_benchmark_suite.py --from-existing <artifact_root> --package-root <out_dir>",
        "```",
        "",
        "See `scripts/dashboard_benchmark_suite_usage.md` for full usage, including live 3-way and scaling modes.",
        "",
    ]
    return "\n".join(lines)


def write_extracted_data(package_dir: Path, variants: list[VariantExtract], parities: dict[str, ParityResult]) -> Path:
    """Persist the machine-readable extraction payload consumed by the profiling notebook."""

    payload = {
        "variants": [asdict(v) for v in variants],
        "parities": {scenario: asdict(parity) for scenario, parity in parities.items()},
        "summary_stages": list(SUMMARY_STAGES),
        "path_display": PATH_DISPLAY,
        "scenario_display": SCENARIO_DISPLAY,
    }
    out_path = package_dir / "extracted_data.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
