from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable

from interpretune.utils.neuronpedia_explanations import DEFAULT_IT_NP_CACHE


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_NAME = "gemma-3-1b-it"
DEFAULT_SOURCE_SET_ID = "gemmascope-2-transcoder-262k-rte"
DEFAULT_RUN_NAME = f"{DEFAULT_MODEL_NAME}_{DEFAULT_SOURCE_SET_ID}"
DEFAULT_BASE_OUTPUT_DIR = Path("/tmp/np_dashboard_config_sweeps")
# IT_BENCH_PYTHON overrides; otherwise the interpreter running this script is used. All artifact roots
# resolve from the Neuronpedia cache root seam ($IT_NP_CACHE, else $HF_HOME/interpretune/neuronpedia).
DEFAULT_PYTHON = os.getenv("IT_BENCH_PYTHON", sys.executable)
NP_CACHE_ROOT = DEFAULT_IT_NP_CACHE
DEFAULT_RUN_ROOT = NP_CACHE_ROOT / "dashboard_runs"

OOM_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "out of memory",
)
SHAPE_MISMATCH_MARKERS = (
    "The size of tensor a",
    "must match the size of tensor b",
)

BATCH_OUTPUT_RE = re.compile(r"Output written to .*?/batch-(\d+)\.json")
POST_BATCH_RE = re.compile(
    r"\[runner_resource\] stage=post_batch_(\d+) .* rss_gib=([0-9.]+).* cuda_max_allocated_gib=([0-9.]+)"
)
RUNNER_PERF_PREFIX = "[runner_perf] "


@dataclass(frozen=True)
class SweepConfig:
    """One batch-shape probe in the dashboard sweep."""

    label: str
    n_features_per_batch: int
    n_prompts_in_forward_pass: int
    primary_acts_batch_size: int | None = None
    prompt_bucket_ceiling: int | None = None


@dataclass
class BatchRecord:
    """Observed metrics for one completed batch."""

    batch_num: int
    detected_at_utc: str
    detected_monotonic: float
    post_batch_rss_gib: float | None = None
    post_batch_cuda_max_gib: float | None = None


@dataclass
class SweepResult:
    """Captured summary for one configuration run."""

    label: str
    n_features_per_batch: int
    n_prompts_in_forward_pass: int
    status: str
    reason: str
    exit_code: int | None
    started_at_utc: str
    ended_at_utc: str
    elapsed_seconds: float
    completed_batches: list[int] = field(default_factory=list)
    avg_batch_seconds: float | None = None
    throughput_features_per_min: float | None = None
    max_tree_rss_gib: float = 0.0
    max_tree_cpu_percent: float = 0.0
    max_host_used_gib: float = 0.0
    max_gpu_process_used_mib: int = 0
    max_gpu_device_used_mib: int = 0
    max_gpu_util_percent: int = 0
    total_model_forward_passes: int | None = None
    avg_model_forward_wall_s: float | None = None
    total_model_forward_wall_s: float | None = None
    avg_get_feature_data_wall_s: float | None = None
    total_get_feature_data_wall_s: float | None = None
    avg_get_feature_data_share_of_batch: float | None = None
    peak_get_feature_data_rss_gib: float | None = None
    peak_get_feature_data_cuda_allocated_gib: float | None = None
    peak_get_feature_data_cuda_reserved_gib: float | None = None
    peak_post_batch_rss_gib: float | None = None
    peak_post_batch_cuda_max_gib: float | None = None
    pipeline_log_path: str = ""
    run_root: str = ""
    command: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureDataSummaryMetrics:
    total_model_forward_passes: int | None = None
    avg_model_forward_wall_s: float | None = None
    total_model_forward_wall_s: float | None = None
    avg_get_feature_data_wall_s: float | None = None
    total_get_feature_data_wall_s: float | None = None
    avg_get_feature_data_share_of_batch: float | None = None
    peak_get_feature_data_rss_gib: float | None = None
    peak_get_feature_data_cuda_allocated_gib: float | None = None
    peak_get_feature_data_cuda_reserved_gib: float | None = None


def round_down_to_multiple(value: int, *, multiple: int) -> int:
    """Round down to a positive multiple while keeping small values intact."""

    if multiple <= 1 or value <= multiple:
        return value
    rounded_value = (value // multiple) * multiple
    return rounded_value if rounded_value > 0 else value


def scale_batch_size(
    base_value: int,
    *,
    scale: float,
    scale_limit: float,
    max_value: int,
    round_to: int,
) -> int:
    """Scale a batch size upward with a cap and a stable rounding policy."""

    scaled_value = max(base_value, int(base_value * min(scale, scale_limit)))
    scaled_value = min(scaled_value, max_value)
    return round_down_to_multiple(scaled_value, multiple=round_to)


def prefer_exact_primary_acts_partition(
    *,
    n_prompts_in_forward_pass: int,
    primary_acts_batch_size: int | None,
    max_prompt_count: int,
    max_prompt_reduction_fraction: float = 0.10,
    max_primary_acts_reduction_fraction: float = 0.10,
) -> tuple[int, int | None]:
    """Prefer an exact prompt-to-acts partition when a nearby downward-only adjustment exists."""

    capped_prompts = min(n_prompts_in_forward_pass, max_prompt_count)
    if primary_acts_batch_size is None:
        return capped_prompts, None

    capped_acts = min(primary_acts_batch_size, capped_prompts)
    if capped_acts <= 0 or capped_prompts % capped_acts == 0:
        return capped_prompts, capped_acts

    min_prompts = max(1, math.ceil(capped_prompts * (1.0 - max_prompt_reduction_fraction)))
    min_acts = max(1, math.ceil(capped_acts * (1.0 - max_primary_acts_reduction_fraction)))

    best_pair: tuple[int, int] | None = None
    for acts in range(capped_acts, min_acts - 1, -1):
        prompts = (capped_prompts // acts) * acts
        if prompts < min_prompts or prompts <= 0:
            continue

        candidate = (prompts, acts)
        if (
            best_pair is None
            or candidate[0] > best_pair[0]
            or (candidate[0] == best_pair[0] and candidate[1] > best_pair[1])
        ):
            best_pair = candidate

    if best_pair is not None:
        return best_pair
    return capped_prompts, capped_acts


def build_bucket_seed_config(
    *,
    label_prefix: str,
    n_features_per_batch: int,
    base_tokens_in_prompt: int,
    base_n_prompts_in_forward_pass: int,
    base_primary_acts_batch_size: int | None,
    bucket_ceiling: int,
    bucket_prompt_count: int,
    prompt_scale_limit: float,
    primary_acts_scale_limit: float,
    round_to: int,
) -> SweepConfig:
    """Build the first bucket-aware probe config from the full-context baseline settings."""

    if bucket_ceiling <= 0:
        raise ValueError("bucket_ceiling must be positive.")
    if bucket_prompt_count <= 0:
        raise ValueError("bucket_prompt_count must be positive.")

    scale = base_tokens_in_prompt / bucket_ceiling
    n_prompts_in_forward_pass = scale_batch_size(
        base_n_prompts_in_forward_pass,
        scale=scale,
        scale_limit=prompt_scale_limit,
        max_value=bucket_prompt_count,
        round_to=round_to,
    )
    primary_acts_batch_size = base_primary_acts_batch_size
    if base_primary_acts_batch_size is not None:
        primary_acts_batch_size = scale_batch_size(
            base_primary_acts_batch_size,
            scale=scale,
            scale_limit=primary_acts_scale_limit,
            max_value=n_prompts_in_forward_pass,
            round_to=round_to,
        )

    n_prompts_in_forward_pass, primary_acts_batch_size = prefer_exact_primary_acts_partition(
        n_prompts_in_forward_pass=n_prompts_in_forward_pass,
        primary_acts_batch_size=primary_acts_batch_size,
        max_prompt_count=bucket_prompt_count,
    )

    acts_label = f"acts{primary_acts_batch_size}" if primary_acts_batch_size is not None else "actsauto"
    return SweepConfig(
        label=(
            f"{label_prefix}-bucket{bucket_ceiling}-{n_features_per_batch}x{n_prompts_in_forward_pass}-{acts_label}"
        ),
        n_features_per_batch=n_features_per_batch,
        n_prompts_in_forward_pass=n_prompts_in_forward_pass,
        primary_acts_batch_size=primary_acts_batch_size,
        prompt_bucket_ceiling=bucket_ceiling,
    )


def build_same_layer_fixed_baseline_config(args: argparse.Namespace) -> SweepConfig:
    """Build a same-layer full-context control from the bucket pilot's base shape."""

    acts_label = (
        f"acts{args.base_primary_acts_batch_size}" if args.base_primary_acts_batch_size is not None else "actsauto"
    )
    return SweepConfig(
        label=(f"same-layer-fixed-{args.base_n_features_per_batch}x{args.base_n_prompts_in_forward_pass}-{acts_label}"),
        n_features_per_batch=args.base_n_features_per_batch,
        n_prompts_in_forward_pass=args.base_n_prompts_in_forward_pass,
        primary_acts_batch_size=args.base_primary_acts_batch_size,
    )


def build_more_conservative_bucket_candidate(
    config: SweepConfig,
    *,
    round_to: int,
) -> SweepConfig | None:
    """Return the next smaller candidate by halving acts first, then logical prompt batch size."""

    if config.primary_acts_batch_size is not None and config.primary_acts_batch_size > round_to:
        reduced_acts_for_batches = round_down_to_multiple(
            max(round_to, config.primary_acts_batch_size // 2),
            multiple=round_to,
        )
        if reduced_acts_for_batches < config.primary_acts_batch_size:
            return SweepConfig(
                label=f"{config.label}-actsdown",
                n_features_per_batch=config.n_features_per_batch,
                n_prompts_in_forward_pass=config.n_prompts_in_forward_pass,
                primary_acts_batch_size=reduced_acts_for_batches,
                prompt_bucket_ceiling=config.prompt_bucket_ceiling,
            )

    if config.n_prompts_in_forward_pass > round_to:
        reduced_prompts = round_down_to_multiple(
            max(round_to, config.n_prompts_in_forward_pass // 2), multiple=round_to
        )
        if reduced_prompts < config.n_prompts_in_forward_pass:
            reduced_acts_for_prompts: int | None = config.primary_acts_batch_size
            if reduced_acts_for_prompts is not None:
                reduced_acts_for_prompts = min(reduced_acts_for_prompts, reduced_prompts)
            return SweepConfig(
                label=f"{config.label}-promptsdown",
                n_features_per_batch=config.n_features_per_batch,
                n_prompts_in_forward_pass=reduced_prompts,
                primary_acts_batch_size=reduced_acts_for_prompts,
                prompt_bucket_ceiling=config.prompt_bucket_ceiling,
            )
    return None


def build_more_aggressive_bucket_candidate(
    config: SweepConfig,
    *,
    bucket_prompt_count: int,
    growth_factor: float,
    round_to: int,
) -> SweepConfig | None:
    """Return a larger candidate for the same bucket, capped by the staged prompt count."""

    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than 1.0.")

    next_prompts = round_down_to_multiple(
        min(
            bucket_prompt_count,
            max(
                config.n_prompts_in_forward_pass + round_to,
                int(config.n_prompts_in_forward_pass * growth_factor),
            ),
        ),
        multiple=round_to,
    )
    if next_prompts <= config.n_prompts_in_forward_pass:
        return None

    next_acts = config.primary_acts_batch_size
    if next_acts is not None:
        next_acts = round_down_to_multiple(
            min(next_prompts, max(next_acts + round_to, int(next_acts * growth_factor))),
            multiple=round_to,
        )
        next_acts = min(next_prompts, next_acts)

    next_prompts, next_acts = prefer_exact_primary_acts_partition(
        n_prompts_in_forward_pass=next_prompts,
        primary_acts_batch_size=next_acts,
        max_prompt_count=bucket_prompt_count,
    )

    acts_label = f"acts{next_acts}" if next_acts is not None else "actsauto"
    bucket_label = config.prompt_bucket_ceiling if config.prompt_bucket_ceiling is not None else "full"
    return SweepConfig(
        label=f"bucket{bucket_label}-{config.n_features_per_batch}x{next_prompts}-{acts_label}",
        n_features_per_batch=config.n_features_per_batch,
        n_prompts_in_forward_pass=next_prompts,
        primary_acts_batch_size=next_acts,
        prompt_bucket_ceiling=config.prompt_bucket_ceiling,
    )


def load_bucket_manifest(bucket_manifest_path: Path) -> dict:
    """Load a staged prompt bucket manifest from disk."""

    payload = json.loads(bucket_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Bucket manifest must decode to a mapping: {bucket_manifest_path}")
    buckets = payload.get("buckets")
    if not isinstance(buckets, list):
        raise ValueError(f"Bucket manifest is missing a valid buckets list: {bucket_manifest_path}")
    return payload


def select_bucket_entries(manifest: dict, requested_ceilings: list[int] | None = None) -> list[dict]:
    """Return requested bucket entries, or all staged buckets when none are explicitly requested."""

    buckets = [bucket for bucket in manifest.get("buckets", []) if isinstance(bucket, dict)]
    if not requested_ceilings:
        return buckets

    requested = {int(value) for value in requested_ceilings}
    selected = [bucket for bucket in buckets if int(bucket.get("upper_inclusive", -1)) in requested]
    if len(selected) != len(requested):
        available = [int(bucket.get("upper_inclusive", -1)) for bucket in buckets]
        missing = sorted(requested.difference({int(bucket.get("upper_inclusive", -1)) for bucket in selected}))
        raise ValueError(f"Requested bucket ceilings are unavailable: missing={missing} available={available}")
    return selected


def candidate_gain_is_worthwhile(
    baseline: SweepResult,
    candidate: SweepResult,
    *,
    min_throughput_gain_fraction: float,
    max_vram_growth_per_gain: float,
) -> bool:
    """Return whether a faster candidate is efficient enough to justify higher VRAM use."""

    if baseline.throughput_features_per_min is None or candidate.throughput_features_per_min is None:
        return False
    if baseline.throughput_features_per_min <= 0:
        return True

    throughput_gain_fraction = (
        candidate.throughput_features_per_min - baseline.throughput_features_per_min
    ) / baseline.throughput_features_per_min
    if throughput_gain_fraction <= 0:
        return False

    baseline_vram = baseline.max_gpu_process_used_mib
    candidate_vram = candidate.max_gpu_process_used_mib
    if baseline_vram <= 0 or candidate_vram <= 0:
        return throughput_gain_fraction >= min_throughput_gain_fraction

    vram_growth_fraction = max(0.0, (candidate_vram - baseline_vram) / baseline_vram)
    if throughput_gain_fraction >= min_throughput_gain_fraction:
        return vram_growth_fraction <= throughput_gain_fraction * max_vram_growth_per_gain
    return vram_growth_fraction <= throughput_gain_fraction * max_vram_growth_per_gain


def result_exceeds_target_vram(
    result: SweepResult,
    *,
    target_max_gpu_process_used_mib: int | None,
) -> bool:
    """Return whether a probe exceeded the optional target GPU-process memory budget."""

    if target_max_gpu_process_used_mib is None or target_max_gpu_process_used_mib <= 0:
        return False
    if result.max_gpu_process_used_mib <= 0:
        return False
    return result.max_gpu_process_used_mib > target_max_gpu_process_used_mib


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def _coerce_log_value(value: str) -> Any:
    if value == "None":
        return None
    if value[:1] in {"{", "[", '"', "-"} or value[:1].isdigit() or value in {"true", "false", "null"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_prefixed_kv_line(line: str, prefix: str) -> dict[str, Any] | None:
    if not line.startswith(prefix):
        return None
    fields: dict[str, Any] = {}
    for token in line[len(prefix) :].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = _coerce_log_value(value)
    return fields


def summarize_feature_data_metrics(
    feature_data_summaries: list[dict[str, Any]],
    activation_and_encode_wall_s: list[float],
    batch_total_wall_s: list[float],
    *,
    completed_batch_count: int,
    summary_warmup_batches: int = 0,
) -> FeatureDataSummaryMetrics:
    summary_limit = completed_batch_count if completed_batch_count > 0 else len(feature_data_summaries)
    summary_start = min(summary_warmup_batches, summary_limit)
    selected_summaries = feature_data_summaries[summary_start:summary_limit] if summary_limit > 0 else []
    selected_feature_data_wall_s = (
        activation_and_encode_wall_s[summary_start:completed_batch_count]
        if completed_batch_count > 0
        else activation_and_encode_wall_s
    )
    selected_batch_total_wall_s = (
        batch_total_wall_s[summary_start:completed_batch_count] if completed_batch_count > 0 else batch_total_wall_s
    )

    forward_pass_counts = [
        int(value)
        for summary in selected_summaries
        if isinstance((value := summary.get("model_forward_passes")), (int, float))
    ]
    total_forward_wall_values = [
        float(value)
        for summary in selected_summaries
        if isinstance((value := summary.get("total_forward_wall_s")), (int, float))
    ]
    total_get_feature_data_wall_s: float | None
    avg_get_feature_data_wall_s: float | None
    if selected_feature_data_wall_s:
        total_get_feature_data_wall_s = sum(selected_feature_data_wall_s)
        avg_get_feature_data_wall_s = total_get_feature_data_wall_s / len(selected_feature_data_wall_s)
    else:
        get_feature_data_wall_values = [
            float(value)
            for summary in selected_summaries
            if isinstance((value := summary.get("get_feature_data_wall_s")), (int, float))
        ]
        total_get_feature_data_wall_s = sum(get_feature_data_wall_values) if get_feature_data_wall_values else None
        avg_get_feature_data_wall_s = (
            total_get_feature_data_wall_s / len(get_feature_data_wall_values)
            if get_feature_data_wall_values and total_get_feature_data_wall_s is not None
            else None
        )

    total_model_forward_passes = sum(forward_pass_counts) if forward_pass_counts else None
    total_model_forward_wall_s = sum(total_forward_wall_values) if total_forward_wall_values else None
    avg_model_forward_wall_s = (
        total_model_forward_wall_s / total_model_forward_passes
        if total_model_forward_passes and total_model_forward_wall_s is not None
        else None
    )

    get_feature_data_share_values = [
        feature_wall_s / batch_wall_s
        for feature_wall_s, batch_wall_s in zip(selected_feature_data_wall_s, selected_batch_total_wall_s)
        if batch_wall_s > 0
    ]
    avg_get_feature_data_share_of_batch = (
        sum(get_feature_data_share_values) / len(get_feature_data_share_values)
        if get_feature_data_share_values
        else None
    )

    peak_get_feature_data_rss_gib = max(
        [
            float(value)
            for summary in selected_summaries
            if isinstance((value := summary.get("peak_rss_gib")), (int, float))
        ],
        default=None,
    )
    peak_get_feature_data_cuda_allocated_gib = max(
        [
            float(value)
            for summary in selected_summaries
            if isinstance((value := summary.get("peak_cuda_allocated_gib")), (int, float))
        ],
        default=None,
    )
    peak_get_feature_data_cuda_reserved_gib = max(
        [
            float(value)
            for summary in selected_summaries
            if isinstance((value := summary.get("peak_cuda_reserved_gib")), (int, float))
        ],
        default=None,
    )

    return FeatureDataSummaryMetrics(
        total_model_forward_passes=total_model_forward_passes,
        avg_model_forward_wall_s=avg_model_forward_wall_s,
        total_model_forward_wall_s=total_model_forward_wall_s,
        avg_get_feature_data_wall_s=avg_get_feature_data_wall_s,
        total_get_feature_data_wall_s=total_get_feature_data_wall_s,
        avg_get_feature_data_share_of_batch=avg_get_feature_data_share_of_batch,
        peak_get_feature_data_rss_gib=peak_get_feature_data_rss_gib,
        peak_get_feature_data_cuda_allocated_gib=peak_get_feature_data_cuda_allocated_gib,
        peak_get_feature_data_cuda_reserved_gib=peak_get_feature_data_cuda_reserved_gib,
    )


def gib_from_kib(value_kib: int) -> float:
    """Convert KiB to GiB."""

    return value_kib / (1024**2)


def parse_config_spec(spec: str) -> SweepConfig:
    """Parse a config spec in either label:features:prompts or features:prompts form."""

    parts = spec.split(":")
    if len(parts) == 2:
        features = int(parts[0])
        prompts = int(parts[1])
        label = f"{features}x{prompts}"
    elif len(parts) == 3:
        label = parts[0]
        features = int(parts[1])
        prompts = int(parts[2])
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid config spec '{spec}'. Use features:prompts or label:features:prompts."
        )
    return SweepConfig(label=label, n_features_per_batch=features, n_prompts_in_forward_pass=prompts)


def default_sweep_configs(include_baseline: bool, include_suggestions: bool) -> list[SweepConfig]:
    """Return the default sweep order for the current RTE dashboard workload."""

    configs: list[SweepConfig] = []
    if include_baseline:
        configs.append(SweepConfig(label="baseline-512x32", n_features_per_batch=512, n_prompts_in_forward_pass=32))
    configs.extend(
        [
            SweepConfig(label="1024x32", n_features_per_batch=1024, n_prompts_in_forward_pass=32),
            SweepConfig(label="512x64", n_features_per_batch=512, n_prompts_in_forward_pass=64),
            SweepConfig(label="1024x64", n_features_per_batch=1024, n_prompts_in_forward_pass=64),
            SweepConfig(label="2048x64", n_features_per_batch=2048, n_prompts_in_forward_pass=64),
            SweepConfig(label="2048x128", n_features_per_batch=2048, n_prompts_in_forward_pass=128),
        ]
    )
    if include_suggestions:
        configs.append(SweepConfig(label="suggested-2048x32", n_features_per_batch=2048, n_prompts_in_forward_pass=32))
    return configs


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the dashboard config sweep."""

    parser = argparse.ArgumentParser(
        description=(
            "Run short Neuronpedia dashboard generation probes across batch-shape settings, "
            "collect throughput and resource metrics, and stop each run after a small number of batches "
            "or a safety limit is reached."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config spec in features:prompts or label:features:prompts form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the current 512x32 control run when no explicit configs are provided.",
    )
    parser.add_argument(
        "--include-suggestions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one additional suggested power-of-two config when no explicit configs are provided.",
    )
    parser.add_argument("--layer", type=int, default=2, help="Single layer index to probe. Defaults to 2.")
    parser.add_argument(
        "--target-batches",
        type=int,
        default=4,
        help="Stop each config after this many completed batches. Defaults to 4 for batch-1-through-3 summaries.",
    )
    parser.add_argument(
        "--summary-warmup-batches",
        type=int,
        default=1,
        help="Exclude completed batches below this batch number from reported steady-state averages. Defaults to 1.",
    )
    parser.add_argument(
        "--max-tree-rss-gib",
        type=float,
        default=42.0,
        help="Kill the process tree if its sampled RSS exceeds this GiB threshold. Defaults to 42.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval in seconds for log parsing and resource sampling. Defaults to 2.",
    )
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=600.0,
        help="Kill a run if it completes no new batch for this many seconds. Defaults to 600.",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=DEFAULT_BASE_OUTPUT_DIR,
        help="Directory under which the sweep session outputs will be created.",
    )
    parser.add_argument(
        "--cleanup-run-dirs",
        action="store_true",
        help="Delete the per-config run directories after metrics are captured.",
    )
    parser.add_argument(
        "--python-executable",
        default=DEFAULT_PYTHON,
        help=f"Python interpreter for the dashboard pipeline. Defaults to {DEFAULT_PYTHON}.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Shared dashboard run root to mirror the live environment defaults.",
    )
    parser.add_argument(
        "--prompts-pretokenized-dataset-path",
        type=Path,
        help="Optional local HuggingFace dataset with an input_ids column to pass through to the dashboard runner.",
    )
    parser.add_argument(
        "--bucket-manifest-path",
        type=Path,
        help="Optional staged prompt bucket manifest for bucket-aware pilot probes.",
    )
    parser.add_argument(
        "--bucket-ceiling",
        action="append",
        type=int,
        default=[],
        help=(
            "Prompt bucket ceiling to probe. Can be passed multiple times. "
            "Defaults to all staged buckets when a manifest is provided."
        ),
    )
    parser.add_argument(
        "--base-tokens-in-prompt",
        type=int,
        default=319,
        help=("Full-context token count used as the scaling reference for bucket-aware probes. Defaults to 319."),
    )
    parser.add_argument(
        "--base-n-features-per-batch",
        type=int,
        default=1024,
        help="Feature batch size to keep fixed across bucket-aware probes. Defaults to 1024.",
    )
    parser.add_argument(
        "--base-n-prompts-in-forward-pass",
        type=int,
        default=256,
        help=("Full-context logical prompt batch size used to seed bucket-aware probes. Defaults to 256."),
    )
    parser.add_argument(
        "--base-primary-acts-batch-size",
        type=int,
        default=64,
        help=("Full-context primary activation subbatch size used to seed bucket-aware probes. Defaults to 64."),
    )
    parser.add_argument(
        "--prompt-scale-limit",
        type=float,
        default=4.0,
        help=(
            "Maximum multiplicative prompt-batch scale applied when seeding a shorter-context bucket. Defaults to 4.0."
        ),
    )
    parser.add_argument(
        "--primary-acts-scale-limit",
        type=float,
        default=4.0,
        help=(
            "Maximum multiplicative primary-acts scale applied when seeding a shorter-context bucket. Defaults to 4.0."
        ),
    )
    parser.add_argument(
        "--batch-size-round-to",
        type=int,
        default=8,
        help="Round derived batch sizes down to this multiple. Defaults to 8.",
    )
    parser.add_argument(
        "--probe-growth-factor",
        type=float,
        default=1.5,
        help=("Growth factor for the next more aggressive bucket probe after a successful run. Defaults to 1.5."),
    )
    parser.add_argument(
        "--max-probe-attempts-per-bucket",
        type=int,
        default=4,
        help=("Maximum probe attempts for one bucket before settling on the best successful config. Defaults to 4."),
    )
    parser.add_argument(
        "--min-throughput-gain-fraction",
        type=float,
        default=0.03,
        help=(
            "Minimum relative throughput gain that should justify moving away from the current best config. "
            "Defaults to 0.03."
        ),
    )
    parser.add_argument(
        "--max-vram-growth-per-gain",
        type=float,
        default=8.0,
        help="Maximum allowed VRAM growth fraction per unit of throughput gain fraction. Defaults to 8.0.",
    )
    parser.add_argument(
        "--target-max-gpu-process-used-mib",
        type=int,
        default=None,
        help=(
            "Optional target ceiling for per-process GPU memory. Successful probes above this budget are recorded but "
            "will not become the selected config for further aggressive follow-ups."
        ),
    )
    parser.add_argument(
        "--runner-use-cached-activations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse runner activation caches during sweep probes. Defaults to disabled so forward-pass-aware sweeps "
            "still measure real model execution."
        ),
    )
    parser.add_argument(
        "--same-layer-fixed-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In bucket pilot mode, also run one same-layer full-context fixed-shape control using the base_* "
            "settings before the bucket-specific probes."
        ),
    )
    return parser


def read_meminfo() -> dict[str, int]:
    """Read /proc/meminfo values in KiB."""

    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", maxsplit=1)
        meminfo[key] = int(raw_value.strip().split()[0])
    return meminfo


def parse_ps_table() -> dict[int, tuple[int, int, float, str]]:
    """Return pid -> (ppid, rss_kib, cpu_pct, cmd) for the current process table."""

    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,rss=,%cpu=,cmd="],
        capture_output=True,
        text=True,
        check=False,
    )
    process_table: dict[int, tuple[int, int, float, str]] = {}
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_str, ppid_str, rss_str, cpu_str, cmd = stripped.split(None, 4)
        process_table[int(pid_str)] = (int(ppid_str), int(rss_str), float(cpu_str), cmd)
    return process_table


def descendant_pids(root_pid: int, process_table: dict[int, tuple[int, int, float, str]]) -> set[int]:
    """Return the root pid and all of its descendants present in the current process table."""

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _, _) in process_table.items():
            if pid in descendants:
                continue
            if ppid in descendants:
                descendants.add(pid)
                changed = True
    return {pid for pid in descendants if pid in process_table}


def sample_tree_metrics(root_pid: int) -> tuple[float, float, float]:
    """Return tree RSS GiB, summed CPU percent, and host used GiB for the root process tree."""

    process_table = parse_ps_table()
    tree_pids = descendant_pids(root_pid, process_table)
    total_rss_kib = sum(process_table[pid][1] for pid in tree_pids)
    total_cpu_pct = sum(process_table[pid][2] for pid in tree_pids)
    meminfo = read_meminfo()
    host_used_kib = meminfo["MemTotal"] - meminfo["MemAvailable"]
    return gib_from_kib(total_rss_kib), total_cpu_pct, gib_from_kib(host_used_kib)


def sample_gpu_metrics(root_pid: int) -> tuple[int, int, int]:
    """Return process GPU MiB, device GPU MiB, and device util percent maxima for the process tree."""

    process_table = parse_ps_table()
    tree_pids = descendant_pids(root_pid, process_table)

    query_gpus = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    max_device_used_mib = 0
    max_device_util_pct = 0
    for line in query_gpus.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        _, used_mib, util_pct = [part.strip() for part in stripped.split(",")[:3]]
        max_device_used_mib = max(max_device_used_mib, int(used_mib))
        max_device_util_pct = max(max_device_util_pct, int(util_pct))

    query_apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    process_used_mib = 0
    for line in query_apps.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_str, used_mib = [part.strip() for part in stripped.split(",")[:2]]
        pid = int(pid_str)
        if pid in tree_pids:
            process_used_mib += int(used_mib)

    return process_used_mib, max_device_used_mib, max_device_util_pct


def build_env() -> dict[str, str]:
    """Build the environment for short dashboard probe runs."""

    env = os.environ.copy()
    hf_home = Path(env.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    env.setdefault("IT_NP_CACHE", str(NP_CACHE_ROOT))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def resolve_pretokenized_dataset_path(
    *,
    bucket_manifest_path: Path | None,
    pretokenized_dataset_path: Path | None,
) -> Path | None:
    """Return the dataset path to stage prompt buckets, inferring it from the manifest when needed."""

    if pretokenized_dataset_path is not None:
        return pretokenized_dataset_path
    if bucket_manifest_path is not None:
        return bucket_manifest_path.parent
    return None


def build_command(
    config: SweepConfig,
    *,
    layer: int,
    python_executable: str,
    run_root: Path,
    pretokenized_dataset_path: Path | None = None,
    runner_use_cached_activations: bool = False,
) -> list[str]:
    """Build the dashboard pipeline command for one short probe run."""

    command = [
        python_executable,
        "-m",
        "interpretune.utils.neuronpedia_dashboard_pipeline",
        "--model-name",
        DEFAULT_MODEL_NAME,
        "--model-layers",
        "26",
        "--sae-set",
        "gemma-scope-2-1b-it-transcoders-all",
        "--neuronpedia-source-set-id",
        DEFAULT_SOURCE_SET_ID,
        "--neuronpedia-source-set-description",
        "Transcoder - 262k (RTE)",
        "--creator-name",
        "Google DeepMind",
        "--release-id",
        "gemma-scope-2",
        "--release-title",
        "Gemma Scope 2",
        "--release-url",
        "https://huggingface.co/google/gemma-scope-2-1b-it",
        "--hf-weights-repo-id",
        "google/gemma-scope-2-1b-it",
        "--hf-weights-path-template",
        "transcoder_all/layer_{layer}_width_262k_l0_small_affine",
        "--hook-point",
        "hook_mlp_in",
        "--prompts-huggingface-dataset-path",
        "aps/super_glue",
        "--prompts-huggingface-dataset-config-name",
        "rte",
        "--prompts-huggingface-dataset-split",
        "train",
        "--model-wrapper",
        "bridge",
        "--bridge-enable-compatibility-mode",
        "--runner-log-performance",
        "--runner-log-resource-snapshots",
        "--start-layer",
        str(layer),
        "--end-layer",
        str(layer),
        "--start-batch",
        "0",
        "--n-prompts-total",
        "2490",
        "--n-features-per-batch",
        str(config.n_features_per_batch),
        "--n-prompts-in-forward-pass",
        str(config.n_prompts_in_forward_pass),
        "--no-archive-partials",
        "--sae-path-template",
        "layer_{layer}_width_262k_l0_small_affine",
        "--python-executable",
        python_executable,
        "--cuda-visible-devices",
        "0",
        "--heartbeat-seconds",
        "60",
        "--stall-timeout-seconds",
        "1800",
        "--use-skip-transcoder",
        "--skip-local-db-import",
        "--run-root",
        str(run_root),
    ]
    command.append(
        "--runner-use-cached-activations" if runner_use_cached_activations else "--no-runner-use-cached-activations"
    )
    if pretokenized_dataset_path is not None:
        command.extend(["--prompts-pretokenized-dataset-path", str(pretokenized_dataset_path)])
    if config.primary_acts_batch_size is not None:
        command.extend(["--primary-acts-batch-size", str(config.primary_acts_batch_size)])
    if config.prompt_bucket_ceiling is not None:
        command.extend(["--prompt-bucket-ceiling", str(config.prompt_bucket_ceiling)])
    return command


def terminate_process_group(process: subprocess.Popen[str], reason: str) -> None:
    """Terminate the process group started for a probe run."""

    if process.poll() is not None:
        return
    print(f"[{utc_now_iso()}] terminating pid={process.pid} reason={reason}", flush=True)
    os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)


def classify_error_lines(lines: Iterable[str]) -> tuple[str | None, list[str]]:
    """Classify notable error markers from newly appended log lines."""

    captured: list[str] = []
    classification: str | None = None
    for line in lines:
        if any(marker in line for marker in OOM_MARKERS):
            classification = "gpu_oom"
            captured.append(line.strip())
        elif any(marker in line for marker in SHAPE_MISMATCH_MARKERS):
            classification = classification or "shape_mismatch"
            captured.append(line.strip())
        elif "Traceback (most recent call last)" in line or "RuntimeError:" in line:
            captured.append(line.strip())
    return classification, captured[-8:]


def read_new_log_lines(log_path: Path, offset: int) -> tuple[list[str], int]:
    """Read newly appended log lines from the pipeline log path."""

    if not log_path.exists():
        return [], offset
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        return lines, handle.tell()


def summarize_batches(
    batch_records: list[BatchRecord],
    config: SweepConfig,
    *,
    summary_warmup_batches: int = 0,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return avg batch seconds, features/min, max post-batch RSS, and max post-batch CUDA max GiB."""

    selected_records = [record for record in batch_records if record.batch_num >= summary_warmup_batches]
    avg_batch_seconds: float | None = None
    throughput_features_per_min: float | None = None
    if len(selected_records) >= 2:
        intervals = [
            later.detected_monotonic - earlier.detected_monotonic
            for earlier, later in zip(selected_records, selected_records[1:], strict=False)
        ]
        avg_batch_seconds = sum(intervals) / len(intervals)
        throughput_features_per_min = (config.n_features_per_batch * 60.0) / avg_batch_seconds

    rss_values = [record.post_batch_rss_gib for record in selected_records if record.post_batch_rss_gib is not None]
    cuda_values = [
        record.post_batch_cuda_max_gib for record in selected_records if record.post_batch_cuda_max_gib is not None
    ]
    peak_post_batch_rss = max(rss_values) if rss_values else None
    peak_post_batch_cuda = max(cuda_values) if cuda_values else None
    return avg_batch_seconds, throughput_features_per_min, peak_post_batch_rss, peak_post_batch_cuda


def write_results(session_dir: Path, results: list[SweepResult]) -> None:
    """Persist the sweep results as JSON and markdown summaries."""

    results_path = session_dir / "results.json"
    results_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")

    markdown_lines = [
        "# Dashboard Sweep Results",
        "",
        (
            "| Config | Status | Batches | Avg Batch s | Features/min | Fwd Passes | Avg Fwd s | "
            "Avg get_feature_data s | get_feature_data/batch | Max Tree RSS GiB | Max Tree CPU % | "
            "Max Host Used GiB | Max GPU Proc MiB | Max GPU Dev MiB | Max GPU Util % | Notes |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        notes = "; ".join(result.notes) if result.notes else ""
        markdown_lines.append(
            (
                "| {label} | {status} | {batches} | {avg_batch} | {features_per_min} | "
                "{forward_passes} | {avg_forward_s} | {avg_get_feature_data_s} | "
                "{get_feature_data_share} | {tree_rss} | {tree_cpu} | {host_used} | "
                "{gpu_proc} | {gpu_dev} | {gpu_util} | {notes} |"
            ).format(
                label=result.label,
                status=result.status,
                batches=len(result.completed_batches),
                avg_batch=f"{result.avg_batch_seconds:.1f}" if result.avg_batch_seconds is not None else "-",
                features_per_min=(
                    f"{result.throughput_features_per_min:.0f}"
                    if result.throughput_features_per_min is not None
                    else "-"
                ),
                forward_passes=result.total_model_forward_passes
                if result.total_model_forward_passes is not None
                else "-",
                avg_forward_s=(
                    f"{result.avg_model_forward_wall_s:.3f}" if result.avg_model_forward_wall_s is not None else "-"
                ),
                avg_get_feature_data_s=(
                    f"{result.avg_get_feature_data_wall_s:.3f}"
                    if result.avg_get_feature_data_wall_s is not None
                    else "-"
                ),
                get_feature_data_share=(
                    f"{result.avg_get_feature_data_share_of_batch:.2%}"
                    if result.avg_get_feature_data_share_of_batch is not None
                    else "-"
                ),
                tree_rss=f"{result.max_tree_rss_gib:.2f}",
                tree_cpu=f"{result.max_tree_cpu_percent:.1f}",
                host_used=f"{result.max_host_used_gib:.2f}",
                gpu_proc=result.max_gpu_process_used_mib,
                gpu_dev=result.max_gpu_device_used_mib,
                gpu_util=result.max_gpu_util_percent,
                notes=notes,
            )
        )
    (session_dir / "results.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def write_selected_bucket_configs(
    session_dir: Path,
    *,
    layer: int,
    device: str,
    selections: list[dict],
    target_max_gpu_process_used_mib: int | None,
    runner_use_cached_activations: bool,
    same_layer_fixed_baseline: dict[str, Any] | None,
) -> None:
    """Persist the selected per-bucket probe settings for one layer/device pilot."""

    payload = {
        "layer": layer,
        "device": device,
        "selected_at_utc": utc_now_iso(),
        "bucket_count": len(selections),
        "total_prompt_count": sum(int(selection.get("prompt_count", 0)) for selection in selections),
        "target_max_gpu_process_used_mib": target_max_gpu_process_used_mib,
        "runner_use_cached_activations": runner_use_cached_activations,
        "same_layer_fixed_baseline": same_layer_fixed_baseline,
        "buckets": selections,
        "future_note": (
            "A target GPU VRAM budget could be added later to constrain candidate generation directly instead of "
            "using only the current throughput-vs-VRAM tradeoff heuristic, while still allowing larger prompt and "
            "primary-acts batches as later substage optimizations improve the useful VRAM envelope."
        ),
    }
    (session_dir / "selected_bucket_configs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_probe(
    config: SweepConfig,
    *,
    args: argparse.Namespace,
    session_dir: Path,
    env: dict[str, str],
) -> SweepResult:
    """Run one probe configuration and collect throughput/resource metrics."""

    config_dir = session_dir / config.label
    run_root = config_dir / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    pipeline_log_path = run_root / DEFAULT_RUN_NAME / f"run.resume-{args.layer}-{args.layer}.log"
    command = build_command(
        config,
        layer=args.layer,
        python_executable=args.python_executable,
        run_root=run_root,
        pretokenized_dataset_path=args.prompts_pretokenized_dataset_path,
        runner_use_cached_activations=args.runner_use_cached_activations,
    )
    stdout_path = config_dir / "outer.stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    started_at_utc = utc_now_iso()
    started_monotonic = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    print(f"[{started_at_utc}] started {config.label}: {shlex.join(command)}", flush=True)

    log_offset = 0
    completed_batches: dict[int, BatchRecord] = {}
    notes: list[str] = []
    last_batch_progress_monotonic = started_monotonic
    last_error_lines: list[str] = []
    classified_error: str | None = None
    status = "running"
    reason = ""
    feature_data_summaries: list[dict[str, Any]] = []
    activation_and_encode_wall_s: list[float] = []
    batch_total_wall_s: list[float] = []

    max_tree_rss_gib = 0.0
    max_tree_cpu_percent = 0.0
    max_host_used_gib = 0.0
    max_gpu_process_used_mib = 0
    max_gpu_device_used_mib = 0
    max_gpu_util_percent = 0

    try:
        while True:
            time.sleep(args.poll_seconds)

            new_lines, log_offset = read_new_log_lines(pipeline_log_path, log_offset)
            for line in new_lines:
                batch_match = BATCH_OUTPUT_RE.search(line)
                if batch_match:
                    batch_num = int(batch_match.group(1))
                    completed_batches.setdefault(
                        batch_num,
                        BatchRecord(
                            batch_num=batch_num,
                            detected_at_utc=utc_now_iso(),
                            detected_monotonic=time.monotonic(),
                        ),
                    )
                    last_batch_progress_monotonic = time.monotonic()

                post_match = POST_BATCH_RE.search(line)
                if post_match:
                    batch_num = int(post_match.group(1))
                    record = completed_batches.setdefault(
                        batch_num,
                        BatchRecord(
                            batch_num=batch_num,
                            detected_at_utc=utc_now_iso(),
                            detected_monotonic=time.monotonic(),
                        ),
                    )
                    record.post_batch_rss_gib = float(post_match.group(2))
                    record.post_batch_cuda_max_gib = float(post_match.group(3))

                perf_fields = parse_prefixed_kv_line(line, RUNNER_PERF_PREFIX)
                if perf_fields is not None:
                    event = perf_fields.get("event")
                    if event == "get_feature_data_summary":
                        feature_data_summaries.append(perf_fields)
                    elif event == "stage_timing" and perf_fields.get("stage") == "activation_and_encode_total":
                        wall_s = perf_fields.get("wall_s")
                        if isinstance(wall_s, (int, float)):
                            activation_and_encode_wall_s.append(float(wall_s))
                    elif event == "batch_total":
                        wall_s = perf_fields.get("wall_s")
                        if isinstance(wall_s, (int, float)):
                            batch_total_wall_s.append(float(wall_s))

            error_kind, error_lines = classify_error_lines(new_lines)
            if error_lines:
                last_error_lines = error_lines
            if error_kind is not None:
                classified_error = error_kind

            if process.poll() is None:
                tree_rss_gib, tree_cpu_percent, host_used_gib = sample_tree_metrics(process.pid)
                max_tree_rss_gib = max(max_tree_rss_gib, tree_rss_gib)
                max_tree_cpu_percent = max(max_tree_cpu_percent, tree_cpu_percent)
                max_host_used_gib = max(max_host_used_gib, host_used_gib)

                gpu_process_used_mib, gpu_device_used_mib, gpu_util_percent = sample_gpu_metrics(process.pid)
                max_gpu_process_used_mib = max(max_gpu_process_used_mib, gpu_process_used_mib)
                max_gpu_device_used_mib = max(max_gpu_device_used_mib, gpu_device_used_mib)
                max_gpu_util_percent = max(max_gpu_util_percent, gpu_util_percent)

                if tree_rss_gib > args.max_tree_rss_gib:
                    status = "killed"
                    reason = f"tree_rss_limit>{args.max_tree_rss_gib:.2f}GiB"
                    notes.append(f"Killed at sampled tree RSS {tree_rss_gib:.2f} GiB.")
                    terminate_process_group(process, reason)
                    break

                if len(completed_batches) >= args.target_batches:
                    status = "target_reached"
                    reason = f"completed_{args.target_batches}_batches"
                    notes.append(f"Stopped after {args.target_batches} completed batches.")
                    terminate_process_group(process, reason)
                    break

                stalled_for = time.monotonic() - last_batch_progress_monotonic
                if args.stall_seconds > 0 and stalled_for > args.stall_seconds:
                    status = "killed"
                    reason = f"stall_timeout>{args.stall_seconds:.0f}s"
                    notes.append(f"Killed after {stalled_for:.1f}s without a new completed batch.")
                    terminate_process_group(process, reason)
                    break
                continue

            break
    finally:
        exit_code = process.wait()

    if status == "running":
        if exit_code == 0:
            status = "exited"
            reason = "exit_0"
        elif classified_error == "gpu_oom":
            status = "failed"
            reason = "gpu_oom"
            notes.append("Process exited after a GPU OOM marker in the log.")
        elif classified_error == "shape_mismatch":
            status = "failed"
            reason = "shape_mismatch"
            notes.append("Process exited after the ignore-mask shape-mismatch seam.")
        else:
            status = "failed"
            reason = f"exit_{exit_code}"
            if last_error_lines:
                notes.append(last_error_lines[-1])

    sorted_records = [completed_batches[batch_num] for batch_num in sorted(completed_batches)]
    avg_batch_seconds, throughput_features_per_min, peak_post_batch_rss, peak_post_batch_cuda = summarize_batches(
        sorted_records,
        config,
        summary_warmup_batches=args.summary_warmup_batches,
    )
    feature_data_metrics = summarize_feature_data_metrics(
        feature_data_summaries,
        activation_and_encode_wall_s,
        batch_total_wall_s,
        completed_batch_count=len(sorted_records),
        summary_warmup_batches=args.summary_warmup_batches,
    )
    if args.summary_warmup_batches > 0:
        notes.append(f"Reported averages exclude completed batches before {args.summary_warmup_batches}.")

    ended_at_utc = utc_now_iso()
    result = SweepResult(
        label=config.label,
        n_features_per_batch=config.n_features_per_batch,
        n_prompts_in_forward_pass=config.n_prompts_in_forward_pass,
        status=status,
        reason=reason,
        exit_code=exit_code,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        elapsed_seconds=time.monotonic() - started_monotonic,
        completed_batches=[record.batch_num for record in sorted_records],
        avg_batch_seconds=avg_batch_seconds,
        throughput_features_per_min=throughput_features_per_min,
        max_tree_rss_gib=max_tree_rss_gib,
        max_tree_cpu_percent=max_tree_cpu_percent,
        max_host_used_gib=max_host_used_gib,
        max_gpu_process_used_mib=max_gpu_process_used_mib,
        max_gpu_device_used_mib=max_gpu_device_used_mib,
        max_gpu_util_percent=max_gpu_util_percent,
        total_model_forward_passes=feature_data_metrics.total_model_forward_passes,
        avg_model_forward_wall_s=feature_data_metrics.avg_model_forward_wall_s,
        total_model_forward_wall_s=feature_data_metrics.total_model_forward_wall_s,
        avg_get_feature_data_wall_s=feature_data_metrics.avg_get_feature_data_wall_s,
        total_get_feature_data_wall_s=feature_data_metrics.total_get_feature_data_wall_s,
        avg_get_feature_data_share_of_batch=feature_data_metrics.avg_get_feature_data_share_of_batch,
        peak_get_feature_data_rss_gib=feature_data_metrics.peak_get_feature_data_rss_gib,
        peak_get_feature_data_cuda_allocated_gib=feature_data_metrics.peak_get_feature_data_cuda_allocated_gib,
        peak_get_feature_data_cuda_reserved_gib=feature_data_metrics.peak_get_feature_data_cuda_reserved_gib,
        peak_post_batch_rss_gib=peak_post_batch_rss,
        peak_post_batch_cuda_max_gib=peak_post_batch_cuda,
        pipeline_log_path=str(pipeline_log_path),
        run_root=str(run_root),
        command=shlex.join(command),
        notes=notes + last_error_lines,
    )

    if args.cleanup_run_dirs:
        shutil.rmtree(run_root, ignore_errors=True)

    print(
        f"[{ended_at_utc}] finished {config.label}: status={result.status} reason={result.reason} "
        f"batches={len(result.completed_batches)} avg_batch_seconds={result.avg_batch_seconds} "
        f"throughput_features_per_min={result.throughput_features_per_min} "
        f"max_tree_rss_gib={result.max_tree_rss_gib:.2f} "
        f"max_gpu_process_used_mib={result.max_gpu_process_used_mib}",
        flush=True,
    )
    return result


def run_bucket_pilot(args: argparse.Namespace, *, session_dir: Path, env: dict[str, str]) -> list[SweepResult]:
    """Run one bucket-aware pilot sweep with OOM-aware fallback and persisted selected configs."""

    manifest = load_bucket_manifest(args.bucket_manifest_path)
    bucket_entries = select_bucket_entries(manifest, args.bucket_ceiling)
    results: list[SweepResult] = []
    selected_configs: list[dict] = []
    same_layer_fixed_baseline_payload: dict[str, Any] | None = None

    if args.same_layer_fixed_baseline:
        fixed_config = build_same_layer_fixed_baseline_config(args)
        fixed_result = run_probe(fixed_config, args=args, session_dir=session_dir, env=env)
        fixed_result.notes.append("same_layer_fixed_baseline")
        results.append(fixed_result)
        write_results(session_dir, results)
        same_layer_fixed_baseline_payload = {
            "config": asdict(fixed_config),
            "result": asdict(fixed_result),
        }

    for bucket in bucket_entries:
        bucket_ceiling = int(bucket["upper_inclusive"])
        bucket_prompt_count = int(bucket["prompt_count"])
        current_config: SweepConfig | None = build_bucket_seed_config(
            label_prefix=f"gpu{env.get('CUDA_VISIBLE_DEVICES', '0')}-layer{args.layer}",
            n_features_per_batch=args.base_n_features_per_batch,
            base_tokens_in_prompt=args.base_tokens_in_prompt,
            base_n_prompts_in_forward_pass=args.base_n_prompts_in_forward_pass,
            base_primary_acts_batch_size=args.base_primary_acts_batch_size,
            bucket_ceiling=bucket_ceiling,
            bucket_prompt_count=bucket_prompt_count,
            prompt_scale_limit=args.prompt_scale_limit,
            primary_acts_scale_limit=args.primary_acts_scale_limit,
            round_to=args.batch_size_round_to,
        )
        best_result: SweepResult | None = None
        best_config: SweepConfig | None = None
        attempts = 0

        while attempts < args.max_probe_attempts_per_bucket and current_config is not None:
            attempts += 1
            result = run_probe(current_config, args=args, session_dir=session_dir, env=env)
            result.notes.append(f"bucket_ceiling={bucket_ceiling}")
            if result_exceeds_target_vram(
                result,
                target_max_gpu_process_used_mib=args.target_max_gpu_process_used_mib,
            ):
                result.notes.append(f"Exceeded target_max_gpu_process_used_mib={args.target_max_gpu_process_used_mib}.")
            results.append(result)
            write_results(session_dir, results)

            if result.status == "failed" and result.reason == "gpu_oom":
                current_config = build_more_conservative_bucket_candidate(
                    current_config,
                    round_to=args.batch_size_round_to,
                )
                continue

            if result.status not in {"target_reached", "exited"}:
                break

            if best_result is None:
                best_result = result
                best_config = current_config
                if result_exceeds_target_vram(
                    result,
                    target_max_gpu_process_used_mib=args.target_max_gpu_process_used_mib,
                ):
                    break
                current_config = build_more_aggressive_bucket_candidate(
                    current_config,
                    bucket_prompt_count=bucket_prompt_count,
                    growth_factor=args.probe_growth_factor,
                    round_to=args.batch_size_round_to,
                )
                continue

            if result_exceeds_target_vram(
                result,
                target_max_gpu_process_used_mib=args.target_max_gpu_process_used_mib,
            ):
                break

            if candidate_gain_is_worthwhile(
                best_result,
                result,
                min_throughput_gain_fraction=args.min_throughput_gain_fraction,
                max_vram_growth_per_gain=args.max_vram_growth_per_gain,
            ):
                best_result = result
                best_config = current_config
                current_config = build_more_aggressive_bucket_candidate(
                    current_config,
                    bucket_prompt_count=bucket_prompt_count,
                    growth_factor=args.probe_growth_factor,
                    round_to=args.batch_size_round_to,
                )
                continue

            break

        selected_configs.append(
            {
                "bucket_ceiling": bucket_ceiling,
                "prompt_count": bucket_prompt_count,
                "effective_length_min": int(bucket.get("effective_length_min", bucket_ceiling)),
                "effective_length_max": int(bucket.get("effective_length_max", bucket_ceiling)),
                "selected_config": asdict(best_config) if best_config is not None else None,
                "selected_result": asdict(best_result) if best_result is not None else None,
                "attempts": attempts,
            }
        )

    write_selected_bucket_configs(
        session_dir,
        layer=args.layer,
        device=str(env.get("CUDA_VISIBLE_DEVICES", "0")),
        selections=selected_configs,
        target_max_gpu_process_used_mib=args.target_max_gpu_process_used_mib,
        runner_use_cached_activations=args.runner_use_cached_activations,
        same_layer_fixed_baseline=same_layer_fixed_baseline_payload,
    )
    return results


def main() -> int:
    """Run the requested dashboard configuration sweep."""

    parser = build_parser()
    args = parser.parse_args()

    configs = [parse_config_spec(spec) for spec in args.config]
    if not configs:
        configs = default_sweep_configs(args.include_baseline, args.include_suggestions)
    if not configs:
        parser.error("No configs selected.")

    session_dir = args.session_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    env = build_env()
    args.prompts_pretokenized_dataset_path = resolve_pretokenized_dataset_path(
        bucket_manifest_path=args.bucket_manifest_path,
        pretokenized_dataset_path=args.prompts_pretokenized_dataset_path,
    )

    print(f"Session directory: {session_dir}", flush=True)
    results: list[SweepResult] = []
    if args.bucket_manifest_path is not None:
        print(
            (
                f"Bucket pilot mode: manifest={args.bucket_manifest_path} "
                f"requested_ceilings={args.bucket_ceiling or 'all'}"
            ),
            flush=True,
        )
        results = run_bucket_pilot(args, session_dir=session_dir, env=env)
    else:
        print(f"Configs: {[config.label for config in configs]}", flush=True)
        for config in configs:
            result = run_probe(config, args=args, session_dir=session_dir, env=env)
            results.append(result)
            write_results(session_dir, results)

    print(f"Wrote results to {session_dir / 'results.json'} and {session_dir / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
