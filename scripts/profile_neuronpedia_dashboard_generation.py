from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
from typing import Any, Iterable

from interpretune.utils import neuronpedia_dashboard_pipeline as dashboard_pipeline
from interpretune.utils.neuronpedia_db_utils import resolve_local_neuronpedia_db_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_NAME = "gemma-3-1b-it"
DEFAULT_SOURCE_SET_ID = "gemmascope-2-transcoder-262k-rte"
DEFAULT_RUN_NAME = f"{DEFAULT_MODEL_NAME}_{DEFAULT_SOURCE_SET_ID}"
DEFAULT_SESSION_ROOT = Path("/tmp/np_dashboard_generation_profiles")
DEFAULT_PYTHON = "/mnt/cache/speediedan/.venvs/it_latest/bin/python"
DEFAULT_PY_SPY = "/mnt/cache/speediedan/.venvs/it_latest/bin/py-spy"
DEFAULT_RUN_ROOT = Path("/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs")
DEFAULT_PRETOKENIZED_DATASET = Path(
    "/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/"
    "gemma-3-1b-it_rte_boolq_context128"
)
DEFAULT_CACHE_PATH = Path("/mnt/cache_extended")

BATCH_OUTPUT_RE = re.compile(r"Output written to .*?/batch-(\d+)\.json")
COLUMNAR_BATCH_OUTPUT_RE = re.compile(r"\bevent=columnar_output_summary\b.*?/batch-(\d+)\.columnar\b")
COLUMNAR_MANIFEST_OUTPUT_RE = re.compile(r"Columnar output written to .*?/batch-(\d+)\.columnar/manifest\.json")
RESOURCE_RE = re.compile(
    r"\[runner_resource\] stage=(\S+) .* rss_gib=([0-9.]+).* cuda_allocated_gib=([0-9.]+).* "
    r"cuda_reserved_gib=([0-9.]+).* cuda_max_allocated_gib=([0-9.]+)"
)
PRE_BATCH_RE = re.compile(r"pre_batch_(\d+)")
POST_BATCH_RE = re.compile(r"post_batch_(\d+)")

OOM_MARKERS = ("CUDA out of memory", "torch.OutOfMemoryError", "out of memory")
SHAPE_MISMATCH_MARKERS = ("The size of tensor a", "must match the size of tensor b")


@dataclass(frozen=True)
class ProfileConfig:
    """One dashboard generation profile configuration."""

    label: str
    n_features_per_batch: int
    n_prompts_in_forward_pass: int


@dataclass
class ProcessIoTotals:
    """Aggregated /proc/<pid>/io counters for a process tree."""

    read_bytes: int = 0
    write_bytes: int = 0
    read_chars: int = 0
    write_chars: int = 0
    syscr: int = 0
    syscw: int = 0
    cancelled_write_bytes: int = 0


@dataclass
class DiskStats:
    """Selected Linux diskstats counters for one block device."""

    reads_completed: int = 0
    sectors_read: int = 0
    ms_reading: int = 0
    writes_completed: int = 0
    sectors_written: int = 0
    ms_writing: int = 0
    ios_in_progress: int = 0
    ms_doing_io: int = 0
    weighted_ms_doing_io: int = 0


@dataclass
class ResourceSample:
    """One process-tree resource sample."""

    elapsed_seconds: float
    timestamp_utc: str
    tree_pid_count: int
    tree_rss_gib: float
    tree_cpu_percent: float
    host_used_gib: float
    gpu_process_used_mib: int
    gpu_device_used_mib: int
    gpu_util_percent: int
    proc_read_mib_s: float | None = None
    proc_write_mib_s: float | None = None
    proc_syscr_s: float | None = None
    proc_syscw_s: float | None = None
    cache_read_mib_s: float | None = None
    cache_write_mib_s: float | None = None
    cache_read_ms_per_op: float | None = None
    cache_write_ms_per_op: float | None = None
    cache_io_util_percent: float | None = None


@dataclass
class RunnerResourceEvent:
    """A resource snapshot emitted by the SAEDashboard runner log."""

    stage: str
    detected_elapsed_seconds: float
    timestamp_utc: str
    rss_gib: float
    cuda_allocated_gib: float
    cuda_reserved_gib: float
    cuda_max_allocated_gib: float


@dataclass
class BatchEvent:
    """Detected generation progress for one batch."""

    batch_num: int
    detected_elapsed_seconds: float
    timestamp_utc: str


@dataclass
class SegmentSummary:
    """Resource and throughput summary for startup or one batch segment."""

    segment: str
    duration_seconds: float | None
    throughput_features_per_min: float | None
    samples: int
    avg_tree_cpu_percent: float | None
    max_tree_cpu_percent: float | None
    max_tree_rss_gib: float | None
    max_gpu_process_used_mib: int | None
    avg_gpu_util_percent: float | None
    max_gpu_util_percent: int | None
    avg_proc_read_mib_s: float | None
    avg_proc_write_mib_s: float | None
    avg_cache_read_mib_s: float | None
    avg_cache_write_mib_s: float | None
    avg_cache_read_ms_per_op: float | None
    avg_cache_write_ms_per_op: float | None
    avg_cache_io_util_percent: float | None


@dataclass
class SpeedscopeSummary:
    """Small machine-readable summary extracted from a py-spy speedscope file."""

    total_samples: int
    top_leaf_frames: list[dict[str, Any]] = field(default_factory=list)
    top_files: list[dict[str, Any]] = field(default_factory=list)
    top_dashboard_leaf_frames: list[dict[str, Any]] = field(default_factory=list)
    dashboard_sample_percent: float | None = None


@dataclass
class ImportStageProfile:
    """Measured import-stage timing for one exact generation lineage."""

    mode: str
    import_root: str
    wall_seconds: float
    conversion_seconds: float
    activation_load_seconds: float | None
    activation_table_load_seconds: float | None
    activation_import_seconds: float | None
    activation_copy_write_seconds: float | None
    activation_copy_stream_close_seconds: float | None
    activation_insert_from_stage_seconds: float | None
    imported_activation_rows: int | None
    imported_neuron_rows: int | None
    table_load_seconds: dict[str, float] = field(default_factory=dict)
    table_import_seconds: dict[str, float] = field(default_factory=dict)
    table_import_substage_seconds: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Captured summary for one profiled dashboard generation run."""

    label: str
    n_features_per_batch: int
    n_prompts_in_forward_pass: int
    status: str
    reason: str
    exit_code: int | None
    started_at_utc: str
    ended_at_utc: str
    elapsed_seconds: float
    completed_batches: list[int]
    avg_batch_seconds: float | None
    throughput_features_per_min: float | None
    max_tree_rss_gib: float
    max_tree_cpu_percent: float
    max_host_used_gib: float
    max_gpu_process_used_mib: int
    max_gpu_device_used_mib: int
    max_gpu_util_percent: int
    avg_proc_read_mib_s: float | None
    avg_proc_write_mib_s: float | None
    avg_cache_read_mib_s: float | None
    avg_cache_write_mib_s: float | None
    avg_cache_read_ms_per_op: float | None
    avg_cache_write_ms_per_op: float | None
    avg_cache_io_util_percent: float | None
    pipeline_log_path: str
    stdout_log_path: str
    resource_samples_path: str
    stage_summary_path: str
    py_spy_output_path: str | None
    py_spy_command: str | None
    run_root: str
    command: str
    cache_device: str | None
    speedscope_summary: SpeedscopeSummary | None = None
    import_stage_profile: ImportStageProfile | None = None
    notes: list[str] = field(default_factory=list)


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def gib_from_kib(value_kib: int) -> float:
    """Convert KiB to GiB."""

    return value_kib / (1024**2)


def mib_from_bytes(value_bytes: int) -> float:
    """Convert bytes to MiB."""

    return value_bytes / (1024**2)


def parse_config_spec(spec: str) -> ProfileConfig:
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
    return ProfileConfig(label=label, n_features_per_batch=features, n_prompts_in_forward_pass=prompts)


def default_profile_configs(include_comparisons: bool) -> list[ProfileConfig]:
    """Return the default profiling run set for the RTE dashboard workload."""

    configs = [ProfileConfig(label="primary-512x128", n_features_per_batch=512, n_prompts_in_forward_pass=128)]
    if include_comparisons:
        configs.extend(
            [
                ProfileConfig(label="prompt-scale-512x256", n_features_per_batch=512, n_prompts_in_forward_pass=256),
                ProfileConfig(label="feature-scale-1024x128", n_features_per_batch=1024, n_prompts_in_forward_pass=128),
                ProfileConfig(label="large-2048x128", n_features_per_batch=2048, n_prompts_in_forward_pass=128),
            ]
        )
    return configs


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile Neuronpedia dashboard generation with py-spy plus process-tree CPU/RSS/GPU and IO sampling. "
            "The default workload is the gemmascope-2-transcoder-262k-rte source set with TransformerBridge and "
            "the pretokenized RTE prompt cache."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Config spec in features:prompts or label:features:prompts form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-comparisons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include 512x256, 1024x128, and 2048x128 comparison runs when no explicit configs are provided.",
    )
    parser.add_argument("--layer", type=int, default=2, help="Single layer index to profile. Defaults to 2.")
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
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Polling interval for resource and IO samples. Defaults to 1 second.",
    )
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=900.0,
        help="Kill a run if no batch output is detected for this many seconds. Defaults to 900.",
    )
    parser.add_argument(
        "--max-tree-rss-gib",
        type=float,
        default=46.0,
        help="Kill the process tree if sampled RSS exceeds this GiB threshold. Defaults to 46.",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=DEFAULT_SESSION_ROOT,
        help=f"Directory for profiling outputs. Defaults to {DEFAULT_SESSION_ROOT}.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Dashboard run root used by the child pipeline.",
    )
    parser.add_argument(
        "--python-executable",
        default=DEFAULT_PYTHON,
        help=f"Python interpreter for the dashboard pipeline. Defaults to {DEFAULT_PYTHON}.",
    )
    parser.add_argument(
        "--prompts-pretokenized-dataset-path",
        type=Path,
        default=DEFAULT_PRETOKENIZED_DATASET,
        help="Local HuggingFace dataset with an input_ids column.",
    )
    parser.add_argument(
        "--primary-acts-batch-size",
        type=int,
        help=(
            "Optional internal activation-capture chunk size passed to SAEDashboard. This preserves "
            "--n-prompts-in-forward-pass as the logical dashboard batch while lowering model-forward peak memory."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES value passed to the dashboard pipeline. Defaults to 0.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Mounted cache path whose block-device IO should be sampled. Defaults to /mnt/cache_extended.",
    )
    parser.add_argument(
        "--py-spy-executable",
        default=DEFAULT_PY_SPY,
        help=f"py-spy executable path. Defaults to {DEFAULT_PY_SPY}.",
    )
    parser.add_argument(
        "--py-spy-format",
        choices=("speedscope", "raw", "flamegraph"),
        default="speedscope",
        help="py-spy output format. Defaults to speedscope.",
    )
    parser.add_argument(
        "--no-py-spy",
        action="store_true",
        help="Run without py-spy while still collecting resource and IO samples.",
    )
    parser.add_argument(
        "--cleanup-run-dirs",
        action="store_true",
        help="Delete per-config dashboard run directories after metrics are captured.",
    )
    parser.add_argument(
        "--profile-import-stage",
        action="store_true",
        help=(
            "After generation profiling, measure the exact-lineage local DB import stage. "
            "Legacy JSONL runs include explicit conversion time in the activation-load surface."
        ),
    )
    parser.add_argument(
        "--local-db-url",
        default=None,
        help=(
            "Optional local Neuronpedia Postgres URL for --profile-import-stage. "
            "Defaults to LOCAL_NEURONPEDIA_DB_URL / POSTGRES_URL_NON_POOLING when unset."
        ),
    )
    parser.add_argument(
        "--dashboard-extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument appended to the child neuronpedia_dashboard_pipeline command. "
            "Use --dashboard-extra-arg=--flag or --dashboard-extra-arg=--option=value; can be repeated."
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
    """Return the root pid and all descendants present in the current process table."""

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


def read_tree_io_totals(pids: Iterable[int]) -> ProcessIoTotals:
    """Read and aggregate /proc/<pid>/io for a process tree."""

    totals = ProcessIoTotals()
    for pid in pids:
        io_path = Path("/proc") / str(pid) / "io"
        try:
            lines = io_path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        counters: dict[str, int] = {}
        for line in lines:
            key, raw_value = line.split(":", maxsplit=1)
            counters[key] = int(raw_value.strip())
        totals.read_bytes += counters.get("read_bytes", 0)
        totals.write_bytes += counters.get("write_bytes", 0)
        totals.read_chars += counters.get("rchar", 0)
        totals.write_chars += counters.get("wchar", 0)
        totals.syscr += counters.get("syscr", 0)
        totals.syscw += counters.get("syscw", 0)
        totals.cancelled_write_bytes += counters.get("cancelled_write_bytes", 0)
    return totals


def resolve_block_device(path: Path) -> str | None:
    """Return the /proc/diskstats device name backing a mounted path when it is discoverable."""

    completed = subprocess.run(["df", "-P", str(path)], capture_output=True, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    source = lines[1].split()[0]
    if not source.startswith("/dev/"):
        return None
    real_device = Path(source).resolve()
    return real_device.name


def read_diskstats(device_name: str | None) -> DiskStats | None:
    """Read selected /proc/diskstats counters for one block device."""

    if device_name is None:
        return None
    try:
        lines = Path("/proc/diskstats").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) < 14 or parts[2] != device_name:
            continue
        return DiskStats(
            reads_completed=int(parts[3]),
            sectors_read=int(parts[5]),
            ms_reading=int(parts[6]),
            writes_completed=int(parts[7]),
            sectors_written=int(parts[9]),
            ms_writing=int(parts[10]),
            ios_in_progress=int(parts[11]),
            ms_doing_io=int(parts[12]),
            weighted_ms_doing_io=int(parts[13]),
        )
    return None


def sample_gpu_metrics(tree_pids: set[int]) -> tuple[int, int, int]:
    """Return process GPU MiB, max device GPU MiB, and max device util percent."""

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
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
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
        try:
            pid = int(pid_str)
            used = int(used_mib)
        except ValueError:
            continue
        if pid in tree_pids:
            process_used_mib += used

    return process_used_mib, max_device_used_mib, max_device_util_pct


def make_resource_sample(
    *,
    root_pid: int,
    started_monotonic: float,
    previous_sample_monotonic: float | None,
    previous_io: ProcessIoTotals | None,
    previous_diskstats: DiskStats | None,
    cache_device: str | None,
) -> tuple[ResourceSample, ProcessIoTotals, DiskStats | None, float]:
    """Capture one process-tree resource and IO sample."""

    now_monotonic = time.monotonic()
    process_table = parse_ps_table()
    tree_pids = descendant_pids(root_pid, process_table)
    total_rss_kib = sum(process_table[pid][1] for pid in tree_pids)
    total_cpu_pct = sum(process_table[pid][2] for pid in tree_pids)
    meminfo = read_meminfo()
    host_used_kib = meminfo["MemTotal"] - meminfo["MemAvailable"]
    gpu_process_used_mib, gpu_device_used_mib, gpu_util_percent = sample_gpu_metrics(tree_pids)
    current_io = read_tree_io_totals(tree_pids)
    current_diskstats = read_diskstats(cache_device)

    sample = ResourceSample(
        elapsed_seconds=now_monotonic - started_monotonic,
        timestamp_utc=utc_now_iso(),
        tree_pid_count=len(tree_pids),
        tree_rss_gib=gib_from_kib(total_rss_kib),
        tree_cpu_percent=total_cpu_pct,
        host_used_gib=gib_from_kib(host_used_kib),
        gpu_process_used_mib=gpu_process_used_mib,
        gpu_device_used_mib=gpu_device_used_mib,
        gpu_util_percent=gpu_util_percent,
    )

    if previous_sample_monotonic is not None and previous_io is not None:
        elapsed = max(now_monotonic - previous_sample_monotonic, 1e-9)
        sample.proc_read_mib_s = mib_from_bytes(max(current_io.read_bytes - previous_io.read_bytes, 0)) / elapsed
        sample.proc_write_mib_s = mib_from_bytes(max(current_io.write_bytes - previous_io.write_bytes, 0)) / elapsed
        sample.proc_syscr_s = max(current_io.syscr - previous_io.syscr, 0) / elapsed
        sample.proc_syscw_s = max(current_io.syscw - previous_io.syscw, 0) / elapsed

        if previous_diskstats is not None and current_diskstats is not None:
            read_ops = max(current_diskstats.reads_completed - previous_diskstats.reads_completed, 0)
            write_ops = max(current_diskstats.writes_completed - previous_diskstats.writes_completed, 0)
            read_ms = max(current_diskstats.ms_reading - previous_diskstats.ms_reading, 0)
            write_ms = max(current_diskstats.ms_writing - previous_diskstats.ms_writing, 0)
            sample.cache_read_mib_s = (
                max(current_diskstats.sectors_read - previous_diskstats.sectors_read, 0) * 512 / (1024**2) / elapsed
            )
            sample.cache_write_mib_s = (
                max(current_diskstats.sectors_written - previous_diskstats.sectors_written, 0)
                * 512
                / (1024**2)
                / elapsed
            )
            sample.cache_read_ms_per_op = read_ms / read_ops if read_ops else 0.0
            sample.cache_write_ms_per_op = write_ms / write_ops if write_ops else 0.0
            sample.cache_io_util_percent = (
                max(current_diskstats.ms_doing_io - previous_diskstats.ms_doing_io, 0) / (elapsed * 1000) * 100
            )

    return sample, current_io, current_diskstats, now_monotonic


def build_env() -> dict[str, str]:
    """Build the environment for dashboard profiling runs."""

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/mnt/cache_extended/speediedan/.cache/huggingface")
    env.setdefault("HF_DATASETS_CACHE", "/mnt/cache_extended/speediedan/.cache/huggingface/datasets")
    env.setdefault("HF_HUB_CACHE", "/mnt/cache_extended/speediedan/.cache/huggingface/hub")
    env.setdefault("IT_NP_CACHE", "/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_dashboard_command(
    config: ProfileConfig,
    *,
    layer: int,
    python_executable: str,
    run_root: Path,
    pretokenized_dataset_path: Path | None,
    primary_acts_batch_size: int | None,
    cuda_visible_devices: str,
    dashboard_extra_args: Iterable[str] = (),
    profile_import_stage: bool = False,
) -> list[str]:
    """Build the dashboard pipeline command for one profiled run."""

    n_tokens_in_prompt = resolve_n_tokens_in_prompt(pretokenized_dataset_path)
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
        "--runner-log-resource-snapshots",
        "--start-layer",
        str(layer),
        "--end-layer",
        str(layer),
        "--start-batch",
        "0",
        "--n-prompts-total",
        "2490",
        "--n-tokens-in-prompt",
        str(n_tokens_in_prompt),
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
        cuda_visible_devices,
        "--heartbeat-seconds",
        "60",
        "--stall-timeout-seconds",
        "1800",
        "--use-skip-transcoder",
        "--skip-local-db-import",
        "--run-root",
        str(run_root),
    ]
    if pretokenized_dataset_path is not None:
        command.extend(["--prompts-pretokenized-dataset-path", str(pretokenized_dataset_path)])
    if primary_acts_batch_size is not None:
        command.extend(["--primary-acts-batch-size", str(primary_acts_batch_size)])
    dashboard_extra_args = list(dashboard_extra_args)
    requested_output_format = dashboard_extra_arg_value(dashboard_extra_args, "--runner-dashboard-output-format")
    if (
        profile_import_stage
        and requested_output_format == "columnar"
        and "--runner-emit-activation-copy-rows" not in dashboard_extra_args
        and "--no-runner-emit-activation-copy-rows" not in dashboard_extra_args
    ):
        dashboard_extra_args.append("--runner-emit-activation-copy-rows")
    command.extend(dashboard_extra_args)
    return command


def dashboard_extra_arg_value(extra_args: Iterable[str], option_name: str) -> str | None:
    """Return the final value passed for an extra child dashboard option."""

    option_prefix = f"{option_name}="
    pending_option = False
    resolved_value = None
    for extra_arg in extra_args:
        if pending_option:
            resolved_value = extra_arg
            pending_option = False
            continue
        if extra_arg == option_name:
            pending_option = True
        elif extra_arg.startswith(option_prefix):
            resolved_value = extra_arg[len(option_prefix) :]
    return resolved_value


def dashboard_run_name(extra_args: Iterable[str]) -> str:
    """Return the dashboard run directory name implied by the child pipeline args."""

    model_name = dashboard_extra_arg_value(extra_args, "--model-name") or DEFAULT_MODEL_NAME
    source_set_id = dashboard_extra_arg_value(extra_args, "--neuronpedia-source-set-id") or DEFAULT_SOURCE_SET_ID
    run_name_suffix = dashboard_extra_arg_value(extra_args, "--run-name-suffix")
    base_name = f"{model_name}_{source_set_id}"
    if run_name_suffix:
        return f"{base_name}_{run_name_suffix}"
    return base_name


def py_spy_extension(py_spy_format: str) -> str:
    """Return the preferred file extension for a py-spy output format."""

    if py_spy_format == "flamegraph":
        return "svg"
    if py_spy_format == "raw":
        return "raw"
    return "speedscope.json"


def build_launch_command(
    dashboard_command: list[str],
    *,
    args: argparse.Namespace,
    config_dir: Path,
) -> tuple[list[str], Path | None, str | None]:
    """Build either the dashboard command or a py-spy wrapper command."""

    if args.no_py_spy:
        return dashboard_command, None, None
    py_spy_executable = args.py_spy_executable or shutil.which("py-spy")
    if py_spy_executable is None or not Path(py_spy_executable).exists():
        raise FileNotFoundError(f"py-spy executable was not found: {py_spy_executable}")
    output_path = config_dir / f"py_spy_{args.py_spy_format}.{py_spy_extension(args.py_spy_format)}"
    command = [py_spy_executable, "record", "--subprocesses", "--nonblocking", "-o", str(output_path)]
    if args.py_spy_format != "flamegraph":
        command.extend(["--format", args.py_spy_format])
    command.extend(["--", *dashboard_command])
    return command, output_path, shlex.join(command)


def terminate_process_group(process: subprocess.Popen[str], reason: str, *, interrupt_first: bool = False) -> None:
    """Terminate the process group started for a profile run."""

    if process.poll() is not None:
        return
    print(f"[{utc_now_iso()}] terminating pid={process.pid} reason={reason}", flush=True)
    if interrupt_first:
        os.killpg(process.pid, signal.SIGINT)
        interrupt_deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < interrupt_deadline:
            time.sleep(0.5)
        if process.poll() is not None:
            return
    os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)


def read_new_log_lines(log_path: Path, offset: int) -> tuple[list[str], int]:
    """Read newly appended lines from a log path."""

    if not log_path.exists():
        return [], offset
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        return lines, handle.tell()


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


def parse_runner_events(
    lines: Iterable[str],
    *,
    started_monotonic: float,
) -> tuple[list[BatchEvent], list[RunnerResourceEvent]]:
    """Parse batch completion and runner resource events from new log lines."""

    batch_events: list[BatchEvent] = []
    resource_events: list[RunnerResourceEvent] = []
    for line in lines:
        now_monotonic = time.monotonic()
        elapsed = now_monotonic - started_monotonic
        batch_match = BATCH_OUTPUT_RE.search(line)
        if batch_match:
            batch_events.append(BatchEvent(int(batch_match.group(1)), elapsed, utc_now_iso()))
        columnar_batch_match = COLUMNAR_BATCH_OUTPUT_RE.search(line)
        if columnar_batch_match:
            batch_events.append(BatchEvent(int(columnar_batch_match.group(1)), elapsed, utc_now_iso()))
        columnar_manifest_match = COLUMNAR_MANIFEST_OUTPUT_RE.search(line)
        if columnar_manifest_match:
            batch_events.append(BatchEvent(int(columnar_manifest_match.group(1)), elapsed, utc_now_iso()))

        resource_match = RESOURCE_RE.search(line)
        if resource_match:
            stage = resource_match.group(1)
            post_batch_match = POST_BATCH_RE.search(stage)
            if post_batch_match:
                batch_events.append(BatchEvent(int(post_batch_match.group(1)), elapsed, utc_now_iso()))
            resource_events.append(
                RunnerResourceEvent(
                    stage=stage,
                    detected_elapsed_seconds=elapsed,
                    timestamp_utc=utc_now_iso(),
                    rss_gib=float(resource_match.group(2)),
                    cuda_allocated_gib=float(resource_match.group(3)),
                    cuda_reserved_gib=float(resource_match.group(4)),
                    cuda_max_allocated_gib=float(resource_match.group(5)),
                )
            )
    return batch_events, resource_events


def finite_average(values: Iterable[float | None]) -> float | None:
    """Average non-None values."""

    concrete = [value for value in values if value is not None]
    if not concrete:
        return None
    return sum(concrete) / len(concrete)


def summarize_samples(
    segment: str,
    samples: list[ResourceSample],
    *,
    duration_seconds: float | None,
    features_in_segment: int | None = None,
) -> SegmentSummary:
    """Build a compact resource summary for a set of samples."""

    throughput = None
    if duration_seconds is not None and duration_seconds > 0 and features_in_segment is not None:
        throughput = features_in_segment * 60.0 / duration_seconds
    return SegmentSummary(
        segment=segment,
        duration_seconds=duration_seconds,
        throughput_features_per_min=throughput,
        samples=len(samples),
        avg_tree_cpu_percent=finite_average(sample.tree_cpu_percent for sample in samples),
        max_tree_cpu_percent=max((sample.tree_cpu_percent for sample in samples), default=None),
        max_tree_rss_gib=max((sample.tree_rss_gib for sample in samples), default=None),
        max_gpu_process_used_mib=max((sample.gpu_process_used_mib for sample in samples), default=None),
        avg_gpu_util_percent=finite_average(float(sample.gpu_util_percent) for sample in samples),
        max_gpu_util_percent=max((sample.gpu_util_percent for sample in samples), default=None),
        avg_proc_read_mib_s=finite_average(sample.proc_read_mib_s for sample in samples),
        avg_proc_write_mib_s=finite_average(sample.proc_write_mib_s for sample in samples),
        avg_cache_read_mib_s=finite_average(sample.cache_read_mib_s for sample in samples),
        avg_cache_write_mib_s=finite_average(sample.cache_write_mib_s for sample in samples),
        avg_cache_read_ms_per_op=finite_average(sample.cache_read_ms_per_op for sample in samples),
        avg_cache_write_ms_per_op=finite_average(sample.cache_write_ms_per_op for sample in samples),
        avg_cache_io_util_percent=finite_average(sample.cache_io_util_percent for sample in samples),
    )


def build_segment_summaries(
    *,
    config: ProfileConfig,
    samples: list[ResourceSample],
    resource_events: list[RunnerResourceEvent],
    batch_events: dict[int, BatchEvent],
) -> list[SegmentSummary]:
    """Summarize startup and per-batch resource windows."""

    summaries: list[SegmentSummary] = []
    pre_batch_times: dict[int, float] = {}
    post_batch_times: dict[int, float] = {}
    for event in resource_events:
        pre_match = PRE_BATCH_RE.search(event.stage)
        if pre_match:
            pre_batch_times[int(pre_match.group(1))] = event.detected_elapsed_seconds
        post_match = POST_BATCH_RE.search(event.stage)
        if post_match:
            post_batch_times[int(post_match.group(1))] = event.detected_elapsed_seconds

    first_pre_time = min(pre_batch_times.values(), default=None)
    if first_pre_time is not None:
        startup_samples = [sample for sample in samples if sample.elapsed_seconds <= first_pre_time]
        summaries.append(
            summarize_samples("startup_to_first_pre_batch", startup_samples, duration_seconds=first_pre_time)
        )

    for batch_num in sorted(batch_events):
        start_time = pre_batch_times.get(batch_num)
        end_time = post_batch_times.get(batch_num) or batch_events[batch_num].detected_elapsed_seconds
        if start_time is None and batch_num > 0:
            start_time = (
                post_batch_times.get(batch_num - 1)
                or batch_events.get(batch_num - 1, batch_events[batch_num]).detected_elapsed_seconds
            )
        if start_time is None:
            start_time = 0.0
        segment_samples = [sample for sample in samples if start_time <= sample.elapsed_seconds <= end_time]
        summaries.append(
            summarize_samples(
                f"batch_{batch_num}",
                segment_samples,
                duration_seconds=max(end_time - start_time, 0.0),
                features_in_segment=config.n_features_per_batch,
            )
        )

    return summaries


def write_resource_samples(path: Path, samples: list[ResourceSample]) -> None:
    """Write resource samples to CSV for spreadsheet or notebook analysis."""

    if not samples:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0]).keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def write_stage_summary(path: Path, summaries: list[SegmentSummary]) -> None:
    """Write stage summaries as markdown."""

    lines = [
        "| Segment | Duration s | Features/min | Samples | Avg CPU % | Max CPU % | Max RSS GiB | "
        "Max GPU MiB | Avg GPU util % | Max GPU util % | Proc read MiB/s | Proc write MiB/s | "
        "Cache read MiB/s | Cache write MiB/s | Cache read ms/op | Cache write ms/op | Cache IO util % |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            (
                "| {segment} | {duration} | {throughput} | {samples} | {avg_cpu} | {max_cpu} | {rss} | "
                "{gpu_mib} | {avg_gpu} | {max_gpu} | {proc_read} | {proc_write} | {cache_read} | "
                "{cache_write} | {read_lat} | {write_lat} | {io_util} |"
            ).format(
                segment=summary.segment,
                duration=f"{summary.duration_seconds:.1f}" if summary.duration_seconds is not None else "-",
                throughput=(
                    f"{summary.throughput_features_per_min:.0f}"
                    if summary.throughput_features_per_min is not None
                    else "-"
                ),
                samples=summary.samples,
                avg_cpu=f"{summary.avg_tree_cpu_percent:.1f}" if summary.avg_tree_cpu_percent is not None else "-",
                max_cpu=f"{summary.max_tree_cpu_percent:.1f}" if summary.max_tree_cpu_percent is not None else "-",
                rss=f"{summary.max_tree_rss_gib:.2f}" if summary.max_tree_rss_gib is not None else "-",
                gpu_mib=summary.max_gpu_process_used_mib if summary.max_gpu_process_used_mib is not None else "-",
                avg_gpu=f"{summary.avg_gpu_util_percent:.1f}" if summary.avg_gpu_util_percent is not None else "-",
                max_gpu=summary.max_gpu_util_percent if summary.max_gpu_util_percent is not None else "-",
                proc_read=f"{summary.avg_proc_read_mib_s:.2f}" if summary.avg_proc_read_mib_s is not None else "-",
                proc_write=f"{summary.avg_proc_write_mib_s:.2f}" if summary.avg_proc_write_mib_s is not None else "-",
                cache_read=f"{summary.avg_cache_read_mib_s:.2f}" if summary.avg_cache_read_mib_s is not None else "-",
                cache_write=f"{summary.avg_cache_write_mib_s:.2f}"
                if summary.avg_cache_write_mib_s is not None
                else "-",
                read_lat=(
                    f"{summary.avg_cache_read_ms_per_op:.3f}" if summary.avg_cache_read_ms_per_op is not None else "-"
                ),
                write_lat=(
                    f"{summary.avg_cache_write_ms_per_op:.3f}" if summary.avg_cache_write_ms_per_op is not None else "-"
                ),
                io_util=(
                    f"{summary.avg_cache_io_util_percent:.1f}" if summary.avg_cache_io_util_percent is not None else "-"
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_speedscope(path: Path, *, top_n: int = 20) -> SpeedscopeSummary | None:
    """Extract top frame/file summaries from a py-spy speedscope JSON file."""

    if not path.exists() or path.suffix != ".json":
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    profiles = data.get("profiles", [])
    if not profiles:
        return SpeedscopeSummary(total_samples=0)
    shared = data.get("shared", {})
    frames = shared.get("frames") or profiles[0].get("frames") or []
    leaf_counts: dict[str, int] = {}
    file_counts: dict[str, int] = {}
    dashboard_leaf_counts: dict[str, int] = {}
    dashboard_samples = 0
    total_samples = 0

    for profile in profiles:
        samples = profile.get("samples", [])
        total_samples += len(samples)
        for sample in samples:
            frame_indexes = sample if isinstance(sample, list) else [sample]
            frame_dicts = []
            for frame_index in frame_indexes:
                if frame_index is None:
                    continue
                try:
                    concrete_index = int(frame_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= concrete_index < len(frames):
                    frame_dicts.append(frames[concrete_index])
            if not frame_dicts:
                continue
            sample_has_dashboard_frame = any("sae_dashboard" in (frame.get("file") or "") for frame in frame_dicts)
            if sample_has_dashboard_frame:
                dashboard_samples += 1
            leaf_frame = frame_dicts[-1]
            name = leaf_frame.get("name") or "<unknown>"
            file_path = leaf_frame.get("file") or "<unknown>"
            line = leaf_frame.get("line")
            leaf_key = f"{name} ({file_path}:{line})" if line is not None else f"{name} ({file_path})"
            leaf_counts[leaf_key] = leaf_counts.get(leaf_key, 0) + 1
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
            if "sae_dashboard" in file_path or sample_has_dashboard_frame:
                dashboard_leaf_counts[leaf_key] = dashboard_leaf_counts.get(leaf_key, 0) + 1

    return SpeedscopeSummary(
        total_samples=total_samples,
        top_leaf_frames=top_count_dicts(leaf_counts, total_samples, top_n),
        top_files=top_count_dicts(file_counts, total_samples, top_n),
        top_dashboard_leaf_frames=top_count_dicts(dashboard_leaf_counts, total_samples, top_n),
        dashboard_sample_percent=(dashboard_samples * 100.0 / total_samples if total_samples else None),
    )


def top_count_dicts(counts: dict[str, int], total: int, top_n: int) -> list[dict[str, Any]]:
    """Convert count mappings to sorted summary dictionaries."""

    return [
        {"name": name, "samples": count, "percent": count * 100.0 / total if total else 0.0}
        for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    ]


def resolve_n_tokens_in_prompt(pretokenized_dataset_path: Path | None) -> int:
    """Resolve the intended prompt token length for the child pipeline command."""

    if pretokenized_dataset_path is None:
        return 128
    metadata_path = pretokenized_dataset_path / "sae_lens.json"
    if not metadata_path.exists():
        return 128
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 128
    for key in ("effective_context_size", "context_size"):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            resolved_value = int(value)
        except (TypeError, ValueError):
            continue
        if resolved_value > 0:
            return resolved_value
    return 128


def build_pipeline_config_from_dashboard_command(
    dashboard_command: list[str],
    *,
    local_db_url: str | None,
) -> dashboard_pipeline.NeuronpediaDashboardPipelineConfig:
    """Rebuild the child pipeline config from the profiler's dashboard command."""

    if len(dashboard_command) < 4 or dashboard_command[1:3] != [
        "-m",
        "interpretune.utils.neuronpedia_dashboard_pipeline",
    ]:
        raise ValueError("Expected a dashboard pipeline module command.")
    argv = [arg for arg in dashboard_command[3:] if arg != "--skip-local-db-import"]
    config = dashboard_pipeline._build_dashboard_pipeline_config(dashboard_pipeline._parse_args(argv))
    config.import_to_local_db = True
    config.local_db_url = resolve_local_neuronpedia_db_url(local_db_url)
    return config


def _coerce_float_mapping(values: Any) -> dict[str, float]:
    """Normalize string-keyed numeric mappings for JSON serialization."""

    if not values:
        return {}
    return {str(key): float(value) for key, value in dict(values).items()}


def _coerce_nested_float_mapping(values: Any) -> dict[str, dict[str, float]]:
    """Normalize nested numeric mappings for JSON serialization."""

    if not values:
        return {}
    return {str(key): _coerce_float_mapping(value) for key, value in dict(values).items()}


def profile_import_stage(
    dashboard_command: list[str],
    *,
    local_db_url: str | None,
) -> ImportStageProfile:
    """Measure local DB import timing for the exact generation lineage produced by the child run."""

    config = build_pipeline_config_from_dashboard_command(dashboard_command, local_db_url=local_db_url)
    layer_num = config.start_layer
    output_dir = config.output_dir_for_layer(layer_num)
    if not output_dir.exists():
        raise FileNotFoundError(f"Expected profiled output directory does not exist: {output_dir}")

    dashboard_output_format = dashboard_pipeline._resolve_runner_dashboard_output_format(config)
    wall_start = time.monotonic()
    conversion_seconds = 0.0
    import_root: Path
    if dashboard_output_format == "columnar":
        import_root = output_dir
        import_summary = dashboard_pipeline.import_columnar_dashboard_output(
            config,
            layer_num=layer_num,
            output_dir=output_dir,
            activation_use_stage_table=False,
        )
    else:
        conversion_start = time.monotonic()
        import_root = dashboard_pipeline.convert_dashboard_output(
            config,
            layer_num=layer_num,
            output_dir=output_dir,
            logger=logging.getLogger(__name__),
        )
        conversion_seconds = time.monotonic() - conversion_start
        import_summary = dashboard_pipeline.import_neuronpedia_export_bundle_local_db(
            import_root,
            local_db_url=config.local_db_url or "",
            prefer_arrow_for_tables=dashboard_pipeline.DEFAULT_LOCAL_DB_COLUMNAR_IMPORT_TABLES,
            prefer_copy_for_tables=dashboard_pipeline.DEFAULT_LOCAL_DB_COLUMNAR_COPY_TABLES,
        )
    wall_seconds = time.monotonic() - wall_start

    table_load_seconds = _coerce_float_mapping(getattr(import_summary, "table_load_seconds", {}))
    table_import_seconds = _coerce_float_mapping(getattr(import_summary, "table_import_seconds", {}))
    table_import_substage_seconds = _coerce_nested_float_mapping(
        getattr(import_summary, "table_import_substage_seconds", {})
    )
    activation_table_load_seconds = table_load_seconds.get("Activation")
    activation_import_seconds = table_import_seconds.get("Activation")
    activation_substages = table_import_substage_seconds.get("Activation", {})
    activation_load_seconds = None
    if activation_table_load_seconds is not None or conversion_seconds:
        activation_load_seconds = (activation_table_load_seconds or 0.0) + conversion_seconds

    imported_row_counts = dict(getattr(import_summary, "imported_row_counts", {}) or {})
    return ImportStageProfile(
        mode=dashboard_output_format,
        import_root=str(import_root),
        wall_seconds=wall_seconds,
        conversion_seconds=conversion_seconds,
        activation_load_seconds=activation_load_seconds,
        activation_table_load_seconds=activation_table_load_seconds,
        activation_import_seconds=activation_import_seconds,
        activation_copy_write_seconds=activation_substages.get("copy_write"),
        activation_copy_stream_close_seconds=activation_substages.get("copy_stream_close"),
        activation_insert_from_stage_seconds=activation_substages.get("insert_from_stage"),
        imported_activation_rows=(
            int(imported_row_counts["Activation"]) if imported_row_counts.get("Activation") is not None else None
        ),
        imported_neuron_rows=(
            int(imported_row_counts["Neuron"]) if imported_row_counts.get("Neuron") is not None else None
        ),
        table_load_seconds=table_load_seconds,
        table_import_seconds=table_import_seconds,
        table_import_substage_seconds=table_import_substage_seconds,
    )


def summarize_batch_throughput(
    config: ProfileConfig,
    batch_events: dict[int, BatchEvent],
    *,
    summary_warmup_batches: int = 0,
) -> tuple[float | None, float | None]:
    """Return average detected batch interval and throughput features/min."""

    sorted_events = [
        batch_events[batch_num] for batch_num in sorted(batch_events) if batch_num >= summary_warmup_batches
    ]
    if len(sorted_events) < 2:
        return None, None
    intervals = [
        later.detected_elapsed_seconds - earlier.detected_elapsed_seconds
        for earlier, later in zip(sorted_events, sorted_events[1:], strict=False)
    ]
    avg_batch_seconds = sum(intervals) / len(intervals)
    return avg_batch_seconds, config.n_features_per_batch * 60.0 / avg_batch_seconds


def run_profile(
    config: ProfileConfig,
    *,
    args: argparse.Namespace,
    session_dir: Path,
    env: dict[str, str],
    cache_device: str | None,
) -> ProfileResult:
    """Run one profiled dashboard generation command."""

    config_dir = session_dir / config.label
    run_root = config_dir / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    pipeline_log_path = (
        run_root / dashboard_run_name(args.dashboard_extra_arg) / f"run.resume-{args.layer}-{args.layer}.log"
    )
    dashboard_command = build_dashboard_command(
        config,
        layer=args.layer,
        python_executable=args.python_executable,
        run_root=run_root,
        pretokenized_dataset_path=args.prompts_pretokenized_dataset_path,
        primary_acts_batch_size=args.primary_acts_batch_size,
        cuda_visible_devices=args.cuda_visible_devices,
        dashboard_extra_args=args.dashboard_extra_arg,
        profile_import_stage=args.profile_import_stage,
    )
    launch_command, py_spy_output_path, py_spy_command = build_launch_command(
        dashboard_command,
        args=args,
        config_dir=config_dir,
    )
    stdout_path = config_dir / "outer.stdout.log"
    resource_samples_path = config_dir / "resource_samples.csv"
    stage_summary_path = config_dir / "stage_summary.md"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    started_at_utc = utc_now_iso()
    started_monotonic = time.monotonic()
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        launch_command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    stdout_handle.close()
    print(f"[{started_at_utc}] started {config.label}: {shlex.join(launch_command)}", flush=True)

    log_offset = 0
    samples: list[ResourceSample] = []
    runner_events: list[RunnerResourceEvent] = []
    batch_events: dict[int, BatchEvent] = {}
    notes: list[str] = []
    last_error_lines: list[str] = []
    classified_error: str | None = None
    status = "running"
    reason = ""
    last_batch_progress_monotonic = started_monotonic
    previous_sample_monotonic: float | None = None
    previous_io: ProcessIoTotals | None = None
    previous_diskstats: DiskStats | None = None

    try:
        while True:
            time.sleep(args.sample_seconds)
            new_lines, log_offset = read_new_log_lines(pipeline_log_path, log_offset)
            new_batch_events, new_runner_events = parse_runner_events(new_lines, started_monotonic=started_monotonic)
            for event in new_batch_events:
                if event.batch_num not in batch_events:
                    batch_events[event.batch_num] = event
                    last_batch_progress_monotonic = time.monotonic()
            runner_events.extend(new_runner_events)

            error_kind, error_lines = classify_error_lines(new_lines)
            if error_lines:
                last_error_lines = error_lines
            if error_kind is not None:
                classified_error = error_kind

            if process.poll() is None:
                sample, previous_io, previous_diskstats, previous_sample_monotonic = make_resource_sample(
                    root_pid=process.pid,
                    started_monotonic=started_monotonic,
                    previous_sample_monotonic=previous_sample_monotonic,
                    previous_io=previous_io,
                    previous_diskstats=previous_diskstats,
                    cache_device=cache_device,
                )
                samples.append(sample)

                if sample.tree_rss_gib > args.max_tree_rss_gib:
                    status = "killed"
                    reason = f"tree_rss_limit>{args.max_tree_rss_gib:.2f}GiB"
                    notes.append(f"Killed at sampled tree RSS {sample.tree_rss_gib:.2f} GiB.")
                    terminate_process_group(process, reason, interrupt_first=py_spy_output_path is not None)
                    break

                if len(batch_events) >= args.target_batches:
                    status = "target_reached"
                    reason = f"completed_{args.target_batches}_batches"
                    notes.append(f"Stopped after {args.target_batches} completed batches.")
                    terminate_process_group(process, reason, interrupt_first=py_spy_output_path is not None)
                    break

                stalled_for = time.monotonic() - last_batch_progress_monotonic
                if args.stall_seconds > 0 and stalled_for > args.stall_seconds:
                    status = "killed"
                    reason = f"stall_timeout>{args.stall_seconds:.0f}s"
                    notes.append(f"Killed after {stalled_for:.1f}s without a new completed batch.")
                    terminate_process_group(process, reason, interrupt_first=py_spy_output_path is not None)
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
            notes.append("Process exited after the ignore-mask shape-mismatch marker.")
        else:
            status = "failed"
            reason = f"exit_{exit_code}"
            notes.extend(last_error_lines[-3:])

    segment_summaries = build_segment_summaries(
        config=config,
        samples=samples,
        resource_events=runner_events,
        batch_events=batch_events,
    )
    write_resource_samples(resource_samples_path, samples)
    write_stage_summary(stage_summary_path, segment_summaries)
    (config_dir / "runner_resource_events.json").write_text(
        json.dumps([asdict(event) for event in runner_events], indent=2),
        encoding="utf-8",
    )

    avg_batch_seconds, throughput_features_per_min = summarize_batch_throughput(
        config,
        batch_events,
        summary_warmup_batches=args.summary_warmup_batches,
    )
    if args.summary_warmup_batches > 0:
        notes.append(f"Reported averages exclude completed batches before {args.summary_warmup_batches}.")
    speedscope_summary = summarize_speedscope(py_spy_output_path) if py_spy_output_path is not None else None
    import_stage_profile = None
    if (
        args.profile_import_stage
        and len(batch_events) >= args.target_batches
        and status in {"target_reached", "exited"}
    ):
        import_stage_profile = profile_import_stage(dashboard_command, local_db_url=args.local_db_url)
        notes.append(
            "Measured exact-lineage import stage profile"
            f" mode={import_stage_profile.mode} wall_seconds={import_stage_profile.wall_seconds:.2f}."
        )

    ended_at_utc = utc_now_iso()
    result = ProfileResult(
        label=config.label,
        n_features_per_batch=config.n_features_per_batch,
        n_prompts_in_forward_pass=config.n_prompts_in_forward_pass,
        status=status,
        reason=reason,
        exit_code=exit_code,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        elapsed_seconds=time.monotonic() - started_monotonic,
        completed_batches=sorted(batch_events),
        avg_batch_seconds=avg_batch_seconds,
        throughput_features_per_min=throughput_features_per_min,
        max_tree_rss_gib=max((sample.tree_rss_gib for sample in samples), default=0.0),
        max_tree_cpu_percent=max((sample.tree_cpu_percent for sample in samples), default=0.0),
        max_host_used_gib=max((sample.host_used_gib for sample in samples), default=0.0),
        max_gpu_process_used_mib=max((sample.gpu_process_used_mib for sample in samples), default=0),
        max_gpu_device_used_mib=max((sample.gpu_device_used_mib for sample in samples), default=0),
        max_gpu_util_percent=max((sample.gpu_util_percent for sample in samples), default=0),
        avg_proc_read_mib_s=finite_average(sample.proc_read_mib_s for sample in samples),
        avg_proc_write_mib_s=finite_average(sample.proc_write_mib_s for sample in samples),
        avg_cache_read_mib_s=finite_average(sample.cache_read_mib_s for sample in samples),
        avg_cache_write_mib_s=finite_average(sample.cache_write_mib_s for sample in samples),
        avg_cache_read_ms_per_op=finite_average(sample.cache_read_ms_per_op for sample in samples),
        avg_cache_write_ms_per_op=finite_average(sample.cache_write_ms_per_op for sample in samples),
        avg_cache_io_util_percent=finite_average(sample.cache_io_util_percent for sample in samples),
        pipeline_log_path=str(pipeline_log_path),
        stdout_log_path=str(stdout_path),
        resource_samples_path=str(resource_samples_path),
        stage_summary_path=str(stage_summary_path),
        py_spy_output_path=str(py_spy_output_path) if py_spy_output_path is not None else None,
        py_spy_command=py_spy_command,
        run_root=str(run_root),
        command=shlex.join(dashboard_command),
        cache_device=cache_device,
        speedscope_summary=speedscope_summary,
        import_stage_profile=import_stage_profile,
        notes=notes + last_error_lines,
    )
    (config_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    if args.cleanup_run_dirs:
        shutil.rmtree(run_root, ignore_errors=True)

    print(
        f"[{ended_at_utc}] finished {config.label}: status={result.status} reason={result.reason} "
        f"batches={len(result.completed_batches)} avg_batch_seconds={result.avg_batch_seconds} "
        f"throughput_features_per_min={result.throughput_features_per_min} "
        f"max_tree_rss_gib={result.max_tree_rss_gib:.2f} max_gpu_process_used_mib={result.max_gpu_process_used_mib}",
        flush=True,
    )
    return result


def write_results(session_dir: Path, results: list[ProfileResult]) -> None:
    """Persist session-level JSON and markdown summaries."""

    results_path = session_dir / "results.json"
    results_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    lines = [
        "# Dashboard Generation Profiling Results",
        "",
        "| Config | Status | Batches | Avg batch s | Features/min | Max RSS GiB | Max CPU % | "
        "Max GPU MiB | Max GPU util % | Proc read MiB/s | Proc write MiB/s | Cache read MiB/s | "
        "Cache write MiB/s | Cache read ms/op | Cache write ms/op | Cache IO util % | Import wall s | "
        "Import load s | Import act s | Import rows | py-spy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        import_profile = result.import_stage_profile
        lines.append(
            (
                "| {label} | {status} | {batches} | {avg_batch} | {throughput} | {rss} | {cpu} | "
                "{gpu_mib} | {gpu_util} | {proc_read} | {proc_write} | {cache_read} | {cache_write} | "
                "{read_lat} | {write_lat} | {io_util} | {import_wall} | {import_load} | {import_act} | "
                "{import_rows} | {py_spy} |"
            ).format(
                label=result.label,
                status=result.status,
                batches=len(result.completed_batches),
                avg_batch=f"{result.avg_batch_seconds:.1f}" if result.avg_batch_seconds is not None else "-",
                throughput=(
                    f"{result.throughput_features_per_min:.0f}"
                    if result.throughput_features_per_min is not None
                    else "-"
                ),
                rss=f"{result.max_tree_rss_gib:.2f}",
                cpu=f"{result.max_tree_cpu_percent:.1f}",
                gpu_mib=result.max_gpu_process_used_mib,
                gpu_util=result.max_gpu_util_percent,
                proc_read=f"{result.avg_proc_read_mib_s:.2f}" if result.avg_proc_read_mib_s is not None else "-",
                proc_write=f"{result.avg_proc_write_mib_s:.2f}" if result.avg_proc_write_mib_s is not None else "-",
                cache_read=f"{result.avg_cache_read_mib_s:.2f}" if result.avg_cache_read_mib_s is not None else "-",
                cache_write=f"{result.avg_cache_write_mib_s:.2f}" if result.avg_cache_write_mib_s is not None else "-",
                read_lat=f"{result.avg_cache_read_ms_per_op:.3f}"
                if result.avg_cache_read_ms_per_op is not None
                else "-",
                write_lat=(
                    f"{result.avg_cache_write_ms_per_op:.3f}" if result.avg_cache_write_ms_per_op is not None else "-"
                ),
                io_util=f"{result.avg_cache_io_util_percent:.1f}"
                if result.avg_cache_io_util_percent is not None
                else "-",
                import_wall=f"{import_profile.wall_seconds:.2f}" if import_profile is not None else "-",
                import_load=(
                    f"{import_profile.activation_load_seconds:.2f}"
                    if import_profile is not None and import_profile.activation_load_seconds is not None
                    else "-"
                ),
                import_act=(
                    f"{import_profile.activation_import_seconds:.2f}"
                    if import_profile is not None and import_profile.activation_import_seconds is not None
                    else "-"
                ),
                import_rows=import_profile.imported_activation_rows if import_profile is not None else "-",
                py_spy=Path(result.py_spy_output_path).name if result.py_spy_output_path else "-",
            )
        )
    (session_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run requested dashboard profiling probes."""

    parser = build_parser()
    args = parser.parse_args()
    configs = [parse_config_spec(spec) for spec in args.config]
    if not configs:
        configs = default_profile_configs(args.include_comparisons)
    if not configs:
        parser.error("No configs selected.")

    session_dir = args.session_root / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    cache_device = resolve_block_device(args.cache_path)
    env = build_env()

    print(f"Session directory: {session_dir}", flush=True)
    print(f"Configs: {[config.label for config in configs]}", flush=True)
    print(f"Cache device for {args.cache_path}: {cache_device or 'unavailable'}", flush=True)

    results: list[ProfileResult] = []
    for config in configs:
        result = run_profile(config, args=args, session_dir=session_dir, env=env, cache_device=cache_device)
        results.append(result)
        write_results(session_dir, results)

    print(f"Wrote results to {session_dir / 'results.json'} and {session_dir / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
