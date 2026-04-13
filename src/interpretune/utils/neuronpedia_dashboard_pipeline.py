from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import certifi

from interpretune.utils.neuronpedia_explanations import (
    DEFAULT_EXPLANATION_AUTHOR_ID,
    DEFAULT_IT_NP_CACHE,
    DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL,
    LocalNeuronpediaServiceStatus,
    NeuronpediaLocalImportSummary,
    check_local_neuronpedia_services,
    import_neuronpedia_export_bundle_local_db,
)


DONE_LAYER_RE = re.compile(r"^DONE layer=(\d+) ")


@dataclass(frozen=True)
class NeuronpediaDashboardLayerResult:
    """Result metadata for one generated layer."""

    layer_num: int
    output_dir: Path
    export_root: Path | None
    import_summary: NeuronpediaLocalImportSummary | None
    elapsed_seconds: float
    skipped: bool = False


@dataclass
class NeuronpediaDashboardPipelineConfig:
    """Configuration for generating, converting, and importing Neuronpedia dashboard layers."""

    model_name: str
    sae_set: str
    neuronpedia_source_set_id: str
    neuronpedia_source_set_description: str
    creator_name: str
    release_id: str
    release_title: str
    release_url: str
    hf_weights_repo_id: str
    hf_weights_path_template: str
    hook_point: str
    prompts_huggingface_dataset_path: str
    start_layer: int
    end_layer: int
    sae_path_template: str
    run_root: Path = DEFAULT_IT_NP_CACHE / "dashboard_runs"
    export_root: Path = Path("/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports")
    existing_log_path: Path | None = None
    pipeline_log_path: Path | None = None
    saedashboard_repo_root: Path = Path("/home/speediedan/repos/SAEDashboard")
    saelens_repo_root: Path = Path("/home/speediedan/repos/SAELens")
    neuronpedia_utils_root: Path = Path("/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils")
    interpretune_env_file: Path | None = Path("/home/speediedan/repos/interpretune/.env")
    python_executable: str = sys.executable
    use_skip_transcoder: bool = False
    sae_dtype: str = "float32"
    model_dtype: str = "bfloat16"
    sparsity_threshold: int = 1
    n_prompts_total: int = 24576
    n_tokens_in_prompt: int = 128
    n_features_per_batch: int = 128
    n_prompts_in_forward_pass: int = 32
    zero_out_bos_token: bool = False
    cuda_visible_devices: str | None = "0"
    heartbeat_seconds: int = 60
    stall_timeout_seconds: int = 0
    import_to_local_db: bool = True
    local_db_url: str | None = None
    webapp_url: str = DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL
    archive_partial_dirs: bool = True
    resume_from_existing_logs: bool = True
    cert_bundle_path: Path = field(default_factory=lambda: Path(certifi.where()))

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root)
        self.export_root = Path(self.export_root)
        self.saedashboard_repo_root = Path(self.saedashboard_repo_root)
        self.saelens_repo_root = Path(self.saelens_repo_root)
        self.neuronpedia_utils_root = Path(self.neuronpedia_utils_root)
        if self.interpretune_env_file is not None:
            self.interpretune_env_file = Path(self.interpretune_env_file)
        self.cert_bundle_path = Path(self.cert_bundle_path)
        if self.existing_log_path is None:
            self.existing_log_path = self.run_directory / "run.log"
        if self.pipeline_log_path is None:
            self.pipeline_log_path = self.run_directory / f"run.resume-{self.start_layer}-{self.end_layer}.log"

    @property
    def run_name(self) -> str:
        return f"{self.model_name}_{self.neuronpedia_source_set_id}"

    @property
    def run_directory(self) -> Path:
        return self.run_root / self.run_name

    def sae_path_for_layer(self, layer_num: int) -> str:
        return self.sae_path_template.format(layer=layer_num)

    def hf_weights_path_for_layer(self, layer_num: int) -> str:
        return self.hf_weights_path_template.format(layer=layer_num)

    def output_dir_for_layer(self, layer_num: int) -> Path:
        return self.run_directory / f"layer_{layer_num}"


def _load_env_file_values(env_file: Path | None) -> dict[str, str]:
    if env_file is None or not env_file.exists():
        return {}

    from dotenv import dotenv_values

    return {key: value for key, value in dotenv_values(env_file).items() if value is not None}


def completed_layers_from_logs(*log_paths: Path) -> set[int]:
    """Parse all completed layer markers from one or more pipeline log files."""

    completed: set[int] = set()
    for log_path in log_paths:
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = DONE_LAYER_RE.match(line.strip())
            if match:
                completed.add(int(match.group(1)))
    return completed


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"interpretune.neuronpedia_dashboard_pipeline.{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _build_generation_env(config: NeuronpediaDashboardPipelineConfig) -> dict[str, str]:
    env = os.environ.copy()
    env_file_values = _load_env_file_values(config.interpretune_env_file)
    for key, value in env_file_values.items():
        env.setdefault(key, value)

    default_hf_home = str(DEFAULT_IT_NP_CACHE.parents[2])
    env["IT_NP_CACHE"] = env.get("IT_NP_CACHE", str(DEFAULT_IT_NP_CACHE))
    env["HF_HOME"] = env.get("HF_HOME", default_hf_home)
    env["HF_DATASETS_CACHE"] = env.get("HF_DATASETS_CACHE", os.path.join(env["HF_HOME"], "datasets"))
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
    env["HF_TOKEN"] = env.get("HF_TOKEN") or env.get("HF_GATED_PUBLIC_REPO_AUTH_KEY") or env.get("HF_MCP_TOKEN_RW", "")
    env["SSL_CERT_FILE"] = env.get("SSL_CERT_FILE", str(config.cert_bundle_path))
    env["REQUESTS_CA_BUNDLE"] = env.get("REQUESTS_CA_BUNDLE", str(config.cert_bundle_path))
    env["CURL_CA_BUNDLE"] = env.get("CURL_CA_BUNDLE", str(config.cert_bundle_path))
    env["DEFAULT_CREATOR_ID"] = env.get("DEFAULT_CREATOR_ID", DEFAULT_EXPLANATION_AUTHOR_ID)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
    env["TQDM_DISABLE"] = env.get("TQDM_DISABLE", "1")
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = env.get("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if config.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    pythonpath_entries = [str(config.saelens_repo_root), str(config.saedashboard_repo_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _archive_partial_output(output_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = output_dir.with_name(f"{output_dir.name}.partial.{timestamp}")
    output_dir.rename(archive_path)
    return archive_path


def _directory_stats(root_dir: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    if not root_dir.exists():
        return file_count, total_bytes
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return file_count, total_bytes


def _dashboard_leaf_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    leaf_dirs: list[Path] = []
    for root, _, files in os.walk(output_dir):
        if any(file_name.startswith("batch-") and file_name.endswith(".json") for file_name in files):
            leaf_dirs.append(Path(root))
    return sorted(leaf_dirs)


def _resolve_dashboard_leaf_dir(output_dir: Path) -> Path:
    leaf_dirs = _dashboard_leaf_dirs(output_dir)
    if not leaf_dirs:
        raise RuntimeError(f"No dashboard leaf directory with batch JSON files found under {output_dir}")
    if len(leaf_dirs) == 1:
        return leaf_dirs[0]
    return max(
        leaf_dirs,
        key=lambda path: (len(list(path.glob("batch-*.json"))), len(path.parts), str(path)),
    )


def _run_command_lines(command: list[str]) -> list[str]:
    executable = shutil.which(command[0])
    if executable is None:
        return [f"missing executable: {command[0]}"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except Exception as exc:
        return [f"command failed: {' '.join(command)}: {exc}"]
    stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    stderr_lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    lines = stdout_lines + stderr_lines
    if not lines:
        lines = [f"exit={completed.returncode} no output"]
    return lines[:20]


def _process_snapshot(pid: int) -> str:
    lines = _run_command_lines(
        [
            "ps",
            "-o",
            "pid=,ppid=,pgid=,stat=,%cpu=,%mem=,rss=,vsz=,etimes=,cmd=",
            "-p",
            str(pid),
        ]
    )
    return " | ".join(lines)


def _gpu_snapshot(pid: int) -> str:
    lines = _run_command_lines(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_lines = _run_command_lines(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    pid_prefix = f"{pid},"
    matching_process_lines = [line for line in process_lines if line.startswith(pid_prefix)]
    return "gpus=" + " || ".join(lines[:8]) + " ; pid=" + " || ".join(matching_process_lines or ["not listed"])


def _kernel_snapshot() -> str:
    return " || ".join(_run_command_lines(["dmesg", "-T", "--level=err,crit,alert,emerg"]))


def _log_runtime_diagnostics(logger: logging.Logger, *, pid: int, output_dir: Path, reason: str) -> None:
    file_count, total_bytes = _directory_stats(output_dir)
    leaf_dirs = _dashboard_leaf_dirs(output_dir)
    logger.info(
        "Diagnostics reason=%s pid=%s files=%s bytes=%s leaf_dirs=%s ps=%s gpu=%s kernel=%s",
        reason,
        pid,
        file_count,
        total_bytes,
        [str(path) for path in leaf_dirs[:4]],
        _process_snapshot(pid),
        _gpu_snapshot(pid),
        _kernel_snapshot(),
    )


def _layer_runner_command(config: NeuronpediaDashboardPipelineConfig, layer_num: int, output_dir: Path) -> list[str]:
    command = [
        config.python_executable,
        "-m",
        "sae_dashboard.neuronpedia.neuronpedia_runner",
        f"--sae-set={config.sae_set}",
        f"--sae-path={config.sae_path_for_layer(layer_num)}",
        f"--np-set-name={config.neuronpedia_source_set_id}",
        f"--dataset-path={config.prompts_huggingface_dataset_path}",
        f"--output-dir={output_dir}",
        f"--sae_dtype={config.sae_dtype}",
        f"--model_dtype={config.model_dtype}",
        f"--sparsity-threshold={config.sparsity_threshold}",
        f"--n-prompts={config.n_prompts_total}",
        f"--n-tokens-in-prompt={config.n_tokens_in_prompt}",
        f"--n-features-per-batch={config.n_features_per_batch}",
        f"--n-prompts-in-forward-pass={config.n_prompts_in_forward_pass}",
    ]
    if config.use_skip_transcoder:
        command.append("--use-skip-transcoder")
    if config.zero_out_bos_token:
        command.append("--zero-out-bos-token")
    return command


def _monitor_process(
    process: subprocess.Popen[str],
    *,
    output_dir: Path,
    logger: logging.Logger,
    heartbeat_seconds: int,
    stall_timeout_seconds: int,
) -> int:
    last_seen_size = 0
    last_seen_file_count = 0
    last_growth_time = time.monotonic()
    while True:
        return_code = process.poll()
        if return_code is not None:
            return return_code
        if heartbeat_seconds > 0:
            time.sleep(heartbeat_seconds)

        current_file_count, current_size = _directory_stats(output_dir)
        if current_size > last_seen_size or current_file_count > last_seen_file_count:
            last_seen_size = current_size
            last_seen_file_count = current_file_count
            last_growth_time = time.monotonic()

        elapsed_without_growth = time.monotonic() - last_growth_time
        logger.info(
            "Heartbeat output_dir=%s files=%s bytes=%s elapsed_without_growth=%.1fs pid=%s ps=%s gpu=%s",
            output_dir,
            current_file_count,
            current_size,
            elapsed_without_growth,
            process.pid,
            _process_snapshot(process.pid),
            _gpu_snapshot(process.pid),
        )
        if stall_timeout_seconds > 0 and elapsed_without_growth >= stall_timeout_seconds:
            _log_runtime_diagnostics(logger, pid=process.pid, output_dir=output_dir, reason="stall-timeout")
            process.terminate()
            raise RuntimeError(
                f"Dashboard generation stalled for {elapsed_without_growth:.0f}s without output growth: {output_dir}"
            )


def _load_converter_module(neuronpedia_utils_root: Path) -> ModuleType:
    module_name = "interpretune_np_convert_saedashboard"
    script_path = neuronpedia_utils_root / "neuronpedia_utils" / "convert-saedashboard-to-neuronpedia-export.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find converter script at {script_path}")
    if str(neuronpedia_utils_root) not in sys.path:
        sys.path.insert(0, str(neuronpedia_utils_root))
    os.environ.setdefault("DEFAULT_CREATOR_ID", DEFAULT_EXPLANATION_AUTHOR_ID)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_source_id(output_dir: Path, layer_num: int, neuronpedia_source_set_id: str) -> str:
    batch_files = sorted(output_dir.glob("batch-*.json"))
    if not batch_files:
        return f"{layer_num}-{neuronpedia_source_set_id}"
    batch_data = json.loads(batch_files[0].read_text(encoding="utf-8"))
    source_suffix = batch_data.get("sae_id_suffix") or ""
    if source_suffix:
        return f"{layer_num}-{neuronpedia_source_set_id}__{source_suffix}"
    return f"{layer_num}-{neuronpedia_source_set_id}"


def _resolve_export_root(export_parent: Path, source_id: str) -> Path:
    direct_path = export_parent / source_id
    if direct_path.exists():
        return direct_path
    candidates = sorted(export_parent.glob(f"{source_id}*"))
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Expected Neuronpedia export root was not created: {direct_path}. candidates={candidates}")


def convert_dashboard_output(
    config: NeuronpediaDashboardPipelineConfig,
    *,
    layer_num: int,
    output_dir: Path,
) -> Path:
    """Convert a SAEDashboard layer output into a Neuronpedia export bundle."""

    dashboard_leaf_dir = _resolve_dashboard_leaf_dir(output_dir)
    module = _load_converter_module(config.neuronpedia_utils_root)
    module_any = cast(Any, module)
    module_any.OUTPUT_DIR = str(config.export_root)
    params = {
        "saedashboard_output_dir": str(dashboard_leaf_dir),
        "creator_name": config.creator_name,
        "release_id": config.release_id,
        "release_title": config.release_title,
        "url": config.release_url,
        "model_name": config.model_name,
        "neuronpedia_source_set_id": config.neuronpedia_source_set_id,
        "neuronpedia_source_set_description": config.neuronpedia_source_set_description,
        "hf_weights_repo_id": config.hf_weights_repo_id,
        "hf_weights_path": config.hf_weights_path_for_layer(layer_num),
        "hook_point": module_any.HOOK_POINT_TYPE_CHOICES(config.hook_point),
        "layer_num": layer_num,
        "prompts_huggingface_dataset_path": config.prompts_huggingface_dataset_path,
        "n_prompts_total": config.n_prompts_total,
        "n_tokens_in_prompt": config.n_tokens_in_prompt,
        "zero_out_bos_token": config.zero_out_bos_token,
    }
    module_any.main(SimpleNamespace(params=params), **params)
    source_id = _resolve_source_id(dashboard_leaf_dir, layer_num, config.neuronpedia_source_set_id)
    export_root = _resolve_export_root(config.export_root / config.model_name, source_id)
    return export_root


def run_dashboard_pipeline(config: NeuronpediaDashboardPipelineConfig) -> list[NeuronpediaDashboardLayerResult]:
    """Run dashboard generation, conversion, and optional local import for a layer range."""

    config.run_directory.mkdir(parents=True, exist_ok=True)
    pipeline_log_path = cast(Path, config.pipeline_log_path)
    existing_log_path = cast(Path, config.existing_log_path)
    logger = _configure_logger(pipeline_log_path)
    env = _build_generation_env(config)

    service_status: LocalNeuronpediaServiceStatus | None = None
    should_import = config.import_to_local_db
    if config.import_to_local_db:
        service_status = check_local_neuronpedia_services(
            local_db_url=config.local_db_url,
            webapp_url=config.webapp_url,
        )
        if not service_status.db_available:
            should_import = False
            logger.warning("Local DB unavailable; continuing without DB import: %s", service_status.db_error)
        if not service_status.webapp_available:
            logger.warning("Local Neuronpedia webapp unavailable: %s", service_status.webapp_error)
        if service_status.db_url_redacted:
            logger.info("Resolved local DB URL: %s", service_status.db_url_redacted)

    completed_layers: set[int] = set()
    if config.resume_from_existing_logs:
        completed_layers = completed_layers_from_logs(existing_log_path, pipeline_log_path)

    logger.info(
        (
            "Starting dashboard pipeline model=%s set=%s layers=%s-%s run_directory=%s "
            "IT_NP_CACHE=%s HF_HOME=%s CUDA_VISIBLE_DEVICES=%s"
        ),
        config.model_name,
        config.neuronpedia_source_set_id,
        config.start_layer,
        config.end_layer,
        config.run_directory,
        env.get("IT_NP_CACHE"),
        env.get("HF_HOME"),
        env.get("CUDA_VISIBLE_DEVICES"),
    )

    results: list[NeuronpediaDashboardLayerResult] = []
    for layer_num in range(config.start_layer, config.end_layer + 1):
        output_dir = config.output_dir_for_layer(layer_num)
        if layer_num in completed_layers:
            logger.info("Skipping already completed layer=%s based on existing logs.", layer_num)
            results.append(
                NeuronpediaDashboardLayerResult(
                    layer_num=layer_num,
                    output_dir=output_dir,
                    export_root=None,
                    import_summary=None,
                    elapsed_seconds=0.0,
                    skipped=True,
                )
            )
            continue

        if output_dir.exists() and config.archive_partial_dirs:
            archive_path = _archive_partial_output(output_dir)
            logger.info("Archived partial output for layer=%s to %s", layer_num, archive_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        layer_start = time.monotonic()
        logger.info(
            "START layer=%s sae_path=%s output_dir=%s",
            layer_num,
            config.sae_path_for_layer(layer_num),
            output_dir,
        )
        with pipeline_log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                _layer_runner_command(config, layer_num, output_dir),
                cwd=str(config.saedashboard_repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = _monitor_process(
                process,
                output_dir=output_dir,
                logger=logger,
                heartbeat_seconds=config.heartbeat_seconds,
                stall_timeout_seconds=config.stall_timeout_seconds,
            )

        if return_code != 0:
            _log_runtime_diagnostics(
                logger,
                pid=process.pid,
                output_dir=output_dir,
                reason=f"nonzero-exit-{return_code}",
            )
            raise RuntimeError(f"Dashboard generation failed for layer {layer_num} with exit code {return_code}")

        logger.info(
            "Layer=%s generation completed; dashboard_leaf_dir=%s",
            layer_num,
            _resolve_dashboard_leaf_dir(output_dir),
        )
        export_root = convert_dashboard_output(config, layer_num=layer_num, output_dir=output_dir)
        logger.info("Converted layer=%s to export_root=%s", layer_num, export_root)

        import_summary = None
        if should_import:
            import_summary = import_neuronpedia_export_bundle_local_db(
                export_root,
                local_db_url=config.local_db_url or "",
            )
            logger.info("Imported layer=%s into local DB counts=%s", layer_num, import_summary.imported_row_counts)

        elapsed_seconds = time.monotonic() - layer_start
        logger.info("DONE layer=%s elapsed_seconds=%.1f", layer_num, elapsed_seconds)
        results.append(
            NeuronpediaDashboardLayerResult(
                layer_num=layer_num,
                output_dir=output_dir,
                export_root=export_root,
                import_summary=import_summary,
                elapsed_seconds=elapsed_seconds,
            )
        )

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAEDashboard layers and import them into a local Neuronpedia DB."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-set", required=True)
    parser.add_argument("--neuronpedia-source-set-id", required=True)
    parser.add_argument("--neuronpedia-source-set-description", required=True)
    parser.add_argument("--creator-name", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-title", required=True)
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--hf-weights-repo-id", required=True)
    parser.add_argument("--hf-weights-path-template", required=True)
    parser.add_argument("--hook-point", required=True)
    parser.add_argument("--prompts-huggingface-dataset-path", required=True)
    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--sae-path-template", required=True)
    parser.add_argument("--run-root", default=str(DEFAULT_IT_NP_CACHE / "dashboard_runs"))
    parser.add_argument(
        "--export-root",
        default="/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports",
    )
    parser.add_argument("--saedashboard-repo-root", default="/home/speediedan/repos/SAEDashboard")
    parser.add_argument("--saelens-repo-root", default="/home/speediedan/repos/SAELens")
    parser.add_argument(
        "--neuronpedia-utils-root",
        default="/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils",
    )
    parser.add_argument("--interpretune-env-file", default="/home/speediedan/repos/interpretune/.env")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--sae-dtype", default="float32")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--sparsity-threshold", type=int, default=1)
    parser.add_argument("--n-prompts-total", type=int, default=24576)
    parser.add_argument("--n-tokens-in-prompt", type=int, default=128)
    parser.add_argument("--n-features-per-batch", type=int, default=128)
    parser.add_argument("--n-prompts-in-forward-pass", type=int, default=32)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--stall-timeout-seconds", type=int, default=0)
    parser.add_argument("--local-db-url")
    parser.add_argument("--webapp-url", default=DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL)
    parser.add_argument("--use-skip-transcoder", action="store_true")
    parser.add_argument("--zero-out-bos-token", action="store_true")
    parser.add_argument("--skip-local-db-import", action="store_true")
    parser.add_argument("--no-archive-partials", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = NeuronpediaDashboardPipelineConfig(
        model_name=args.model_name,
        sae_set=args.sae_set,
        neuronpedia_source_set_id=args.neuronpedia_source_set_id,
        neuronpedia_source_set_description=args.neuronpedia_source_set_description,
        creator_name=args.creator_name,
        release_id=args.release_id,
        release_title=args.release_title,
        release_url=args.release_url,
        hf_weights_repo_id=args.hf_weights_repo_id,
        hf_weights_path_template=args.hf_weights_path_template,
        hook_point=args.hook_point,
        prompts_huggingface_dataset_path=args.prompts_huggingface_dataset_path,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        sae_path_template=args.sae_path_template,
        run_root=Path(args.run_root),
        export_root=Path(args.export_root),
        saedashboard_repo_root=Path(args.saedashboard_repo_root),
        saelens_repo_root=Path(args.saelens_repo_root),
        neuronpedia_utils_root=Path(args.neuronpedia_utils_root),
        interpretune_env_file=Path(args.interpretune_env_file) if args.interpretune_env_file else None,
        python_executable=args.python_executable,
        use_skip_transcoder=args.use_skip_transcoder,
        sae_dtype=args.sae_dtype,
        model_dtype=args.model_dtype,
        sparsity_threshold=args.sparsity_threshold,
        n_prompts_total=args.n_prompts_total,
        n_tokens_in_prompt=args.n_tokens_in_prompt,
        n_features_per_batch=args.n_features_per_batch,
        n_prompts_in_forward_pass=args.n_prompts_in_forward_pass,
        zero_out_bos_token=args.zero_out_bos_token,
        cuda_visible_devices=args.cuda_visible_devices,
        heartbeat_seconds=args.heartbeat_seconds,
        stall_timeout_seconds=args.stall_timeout_seconds,
        import_to_local_db=not args.skip_local_db_import,
        local_db_url=args.local_db_url,
        webapp_url=args.webapp_url,
        archive_partial_dirs=not args.no_archive_partials,
        resume_from_existing_logs=not args.no_resume,
    )
    run_dashboard_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
