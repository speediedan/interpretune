#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from interpretune.utils import neuronpedia_dashboard_pipeline as dashboard_pipeline


WORKER_CLI_FLAG_BY_KEY = {
    "cuda_visible_devices": "--cuda-visible-devices",
    "start_layer": "--start-layer",
    "end_layer": "--end-layer",
    "n_features_per_batch": "--n-features-per-batch",
    "n_prompts_in_forward_pass": "--n-prompts-in-forward-pass",
    "primary_acts_batch_size": "--primary-acts-batch-size",
    "runner_log_performance": "--runner-log-performance",
    "runner_profile_rolling_substages": "--runner-profile-rolling-substages",
    "runner_torch_profile": "--runner-torch-profile",
    "runner_torch_profile_dir": "--runner-torch-profile-dir",
    "heartbeat_seconds": "--heartbeat-seconds",
    "stall_timeout_seconds": "--stall-timeout-seconds",
    "layer_lock_stale_seconds": "--layer-lock-stale-seconds",
}
ALLOWED_WORKER_KEYS = {"id", *WORKER_CLI_FLAG_BY_KEY}
LAUNCHER_EXIT_CODE_RE = re.compile(
    r"(?:Diagnostics reason=nonzero-exit-|Dashboard generation failed for layer \d+ with exit code )(?P<code>-?\d+)"
)


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the Neuronpedia dashboard pipeline from a YAML config. Extra arguments are forwarded to the "
            "pipeline CLI and override config-file values."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the dashboard pipeline YAML config file.")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run the pipeline in the foreground even if launcher.background is true in the config.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the fully resolved pipeline command before launching it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command(s) and exit without launching any dashboard workers.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Launch a detached monitor process that restarts workers after OOMs with safer batch settings.",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable launcher.monitor from the config file.",
    )
    parser.add_argument(
        "--monitor-foreground",
        action="store_true",
        help="Run the monitor loop in this process. Intended for testing and debugging the monitor.",
    )
    parser.add_argument(
        "--monitor-heartbeat-seconds",
        type=int,
        help="Seconds between monitor heartbeats when monitor mode is enabled.",
    )
    parser.add_argument(
        "--run-monitor",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--monitor-manifest",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--monitor-log-path",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def _build_pipeline_config(
    config_path: Path,
    passthrough_args: list[str],
) -> dashboard_pipeline.NeuronpediaDashboardPipelineConfig:
    parsed_args = dashboard_pipeline._parse_args(["--config", str(config_path), *passthrough_args])
    return dashboard_pipeline._build_dashboard_pipeline_config(parsed_args)


def _worker_passthrough_args(base_passthrough_args: list[str], worker: dict[str, Any]) -> list[str]:
    unknown_keys = sorted(set(worker) - ALLOWED_WORKER_KEYS)
    if unknown_keys:
        raise ValueError(
            "launcher.workers entries only support these keys: "
            f"{', '.join(sorted(ALLOWED_WORKER_KEYS))}; unknown: {', '.join(unknown_keys)}"
        )
    worker_id = str(worker.get("id", "")).strip()
    if not worker_id:
        raise ValueError("Each launcher.workers entry must define a non-empty id.")
    args = [*base_passthrough_args, "--worker-id", worker_id, "--enable-layer-locks"]
    for key, flag_name in WORKER_CLI_FLAG_BY_KEY.items():
        value = worker.get(key)
        if value is None:
            continue
        args.extend([flag_name, str(value)])
    return args


def _pipeline_command(config_path: Path, passthrough_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "interpretune.utils.neuronpedia_dashboard_pipeline",
        "--config",
        str(config_path),
        *passthrough_args,
    ]


def _monitor_command(
    *,
    config_path: Path,
    passthrough_args: list[str],
    manifest_path: Path,
    monitor_log_path: Path,
    heartbeat_seconds: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--run-monitor",
        "--monitor-manifest",
        str(manifest_path),
        "--monitor-log-path",
        str(monitor_log_path),
        "--monitor-heartbeat-seconds",
        str(heartbeat_seconds),
        *passthrough_args,
    ]


def _launch_pipeline_process(
    *,
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as launcher_log:
        return subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=launcher_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def _launch_monitor_process(
    *,
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as monitor_log:
        return subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=monitor_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _current_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _halve_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    int_value = int(value)
    if int_value <= 1:
        return None
    return max(1, int_value // 2)


def _worker_with_oom_reduction(worker: Mapping[str, Any], oom_count: int) -> dict[str, Any] | None:
    updated_worker = dict(worker)
    if oom_count == 1:
        reduced_primary_acts_batch_size = _halve_positive_int(updated_worker.get("primary_acts_batch_size"))
        if reduced_primary_acts_batch_size is not None:
            updated_worker["primary_acts_batch_size"] = reduced_primary_acts_batch_size
            return updated_worker
        reduced_prompt_batch_size = _halve_positive_int(updated_worker.get("n_prompts_in_forward_pass"))
        if reduced_prompt_batch_size is not None:
            updated_worker["n_prompts_in_forward_pass"] = reduced_prompt_batch_size
            return updated_worker
        return None
    if oom_count == 2:
        reduced_prompt_batch_size = _halve_positive_int(updated_worker.get("n_prompts_in_forward_pass"))
        if reduced_prompt_batch_size is not None:
            updated_worker["n_prompts_in_forward_pass"] = reduced_prompt_batch_size
            return updated_worker
        return None
    return None


def _worker_progress_summary(worker_config: dashboard_pipeline.NeuronpediaDashboardPipelineConfig) -> dict[str, int]:
    progress: dict[str, int] = {}
    for layer_dir in sorted(worker_config.run_directory.glob("layer_*")):
        if not layer_dir.is_dir():
            continue
        progress[layer_dir.name] = len(list(layer_dir.rglob("batch-*.json")))
    return progress


def _worker_requested_layers_complete(worker_config: dashboard_pipeline.NeuronpediaDashboardPipelineConfig) -> bool:
    requested_layers = set(range(worker_config.start_layer, worker_config.end_layer + 1))
    completed_layers = dashboard_pipeline.completed_layers_from_logs(
        *dashboard_pipeline._completed_log_paths(worker_config)
    )
    return bool(requested_layers) and requested_layers <= completed_layers


def _launcher_log_exit_code(log_path: Path, *, start_offset: int = 0) -> int | None:
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        if start_offset > 0:
            log_file.seek(start_offset)
        exit_code: int | None = None
        for line in log_file:
            match = LAUNCHER_EXIT_CODE_RE.search(line)
            if match is not None:
                exit_code = int(match.group("code"))
        return exit_code


def _worker_runtime_snapshot(pid: int) -> str:
    try:
        process_group_id = os.getpgid(pid)
    except ProcessLookupError:
        return "pid=not-running"
    completed = subprocess.run(
        ["bash", "-lc", f"ps -o pid,ppid,pgid,stat,etimes,%cpu,%mem,rss,args -g {process_group_id} --no-headers"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip().replace("\n", " || ") or f"pgid={process_group_id} empty"


def _restart_worker(
    *,
    state: dict[str, Any],
    config_path: Path,
    passthrough_args: list[str],
    repo_root: Path,
    env: dict[str, str],
    timestamp: str,
) -> subprocess.Popen[str]:
    worker = state["worker"]
    worker_args = _worker_passthrough_args(passthrough_args, worker)
    worker_config = _build_pipeline_config(config_path, worker_args)
    worker_id = worker_config.worker_id or str(worker["id"])
    worker_command = _pipeline_command(config_path, worker_args)
    worker_log_path = worker_config.run_directory / f"launcher.{worker_id}.monitor.{timestamp}.log"
    pipeline_log_path = worker_config.pipeline_log_path
    assert pipeline_log_path is not None
    state["pipeline_log"] = pipeline_log_path
    state["launcher_log"] = worker_log_path
    state["pipeline_log_offset"] = _current_file_size(pipeline_log_path)
    state["launcher_log_offset"] = 0
    state["worker_config"] = worker_config
    process = _launch_pipeline_process(
        command=worker_command,
        repo_root=repo_root,
        env=env,
        log_path=worker_log_path,
    )
    state["pid"] = process.pid
    state["process"] = process
    return process


def _run_monitor_loop(
    *,
    config_path: Path,
    passthrough_args: list[str],
    manifest_path: Path,
    heartbeat_seconds: int,
    repo_root: Path,
    env: dict[str, str],
) -> int:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError(f"Monitor manifest must contain a list of workers: {manifest_path}")

    states: list[dict[str, Any]] = []
    for worker_manifest in manifest_payload:
        worker = dict(worker_manifest["worker"])
        worker_config = _build_pipeline_config(config_path, _worker_passthrough_args(passthrough_args, worker))
        states.append(
            {
                "worker_id": worker_manifest["worker_id"],
                "worker": worker,
                "pid": int(worker_manifest["pid"]),
                "process": None,
                "pipeline_log": Path(worker_manifest["pipeline_log"]),
                "launcher_log": Path(worker_manifest["launcher_log"]),
                "pipeline_log_offset": int(worker_manifest.get("pipeline_log_initial_size", 0)),
                "launcher_log_offset": int(worker_manifest.get("launcher_log_initial_size", 0)),
                "oom_count": 0,
                "completed": False,
                "disabled": False,
                "worker_config": worker_config,
            }
        )

    print(
        "MONITOR_START manifest=%s workers=%s heartbeat_seconds=%s"
        % (manifest_path, ",".join(str(state["worker_id"]) for state in states), heartbeat_seconds),
        flush=True,
    )
    while True:
        for state in states:
            if state["completed"] or state["disabled"]:
                continue
            pid = int(state["pid"])
            running = _pid_is_running(pid)
            process = state.get("process")
            return_code = process.poll() if process is not None else None
            if running and return_code is None:
                worker_config = state["worker_config"]
                print(
                    "MONITOR_HEARTBEAT worker=%s pid=%s oom_count=%s config=%s progress=%s runtime=%s gpu=%s"
                    % (
                        state["worker_id"],
                        pid,
                        state["oom_count"],
                        {
                            "cuda_visible_devices": worker_config.cuda_visible_devices,
                            "n_features_per_batch": worker_config.n_features_per_batch,
                            "n_prompts_in_forward_pass": worker_config.n_prompts_in_forward_pass,
                            "primary_acts_batch_size": worker_config.primary_acts_batch_size,
                        },
                        _worker_progress_summary(worker_config),
                        _worker_runtime_snapshot(pid),
                        dashboard_pipeline._gpu_snapshot(pid),
                    ),
                    flush=True,
                )
                continue

            log_path = Path(state["pipeline_log"])
            log_offset = int(state["pipeline_log_offset"])
            launcher_log_path = Path(state["launcher_log"])
            launcher_log_offset = int(state.get("launcher_log_offset", 0))
            exit_code = _launcher_log_exit_code(launcher_log_path, start_offset=launcher_log_offset)
            oom_detected = dashboard_pipeline.dashboard_log_contains_oom(
                log_path, start_offset=log_offset
            ) or dashboard_pipeline.dashboard_log_contains_oom(launcher_log_path, start_offset=launcher_log_offset)
            if exit_code == -9 and not oom_detected:
                print(
                    "MONITOR_OOM_LIKE_EXIT worker=%s pid=%s return_code=%s launcher_log=%s"
                    % (state["worker_id"], pid, exit_code, launcher_log_path),
                    flush=True,
                )
                oom_detected = True
            if oom_detected:
                state["oom_count"] += 1
                oom_count = int(state["oom_count"])
                print(
                    "MONITOR_OOM_OBSERVED worker=%s pid=%s oom_count=%s logs=%s,%s offsets=%s,%s"
                    % (
                        state["worker_id"],
                        pid,
                        oom_count,
                        log_path,
                        launcher_log_path,
                        log_offset,
                        launcher_log_offset,
                    ),
                    flush=True,
                )
                next_worker = _worker_with_oom_reduction(state["worker"], oom_count)
                if next_worker is None:
                    state["disabled"] = True
                    print(
                        "MONITOR_OOM_DISABLED worker=%s oom_count=%s message=%s"
                        % (
                            state["worker_id"],
                            oom_count,
                            "two automatic OOM mitigations have already been attempted; "
                            "manually inspect and tune this worker config",
                        ),
                        flush=True,
                    )
                    continue
                state["worker"] = next_worker
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                restarted_process = _restart_worker(
                    state=state,
                    config_path=config_path,
                    passthrough_args=passthrough_args,
                    repo_root=repo_root,
                    env=env,
                    timestamp=timestamp,
                )
                worker_config = state["worker_config"]
                print(
                    "MONITOR_RESTART worker=%s pid=%s oom_count=%s config=%s pipeline_log=%s launcher_log=%s"
                    % (
                        state["worker_id"],
                        restarted_process.pid,
                        oom_count,
                        {
                            "cuda_visible_devices": worker_config.cuda_visible_devices,
                            "n_features_per_batch": worker_config.n_features_per_batch,
                            "n_prompts_in_forward_pass": worker_config.n_prompts_in_forward_pass,
                            "primary_acts_batch_size": worker_config.primary_acts_batch_size,
                        },
                        worker_config.pipeline_log_path,
                        state["launcher_log"],
                    ),
                    flush=True,
                )
                continue

            worker_config = state["worker_config"]
            if return_code not in (None, 0) or exit_code is not None:
                state["disabled"] = True
                print(
                    "MONITOR_WORKER_FAILED_WITHOUT_OOM worker=%s pid=%s return_code=%s log=%s launcher_log=%s"
                    % (
                        state["worker_id"],
                        pid,
                        return_code if return_code not in (None, 0) else exit_code,
                        log_path,
                        launcher_log_path,
                    ),
                    flush=True,
                )
            elif return_code == 0 or _worker_requested_layers_complete(worker_config):
                state["completed"] = True
                print(
                    "MONITOR_WORKER_COMPLETED worker=%s pid=%s log=%s" % (state["worker_id"], pid, log_path),
                    flush=True,
                )
            else:
                state["disabled"] = True
                print(
                    "MONITOR_WORKER_STOPPED_WITHOUT_OOM worker=%s pid=%s log=%s progress=%s"
                    % (state["worker_id"], pid, log_path, _worker_progress_summary(worker_config)),
                    flush=True,
                )

        if all(state["completed"] or state["disabled"] for state in states):
            print(
                "MONITOR_EXIT states=%s"
                % [
                    {
                        "worker_id": state["worker_id"],
                        "completed": state["completed"],
                        "disabled": state["disabled"],
                        "oom_count": state["oom_count"],
                    }
                    for state in states
                ],
                flush=True,
            )
            return 0
        time.sleep(max(1, heartbeat_seconds))


def main(argv: list[str] | None = None) -> int:
    parser = _create_argument_parser()
    args, passthrough_args = parser.parse_known_args(argv)

    config_path = args.config.expanduser().resolve()
    pipeline_config = _build_pipeline_config(config_path, passthrough_args)
    launcher_settings = dashboard_pipeline.load_dashboard_launcher_settings(
        config_path,
        pipeline_config=pipeline_config,
    )
    workers = launcher_settings.get("workers", [])
    monitor_enabled = bool(args.monitor or (launcher_settings["monitor"] and not args.no_monitor))
    monitor_heartbeat_seconds = int(args.monitor_heartbeat_seconds or launcher_settings["monitor_heartbeat_seconds"])

    command = _pipeline_command(config_path, passthrough_args)
    env = os.environ.copy()
    env.update(launcher_settings["env"])
    repo_root = Path(__file__).resolve().parents[1]

    if args.run_monitor:
        if args.monitor_manifest is None or args.monitor_log_path is None:
            parser.error("--run-monitor requires --monitor-manifest and --monitor-log-path")
        return _run_monitor_loop(
            config_path=config_path,
            passthrough_args=passthrough_args,
            manifest_path=args.monitor_manifest.expanduser().resolve(),
            heartbeat_seconds=monitor_heartbeat_seconds,
            repo_root=repo_root,
            env=env,
        )

    if (args.print_command or args.dry_run) and not workers:
        print("Resolved command:")
        print("  " + " ".join(shlex.quote(part) for part in command))
        if args.dry_run:
            return 0

    if workers:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        worker_processes: list[tuple[str, Any, Path, Path, subprocess.Popen[str], dict[str, Any], int]] = []
        for worker in workers:
            worker_args = _worker_passthrough_args(passthrough_args, worker)
            worker_config = _build_pipeline_config(config_path, worker_args)
            worker_id = worker_config.worker_id or str(worker["id"])
            worker_command = _pipeline_command(config_path, worker_args)
            worker_log_path = worker_config.run_directory / f"launcher.{worker_id}.{timestamp}.log"
            pipeline_log_path = worker_config.pipeline_log_path
            assert pipeline_log_path is not None
            pipeline_log_initial_size = _current_file_size(pipeline_log_path)
            if args.print_command or args.dry_run:
                print(f"Resolved command for worker {worker_id}:")
                print("  " + " ".join(shlex.quote(part) for part in worker_command))
            if args.dry_run:
                continue
            process = _launch_pipeline_process(
                command=worker_command,
                repo_root=repo_root,
                env=env,
                log_path=worker_log_path,
            )
            worker_processes.append(
                (
                    worker_id,
                    worker_config,
                    worker_log_path,
                    pipeline_log_path,
                    process,
                    dict(worker),
                    pipeline_log_initial_size,
                )
            )

        if args.dry_run:
            return 0

        manifest_path = pipeline_config.run_directory / f"launcher.workers.{timestamp}.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {
                        "worker_id": worker_id,
                        "pid": process.pid,
                        "launcher_log": str(worker_log_path),
                        "pipeline_log": str(pipeline_log_path),
                        "pipeline_log_initial_size": pipeline_log_initial_size,
                        "cuda_visible_devices": worker_config.cuda_visible_devices,
                        "n_features_per_batch": worker_config.n_features_per_batch,
                        "n_prompts_in_forward_pass": worker_config.n_prompts_in_forward_pass,
                        "primary_acts_batch_size": worker_config.primary_acts_batch_size,
                        "worker": worker,
                    }
                    for (
                        worker_id,
                        worker_config,
                        worker_log_path,
                        pipeline_log_path,
                        process,
                        worker,
                        pipeline_log_initial_size,
                    ) in worker_processes
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        for worker_id, worker_config, worker_log_path, _, process, _, _ in worker_processes:
            print(f"Launched dashboard pipeline worker {worker_id} PID {process.pid}")
            print(f"  Launcher log: {worker_log_path}")
            print(f"  Pipeline log: {worker_config.pipeline_log_path}")
            print(f"  tail -f {worker_config.pipeline_log_path}")
        print(f"Worker manifest: {manifest_path}")

        monitor_process: subprocess.Popen[str] | None = None
        if monitor_enabled:
            monitor_log_path = pipeline_config.run_directory / f"monitor.{timestamp}.log"
            monitor_cmd = _monitor_command(
                config_path=config_path,
                passthrough_args=passthrough_args,
                manifest_path=manifest_path,
                monitor_log_path=monitor_log_path,
                heartbeat_seconds=monitor_heartbeat_seconds,
            )
            if args.print_command:
                print("Resolved command for monitor:")
                print("  " + " ".join(shlex.quote(part) for part in monitor_cmd))
            if args.monitor_foreground:
                print(f"Running dashboard monitor in foreground: {monitor_log_path}")
                return _run_monitor_loop(
                    config_path=config_path,
                    passthrough_args=passthrough_args,
                    manifest_path=manifest_path,
                    heartbeat_seconds=monitor_heartbeat_seconds,
                    repo_root=repo_root,
                    env=env,
                )
            monitor_process = _launch_monitor_process(
                command=monitor_cmd,
                repo_root=repo_root,
                env=env,
                log_path=monitor_log_path,
            )
            print(f"Launched dashboard monitor PID {monitor_process.pid}")
            print(f"  Monitor log: {monitor_log_path}")
            print(f"  tail -f {monitor_log_path}")

        if args.foreground or not launcher_settings["background"]:
            return_codes = [process.wait() for _, _, _, _, process, _, _ in worker_processes]
            if monitor_process is not None:
                return_codes.append(monitor_process.wait())
            return max(return_codes, default=0)
        return 0

    if args.foreground or not launcher_settings["background"]:
        completed = subprocess.run(command, cwd=repo_root, env=env, check=False)
        return completed.returncode

    log_path = launcher_settings["log_path"]
    if log_path is None:
        raise ValueError("Background launches require launcher.log_path or a pipeline_config-derived log path.")

    process = _launch_pipeline_process(command=command, repo_root=repo_root, env=env, log_path=log_path)

    print(f"Launched dashboard pipeline PID {process.pid}")
    print(f"Launcher log: {log_path}")
    print(f"Pipeline log: {pipeline_config.pipeline_log_path}")
    print(f"tail -f {pipeline_config.pipeline_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
