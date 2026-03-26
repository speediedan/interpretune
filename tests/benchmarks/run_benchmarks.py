#!/usr/bin/env python
"""
Benchmark Runner
================

Runs experiment configs via the interpretune CLI and compares results
against the benchmark registry. Supports both Lightning CLI (adapter_ctx includes
``lightning``) and core CLI (adapter_ctx uses ``core``) modes.

Usage:
    # Run a specific benchmark (experiment/benchmark_id)
    python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l

    # Run all benchmarks for an experiment
    python tests/benchmarks/run_benchmarks.py --experiment rte_boolq

    # Run all benchmarks
    python tests/benchmarks/run_benchmarks.py --all

    # Run with debug diagnostics
    python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l --debug

    # Update registry after validated run (requires clean working tree)
    python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l --update-registry

    # Force-update registry (bypasses clean working tree check)
    python tests/benchmarks/run_benchmarks.py --all --force-update-registry

    # Limit batches for quick smoke test
    python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l --limit-batches 5
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from tests.benchmarks.benchmark_utils import parse_accuracy

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = Path(__file__).resolve().parent / "benchmark_registry.yaml"
ALLOW_FILE = Path(__file__).resolve().parent / "benchmark_update.allow"
DEBUG_LOG_DIR = Path("/tmp/benchmark_debug")


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False, width=120)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _check_clean_working_tree() -> bool:
    """Return True if the git working tree is clean or benchmark_update.allow exists."""
    if ALLOW_FILE.exists():
        log.warning("benchmark_update.allow found — bypassing clean working tree check")
        return True
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def _get_commit_sha() -> str:
    """Return the current HEAD commit SHA (short form)."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _get_salient_pkg_versions() -> dict[str, str]:
    """Return a dict of salient package versions via collect_env_details."""
    # Import here to avoid hard dependency at module level
    sys.path.insert(0, str(REPO_ROOT / "requirements" / "utils"))
    try:
        from collect_env_details import info_packages

        versions = info_packages()
    finally:
        sys.path.pop(0)
    # Filter to salient packages only (drop torch debug/git metadata)
    salient_keys = [
        "interpretune",
        "lightning",
        "transformer_lens",
        "sae_lens",
        "circuit_tracer",
        "nnsight",
        "transformers",
        "datasets",
        "torch",
    ]
    return {k: str(versions[k]) for k in salient_keys if k in versions}


def _detect_cli_mode(entry: dict) -> str:
    """Detect whether a benchmark uses the Lightning or core CLI from its adapter_ctx."""
    cli_mode = entry.get("cli_mode")
    if cli_mode:
        return cli_mode
    adapter_ctx = entry.get("adapter_ctx", [])
    return "lightning" if "lightning" in adapter_ctx else "core"


def run_cli_benchmark(
    config_path: str,
    limit_batches: int | None = None,
    log_dir: Path | None = None,
    debug: bool = False,
    cli_mode: str = "lightning",
) -> dict:
    """Run a benchmark via the interpretune CLI and capture results.

    Args:
        cli_mode: ``"lightning"`` for Lightning CLI or ``"core"`` for core CLI.

    Returns dict with keys: accuracy, raw_output, return_code, duration_s
    """
    abs_config = REPO_ROOT / config_path
    if not abs_config.exists():
        raise FileNotFoundError(f"Config not found: {abs_config}")

    if cli_mode == "lightning":
        cmd = [
            "interpretune",
            "--lightning_cli",
            "test",
            "--config",
            str(abs_config),
        ]
        if limit_batches is not None:
            cmd.extend(["--trainer.limit_test_batches", str(limit_batches)])
    else:
        cmd = [
            "interpretune",
            "--run_command",
            "test",
            "--config",
            str(abs_config),
        ]
        if limit_batches is not None:
            cmd.extend(["--run_cfg.limit_test_batches", str(limit_batches)])

    env = os.environ.copy()
    if debug:
        env["IT_CI_LOG_LEVEL"] = "DEBUG"

    log.info(f"Running: {' '.join(cmd)}")
    ts = timestamp()

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=1800,  # 30 min max
    )
    duration = time.monotonic() - start

    raw_output = result.stdout + "\n--- STDERR ---\n" + result.stderr

    # Save raw output
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"benchmark_{ts}.log"
        with open(log_file, "w") as f:
            f.write(f"CMD: {' '.join(cmd)}\n")
            f.write(f"RETURN CODE: {result.returncode}\n")
            f.write(f"DURATION: {duration:.1f}s\n")
            f.write(f"{'=' * 80}\n")
            f.write(raw_output)
        log.info(f"Saved log to {log_file}")

    # Parse accuracy from output
    accuracy = parse_accuracy(raw_output)

    return {
        "accuracy": accuracy,
        "raw_output": raw_output,
        "return_code": result.returncode,
        "duration_s": duration,
        "log_file": str(log_file) if log_dir else None,
    }


def run_debug_diagnostics(config_path: str, log_dir: Path, debug_utils_module: str | None = None) -> dict:
    """Run detailed diagnostics for debugging a benchmark config.

    If ``debug_utils_module`` is specified, runs the experiment-specific debug script
    from ``tests/benchmarks/debug_utils/<module>/dbg_<module>.py``. Otherwise runs
    only the shared diagnostics.
    """
    abs_config = REPO_ROOT / config_path
    ts = timestamp()
    diag_log = log_dir / f"diagnostics_{ts}.log"

    if debug_utils_module:
        diag_script = Path(__file__).parent / "debug_utils" / debug_utils_module / f"dbg_{debug_utils_module}.py"
    else:
        diag_script = None

    if diag_script and diag_script.exists():
        cmd = [sys.executable, str(diag_script), "--config", str(abs_config), "--output", str(diag_log)]
    else:
        # Fall back to shared benchmark_utils
        cmd = [
            sys.executable,
            "-c",
            f"from tests.benchmarks.benchmark_utils import run_shared_diagnostics; "
            f"run_shared_diagnostics('{abs_config}', '{diag_log}')",
        ]

    log.info(f"Running diagnostics: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=1800,
    )

    combined = result.stdout + "\n--- STDERR ---\n" + result.stderr
    with open(diag_log, "w") as f:
        f.write(combined)

    return {
        "log_file": str(diag_log),
        "return_code": result.returncode,
        "output": combined,
    }


def _resolve_benchmark(registry: dict, benchmark_spec: str) -> tuple[str, str, dict]:
    """Resolve a benchmark spec (experiment/benchmark_id) to its entry.

    Returns (experiment_name, benchmark_id, entry) tuple.
    """
    if "/" in benchmark_spec:
        experiment_name, benchmark_id = benchmark_spec.split("/", 1)
    else:
        # Try to find the benchmark_id in any experiment
        for exp_name, exp_benchmarks in registry["benchmarks"].items():
            if benchmark_spec in exp_benchmarks:
                return exp_name, benchmark_spec, exp_benchmarks[benchmark_spec]
        raise KeyError(f"Unknown benchmark: {benchmark_spec}")

    if experiment_name not in registry["benchmarks"]:
        raise KeyError(f"Unknown experiment: {experiment_name}")
    if benchmark_id not in registry["benchmarks"][experiment_name]:
        raise KeyError(f"Unknown benchmark: {benchmark_spec}")
    return experiment_name, benchmark_id, registry["benchmarks"][experiment_name][benchmark_id]


def evaluate_result(experiment_name: str, benchmark_id: str, result: dict, entry: dict) -> bool:
    """Compare benchmark result against registry expectation."""
    expected = entry["expected_accuracy"]
    tolerance = entry["tolerance"]
    actual = result["accuracy"]

    if actual is None:
        print(f"  FAIL: Could not parse accuracy from output (return code: {result['return_code']})")
        if result["return_code"] != 0:
            lines = result["raw_output"].strip().split("\n")
            print("  Last 30 lines of output:")
            for line in lines[-30:]:
                print(f"    {line}")
        return False

    if expected is None:
        print(f"  INFO: No expected accuracy set. Got: {actual:.4f}")
        return True

    diff = abs(actual - expected)
    passed = diff <= tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  {status}: accuracy={actual:.4f}  expected={expected:.3f}  diff={diff:.4f}  tol={tolerance}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    parser.add_argument("--benchmark", type=str, help="Benchmark spec: experiment/benchmark_id or just benchmark_id")
    parser.add_argument("--experiment", type=str, help="Run all benchmarks for a specific experiment")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--debug", action="store_true", help="Run with debug diagnostics")
    parser.add_argument("--update-registry", action="store_true", help="Update registry with actual results")
    parser.add_argument(
        "--force-update-registry",
        action="store_true",
        help="Update registry (bypasses clean working tree check)",
    )
    parser.add_argument("--limit-batches", type=int, default=None, help="Limit number of test batches")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # --force-update-registry implies --update-registry
    if args.force_update_registry:
        args.update_registry = True

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    # Validate working tree when updating the registry
    if args.update_registry and not args.force_update_registry:
        if not _check_clean_working_tree():
            log.error(
                "Working tree is not clean. Commit or stash changes before updating the registry, "
                "or use --force-update-registry to bypass this check."
            )
            sys.exit(1)

    registry = load_registry()
    log_dir = Path(args.log_dir) if args.log_dir else DEBUG_LOG_DIR

    # Build list of (experiment_name, benchmark_id, entry) tuples
    benchmark_specs: list[tuple[str, str, dict]] = []

    if args.all:
        for exp_name, exp_benchmarks in registry["benchmarks"].items():
            for bm_id, entry in exp_benchmarks.items():
                benchmark_specs.append((exp_name, bm_id, entry))
    elif args.experiment:
        if args.experiment not in registry["benchmarks"]:
            log.error(f"Unknown experiment: {args.experiment}")
            sys.exit(1)
        for bm_id, entry in registry["benchmarks"][args.experiment].items():
            benchmark_specs.append((args.experiment, bm_id, entry))
    elif args.benchmark:
        benchmark_specs.append(_resolve_benchmark(registry, args.benchmark))
    else:
        parser.error("Specify --benchmark SPEC, --experiment NAME, or --all")
        return

    results = {}
    all_passed = True

    for exp_name, bm_id, entry in benchmark_specs:
        full_id = f"{exp_name}/{bm_id}"
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {full_id}")
        print(f"Config: {entry['config_path']}")
        print(f"Description: {entry['description']}")
        print(f"{'=' * 60}")

        if args.debug:
            print("\n--- Debug Diagnostics ---")
            debug_module = entry.get("debug_utils_module")
            diag = run_debug_diagnostics(entry["config_path"], log_dir, debug_module)
            print(f"  Diagnostics log: {diag['log_file']}")
            if diag["return_code"] != 0:
                print(f"  WARNING: Diagnostics exited with code {diag['return_code']}")

        print("\n--- Benchmark Run ---")
        cli_mode = _detect_cli_mode(entry)
        result = run_cli_benchmark(
            config_path=entry["config_path"],
            limit_batches=args.limit_batches,
            log_dir=log_dir,
            debug=args.debug,
            cli_mode=cli_mode,
        )
        results[full_id] = result

        print(f"  Duration: {result['duration_s']:.1f}s")
        if result["log_file"]:
            print(f"  Log: {result['log_file']}")

        passed = evaluate_result(exp_name, bm_id, result, entry)
        if not passed:
            all_passed = False

        if args.update_registry and result["accuracy"] is not None:
            entry["expected_accuracy"] = round(result["accuracy"], 4)
            entry["last_validated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            entry["commit_sha"] = _get_commit_sha()
            entry["salient_pkg_versions"] = _get_salient_pkg_versions()
            save_registry(registry)
            print(f"  Registry updated: accuracy={result['accuracy']:.4f}  commit={entry['commit_sha']}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for full_id, result in results.items():
        acc = f"{result['accuracy']:.4f}" if result["accuracy"] is not None else "N/A"
        code = result["return_code"]
        print(f"  {full_id}: accuracy={acc}  rc={code}  time={result['duration_s']:.1f}s")

    if not all_passed:
        print("\nSome benchmarks FAILED or could not be evaluated.")
        sys.exit(1)
    else:
        print("\nAll benchmarks PASSED.")


if __name__ == "__main__":
    main()
