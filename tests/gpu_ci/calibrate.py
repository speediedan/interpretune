"""Measure GPU tests' memory needs, and check the ``RunIf(min_gpu_mem_gb=...)`` declarations against them.

Measure (needs a GPU, and an idle one, so hold the GPU lease for the whole run):

    $GPU_LEASE_CMD -- python tests/gpu_ci/calibrate.py measure --device GPU-<uuid> [-k PATTERN]

Each selected GPU test runs in its OWN process, pinned to one device by UUID, under ``tests.gpu_ci.vram_probe``. In a
shared process a test's peak mostly reflects what earlier tests and fixtures left resident, not what it needs.
Results merge into ``tests/gpu_ci/vram_measurements.yaml``, keyed by node id.

Check (CPU only; compares the committed measurements with the current declarations):

    python tests/gpu_ci/calibrate.py check

fails when a declaration is below its measured need plus margin, and warns when it is more than twice it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = Path(__file__).with_name("vram_measurements.yaml")
MARGIN = 1.15  # on the measured need, which already includes the CUDA context
TIER_ENV = (  # the env flag that lets a test of each special tier run (and selects only that tier)
    ("standalone", "IT_RUN_STANDALONE_TESTS", "1"),
    ("profiling_ci", "IT_RUN_PROFILING_TESTS", "1"),
    ("profiling", "IT_RUN_PROFILING_TESTS", "2"),
    ("optional", "IT_RUN_OPTIONAL_TESTS", "1"),
    ("benchmark", "IT_RUN_BENCHMARK_TESTS", "1"),
)


def recommended_gb(need_gb: float) -> float:
    """The declaration a measured need calls for: need plus margin, rounded up to the next half GiB."""
    return max(0.5, math.ceil(need_gb * MARGIN * 2) / 2)


def collect_gpu_items(extra: list[str] | None = None) -> list[dict]:
    """Every GPU-marked test with its declared needs, from a CPU-only collection of the whole suite."""
    with tempfile.TemporaryDirectory() as d:
        report = Path(d) / "gpu_items.json"
        env = {**os.environ, "IT_GPU_DECLARATION_REPORT": str(report), "CUDA_VISIBLE_DEVICES": ""}
        for _, var, _ in TIER_ENV:
            env.pop(var, None)
        env.pop("IT_RUN_CUDA_TESTS", None)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "src/it_examples/tests",
            "-q",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            *(extra or []),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if not report.exists():
            raise RuntimeError(
                f"collection wrote no declaration report (exit {proc.returncode}):\n{proc.stdout[-2000:]}"
            )
        return json.loads(report.read_text())


def _tier_env(item: dict) -> dict[str, str]:
    for key, var, value in TIER_ENV:
        if item.get(key):
            return {var: value}
    return {"IT_RUN_CUDA_TESTS": "1"}


def measure(args) -> int:
    items = collect_gpu_items(["-k", args.k] if args.k else None)
    data = yaml.safe_load(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
    print(f"measuring {len(items)} GPU tests on {args.device}", flush=True)
    for item in items:
        with tempfile.TemporaryDirectory() as d:
            report = Path(d) / "vram.jsonl"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.device, **_tier_env(item)}
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                item["nodeid"],
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "tests.gpu_ci.vram_probe",
                f"--vram-report={report}",
            ]
            try:
                subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT {item['nodeid']}", flush=True)
                continue
            rows = [json.loads(line) for line in report.read_text().splitlines()] if report.exists() else []
        if not rows:
            print(f"NO-REPORT {item['nodeid']} (did not run: deselected, skipped before setup, or crashed)", flush=True)
            continue
        row = rows[-1]
        data[item["nodeid"]] = {
            k: row[k]
            for k in (
                "outcome",
                "need_gb",
                "allocator_peak_gb",
                "device_peak_gb",
                "context_gb",
                "device",
                "torch",
                "seconds",
            )
        }
        data[item["nodeid"]]["date"] = date.today().isoformat()
        print(
            f"{row['outcome']:7s} need={row['need_gb']:6.2f} declared={row['declared_gb']:5.1f} "
            f"recommend={recommended_gb(row['need_gb']):5.1f}  {item['nodeid']}",
            flush=True,
        )
        MEASUREMENTS.write_text(yaml.safe_dump(dict(sorted(data.items())), sort_keys=False))
    return 0


def check(args) -> int:
    data = yaml.safe_load(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
    items = {i["nodeid"]: i for i in collect_gpu_items()}
    bad = []
    for nodeid, m in sorted(data.items()):
        if nodeid not in items or m.get("outcome") != "passed":
            continue
        declared, need = float(items[nodeid].get("min_gpu_mem_gb") or 0), recommended_gb(m["need_gb"])
        if declared < need:
            bad.append(f"UNDER  {nodeid}: declares {declared:g} GiB, measured need calls for {need:g}")
        elif declared > 2 * need and declared - need > 1:
            print(f"OVER   {nodeid}: declares {declared:g} GiB, measured need calls for {need:g}")
    for line in bad:
        print(line)
    print(f"checked {len(data)} measurements against {len(items)} declarations: {len(bad)} under-declared")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="measure GPU tests, one process each, on one device")
    m.add_argument("--device", required=True, help="the device to pin to (a UUID, as CUDA_VISIBLE_DEVICES takes it)")
    m.add_argument("-k", default=None, help="pytest -k expression narrowing the GPU tests measured")
    m.add_argument("--timeout", type=int, default=1800, help="per-test timeout in seconds")
    sub.add_parser("check", help="compare committed measurements with the current declarations")
    args = ap.parse_args(argv)
    return measure(args) if args.cmd == "measure" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
