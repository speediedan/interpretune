"""Two-stage device placement's memory classes (CPU): ``small`` and ``large`` partition a GPU phase."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMALL_MAX_GB = "8"  # the smaller CI device's total memory


def _collect(**env_overrides: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "IT_RUN_CUDA_TESTS": "1", "CUDA_VISIBLE_DEVICES": ""}
    for k in ("IT_GPU_SELECTION_FILE", "IT_GPU_MEM_CLASS", "IT_GPU_SMALL_MAX_GB"):
        env.pop(k, None)
    env.update(env_overrides)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "src/it_examples/tests",
            "-q",
            "--collect-only",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _ids(proc: subprocess.CompletedProcess) -> set[str]:
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return {line.strip() for line in proc.stdout.splitlines() if "::" in line}


def test_the_memory_classes_partition_the_phase(tmp_path):
    report = tmp_path / "decl.json"
    whole = _ids(_collect(IT_GPU_DECLARATION_REPORT=str(report)))
    small = _ids(_collect(IT_GPU_MEM_CLASS="small", IT_GPU_SMALL_MAX_GB=SMALL_MAX_GB))
    large = _ids(_collect(IT_GPU_MEM_CLASS="large", IT_GPU_SMALL_MAX_GB=SMALL_MAX_GB))
    # Both classes must be populated, or the partition proves nothing about the filter.
    assert small and large, (len(small), len(large))
    assert not small & large, sorted(small & large)[:5]
    assert small | large == whole, sorted(whole - (small | large))[:5]
    # bf16 tests go to the large class whatever they declare: the small device may lack bf16, and there they would be
    # skipped rather than run. At least one must exist, or this half checks nothing.
    bf16 = {row["nodeid"] for row in json.loads(report.read_text()) if row.get("bf16_cuda")}
    assert bf16, "no bf16_cuda test in the cuda-marked phase to check the rule against"
    assert not bf16 & small, sorted(bf16 & small)[:5]


def test_a_class_without_its_threshold_is_refused():
    proc = _collect(IT_GPU_MEM_CLASS="small")
    assert proc.returncode != 0 and "needs IT_GPU_SMALL_MAX_GB" in proc.stdout + proc.stderr


def test_an_unknown_class_is_refused():
    proc = _collect(IT_GPU_MEM_CLASS="medium", IT_GPU_SMALL_MAX_GB=SMALL_MAX_GB)
    assert proc.returncode != 0 and "is not a memory class" in proc.stdout + proc.stderr
