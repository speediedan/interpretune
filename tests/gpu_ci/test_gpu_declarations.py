"""Guards for the GPU test declarations and the special-test phases (CPU, except the probe's positive control)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import cuda_phase_selects
from tests.gpu_ci.calibrate import collect_gpu_items, recommended_gb
from tests.runif import RunIf

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gpu_items() -> list[dict]:
    return collect_gpu_items()


def test_every_gpu_test_declares_its_memory(gpu_items):
    """A GPU test says how much device memory it needs, so placement is computed rather than found by an OOM."""
    assert gpu_items, "the declaration report listed no GPU tests: the collection hook did not run"
    undeclared = [i["nodeid"] for i in gpu_items if not i.get("min_gpu_mem_gb")]
    assert not undeclared, "GPU tests without RunIf(min_gpu_mem_gb=...):\n  " + "\n  ".join(undeclared)


def test_the_gpu_count_survives_into_the_mark(gpu_items):
    """``min_cuda_gpus`` is read as a number by selection and the lease, so it must not collapse to a flag."""
    counts = {i["min_cuda_gpus"] for i in gpu_items if "min_cuda_gpus" in i}
    assert counts and all(isinstance(c, int) and not isinstance(c, bool) for c in counts), counts


def test_the_declaration_report_honours_k(gpu_items):
    """The calibration driver narrows with ``-k``: the report must list only what the run would execute."""
    narrowed = collect_gpu_items(["-k", "test_memprofiler_remove_hooks"])
    assert [i["nodeid"] for i in narrowed] == [
        "tests/core/test_memprofiler.py::TestClassMemProfiler::test_memprofiler_remove_hooks"
    ], narrowed
    assert len(gpu_items) > len(narrowed)


def test_a_memory_declaration_without_a_gpu_is_refused():
    with pytest.raises(ValueError, match="declares no GPU"):
        RunIf(min_gpu_mem_gb=1.0)


@pytest.mark.parametrize(
    "kw, selected",
    [
        ({"min_cuda_gpus": 1}, True),
        ({"bf16_cuda": True}, True),
        ({"min_cuda_gpus": 1, "benchmark": True}, False),
        ({"min_cuda_gpus": 1, "standalone": True}, False),
        ({"min_cuda_gpus": 1, "profiling_ci": True}, False),
        ({"min_cuda_gpus": 1, "optional": True}, False),
        ({"lightning": True}, False),
    ],
)
def test_cuda_phase_selection(kw, selected):
    assert cuda_phase_selects(kw) is selected


def test_recommended_declaration_rounds_up_with_margin():
    assert recommended_gb(0.0) == 0.5
    assert recommended_gb(1.0) == 1.5  # 1.15 rounds up to the next half
    assert recommended_gb(6.5) == 7.5


def _special_tests(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "GPU_LEASE_CMD"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    # the harness runs `python` from PATH; make it this interpreter rather than whatever the shell finds first
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", "tests/special_tests.sh", "--mark_type=standalone", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@RunIf(skip_windows=True, skip_mac_os=True)
def test_an_empty_special_phase_fails():
    """A phase that collects nothing must not read as a passing phase."""
    proc = _special_tests("--filter_pattern=no_test_matches_this_pattern_xyz")
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "No tests were selected" in proc.stdout


@RunIf(skip_windows=True, skip_mac_os=True)
def test_an_empty_special_phase_passes_when_declared():
    proc = _special_tests("--filter_pattern=no_test_matches_this_pattern_xyz", "--allow-empty")
    assert proc.returncode == 0, proc.stdout[-2000:]


def test_the_special_phases_collect_the_example_tests():
    """Special-tier tests under src/it_examples/tests were never collected by the special-test harness."""
    env = {**os.environ, "IT_RUN_OPTIONAL_TESTS": "1", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
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
            "-k",
            "test_sae_lens_notebooks",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert "test_sae_lens_notebooks" in proc.stdout, proc.stdout[-2000:]
    harness = (ROOT / "tests" / "special_tests.sh").read_text()
    assert 'collect_dir=${collect_dir:-"tests src/it_examples/tests"}' in harness


@RunIf(min_cuda_gpus=1, min_gpu_mem_gb=1.0)
def test_the_probe_reads_back_a_known_allocation(tmp_path):
    """Positive control on the measurement helper: a 512 MiB tensor must read back as at least that much."""
    probe_test = tmp_path / "test_alloc.py"
    probe_test.write_text(
        "import torch\n"
        "def test_alloc():\n"
        "    x = torch.empty(512 * 2**20, dtype=torch.uint8, device='cuda')\n"
        "    torch.cuda.synchronize()\n"
        "    del x\n"
    )
    report = tmp_path / "vram.jsonl"
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(probe_test),
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.gpu_ci.vram_probe",
            f"--vram-report={report}",
            "--rootdir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    row = json.loads(report.read_text().splitlines()[-1])
    assert row["allocator_peak_gb"] >= 0.5, row
    assert row["need_gb"] >= 0.5 + row["context_gb"] - 1e-3, row
