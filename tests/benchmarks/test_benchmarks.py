"""
Benchmark Tests
===============

Pytest-based benchmark tests that validate experiment reproducibility.
These require GPU and significant runtime, so they use the benchmark mark.

Usage:
    # Run all benchmarks
    IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py -v

    # Run a specific experiment's benchmarks
    IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py \
        -k "rte_boolq" -v

    # Run a specific benchmark config
    IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py \
        -k "gemma2_2b_it_l" -v

    # Via special_tests.sh harness
    ./tests/special_tests.sh --mark_type=benchmark
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.runif import RunIf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = Path(__file__).resolve().parent / "benchmark_registry.yaml"


def _load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def _get_benchmark_params():
    """Generate pytest parameters from the nested benchmark registry.

    Yields (experiment_name, benchmark_id, entry) tuples with test id
    formatted as ``experiment_name/benchmark_id``.
    """
    registry = _load_registry()
    params = []
    for experiment_name, experiment_benchmarks in registry["benchmarks"].items():
        for benchmark_id, entry in experiment_benchmarks.items():
            params.append(
                pytest.param(
                    experiment_name,
                    benchmark_id,
                    entry,
                    id=f"{experiment_name}/{benchmark_id}",
                )
            )
    return params


def _parse_accuracy(output: str) -> float | None:
    """Parse accuracy from CLI benchmark output."""
    for pattern in [
        r"['\"]accuracy['\"]:\s*tensor\(([\d.]+)",
        r"accuracy\s+([\d.]+)",
        r"['\"]accuracy['\"]:\s*([\d.]+)",
    ]:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    return None


@RunIf(benchmark=True, min_cuda_gpus=1)
@pytest.mark.parametrize(
    ("experiment_name", "benchmark_id", "benchmark_entry"),
    _get_benchmark_params(),
)
def test_benchmark(experiment_name, benchmark_id, benchmark_entry):
    """Run a benchmark config and verify accuracy against expected."""
    config_path = REPO_ROOT / benchmark_entry["config_path"]
    assert config_path.exists(), f"Config not found: {config_path}"

    expected = benchmark_entry.get("expected_accuracy")
    tolerance = benchmark_entry.get("tolerance", 0.02)

    # Run via CLI subprocess to match production execution path
    cmd = [
        "interpretune",
        "--lightning_cli",
        "test",
        "--config",
        str(config_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=1800,
    )

    assert result.returncode == 0, (
        f"Benchmark {experiment_name}/{benchmark_id} failed with rc={result.returncode}\n"
        f"STDERR:\n{result.stderr[-2000:]}"
    )

    output = result.stdout + result.stderr
    accuracy = _parse_accuracy(output)

    assert accuracy is not None, (
        f"Could not parse accuracy from {experiment_name}/{benchmark_id} output.\n"
        f"Last 50 lines:\n" + "\n".join(output.strip().split("\n")[-50:])
    )

    if expected is not None:
        assert abs(accuracy - expected) <= tolerance, (
            f"Benchmark {experiment_name}/{benchmark_id}: accuracy={accuracy:.4f}, "
            f"expected={expected:.3f}±{tolerance}"
        )
