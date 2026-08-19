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
    """Parse accuracy from CLI benchmark output.

    Delegates to the shared ``benchmark_utils.parse_accuracy`` — a stale local pattern copy
    previously failed to match Lightning's rich metric table (``accuracy │ 0.808…``), whose
    box-drawing separator the shared pattern handles.
    """
    from tests.benchmarks.benchmark_utils import parse_accuracy

    return parse_accuracy(output)


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

    # Run via CLI subprocess to match production execution path, honoring the registry's
    # cli_mode exactly like run_benchmarks.py does (core-CLI entries compose a non-Lightning
    # InterpretunableModule that the Lightning trainer rejects)
    from tests.benchmarks.run_benchmarks import _detect_cli_mode

    cli_flag = "--lightning_cli" if _detect_cli_mode(benchmark_entry) == "lightning" else "--run_command"
    cmd = [
        "interpretune",
        cli_flag,
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
            f"Benchmark {experiment_name}/{benchmark_id}: accuracy={accuracy:.4f}, expected={expected:.3f}±{tolerance}"
        )


################################################################################
# Accuracy-parsing pins (ungated: pure string handling, no GPU and no benchmark run)
################################################################################
# NOTE [Colorized Metric Tables]: `run_benchmarks.py` scores a run by matching "accuracy" and its value
# in captured output. Lightning renders that value in a rich table which colorizes whenever a terminal is
# signalled, and an ambient FORCE_COLOR does so even into a pipe. The escape runs then land BETWEEN the
# metric name and its value, so the pattern cannot bridge them and the benchmark scores N/A on a run that
# returned 0. That is a silent wrong answer from the instrument, not a loud failure, so it is pinned with
# the exact bytes observed rather than a hand-written approximation.

# Captured verbatim from a gemma3_1b_it_l_ns Lightning run (accuracy 0.75) with FORCE_COLOR=3 ambient.
ANSI_METRIC_TABLE_ROW = (
    "│\x1b[36m \x1b[0m\x1b[36m        accuracy         \x1b[0m\x1b[36m \x1b[0m"
    "│\x1b[35m \x1b[0m\x1b[35m          0.75           \x1b[0m\x1b[35m \x1b[0m│"
)


def test_parse_accuracy_reads_colorized_lightning_table():
    from tests.benchmarks.benchmark_utils import parse_accuracy

    assert parse_accuracy(ANSI_METRIC_TABLE_ROW) == 0.75


def test_parse_accuracy_reads_uncolored_lightning_table():
    """The same row without color must keep working; the fix may not trade one rendering for the other."""
    from tests.benchmarks.benchmark_utils import parse_accuracy, strip_ansi

    assert parse_accuracy(strip_ansi(ANSI_METRIC_TABLE_ROW)) == 0.75


def test_parse_accuracy_reads_core_epoch_end():
    """The core CLI reports through a plain dict rather than a table."""
    from tests.benchmarks.benchmark_utils import parse_accuracy

    assert parse_accuracy("Test epoch end: {'accuracy': 0.7500}") == 0.75


def test_parse_accuracy_returns_none_without_a_metric():
    from tests.benchmarks.benchmark_utils import parse_accuracy

    assert parse_accuracy("Testing ... 2/2 0:00:01") is None


def test_benchmark_subprocess_env_is_color_neutral():
    """The instrument must not inherit the ambient host's color settings.

    Pinned at the source rather than only at the parser: a colorized capture is unreadable for any future consumer of
    the raw log, not just `parse_accuracy`.
    """
    import inspect

    from tests.benchmarks import run_benchmarks

    source = inspect.getsource(run_benchmarks.run_cli_benchmark)
    assert 'env.pop("FORCE_COLOR", None)' in source
    assert 'env["NO_COLOR"] = "1"' in source
