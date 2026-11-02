"""CPU tests for this repository's coverage phases (``[tool.gpu-epsilon.phases]`` in ``pyproject.toml``), which
``scripts/gen_it_coverage.sh`` runs through gpu-epsilon's phase driver.

The driver's own rules are tested in gpu-epsilon. These cases check that this repository's table agrees with what
reads it and what it calls: the special-test marks, the selection variables ``tests/conftest.py`` reads, and the
flags ``gen_it_coverage.sh`` passes.
"""

import re
import shlex

import pytest

phases = pytest.importorskip(
    "gpu_epsilon.phases", reason="gpu-epsilon is not installed (requirements/ci/gpu_epsilon.txt)"
)
special_tests = pytest.importorskip("gpu_epsilon.special_tests")

from tests.gpu_ci import phase_report as pr  # noqa: E402

PYPROJECT = pr.ROOT / "pyproject.toml"
CONFIG = phases.DriverConfig.load(PYPROJECT)
RUNNER = special_tests.RunnerConfig.load(PYPROJECT)
SCRIPT = (pr.ROOT / "scripts" / "gen_it_coverage.sh").read_text()
PHASES = {p.name: p for p in CONFIG.phases}
CUDA_PROBE = ("-c", "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)")


def _with_phases() -> list[str]:
    (names,) = re.findall(r"--with ([\w,]+)", SCRIPT)
    return names.split(",")


def test_the_default_and_full_selections():
    assert [p.name for p in CONFIG.select([], [], [])] == ["main", "cuda", "standalone", "standalone_cpu", "profile_ci"]
    full = [p.name for p in CONFIG.select([], _with_phases(), [])]
    assert full == ["main", "cuda", "standalone", "standalone_cpu", "profile_ci", "profile", "optional", "benchmark"]
    assert all(PHASES[name].optional for name in _with_phases())


def test_coverage_is_erased_first_and_reported_last():
    assert CONFIG.setup == (("-m", "coverage", "erase"),)
    assert CONFIG.report == (("-m", "coverage", "report", "-m"),)
    # nnsight's trace teardown clears sys.settrace, which would stop the default coverage core
    assert CONFIG.env["COVERAGE_CORE"] == "sysmon"


def test_the_main_suite_phases_cover_both_roots():
    for name in ("main", "cuda"):
        run = PHASES[name].run
        assert {"--cov=src/interpretune", "--cov-append", "--cov-report="} <= set(run), name
        assert [a for a in run if not a.startswith("-") and a not in ("faulthandler", "pytest")] == list(
            RUNNER.collect_dirs
        ), name


def test_the_main_suite_phases_partition_on_cuda():
    main, cuda = PHASES["main"], PHASES["cuda"]
    assert main.env == {"CUDA_VISIBLE_DEVICES": ""} and not main.skip_unless
    # every visible device, and only the cuda-marked selection (conftest reads IT_RUN_CUDA_TESTS)
    assert cuda.env == {"IT_RUN_CUDA_TESTS": "1"} and cuda.unset == ("CUDA_VISIBLE_DEVICES",)
    assert cuda.skip_unless == CUDA_PROBE and PHASES["benchmark"].skip_unless == CUDA_PROBE
    assert 'os.getenv("IT_RUN_CUDA_TESTS", "0") == "1"' in (pr.ROOT / "tests" / "conftest.py").read_text()


def test_every_special_test_phase_runs_a_configured_mark():
    marks = {
        p.name: re.findall(r"--mark_type=(\w+)", " ".join(p.special_tests)) for p in CONFIG.phases if p.special_tests
    }
    assert marks == {
        name: [name] for name in ("standalone", "standalone_cpu", "profile_ci", "profile", "optional", "benchmark")
    }
    assert PHASES["standalone_cpu"].env == {"CUDA_VISIBLE_DEVICES": ""}
    for (mark,) in marks.values():
        assert mark in RUNNER.marks and not RUNNER.marks[mark].get("refused")


@pytest.mark.parametrize(
    "flags", ["--reruns=2 --reruns-delay=5", "--reruns=3 --reruns-delay=1 --resource-debug", "--no-reruns"]
)
def test_the_runner_accepts_what_the_script_passes(flags):
    """gen_it_coverage.sh passes its rerun and resource-debug flags to every special-test phase."""
    assert 'special_args+=("--reruns=${reruns_count}" "--reruns-delay=${reruns_delay}")' in SCRIPT
    assert "special_args+=(--no-reruns)" in SCRIPT and "special_args+=(--resource-debug)" in SCRIPT
    for phase in CONFIG.phases:
        if phase.special_tests:
            argv = ["--config", str(PYPROJECT), "--check-args", *phase.special_tests, *shlex.split(flags)]
            assert special_tests.main(argv) == 0, phase.name
