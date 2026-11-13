"""CPU tests for this repository's special-test configuration (``[tool.gpu-epsilon.special-tests]`` in
``pyproject.toml``), which gpu-epsilon's shared runner (``python -m gpu_epsilon.special_tests``) reads.

The runner's own rules are tested in gpu-epsilon. These cases check that this repository's table agrees with what reads
it: the selection variables ``tests/conftest.py`` selects on, the GPU phases that run each mark, and the options the
coverage drivers pass.
"""

import re
from pathlib import Path

import pytest

special_tests = pytest.importorskip(
    "gpu_epsilon.special_tests", reason="gpu-epsilon is not installed (requirements/ci/gpu_epsilon.txt)"
)

phase_evidence = pytest.importorskip("gpu_epsilon.phase_evidence")

ROOT = Path(__file__).resolve().parents[2]
CONFIG = special_tests.RunnerConfig.load(ROOT / "pyproject.toml")
GPU_PHASES = phase_evidence.EvidenceConfig.load(ROOT / "pyproject.toml").phases
CONFTEST = (ROOT / "tests" / "conftest.py").read_text()


def _run(*argv: str):
    return special_tests.Run(CONFIG, special_tests._parser(CONFIG).parse_args(list(argv)))


def test_every_mark_resolves_to_a_conftest_selection():
    assert CONFIG.default_mark == "standalone"
    assert set(CONFIG.marks) == {
        "standalone",
        "standalone_cpu",
        "profile",
        "profile_ci",
        "optional",
        "benchmark",
        "exp_patch",
    }
    for mark, spec in CONFIG.marks.items():
        if spec.get("refused"):
            continue
        (var, value), *rest = spec["env"].items()
        assert not rest, f"{mark}: one selection variable per mark"
        # conftest compares with `== "<value>"` or tests membership in a tuple of values
        assert re.search(rf'os\.getenv\("{var}", "0"\) (== "{value}"|in \([^)]*"{value}")', CONFTEST), mark


def test_exp_patch_is_refused():
    assert "no experimental-patch tests" in CONFIG.marks["exp_patch"]["refused"]


def test_every_selection_variable_conftest_reads_is_cleared():
    """An inherited selection variable (IT_RUN_CUDA_TESTS=1, say) must not change what a mark selects."""
    read = set(re.findall(r'os\.getenv\("(IT_RUN_\w+)"', CONFTEST))
    assert read and read <= CONFIG.selection_vars()


def test_the_gpu_phases_run_configured_marks():
    for phase in GPU_PHASES:
        if "gpu_epsilon.special_tests" in phase.command:
            (mark,) = re.findall(r"--mark_type=(\w+)", phase.command)
            assert mark in CONFIG.marks and not CONFIG.marks[mark].get("refused")


def test_both_roots_are_collected_under_package_coverage():
    assert CONFIG.collect_dirs == ("tests", "src/it_examples/tests")
    assert {"--cov=src/interpretune", "--cov-append"} <= set(CONFIG.exec)
    (override,) = CONFIG.exec_overrides
    assert override["env"] == "COVERAGE_ANALYSIS_CONFIG_FILE" and "--cov-config={value}" in override["exec"]


def test_rerun_options_keep_their_defaults(monkeypatch):
    """The coverage drivers pass --reruns=N, --reruns-delay=N, --no-reruns and --resource-debug through."""
    monkeypatch.delenv("COVERAGE_ANALYSIS_CONFIG_FILE", raising=False)
    assert _run().option_args() == ["--reruns", "2", "--reruns-delay", "5"]
    assert _run("--reruns=3", "--reruns-delay=1").option_args() == ["--reruns", "3", "--reruns-delay", "1"]
    assert _run("--no-reruns").option_args() == []
    assert _run("--resource-debug").environment()["IT_RESOURCE_DEBUG"] == "1"


def test_the_experiments_default_to_off(monkeypatch):
    monkeypatch.chdir(ROOT)
    names = (ROOT / CONFIG.experiments_list).read_text().split()
    run = _run("--mark_type=standalone")
    assert run.experiment_flags() == dict.fromkeys(names, False)
    env = run.environment()
    assert env["IT_RUN_STANDALONE_TESTS"] == "1" and not set(names) & set(env)


def test_the_hosted_linux_job_runs_the_device_free_standalone_tests():
    """``standalone_cpu`` tests left the GPU standalone phase (#731), so the hosted Linux job must run them."""
    workflow = (ROOT / ".github" / "workflows" / "ci_test-full.yml").read_text()
    step = workflow[workflow.index("Run the device-free standalone tests") :]
    step = step[: step.index("- name:")]
    assert "runner.os == 'Linux'" in step and 'CUDA_VISIBLE_DEVICES: ""' in step
    assert "python -m gpu_epsilon.special_tests --config pyproject.toml --mark_type=standalone_cpu" in step
    # and no GPU phase selects them
    assert all("IT_RUN_STANDALONE_CPU_TESTS" not in phase.selection_env for phase in GPU_PHASES)
    assert all("standalone_cpu" not in phase.command for phase in GPU_PHASES)
