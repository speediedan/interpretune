"""CPU tests for this repository's phase-evidence configuration (``[tool.gpu-epsilon.phase-evidence]`` in
``pyproject.toml``), which gpu-epsilon's writer (``python -m gpu_epsilon.phase_evidence``) reads for the GPU job's
phase manifests and reports.

The writer's own rules (collection, per-rank verdicts, classification, gating) are tested in gpu-epsilon, against the
module this repository used to carry. These cases check that this repository's table agrees with what reads it and
what it describes: the GPU job's phase steps, the selection variables ``tests/conftest.py`` reads, the special-test
configuration, the release configuration and the reviewed exclusions.
"""

import json
import re
from pathlib import Path

import pytest

phase_evidence = pytest.importorskip(
    "gpu_epsilon.phase_evidence", reason="gpu-epsilon is not installed (requirements/ci/gpu_epsilon.txt)"
)
special_tests = pytest.importorskip("gpu_epsilon.special_tests")

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
CONFIG = phase_evidence.EvidenceConfig.load(PYPROJECT)
PHASES = {p.name: p for p in CONFIG.phases}
CONFTEST = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
SCRIPTS = ROOT / ".azure-pipelines" / "scripts"


def _session(events, collected, finish=True):
    records = [{"event": "start", "rank": 0}, {"event": "collected", "nodeids": list(collected), "rank": 0}]
    records += [{"event": "outcome", "nodeid": n, "when": "call", "outcome": o, "rank": 0} for n, o in events]
    return records + ([{"event": "finish", "exitstatus": 0, "rank": 0}] if finish else [])


def test_the_phases_are_the_gpu_jobs_phases():
    assert list(PHASES) == ["cuda", "standalone", "profile_ci"]
    assert all(p.release and not p.measured for p in PHASES.values())
    container = (SCRIPTS / "gpu_job_container.sh").read_text(encoding="utf-8")
    (listed,) = re.findall(r"^PHASES=\(([^)]*)\)", container, re.M)
    assert listed.split() == list(PHASES)
    # gpu_phase.sh has one case arm per phase (two share one), and none for anything else
    phase_script = (SCRIPTS / "gpu_phase.sh").read_text(encoding="utf-8")
    arms = {name for arm in re.findall(r"^    ([\w|]+)\)$", phase_script, re.M) for name in arm.split("|")}
    assert arms == set(PHASES)


def test_each_phase_collects_what_its_step_runs():
    runner = special_tests.RunnerConfig.load(PYPROJECT)
    for name, phase in PHASES.items():
        # the phase's step collects both roots, as the runner does
        assert phase.paths == runner.collect_dirs, name
        ((var, value),) = phase.selection_env.items()
        assert var in CONFIG.selection_vars, name
        if "gpu_epsilon.special_tests" in phase.command:
            (mark,) = re.findall(r"--mark_type=(\w+)", phase.command)
            assert mark == name and runner.marks[mark]["env"] == {var: value}, name
    assert CONFIG.selector(PHASES["cuda"]).startswith("IT_RUN_CUDA_TESTS=1; pytest tests src/it_examples/tests")


def test_every_selection_variable_conftest_reads_is_stripped_before_collecting():
    """An inherited selection variable (IT_RUN_STANDALONE_CPU_TESTS=1, say) must not change what a phase collects,
    and the memory class goes too, so a two-stage phase's manifest covers both stages."""
    read = set(re.findall(r'os\.(?:getenv|environ\.get)\("(IT_RUN_\w+)"', CONFTEST))
    assert "IT_RUN_STANDALONE_CPU_TESTS" in read
    assert read | {"IT_GPU_MEM_CLASS", "IT_GPU_SMALL_MAX_GB"} <= set(CONFIG.selection_vars)
    # the selection file narrows a phase in conftest, so it is a scope, never stripped
    assert "IT_GPU_SELECTION_FILE" in CONFTEST and "IT_GPU_SELECTION_FILE" not in CONFIG.selection_vars


def test_a_selection_file_chooses_the_selected_scope(monkeypatch):
    monkeypatch.delenv("IT_GPU_SELECTION_FILE", raising=False)
    assert CONFIG.resolve_scope("cuda") == ("full", None)
    monkeypatch.setenv("IT_GPU_SELECTION_FILE", "/tmp/it_gpu_selection.txt")
    name, scope = CONFIG.resolve_scope("standalone")
    assert name == "selected"
    assert CONFIG.selector(PHASES["standalone"], scope).endswith("; select_file=/tmp/it_gpu_selection.txt")


def test_run_env_records_the_phase_from_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    env = phase_evidence.run_env(CONFIG, "standalone", tmp_path)
    assert env[phase_evidence.PHASE_RECORD_DIR_ENV] == str(tmp_path.resolve() / "records" / "standalone")
    assert env[phase_evidence.PHASE_RECORD_ROOT_ENV] == str(ROOT)
    assert env["PYTEST_ADDOPTS"] == f"-x -p {phase_evidence.PLUGIN}"
    # the pipeline command sets its own selection variables, and no phase is measured, so nothing else is exported
    assert set(env) == {phase_evidence.PHASE_RECORD_DIR_ENV, phase_evidence.PHASE_RECORD_ROOT_ENV, "PYTEST_ADDOPTS"}


def test_the_reports_record_the_versions_the_release_compares():
    release = json.loads((ROOT / "tests" / "gpu_ci" / "release.json").read_text(encoding="utf-8"))
    recorded = {*phase_evidence.BUILTIN_VERSIONS, *CONFIG.versions}
    assert {*release["required_versions"], *release["shared_versions"], *release["shared_builds"]} <= recorded
    assert all(source == f"dist:{key}" for key, source in CONFIG.versions.items())


@pytest.mark.parametrize(
    "classification, scope, job_class, passes",
    [
        ("clean", "full", "ci-release", True),
        ("failing", "selected", "ci-any-vendor", False),
        ("empty", "full", "ci-any-vendor", False),
        ("empty", "selected", "ci-any-vendor", True),
        ("unreviewed-exclusions", "full", "ci-any-vendor", True),
        ("unreviewed-exclusions", "full", "ci-vendor-marked", True),
        ("unreviewed-exclusions", "full", "ci-release", False),
    ],
)
def test_gate(classification, scope, job_class, passes):
    assert phase_evidence.gate(CONFIG, classification, scope, job_class) is passes


def test_cli_report_gates_a_release_leg_on_reviewed_skips(tmp_path, monkeypatch):
    """The report as gpu_job_container.sh runs it, on this repository's configuration: an unreviewed skip passes a
    gate and fails a release leg until tests/gpu_ci/reviewed_exclusions.json-style reasons name it."""
    # device names would initialize CUDA in the test process
    monkeypatch.setattr(phase_evidence, "device_identity", lambda config: {"device_count": 0, "devices": []})
    monkeypatch.chdir(ROOT)
    a, b = "t.py::test_a", "t.py::test_b"
    manifest = {
        "repository": CONFIG.repository,
        "revision": "r",
        "phase": "cuda",
        "scope": "full",
        "selector": CONFIG.selector(PHASES["cuda"]),
        "tree_dirty_paths": [],
        "raw_node_ids": [a, b],
        "selected_node_ids": [a, b],
        "collection_errors": [],
        "collect_exit_codes": {"raw": 0, "selected": 0},
    }
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "cuda.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = tmp_path / "records" / "cuda"
    records.mkdir(parents=True)
    session = _session([(a, "passed"), (b, "skipped")], collected=(a, b))
    (records / "records-rank0-pid1.jsonl").write_text("\n".join(json.dumps(r) for r in session), encoding="utf-8")
    args = ["report", "--phase", "cuda", "--out-dir", str(tmp_path), "--vendor", "nvidia", "--evidence-class"]
    args += ["pipeline"]
    assert phase_evidence.main([*args, "--lease-mode", "tool"]) == 0
    report = json.loads((tmp_path / "reports" / "cuda.json").read_text(encoding="utf-8"))
    assert report["backend"] == "cuda" and report["lease_mode"] == "tool" and "interpretune" in report["versions"]
    assert report["classification"] == "unreviewed-exclusions" and report["coverage"] == {"measured": False}
    assert phase_evidence.main([*args, "--job-class", "ci-release"]) == 1
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps({b: "needs two devices"}), encoding="utf-8")
    assert phase_evidence.main([*args, "--job-class", "ci-release", "--reviewed-exclusions", str(reviewed)]) == 0


def test_reviewed_exclusions_name_existing_tests():
    # every reviewed skip or xfail names a test that still exists (its module, function and parameter id) and gives a
    # reason, so a renamed test cannot leave a stale entry that makes some other skip look reviewed
    path = ROOT / "tests/gpu_ci/reviewed_exclusions.json"
    reviewed = json.loads(path.read_text(encoding="utf-8"))
    for nodeid, reason in reviewed.items():
        assert reason.strip(), nodeid
        module, _, test = nodeid.partition("::")
        name = test.split("::")[-1].partition("[")[0]
        # conformance targets inherit their cases from interpretune.testing.conformance
        sources = [ROOT / module, *(ROOT / "src/interpretune/testing/conformance").glob("*.py")]
        assert any(f"def {name}(" in s.read_text(encoding="utf-8") for s in sources), nodeid
