"""CPU tests for the GPU CI phase manifests and reports (``phase_report.py`` and ``phase_plugin.py``)."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.gpu_ci import phase_report as pr
from tests.gpu_ci.phase_plugin import RECORD_DIR_ENV, RECORD_ROOT_ENV, node_outcome

SYNTHETIC_TESTS = """
import os
import pytest

def test_pass():
    pass

def test_fail():
    assert False

@pytest.mark.skip(reason="synthetic skip")
def test_skip():
    pass

@pytest.mark.xfail(reason="synthetic xfail", strict=True)
def test_xfail():
    assert False

@pytest.mark.xfail(reason="synthetic xpass", strict=False)
def test_xpass():
    pass

@pytest.fixture
def broken():
    raise RuntimeError("setup error")

def test_setup_error(broken):
    pass
"""
# Selects on an environment variable, as tests/conftest.py does, so raw and selected collections differ.
SYNTHETIC_CONFTEST = """
import os

def pytest_collection_modifyitems(items):
    if os.getenv("IT_RUN_CUDA_TESTS") == "1":
        items[:] = [i for i in items if not i.name.startswith("test_pass_only_raw")]
"""
# Fails its first attempt and passes the next, as a flaky test retried by ``--reruns`` does.
FLAKY_TEST = """
from pathlib import Path

def test_flaky(tmp_path_factory):
    marker = Path(__file__).with_name("attempted")
    if not marker.exists():
        marker.write_text("1")
        assert False, "first attempt"
"""


def _session(events, finish=True, collected=("t.py::test_a",)):
    records = [{"event": "start", "rank": 0}, {"event": "collected", "nodeids": list(collected), "rank": 0}]
    records += [{"event": "outcome", "nodeid": n, "when": "call", "outcome": o, "rank": 0} for n, o in events]
    if finish:
        records.append({"event": "finish", "exitstatus": 0, "rank": 0})
    return records


def _manifest(selected, raw=None, scope="full"):
    return {
        "repository": pr.REPOSITORY,
        "revision": "r",
        "phase": "standalone",
        "scope": scope,
        "selector": "s",
        "raw_node_ids": list(raw or selected),
        "selected_node_ids": list(selected),
        "collection_errors": [],
    }


def _report(manifest, sessions, **kwargs):
    return pr.build_report(manifest, sessions, vendor="amd", evidence_class="parity-pre-agent", **kwargs)


def _run_recorded(tmp_path, *args, env_update=None):
    record_dir = tmp_path / "records"
    env = {k: v for k, v in os.environ.items() if k not in pr.SELECTION_VARS}
    env.update(env_update or {}, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(pr.ROOT))
    env.pop("PYTEST_ADDOPTS", None)
    env[RECORD_DIR_ENV] = str(record_dir)
    cmd = [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", "-p", pr.PLUGIN]
    subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    return pr.read_records(record_dir)


def test_digest_is_sorted_compact_json():
    manifest = _manifest(["t.py::test_a"])
    reordered = dict(reversed(list(manifest.items())))
    assert pr.manifest_digest(manifest) == pr.manifest_digest(reordered)
    assert len(pr.manifest_digest(manifest)) == 64


@pytest.mark.parametrize(
    "when, outcome, wasxfail, expected",
    [
        ("call", "failed", False, "failed"),
        ("setup", "failed", False, "errors"),
        ("teardown", "failed", False, "errors"),
        ("setup", "skipped", False, "skipped"),
        ("call", "skipped", True, "xfailed"),
        ("call", "passed", False, "passed"),
        ("call", "passed", True, "xpassed"),
        ("setup", "passed", False, None),
        ("teardown", "passed", False, None),
        # an attempt pytest-rerunfailures retries decides nothing: the final attempt does
        ("call", "rerun", False, None),
    ],
)
def test_node_outcome(when, outcome, wasxfail, expected):
    report = SimpleNamespace(
        when=when,
        outcome=outcome,
        failed=outcome == "failed",
        skipped=outcome == "skipped",
        passed=outcome == "passed",
    )
    if wasxfail:
        report.wasxfail = "reason"
    assert node_outcome(report) == expected


def test_phase_table_matches_the_gpu_phases():
    assert list(pr.PHASES) == ["cuda", "standalone", "profile_ci"]
    assert all(p.release for p in pr.PHASES.values())
    script = (pr.ROOT / ".azure-pipelines/scripts/gpu_phase.sh").read_text(encoding="utf-8")
    for name, phase in pr.PHASES.items():
        assert f"{name})" in script or f"|{name})" in script or f"{name}|" in script, name
        ((var, value),) = phase.selection_env.items()
        assert var in pr.SELECTION_VARS
    assert pr.selector_string(pr.PHASES["cuda"]).startswith("IT_RUN_CUDA_TESTS=1; pytest tests src/it_examples/tests")


def test_selector_and_scope_name_a_selection_file(monkeypatch):
    monkeypatch.setenv("IT_GPU_SELECTION_FILE", "/tmp/it_gpu_selection.txt")
    assert pr.selector_string(pr.PHASES["standalone"]).endswith("; select_file=/tmp/it_gpu_selection.txt")


def test_run_env_records_every_pytest_process(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    env = pr.run_env("standalone", tmp_path)
    assert env[RECORD_DIR_ENV] == str(tmp_path.resolve() / "records" / "standalone")
    assert env[RECORD_ROOT_ENV] == str(pr.ROOT)
    assert env["PYTEST_ADDOPTS"] == f"-x -p {pr.PLUGIN}"
    with pytest.raises(KeyError):
        pr.run_env("extended", tmp_path)


def test_collect_run_and_report_end_to_end(tmp_path):
    """Collect a synthetic tree raw and selected through the plugin, run it recorded, and report it."""
    (tmp_path / "test_synthetic.py").write_text(SYNTHETIC_TESTS + "\ndef test_pass_only_raw():\n    pass\n")
    (tmp_path / "conftest.py").write_text(SYNTHETIC_CONFTEST)
    raw, raw_errors, _ = pr._collect(("test_synthetic.py",), {}, tmp_path)
    selected, sel_errors, _ = pr._collect(("test_synthetic.py",), {"IT_RUN_CUDA_TESTS": "1"}, tmp_path)
    assert not raw_errors and not sel_errors
    assert len(raw) == 7 and len(selected) == 6 and "test_synthetic.py::test_pass_only_raw" not in selected
    sessions = _run_recorded(tmp_path, "test_synthetic.py", env_update={"IT_RUN_CUDA_TESTS": "1"})
    report = _report(_manifest(selected, raw), sessions)
    o = report["outcomes"]
    assert (o["passed"], o["failed"], o["errors"], o["skipped"], o["xfailed"], o["xpassed"]) == (1, 1, 1, 1, 1, 1)
    assert o["attempted"] == 6 and o["unexecuted"] == 0
    assert report["collection"] == {"raw": 7, "deselected": 1, "selected": 6, "errors": 0}
    assert report["classification"] == "failing" and report["evidence_class"] == "parity-pre-agent"
    assert report["nodes"]["test_synthetic.py::test_setup_error"]["outcome"] == "errors"


def test_a_session_rooted_elsewhere_is_not_recorded(tmp_path):
    """A test that runs pytest on a tree of its own inherits the phase's recording; its session is not the phase's
    (measured: a cuda-phase test's ``test_alloc.py`` ran unmanifested and failed the report)."""
    (tmp_path / "test_synthetic.py").write_text(SYNTHETIC_TESTS)
    assert _run_recorded(tmp_path, "test_synthetic.py", env_update={RECORD_ROOT_ENV: str(pr.ROOT)}) == []
    assert _run_recorded(tmp_path, "test_synthetic.py", env_update={RECORD_ROOT_ENV: str(tmp_path)})


def test_a_rerun_that_passes_counts_once_as_passed(tmp_path):
    pytest.importorskip("pytest_rerunfailures")
    (tmp_path / "test_flaky.py").write_text(FLAKY_TEST)
    report = _report(
        _manifest(["test_flaky.py::test_flaky"]), _run_recorded(tmp_path, "test_flaky.py", "--reruns", "1")
    )
    assert report["outcomes"]["passed"] == 1 and report["outcomes"]["failed"] == 0
    assert report["classification"] == "clean"


def test_report_counts_sessions_completions_and_drift():
    a, b, c = "t.py::test_a", "t.py::test_b", "t.py::test_c"
    # special_tests.sh runs each node in its own process: one session per node
    per_node = _report(
        _manifest([a, b]), [_session([(a, "passed")], collected=(a,)), _session([(b, "passed")], collected=(b,))]
    )
    assert per_node["classification"] == "clean" and per_node["nodes"][a]["sessions"] == 1
    assert per_node["manifest_sha256"] == pr.manifest_digest(_manifest([a, b]))
    # a failure in any session fails the node
    split = _report(_manifest([a]), [_session([(a, "passed")]), _session([(a, "failed")])])
    assert split["outcomes"]["failed"] == 1 and split["classification"] == "failing"
    # a session that started and never finished fails the node it gave no verdict for
    died = _report(_manifest([a]), [_session([], finish=False)])
    assert died["missing_completions"] == 1 and died["outcomes"]["failed"] == 1 and died["classification"] == "failing"
    # a selected node with no verdict is unexecuted; a node outside the manifest is a collection error
    drift = _report(_manifest([a, b]), [_session([(a, "passed"), (c, "passed")], collected=(a, b, c))])
    assert drift["outcomes"]["unexecuted"] == 1 and drift["nodes"][b]["outcome"] == "unexecuted"
    assert drift["unmanifested_nodes"] == [c] and drift["collection"]["errors"] == 1
    # skips need a reviewed reason, and a phase with nothing passed is never clean
    skips = _report(_manifest([a, b]), [_session([(a, "passed"), (b, "skipped")], collected=(a, b))])
    assert skips["classification"] == "unreviewed-exclusions" and not skips["exclusions_reviewed"]
    reviewed = _report(
        _manifest([a, b]),
        [_session([(a, "passed"), (b, "skipped")], collected=(a, b))],
        reviewed_exclusions={b: "needs two devices"},
    )
    assert reviewed["classification"] == "clean" and reviewed["exclusions"][0]["reason"] == "needs two devices"
    empty = _report(_manifest([b]), [_session([(b, "skipped")], collected=(b,))], reviewed_exclusions={b: "x"})
    assert empty["classification"] == "empty"
    assert _report(_manifest([]), [])["classification"] == "empty"


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
    assert pr.gate(classification, scope, job_class) is passes


def test_report_refuses_unknown_vendor_or_evidence_class():
    with pytest.raises(ValueError, match="vendor"):
        pr.build_report(_manifest([]), [], vendor="intel", evidence_class="pipeline")
    with pytest.raises(ValueError, match="evidence class"):
        pr.build_report(_manifest([]), [], vendor="nvidia", evidence_class="release")


def _write_phase(tmp_path, phase, manifest, sessions):
    (tmp_path / "manifests").mkdir(exist_ok=True)
    (tmp_path / "manifests" / f"{phase}.json").write_text(json.dumps({**manifest, "phase": phase}))
    records = tmp_path / "records" / phase
    records.mkdir(parents=True)
    for i, session in enumerate(sessions):
        (records / f"records-rank0-pid{i}.jsonl").write_text("\n".join(json.dumps(r) for r in session))


def test_cli_report_round_trip_and_gate(tmp_path, monkeypatch):
    # device names would initialize CUDA in the test process; the identity is checked on its own below
    monkeypatch.setattr(pr, "device_identity", lambda: {"device_count": 0, "devices": []})
    a, b = "t.py::test_a", "t.py::test_b"
    _write_phase(tmp_path, "cuda", _manifest([a, b]), [_session([(a, "passed"), (b, "skipped")], collected=(a, b))])
    args = [
        "report",
        "--phase",
        "cuda",
        "--out-dir",
        str(tmp_path),
        "--vendor",
        "nvidia",
        "--evidence-class",
        "pipeline",
    ]
    assert pr.main([*args, "--lease-mode", "tool"]) == 0
    report = json.loads((tmp_path / "reports" / "cuda.json").read_text(encoding="utf-8"))
    assert report["backend"] == "cuda" and report["lease_mode"] == "tool" and "python" in report["versions"]
    assert report["classification"] == "unreviewed-exclusions" and report["coverage"] == {"measured": False}
    # the same report fails a release leg until the skip is reviewed
    assert pr.main([*args, "--job-class", "ci-release"]) == 1
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps({b: "needs two devices"}))
    assert pr.main([*args, "--job-class", "ci-release", "--reviewed-exclusions", str(reviewed)]) == 0


def test_device_identity_records_visible_device_variables(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    ident = pr.device_identity()
    assert ident["ROCR_VISIBLE_DEVICES"] == "GPU-a,GPU-b" and ident["device_count"] == 0 and ident["devices"] == []
    assert set(pr.DEVICE_VARS) <= set(ident)


def test_reviewed_exclusions_name_existing_tests():
    # every reviewed skip or xfail names a test that still exists (its module, function and parameter id) and gives a
    # reason, so a renamed test cannot leave a stale entry that makes some other skip look reviewed
    path = pr.ROOT / "tests/gpu_ci/reviewed_exclusions.json"
    reviewed = json.loads(path.read_text(encoding="utf-8"))
    for nodeid, reason in reviewed.items():
        assert reason.strip(), nodeid
        module, _, test = nodeid.partition("::")
        name = test.split("::")[-1].partition("[")[0]
        # conformance targets inherit their cases from interpretune.testing.conformance
        sources = [pr.ROOT / module, *(pr.ROOT / "src/interpretune/testing/conformance").glob("*.py")]
        assert any(f"def {name}(" in s.read_text(encoding="utf-8") for s in sources), nodeid
