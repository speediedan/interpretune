"""Contract tests for the release eligibility checker shared with finetuning-scheduler.

The checker is ``gpu_epsilon.release_eligibility`` (speediedan/gpu-epsilon, pinned in
``requirements/ci/gpu_epsilon.txt``), configured for this repository by ``tests/gpu_ci/release.json``. Its own cases
live with it; these check that the configuration describes the evidence ``gpu_epsilon.phase_evidence`` writes under
this repository's ``[tool.gpu-epsilon.phase-evidence]``, and that the checker accepts that evidence the way
``.azure-pipelines/gpu-release.yml`` invokes it.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

eligibility = pytest.importorskip(
    "gpu_epsilon.release_eligibility", reason="gpu-epsilon is not installed (requirements/ci/gpu_epsilon.txt)"
)

phase_evidence = pytest.importorskip("gpu_epsilon.phase_evidence")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tests" / "gpu_ci" / "release.json"
EVIDENCE = phase_evidence.EvidenceConfig.load(ROOT / "pyproject.toml")
REVISION = "0" * 40
IMAGES = {"nvidia": "it:cuda@sha256:" + "a" * 64, "amd": "it:rocm@sha256:" + "b" * 64}
TORCH = {"nvidia": "2.15.0+cu132", "amd": "2.15.0+rocm10.0"}
NODES = [f"tests/test_x.py::test_{i}" for i in range(4)]
SUCCEEDED = {"nvidia": "Succeeded", "amd": "Succeeded"}


@pytest.fixture(scope="module")
def config():
    return eligibility.ReleaseConfig.load(CONFIG_PATH)


def _manifest(phase):
    return {
        "repository": EVIDENCE.repository,
        "revision": REVISION,
        "phase": phase,
        "scope": "full",
        "selector": f"selector for {phase}",
        "tree_dirty_paths": [],
        "raw_node_ids": NODES + ["tests/test_x.py::test_deselected"],
        "selected_node_ids": list(NODES),
        "collection_errors": [],
        "collect_exit_codes": {"raw": 0, "selected": 0},
    }


def _session(finished=True, rank=0, outcome="passed"):
    records = [{"event": "start", "rank": rank}, {"event": "collected", "nodeids": list(NODES), "rank": rank}]
    records += [{"event": "outcome", "nodeid": n, "when": "call", "outcome": outcome, "rank": rank} for n in NODES]
    return records + ([{"event": "finish", "exitstatus": 0, "rank": rank}] if finished else [])


def _leg(config, vendor, sessions=None):
    versions = {
        "python": "3.13.16",
        "torch": TORCH[vendor],
        "torch_git_version": "c" * 40,
        "interpretune": "0.1.0.dev0",
        "lightning": "2.6.1",
        "finetuning-scheduler": "2.15.0",
        "transformer-lens": "4.0.0",
        "nnsight": "0.5.0",
        "sae-lens": "6.0.0",
    }
    leg = {}
    for phase in config.phases:
        manifest = _manifest(phase)
        report = phase_evidence.build_report(
            EVIDENCE,
            manifest,
            sessions or [_session()],
            vendor=vendor,
            evidence_class="pipeline",
            lease_mode="tool",
            image=IMAGES[vendor],
            versions=versions,
        )
        leg[phase] = {"report": report, "manifest": manifest}
    return leg


def test_the_config_describes_the_phase_reports(config):
    assert config.repository == EVIDENCE.repository
    assert config.vendors == EVIDENCE.vendors
    assert config.phases == tuple(phase.name for phase in EVIDENCE.phases if phase.release)
    # the checker fails a leg on exactly the counts the writer fails a phase on
    assert tuple(config.failure_counts) == phase_evidence.FAILURE_COUNTS
    # every version the checker compares is one the writer records
    recorded = {*phase_evidence.BUILTIN_VERSIONS, *EVIDENCE.versions}
    assert {*config.required_versions, *config.shared_versions, *config.shared_builds} <= recorded
    # the release legs publish `<artifact_prefix>-<vendor>-<attempt>` holding the phase evidence directory
    job = (ROOT / ".azure-pipelines" / "templates" / "gpu-job.yml").read_text()
    assert f"artifact: {config.artifact_prefix}-${{{{ parameters.vendor }}}}-$(System.JobAttempt)" in job
    container = (ROOT / ".azure-pipelines" / "scripts" / "gpu_job_container.sh").read_text()
    assert f"PHASES_OUT=/evidence/{config.phases_dir}" in container


def test_two_clean_legs_are_eligible(config):
    legs = {vendor: _leg(config, vendor) for vendor in config.vendors}
    verdict = eligibility.check_release(config, legs, SUCCEEDED, REVISION)
    assert verdict["eligible"] and verdict["repository"] == EVIDENCE.repository
    assert {v: e["image"] for v, e in verdict["evidence"].items()} == IMAGES


def test_a_missing_completion_is_ineligible(config):
    """A phase session that started and never finished fails the release, by this repository's own failure
    count."""
    legs = {vendor: _leg(config, vendor) for vendor in config.vendors}
    legs["amd"] = _leg(config, "amd", sessions=[_session(finished=False)])
    assert all(docs["report"]["missing_rank_completions"] for docs in legs["amd"].values())
    with pytest.raises(eligibility.Ineligible, match="amd"):
        eligibility.check_release(config, legs, SUCCEEDED, REVISION)


def test_a_rank_failure_is_ineligible(config):
    """A non-launching rank that failed fails the release even when rank 0 passed every node."""
    legs = {vendor: _leg(config, vendor) for vendor in config.vendors}
    legs["nvidia"] = _leg(config, "nvidia", sessions=[_session(), _session(rank=1, outcome="failed")])
    assert all(docs["report"]["rank_failures"] == len(NODES) for docs in legs["nvidia"].values())
    with pytest.raises(eligibility.Ineligible, match="nvidia"):
        eligibility.check_release(config, legs, SUCCEEDED, REVISION)


def test_the_release_job_command_writes_a_verdict(config, tmp_path):
    """The checker as the release job runs it: a module of the pinned package, on this config and the downloaded
    artifacts, taking each vendor's latest attempt."""
    for vendor in config.vendors:
        for attempt, leg in ((1, {}), (2, _leg(config, vendor))):
            phases = tmp_path / f"{config.artifact_prefix}-{vendor}-{attempt}" / config.phases_dir
            for kind in ("reports", "manifests"):
                (phases / kind).mkdir(parents=True)
            for phase, docs in leg.items():
                (phases / "reports" / f"{phase}.json").write_text(json.dumps(docs["report"]))
                (phases / "manifests" / f"{phase}.json").write_text(json.dumps(docs["manifest"]))
    out = tmp_path / "it-release-eligibility.json"
    args = ["--config", str(CONFIG_PATH), "--artifacts", str(tmp_path), "--revision", REVISION, "--output", str(out)]
    args += [a for vendor in config.vendors for a in (f"--{vendor}-result", "Succeeded")]
    run = subprocess.run(
        [sys.executable, "-m", "gpu_epsilon.release_eligibility", *args], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr
    assert "ELIGIBLE" in run.stdout and json.loads(out.read_text())["eligible"]
