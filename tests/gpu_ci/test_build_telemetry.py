"""The per-build GPU telemetry record (CPU): what it joins, what it refuses, and what it reports as unknown."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gpu_ci.build_telemetry import SCHEMA, TelemetryError, assemble, main

gpu_epsilon_evidence = pytest.importorskip("gpu_epsilon.evidence")

ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_SH = ROOT / ".azure-pipelines" / "scripts" / "telemetry.sh"
REV = "a" * 40
SELECTOR = "IT_RUN_CUDA_TESTS=1; pytest tests src/it_examples/tests"
FULL_PHASE = [f"tests/test_x.py::test_{i}" for i in range(10)]
RAW = FULL_PHASE + [f"tests/test_cpu.py::test_{i}" for i in range(90)]


def _manifest(selected, scope="full", selector=SELECTOR, revision=REV, phase="cuda") -> dict:
    return {
        "repository": "speediedan/interpretune",
        "revision": revision,
        "phase": phase,
        "scope": scope,
        "selector": selector,
        "tree_dirty_paths": [],
        "raw_node_ids": RAW,
        "selected_node_ids": list(selected),
        "collection_errors": [],
        "collect_exit_codes": {"raw": 0, "selected": 0},
    }


def _report(manifest: dict, **override) -> dict:
    report = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "phase": manifest["phase"],
        "scope": manifest["scope"],
        "selector": manifest["selector"],
        "vendor": "nvidia",
        "lease_mode": "tool",
        "outcomes": {"attempted": len(manifest["selected_node_ids"]), "passed": len(manifest["selected_node_ids"])},
        "classification": "clean",
        "exclusions_reviewed": True,
        "manifest_sha256": gpu_epsilon_evidence.manifest_digest(manifest),
    }
    report.update(override)
    return report


def _plan(mode="full", revision=REV) -> dict:
    return {
        "schema": "interpretune.gpu-ci.plan/1",
        "identity": {"repository": "speediedan/interpretune", "build_id": "7", "merge_revision": revision},
        "mode": mode,
        "why": "test",
        "extended": False,
        "lease_policy": {"wait_s": 2400, "budget_s": 4800, "on_timeout": "fail"},
        "selection": None,
        "changed_paths": None,
        "changed_unknown": "not computed: test",
    }


EVENTS = [
    {"ts": "t0", "epoch": 100, "kind": "lease", "stage": "large", "action": "wait", "attempt": "1"},
    {
        "ts": "t1",
        "epoch": 160,
        "kind": "lease",
        "stage": "large",
        "action": "acquired",
        "grant": "all",
        "devices": "GPU-a;GPU-b",
        "mode": "tool",
    },
    {"ts": "t2", "epoch": 170, "kind": "phase", "phase": "cuda", "class": "all", "action": "start"},
    {
        "ts": "t3",
        "epoch": 470,
        "kind": "phase",
        "phase": "cuda",
        "class": "all",
        "action": "end",
        "rc": "0",
        "ran": "ran",
    },
    {"ts": "t4", "epoch": 500, "kind": "lease", "stage": "large", "action": "released"},
]


def _write(tmp_path: Path, *, manifest=None, report=None, denominator=None, plan=None, events=EVENTS, raw_lines=()):
    manifest = manifest if manifest is not None else _manifest(FULL_PHASE)
    (tmp_path / "manifests").mkdir(parents=True)
    (tmp_path / "manifests" / f"{manifest['phase']}.json").write_text(json.dumps(manifest))
    if report is not False:
        (tmp_path / "reports").mkdir()
        rep = report if report is not None else _report(manifest)
        (tmp_path / "reports" / f"{manifest['phase']}.json").write_text(json.dumps(rep))
    if denominator is not None:
        (tmp_path / "denominator" / "manifests").mkdir(parents=True)
        (tmp_path / "denominator" / "manifests" / f"{manifest['phase']}.json").write_text(json.dumps(denominator))
    (tmp_path / "telemetry").mkdir()
    (tmp_path / "telemetry" / "plan.json").write_text(json.dumps(plan if plan is not None else _plan()))
    lines = [json.dumps(e) for e in events] + list(raw_lines)
    (tmp_path / "telemetry" / "events.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_a_full_run_joins_counts_intervals_and_leases(tmp_path):
    record = assemble(_write(tmp_path))
    assert record["schema"] == SCHEMA
    cuda = record["phases"]["cuda"]
    assert (cuda["selected"], cuda["full_phase"], cuda["raw_collection"]) == (10, 10, 100)
    assert cuda["outcomes"]["passed"] == 10 and cuda["classification"] == "clean"
    assert record["phase_intervals"]["cuda/all"]["seconds"] == 300
    assert record["phase_seconds_total"] == 300
    large = record["leases"]["large"]
    # queueing and holding are separate quantities
    assert (large["wait_s"], large["held_s"], large["acquired"]["devices"]) == (60, 340, ["GPU-a", "GPU-b"])
    assert record["unknowns"] == []


def test_a_selected_run_takes_its_full_phase_from_the_denominator_collection(tmp_path):
    manifest = _manifest(FULL_PHASE[:3], scope="selected", selector=SELECTOR + "; select_file=/tmp/sel.txt")
    record = assemble(_write(tmp_path, manifest=manifest, denominator=_manifest(FULL_PHASE), plan=_plan("selected")))
    cuda = record["phases"]["cuda"]
    assert (cuda["selected"], cuda["full_phase"]) == (3, 10)
    assert "same revision" in cuda["full_phase_source"]


def test_a_selected_run_without_a_denominator_reports_it_unknown_not_raw(tmp_path):
    manifest = _manifest(FULL_PHASE[:3], scope="selected", selector=SELECTOR + "; select_file=/tmp/sel.txt")
    record = assemble(_write(tmp_path, manifest=manifest, plan=_plan("selected")))
    cuda = record["phases"]["cuda"]
    assert cuda["full_phase"] is None and cuda["raw_collection"] == 100
    assert any("no full-scope collection" in u for u in record["unknowns"])


@pytest.mark.parametrize(
    "case, match",
    [
        # the negative control the audit asked for: the manifest changes, the report's digest does not
        ("selected_changed", "does not describe this manifest"),
        ("report_revision", "report's revision"),
        ("report_selector", "report's selector"),
        ("plan_revision", "the plan says the job ran"),
        ("denominator_revision", "denominator was collected at"),
        ("denominator_selector", "not the manifest's without its selection"),
    ],
)
def test_evidence_that_does_not_bind_is_refused_by_name(tmp_path, case, match):
    manifest = _manifest(FULL_PHASE)
    report = _report(manifest)
    plan, denominator = _plan(), None
    if case == "selected_changed":
        manifest = _manifest(FULL_PHASE[:9])
    elif case == "report_revision":
        report = _report(manifest, revision="b" * 40)
    elif case == "report_selector":
        report = _report(manifest, selector="IT_RUN_STANDALONE_TESTS=1; pytest tests")
    elif case == "plan_revision":
        plan = _plan(revision="c" * 40)
    else:
        manifest = _manifest(FULL_PHASE[:3], scope="selected", selector=SELECTOR + "; select_file=/tmp/sel.txt")
        report = _report(manifest)
        if case == "denominator_revision":
            denominator = _manifest(FULL_PHASE, revision="d" * 40)
        else:
            denominator = _manifest(FULL_PHASE, selector="IT_RUN_PROFILING_TESTS=1; pytest tests")
    with pytest.raises(TelemetryError, match=match):
        assemble(_write(tmp_path, manifest=manifest, report=report, plan=plan, denominator=denominator))


def test_an_interrupted_phase_and_an_unreleased_lease_are_unknown_not_zero(tmp_path):
    events = EVENTS[:3]  # the phase started, nothing after it
    record = assemble(_write(tmp_path, events=events, report=False))
    assert record["phase_intervals"]["cuda/all"]["seconds"] is None
    assert record["phase_seconds_total"] is None
    assert record["leases"]["large"]["held_s"] is None
    unknowns = " | ".join(record["unknowns"])
    assert "never recorded its end" in unknowns and "no release was recorded" in unknowns and "no report" in unknowns


def test_devices_that_could_not_be_enumerated_are_named(tmp_path):
    events = [dict(e) for e in EVENTS]
    events[1]["devices"] = "unknown"
    record = assemble(_write(tmp_path, events=events))
    assert record["leases"]["large"]["acquired"]["devices"] is None
    assert any("could not be enumerated" in u for u in record["unknowns"])


def test_a_torn_event_line_is_counted_not_dropped_silently(tmp_path):
    record = assemble(_write(tmp_path, raw_lines=['{"ts": "t5", "epo']))
    assert any("1 event line(s) could not be read" in u for u in record["unknowns"])


def test_an_unknown_plan_schema_is_refused(tmp_path):
    plan = _plan()
    plan["schema"] = "interpretune.gpu-ci.plan/2"
    with pytest.raises(TelemetryError, match="this reader knows"):
        assemble(_write(tmp_path, plan=plan))


def test_the_cli_writes_the_record_and_exits_1_on_a_refusal(tmp_path, capsys):
    evidence = _write(tmp_path / "ok")
    assert main(["assemble", "--dir", str(evidence)]) == 0
    assert json.loads((evidence / "telemetry" / "build_telemetry.json").read_text())["schema"] == SCHEMA
    bad = _write(tmp_path / "bad", plan=_plan(revision="c" * 40))
    assert main(["assemble", "--dir", str(bad)]) == 1
    assert "build telemetry refused" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "linux", reason="the GPU pipeline's steps run in a Linux container")
def test_the_event_helper_writes_json_and_drops_a_value_it_cannot_quote(tmp_path):
    script = f'source "{TELEMETRY_SH}"; telemetry_event kind=lease stage=small bad=\'x"y\' action=wait'
    env = {**os.environ, "IT_TELEMETRY_DIR": str(tmp_path)}
    subprocess.run(["bash", "-c", script], env=env, check=True)
    (event,) = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {k: event[k] for k in ("kind", "stage", "action")} == {"kind": "lease", "stage": "small", "action": "wait"}
    assert "bad" not in event and isinstance(event["epoch"], int)


@pytest.mark.skipif(sys.platform != "linux", reason="the GPU pipeline's steps run in a Linux container")
def test_the_event_helper_writes_nothing_without_a_destination(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "IT_TELEMETRY_DIR"}
    script = f'source "{TELEMETRY_SH}"; [[ -d /evidence ]] && exit 3; telemetry_event kind=lease; echo rc=$?'
    proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, cwd=tmp_path)
    if proc.returncode == 3:
        pytest.skip("this host has an /evidence directory")
    assert proc.stdout.strip() == "rc=0" and not list(tmp_path.iterdir())
