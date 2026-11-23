"""The on-demand GPU CI report (CPU): its minutes, its failure verdicts and the unknowns it keeps visible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.gpu_ci.ci_report import (
    Build,
    Evidence,
    Merge,
    Sources,
    build_row,
    classify_failures,
    full_run_review,
    markdown,
    merge_to_signal,
    pr_number,
    read_evidence,
    task_minutes,
    weekly,
)

gpu_epsilon_evidence = pytest.importorskip("gpu_epsilon.evidence")

REV = "b" * 40
SELECTOR = "IT_RUN_CUDA_TESTS=1; pytest tests src/it_examples/tests"
NODES = [f"tests/test_x.py::test_{i}" for i in range(6)]


def _task(name, start, end, result="succeeded"):
    return {"type": "Task", "name": name, "startTime": start, "finishTime": end, "result": result}


def _evidence_dir(root: Path, selected, failed=(), scope="selected") -> Path:
    """A build's ``it-phases`` directory as a build that predates the telemetry record published it."""
    manifest = {
        "repository": "interpretune/interpretune",
        "revision": REV,
        "phase": "cuda",
        "scope": scope,
        "selector": SELECTOR + ("" if scope == "full" else " --select-file sel.txt"),
        "raw_node_ids": NODES,
        "selected_node_ids": list(selected),
        "collection_errors": [],
    }
    nodes = {n: {"outcome": "failed" if n in failed else "passed"} for n in selected}
    report = {k: manifest[k] for k in ("repository", "revision", "phase", "scope", "selector")}
    report.update(
        outcomes={"attempted": len(selected), "failed": len(failed)},
        nodes=nodes,
        manifest_sha256=gpu_epsilon_evidence.manifest_digest(manifest),
    )
    for sub, doc in (("manifests", manifest), ("reports", report)):
        (root / sub).mkdir(parents=True, exist_ok=True)
        (root / sub / "cuda.json").write_text(json.dumps(doc))
    return root


def _ev(selected=(), mode="selected", failed=None) -> Evidence:
    return Evidence({"phases": {}}, set(selected), failed or {}, mode)


def test_pr_number_reads_the_squash_suffix_only():
    assert pr_number("Add a component map for Qwen3.5 (#947)") == 947
    assert pr_number("Fix #12 in the docs") is None
    assert pr_number("Bump a pin") is None


def test_task_minutes_counts_regular_testing_tasks_and_refuses_a_partial_sum():
    timeline = [
        {"type": "Job", "startTime": "2026-11-18T10:00:00Z", "finishTime": "2026-11-18T10:40:00Z"},
        _task("Testing: standard gpu cuda-marked (small)", "2026-11-18T10:05:00Z", "2026-11-18T10:15:00Z"),
        _task("Testing: standalone gpu (large)", "2026-11-18T10:15:00Z", "2026-11-18T10:20:30Z"),
        _task("Testing: extended optional", "2026-11-18T10:20:30Z", "2026-11-18T10:30:00Z"),
        _task("Install dependencies", "2026-11-18T10:00:00Z", "2026-11-18T10:05:00Z"),
    ]
    assert task_minutes(timeline) == {"agent": 40.0, "testing": 15.5}
    timeline.append(_task("Testing: CI Profiling (large)", "2026-11-18T10:30:00Z", None, result=None))
    out = task_minutes(timeline)
    assert out["testing"] is None and "1 Testing task" in out["unknown"]
    assert task_minutes(None)["agent"] is None


def test_classify_failures_selected_miss_and_undetermined():
    failures = {NODES[0]: "cuda", NODES[1]: "cuda"}
    merges = [Merge("1" * 40, "A (#10)", "2026-11-18T00:00:00Z", 10), Merge("2" * 40, "B (#11)", "t", 11)]
    gates = {10: _ev([NODES[0]]), 11: _ev(mode="none")}
    verdicts = {c["node"]: c for c in classify_failures(failures, merges, gates)}
    assert verdicts[NODES[0]]["verdict"] == "selected" and verdicts[NODES[0]]["selected_by"] == [10]
    # every merge's gate is readable (one ran nothing) and none selected it
    assert verdicts[NODES[1]]["verdict"] == "miss"

    # a full gate selects everything
    full = classify_failures(failures, merges, {10: _ev([NODES[0]]), 11: _ev(mode="full")})
    assert {c["verdict"] for c in full} == {"selected"}

    # a merge without gate evidence, or a commit pushed without a pull request, cannot clear a failure
    for extra, gate in ((Merge("3" * 40, "C (#12)", "t", 12), None), (Merge("4" * 40, "direct push", "t"), None)):
        selections = {**gates, **({extra.pull_request: gate} if extra.pull_request else {})}
        out = {c["node"]: c for c in classify_failures(failures, [*merges, extra], selections)}
        assert out[NODES[1]]["verdict"] == "undetermined"
        assert out[NODES[1]]["without_gate_evidence"] == ["#12" if extra.pull_request else "4" * 12]
        assert out[NODES[0]]["verdict"] == "selected"


def test_an_unreadable_gate_is_not_a_selection_of_nothing():
    unreadable = Evidence(None, unknown="evidence refused: test")
    out = classify_failures({NODES[1]: "cuda"}, [Merge("1" * 40, "A (#10)", "t", 10)], {10: unreadable})
    assert out[0]["verdict"] == "undetermined"


def test_read_evidence_assembles_a_build_that_predates_the_record(tmp_path):
    d = _evidence_dir(tmp_path / "it-phases", NODES[:3], failed=[NODES[2]])
    ev = read_evidence(Build(1, "pullRequest", "failed", "refs/pull/5/merge", REV, None, evidence=d))
    assert ev.selected == set(NODES[:3]) and ev.failed == {NODES[2]: "cuda"} and ev.mode == "selected"
    assert any("no plan record" in u for u in ev.telemetry["unknowns"])

    # a report that does not describe its manifest is refused, never counted
    report = json.loads((d / "reports" / "cuda.json").read_text())
    report["manifest_sha256"] = "0" * 64
    (d / "reports" / "cuda.json").write_text(json.dumps(report))
    ev = read_evidence(Build(1, "pullRequest", "failed", "", REV, None, evidence=d))
    assert ev.telemetry is None and not ev.readable and ev.unknown.startswith("evidence refused")

    none = read_evidence(Build(2, "pullRequest", "succeeded", "", REV, None, evidence_unknown="gone", plan_mode="none"))
    assert none.readable and none.mode == "none"
    assert not read_evidence(Build(3, "pullRequest", "succeeded", "", REV, None, evidence_unknown="gone")).readable


def test_merge_to_signal_takes_the_first_containing_full_run_after_the_merge():
    runs = [
        Build(20, "schedule", "succeeded", "refs/heads/main", "r20", "2026-11-18T12:00:00Z"),
        Build(19, "schedule", "succeeded", "refs/heads/main", "r19", "2026-11-18T02:00:00Z"),
    ]
    merges = [Merge("m1", "A (#1)", "2026-11-18T01:00:00Z", 1), Merge("m2", "B (#2)", "2026-11-18T13:00:00Z", 2)]
    contains = {("m1", "r19"), ("m1", "r20")}
    out = merge_to_signal(merges, runs, lambda sha, rev: (sha, rev) in contains)
    assert out[0]["build"] == 19 and out[0]["minutes"] == 60.0
    assert out[1]["minutes"] is None and "no full run" in out[1]["unknown"]


def test_weekly_keeps_unknown_minutes_and_denominators_out_of_the_sums():
    def row(build, reason, agent, phases, held=None, areas=None):
        return {
            "build": build,
            "reason": reason,
            "result": "succeeded",
            "finished": "2026-11-18T10:00:00Z",
            "agent_minutes": agent,
            "testing_minutes": agent,
            "device_held_minutes": held,
            "phases": phases,
            "areas": areas,
        }

    rows = [
        row(1, "pullRequest", 10.0, {"cuda": {"selected": 5, "full_phase": 50}}, held=4.0, areas=["sae"]),
        row(2, "pullRequest", None, {"cuda": {"selected": 7, "full_phase": None}}, areas=["sae", "tl"]),
        row(3, "schedule", 30.0, {"cuda": {"selected": 50, "full_phase": 50}}),
    ]
    w = weekly(rows)["2026-W47"]
    assert w["builds"] == {"pullRequest": 2, "schedule": 1}
    assert w["agent_minutes"] == {"sum": 40.0, "known": 2, "unknown": 1}
    assert w["device_held_minutes"] == {"sum": 4.0, "known": 1, "unknown": 2}
    assert w["gated_selected_vs_full"] == {"selected": 5, "full_phase": 50, "builds_counted": 1, "builds_unknown": 1}
    assert w["area_share"] == {"gated_builds_with_plan": 2, "areas": {"sae": 1.0, "tl": 0.5}}


class FakeSources(Sources):
    """Builds, gates and main's history from memory: the review's joins without Azure or git."""

    def __init__(self, main_runs, gates, merges, evidence):
        self.main_runs, self.gate_builds, self.merge_list, self.evidence_dirs = main_runs, gates, merges, evidence

    def full_runs_of_main(self, before, top=25):
        return [b for b in self.main_runs if b.finished < before]

    def gate(self, pull_request, merged_at):
        return self.gate_builds.get(pull_request)

    def fill(self, build):
        build.evidence = self.evidence_dirs.get(build.id)
        build.evidence_unknown = None if build.evidence else "no artifact"
        return build

    def merges(self, base, head):
        return self.merge_list


def test_full_run_review_joins_the_interval_to_its_gates(tmp_path):
    previous = Build(1, "schedule", "succeeded", "refs/heads/main", "r1", "2026-11-17T10:00:00Z")
    gate = Build(5, "pullRequest", "succeeded", "refs/pull/10/merge", "g5", "2026-11-17T12:00:00Z", pull_request=10)
    run = Build(9, "schedule", "failed", "refs/heads/main", "r9", "2026-11-18T10:00:00Z")
    src = FakeSources(
        [previous],
        {10: gate},
        [Merge("m10", "A (#10)", "2026-11-17T13:00:00Z", 10)],
        {
            1: _evidence_dir(tmp_path / "1" / "it-phases", NODES, scope="full"),
            5: _evidence_dir(tmp_path / "5" / "it-phases", NODES[:2]),
        },
    )
    failing = read_evidence(
        Build(
            9,
            "",
            "",
            "",
            REV,
            None,
            evidence=_evidence_dir(tmp_path / "9" / "it-phases", NODES, [NODES[1], NODES[4]], "full"),
        )
    )
    review = full_run_review(src, run, failing, {})
    assert review["interval"]["since_build"] == 1 and review["interval"]["merges"][0]["gate"] == 5
    verdicts = {c["node"]: c["verdict"] for c in review["classified"]}
    assert verdicts == {NODES[1]: "selected", NODES[4]: "miss"}

    # with no green full run before it, nothing can be cleared
    alone = full_run_review(FakeSources([], {}, [], {}), run, failing, {})
    assert alone["interval"] is None and {c["verdict"] for c in alone["classified"]} == {"undetermined"}


def test_build_row_and_markdown_say_unknown_rather_than_zero(tmp_path):
    build = Build(7, "pullRequest", "succeeded", "refs/pull/3/merge", REV, "2026-11-18T10:00:00Z", pull_request=3)
    row = build_row(build, read_evidence(build))
    assert row["phases"] is None and row["agent_minutes"] is None and row["unknowns"]
    report = {
        "window": {"since": "2026-11-17", "until": None},
        "weekly": weekly([row]),
        "selector_misses": {"full_runs": 0, "failures": 0, "miss": 0, "selected": 0, "undetermined": 0},
        "full_runs": [],
        "merge_to_full_run": [{"sha": "m" * 40, "pull_request": 3, "minutes": None, "build": None}],
        "builds": [row],
    }
    page = markdown(report)
    assert "| 7 | pullRequest | succeeded | 3 | unknown | unknown |" in page
    assert "| mmmmmmmmmmmm | 3 | unknown | unknown |" in page
