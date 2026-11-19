"""An on-demand report on the GPU pipeline over a window of builds: cost, selection and selector misses.

It answers the review questions of selective GPU runs from the builds' own evidence (the phase manifests and reports,
and ``telemetry/build_telemetry.json`` where the build published one) and their Azure timelines:

- GPU-agent minutes per build (the job's interval) and the minutes its regular ``Testing`` tasks took;
- gated (pull request) and scheduled builds per week;
- the minutes the devices were held, from the lease events, which only builds with a telemetry record carry;
- the selected count against the full phase per pull request build, and the share of gated builds selecting each area;
- **selector misses**: a failure in a full run of ``main`` in a test that none of the pull requests merged since the
  preceding green full run selected in its gate;
- the time from a merge to the first full run of ``main`` that contains it.

What the evidence cannot establish is reported as unknown, never as zero. An interval whose merged pull requests
include one without readable gate evidence (no gate, an expired artifact, a commit pushed without a pull request)
cannot clear a failure, so the failure is ``undetermined`` rather than a miss or a catch, and green pull request gates
alone never establish that there were no misses: only a full run's failures are tested.

Usage, from the repository root in a dev environment with ``main`` fetched (``AZURE_DEVOPS_EXT_PAT`` set)::

    python -m tests.gpu_ci.ci_report --since 2026-11-11 --org https://dev.azure.com/<org> --project interpretune \\
        --definition 2 --out report.json --markdown report.md

Exit status: 0 written, 2 refused.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tests.gpu_ci.build_telemetry import TelemetryError, assemble
from tests.gpu_ci.scheduled_run_disposition import Azure, Refused, _parse_time, plan_mode

SCHEMA = "interpretune.gpu-ci.report/1"
#: The regular GPU test tasks of a build; the extended tiers run only in the weekly build and are reported apart.
TESTING_TASK = re.compile(r"^Testing: (?!extended)")
#: The pull request a squash merge on main came from, by the title suffix GitHub writes.
PR_SUFFIX = re.compile(r"\(#(\d+)\)\s*$")
FAILED = frozenset({"failed", "errors"})


@dataclass
class Build:
    """One completed build of the GPU pipeline and what could be read of it."""

    id: int
    reason: str
    result: str | None
    source_branch: str
    source_version: str
    finished: str | None
    pull_request: int | None = None
    timeline: list[dict] | None = None
    #: the extracted ``it-phases`` directory, or ``None`` with ``evidence_unknown`` saying why
    evidence: Path | None = None
    evidence_unknown: str | None = None
    #: the planner's mode from the plan step's log, read only when there is no evidence artifact
    plan_mode: str | None = None

    @classmethod
    def from_azure(cls, doc: dict) -> Build:
        branch = doc.get("sourceBranch", "")
        pr = re.fullmatch(r"refs/pull/(\d+)/merge", branch)
        return cls(
            id=doc["id"],
            reason=doc.get("reason", ""),
            result=doc.get("result"),
            source_branch=branch,
            source_version=doc.get("sourceVersion", ""),
            finished=doc.get("finishTime"),
            pull_request=int(pr[1]) if pr else None,
        )


@dataclass
class Merge:
    """A first-parent commit of main: the squash merge of ``pull_request``, or a commit pushed without one."""

    sha: str
    subject: str
    merged_at: str
    pull_request: int | None = None


@dataclass
class Evidence:
    """A build's evidence as the report reads it: the telemetry record, its selected and failed nodes."""

    telemetry: dict | None
    selected: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    mode: str | None = None
    unknown: str | None = None

    @property
    def readable(self) -> bool:
        """Whether the build's selection is known: from its evidence, or a plan that ran nothing."""
        return self.telemetry is not None or self.mode == "none"


def pr_number(subject: str) -> int | None:
    """The pull request a squash-merge subject names, or ``None`` for a commit pushed without one."""
    found = PR_SUFFIX.search(subject)
    return int(found[1]) if found else None


def _minutes(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return round((_parse_time(end) - _parse_time(start)).total_seconds() / 60, 2)


def task_minutes(timeline: list[dict] | None) -> dict:
    """The job's minutes on the agent and the regular ``Testing`` tasks' minutes, each ``None`` when incomplete.

    A task that started and never finished makes the testing sum unknown rather than short.
    """
    if timeline is None:
        return {"agent": None, "testing": None, "unknown": "no timeline"}
    jobs = [r for r in timeline if r.get("type") == "Job" and r.get("startTime")]
    agent = sum(m for m in (_minutes(j.get("startTime"), j.get("finishTime")) for j in jobs) if m is not None)
    agent_known = bool(jobs) and all(j.get("finishTime") for j in jobs)
    tasks = [
        r
        for r in timeline
        if r.get("type") == "Task" and TESTING_TASK.match(r.get("name", "")) and r.get("result") != "skipped"
    ]
    spans = [_minutes(t.get("startTime"), t.get("finishTime")) for t in tasks]
    known = all(s is not None for s in spans)
    out = {"agent": round(agent, 2) if agent_known else None, "testing": round(sum(spans), 2) if known else None}
    if not known:
        out["unknown"] = f"{sum(s is None for s in spans)} Testing task(s) without both a start and a finish"
    return out


def read_evidence(build: Build) -> Evidence:
    """The telemetry record (published, or assembled from the phase evidence of a build that predates it), the
    nodes the phases selected and the ones that failed; evidence that does not bind is reported, never counted.

    A build without an evidence artifact is known only if its plan ran nothing (``mode=none`` in the plan log).
    """
    evidence_dir = build.evidence
    if evidence_dir is None:
        return Evidence(None, mode="none" if build.plan_mode == "none" else None, unknown=build.evidence_unknown)
    published = evidence_dir / "telemetry" / "build_telemetry.json"
    try:
        telemetry = json.loads(published.read_text(encoding="utf-8")) if published.is_file() else assemble(evidence_dir)
    except TelemetryError as e:
        return Evidence(None, unknown=f"evidence refused: {e}")
    selected: set[str] = set()
    for path in sorted((evidence_dir / "manifests").glob("*.json")):
        selected.update(json.loads(path.read_text(encoding="utf-8"))["selected_node_ids"])
    failed: dict[str, str] = {}
    for path in sorted((evidence_dir / "reports").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        failed.update({n: report["phase"] for n, v in report["nodes"].items() if v["outcome"] in FAILED})
    scopes = {p["scope"] for p in telemetry["phases"].values()}
    plan_mode = (telemetry.get("plan") or {}).get("mode")
    # the plan's word when there is one; otherwise the manifests' scopes, which say what was actually collected
    mode = plan_mode or ("full" if scopes == {"full"} else "selected" if scopes else None)
    return Evidence(telemetry, selected, failed, mode)


def classify_failures(
    failures: dict[str, str], merges: list[Merge], selections: dict[int, Evidence | None]
) -> list[dict]:
    """Each full-run failure against the gates of the pull requests merged since the preceding green full run.

    ``selected``: a merged pull request's gate selected the test (the gate ran it and the selection did its job).
    ``miss``: every merge in the interval has gate evidence and none selected it. ``undetermined``: none of the readable
    gates selected it, but a merge without readable gate evidence might have.
    """
    unreadable = sorted(
        f"#{m.pull_request}" if m.pull_request else m.sha[:12]
        for m in merges
        if m.pull_request is None or (ev := selections.get(m.pull_request)) is None or not ev.readable
    )
    out = []
    for node, phase in sorted(failures.items()):
        by = sorted(
            m.pull_request
            for m in merges
            if m.pull_request is not None
            and (ev := selections.get(m.pull_request)) is not None
            and ev.readable
            and (ev.mode == "full" or node in ev.selected)
        )
        verdict = "selected" if by else ("undetermined" if unreadable else "miss")
        entry = {"node": node, "phase": phase, "verdict": verdict, "selected_by": by}
        if verdict == "undetermined":
            entry["without_gate_evidence"] = unreadable
        out.append(entry)
    return out


def merge_to_signal(
    merges: Iterable[Merge], full_runs: list[Build], contains: Callable[[str, str], bool]
) -> list[dict]:
    """Minutes from each merge to the first full run of main that finished after it and contains it."""
    runs = sorted((r for r in full_runs if r.finished), key=lambda r: _parse_time(r.finished))
    out = []
    for m in merges:
        first = next(
            (
                r
                for r in runs
                if _parse_time(r.finished) >= _parse_time(m.merged_at) and contains(m.sha, r.source_version)
            ),
            None,
        )
        entry = {"sha": m.sha, "pull_request": m.pull_request, "merged_at": m.merged_at}
        if first is None:
            entry.update(build=None, minutes=None, unknown="no full run of main containing it finished in the window")
        else:
            entry.update(build=first.id, result=first.result, minutes=_minutes(m.merged_at, first.finished))
        out.append(entry)
    return out


def build_row(build: Build, evidence: Evidence) -> dict:
    """One build's line in the report: identity, minutes, per-phase counts and lease holds, and what is unknown."""
    minutes = task_minutes(build.timeline)
    unknowns = [f"minutes: {minutes['unknown']}"] if minutes.get("unknown") else []
    row = {
        "build": build.id,
        "reason": build.reason,
        "result": build.result,
        "pull_request": build.pull_request,
        "revision": build.source_version,
        "finished": build.finished,
        "mode": evidence.mode,
        "agent_minutes": minutes["agent"],
        "testing_minutes": minutes["testing"],
        "phases": None,
        "device_held_minutes": None,
        "areas": None,
    }
    if evidence.telemetry is None:
        unknowns.append(evidence.unknown or "no telemetry")
        row["unknowns"] = unknowns
        return row
    t = evidence.telemetry
    row["phases"] = {
        name: {k: p.get(k) for k in ("scope", "selected", "full_phase", "outcomes")} for name, p in t["phases"].items()
    }
    holds = [s.get("held_s") for s in t.get("leases", {}).values() if s.get("acquired")]
    if holds and all(h is not None for h in holds):
        row["device_held_minutes"] = round(sum(holds) / 60, 2)
    elif t.get("phase_intervals"):
        # phase events without a lease acquisition: the job ran unleased, or its lease events were lost
        unknowns.append("phases recorded their intervals but no lease acquisition was recorded")
    selection = (t.get("plan") or {}).get("selection")
    row["areas"] = selection.get("areas") if selection else None
    row["unknowns"] = unknowns + list(t.get("unknowns", []))
    return row


def _week(when: str) -> str:
    year, week, _ = _parse_time(when).isocalendar()
    return f"{year}-W{week:02d}"


def _known_sum(values: list[float | None]) -> dict:
    known = [v for v in values if v is not None]
    return {"sum": round(sum(known), 2), "known": len(known), "unknown": len(values) - len(known)}


def weekly(rows: list[dict]) -> dict:
    """Per ISO week: builds by reason, minutes and device holds (with how many builds could not say), selected
    against full, and the share of gated builds that selected each area."""
    by_week: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["finished"]:
            by_week[_week(row["finished"])].append(row)
    out = {}
    for week, items in sorted(by_week.items()):
        gated = [r for r in items if r["reason"] == "pullRequest"]
        with_areas = [r for r in gated if r["areas"] is not None]
        area_counts = Counter(a for r in with_areas for a in r["areas"])
        selected = full = 0
        fraction_unknown = 0
        for r in gated:
            phases = r["phases"]
            if not phases or any(p["full_phase"] is None for p in phases.values()):
                fraction_unknown += 1
                continue
            selected += sum(p["selected"] for p in phases.values())
            full += sum(p["full_phase"] for p in phases.values())
        out[week] = {
            "builds": dict(Counter(r["reason"] for r in items)),
            "results": dict(Counter(f"{r['reason']}:{r['result']}" for r in items)),
            "agent_minutes": _known_sum([r["agent_minutes"] for r in items]),
            "testing_minutes": _known_sum([r["testing_minutes"] for r in items]),
            "device_held_minutes": _known_sum([r["device_held_minutes"] for r in items]),
            "gated_selected_vs_full": {
                "selected": selected,
                "full_phase": full,
                "builds_counted": len(gated) - fraction_unknown,
                "builds_unknown": fraction_unknown,
            },
            "area_share": {
                "gated_builds_with_plan": len(with_areas),
                "areas": {a: round(n / len(with_areas), 3) for a, n in sorted(area_counts.items())},
            },
        }
    return out


class Sources:
    """The reads the report needs: Azure builds, timelines and evidence artifacts, and main's history from git."""

    def __init__(self, az: Azure, project: str, definition: int, cache: Path, root: Path = Path(".")):
        self.az, self.project, self.definition, self.cache, self.root = az, project, definition, cache, root

    def builds(self, since: str, until: str | None) -> list[Build]:
        query = {"definitions": self.definition, "statusFilter": "completed", "minTime": since, "$top": 500}
        if until:
            query["maxTime"] = until
        return [Build.from_azure(b) for b in self.az.get(self.project, "build/builds", **query)["value"]]

    def full_runs_of_main(self, before: str, top: int = 25) -> list[Build]:
        """Completed builds of main finishing before ``before``, newest first (their mode is read from
        evidence)."""
        query = {
            "definitions": self.definition,
            "branchName": "refs/heads/main",
            "statusFilter": "completed",
            "maxTime": before,
            "queryOrder": "finishTimeDescending",
            "$top": top,
        }
        return [Build.from_azure(b) for b in self.az.get(self.project, "build/builds", **query)["value"]]

    def gate(self, pull_request: int, merged_at: str) -> Build | None:
        """The pull request's last completed gate before it merged."""
        query = {
            "definitions": self.definition,
            "branchName": f"refs/pull/{pull_request}/merge",
            "statusFilter": "completed",
            "maxTime": merged_at,
            "queryOrder": "finishTimeDescending",
            "$top": 1,
        }
        found = self.az.get(self.project, "build/builds", **query)["value"]
        return Build.from_azure(found[0]) if found else None

    def fill(self, build: Build) -> Build:
        """Read the build's timeline and fetch its evidence artifact (cached by build id)."""
        build.timeline = self.az.get(self.project, f"build/builds/{build.id}/timeline").get("records")
        target = self.cache / str(build.id)
        if not (target / "it-phases").is_dir():
            artifacts = sorted(
                (
                    a
                    for a in self.az.get(self.project, f"build/builds/{build.id}/artifacts")["value"]
                    if a["name"].startswith("it-pytest-")
                ),
                key=lambda a: a["name"],
            )
            if not artifacts:
                build.evidence_unknown = "no it-pytest-* artifact (never published, or past retention)"
                build.plan_mode = self._plan_mode(build)
                return build
            # the last attempt's artifact: a rerun publishes under the next attempt number
            self._extract(artifacts[-1]["resource"]["downloadUrl"], artifacts[-1]["name"], target)
        build.evidence = target / "it-phases"
        return build

    def _plan_mode(self, build: Build) -> str | None:
        logs = [
            r["log"]["id"]
            for r in build.timeline or []
            if r.get("type") == "Task" and r.get("name") == "Plan the GPU run" and r.get("log")
        ]
        modes = [plan_mode(self.az.text(self.project, f"build/builds/{build.id}/logs/{log}")) for log in logs]
        return next((m for m in reversed(modes) if m), None)

    def _extract(self, url: str, name: str, target: Path) -> None:
        req = urllib.request.Request(url, headers={"Authorization": self.az.auth})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        target.mkdir(parents=True, exist_ok=True)
        prefix = f"{name}/it-phases/"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                rel = member[len(prefix) :] if member.startswith(prefix) else None
                if not rel or member.endswith("/") or ".." in Path(rel).parts:
                    continue
                out = target / "it-phases" / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(member))

    def _git(self, *args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)
        if proc.returncode:
            raise Refused(f"git {' '.join(args)} failed: {proc.stderr.strip()[-300:]} (is main fetched?)")
        return proc.stdout

    def _merges(self, *args: str) -> list[Merge]:
        out = []
        for line in self._git("log", "--first-parent", "--reverse", "--format=%H%x1f%ct%x1f%s", *args).splitlines():
            sha, epoch, subject = line.split("\x1f", 2)
            when = datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            out.append(Merge(sha, subject, when, pr_number(subject)))
        return out

    def merges(self, base: str, head: str) -> list[Merge]:
        """First-parent commits of ``head`` after ``base``, oldest first."""
        return self._merges(f"{base}..{head}")

    def merges_between(self, since: str, until: str | None, ref: str) -> list[Merge]:
        """First-parent commits of ``ref`` committed in the window, oldest first."""
        return self._merges(f"--since={since}", *([f"--until={until}"] if until else []), ref)

    def contains(self, sha: str, revision: str) -> bool:
        proc = subprocess.run(["git", "merge-base", "--is-ancestor", sha, revision], cwd=self.root, capture_output=True)
        if proc.returncode > 1:
            raise Refused(f"git cannot relate {sha} to {revision} (is main fetched?)")
        return proc.returncode == 0


def full_run_review(src: Sources, run: Build, ev: Evidence, cache: dict[int, Evidence]) -> dict:
    """One full run of main: its failures, the interval since the preceding green full run, and each failure's
    verdict against the gates of the pull requests merged in that interval."""
    entry: dict = {"build": run.id, "result": run.result, "revision": run.source_version, "failures": len(ev.failed)}
    previous = None
    for b in src.full_runs_of_main(before=run.finished or ""):
        if b.id == run.id or b.result != "succeeded":
            continue
        if b.id not in cache:
            cache[b.id] = read_evidence(src.fill(b))
        if cache[b.id].mode == "full":
            previous = b
            break
    if previous is None:
        entry.update(interval=None, unknown="no preceding green full run of main could be read")
        if ev.failed:
            entry["classified"] = [
                {"node": n, "phase": p, "verdict": "undetermined", "selected_by": []}
                for n, p in sorted(ev.failed.items())
            ]
        return entry
    merges = src.merges(previous.source_version, run.source_version)
    gates: dict[int, Evidence | None] = {}
    gate_builds: dict[int, int | None] = {}
    for m in merges:
        if m.pull_request is None:
            continue
        gate = src.gate(m.pull_request, m.merged_at)
        gate_builds[m.pull_request] = gate.id if gate else None
        if gate is None:
            gates[m.pull_request] = None
            continue
        if gate.id not in cache:
            cache[gate.id] = read_evidence(src.fill(gate))
        gates[m.pull_request] = cache[gate.id]
    entry["interval"] = {
        "since_build": previous.id,
        "since_revision": previous.source_version,
        "merges": [
            {"sha": m.sha, "pull_request": m.pull_request, "gate": gate_builds.get(m.pull_request)} for m in merges
        ],
    }
    entry["classified"] = classify_failures(ev.failed, merges, gates)
    return entry


def build_report(src: Sources, since: str, until: str | None, main_ref: str = "origin/main") -> dict:
    """The report over builds completing in [``since``, ``until``)."""
    builds = src.builds(since, until)
    cache: dict[int, Evidence] = {}
    rows = []
    for b in builds:
        cache[b.id] = read_evidence(src.fill(b))
        rows.append(build_row(b, cache[b.id]))
    main_runs = [b for b in builds if b.source_branch == "refs/heads/main"]
    full_runs = [b for b in main_runs if cache[b.id].mode == "full"]
    reviews = [full_run_review(src, run, cache[run.id], cache) for run in full_runs]
    verdicts = Counter(c["verdict"] for r in reviews for c in r.get("classified", []))
    return {
        "schema": SCHEMA,
        "window": {"since": since, "until": until},
        "builds": rows,
        "weekly": weekly(rows),
        "full_runs": reviews,
        "selector_misses": {
            "full_runs": len(reviews),
            "failures": sum(r["failures"] for r in reviews),
            **{v: verdicts.get(v, 0) for v in ("miss", "selected", "undetermined")},
        },
        "merge_to_full_run": merge_to_signal(src.merges_between(since, until, main_ref), full_runs, src.contains),
        "main_builds_without_evidence": sorted(b.id for b in main_runs if cache[b.id].mode is None),
    }


def _cell(value) -> str:
    return "unknown" if value is None else str(value)


def markdown(report: dict) -> str:
    """The report as a readable page; the JSON is the record, this is the summary."""
    w = report["window"]
    lines = [f"# GPU CI report: {w['since']} to {w['until'] or 'now'}", ""]
    lines += [
        "## Per week",
        "",
        "| week | builds | agent min (unknown) | testing min | device held min | gated selected / full (unknown) |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for week, s in report["weekly"].items():
        builds = ", ".join(f"{k} {v}" for k, v in sorted(s["builds"].items()))
        f = s["gated_selected_vs_full"]
        lines.append(
            f"| {week} | {builds} | {s['agent_minutes']['sum']} ({s['agent_minutes']['unknown']}) | "
            f"{s['testing_minutes']['sum']} | {s['device_held_minutes']['sum']} "
            f"({s['device_held_minutes']['unknown']} unknown) | {f['selected']} / {f['full_phase']} "
            f"({f['builds_unknown']}) |"
        )
    m = report["selector_misses"]
    lines += [
        "",
        "## Selector misses",
        "",
        f"{m['full_runs']} full run(s) of main, {m['failures']} failure(s): {m['miss']} miss, "
        f"{m['selected']} selected by a merged gate, {m['undetermined']} undetermined.",
        "",
    ]
    for r in report["full_runs"]:
        interval = r.get("interval")
        since = (
            f"since build {interval['since_build']} ({len(interval['merges'])} merges)" if interval else r["unknown"]
        )
        lines.append(f"- build {r['build']} ({r['result']}, {r['revision'][:12]}): {r['failures']} failure(s), {since}")
        for c in r.get("classified", []):
            lines.append(f"  - `{c['node']}` ({c['phase']}): {c['verdict']} {c.get('selected_by') or ''}".rstrip())
    lines += [
        "",
        "## Merge to first full run",
        "",
        "| merge | pull request | minutes | build |",
        "| --- | --- | ---: | --- |",
    ]
    for e in report["merge_to_full_run"]:
        lines.append(f"| {e['sha'][:12]} | {_cell(e['pull_request'])} | {_cell(e['minutes'])} | {_cell(e['build'])} |")
    lines += [
        "",
        "## Builds",
        "",
        "| build | reason | result | PR | mode | agent min | testing min | held min | unknowns |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in report["builds"]:
        lines.append(
            f"| {r['build']} | {r['reason']} | {r['result']} | {_cell(r['pull_request'])} | {_cell(r['mode'])} | "
            f"{_cell(r['agent_minutes'])} | {_cell(r['testing_minutes'])} | {_cell(r['device_held_minutes'])} | "
            f"{len(r['unknowns'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", required=True, help="the window's start (ISO date or time, UTC)")
    ap.add_argument("--until", default=None, help="the window's end (default: now)")
    ap.add_argument("--org", required=True, help="https://dev.azure.com/<org>")
    ap.add_argument("--project", required=True, help="the pipeline's Azure project")
    ap.add_argument("--definition", type=int, required=True, help="the GPU pipeline's definition id")
    ap.add_argument("--main-ref", default="origin/main", help="the ref main's history is read from (fetch it first)")
    ap.add_argument("--cache", type=Path, default=Path(".gpu_ci_report_cache"), help="where artifacts are extracted")
    ap.add_argument("--out", type=Path, required=True, help="the report's JSON")
    ap.add_argument("--markdown", type=Path, default=None, help="also write the summary as Markdown")
    args = ap.parse_args(argv)
    try:
        az = Azure(args.org, os.environ.get("AZURE_DEVOPS_EXT_PAT"))
        report = build_report(
            Sources(az, args.project, args.definition, args.cache), args.since, args.until, args.main_ref
        )
    except (Refused, urllib.error.URLError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(report))
    m = report["selector_misses"]
    counts = f"{m['miss']} miss, {m['selected']} selected, {m['undetermined']} undetermined"
    print(f"{len(report['builds'])} builds, {m['full_runs']} full runs: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
