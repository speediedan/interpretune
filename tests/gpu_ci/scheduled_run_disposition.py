"""Decide what to do with a queued scheduled GPU build of main: release it, reject it, or hold it.

Main is not built on merge, so a scheduled build is the automatic way main gets a full GPU run, and it has to meet the
same two tests a manual run of main does: there is a reason to run, and the GPU agents are otherwise idle. The verdict:

- ``REJECT covered``: a green **full** run already ran this exact tree (same git tree object). "Full" is the planner's
  own decision for that run, read from its plan step (``GPU run plan: mode=full``), so a selected gate never counts:
  catching what selection misses is half of what the nightly is for. The weekly run is never rejected this way, since
  its extended tiers run in no other build.
- ``RELEASE``: the tree is uncovered and the host is quiet (nothing running, nothing approved and waiting, no
  releasable pull request gate, no lease held or waited on).
- ``HOLD``: uncovered but the host is busy, named; ask again later.
- ``REJECT stale``: still not releasable once the build has waited ``APPROVAL_CAP``, naming what held it, so a
  ``batch: true`` schedule is never blocked behind an old build. The next scheduled run is a fresh attempt.

It decides and prints; releasing or rejecting the approval stays with whoever operates the pool. Anything it cannot
establish is refused by name rather than guessed: a build that is not a queued scheduled build, a missing credential,
no lease state (pass ``--no-lease`` to say there is none), a tree that cannot be read.

**In the job** (``--in-job``, the first GPU step of a scheduled build): scheduled builds of main are not held for
approval, so the coverage test runs inside the job instead. A nightly run whose tree a green full run already covered
cancels itself, through the build REST API with the job's access token, as a run whose lease wait times out does. The
idle test is that lease wait, capped at two hours by the planner. The weekly run and every other build reason always
run. This direction never fails closed: anything the step cannot establish (the API, the tree, the cancel) lets the
run go ahead in full, because an extra full run costs a slot and a skipped one could hide a regression.

Usage, from the repository root in a dev environment (the module itself uses only the standard library)::

    python -m tests.gpu_ci.scheduled_run_disposition --build 1704 \\
        --org https://dev.azure.com/<org> --project interpretune --pool-project <another project on the pool> \\
        --lease-status-cmd "$GPU_LEASE_CMD --status"
    python -m tests.gpu_ci.scheduled_run_disposition --in-job   # in a pipeline step, from its predefined variables

Exit status: 0 release, 1 reject, 3 hold, 2 refused (nothing decided). ``--in-job`` always exits 0.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from tests.gpu_ci.plan_gpu_run import WEEKLY_SCHEDULE

#: How long a scheduled build may wait to be released before it is rejected as stale (the maintainer's two-hour cap).
APPROVAL_CAP = timedelta(hours=2)
#: The planner's own line in a run's plan step; only a resolved run prints a mode after ``mode=``.
PLAN_LINE = re.compile(r"GPU run plan: mode=(full|selected|none)\b")
#: A held lease, or a ticket waiting for one, in a lease tool's ``--status`` output.
LEASE_HELD = re.compile(r"^(\S+)\s+HELD\s+(.*)$")
LEASE_TICKET = re.compile(r"^\s*ticket\s+\S+\s+(\S+)")
#: Check runs that belong to the GPU pipeline itself, not to the hosted CPU matrix a gate's release waits for.
GPU_CHECK_APPS = frozenset({"azure-pipelines"})
GREEN_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})
RELEASE, REJECT, HOLD = "RELEASE", "REJECT", "HOLD"
EXIT = {RELEASE: 0, REJECT: 1, HOLD: 3}


class Refused(RuntimeError):
    """The inputs do not let the disposition be decided."""


@dataclass(frozen=True)
class FullRun:
    """A green run of the GPU pipeline that the planner ran in full, and the tree it ran (``None`` if
    unreadable)."""

    build_id: int
    sha: str
    tree: str | None


@dataclass
class HostState:
    """What would make releasing now cost someone else: each list names its builds or holders."""

    running: list[str] = field(default_factory=list)
    approved_waiting: list[str] = field(default_factory=list)
    releasable_gates: list[str] = field(default_factory=list)
    lease: list[str] = field(default_factory=list)

    def blockers(self) -> list[str]:
        named = (
            ("running", self.running),
            ("approved and waiting", self.approved_waiting),
            ("releasable PR gate", self.releasable_gates),
            ("lease", self.lease),
        )
        return [f"{what}: {', '.join(items)}" for what, items in named if items]


@dataclass(frozen=True)
class Verdict:
    action: str
    reason: str

    def line(self, build_id: int) -> str:
        return f"{self.action} scheduled build {build_id}: {self.reason}"


def dispose(*, tree: str, weekly: bool, full_runs: Iterable[FullRun], host: HostState, waited: timedelta) -> Verdict:
    """The verdict for a queued scheduled build whose commit has ``tree``, after ``waited`` in the queue."""
    if not weekly and (run := covering_run(tree, full_runs)) is not None:
        return Verdict(REJECT, f"covered by build {run.build_id} ({run.sha}), a green full run of tree {tree}")
    blockers = host.blockers()
    why = "weekly extended run" if weekly else f"tree {tree} has no green full run"
    if not blockers:
        return Verdict(RELEASE, f"{why}; host quiet")
    if waited >= APPROVAL_CAP:
        return Verdict(REJECT, f"stale: not releasable after {_minutes(waited)} min; held by {'; '.join(blockers)}")
    return Verdict(HOLD, f"{why}; {'; '.join(blockers)} (waited {_minutes(waited)} min of {_minutes(APPROVAL_CAP)})")


def plan_mode(log_text: str) -> str | None:
    """The mode the planner chose in a run's plan step log, or ``None`` when the log carries no plan line."""
    found = PLAN_LINE.findall(log_text)
    return found[-1] if found else None


def lease_blockers(status_text: str) -> list[str]:
    """Held leases and waiting tickets in a lease tool's ``--status`` output (free leases name nothing)."""
    out = []
    for line in status_text.splitlines():
        held = LEASE_HELD.match(line.strip())
        if held:
            out.append(f"{held[1]} held ({held[2].strip()})")
            continue
        ticket = LEASE_TICKET.match(line)
        if ticket:
            out.append(f"{ticket[1]} waiting")
    return out


def hosted_checks_green(check_runs: list[dict]) -> bool:
    """Whether a pull request head's hosted checks are all finished and green; no hosted check is not green."""
    hosted = [c for c in check_runs if (c.get("app") or {}).get("slug") not in GPU_CHECK_APPS]
    return bool(hosted) and all(
        c.get("status") == "completed" and c.get("conclusion") in GREEN_CONCLUSIONS for c in hosted
    )


def _minutes(delta: timedelta) -> int:
    return int(delta.total_seconds() // 60)


def _parse_time(value: str) -> datetime:
    # Azure returns up to 7 fractional digits; datetime accepts 6
    base, _, rest = value.rstrip("Z").partition(".")
    frac = re.match(r"\d*", rest)[0][:6]
    return datetime.fromisoformat(f"{base}.{frac or '0'}").replace(tzinfo=timezone.utc)


class Azure:
    """The few Azure DevOps REST reads the disposition needs, with the PAT from ``AZURE_DEVOPS_EXT_PAT``."""

    def __init__(self, org: str, pat: str | None = None, *, bearer: str | None = None):
        if bearer:
            self.auth = f"Bearer {bearer}"
        elif pat:
            self.auth = "Basic " + base64.b64encode(f":{pat}".encode()).decode()
        else:
            raise Refused("AZURE_DEVOPS_EXT_PAT is not set: the build and approval state cannot be read")
        self.org = org.rstrip("/")

    def get(self, project: str, path: str, **query) -> dict:
        query.setdefault("api-version", "7.1")
        url = f"{self.org}/{urllib.parse.quote(project)}/_apis/{path}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url, headers={"Authorization": self.auth, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
        try:
            return json.loads(body)
        except ValueError as exc:
            # a rejected credential can come back as a sign-in page, not an error: never read it as an empty list
            raise Refused(f"Azure answered {path} with something that is not JSON (is the PAT valid?)") from exc

    def patch(self, project: str, path: str, body: dict) -> None:
        url = f"{self.org}/{urllib.parse.quote(project)}/_apis/{path}?api-version=7.1"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            method="PATCH",
            headers={"Authorization": self.auth, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30):
            pass

    def text(self, project: str, path: str) -> str:
        url = f"{self.org}/{urllib.parse.quote(project)}/_apis/{path}?api-version=7.1"
        req = urllib.request.Request(url, headers={"Authorization": self.auth, "Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode(errors="replace")


def github(path: str) -> dict | list:
    """A public GitHub API read (``GITHUB_TOKEN`` used when set, for the rate limit)."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "interpretune-gpu-ci"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    req = urllib.request.Request(f"https://api.github.com/{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def commit_tree(repo: str, sha: str) -> str | None:
    """The tree of ``sha``, or ``None`` when GitHub no longer serves the commit (an old merge ref, say)."""
    try:
        return github(f"repos/{repo}/git/commits/{sha}")["tree"]["sha"]
    except urllib.error.HTTPError as e:
        if e.code in (404, 422):
            return None
        raise


def full_runs_since(az: Azure, project: str, definition: int, repo: str, since: str | None) -> list[FullRun]:
    """Green runs of ``definition`` since ``since`` (or ever) whose plan step chose ``mode=full``."""
    query = {"definitions": definition, "resultFilter": "succeeded", "queryOrder": "finishTimeDescending", "$top": 50}
    if since:
        query["minTime"] = since
    runs = []
    for build in az.get(project, "build/builds", **query)["value"]:
        timeline = az.get(project, f"build/builds/{build['id']}/timeline")
        plan_logs = [
            r["log"]["id"]
            for r in timeline.get("records") or []
            if r.get("type") == "Task" and r.get("name") == "Plan the GPU run" and r.get("log")
        ]
        modes = [plan_mode(az.text(project, f"build/builds/{build['id']}/logs/{log}")) for log in plan_logs]
        if "full" in modes:
            runs.append(FullRun(build["id"], build["sourceVersion"], commit_tree(repo, build["sourceVersion"])))
    return runs


def last_green_scheduled(az: Azure, project: str, definition: int) -> str | None:
    found = az.get(
        project,
        "build/builds",
        definitions=definition,
        reasonFilter="schedule",
        resultFilter="succeeded",
        queryOrder="finishTimeDescending",
        **{"$top": 1},
    )["value"]
    return found[0]["finishTime"] if found else None


def host_state(
    az: Azure, projects: list[str], candidate: int, lease_cmds: list[str], run: Callable[[str], str]
) -> HostState:
    """The pool's state across every project that shares it, plus each host's lease status."""
    state = HostState()
    for project in projects:
        state.running += [
            f"{project} {b['id']}" for b in az.get(project, "build/builds", statusFilter="inProgress")["value"]
        ]
        pending = {
            int(a["pipeline"]["owner"]["id"])
            for a in az.get(project, "pipelines/approvals", state="pending", **{"api-version": "7.1-preview.1"})[
                "value"
            ]
        }
        for b in az.get(project, "build/builds", statusFilter="notStarted")["value"]:
            if b["id"] == candidate:
                continue
            if b["id"] not in pending:
                state.approved_waiting.append(f"{project} {b['id']}")
            elif b.get("reason") == "pullRequest":
                head = (b.get("triggerInfo") or {}).get("pr.sourceSha")
                repo = b.get("repository", {}).get("id", "")
                if (
                    head
                    and repo
                    and hosted_checks_green(github(f"repos/{repo}/commits/{head}/check-runs")["check_runs"])
                ):
                    state.releasable_gates.append(f"{project} {b['id']}")
    for cmd in lease_cmds:
        state.lease += lease_blockers(run(cmd))
    return state


def _run(cmd: str) -> str:
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=60)
    if proc.returncode:
        raise Refused(f"lease status command failed (exit {proc.returncode}): {cmd}: {proc.stderr.strip()[-300:]}")
    return proc.stdout


def covering_run(tree: str, runs: Iterable[FullRun]) -> FullRun | None:
    """The first green full run of exactly ``tree``, if any; a run whose tree could not be read covers nothing."""
    return next((run for run in runs if run.tree is not None and run.tree == tree), None)


def in_job(env: dict, sleep: Callable[[float], None] = time.sleep) -> int:
    """The in-job coverage test for a scheduled build: cancel a nightly run of a covered tree, run anything else."""

    def proceed(why: str) -> int:
        print(f"scheduled-run coverage: {why}; running")
        return 0

    if env.get("BUILD_REASON") != "Schedule":
        return proceed(f"build reason {env.get('BUILD_REASON')!r}, not a scheduled build")
    if env.get("BUILD_CRONSCHEDULE_DISPLAYNAME") == WEEKLY_SCHEDULE:
        return proceed("the weekly extended run, whose extra tiers run in no other build")
    try:
        az = Azure(env["SYSTEM_COLLECTIONURI"], bearer=env.get("SYSTEM_ACCESSTOKEN"))
        project, build_id = env["SYSTEM_TEAMPROJECT"], int(env["BUILD_BUILDID"])
        repo, sha = env["BUILD_REPOSITORY_NAME"], env["BUILD_SOURCEVERSION"]
        definition = az.get(project, f"build/builds/{build_id}")["definition"]["id"]
        tree = commit_tree(repo, sha)
        if tree is None:
            return proceed(f"the tree of {sha} cannot be read")
        covering = covering_run(
            tree, full_runs_since(az, project, definition, repo, last_green_scheduled(az, project, definition))
        )
    except (Refused, urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"##vso[task.logissue type=warning]Scheduled-run coverage could not be established ({exc}); running.")
        return 0
    if covering is None:
        return proceed(f"tree {tree} has no green full run since the last green scheduled run")
    print(
        f"##vso[task.logissue type=warning]This scheduled run's tree {tree} was already run green in full by build "
        f"{covering.build_id} ({covering.sha}); cancelling this run rather than repeating it."
    )
    try:
        az.patch(project, f"build/builds/{build_id}", {"status": "cancelling"})
    except (urllib.error.URLError, ValueError) as exc:
        print(f"##vso[task.logissue type=warning]The cancel request failed ({exc}); running.")
        return 0
    sleep(300)  # the cancel ends this step; reaching the next line means it did not take effect
    print("##vso[task.logissue type=warning]The cancel did not take effect within five minutes; running.")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--in-job"]:
        return in_job(dict(os.environ))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", type=int, required=True, help="the queued scheduled build")
    ap.add_argument("--org", required=True, help="https://dev.azure.com/<org>")
    ap.add_argument("--project", required=True, help="the build's Azure project")
    ap.add_argument("--repo", default="speediedan/interpretune", help="the GitHub repository the pipeline builds")
    ap.add_argument(
        "--pool-project", action="append", default=[], help="another project whose builds share the pool (repeatable)"
    )
    leases = ap.add_mutually_exclusive_group(required=True)
    leases.add_argument(
        "--lease-status-cmd", action="append", default=[], help="a host's lease status command (repeatable)"
    )
    leases.add_argument("--no-lease", action="store_true", help="the hosts have no lease tool; skip the lease check")
    args = ap.parse_args(argv)
    try:
        az = Azure(args.org, os.environ.get("AZURE_DEVOPS_EXT_PAT"))
        build = az.get(args.project, f"build/builds/{args.build}")
        if build.get("reason") != "schedule" or build.get("status") != "notStarted":
            raise Refused(
                f"build {args.build} is not a queued scheduled build "
                f"(reason {build.get('reason')!r}, status {build.get('status')!r})"
            )
        tree = commit_tree(args.repo, build["sourceVersion"])
        if tree is None:
            raise Refused(f"the tree of {build['sourceVersion']} cannot be read from {args.repo}")
        definition = build["definition"]["id"]
        weekly = (build.get("triggerInfo") or {}).get("scheduleName") == WEEKLY_SCHEDULE
        since = last_green_scheduled(az, args.project, definition)
        runs = [] if weekly else full_runs_since(az, args.project, definition, args.repo, since)
        host = host_state(az, [args.project, *args.pool_project], args.build, args.lease_status_cmd, _run)
        waited = datetime.now(timezone.utc) - _parse_time(build["queueTime"])
        verdict = dispose(tree=tree, weekly=weekly, full_runs=runs, host=host, waited=waited)
    except (Refused, urllib.error.URLError, KeyError) as exc:
        print(f"REFUSED scheduled build {args.build}: {exc}", file=sys.stderr)
        return 2
    print(verdict.line(args.build))
    return EXIT[verdict.action]


if __name__ == "__main__":
    sys.exit(main())
