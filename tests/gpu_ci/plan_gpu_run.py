"""Plan one GPU pipeline run: full or selected, extended tiers or not, and how long to wait for the host leases.

The pipeline runs this once, before any GPU work, and exports the plan to the later steps as Azure variables:

- ``IT_GPU_SELECTION_MODE``: ``full``, ``selected`` (only the GPU tests the change selects) or ``none`` (the change
  selects no GPU test, so the GPU phases are skipped);
- ``IT_GPU_SELECTION_FILE``: the file ``tests/conftest.py`` narrows each GPU phase with, set only when selected;
- ``IT_GPU_EXTENDED``: ``1`` when the extended tiers (optional GPU tests, the full profiling tier) also run;
- ``IT_GPU_LEASE_WAIT`` and ``IT_GPU_LEASE_ON_TIMEOUT``: how long the job waits for the host leases, and whether
  running out of time fails the build (a pull request gate) or cancels it (a scheduled run, which retries next time).

Only a pull request gate selects. Every other build runs the full set, and anything the planner cannot establish
(the change set, the pull request's labels) also runs the full set, named in the log, rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tests.gpu_ci.select_gpu_tests import Selection, changed_files, load_areas, select, write_selection_file

#: The pull request label that forces a full GPU run.
FULL_LABEL = "ci:gpu-full"
#: Must match the weekly schedule's ``displayName`` in ``.azure-pipelines/gpu-tests.yml``.
WEEKLY_SCHEDULE = "weekly extended GPU run"
MODES = ("auto", "full")
#: Seconds a PR gate waits for each lease before failing closed, and the whole wait a scheduled run allows before it
#: cancels itself (the next scheduled run is a fresh attempt, so a busy host costs a night, not a red build).
PR_LEASE_WAIT, SCHEDULED_LEASE_WAIT = 2400, 7200


@dataclass
class Plan:
    mode: str  # full | selected | none
    extended: bool
    lease_wait: int
    on_lease_timeout: str  # fail | cancel
    why: str
    selection: Selection | None = None


def plan_run(
    *,
    build_reason: str,
    source_branch: str,
    cron_name: str,
    mode: str,
    extended: bool,
    pr_labels: Callable[[], list[str]],
    changed: Callable[[], list[str]],
    spec: dict,
    root: Path | None,
) -> Plan:
    """Decide what this run executes; every input the decision depends on is passed in, so it is testable."""
    if mode not in MODES:
        raise ValueError(f"unknown GPU run mode {mode!r}; expected one of {MODES}")
    extended = (
        extended or cron_name == WEEKLY_SCHEDULE or source_branch.startswith(("refs/heads/release/", "refs/tags/"))
    )
    scheduled = build_reason == "Schedule"
    lease = (SCHEDULED_LEASE_WAIT, "cancel") if scheduled else (PR_LEASE_WAIT, "fail")

    def full(why: str, selection: Selection | None = None) -> Plan:
        return Plan("full", extended, *lease, why, selection)

    if mode == "full":
        return full("mode=full requested")
    if extended:
        return full("an extended run is always full")
    if build_reason != "PullRequest":
        return full(f"a {build_reason or 'non-PR'} build; only pull request gates select")
    try:
        labels = pr_labels()
    except Exception as e:  # any failure to read the labels must run the full set, not guess
        return full(f"could not read the pull request's labels ({e}); running the full set rather than guessing")
    if FULL_LABEL in labels:
        return full(f"the pull request carries the {FULL_LABEL!r} label")
    try:
        files = changed()
    except Exception as e:  # likewise: an unknown change set selects everything
        return full(f"could not determine the change set ({e}); running the full set rather than guessing")
    sel = select(files, spec, root)
    if sel.full:
        hits = sorted(p for p, r in sel.reasons.items() if r == "run_all" or r.startswith("UNCLASSIFIED"))
        return full("the change touches " + ", ".join(hits), sel)
    if sel.empty:
        return Plan("none", False, *lease, "the change selects no GPU test", sel)
    return Plan("selected", False, *lease, "per-change selection", sel)


def merge_commit_changes(cwd: Path | None = None) -> list[str]:
    """Paths a pull request's merge commit changes relative to its target branch (``HEAD^1``).

    Refuses when ``HEAD`` is not a merge commit: the pipeline validates the merge ref, and diffing anything else against
    its first parent would describe one commit, not the pull request.
    """
    probe = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD^2"], cwd=cwd, capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError("HEAD is not a merge commit (is the checkout fetchDepth at least 2?)")
    return changed_files("HEAD^1", cwd=cwd)


def github_labels(repo: str, number: str, timeout: float = 20.0) -> list[str]:
    """The labels on a public repository's pull request, read without a token."""
    if not (repo and number):
        raise RuntimeError(f"no repository or pull request number (repo={repo!r}, number={number!r})")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues/{number}/labels",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "interpretune-gpu-ci"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return [label["name"] for label in json.load(resp)]


def azure_variables(plan: Plan, selection_file: Path | None) -> list[str]:
    """The ``##vso`` logging commands that export ``plan`` to the job's later steps."""
    out = {
        "IT_GPU_SELECTION_MODE": plan.mode,
        "IT_GPU_EXTENDED": "1" if plan.extended else "0",
        "IT_GPU_LEASE_WAIT": str(plan.lease_wait),
        "IT_GPU_LEASE_ON_TIMEOUT": plan.on_lease_timeout,
    }
    if plan.mode == "selected" and selection_file is not None:
        out["IT_GPU_SELECTION_FILE"] = str(selection_file)
    return [f"##vso[task.setvariable variable={k}]{v}" for k, v in out.items()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="auto", choices=MODES)
    ap.add_argument("--extended", default="false", help="the pipeline's boolean parameter, as Azure renders it")
    args = ap.parse_args(argv)
    env = os.environ
    plan = plan_run(
        build_reason=env.get("BUILD_REASON", ""),
        source_branch=env.get("BUILD_SOURCEBRANCH", ""),
        cron_name=env.get("BUILD_CRONSCHEDULE_DISPLAYNAME", ""),
        mode=args.mode,
        extended=args.extended.lower() == "true",
        pr_labels=lambda: github_labels(
            env.get("BUILD_REPOSITORY_NAME", ""), env.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", "")
        ),
        changed=merge_commit_changes,
        spec=load_areas(),
        root=Path.cwd(),
    )
    selection_file = None
    if plan.selection is not None and plan.mode == "selected":
        selection_file = Path(env.get("AGENT_TEMPDIRECTORY") or tempfile.gettempdir()) / "it_gpu_selection.txt"
        write_selection_file(plan.selection, load_areas(), selection_file)
    print(
        f"GPU run plan: mode={plan.mode} extended={plan.extended} lease_wait={plan.lease_wait}s "
        f"on_lease_timeout={plan.on_lease_timeout}: {plan.why}"
    )
    if plan.selection is not None:
        json.dump(plan.selection.to_json(), sys.stdout, indent=2)
        print()
    if selection_file is not None:
        print(f"selection file {selection_file}:\n{selection_file.read_text()}", end="")
    for line in azure_variables(plan, selection_file):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
