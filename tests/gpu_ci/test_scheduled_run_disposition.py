"""The scheduled-run disposition (CPU): each verdict, the parsers it reads, and that it refuses rather than
guesses."""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.gpu_ci import scheduled_run_disposition as srd
from tests.gpu_ci.scheduled_run_disposition import APPROVAL_CAP, FullRun, HostState, dispose

TREE = "0567c2989ecf3387734a54afd5b42771cb452ecf"
OTHER = "d33e1997" + "0" * 32
FULL = [FullRun(1724, "ea2b8ce9" + "0" * 32, TREE), FullRun(1719, "d6438182" + "0" * 32, OTHER)]
QUIET = HostState()
SOON, LATE = timedelta(minutes=5), APPROVAL_CAP + timedelta(minutes=1)


def test_covered_tree_is_rejected_naming_the_full_run():
    v = dispose(tree=TREE, weekly=False, full_runs=FULL, host=QUIET, waited=SOON)
    assert (v.action, "covered by build 1724") == (srd.REJECT, v.reason[: len("covered by build 1724")])


def test_weekly_run_is_never_rejected_as_covered():
    v = dispose(tree=TREE, weekly=True, full_runs=FULL, host=QUIET, waited=SOON)
    assert v.action == srd.RELEASE and "weekly" in v.reason


def test_uncovered_tree_on_a_quiet_host_is_released():
    v = dispose(tree="f" * 40, weekly=False, full_runs=FULL, host=QUIET, waited=SOON)
    assert v.action == srd.RELEASE


def test_an_unreadable_full_run_tree_covers_nothing():
    runs = [FullRun(1600, "a" * 40, None)]
    assert dispose(tree=TREE, weekly=False, full_runs=runs, host=QUIET, waited=SOON).action == srd.RELEASE


@pytest.mark.parametrize(
    "host, named",
    [
        (HostState(running=["interpretune 1730"]), "running: interpretune 1730"),
        (HostState(approved_waiting=["finetuning-scheduler 1731"]), "approved and waiting: finetuning-scheduler 1731"),
        (HostState(releasable_gates=["interpretune 1729"]), "releasable PR gate: interpretune 1729"),
        (HostState(lease=["gpu held (pid=1 project=x)"]), "lease: gpu held"),
    ],
)
def test_a_busy_host_holds_and_names_what_holds_it(host, named):
    v = dispose(tree="f" * 40, weekly=False, full_runs=FULL, host=host, waited=SOON)
    assert v.action == srd.HOLD and named in v.reason


def test_a_build_still_held_at_the_cap_is_rejected_as_stale():
    v = dispose(tree="f" * 40, weekly=False, full_runs=FULL, host=HostState(lease=["gpu held (x)"]), waited=LATE)
    assert v.action == srd.REJECT and v.reason.startswith("stale") and "gpu held" in v.reason


def test_a_quiet_host_releases_even_past_the_cap():
    assert dispose(tree="f" * 40, weekly=False, full_runs=[], host=QUIET, waited=LATE).action == srd.RELEASE


@pytest.mark.parametrize(
    "log, mode",
    [
        (
            "2026-10-25T20:40:01Z GPU run plan: mode=full extended=False lease_wait=2400s on_lease_timeout=fail: x",
            "full",
        ),
        ("GPU run plan: mode=selected extended=False lease_wait=2400s on_lease_timeout=fail: 3 areas", "selected"),
        ("GPU run plan: mode=none extended=False lease_wait=2400s", "none"),
        # the step's source echoed without running prints no resolved mode, so it must not read as a plan
        ('python -m tests.gpu_ci.plan_gpu_run\nprint(f"GPU run plan: mode={plan.mode}")', None),
        ("", None),
    ],
)
def test_plan_mode_reads_only_a_resolved_plan_line(log, mode):
    assert srd.plan_mode(log) == mode


def test_lease_blockers_name_holders_and_waiters_and_nothing_when_free():
    status = (
        "gpu        HELD  pid=294555 project=v7-research since=2026-10-25T11:13:59-07:00\n"
        "  NVIDIA GeForce RTX 4090 (24564 MiB) GPU-af39 free\n"
        "cpu-heavy  free\n"
        "queue (oldest first):\n"
        "  ticket 0000000057  fts-core-104-nvidia  wants all\n"
    )
    assert srd.lease_blockers(status) == [
        "gpu held (pid=294555 project=v7-research since=2026-10-25T11:13:59-07:00)",
        "fts-core-104-nvidia waiting",
    ]
    assert srd.lease_blockers("gpu        free\ncpu-heavy  free\n") == []


def _check(name, status="completed", conclusion="success", app="github-actions"):
    return {"name": name, "status": status, "conclusion": conclusion, "app": {"slug": app}}


@pytest.mark.parametrize(
    "runs, green",
    [
        ([_check("cpu (ubuntu)"), _check("Lint"), _check("gpu", "queued", None, "azure-pipelines")], True),
        ([_check("cpu (ubuntu)"), _check("cpu (macos)", "in_progress", None)], False),
        ([_check("cpu (ubuntu)"), _check("Lint", conclusion="failure")], False),
        # no hosted check at all is not a green matrix
        ([_check("gpu", "queued", None, "azure-pipelines")], False),
        ([], False),
    ],
)
def test_hosted_checks_green(runs, green):
    assert srd.hosted_checks_green(runs) is green


class FakeAzure:
    """Builds and approvals per project, shaped like the REST answers the disposition reads."""

    def __init__(self, builds: dict, approvals: dict):
        self.builds, self.approvals = builds, approvals

    def get(self, project, path, **query):
        if path == "pipelines/approvals":
            return {"value": [{"pipeline": {"owner": {"id": i}}} for i in self.approvals.get(project, [])]}
        status = query["statusFilter"]
        return {"value": [b for b in self.builds.get(project, []) if b["status"] == status]}


def test_host_state_reads_every_project_on_the_pool(monkeypatch):
    builds = {
        "interpretune": [
            {"id": 1700, "status": "notStarted", "reason": "schedule"},  # the candidate itself
            {
                "id": 1701,
                "status": "notStarted",
                "reason": "pullRequest",
                "triggerInfo": {"pr.sourceSha": "a" * 40},
                "repository": {"id": "speediedan/interpretune"},
            },
        ],
        "finetuning-scheduler": [
            {"id": 1702, "status": "inProgress"},
            {"id": 1703, "status": "notStarted", "reason": "manual"},  # approved, waiting for an agent
        ],
    }
    az = FakeAzure(builds, {"interpretune": [1700, 1701]})
    monkeypatch.setattr(srd, "github", lambda path: {"check_runs": [_check("cpu (ubuntu)")]})
    state = srd.host_state(az, ["interpretune", "finetuning-scheduler"], 1700, ["lease-status"], lambda cmd: "gpu free")
    assert state.running == ["finetuning-scheduler 1702"]
    assert state.approved_waiting == ["finetuning-scheduler 1703"]
    assert state.releasable_gates == ["interpretune 1701"]
    assert state.lease == []


def test_cli_refuses_without_a_credential(monkeypatch, capsys):
    monkeypatch.delenv("AZURE_DEVOPS_EXT_PAT", raising=False)
    argv = ["--build", "1", "--org", "https://dev.azure.com/x", "--project", "p", "--no-lease"]
    assert srd.main(argv) == 2
    assert "AZURE_DEVOPS_EXT_PAT is not set" in capsys.readouterr().err


def test_cli_requires_the_lease_state_or_an_explicit_no_lease():
    with pytest.raises(SystemExit):
        srd.main(["--build", "1", "--org", "https://dev.azure.com/x", "--project", "p"])


def test_cli_refuses_a_build_that_is_not_a_queued_scheduled_build(monkeypatch, capsys):
    monkeypatch.setenv("AZURE_DEVOPS_EXT_PAT", "x")
    monkeypatch.setattr(srd.Azure, "get", lambda self, project, path, **q: {"reason": "manual", "status": "completed"})
    argv = ["--build", "1721", "--org", "https://dev.azure.com/x", "--project", "p", "--no-lease"]
    assert srd.main(argv) == 2
    assert "not a queued scheduled build (reason 'manual', status 'completed')" in capsys.readouterr().err


def test_azure_refuses_a_non_json_answer(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>sign in</html>"

    monkeypatch.setattr(srd.urllib.request, "urlopen", lambda req, timeout: Resp())
    with pytest.raises(srd.Refused, match="not JSON"):
        srd.Azure("https://dev.azure.com/x", "pat").get("p", "build/builds")


def test_parse_time_accepts_azure_precision():
    assert srd._parse_time("2026-10-25T10:00:08.2352154Z").microsecond == 235215
    assert srd._parse_time("2026-10-25T10:00:08Z").second == 8


# --in-job: the coverage test inside a scheduled build, which is not held for approval

JOB_ENV = {
    "BUILD_REASON": "Schedule",
    "BUILD_CRONSCHEDULE_DISPLAYNAME": "nightly full GPU run",
    "SYSTEM_COLLECTIONURI": "https://dev.azure.com/x/",
    "SYSTEM_TEAMPROJECT": "interpretune",
    "SYSTEM_ACCESSTOKEN": "token",
    "BUILD_BUILDID": "1800",
    "BUILD_REPOSITORY_NAME": "speediedan/interpretune",
    "BUILD_SOURCEVERSION": "a855ed3f" + "0" * 32,
}


@pytest.fixture
def job(monkeypatch):
    """Azure and GitHub replaced: the build's definition, its tree, the full runs, and a record of cancels."""
    calls = {"patch": [], "sleep": []}
    monkeypatch.setattr(srd.Azure, "get", lambda self, project, path, **q: {"definition": {"id": 2}})
    monkeypatch.setattr(srd.Azure, "patch", lambda self, project, path, body: calls["patch"].append((path, body)))
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: TREE)
    monkeypatch.setattr(srd, "last_green_scheduled", lambda az, project, definition: "2026-10-12T16:49:20Z")
    monkeypatch.setattr(srd, "full_runs_since", lambda az, project, definition, repo, since: FULL)
    return calls


def _in_job(calls, **env):
    return srd.in_job({**JOB_ENV, **env}, sleep=calls["sleep"].append)


def test_in_job_cancels_a_nightly_of_a_covered_tree(job, capsys):
    assert _in_job(job) == 0
    assert job["patch"] == [("build/builds/1800", {"status": "cancelling"})]
    assert job["sleep"] == [300]
    out = capsys.readouterr().out
    assert "already run green in full by build 1724" in out and "did not take effect" in out


def test_in_job_runs_an_uncovered_tree(job, monkeypatch, capsys):
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: "f" * 40)
    assert _in_job(job) == 0 and job["patch"] == []
    assert "has no green full run" in capsys.readouterr().out


@pytest.mark.parametrize(
    "env, why",
    [
        ({"BUILD_CRONSCHEDULE_DISPLAYNAME": srd.WEEKLY_SCHEDULE}, "weekly extended run"),
        ({"BUILD_REASON": "PullRequest"}, "not a scheduled build"),
        ({"BUILD_REASON": "Manual"}, "not a scheduled build"),
    ],
)
def test_in_job_never_skips_the_weekly_run_or_other_builds(job, capsys, env, why):
    assert _in_job(job, **env) == 0 and job["patch"] == []
    assert why in capsys.readouterr().out


def test_in_job_runs_when_coverage_cannot_be_established(job, monkeypatch, capsys):
    def down(*a, **k):
        raise srd.urllib.error.URLError("unreachable")

    monkeypatch.setattr(srd, "full_runs_since", down)
    assert _in_job(job) == 0 and job["patch"] == []
    assert "could not be established" in capsys.readouterr().out


def test_in_job_runs_when_the_tree_cannot_be_read(job, monkeypatch, capsys):
    monkeypatch.setattr(srd, "commit_tree", lambda repo, sha: None)
    assert _in_job(job) == 0 and job["patch"] == []
    assert "cannot be read" in capsys.readouterr().out


def test_in_job_runs_when_the_cancel_request_fails(job, monkeypatch, capsys):
    def refused(self, project, path, body):
        raise srd.urllib.error.HTTPError("u", 403, "forbidden", None, None)

    monkeypatch.setattr(srd.Azure, "patch", refused)
    assert _in_job(job) == 0 and job["sleep"] == []
    assert "cancel request failed" in capsys.readouterr().out


def test_in_job_without_a_token_runs(job, capsys):
    assert _in_job(job, SYSTEM_ACCESSTOKEN="") == 0 and job["patch"] == []
    assert "could not be established" in capsys.readouterr().out


def test_main_dispatches_in_job(monkeypatch):
    seen = {}
    monkeypatch.setattr(srd, "in_job", lambda env: seen.setdefault("env", env) and 0)
    monkeypatch.setenv("BUILD_REASON", "Schedule")
    assert srd.main(["--in-job"]) == 0
    assert seen["env"]["BUILD_REASON"] == "Schedule"
