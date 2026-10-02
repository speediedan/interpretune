"""The GPU run planner (CPU): which builds select, which run full, and that it refuses to guess."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.gpu_ci.plan_gpu_run import (
    FULL_LABEL,
    PR_LEASE_WAIT,
    SCHEDULED_LEASE_WAIT,
    WEEKLY_SCHEDULE,
    azure_variables,
    merge_commit_changes,
    plan_run,
)
from tests.gpu_ci.select_gpu_tests import load_areas

ROOT = Path(__file__).resolve().parents[2]
JLENS_SRC = "src/interpretune/analysis/ops/bundled/jlens/jlens_ops.py"


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_areas()


def _plan(
    spec,
    *,
    reason="PullRequest",
    branch="refs/pull/1/merge",
    cron="",
    mode="auto",
    extended=False,
    labels=(),
    files=(JLENS_SRC,),
):
    def pr_labels():
        if isinstance(labels, Exception):
            raise labels
        return list(labels)

    def changed():
        if isinstance(files, Exception):
            raise files
        return list(files)

    return plan_run(
        build_reason=reason,
        source_branch=branch,
        cron_name=cron,
        mode=mode,
        extended=extended,
        pr_labels=pr_labels,
        changed=changed,
        spec=spec,
        root=ROOT,
    )


def test_a_pull_request_gate_selects(spec):
    plan = _plan(spec)
    assert plan.mode == "selected" and "jlens" in plan.selection.areas
    assert (plan.lease_wait, plan.on_lease_timeout) == (PR_LEASE_WAIT, "fail")


@pytest.mark.parametrize(
    "kwargs, why",
    [
        ({"mode": "full"}, "mode=full"),
        ({"labels": (FULL_LABEL, "bug")}, FULL_LABEL),
        ({"reason": "Manual"}, "only pull request gates select"),
        ({"reason": "IndividualCI", "branch": "refs/heads/release/0.1"}, "extended"),
        ({"files": ("pyproject.toml", JLENS_SRC)}, "pyproject.toml"),
        ({"files": ("src/interpretune/brand_new_module.py",)}, "brand_new_module.py"),
    ],
)
def test_what_runs_the_full_set(spec, kwargs, why):
    plan = _plan(spec, **kwargs)
    assert plan.mode == "full" and why in plan.why, plan


@pytest.mark.parametrize(
    "kwargs, what",
    [
        ({"labels": RuntimeError("HTTP 403")}, "labels"),
        ({"files": RuntimeError("HEAD is not a merge commit")}, "change set"),
    ],
)
def test_an_unknown_input_runs_the_full_set_by_name(spec, kwargs, what):
    """A failure to read what selection depends on must widen to the full set, never select from a guess."""
    plan = _plan(spec, **kwargs)
    assert plan.mode == "full" and what in plan.why and "rather than guessing" in plan.why, plan


def test_a_change_selecting_no_gpu_test_skips_the_gpu_phases(spec):
    plan = _plan(spec, files=("docs/index.md",))
    assert plan.mode == "none"
    assert not any("IT_GPU_SELECTION_FILE" in v for v in azure_variables(plan, Path("/tmp/x")))


def test_scheduled_runs_wait_longer_and_cancel_instead_of_failing(spec):
    nightly = _plan(spec, reason="Schedule", branch="refs/heads/main", cron="nightly full GPU run")
    weekly = _plan(spec, reason="Schedule", branch="refs/heads/main", cron=WEEKLY_SCHEDULE)
    for plan in (nightly, weekly):
        assert plan.mode == "full"
        assert (plan.lease_wait, plan.on_lease_timeout) == (SCHEDULED_LEASE_WAIT, "cancel")
    assert not nightly.extended and weekly.extended


def test_the_weekly_schedule_name_matches_the_pipeline(spec):
    """The planner knows the weekly run by its schedule's name; a rename must break this test, not the run."""
    pipeline = (ROOT / ".azure-pipelines" / "gpu-tests.yml").read_text()
    assert f'displayName: "{WEEKLY_SCHEDULE}"' in pipeline


def test_an_unknown_mode_is_refused(spec):
    with pytest.raises(ValueError, match="unknown GPU run mode 'selected'"):
        _plan(spec, mode="selected")


def test_the_exported_variables(spec):
    plan = _plan(spec)
    sel_file = Path("/agent/tmp/sel.txt")
    lines = azure_variables(plan, sel_file)
    assert "##vso[task.setvariable variable=IT_GPU_SELECTION_MODE]selected" in lines
    assert f"##vso[task.setvariable variable=IT_GPU_SELECTION_FILE]{sel_file}" in lines
    assert "##vso[task.setvariable variable=IT_GPU_LEASE_ON_TIMEOUT]fail" in lines


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _pr_merge_repo(tmp_path: Path, check_before_merge: bool = False) -> Path:
    """A repository whose HEAD merges a pull request branch into a target branch that moved after it branched."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "ci")
    (repo / "a.txt").write_text("a")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    if check_before_merge:
        with pytest.raises(RuntimeError, match="not a merge commit"):
            merge_commit_changes(repo)
    _git(repo, "checkout", "-q", "-b", "pr")
    (repo / "b.txt").write_text("b")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pr change")
    _git(repo, "checkout", "-q", "main")
    (repo / "c.txt").write_text("c")  # a target-branch change the pull request did not make
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "main moved")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge", "pr")
    return repo


def test_the_change_set_is_read_from_the_merge_commit(tmp_path):
    assert merge_commit_changes(_pr_merge_repo(tmp_path, check_before_merge=True)) == ["b.txt"]


def test_a_git_failure_reports_gits_own_message(tmp_path):
    """Only a quiet miss means "not a merge commit"; any other failure surfaces git's message, not a guess."""
    with pytest.raises(RuntimeError, match=r"git could not read the checkout \(exit \d+\): .*not a git repository"):
        merge_commit_changes(tmp_path)


def test_a_checkout_owned_by_another_user_is_read_once_trusted(tmp_path, monkeypatch):
    """What the pipeline's container meets: the agent checks out on the host, so git sees another uid's repository.

    ``GIT_TEST_ASSUME_DIFFERENT_OWNER`` reproduces that refusal without a second uid, and the command-line-scope
    ``safe.directory`` the pipeline step sets must lift it. Global and system configuration are shut out, since a
    runner that already trusts every directory (``safe.directory = *``) would never refuse.
    """
    repo = _pr_merge_repo(tmp_path)
    empty_global = tmp_path / "empty.gitconfig"
    empty_global.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_global))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    with pytest.raises(RuntimeError, match="dubious ownership"):
        merge_commit_changes(repo)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", repo.as_posix())
    assert merge_commit_changes(repo) == ["b.txt"]
