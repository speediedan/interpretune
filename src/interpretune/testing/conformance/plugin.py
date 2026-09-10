"""pytest plugin: gate enforcement, the selection report, and the vacuity guards.

Consumers add ``pytest_plugins = ["interpretune.testing.conformance.plugin"]`` to their ``conftest.py``.
"""

from __future__ import annotations

import os

import pytest

from .gates import UNDECLARED, SelectionReport, gate_of

_REPORT_KEY = pytest.StashKey[SelectionReport]()
STRICT_ENV = "IT_CONFORMANCE_STRICT"


def pytest_configure(config):
    """Register the marker and the per-session selection report."""
    config.addinivalue_line("markers", "conformance: a capability-gated conformance case")
    config.stash[_REPORT_KEY] = SelectionReport()


def pytest_collection_modifyitems(config, items):
    """Mark every case that carries a gate, so the report can tell cases from ordinary tests.

    Never narrows ``items``: this is a marker pass, and a selector here is the exact regression the
    interpretune conftest documents.
    """
    for item in items:
        fn = getattr(item, "obj", None)
        if fn is not None and gate_of(fn) is not None:
            item.add_marker(pytest.mark.conformance)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Classify each conformance case's outcome for the report."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    if item.get_closest_marker("conformance") is None:
        return
    selection = item.config.stash[_REPORT_KEY]
    if report.passed:
        selection.record(item.name, "ran")
    elif report.failed:
        selection.record(item.name, "failed")
    elif report.skipped:
        reason = ""
        if isinstance(report.longrepr, tuple):
            reason = str(report.longrepr[2])
        else:
            reason = str(report.longrepr)
        selection.record(item.name, "skipped-undeclared" if UNDECLARED in reason else "skipped-other")


def collected_any_case(selection: SelectionReport) -> bool:
    """Whether this run had a conformance case in scope at all.

    The report and the vacuity guards are meaningless for a run that collected no case (a targeted run of an unrelated
    file, `-k` selecting ordinary tests): printing "no conformance case ran" there asserts something false about the run
    and trains the reader to skip the marker, which is how a real vacuity then reads as noise.
    """
    return bool(selection.ran or selection.skipped_undeclared or selection.skipped_other or selection.failed)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the selection report and any vacuity problem after the summary, for a run that had cases in scope."""
    selection = config.stash.get(_REPORT_KEY, None)
    if selection is None or not collected_any_case(selection):
        return
    terminalreporter.write_sep("-", "conformance")
    terminalreporter.write_line(selection.render())
    problems = vacuity_problems(selection, strict=os.getenv(STRICT_ENV, "0") == "1")
    for p in problems:
        terminalreporter.write_line(f"VACUITY: {p}")


REPORT_PATH_ENV = "IT_CONFORMANCE_REPORT"
COMPONENT_DIR_ENV = "IT_CONFORMANCE_COMPONENT_DIR"


def write_report_artifact(selection: SelectionReport, path: str, *, exitstatus: int) -> None:
    """Write the selection report as JSON to ``path``, with the provenance a consumer needs to judge it.

    Provenance names the interpretune version, the git head of the tree that ran (when it is a checkout), the component
    directory and its own revision when ``IT_CONFORMANCE_COMPONENT_DIR`` names it (the last commit touching that
    directory, the key a card compares at publish), the time, and the exit status, so a reader can tell which source
    measured the declaration and whether the run that produced it was green. The artifact is what the suite measured on
    a composed session; it is never a manifest's claim.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("interpretune")

    from interpretune.hub.revisions import directory_revision, repo_head

    component_dir = os.getenv(COMPONENT_DIR_ENV)
    artifact = selection.as_artifact(
        interpretune_version=__version__,
        git_head=repo_head(Path.cwd()),
        component_dir=component_dir,
        component_revision=directory_revision(Path(component_dir)) if component_dir else None,
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        exit_status=int(exitstatus),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):
    """Turn a vacuous green into a failure, and write the report artifact when a path was given."""
    selection = session.config.stash.get(_REPORT_KEY, None)
    if selection is None:
        return
    report_path = os.getenv(REPORT_PATH_ENV)
    if report_path and collected_any_case(selection):
        write_report_artifact(selection, report_path, exitstatus=int(exitstatus))
    if not collected_any_case(selection):
        return  # no conformance cases were collected at all: this run was not a conformance run
    if vacuity_problems(selection, strict=os.getenv(STRICT_ENV, "0") == "1") and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def vacuity_problems(selection: SelectionReport, *, strict: bool) -> list[str]:
    """The two guards.

    A green run that proved nothing must not read as green.
    """
    problems = []
    if not selection.ran and not selection.failed:
        problems.append(
            "no conformance case ran: every gated case was skipped, so nothing about the adapter beyond "
            "composition was checked"
        )
    if strict and selection.skipped_other:
        problems.append(
            f"{len(selection.skipped_other)} case(s) skipped for a reason other than an undeclared gate under "
            f"{STRICT_ENV}=1: {selection.skipped_other}"
        )
    return problems
