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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the selection report and any vacuity problem after the summary."""
    selection = config.stash.get(_REPORT_KEY, None)
    if selection is None:
        return
    terminalreporter.write_sep("-", "conformance")
    terminalreporter.write_line(selection.render())
    problems = vacuity_problems(selection, strict=os.getenv(STRICT_ENV, "0") == "1")
    for p in problems:
        terminalreporter.write_line(f"VACUITY: {p}")


def pytest_sessionfinish(session, exitstatus):
    """Turn a vacuous green into a failure."""
    selection = session.config.stash.get(_REPORT_KEY, None)
    if selection is None:
        return
    if not (selection.ran or selection.skipped_undeclared or selection.skipped_other or selection.failed):
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
