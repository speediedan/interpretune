"""A pytest plugin recording what a GPU CI phase collected and how each node ended (#703).

Ported from finetuning-scheduler's ``tests/gpu_ci/phase_plugin.py`` so both repositories' phase reports read the same
records. Load it with ``-p tests.gpu_ci.phase_plugin`` (or through ``PYTEST_ADDOPTS``, which the per-test pytest
processes ``tests/special_tests.sh`` starts inherit). It does nothing unless ``IT_PHASE_RECORD_DIR`` names a directory;
then each pytest session appends JSON lines to its own file there:

- ``{"event": "start"}`` when the session starts;
- ``{"event": "collect_error", "nodeid": ...}`` for each collector that failed (an import error, say);
- ``{"event": "collected", "nodeids": [...]}`` after collection, so after ``tests/conftest.py`` has selected the phase's
  tests (and, in a two-stage run, the stage's memory class);
- ``{"event": "outcome", "nodeid": ..., "outcome": ...}`` for each test stage that decides a node's outcome;
- ``{"event": "finish", "exitstatus": ...}`` when the session finishes. A session that died never writes it, which
  ``phase_report.py`` counts as a missing completion.

Every line carries the session's ``rank`` (``LOCAL_RANK``, else ``RANK``, else 0), ``world_size`` when a launcher set
it, and ``pid``.
"""

import json
import os
from pathlib import Path

RECORD_DIR_ENV = "IT_PHASE_RECORD_DIR"


def _identity() -> dict:
    world_size = os.environ.get("WORLD_SIZE", "")
    return {
        "rank": int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))),
        "world_size": int(world_size) if world_size.isdigit() else None,
        "pid": os.getpid(),
    }


def node_outcome(report) -> str | None:
    """The phase-report outcome one ``pytest_runtest_logreport`` report decides, or ``None`` when it decides
    nothing.

    Mirrors pytest's own terminal categories: a failed setup or teardown is an error, a failed call a failure, an
    expected failure ``xfailed`` (a strict XPASS already arrives as a failed call), a skip at any stage ``skipped``. An
    attempt pytest-rerunfailures retries (outcome ``rerun``; the cuda phase runs ``--reruns 2``) decides nothing: the
    final attempt does.
    """
    if report.outcome == "rerun":
        return None
    if report.failed:
        return "failed" if report.when == "call" else "errors"
    if report.skipped:
        return "xfailed" if hasattr(report, "wasxfail") else "skipped"
    if report.when == "call":
        return "xpassed" if hasattr(report, "wasxfail") else "passed"
    return None


class _PhaseRecorder:
    """Appends one session's records to its own JSON-lines file, so concurrent sessions never interleave writes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**record, **_identity()}) + "\n")

    def pytest_sessionstart(self, session):
        self._write({"event": "start"})

    def pytest_collectreport(self, report):
        if report.failed:
            self._write({"event": "collect_error", "nodeid": report.nodeid})

    def pytest_collection_finish(self, session):
        self._write({"event": "collected", "nodeids": [item.nodeid for item in session.items]})

    def pytest_runtest_logreport(self, report):
        outcome = node_outcome(report)
        if outcome is not None:
            self._write({"event": "outcome", "nodeid": report.nodeid, "when": report.when, "outcome": outcome})

    def pytest_sessionfinish(self, session, exitstatus):
        self._write({"event": "finish", "exitstatus": int(exitstatus)})


def pytest_configure(config):
    record_dir = os.environ.get(RECORD_DIR_ENV)
    if not record_dir:
        return
    Path(record_dir).mkdir(parents=True, exist_ok=True)
    ident = _identity()
    path = Path(record_dir) / f"records-rank{ident['rank']}-pid{ident['pid']}.jsonl"
    config.pluginmanager.register(_PhaseRecorder(path), "it_phase_recorder")
