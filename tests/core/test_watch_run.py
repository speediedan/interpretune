"""The armed watcher (scripts/watch_run.sh) covers every terminal state by construction.

Each case asserts that exactly one ``WATCH`` line is printed and that it names the right state; the timeout
and query-failed cases are the ones a hand-written loop silently omits.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "watch_run.sh"
pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="bash helper")


def _run(*args: str, env: dict | None = None, timeout: int = 60) -> tuple[int, list[str]]:
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout, env=env or os.environ.copy()
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("WATCH ")]
    return proc.returncode, lines


class TestChildCommand:
    def test_exit_zero_is_green(self):
        code, lines = _run("--", "true")
        assert code == 0 and len(lines) == 1 and " GREEN " in lines[0]

    def test_nonzero_exit_is_red(self):
        code, lines = _run("--", "bash", "-c", "exit 3")
        assert code == 1 and len(lines) == 1 and " RED exit 3" in lines[0]

    def test_a_signal_is_dead_not_red(self):
        code, lines = _run("--", "bash", "-c", "kill -KILL $$")
        assert code == 3 and len(lines) == 1 and " DEAD killed by signal 9" in lines[0]

    def test_the_deadline_is_a_distinct_terminal_state(self):
        code, lines = _run("--timeout", "1", "--", "sleep", "5")
        assert code == 2 and len(lines) == 1 and " TIMEOUT " in lines[0]


class TestPidWithLog:
    def _sleeper(self) -> subprocess.Popen:
        return subprocess.Popen(["sleep", "1"])

    def test_a_pytest_summary_with_failures_is_red(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("...\n=========== 2 failed, 40 passed in 3.00s ===========\n")
        p = self._sleeper()
        code, lines = _run("--pid", str(p.pid), "--log", str(log), "--interval", "1")
        assert code == 1 and len(lines) == 1 and " RED " in lines[0] and "2 failed" in lines[0]

    def test_a_clean_summary_is_green(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("=========== 42 passed, 3 skipped in 3.00s ===========\n")
        p = self._sleeper()
        code, lines = _run("--pid", str(p.pid), "--log", str(log), "--interval", "1")
        assert code == 0 and len(lines) == 1 and " GREEN " in lines[0]

    def test_no_summary_means_dead_not_green(self, tmp_path):
        """The state a hand-written loop reports as nothing: the process is gone and the log stops mid-run."""
        log = tmp_path / "run.log"
        log.write_text("tests/core/test_x.py::test_y PASSED\n")
        p = self._sleeper()
        code, lines = _run("--pid", str(p.pid), "--log", str(log), "--interval", "1")
        assert code == 3 and len(lines) == 1 and " DEAD " in lines[0]

    def test_a_pid_that_is_already_gone_is_dead_at_start(self):
        p = subprocess.Popen(["true"])
        p.wait()
        time.sleep(0.2)
        code, lines = _run("--pid", str(p.pid), "--interval", "1")
        assert code == 3 and len(lines) == 1 and "DEAD not running at watch start" in lines[0]


class TestFind:
    def test_a_pattern_in_the_watchers_own_command_line_does_not_match_itself(self):
        """Rule 1 as a property: the pattern is spelled on this very command line and must resolve to nothing."""
        marker = "zz-watch-self-match-probe-zz"
        code, lines = _run("--find", marker, "--interval", "1")
        assert code == 3 and len(lines) == 1 and " DEAD no process matches outside this watcher's own chain" in lines[0]

    def test_a_real_process_is_found_and_watched(self, tmp_path):
        marker = tmp_path.name
        # Two commands, so bash stays the parent and its command line (with the marker) is what pgrep sees; a
        # single command would be exec'd and the marker would vanish with the shell.
        p = subprocess.Popen(["bash", "-c", f"sleep 1; : {marker}"])
        code, lines = _run("--find", f"sleep 1; : {marker}", "--uid", "me", "--interval", "1")
        p.wait()
        assert code == 0 and len(lines) == 1 and " EXITED " in lines[0]

    def test_an_ambiguous_match_is_refused_not_guessed(self, tmp_path):
        marker = tmp_path.name
        ps = [subprocess.Popen(["bash", "-c", f"sleep 2; : dup {marker}"]) for _ in range(2)]
        code, lines = _run("--find", f"sleep 2; : dup {marker}", "--interval", "1")
        for p in ps:
            p.wait()
        assert code == 4 and len(lines) == 1 and " QUERY-FAILED ambiguous: 2 processes" in lines[0]


class TestBrokenProbe:
    def test_a_probe_that_returns_nothing_is_query_failed_not_pending(self, tmp_path):
        """A broken az on PATH must surface as its own state, never as a healthy poll that keeps waiting."""
        fake = tmp_path / "az"
        fake.write_text("#!/usr/bin/env bash\nexit 1\n")
        fake.chmod(0o755)
        env = {**os.environ, "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}"}
        code, lines = _run("--azure-build", "1", "--interval", "0", "--max-query-failures", "3", env=env)
        assert code == 4 and len(lines) == 1 and " QUERY-FAILED 3 consecutive probe failures" in lines[0]

    def test_a_completed_build_is_classified_from_its_result(self, tmp_path):
        fake = tmp_path / "az"
        # The real `az ... --query "[status,result]" -o tsv` prints one element per LINE (captured, not assumed);
        # the first fake wrote them tab-separated and passed while the live command was misread.
        fake.write_text("#!/usr/bin/env bash\nprintf 'completed\\ncanceled\\n'\n")
        fake.chmod(0o755)
        env = {**os.environ, "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}"}
        code, lines = _run("--azure-build", "1", "--interval", "0", env=env)
        assert code == 3 and len(lines) == 1 and " DEAD canceled" in lines[0]
