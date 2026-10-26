"""The ``gpu_lease_wrap.sh`` drift check, offline: identical, drifted, line-ending-only drift, and an unreadable
copy.

The CI job reads finetuning-scheduler's copy from GitHub. These tests use ``--against`` with local files, so a change
that breaks the comparison fails here instead of a drift passing unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_gpu_lease_wrap_sync.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_gpu_lease_wrap_sync", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_identical_copy_passes(checker, tmp_path, capsys):
    other = tmp_path / "gpu_lease_wrap.sh"
    other.write_bytes(checker.LOCAL.read_bytes())
    assert checker.main(["--against", str(other)]) == 0
    assert "byte-identical" in capsys.readouterr().out


def test_drifted_copy_fails_with_a_diff(checker, tmp_path, capsys):
    other = tmp_path / "gpu_lease_wrap.sh"
    other.write_bytes(checker.LOCAL.read_bytes() + b"# a line only the other copy has\n")
    assert checker.main(["--against", str(other)]) == 1
    out = capsys.readouterr().out
    assert "has drifted" in out and "+# a line only the other copy has" in out


def test_line_ending_drift_is_still_drift(checker, tmp_path, capsys):
    other = tmp_path / "gpu_lease_wrap.sh"
    other.write_bytes(checker.LOCAL.read_bytes().replace(b"\n", b"\r\n"))
    assert checker.main(["--against", str(other)]) == 1


def test_unreadable_copy_is_refused(checker, tmp_path, capsys):
    assert checker.main(["--against", str(tmp_path / "missing.sh")]) == 2
    assert "REFUSED" in capsys.readouterr().err
