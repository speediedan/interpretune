"""The shared-scripts drift check, offline: identical, drifted, line-ending-only drift, and an unreadable copy.

The CI job reads finetuning-scheduler's copies from GitHub. These tests use ``--against-dir`` with local copies, so a
change that breaks the comparison fails here instead of a drift passing unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_shared_scripts_sync.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_shared_scripts_sync", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def copies(checker, tmp_path):
    """Byte-identical copies of every shared script in a directory, keyed by file name."""
    for path in checker.SHARED:
        (tmp_path / Path(path).name).write_bytes((checker.REPO / path).read_bytes())
    return tmp_path


def test_every_shared_script_is_checked(checker):
    assert set(checker.SHARED) >= {
        "scripts/gpu_lease_wrap.sh",
        "scripts/infra_utils.sh",
        "scripts/manage_standalone_processes.sh",
    }
    assert all((checker.REPO / path).is_file() for path in checker.SHARED)


def test_identical_copies_pass(checker, copies, capsys):
    assert checker.main(["--against-dir", str(copies)]) == 0
    out = capsys.readouterr().out
    assert all(f"{Path(p).name} is byte-identical" in out for p in checker.SHARED)


@pytest.mark.parametrize("name", ["gpu_lease_wrap.sh", "manage_standalone_processes.sh"])
def test_a_drifted_copy_fails_with_a_diff(checker, copies, capsys, name):
    target = copies / name
    target.write_bytes(target.read_bytes() + b"# a line only the other copy has\n")
    assert checker.main(["--against-dir", str(copies)]) == 1
    out = capsys.readouterr().out
    assert f"{name} has drifted" in out and "+# a line only the other copy has" in out


def test_line_ending_drift_is_still_drift(checker, copies):
    target = copies / "manage_standalone_processes.sh"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert checker.main(["--against-dir", str(copies)]) == 1


def test_an_unreadable_copy_is_refused_and_outranks_drift(checker, copies, capsys):
    (copies / "gpu_lease_wrap.sh").unlink()
    drifted = copies / "manage_standalone_processes.sh"
    drifted.write_bytes(drifted.read_bytes() + b"#\n")
    assert checker.main(["--against-dir", str(copies)]) == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err and "manage_standalone_processes.sh has drifted" in captured.out
