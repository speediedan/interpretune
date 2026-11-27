"""Every text-mode read or write in this repository names its encoding.

Without ``encoding=`` Python decodes with the platform's default, which is cp1252 on a Windows host that is not in
UTF-8 mode. Most UTF-8 bytes decode there into the wrong characters without an error, and five byte values raise. The
hosted Windows job runs with ``PYTHONUTF8=1``, so it cannot see either failure; this check reads the source instead.

Ruff's ``unspecified-encoding`` rule (a preview rule) is not used here because it only reports a ``read_text`` or
``write_text`` whose receiver it can infer to be a ``Path``, which is a small part of the calls in this tree.

Not covered: a call whose mode is computed, a method ``open`` taking a path (fsspec, ``os.open``), ``subprocess`` with
``text=True`` (its encoding is the child's), and notebooks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("src", "tests", "scripts", "requirements")

# Trees this repository does not own the text of: imported research code.
EXEMPT_PREFIXES = (
    "tests/ioi_parity/",
    "tests/mech_interp_parity/",
    "tests/concept_direction_approach_parity/",
)


def _constant_mode(node: ast.Call, position: int) -> object:
    if len(node.args) > position:
        arg = node.args[position]
        return arg.value if isinstance(arg, ast.Constant) else ...
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return keyword.value.value if isinstance(keyword.value, ast.Constant) else ...
    return None


def _is_text_mode(mode: object) -> bool:
    return mode is None or (isinstance(mode, str) and "b" not in mode)


def _omits_encoding(node: ast.Call) -> bool:
    keywords = {keyword.arg for keyword in node.keywords}
    if "encoding" in keywords or None in keywords:
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "open":
        return len(node.args) <= 3 and _is_text_mode(_constant_mode(node, 1))
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "read_text":
        return not node.args
    if func.attr == "write_text":
        return len(node.args) < 2
    if func.attr == "FileHandler":
        return len(node.args) < 3
    if func.attr in ("NamedTemporaryFile", "TemporaryFile"):
        mode = _constant_mode(node, 0)
        return isinstance(mode, str) and "b" not in mode
    if func.attr == "open" and len(node.args) == 1:
        mode = _constant_mode(node, 0)
        return isinstance(mode, str) and len(mode) <= 3 and set(mode) <= set("rwaxt+")
    return False


def unspecified_encodings(source: str) -> list[int]:
    """Return the line of each text-mode call in ``source`` that names no encoding."""
    return [n.lineno for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and _omits_encoding(n)]


def _scanned_files() -> list[Path]:
    files = []
    for root in SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if not path.relative_to(REPO_ROOT).as_posix().startswith(EXEMPT_PREFIXES):
                files.append(path)
    return files


def test_every_text_read_and_write_names_its_encoding():
    files = _scanned_files()
    scanned = {path.relative_to(REPO_ROOT).as_posix() for path in files}
    # One file per scanned root, so a root that stopped matching cannot pass by scanning nothing.
    for known in (
        "src/interpretune/session.py",
        "tests/core/test_explicit_text_encoding.py",
        "scripts/generate_op_stubs.py",
        "requirements/utils/collect_env_details.py",
    ):
        assert known in scanned, f"{known} was not scanned"
    offenders = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{line}"
        for path in files
        for line in unspecified_encodings(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, 'pass encoding="utf-8":\n' + "\n".join(offenders)


@pytest.mark.parametrize(
    "source",
    [
        "p.read_text()",
        "p.write_text(s)",
        "p.write_text(data=s)",
        "open(p)",
        "open(p, 'r')",
        "open(p, mode='w')",
        "p.open('a')",
        "tempfile.NamedTemporaryFile(mode='w', suffix='.py')",
        "logging.FileHandler(p, mode='a')",
    ],
)
def test_a_call_without_an_encoding_is_reported(source):
    assert unspecified_encodings(source) == [1]


@pytest.mark.parametrize(
    "source",
    [
        "p.read_text(encoding='utf-8')",
        "p.read_text('utf-8')",
        "p.write_text(s, 'utf-8')",
        "open(p, 'rb')",
        "open(p, mode='wb')",
        "open(p, encoding='utf-8')",
        "open(p, mode)",
        "open(p, **kwargs)",
        "os.open(p, os.O_RDONLY)",
        "fs.open(p, 'rb')",
        "tempfile.NamedTemporaryFile()",
        "distribution.read_text('METADATA')",
        "logging.FileHandler(p, encoding='utf-8')",
    ],
)
def test_a_binary_or_explicit_call_is_accepted(source):
    assert unspecified_encodings(source) == []
