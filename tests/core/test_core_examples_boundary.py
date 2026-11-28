"""The core package and its tests import nothing from ``it_examples``.

The examples moved to the interpretune/examples repository and interpretune's ``it_examples`` tree is gone, so an
import of it would fail in every install. Example components reach core tests from the warmed
components cache instead (``tests/hf_warm_manifest.yaml``). Strings that name ``it_examples`` (snapshot-rewrite tables,
subprocess probes, path fixtures) are fine, so the scan reads import statements and ``importlib.import_module`` /
``__import__`` calls, never string contents.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_ROOTS = ("src/interpretune", "tests")


def examples_imports(path: Path) -> list[str]:
    """Every ``it_examples`` module that ``path`` imports, as ``<file>:<line>:<module>``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module else []
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            value = node.args[0].value
            modules = [value] if name in ("import_module", "__import__") and isinstance(value, str) else []
        else:
            continue
        found.extend(
            f"{path.name}:{node.lineno}:{module}"
            for module in modules
            if module == "it_examples" or module.startswith("it_examples.")
        )
    return found


def test_scan_finds_every_import_form(tmp_path):
    """The positive control: a scan that cannot find a planted import proves nothing by finding none."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import importlib\n"
        "import it_examples\n"
        "from it_examples.seeds import ensure_local_seeds\n"
        "def lazy():\n"
        "    from it_examples import utils\n"
        "    return importlib.import_module('it_examples.utils.nb_ui_utils')\n"
        "FIXTURE = 'from it_examples.examples import rte'\n",
        encoding="utf-8",
    )
    assert examples_imports(planted) == [
        "planted.py:2:it_examples",
        "planted.py:3:it_examples.seeds",
        "planted.py:5:it_examples",
        "planted.py:6:it_examples.utils.nb_ui_utils",
    ]


@pytest.mark.parametrize("root", SCANNED_ROOTS)
def test_no_examples_imports(root):
    files = sorted((REPO_ROOT / root).rglob("*.py"))
    assert len(files) > 50, f"only {len(files)} Python files under {root}: the scan would pass on nothing"
    offenders = [ref for path in files for ref in examples_imports(path)]
    assert not offenders, (
        f"{root} imports it_examples, which the core package and its tests must not depend on. Read example "
        f"components from the warmed components cache instead: {offenders}"
    )
