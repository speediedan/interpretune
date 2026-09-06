"""Shim over the notebook publishing engine, kept so the pre-commit hook and existing habits keep working.

The engine is ``interpretune.utils.notebook_publishing`` and ships as the console script
``interpretune-publish-notebooks``, parameterized from ``[tool.interpretune.notebooks]`` in ``pyproject.toml``, so an
adapter repository publishes its own example notebooks with the same engine and copies nothing.

The module is loaded from its file rather than imported through the package: the pre-commit and docs jobs run
this without interpretune's dependencies installed, and ``import interpretune`` would pull in torch. The engine
itself imports only the standard library.

Usage:
    python scripts/publish_notebooks.py [--dry-run] [--check-only] [--force]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE = REPO_ROOT / "src" / "interpretune" / "utils" / "notebook_publishing.py"


def _load_engine():
    spec = importlib.util.spec_from_file_location("it_notebook_publishing", ENGINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve `from __future__` annotations through sys.modules
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    sys.exit(_load_engine().main(root=REPO_ROOT))
