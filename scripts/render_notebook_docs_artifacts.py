#!/usr/bin/env python3
"""Shim over the docs-artifact renderer, kept so docs-build, pre-commit habits and existing commands keep working.

The engine is ``interpretune.utils.notebook_docs`` and ships as the console script
``interpretune-render-notebook-docs``, configured from ``[tool.notebook-publishing]`` in ``pyproject.toml``.
Interpretune's own op-surface checks are its hooks file, ``scripts/notebook_docs_hooks.py``.

The module is loaded from its file rather than imported through the package: docs-build runs ``--check-stale``
before installing anything, and ``import interpretune`` would pull in torch. The engine itself needs only the
standard library (papermill, only to execute).

Usage:
    python scripts/render_notebook_docs_artifacts.py --no-execute
    python scripts/render_notebook_docs_artifacts.py --notebook saelens_adapter_example
    python scripts/render_notebook_docs_artifacts.py --check-stale
    python scripts/render_notebook_docs_artifacts.py --list
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE = REPO_ROOT / "src" / "interpretune" / "utils" / "notebook_docs.py"


def _load_engine():
    spec = importlib.util.spec_from_file_location("it_notebook_docs", ENGINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve `from __future__` annotations through sys.modules
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    sys.exit(_load_engine().main(root=REPO_ROOT))
