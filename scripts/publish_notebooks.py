"""Shim over the notebook publishing engine, kept so the pre-commit hook and existing habits keep working.

The engine is ``interpretune.utils.notebook_publishing`` and ships as the console script
``interpretune-publish-notebooks``, parameterized from ``[tool.interpretune.notebooks]`` in ``pyproject.toml``, so an
adapter repository publishes its own example notebooks with the same engine and copies nothing.

Usage:
    python scripts/publish_notebooks.py [--dry-run] [--check-only] [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from interpretune.utils.notebook_publishing import main

    sys.exit(main(root=Path(__file__).resolve().parent.parent))
