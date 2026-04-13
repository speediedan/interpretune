from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def bootstrap_notebook_imports(cwd: Path | None = None) -> SimpleNamespace:
    """Ensure the concept-direction notebook can import the repo, tests, and harness modules."""

    resolved_cwd = (cwd or Path.cwd()).resolve()
    repo_root = resolved_cwd if (resolved_cwd / "pyproject.toml").exists() else resolved_cwd.parents[1]
    tests_dir = repo_root / "tests"
    harness_dir = (
        resolved_cwd
        if (resolved_cwd / "concept_direction_experiment_utils.py").exists()
        else tests_dir / "concept_direction_approach_parity"
    )
    for path in (repo_root, tests_dir, harness_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return SimpleNamespace(repo_root=repo_root, tests_dir=tests_dir, harness_dir=harness_dir)