from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


def bootstrap_notebook_imports(
    cwd: Path | None = None,
    *,
    extra_paths: Iterable[str | Path] | None = None,
) -> SimpleNamespace:
    """Ensure experiment notebooks can import the repo, tests, and shared harness modules."""

    resolved_cwd = (cwd or Path.cwd()).resolve()
    repo_root = resolved_cwd if (resolved_cwd / "pyproject.toml").exists() else resolved_cwd.parents[1]
    tests_dir = repo_root / "tests"
    harness_dir = tests_dir / "nb_experiment_harness"

    path_candidates = [repo_root, tests_dir, harness_dir, resolved_cwd]
    if extra_paths is not None:
        path_candidates.extend(Path(path).expanduser().resolve() for path in extra_paths)

    for path in path_candidates:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return SimpleNamespace(
        repo_root=repo_root,
        tests_dir=tests_dir,
        harness_dir=harness_dir,
        working_dir=resolved_cwd,
    )
