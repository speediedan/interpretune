from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from interpretune.utils.notebook_experiments import bootstrap_experiment_imports


def bootstrap_notebook_imports(
    cwd: Path | None = None,
    *,
    extra_paths: Iterable[str | Path] | None = None,
) -> SimpleNamespace:
    """Put this repository's root and any declared harness paths on ``sys.path`` for a notebook.

    A thin adapter over the shared rails, kept so in-tree notebooks importing this name keep working.
    The rails themselves live in the core package, so an out-of-tree experiment uses them directly
    rather than reaching into `it_examples`.

    Two behaviours changed with the move, both deliberate. The root is found by walking up to the
    nearest `pyproject.toml` rather than assuming the notebook sits two levels below it, and the
    unconditional `tests/` and `tests/nb_experiments/` appends are gone: the latter no longer exists
    even here, which is the argument against encoding a layout instead of reading one.
    """
    config = bootstrap_experiment_imports(cwd, extra_paths=extra_paths)
    return SimpleNamespace(
        repo_root=config.root,
        harness_paths=config.harness_paths,
        working_dir=(cwd or Path.cwd()).resolve(),
    )
