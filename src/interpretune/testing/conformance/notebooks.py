"""The static notebook-form cases: a repository's published example notebooks are what its dev copies say they
are, and they can be read by a consumer who has the published package and nothing else.

Subclass ``NotebookFormConformance`` and set ``root`` (the directory holding the ``pyproject.toml`` whose
``[tool.interpretune.notebooks]`` table describes the layout). Nothing here executes a notebook; the adapter
repository's own CI does that. These cases are the ones that hold without a kernel: sync, shape, and imports.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any, ClassVar

import pytest

from interpretune.utils.notebook_publishing import (
    PublisherConfig,
    changed_files,
    load_notebook,
    load_notebook_hashes,
)

MAGIC_PREFIXES = ("%", "!", "?")


def code_cells(notebook: dict[str, Any]) -> list[str]:
    """Each code cell's source as one string, with IPython magics and shell lines blanked so ``ast`` can parse."""
    out = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else source
        out.append("\n".join("" if line.lstrip().startswith(MAGIC_PREFIXES) else line for line in text.splitlines()))
    return out


def imported_modules(notebook: dict[str, Any]) -> set[str]:
    """Top-level module names a notebook imports, from every parseable code cell."""
    names: set[str] = set()
    for text in code_cells(notebook):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue  # a cell that is not Python (a shell recipe, say) has no imports to check
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module.split(".")[0])
    return names


class NotebookFormConformance:
    """Subclass, set ``root``, and pytest does the rest."""

    root: ClassVar[Path]
    forbidden_imports: ClassVar[tuple[str, ...]] = ()
    """Top-level modules a published notebook must not import directly: an adapter whose hub form is the subject
    of its notebooks lists its own pip-form package here, so the notebook demonstrates the hub path."""
    ignored_imports: ClassVar[tuple[str, ...]] = ()
    """Top-level modules the resolvability case skips (optional extras the notebook itself gates)."""

    @pytest.fixture(scope="class")
    def cfg(self, request) -> PublisherConfig:
        """The repository's publishing configuration, read from its pyproject."""
        return PublisherConfig.from_pyproject(Path(request.cls.root))

    @pytest.fixture(scope="class")
    def published(self, cfg) -> list[Path]:
        """Every published notebook."""
        notebooks = sorted(cfg.publish_dir.rglob("*.ipynb"))
        assert notebooks, f"no published notebooks under {cfg.publish_dir}"
        return notebooks

    def test_published_copies_are_in_sync_with_dev(self, cfg):
        """Every dev file's stored hash matches its content: nothing was edited without republishing."""
        stale = changed_files(cfg, load_notebook_hashes(cfg.publish_dir))
        assert not stale, "run interpretune-publish-notebooks; out of date: " + ", ".join(
            cfg.hash_key(p) for p in stale
        )

    def test_every_dev_notebook_has_a_published_copy(self, cfg, published):
        """No dev notebook is missing from the publish tree."""
        dev = {p.relative_to(cfg.dev_dir) for p in cfg.dev_dir.rglob("*.ipynb") if "__pycache__" not in p.parts}
        pub = {p.relative_to(cfg.publish_dir) for p in published}
        assert dev <= pub, f"dev notebooks without a published copy: {sorted(str(p) for p in dev - pub)}"

    def test_published_notebooks_open_with_the_badge_and_install_cell(self, cfg, published):
        """The two prepended cells lead, and the badge opens this repository at the configured branch."""
        for path in published:
            cells = load_notebook(path).get("cells", [])
            ids = [c.get("metadata", {}).get("id") for c in cells[:2]]
            assert ids == ["colab-badge", "install-deps"], f"{path.name}: leading cells are {ids}"
            badge = "".join(cells[0]["source"])
            assert f"github/{cfg.colab_repo}/blob/{cfg.colab_branch}/" in badge, f"{path.name}: badge points elsewhere"

    def test_published_notebooks_keep_no_remove_cell_tags(self, published):
        """Cells tagged for removal were removed, not merely tagged."""
        for path in published:
            tagged = [
                i
                for i, c in enumerate(load_notebook(path).get("cells", []))
                if "remove-cell" in (c.get("metadata", {}).get("tags") or [])
            ]
            assert not tagged, f"{path.name}: cells {tagged} carry remove-cell and were published anyway"

    def test_imports_resolve_in_this_environment(self, published):
        """Every top-level module a published notebook imports is importable here (the environment the adapter's CI
        runs the notebook in), except those the class lists as gated extras."""
        unresolved: dict[str, set[str]] = {}
        for path in published:
            missing = {
                name
                for name in imported_modules(load_notebook(path))
                if name not in self.ignored_imports and importlib.util.find_spec(name) is None
            }
            if missing:
                unresolved[path.name] = missing
        assert not unresolved, f"unresolvable imports: {unresolved}"

    def test_published_notebooks_avoid_the_forbidden_imports(self, published):
        """Where the hub form is the subject, the notebook must not import the pip form directly."""
        if not self.forbidden_imports:
            pytest.skip("no forbidden imports declared for this repository")
        hits = {
            path.name: sorted(imported_modules(load_notebook(path)) & set(self.forbidden_imports)) for path in published
        }
        hits = {k: v for k, v in hits.items() if v}
        assert not hits, f"published notebooks import the pip form directly where the hub form is the subject: {hits}"


__all__ = ["NotebookFormConformance", "code_cells", "imported_modules"]
