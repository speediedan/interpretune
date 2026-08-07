"""Sphinx configuration for interpretune documentation."""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

project = "interpretune"
author = "Daniel Dale"
copyright = "2023-2026, Daniel Dale"

try:
    from interpretune.__about__ import __version__ as release
except Exception:
    release = "0.1.0.dev0"
version = release

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    # myst_nb supersedes myst_parser (it activates it internally) and adds .ipynb as a source
    # suffix. Do NOT also list myst_parser — registering both raises an extension conflict.
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_member_order = "groupwise"
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "methods": True,
    "exclude-members": "_abc_impl",
}
autosectionlabel_prefix_document = True

# Optional adapter deps are mocked on doc builders (Read the Docs / the docs-build CI smoke) —
# set SPHINX_MOCK_REQUIREMENTS=1 there; leave unset locally where the real env is available.
_MOCK_PACKAGES = [
    # Only the OPTIONAL adapter stack is mocked; interpretune's base dependencies (torch CPU,
    # transformers, datasets, + light utils) are installed REAL in docs builds — the core import
    # graph evaluates their types/paths at import time in ways mocks cannot represent.
    "transformer_lens",
    "sae_lens",
    "sae_dashboard",
    "circuit_tracer",
    "nnsight",
    "finetuning_scheduler",
    "lightning",
    "umap",
    "plotly",
    "matplotlib",
]
_ON_DOC_BUILDER = os.environ.get("SPHINX_MOCK_REQUIREMENTS", "0") == "1" or os.environ.get("READTHEDOCS") == "True"
autodoc_mock_imports = _MOCK_PACKAGES if _ON_DOC_BUILDER else []

if autodoc_mock_imports:
    # The adapter config hierarchy deliberately avoids PEP 563 (jsonargparse postponed-annotation
    # resolution breaks on it), so class-body annotations like `SAEConfig | dict[str, Any]` are
    # EVALUATED at import — teach sphinx's mock objects union syntax so mocked adapter types can
    # participate (the resulting annotation value is typing.Any, which renders fine).
    from typing import Any as _Any

    from sphinx.ext.autodoc.mock import _MockObject

    _MockObject.__or__ = lambda self, other: _Any  # type: ignore[assignment]
    _MockObject.__ror__ = lambda self, other: _Any  # type: ignore[assignment]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "linkify"]
myst_heading_anchors = 3

# --- Example notebooks -------------------------------------------------------------------------
# NEVER execute notebooks at build time. The demos need bf16 CUDA, gated gemma weights, the
# git-pinned adapter stack (which this very config MOCKS), and for the local-dashboard paths a live
# Neuronpedia webapp + Postgres. Outputs are baked in ahead of time instead (see the staging step
# below), so the docs render whatever the artifact carries and RTD stays a pure-CPU build.
nb_execution_mode = "off"
nb_merge_streams = True

# The notebooks live outside `docs/source`, so they are STAGED into it at build time (the target
# `docs/source/notebooks/` is gitignored). Two sources, in priority order:
#   1. `docs/notebook_artifacts/` — executed copies WITH outputs, regenerated on a GPU host. These
#      are what a reader should see. Deliberately NOT shipped in the sdist (see MANIFEST.in).
#   2. `src/it_examples/notebooks/publish/` — the stripped, runnable notebooks users clone. These
#      are the fallback so the docs still build (code-only) before any executed copy exists.
# Keeping the shipped notebooks stripped is intentional: an executed copy is ~27x larger, outputs
# are base64 blobs that never delta-compress, and MANIFEST.in ships the publish tree to every
# `pip install`.
_NOTEBOOK_SOURCES = (
    _REPO_ROOT / "docs" / "notebook_artifacts",
    _REPO_ROOT / "src" / "it_examples" / "notebooks" / "publish",
)
_NOTEBOOK_STAGE_DIR = Path(__file__).parent / "notebooks"


def _load_notebook_docs_helpers():
    """Reuse the cell-exclusion rules from scripts/render_notebook_docs_artifacts.py.

    Imported rather than duplicated so DOCS_EXCLUDED_CELL_IDS has exactly one definition. That module's only non-stdlib
    import (papermill) is deferred into its execute() path, so importing it here is cheap and safe.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_it_notebook_docs", _REPO_ROOT / "scripts" / "render_notebook_docs_artifacts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_example_notebooks() -> None:
    """Copy example notebooks (and their sidecar assets) into the Sphinx source tree.

    Later sources do not overwrite earlier ones, so an executed artifact always wins over the stripped fallback for the
    same relative path.

    Docs-excluded cells (the commented-out ``install-deps`` installer) are stripped HERE rather than only in the
    artifact builder, because most notebooks have no executed artifact yet and therefore fall back to the publish lane,
    which the artifact builder never touches. Stripping at staging time makes the rule global: it holds for every
    notebook on the docs site regardless of which lane it came from, including notebooks added later.
    """
    import shutil

    if _NOTEBOOK_STAGE_DIR.exists():
        shutil.rmtree(_NOTEBOOK_STAGE_DIR)

    helpers = _load_notebook_docs_helpers()

    staged: set[Path] = set()
    for source_dir in _NOTEBOOK_SOURCES:
        if not source_dir.is_dir():
            continue
        for source_path in sorted(source_dir.rglob("*")):
            if source_path.is_dir() or source_path.name.startswith("."):
                continue
            relative_path = source_path.relative_to(source_dir)
            if relative_path in staged:
                continue
            target_path = _NOTEBOOK_STAGE_DIR / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.suffix == ".ipynb":
                notebook, _ = helpers.strip_docs_excluded_cells(helpers.load_notebook(source_path))
                helpers.save_notebook(notebook, target_path)
            else:
                shutil.copy2(source_path, target_path)
            staged.add(relative_path)


_stage_example_notebooks()

# NOTE: no explicit source_suffix — myst_parser registers .md itself (an explicit mapping breaks under myst-parser 5.x)
templates_path = ["_templates"]
exclude_patterns = ["_build"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Theme: vanilla pydata-sphinx-theme for the bootstrap. Swap point for the planned
# `interpretune_sphinx_theme` child-theme fork: change html_theme + add the package to
# requirements/docs.txt (see the docs plan in the maintainer's PR-prep notes).
html_theme = "pydata_sphinx_theme"
html_title = "Interpretune"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/images/logos/logo_interpretune.svg"
html_favicon = "_static/images/logos/icon_interpretune.svg"
html_theme_options = {
    "github_url": "https://github.com/speediedan/interpretune",
    "navbar_align": "content",
    "show_toc_level": 2,
    # 0 => every caption group in the sidebar starts collapsed. The index page carries a quickstart
    # instead of an inline copy of the navigation, so the sidebar is the single place nav lives.
    "show_nav_level": 0,
    "navigation_with_keys": False,
}
# Global site navigation in the primary sidebar on EVERY page (including the landing page):
# the theme's stock sidebar-nav-bs renders only the active top-level section (empty at the root),
# so we render the full toctree from depth 0 via a custom template.
html_sidebars = {"**": ["sidebar-nav-global.html"]}
html_context = {
    "github_user": "speediedan",
    "github_repo": "interpretune",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Existing docs carry informal cross-doc links; do not fail the build on nitpicks during the
# bootstrap wave (the docs-build CI smoke still fails on ERRORS). Tighten after the coherence pass.
suppress_warnings = [
    # Structural, not content defects:
    "autosectionlabel.*",  # include-stub pages re-register the legacy guides' section labels
    "myst.xref_missing",  # legacy guides carry GitHub-style ../src/...#L links (linkcheck covers rot)
    "ref.python",  # re-exported symbols (interpretune.analysis.X vs submodule X) are intentionally dual-pathed
    "sphinx_autodoc_typehints.forward_reference",  # AnalysisCfgProtocol fwd-refs (PEP 563 deliberately off)
    "autosummary.import_cycle",  # api.rst lists fully-qualified top-level modules by design
    # Example notebooks are authored for readers running them (Colab/Jupyter), where heading level
    # is a styling choice, not a document outline. Skipping a level or opening at H2 is deliberate
    # there; rewriting the markdown to satisfy Sphinx would churn artifacts users actually execute.
    "myst.header",
]

nitpicky = False
