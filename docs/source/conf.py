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
    "myst_parser",
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
html_title = "interpretune"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/speediedan/interpretune",
    "navbar_align": "content",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
html_context = {
    "github_user": "speediedan",
    "github_repo": "interpretune",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Existing docs carry informal cross-doc links; do not fail the build on nitpicks during the
# bootstrap wave (the docs-build CI smoke still fails on ERRORS). Tighten after the coherence pass.
nitpicky = False
