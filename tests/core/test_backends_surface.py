"""Contract tests for the ``interpretune.analysis.backends`` façade.

``analysis/backends/__init__.py`` is a re-export façade: the seam's implementation lives in named
modules (``interventions``, ``feature_selection``, ``capabilities``, ``protocols``) and the package
``__all__`` is the public surface op authors are told to rely on. These tests pin the two invariants
that make that arrangement safe to extend.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

BACKENDS_DIR = Path(__file__).parent.parent.parent / "src" / "interpretune" / "analysis" / "backends"
SEAM_MODULES = ("interventions", "feature_selection", "capabilities", "protocols")


def _public_module_level_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.add(target.id)
    return names


@pytest.mark.parametrize("seam_module", SEAM_MODULES)
def test_seam_module_public_names_are_reexported(seam_module: str):
    """Every public name defined in a seam module must be reachable from the façade.

    Without this, adding a public helper to (say) ``feature_selection.py`` and forgetting the
    ``__init__`` re-export produces a name that is public by convention but unreachable by the
    documented import path.
    """
    backends = importlib.import_module("interpretune.analysis.backends")
    defined = _public_module_level_names(BACKENDS_DIR / f"{seam_module}.py")
    missing = sorted(name for name in defined if name not in set(backends.__all__))
    assert not missing, f"backends/{seam_module}.py defines public names missing from the package __all__: {missing}"


def test_facade_all_resolves_and_is_deduplicated():
    backends = importlib.import_module("interpretune.analysis.backends")
    exported = list(backends.__all__)
    assert len(exported) == len(set(exported)), "duplicate names in interpretune.analysis.backends.__all__"
    unresolvable = [name for name in exported if not hasattr(backends, name)]
    assert not unresolvable, f"__all__ names not importable from the façade: {unresolvable}"


@pytest.mark.parametrize("seam_module", SEAM_MODULES)
def test_seam_modules_do_not_register_as_named_backends(seam_module: str):
    """The seam modules must stay inert with respect to by-name backend resolution.

    ``resolve_analysis_backend`` treats ``backends.<name>`` as a lookup namespace: on a registry miss
    it imports ``interpretune.analysis.backends.<name>`` hoping the module self-registers. The four
    seam modules are siblings in that namespace, so importing one must not mutate the registry (and
    resolving one by name must still raise).
    """
    from interpretune.analysis.backends import ANALYSIS_BACKEND_REGISTRY, resolve_analysis_backend

    before = dict(ANALYSIS_BACKEND_REGISTRY)
    importlib.import_module(f"interpretune.analysis.backends.{seam_module}")
    assert dict(ANALYSIS_BACKEND_REGISTRY) == before, f"importing backends.{seam_module} mutated the backend registry"
    with pytest.raises(KeyError):
        resolve_analysis_backend(seam_module)
