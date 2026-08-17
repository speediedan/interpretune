"""Publishability lint for the bundled op families.

Bundled ops must follow the same compositional contract we require of hub op collections: each
family implementation module must be liftable into a standalone ``kinds: [ops]`` hub repo
unchanged. Concretely, a family module may only import:

- the sanctioned op-authoring surfaces: ``interpretune.analysis.optools``,
  ``interpretune.analysis.backends``, ``interpretune.analysis.inputs``, and the public op classes
  in ``interpretune.analysis.ops.base`` (``AnalysisBatch``/``OpSchema``/``ColCfg`` and
  ``get_batch_input``),
- ``interpretune.protocol`` types,
- the bare public ``interpretune`` surface (function-level only), used to invoke ops the family's
  YAML declares via ``required_ops`` (see NOTE [Op-Driven Transitive Dependency Atomicity]),
- its own family module (relative or absolute),
- the standard library and the declared third-party op dependencies (torch, transformers,
  jaxtyping, transformer_lens).

Anything else (another family, runners, config internals, private modules elsewhere in
interpretune) is a latent privileged dependency and fails this lint. The family YAMLs are held to
the matching rule: ``implementation``/``importable_params`` may reference only the family's own
module or ``interpretune.analysis.optools``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml

BUNDLED_ROOT = Path(__file__).parent.parent.parent / "src" / "interpretune" / "analysis" / "ops" / "bundled"
BUNDLED_PKG = "interpretune.analysis.ops.bundled"

SANCTIONED_IT_PREFIXES = (
    "interpretune.analysis.optools",
    "interpretune.analysis.backends",
    "interpretune.analysis.inputs",
    "interpretune.analysis.ops.base",
    "interpretune.protocol",
)
ALLOWED_THIRD_PARTY = {"torch", "transformers", "jaxtyping", "transformer_lens"}
STDLIB = set(sys.stdlib_module_names)


def _family_impl_modules() -> list[Path]:
    modules = sorted(p for p in BUNDLED_ROOT.glob("*/*.py") if p.name != "__init__.py")
    assert modules, f"no bundled family impl modules found under {BUNDLED_ROOT}"
    return modules


def _family_yamls() -> list[Path]:
    yamls = sorted(BUNDLED_ROOT.glob("*/*.yaml"))
    assert yamls, f"no bundled family YAMLs found under {BUNDLED_ROOT}"
    return yamls


def _collect_imports(tree: ast.AST) -> list[tuple[str, int, bool]]:
    """Return (module_path, level, is_function_level) for every import in the tree."""
    imports: list[tuple[str, int, bool]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._depth = 0

        def visit_FunctionDef(self, node):
            self._depth += 1
            self.generic_visit(node)
            self._depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for alias in node.names:
                imports.append((alias.name, 0, self._depth > 0))

        def visit_ImportFrom(self, node):
            imports.append((node.module or "", node.level, self._depth > 0))

    Visitor().visit(tree)
    return imports


@pytest.mark.parametrize("module_path", _family_impl_modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_bundled_family_module_imports_are_sanctioned(module_path: Path):
    family_prefix = f"{BUNDLED_PKG}.{module_path.parent.name}"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    violations: list[str] = []

    for module_name, level, is_function_level in _collect_imports(tree):
        if level > 0:
            # Relative imports resolve within the family package and are fine.
            continue
        top = module_name.split(".")[0]
        if top in STDLIB or top in ALLOWED_THIRD_PARTY:
            continue
        if top == "interpretune":
            if module_name == "interpretune":
                # Bare public surface: sanctioned for declared required_ops calls, and only
                # inside a function body so import-time cycles stay impossible.
                if not is_function_level:
                    violations.append("module-level 'import interpretune' (must be function-level)")
                continue
            if module_name.startswith(family_prefix) or any(
                module_name == p or module_name.startswith(p + ".") for p in SANCTIONED_IT_PREFIXES
            ):
                continue
            violations.append(module_name)
            continue
        violations.append(module_name)

    assert not violations, (
        f"{module_path.parent.name}/{module_path.name} imports unsanctioned modules "
        f"(latent privileged dependencies): {violations}"
    )


@pytest.mark.parametrize("yaml_path", _family_yamls(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_bundled_family_yaml_references_are_sanctioned(yaml_path: Path):
    if yaml_path.name == "composites.yaml":
        pytest.skip("composites reference ops by name, not implementation paths")
    family_prefix = f"{BUNDLED_PKG}.{yaml_path.parent.name}"
    content = yaml.safe_load(yaml_path.read_text()) or {}
    violations: list[str] = []
    for op_name, op_def in content.items():
        if op_name == "composite_operations" or not isinstance(op_def, dict):
            continue
        refs = [op_def.get("implementation", "")]
        refs.extend((op_def.get("importable_params") or {}).values())
        for ref in refs:
            if not ref:
                continue
            if ref.startswith(family_prefix + ".") or ref.startswith("interpretune.analysis.optools."):
                continue
            violations.append(f"{op_name}: {ref}")
    assert not violations, (
        f"{yaml_path.parent.name}/{yaml_path.name} references implementations outside its own "
        f"family module or optools: {violations}"
    )


def test_bundled_yaml_op_names_are_globally_unique():
    seen: dict[str, Path] = {}
    for yaml_path in _family_yamls():
        content = yaml.safe_load(yaml_path.read_text()) or {}
        for op_name in content:
            if op_name == "composite_operations":
                continue
            assert op_name not in seen, f"op '{op_name}' defined in both {seen[op_name]} and {yaml_path}"
            seen[op_name] = yaml_path
    assert seen, "expected at least one bundled op definition"


def _single_sourced_all(module_path: Path) -> tuple[list[str], set[str]]:
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    all_assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    ]
    assert len(all_assignments) == 1, (
        f"{module_path.name} must define __all__ exactly once (found {len(all_assignments)}); "
        "a second binding silently shadows the first"
    )
    exported = [ast.literal_eval(elt) for elt in all_assignments[0].value.elts]  # type: ignore[attr-defined]
    public_defs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    }
    public_constants = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and not target.id.startswith("_") and target.id != "__all__"
    }
    return exported, public_defs | public_constants


@pytest.mark.parametrize(
    "module_rel",
    ["analysis/optools.py", "analysis/inputs.py", "analysis/ops/bundled/concept/concept_ops.py"],
)
def test_sanctioned_surface_all_is_single_sourced_and_complete(module_rel: str):
    module_path = Path(__file__).parent.parent.parent / "src" / "interpretune" / module_rel
    exported, public_names = _single_sourced_all(module_path)
    assert len(exported) == len(set(exported)), f"duplicate names in {module_rel} __all__"
    missing_from_all = public_names - set(exported)
    if module_rel.endswith("concept_ops.py"):
        # concept_ops defines type aliases via imports only; every module-level public def/const
        # is part of its exported surface.
        pass
    assert not missing_from_all, f"{module_rel}: public names missing from __all__: {sorted(missing_from_all)}"
    phantom = set(exported) - public_names
    assert not phantom, f"{module_rel}: __all__ exports names not defined at module level: {sorted(phantom)}"
