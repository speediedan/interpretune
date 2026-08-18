"""Publishability lint for the bundled op families.

Bundled ops must follow the same compositional contract we require of hub op collections: no family
may carry a privileged latent dependency on interpretune internals of the kind ``definitions.py``
carried on ``helpers.py``. Concretely, a family module may only import:

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
interpretune) is a latent privileged dependency and fails this lint. Sanctioned modules may be
imported only for their **public** names: an underscore-prefixed import from interpretune is the same
defect at name granularity, so whatever a family genuinely needs is promoted in the sanctioned
surface instead. The family YAMLs are held to the matching rule:
``implementation``/``importable_params`` may reference only the family's own module or
``interpretune.analysis.optools``.

Scope limits (deliberate, so this lint is not read as more than it proves):

- ``required_ops`` resolution is NOT checked here, and real cross-family coupling exists today
  (``sae`` and ``attribution`` both require ``get_answer_indices``, which only ``core`` defines;
  ``attribution`` also requires ``get_alive_latents``, which only ``sae`` defines). An unresolvable
  ``required_ops`` entry is dropped with a warning by
  ``AnalysisOpDispatcher._compile_required_ops_schemas`` rather than failing loudly.
- The ``implementation`` grammar this lint mandates (fully-qualified interpretune paths) is not the
  grammar the hub loader accepts: ``get_function_from_dynamic_module`` parses a bare
  ``module.function`` pair, so publishing a family verbatim would require rewriting those paths.
- ``composites.yaml`` is skipped: compositions reference ops by name and cross family boundaries.

So this lint establishes the *import* half of the contract (no privileged latent dependency), not
literal publish-as-is liftability. Closing the remaining gaps is tracked in issue #266.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml

from interpretune.analysis.ops.collection import COLLECTION_HEADER_KEY

BUNDLED_ROOT = Path(__file__).parent.parent.parent / "src" / "interpretune" / "analysis" / "ops" / "bundled"
BUNDLED_PKG = "interpretune.analysis.ops.bundled"

SANCTIONED_IT_PREFIXES = (
    "interpretune.analysis.optools",
    "interpretune.analysis.backends",
    "interpretune.analysis.inputs",
    "interpretune.analysis.ops.base",
    "interpretune.protocol",
)

# Checked BEFORE the prefix allow above, and deliberately an exclusion list rather than a narrowed
# prefix: `SANCTIONED_IT_PREFIXES` matches by prefix, so every submodule of a sanctioned package is
# sanctioned by default. `backends.impls` holds the concrete per-library backends, which an op must
# consume through the protocols and capability helpers rather than importing -- importing one is the
# backend entanglement the seam exists to prevent, and it would pass the prefix check unnoticed.
# Expressed as an exclusion so that adding a future seam module cannot silently re-open it.
UNSANCTIONED_IT_PREFIXES = ("interpretune.analysis.backends.impls",)
ALLOWED_THIRD_PARTY = {"torch", "transformers", "jaxtyping", "transformer_lens"}
STDLIB = set(sys.stdlib_module_names)


def _is_unsanctioned_it_module(module_name: str) -> bool:
    """Whether an interpretune module is explicitly denied to op implementations."""
    return any(module_name == p or module_name.startswith(p + ".") for p in UNSANCTIONED_IT_PREFIXES)


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
    return [(module, level, fn_level) for module, level, fn_level, _ in _collect_imports_with_names(tree)]


def _collect_imports_with_names(tree: ast.AST) -> list[tuple[str, int, bool, tuple[str, ...]]]:
    """Return (module_path, level, is_function_level, imported_names) for every import in the tree.

    ``imported_names`` is empty for plain ``import x`` statements (there the module path itself is
    the thing being checked) and carries the ``from x import a, b`` names otherwise.
    """
    imports: list[tuple[str, int, bool, tuple[str, ...]]] = []

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
                imports.append((alias.name, 0, self._depth > 0, ()))

        def visit_ImportFrom(self, node):
            imports.append((node.module or "", node.level, self._depth > 0, tuple(a.name for a in node.names)))

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
            if _is_unsanctioned_it_module(module_name):
                # Denied explicitly, and checked BEFORE the prefix allow: these live *under* a
                # sanctioned prefix, so the allow below would otherwise pass them.
                violations.append(f"{module_name} (concrete backend; use the backends seam instead)")
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


@pytest.mark.parametrize("module_path", _family_impl_modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_bundled_family_imports_no_private_names(module_path: Path):
    """A family may import from a sanctioned module, but only its PUBLIC names.

    The module allowlist above is necessary but not sufficient: a hub op collection cannot depend on
    ``interpretune.analysis.backends._select_top_feature_indices`` any more than the old
    ``definitions.py`` could legitimately depend on ``helpers.py``. Whatever a family genuinely needs
    gets promoted to a public name in the sanctioned surface instead.
    """
    family_prefix = f"{BUNDLED_PKG}.{module_path.parent.name}"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    violations: list[str] = []

    for module_name, level, _is_function_level, names in _collect_imports_with_names(tree):
        if level > 0 or module_name.startswith(family_prefix):
            # A family's own module may share privates within the family (a hub op file may too).
            continue
        if module_name.split(".")[0] != "interpretune":
            continue
        violations.extend(f"{module_name}.{name}" for name in names if name.startswith("_"))

    assert not violations, (
        f"{module_path.parent.name}/{module_path.name} imports private names from interpretune "
        f"(promote them in the sanctioned surface instead): {violations}"
    )


class TestConcreteBackendsAreDenied:
    """The ``backends.impls`` deny must actually bite, since a prefix match would otherwise allow it.

    ``interpretune.analysis.backends`` is sanctioned, and ``SANCTIONED_IT_PREFIXES`` matches by prefix, so every
    submodule of it is sanctioned by default -- including the concrete per-library backends that an op must never
    import. Without an explicit exclusion the reorg that moved them under ``impls/`` would be inert against this
    contract, so these tests drive the real check with synthetic family modules rather than asserting the constant.
    """

    @staticmethod
    def _family_module(tmp_path, source: str):
        family_dir = tmp_path / "synthetic_family"
        family_dir.mkdir()
        module_path = family_dir / "synthetic_ops.py"
        module_path.write_text(source)
        return module_path

    def test_importing_a_concrete_backend_is_rejected(self, tmp_path):
        module_path = self._family_module(
            tmp_path,
            "from interpretune.analysis.backends.impls.nnsight import NNsightModelBackend\n",
        )
        with pytest.raises(AssertionError, match="concrete backend"):
            test_bundled_family_module_imports_are_sanctioned(module_path)

    def test_importing_the_seam_is_still_allowed(self, tmp_path):
        module_path = self._family_module(
            tmp_path,
            "from interpretune.analysis.backends import require_analysis_backend, FeatureSelectionSpec\n"
            "from interpretune.analysis.backends.protocols import AnalysisBackend\n",
        )
        test_bundled_family_module_imports_are_sanctioned(module_path)

    def test_private_names_from_impls_are_also_rejected(self, tmp_path):
        """The private-name rule spans all interpretune modules, so it covers impls without its own deny."""
        module_path = self._family_module(
            tmp_path,
            "from interpretune.analysis.backends.impls.nnsight import _navigate_envoy\n",
        )
        with pytest.raises(AssertionError, match="private names"):
            test_bundled_family_imports_no_private_names(module_path)

    @pytest.mark.parametrize(
        "module_name, denied",
        [
            ("interpretune.analysis.backends.impls", True),
            ("interpretune.analysis.backends.impls.circuit_tracer", True),
            ("interpretune.analysis.backends", False),
            ("interpretune.analysis.backends.protocols", False),
            # Guards against expressing the deny as a substring or narrowed prefix.
            ("interpretune.analysis.backends.implicit_thing", False),
        ],
    )
    def test_deny_matches_on_path_boundaries(self, module_name, denied):
        assert _is_unsanctioned_it_module(module_name) is denied


@pytest.mark.parametrize("yaml_path", _family_yamls(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_bundled_family_yaml_references_are_sanctioned(yaml_path: Path):
    if yaml_path.name == "composites.yaml":
        pytest.skip("composites reference ops by name, not implementation paths")
    family_prefix = f"{BUNDLED_PKG}.{yaml_path.parent.name}"
    content = yaml.safe_load(yaml_path.read_text()) or {}
    violations: list[str] = []
    for op_name, op_def in content.items():
        if op_name in (COLLECTION_HEADER_KEY, "composite_operations") or not isinstance(op_def, dict):
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
            # `collection:` is family metadata, not an op; every family declares one, so treating it as an
            # op name made this fail on the second family with a misleading duplicate-op message.
            if op_name in (COLLECTION_HEADER_KEY, "composite_operations"):
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
