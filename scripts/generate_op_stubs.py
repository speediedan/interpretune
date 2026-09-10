#!/usr/bin/env python
"""Generate type stubs for analysis operations to improve IDE support."""

import ast
import sys
import inspect
import importlib
from pathlib import Path
import yaml
from typing import Dict, Any, Callable, List, Union

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from interpretune.analysis.ops.collection import COLLECTION_HEADER_KEY  # noqa: E402  (needs project_root)


def import_callable(callable_path: str) -> Callable:
    """Import a callable from a path."""
    module_path, func_name = callable_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def format_type_annotation(annotation):
    """Format type annotation for stub file."""
    if annotation is inspect.Parameter.empty:
        return ""

    # Handle common types directly
    if annotation is None:
        return "None"
    if isinstance(annotation, type) and hasattr(annotation, "__name__"):
        return annotation.__name__

    # Handle complex typing constructs
    import typing

    if hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        if origin is Union:
            return f"Union[{', '.join(format_type_annotation(arg) for arg in args)}]"
        elif origin is list:
            if args:
                return f"List[{format_type_annotation(args[0])}]"
            return "List"
        elif origin is dict:
            if len(args) == 2:
                return f"Dict[{format_type_annotation(args[0])}, {format_type_annotation(args[1])}]"
            return "Dict"
        elif origin:
            formatted_args = ", ".join(format_type_annotation(arg) for arg in args)
            return f"{origin.__name__}[{formatted_args}]"

    # Default: convert to string and clean up formats
    return str(annotation).replace("<class '", "").replace("'>", "").replace("<", "[").replace(">", "]")


def format_schema_doc(schema_dict: Dict) -> str:
    """Format schema dictionary into a readable docstring section.

    Accepts both the raw YAML shape (field -> mapping) and the compiled shape (``OpSchema`` of ``ColCfg``).
    The collection-stub path sources definitions from the dispatcher, so its fields arrive as ``ColCfg``
    dataclasses; reading only mappings silently emitted every schema section empty.
    """
    if not schema_dict:
        return ""

    lines = []
    for field_name, field_def in schema_dict.items():
        if isinstance(field_def, dict):
            dtype, required = field_def.get("datasets_dtype"), field_def.get("required")
        elif hasattr(field_def, "datasets_dtype"):
            dtype, required = field_def.datasets_dtype, getattr(field_def, "required", None)
        else:
            continue
        field_str = f"{field_name}"
        if dtype:
            field_str += f" ({dtype})"
        if required:
            field_str += " (required)"
        lines.append(field_str)

    return "\n".join(lines)


def wrap_signature(name: str, params: List[str], return_type: str = "", max_width: int = 120) -> str:
    """Generate a properly wrapped function signature."""
    signature = f"def {name}("

    # Always format with one parameter per line for consistency
    if params:
        signature += "\n"
        for i, param in enumerate(params):
            if i < len(params) - 1:
                signature += f"    {param},\n"
            else:
                signature += f"    {param}\n"
        signature += ")"
    else:
        signature += ")"

    # Add return type if provided
    if return_type:
        signature += f" -> {return_type}"

    return signature


def format_docstring(
    description: str, input_schema: Dict, output_schema: Dict, function_param_defaults: Dict[str, str] | None = None
) -> str:
    """Format a docstring with proper wrapping and sections."""
    doc_lines = [f'"""{description}']

    if input_schema:
        doc_lines.append("\n    Input Schema:")
        schema_doc = format_schema_doc(input_schema)
        if schema_doc:
            # Add proper indentation to each line
            indented_schema = "\n".join(f"        {line}" for line in schema_doc.split("\n"))
            doc_lines.append(indented_schema)

    if output_schema:
        doc_lines.append("\n    Output Schema:")
        schema_doc = format_schema_doc(output_schema)
        if schema_doc:
            # Add proper indentation to each line
            indented_schema = "\n".join(f"        {line}" for line in schema_doc.split("\n"))
            doc_lines.append(indented_schema)

    # Document any function-parameter defaults that were present in the YAML (FQ callable paths).
    if function_param_defaults:
        doc_lines.append("\n    Function parameter defaults (from YAML):")
        for param_name, fq_path in function_param_defaults.items():
            doc_lines.append(f"        {param_name}: {fq_path}")

    doc_lines.append('"""')
    return "\n".join(doc_lines)


def alias_assignments(op_name: str, op_def: Dict[str, Any]) -> List[str]:
    """Alias lines for an op, always under a BARE name.

    A hub op's aliases are namespaced, and ``user.repo.alias = op`` is not valid Python -- it reads as an
    attribute assignment on ``user``. Shared by the introspected and YAML-derived paths so a collection whose
    implementation cannot be imported does not silently lose its aliases too.
    """
    aliases = []
    for alias in op_def.get("aliases") or []:
        bare_alias = alias.split(".")[-1]
        if bare_alias != op_name:
            aliases.append(f"{bare_alias} = {op_name}")
    single = op_def.get("alias")
    if single and single.split(".")[-1] != op_name:
        aliases.append(f"{single.split('.')[-1]} = {op_name}")
    return aliases


def yaml_derived_stub(op_name: str, op_def: Dict[str, Any], reason: str) -> str:
    """Stub for an op whose implementation could not be imported, derived from its YAML declaration.

    Replaces a fallback that emitted a bare untyped signature with no docstring beyond the failure text, which
    degraded silently: the op still appeared in the stub, so nothing downstream could tell a YAML-derived stub
    from a real one, and the schema documentation was lost precisely when introspection was least available.

    The signature stays the canonical op-protocol shape rather than being invented from ``input_schema``. The
    schema names an op's DATA contract, not its Python parameters, so synthesizing parameters from it would
    produce a stub that type-checks calls the runtime rejects -- worse than a conservative one.
    """
    docstring = format_docstring(
        op_def.get("description", ""),
        op_def.get("input_schema", {}),
        op_def.get("output_schema", {}),
        op_def.get("importable_params") or None,
    )
    signature = wrap_signature(
        op_name,
        ["module", "analysis_batch: Optional[BaseAnalysisBatchProtocol]", "batch", "batch_idx: int", "**kwargs"],
        "BaseAnalysisBatchProtocol",
    )
    annotated = docstring.replace(
        '"""',
        f'"""[YAML-derived stub: {reason}] ',
        1,
    )
    stub = f"{signature}:\n    {annotated}\n    ...\n\n"
    if aliases := alias_assignments(op_name, op_def):
        stub += "\n".join(aliases) + "\n\n"
    return stub


def generate_operation_stub(
    op_name: str,
    op_def: Dict[str, Any],
    yaml_content: Dict[str, Any],
    require_importable: bool = True,
    resolve_impl: Callable | None = None,
) -> str:
    """Generate type stub for a single analysis operation.

    ``require_importable`` distinguishes the two callers. For the committed bundled stub an unimportable
    implementation is a genuine defect (the module ships in the wheel), so it raises rather than quietly
    emitting a degraded stub that the stale-stubs check would then happily accept. For local and hub
    collections the impl legitimately may not be importable in the generating environment, so those fall back
    to a YAML-derived stub that says so.

    ``resolve_impl`` overrides how the implementation is imported. A hub op's ``implementation`` is a
    repo-relative ``<module>.<function>`` pair resolved through the dynamic-module machinery, so a plain
    ``import_callable`` cannot find it and EVERY hub op would degrade to a YAML-derived stub -- which is
    exactly the silent degradation this pass exists to remove.
    """
    try:
        # Import the implementation function
        impl_path = op_def["implementation"]
        func = resolve_impl(op_name, op_def) if resolve_impl is not None else import_callable(impl_path)

        # Get function signature
        sig = inspect.signature(func)

        # Create parameters list
        params = []
        # Collect function-parameter defaults to document them in the docstring
        function_param_defaults: Dict[str, str] = {}
        for name, param in sig.parameters.items():
            annotation = format_type_annotation(param.annotation)
            if annotation:
                annotation = f": {annotation}"

            # Handle *args and **kwargs
            prefix = ""
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                prefix = "*"
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                prefix = "**"

            default = ""
            if param.default is not param.empty:
                # Check if this parameter has a corresponding importable_param in the YAML definition
                if "importable_params" in op_def and name in op_def["importable_params"]:
                    # DO NOT emit the FQ path as the default in the stub (string default breaks type checkers).
                    # Instead, set default to ... and record the FQ path for documentation in the docstring.
                    default = " = ..."
                    function_param_defaults[name] = op_def["importable_params"][name]
                elif param.default is None:
                    default = " = None"
                elif isinstance(param.default, str):
                    default = f" = '{param.default}'"
                else:
                    default = f" = {param.default}"

            params.append(f"{prefix}{name}{annotation}{default}")

        # Get return type
        return_type = format_type_annotation(sig.return_annotation)

        # Create function signature
        signature = wrap_signature(op_name, params, return_type)

        # Create formatted docstring (include the recorded function_param_defaults)
        docstring = format_docstring(
            op_def.get("description", ""),
            op_def.get("input_schema", {}),
            op_def.get("output_schema", {}),
            function_param_defaults or None,
        )

        # Build the complete stub
        stub = f"{signature}:\n    {docstring}\n    ...\n\n"

        aliases = alias_assignments(op_name, op_def)
        if aliases:
            stub += "\n".join(aliases) + "\n\n"

        return stub

    except (ImportError, AttributeError, ValueError) as e:
        if require_importable:
            raise RuntimeError(
                f"Cannot generate a stub for bundled op {op_name!r}: its implementation "
                f"{op_def.get('implementation')!r} could not be imported ({type(e).__name__}: {e}). A bundled "
                "implementation ships in the wheel, so this is a defect rather than a degraded environment; "
                "emitting a fallback stub here would let the stale-stubs check pass over it."
            ) from e
        print(f"Falling back to a YAML-derived stub for {op_name}: {type(e).__name__}: {e}")
        return yaml_derived_stub(op_name, op_def, f"{op_def.get('implementation')} not importable: {e}")


def composite_protocol_name(op_def: Dict[str, Any], definitions: Dict[str, Any] | None = None) -> str:
    """The protocol a composite's stub should name, in descending order of what is actually known (#60).

    A composite has no single implementation to introspect, which is why this was hardcoded to the base
    protocol. Two better answers exist now:

    1. a **declared** ``protocol_cls`` (#56) -- an explicit answer beats an inferred one;
    2. the protocol its constituents agree on -- they are the ops that actually populate the batch, so if
       every one of them names the same protocol the composite returns that too.

    Falls back to ``BaseAnalysisBatchProtocol`` when the constituents disagree or cannot be resolved.
    Deliberately does NOT assert ``Default`` everywhere: naming a protocol the batch may not satisfy is
    worse than naming the weaker one that it certainly does.
    """
    if declared := op_def.get("protocol_cls"):
        return str(declared).rpartition(".")[2]

    composition = op_def.get("composition") or []
    if isinstance(composition, str):
        composition = composition.split(".")
    if not composition or definitions is None:
        return "BaseAnalysisBatchProtocol"

    named = set()
    for member in composition:
        member_def = definitions.get(member)
        if member_def is None:
            return "BaseAnalysisBatchProtocol"  # unresolvable constituent: do not guess for the whole
        if member_declared := member_def.get("protocol_cls"):
            named.add(str(member_declared).rpartition(".")[2])
        else:
            named.add("DefaultAnalysisBatchProtocol")
    return named.pop() if len(named) == 1 else "BaseAnalysisBatchProtocol"


def generate_composition_stub(op_name: str, op_def: Dict[str, Any], definitions: Dict[str, Any] | None = None) -> str:
    """Generate type stub for a composite operation."""
    composition = op_def.get("composition", "")
    composition_str = composition if isinstance(composition, str) else ".".join(composition)
    protocol = composite_protocol_name(op_def, definitions)

    signature = wrap_signature(
        op_name,
        ["module", f"analysis_batch: Optional[{protocol}]", "batch", "batch_idx: int"],
        protocol,
    )

    # A composite's compiled schemas are already merged onto its OpDef, so it can document the same
    # Input/Output sections a simple op gets rather than only naming its constituents. `format_docstring`
    # is the same renderer the introspected path uses, so the two stay consistent by construction.
    description = f"Composition of operations:\n    {composition_str}"
    if "description" in op_def:
        description += f"\n\n    {op_def['description']}"
    doc = "    " + format_docstring(
        description,
        op_def.get("input_schema", {}) or {},
        op_def.get("output_schema", {}) or {},
    )

    stub = f"{signature}:\n{doc}\n    ...\n\n"

    if "alias" in op_def and op_def["alias"] != op_name:
        stub += f"{op_def['alias']} = {op_name}\n\n"

    return stub


def load_bundled_definitions(yaml_paths: List[Path]) -> Dict[str, Any]:
    """Load and merge the bundled op-family YAMLs into a single definitions mapping.

    Non-op top-level keys are skipped. Every family declares a ``collection:`` header, and treating it as an
    op made the generator fail on its second family with a misleading "Duplicate bundled op definition
    'collection'" -- the same shape of bug the header caused in the op compiler.
    """
    non_op_keys = {COLLECTION_HEADER_KEY}
    merged: Dict[str, Any] = {}
    for yaml_path in yaml_paths:
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        for op_name, op_def in content.items():
            if op_name in non_op_keys:
                continue
            if op_name == "composite_operations":
                merged.setdefault("composite_operations", {}).update(op_def)
            else:
                if op_name in merged:
                    raise ValueError(f"Duplicate bundled op definition '{op_name}' in {yaml_path}")
                merged[op_name] = op_def
    return merged


def _binds_statically(tree: ast.Module, attr: str) -> bool:
    """Whether ``attr`` is bound in this module by something a type checker can see."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == attr:
            return True
        if isinstance(node, ast.ImportFrom) and any((a.asname or a.name) == attr for a in node.names):
            return True
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == attr for t in node.targets):
            return True
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == attr:
            return True
    return False


def _follow_lazy_reexports(init_path: Path, module: str, attr: str, _limit: int = 8) -> tuple:
    """Follow ``module.attr`` through any package that re-exports it lazily, to the one that defines it.

    A PEP 562 package binds nothing statically, so a type checker cannot resolve ``from that_package import
    X`` at all. `interpretune.adapters` is one, and eight public names route through it, so pointing the stub
    at the intermediate package produces re-exports that read fine and resolve to nothing. Followed to the
    defining module instead.
    """
    package_root = init_path.parent.parent  # .../src
    for _ in range(_limit):
        candidate = package_root.joinpath(*module.split(".")) / "__init__.py"
        if not candidate.is_file():
            return module, attr
        try:
            sub = ast.parse(candidate.read_text())
        except SyntaxError:
            return module, attr
        # The map's NAME varies by package (`_LAZY_MODULE_ATTRS`, `_LAZY_ADAPTER_ATTRS`), so match the
        # convention rather than one spelling. Getting this wrong is silent: the follower returns the
        # intermediate package unchanged and the stub re-exports from something that binds nothing.
        mapping = {}
        lazy_package = False
        for node in ast.walk(sub):
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                lazy_package = True
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for target in node.targets:
                    tid = getattr(target, "id", "")
                    if tid.startswith("_LAZY_") and tid.endswith("_ATTRS"):
                        mapping.update({k.value: v.value for k, v in zip(node.value.keys, node.value.values)})
        if attr not in mapping:
            # A package can be lazy for SOME names and bind others statically (`interpretune.config`
            # imports ITConfig outright while exporting adapter configs lazily). A static binding
            # resolves for a type checker, so only an attr with NEITHER is unreachable.
            if lazy_package and not _binds_statically(sub, attr):
                # A package that binds nothing statically cannot satisfy `from <pkg> import <attr>` for a
                # type checker. Emitting it anyway is what produced ten unresolvable re-exports that every
                # AST-level check called present. Refuse so a renamed map fails loudly.
                raise RuntimeError(
                    f"{module!r} exports lazily and no `_LAZY_*_ATTRS` mapping in it names {attr!r}, so the "
                    "stub cannot point at a module that actually binds it. If that package's mapping was "
                    "renamed, teach `_follow_lazy_reexports` the new name."
                )
            return module, attr
        nxt = mapping[attr]
        nxt_module, _, nxt_attr = nxt.rpartition(".")
        if not nxt_module or (nxt_module, nxt_attr) == (module, attr):
            return module, attr
        module, attr = nxt_module, nxt_attr
    return module, attr


def public_surface_imports(init_path: Path) -> List[str]:
    """Re-export lines covering every name in ``interpretune.__all__``, derived rather than listed.

    A ``.pyi`` SHADOWS the module it names for a type checker rather than adding to it, so any public
    name the stub omits is invisible to a typed consumer even though it exists and is exported. This
    list was hand-maintained inside the generator and had drifted to 23 of 75.

    Read STATICALLY from ``__init__.py``'s own ``__all__``, ``_LAZY_MODULE_ATTRS`` and top-level
    imports, with no import of interpretune: resolving the lazy names would import every optional
    framework as a side effect of generating stubs, and the stale-stubs CI check has to stay hermetic
    and runnable without them.
    """
    tree = ast.parse(init_path.read_text())
    lazy: Dict[str, str] = {}
    exported: List[str] = []
    from_imports: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = getattr(target, "id", "")
                if name == "_LAZY_MODULE_ATTRS" and isinstance(node.value, ast.Dict):
                    lazy = {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
                elif name == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                    exported = [e.value for e in node.value.elts]
    # `ast.walk`, not `tree.body`: `__init__.py` imports much of the public surface inside an
    # `if TYPE_CHECKING:` block, which is a nested node. Reading only the top level silently misses those.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                from_imports.setdefault(alias.asname or alias.name, node.module)

    by_module: Dict[str, List[str]] = {}
    unresolved: List[str] = []
    for name in exported:
        if name in lazy:
            target = lazy[name]
            module, _, attr = target.rpartition(".")
            # A direct submodule (`it.hub`) is spelled `from interpretune import hub`, not
            # `from interpretune.hub import hub`, which would not resolve.
            module, attr = (module, attr) if attr == name else (target, name)
            module, attr = _follow_lazy_reexports(init_path, module, attr)
        elif name in from_imports:
            module, attr = from_imports[name], name
        else:
            unresolved.append(name)
            continue
        by_module.setdefault(module, []).append(attr)

    if unresolved:
        # Refused rather than skipped: a silently omitted name is exactly the defect this function
        # exists to remove, and it would be invisible in the generated file.
        raise RuntimeError(
            f"cannot resolve a module for these `__all__` names: {sorted(unresolved)}. Add them to "
            "`_LAZY_MODULE_ATTRS` or import them at the top of `interpretune/__init__.py`."
        )

    lines = [
        "# The public surface, DERIVED from `interpretune.__all__` rather than listed here.",
        "# A .pyi shadows the module it names, so every public name absent from this file is invisible",
        "# to a type checker even though it exists and is exported.",
    ]
    for module in sorted(by_module):
        names = sorted(set(by_module[module]))
        if len(names) == 1:
            lines.append(f"from {module} import {names[0]} as {names[0]}")
        else:
            lines.append(f"from {module} import (")
            lines.extend(f"    {n} as {n}," for n in names)
            lines.append(")")
    return lines


def generate_stubs(yaml_paths: Union[Path, List[Path]], output_path: Path) -> None:
    """Generate type stubs for all operations in the bundled op-family YAML files."""
    # Load YAML definitions (committed stubs are derived from the bundled op set only, so the
    # stale-stubs CI check stays hermetic and network-independent)
    if isinstance(yaml_paths, Path):
        yaml_paths = [yaml_paths]
    yaml_content = load_bundled_definitions(sorted(yaml_paths))

    # Start with header
    stubs = [
        '"""Type stubs for Interpretune analysis operations."""',
        "# This file is auto-generated. Do not modify directly.",
        "",
        "from typing import Callable, Optional",
        "import torch",
        "from transformers import BatchEncoding",
        # `BaseAnalysisBatchProtocol` / `DefaultAnalysisBatchProtocol` are NOT imported here even though the
        # op signatures below use them. Both are in `__all__`, so the derived public surface immediately
        # below emits them in the `X as X` form a stub needs to re-export a name. Importing them here too
        # made the generator emit each twice; ruff then removed one copy on commit, so the committed file
        # stopped matching a fresh generation and the stale-stubs check went red on a no-op.
        "",
    ]
    stubs.extend(public_surface_imports(Path(__file__).parent.parent / "src" / "interpretune" / "__init__.py"))
    stubs.extend(
        [
            "",
            "# Basic operations",
            "",
        ]
    )

    # Process individual operations
    for op_name, op_def in sorted(yaml_content.items()):
        # Skip composite operations section
        if op_name == "composite_operations":
            continue

        op_stub = generate_operation_stub(op_name, op_def, yaml_content)
        stubs.append(op_stub)

    # Process composite operations
    if "composite_operations" in yaml_content:
        stubs.append("# Composite operations\n")
        comp_ops = yaml_content["composite_operations"]
        for op_name, op_def in sorted(comp_ops.items()):
            op_stub = generate_composition_stub(op_name, op_def, yaml_content)
            stubs.append(op_stub)

    # Write to output file
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        # Emit the protocol import line to match what the stub body actually references. The generator
        # runs `ruff-format`, which does NOT remove unused imports, but the repo's pre-commit runs
        # `ruff-check --fix`, which does. Emitting an unused `BaseAnalysisBatchProtocol` therefore
        # oscillates: the generator adds it, pre-commit strips it on commit, and CI's regeneration adds it
        # back and reports the committed stub stale. Deciding it here makes generator output identical to
        # post-pre-commit output, so the freshness check is stable. Since #60, most composites resolve to
        # `Default`, and `Base` appears only when something genuinely falls back to it.
        rendered = "\n".join(stubs)
        # Count rather than search-after-the-import: the import LINE itself contains the name, so any
        # "does the rest mention it" test is trivially true. One occurrence means only the import has it.
        if rendered.count("BaseAnalysisBatchProtocol") == 1:
            rendered = rendered.replace(
                "from interpretune.protocol import BaseAnalysisBatchProtocol, DefaultAnalysisBatchProtocol",
                "from interpretune.protocol import DefaultAnalysisBatchProtocol",
                1,
            )
        f.write(rendered)

    print(f"Stubs generated at {output_path}")

    # Apply formatting to match pre-commit hooks
    try:
        import subprocess

        # Ensure pre-commit hooks are installed
        # We run install every time since it's idempotent and fast if already installed
        install_result = subprocess.run(["pre-commit", "install"], capture_output=True, text=True, cwd=project_root)

        if install_result.returncode != 0:
            print(f"Warning: Failed to install pre-commit hooks: {install_result.stderr}")
            print("Skipping formatting step")
            return

        # Run pre-commit ruff formatting on the generated file to match pre-commit formatting
        result = subprocess.run(
            ["pre-commit", "run", "ruff-format", "--files", str(output_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode == 0:
            print(f"Applied ruff formatting to {output_path}")
        else:
            # Pre-commit returns non-zero when it makes changes, which is expected
            if "reformatted" in result.stdout or "Passed" in result.stdout:
                print(f"Applied ruff formatting to {output_path}")
            else:
                print(f"Warning: ruff formatting may have failed: {result.stdout}")

    except Exception as e:
        print(f"Warning: Could not apply formatting: {e}")


def stub_module_name(collection_key: str) -> str:
    """Sanitize a collection key into a module name a type checker can import.

    ``speediedan.concept_direction_ops`` -> ``speediedan__concept_direction_ops``. A namespaced op name is not
    a Python identifier, so a stub cannot declare it directly; each collection gets its own stub module whose
    ops are declared under their BARE names, which is how the collection's ops are called once
    ``it.hub.prefer_ops`` is in effect.
    """
    return collection_key.replace("/", "__").replace(".", "__").replace("-", "_")


def group_definitions_by_collection(definitions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group non-bundled op definitions by the collection they came from.

    Keyed by hub namespace (or ``local``) rather than the declared collection name: the namespace is what
    addresses the ops and what ``prefer_ops`` takes, and a collection is free to declare any handle it likes
    -- including one that collides with another collection's.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for name, op_def in definitions.items():
        source = getattr(op_def, "source", "bundled")
        if source == "bundled":
            continue
        if name != getattr(op_def, "name", name):
            continue  # alias entry pointing at the same OpDef
        key = source.split(":", 1)[1] if source.startswith("hub:") else "local"
        grouped.setdefault(key, {})[name] = op_def
    return grouped


def _resolve_through_dispatcher(op_name: str, op_def: Any) -> Callable:
    """Import a collection op's implementation the way the dispatcher does at runtime.

    Hub ops resolve through the dynamic-module machinery (their ``implementation`` is repo-relative), and local
    collection ops resolve as ordinary imports because their directory is on ``sys.path``. Using the
    dispatcher's own resolution means a generated stub reflects the signature that will actually be called.
    """
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    source = getattr(op_def, "source", "")
    if source.startswith("hub:"):
        return DISPATCHER._import_hub_callable(op_name, op_def)
    return import_callable(op_def.implementation if hasattr(op_def, "implementation") else op_def["implementation"])


def generate_collection_stubs(output_dir: Path) -> List[Path]:
    """Generate one ``.pyi`` per local/hub op collection into ``output_dir``; returns the files written.

    Deliberately NOT part of the committed stub (#58/#60, §3.10). The committed
    ``src/interpretune/__init__.pyi`` stays bundled-only and offline-derivable so the stale-stubs CI check
    never depends on a network fetch or a trust decision; collection stubs are generated on demand into the
    analysis cache, where an IDE can be pointed at them.
    """
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    DISPATCHER.load_definitions()
    grouped = group_definitions_by_collection(DISPATCHER.registered_ops)
    if not grouped:
        print("No local or hub op collections found; nothing to generate (bundled ops live in the committed stub).")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for key, ops in sorted(grouped.items()):
        lines = [
            f'"""Type stubs for the {key!r} interpretune op collection."""',
            "# This file is auto-generated. Do not modify directly.",
            "",
            "from typing import Callable, Optional",
            "import torch",
            "from transformers import BatchEncoding",
            "from interpretune.analysis.ops import AnalysisBatch as AnalysisBatch",
            "from interpretune.protocol import BaseAnalysisBatchProtocol, DefaultAnalysisBatchProtocol",
            "",
        ]
        for name, op_def in sorted(ops.items()):
            bare = name.split(".")[-1]
            as_dict = op_def.to_dict() if hasattr(op_def, "to_dict") else dict(op_def)
            lines.append(f"# {name}")
            lines.append(
                generate_operation_stub(
                    bare,
                    as_dict,
                    {},
                    require_importable=False,
                    resolve_impl=lambda _bare, _def, _name=name, _od=op_def: _resolve_through_dispatcher(_name, _od),
                )
            )
        path = output_dir / f"{stub_module_name(key)}.pyi"
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(path)
        print(f"Generated {len(ops)} op stubs for {key} at {path}")
    print(
        "\nPoint your IDE's stub path at this directory to pick these up "
        "(e.g. pyright `stubPath`, or add it to `extraPaths`)."
    )
    return written


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate type stubs for interpretune analysis operations.")
    parser.add_argument(
        "--include-collections",
        action="store_true",
        help="also generate per-collection stubs for local/hub collections into the analysis cache",
    )
    parser.add_argument(
        "--collections-only",
        action="store_true",
        help="generate ONLY the per-collection stubs, leaving the committed bundled stub untouched",
    )
    parser.add_argument("--collection-stub-dir", type=Path, help="where to write collection stubs")
    args = parser.parse_args()

    if not args.collections_only:
        bundled_dir = project_root / "src" / "interpretune" / "analysis" / "ops" / "bundled"
        generate_stubs(sorted(bundled_dir.glob("**/*.yaml")), project_root / "src" / "interpretune" / "__init__.pyi")

    if args.include_collections or args.collections_only:
        stub_dir = args.collection_stub_dir
        if stub_dir is None:
            from interpretune.analysis import IT_ANALYSIS_CACHE

            stub_dir = Path(IT_ANALYSIS_CACHE) / "op_stubs"
        generate_collection_stubs(stub_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
