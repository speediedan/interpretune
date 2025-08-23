#!/usr/bin/env python
"""Generate type stubs for analysis operations to improve IDE support."""

import sys
import inspect
import importlib
from pathlib import Path
import yaml
from typing import Dict, Any, Callable, List, Union

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
    """Format schema dictionary into a readable docstring section."""
    if not schema_dict:
        return ""

    lines = []
    for field_name, field_def in schema_dict.items():
        if isinstance(field_def, dict):
            field_str = f"{field_name}"
            if "datasets_dtype" in field_def:
                field_str += f" ({field_def['datasets_dtype']})"
            if "required" in field_def and field_def["required"]:
                field_str += " (required)"
            lines.append(field_str)

    return "\n    ".join(lines)


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


def format_docstring(description: str, input_schema: Dict, output_schema: Dict) -> str:
    """Format a docstring with proper wrapping and sections."""
    doc_lines = [f'"""{description}']

    if input_schema:
        doc_lines.append("\nInput Schema:")
        doc_lines.append(f"    {format_schema_doc(input_schema)}")

    if output_schema:
        doc_lines.append("\nOutput Schema:")
        doc_lines.append(f"    {format_schema_doc(output_schema)}")

    doc_lines.append('"""')
    return "\n".join(doc_lines)


def generate_operation_stub(op_name: str, op_def: Dict[str, Any], yaml_content: Dict[str, Any]) -> str:
    """Generate type stub for a single analysis operation."""
    try:
        # Import the implementation function
        impl_path = op_def["implementation"]
        func = import_callable(impl_path)

        # Get function signature
        sig = inspect.signature(func)

        # Create parameters list
        params = []
        for name, param in sig.parameters.items():
            annotation = format_type_annotation(param.annotation)
            if annotation:
                annotation = f": {annotation}"

            default = ""
            if param.default is not param.empty:
                # Check if this parameter has a corresponding function_param in the YAML definition
                if "function_params" in op_def and name in op_def["function_params"]:
                    # Use fully qualified function name as a string
                    default = f' = "{op_def["function_params"][name]}"'
                elif param.default is None:
                    default = " = None"
                elif isinstance(param.default, str):
                    default = f" = '{param.default}'"
                else:
                    default = f" = {param.default}"

            params.append(f"{name}{annotation}{default}")

        # Get return type
        return_type = format_type_annotation(sig.return_annotation)

        # Create function signature
        signature = wrap_signature(op_name, params, return_type)

        # Create formatted docstring
        docstring = format_docstring(
            op_def.get("description", ""), op_def.get("input_schema", {}), op_def.get("output_schema", {})
        )

        # Build the complete stub
        stub = f"{signature}:\n    {docstring}\n    ...\n\n"

        # Add aliases
        aliases = []
        if "aliases" in op_def:
            for alias in op_def["aliases"]:
                aliases.append(f"{alias} = {op_name}")

        if "alias" in op_def and op_def["alias"] != op_name:
            aliases.append(f"{op_def['alias']} = {op_name}")

        if aliases:
            stub += "\n".join(aliases) + "\n\n"

        return stub

    except (ImportError, AttributeError) as e:
        print(f"Error generating stub for {op_name}: {e}")
        # Fallback to a basic stub
        return (
            f"def {op_name}(module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, "
            f"batch_idx: int) -> BaseAnalysisBatchProtocol:\n"
            f'    """Operation {op_name} (import failed: {e})"""\n'
            f"    ...\n\n"
        )


def generate_composition_stub(op_name: str, op_def: Dict[str, Any]) -> str:
    """Generate type stub for a composite operation."""
    composition = op_def.get("composition", "")
    composition_str = composition if isinstance(composition, str) else ".".join(composition)

    # Create a standardized signature for composite operations
    signature = wrap_signature(
        op_name,
        ["module", "analysis_batch: Optional[BaseAnalysisBatchProtocol]", "batch", "batch_idx: int"],
        "BaseAnalysisBatchProtocol",
    )

    # Create docstring
    doc = f'    """Composition of operations: {composition_str}'
    if "description" in op_def:
        doc += f"\n\n    {op_def['description']}"
    doc += '\n    """'

    stub = f"{signature}:\n{doc}\n    ...\n\n"

    if "alias" in op_def and op_def["alias"] != op_name:
        stub += f"{op_def['alias']} = {op_name}\n\n"

    return stub


def generate_stubs(yaml_path: Path, output_path: Path) -> None:
    """Generate type stubs for all operations in the YAML file."""
    # Load YAML definitions
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_content = yaml.safe_load(f)

    # Start with header
    stubs = [
        '"""Type stubs for Interpretune analysis operations."""',
        "# This file is auto-generated. Do not modify directly.",
        "",
        "from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Sequence, Literal",
        "import torch",
        "from transformers import BatchEncoding",
        "from interpretune.protocol import BaseAnalysisBatchProtocol, DefaultAnalysisBatchProtocol",
        "",
        "# Basic operations",
        "",
    ]

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
            op_stub = generate_composition_stub(op_name, op_def)
            stubs.append(op_stub)

    # Write to output file
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        f.write("\n".join(stubs))

    print(f"Stubs generated at {output_path}")


if __name__ == "__main__":
    yaml_path = project_root / "src" / "interpretune" / "analysis" / "ops" / "native_analysis_functions.yaml"
    output_path = project_root / "src" / "interpretune" / "__init__.pyi"

    generate_stubs(yaml_path, output_path)
