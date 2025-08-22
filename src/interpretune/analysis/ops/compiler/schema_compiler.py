"""Schema compiler for analysis operations to maintain field propagation while minimizing schemas."""

from typing import Dict, List, Tuple, Any, Union, TypeVar, Callable, Set
from dataclasses import replace
from copy import deepcopy
import re

from ..base import OpSchema, ColCfg, AnalysisOp
from interpretune.utils.logging import rank_zero_warn


# Type variables for schema and field definition types
T_Schema = TypeVar('T_Schema')  # Schema container type (Dict or OpSchema)
T_Field = TypeVar('T_Field')    # Field definition type (Dict or ColCfg)

def _compile_composition_schema_core(
    operations: List[Any],
    get_schemas_fn: Callable[[Any], Tuple[Dict[str, T_Field], Dict[str, T_Field]]],
    is_intermediate_fn: Callable[[T_Field], bool],
    handle_object_field_fn: Callable[[T_Field], T_Field],
    create_schema_fn: Callable[[Dict[str, T_Field]], T_Schema]
) -> Tuple[T_Schema, T_Schema]:
    """Core logic for compiling composition schemas with customizable handling of types.

    Args:
        operations: List of operations (strings, AnalysisOp objects, etc.)
        get_schemas_fn: Function to extract input and output schemas from an operation
        is_intermediate_fn: Function to check if a field is intermediate-only
        handle_object_field_fn: Function to handle object field type conversion
        create_schema_fn: Function to create the final schema container

    Returns:
        Tuple of (input_schema, output_schema)
    """
    input_fields: Dict[str, T_Field] = {}
    output_fields: Dict[str, T_Field] = {}
    intermediate_fields: Dict[str, T_Field] = {}

    if not operations:
        raise ValueError("No operations provided for composite schema compilation")

    for op in operations:
        # Extract input and output schemas using the provided function
        input_schema, output_schema = get_schemas_fn(op)

        # Add input fields
        for field_name, field_def in input_schema.items():
            input_fields[field_name] = field_def

        # Add output fields
        for field_name, field_def in output_schema.items():
            # Check if field is marked as intermediate only
            if is_intermediate_fn(field_def):
                intermediate_fields[field_name] = field_def
            else:
                # Handle object type fields for PyArrow compatibility
                output_fields[field_name] = handle_object_field_fn(field_def)

    # Create and return the final schemas using the provided function
    return create_schema_fn(input_fields), create_schema_fn(output_fields)

def jit_compile_composition_schema(
    operations: List[Union[str, AnalysisOp]],
    op_definitions: Dict[str, Dict]
) -> Tuple[OpSchema, OpSchema]:
    """Compile the complete schema for a composition of operations using operation definitions.

    Args:
        operations: List of operation names or AnalysisOp instances
        op_definitions: Dictionary of operation definitions (typically dispatcher._op_definitions)

    Returns:
        Tuple of (input_schema, output_schema) representing the complete schemas

    Raises:
        ValueError: If operations is empty, an operation name is not found or lacks required schemas
        TypeError: If operations contain invalid types
    """
    def get_schemas(op):
        # Get operation schema based on input type
        if isinstance(op, str):
            # Get the operation definition directly from op_definitions
            op_def = op_definitions.get(op)
            if not op_def:
                raise ValueError(f"Operation {op} not found in definitions")

            # Extract schema information from the definition
            if 'input_schema' not in op_def or 'output_schema' not in op_def:
                raise ValueError(f"Operation {op} is missing required schemas")

            # Convert dictionary schemas to ColCfg objects if needed
            def get_schema(schema_dict):
                if not schema_dict:
                    return {}
                result = {}
                for field_name, field_def in schema_dict.items():
                    if isinstance(field_def, ColCfg):
                        result[field_name] = field_def
                    else:
                        try:
                            if isinstance(field_def, dict):
                                result[field_name] = ColCfg(**field_def)
                            else:
                                # Attempt to convert to ColCfg using the constructor
                                result[field_name] = ColCfg.from_dict(field_def.__dict__)
                        except Exception as e:
                            # If the field_def is not a dict or ColCfg, log a warning but allow compilation to proceed
                            rank_zero_warn(
                                f"Conversion to ColCfg of field {field_name} in operation {op} did not succeed. "
                                f"Field `{field_name}` (type: {type(field_def)}) will be skipped, error: {e}"
                            )

                return result

            return get_schema(op_def.get('input_schema', {})), get_schema(op_def.get('output_schema', {}))

        elif hasattr(op, 'input_schema') and hasattr(op, 'output_schema'):
            # It's already an AnalysisOp instance
            return op.input_schema or {}, op.output_schema or {}
        else:
            raise TypeError(f"Operations must be strings or AnalysisOp instances with schemas, got {type(op)}")

    def is_intermediate(field_def):
        return getattr(field_def, 'intermediate_only', False)

    def handle_object_field(field_def):
        datasets_dtype = field_def.datasets_dtype

        if datasets_dtype == 'object':
            # Use replace to create a new instance instead of modifying
            return replace(field_def, datasets_dtype='string', non_tensor=True)
        return field_def

    def create_schema(fields):
        return OpSchema(fields)

    return _compile_composition_schema_core(
        operations=operations,
        get_schemas_fn=get_schemas,
        is_intermediate_fn=is_intermediate,
        handle_object_field_fn=handle_object_field,
        create_schema_fn=create_schema
    )

def compile_operation_composition_schema(
    operations: List[str], all_operations_dict: Dict[str, Dict]
) -> Tuple[Dict, Dict]:
    """Compile the complete schema for a composition of operations.

    Args:
        operations: List of operation names in the composition
        all_operations_dict: Dictionary of all operation definitions

    Returns:
        Tuple of (input_schema, output_schema) representing the complete schemas
    """
    def get_schemas(op_name):
        op_def = all_operations_dict.get(op_name)
        if not op_def:
            # Search for namespaced versions of the operation
            matching_ops = []
            for full_op_name in all_operations_dict.keys():
                # Check if the full name ends with the requested op_name
                # Format: namespace.collection.op_name
                if '.' in full_op_name and full_op_name.split('.')[-1] == op_name:
                    matching_ops.append(full_op_name)

            if not matching_ops:
                raise ValueError(f"Operation {op_name} not found")
            elif len(matching_ops) > 1:
                # Multiple matches found - issue warning and use first one
                resolved_op_name = matching_ops[0]
                rank_zero_warn(
                    f"Multiple operations matching '{op_name}' were found: {matching_ops}. "
                    f"Using '{resolved_op_name}'. Consider using the fully-qualified operation name "
                    f"in your composition definition to avoid ambiguity.",
                    stacklevel=3
                )
                op_def = all_operations_dict[resolved_op_name]
            else:
                # Single match found - use it
                op_def = all_operations_dict[matching_ops[0]]

        return op_def.get('input_schema', {}), op_def.get('output_schema', {})

    def is_intermediate(field_def):
        return field_def.get('intermediate_only', False) if isinstance(field_def, dict) else False

    def handle_object_field(field_def):
        if isinstance(field_def, dict) and field_def.get('datasets_dtype') == 'object':
            field_def_copy = field_def.copy()
            field_def_copy['datasets_dtype'] = 'string'
            field_def_copy['non_tensor'] = True
            return field_def_copy
        return field_def

    def create_schema(fields):
        return fields  # Just return the dictionary directly

    return _compile_composition_schema_core(
        operations=operations,
        get_schemas_fn=get_schemas,
        is_intermediate_fn=is_intermediate,
        handle_object_field_fn=handle_object_field,
        create_schema_fn=create_schema
    )

def resolve_required_ops(op_name: str, op_def: Dict[str, Any], op_definitions: Dict[str, Dict]) -> List[str]:
    """Resolve required_ops for an operation, handling namespaced operations.

    Args:
        op_name: Name of the operation whose required_ops need resolution
        op_def: Operation definition containing required_ops
        op_definitions: Dictionary of all operation definitions

    Returns:
        List of resolved operation names

    Raises:
        ValueError: If required operations cannot be resolved
    """
    required_ops = op_def.get('required_ops', [])
    if not required_ops:
        return []

    resolved_ops = []

    for required_op in required_ops:
        # First, check if the required op exists exactly as specified
        if required_op in op_definitions:
            resolved_ops.append(required_op)
            continue

        # If not found exactly, search for namespaced operations with matching basename
        # Extract basename from required_op (handle case where required_op might already be namespaced)
        required_basename = required_op.split('.')[-1]

        # Find all ops whose basename matches the required basename
        matching_ops = []
        for existing_op_name in op_definitions:
            existing_basename = existing_op_name.split('.')[-1]
            if existing_basename == required_basename:
                matching_ops.append(existing_op_name)

        if not matching_ops:
            # No matching operations found
            # rank_zero_warn(f"Operation '{op_name}' requires '{required_op}' but no matching operation found. "
            #               f"Operation '{op_name}' will be skipped.")
            raise ValueError(f"Required operation '{required_op}' not found for operation '{op_name}'.")

        if len(matching_ops) == 1:
            # Single match - use it
            resolved_ops.append(matching_ops[0])
        else:
            # Multiple matches - issue warning and use first match
            rank_zero_warn(f"Operation '{op_name}' requires '{required_op}' but multiple matching operations found: "
                          f"{matching_ops}. Using '{matching_ops[0]}'. Consider using a fully-qualified name "
                          f"if a different operation should be used.")
            resolved_ops.append(matching_ops[0])

    return resolved_ops


def build_operation_compositions(yaml_config: Dict) -> Dict[str, Any]:
    """Build operation compositions with compiled schemas from YAML configuration.

    Args:
        yaml_config: YAML configuration dictionary

    Returns:
        Updated configuration with compiled operation compositions
    """
    ops = yaml_config.copy()
    composite_ops = ops.pop('composite_operations', {})

    all_ops_dict = {}
    # For the non-composite ops, add them to our operations dictionary
    for k, v in ops.items():
        if isinstance(v, dict):
            # Fix any 'object' type fields to use 'string' for PyArrow compatibility
            if 'output_schema' in v:
                for field_name, field_def in v['output_schema'].items():
                    if isinstance(field_def, dict) and field_def.get('datasets_dtype') == 'object':
                        field_def['datasets_dtype'] = 'string'
                        field_def['non_tensor'] = True
            all_ops_dict[k] = v

    # Build compiled operations
    for name, composition_def in composite_ops.items():
        composition_str = composition_def.get('composition', '')
        aliases = composition_def.get('aliases', [])

        # Parse composition string to handle parentheses-wrapped namespaced operations
        composition = _parse_composition_string(composition_str)

        try:
            input_schema, output_schema = compile_operation_composition_schema(composition, all_ops_dict)
        except Exception as e:
            rank_zero_warn(f"Failed to compile operation '{name}' with composition {composition}: {e}")
            continue

        # Create a new operation definition with the compiled schemas
        ops[name] = {
            'description': f"Compiled composition: {'.'.join(composition)}",
            'composition': composition,
            'input_schema': input_schema,
            'output_schema': output_schema,
            'aliases': aliases
        }

    return ops

def _parse_composition_string(composition_str: str) -> List[str]:
    """Parse composition string to handle parentheses-wrapped namespaced operations.

    Examples:
        "op1.op2.op3" -> ["op1", "op2", "op3"]
        "op1.(namespace.op2).op3" -> ["op1", "namespace.op2", "op3"]
        "trivial_local_test_op.(speediedan.trivial_op_repo.trivial_test_op)" ->
            ["trivial_local_test_op", "speediedan.trivial_op_repo.trivial_test_op"]

    Args:
        composition_str: Dot-separated composition string with optional parentheses-wrapped operations

    Returns:
        List of operation names, with parentheses-wrapped operations resolved to their full names
    """
    if not composition_str:
        return []

    # Check for unbalanced parentheses
    open_count = composition_str.count('(')
    close_count = composition_str.count(')')
    if open_count != close_count:
        raise ValueError(f"Unbalanced parentheses in composition string: '{composition_str}'")

    # Pattern to match either:
    # 1. Parentheses-wrapped operations: (content) -> `\(([^)]+)\)`
    # 2. Regular operation names (sequences of non-dot, non-paren chars) -> `([^.()]+)`
    # Split by dots, but treat parentheses-wrapped content as single units
    pattern = r'\(([^)]+)\)|([^.()]+)'

    operations = []
    for match in re.finditer(pattern, composition_str):
        if match.group(1):  # Parentheses-wrapped operation
            operations.append(match.group(1).strip())
        elif match.group(2) and match.group(2).strip():  # Regular operation (non-empty)
            operations.append(match.group(2).strip())

    return operations

def compile_op_schema(op_name: str, op_definitions: Dict[str, Dict[str, Any]],
                      _processing: Set[str] = None) -> Dict[str, Any]:  # type: ignore[assignment]
    """Compile operation schema by merging schemas from required operations.

    Args:
        op_name: Name of the operation to compile
        op_definitions: Dictionary of all available operation definitions
        _processing: Set of operations currently being processed (used internally for circular dependency detection)

    Returns:
        Compiled operation definition with merged schemas

    Raises:
        ValueError: If the operation or a required operation is not found in definitions
    """
    if op_name not in op_definitions:
        raise ValueError(f"Operation {op_name} not found in definitions")

    # Initialize processing set on first call
    if _processing is None:
        _processing = set()


    op_def = op_definitions[op_name]

    # Start with a deep copy of the original definition
    compiled_def = deepcopy(op_def)

    # Check for circular dependency
    if op_name in _processing:
        # TODO: look into handling circular dependencies more gracefully, being cautious and rejecting the operation
        raise ValueError(f"Circular dependency detected: {' -> '.join(_processing)} -> {op_name}")

    # Get required operations
    # Resolve required_ops first
    try:
        resolved_required_ops = resolve_required_ops(op_name, op_def, op_definitions)
        compiled_def['required_ops'] = resolved_required_ops
    except ValueError as e:
        # If we can't resolve required ops, skip this operation
        raise ValueError(f"Operation '{op_name}' cannot be compiled due to unresolved required operations: {e}")

    if not resolved_required_ops:
        op_definitions[op_name] = compiled_def
        return op_definitions[op_name]  # No required ops, return as is

    # Initialize schemas if they don't exist
    if "input_schema" not in compiled_def:
        compiled_def["input_schema"] = {}
    if "output_schema" not in compiled_def:
        compiled_def["output_schema"] = {}

    # Add current operation to processing set
    _processing.add(op_name)

    try:
        # Recursively merge schemas from required operations
        for req_op_name in resolved_required_ops:

            # Recursively compile the required operation first
            compiled_req_def = compile_op_schema(req_op_name, op_definitions, _processing)

            # Merge input schemas (required op schemas have lower precedence)
            req_input_schema = compiled_req_def.get("input_schema", {})
            for field_name, field_config in req_input_schema.items():
                if field_name not in compiled_def["input_schema"]:
                    compiled_def["input_schema"][field_name] = deepcopy(field_config)
                    # TODO: Mark as potentially available output for this operation?
                    if field_name not in compiled_def["output_schema"]:
                        compiled_def["output_schema"][field_name] = field_config.copy()
                        # Mark as not required since it comes from a required op
                        compiled_def["output_schema"][field_name]['required'] = False

            # Merge required op output schemas (required op schemas have lower precedence)
            req_output_schema = compiled_req_def.get("output_schema", {})
            for field_name, field_config in req_output_schema.items():
                if field_name not in compiled_def["output_schema"]:
                    compiled_def["output_schema"][field_name] = deepcopy(field_config)
                    compiled_def["output_schema"][field_name]['required'] = False

    finally:
        # Remove current operation from processing set
        _processing.discard(op_name)

    op_definitions[op_name] = compiled_def
    return op_definitions[op_name]  # ref usually not usually needed, but return for consistency
