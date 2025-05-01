"""Schema compiler for analysis operations to maintain field propagation while minimizing schemas."""

from typing import Dict, List, Tuple, Any, Union, TypeVar, Callable

from ..base import OpSchema, ColCfg, AnalysisOp


# Type variables for schema and field definition types
T_Schema = TypeVar('T_Schema')  # Schema container type (Dict or OpSchema)
T_Field = TypeVar('T_Field')    # Field definition type (Dict or ColCfg)

def _compile_chain_schema_core(
    operations: List[Any],
    get_schemas_fn: Callable[[Any], Tuple[Dict[str, T_Field], Dict[str, T_Field]]],
    is_intermediate_fn: Callable[[T_Field], bool],
    handle_object_field_fn: Callable[[T_Field], T_Field],
    create_schema_fn: Callable[[Dict[str, T_Field]], T_Schema]
) -> Tuple[T_Schema, T_Schema]:
    """Core logic for compiling chain schemas with customizable handling of types.

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
        raise ValueError("No operations provided for chain schema compilation")

    for op in operations:
        # Extract input and output schemas using the provided function
        input_schema, output_schema = get_schemas_fn(op)

        # Add input fields that aren't already outputs or intermediates
        for field_name, field_def in input_schema.items():
            if field_name not in output_fields and field_name not in intermediate_fields:
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

def jit_compile_chain_schema(
    operations: List[Union[str, AnalysisOp]],
    op_definitions: Dict[str, Dict]
) -> Tuple[OpSchema, OpSchema]:
    """Compile the complete schema for a chain of operations using operation definitions.

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

            # Convert dictionary schemas to OpSchema objects if needed
            def get_schema(schema_dict):
                if not schema_dict:
                    return {}
                result = {}
                for field_name, field_def in schema_dict.items():
                    if isinstance(field_def, dict):
                        result[field_name] = ColCfg(**field_def)
                    elif isinstance(field_def, ColCfg):
                        result[field_name] = field_def
                return result

            return get_schema(op_def.get('input_schema', {})), get_schema(op_def.get('output_schema', {}))

        elif hasattr(op, 'input_schema') and hasattr(op, 'output_schema'):
            # It's already an AnalysisOp instance
            return op.input_schema or {}, op.output_schema or {}
        else:
            raise TypeError(f"Operations must be strings or AnalysisOp instances with schemas, got {type(op)}")

    def is_intermediate(field_def):
        if isinstance(field_def, dict):
            return field_def.get('intermediate_only', False)
        elif hasattr(field_def, 'intermediate_only'):
            return field_def.intermediate_only
        return False

    def handle_object_field(field_def):
        datasets_dtype = None
        if isinstance(field_def, dict):
            datasets_dtype = field_def.get('datasets_dtype')
        elif hasattr(field_def, 'datasets_dtype'):
            datasets_dtype = field_def.datasets_dtype

        if datasets_dtype == 'object':
            if isinstance(field_def, dict):
                field_def_copy = field_def.copy()
                field_def_copy['datasets_dtype'] = 'string'
                field_def_copy['non_tensor'] = True
                return field_def_copy
            elif isinstance(field_def, ColCfg):
                # Create a modified ColCfg for object fields
                import copy
                field_def_copy = copy.copy(field_def)
                field_def_copy.datasets_dtype = 'string'
                field_def_copy.non_tensor = True
                return field_def_copy
        return field_def

    def create_schema(fields):
        return OpSchema(fields)

    return _compile_chain_schema_core(
        operations=operations,
        get_schemas_fn=get_schemas,
        is_intermediate_fn=is_intermediate,
        handle_object_field_fn=handle_object_field,
        create_schema_fn=create_schema
    )

def compile_operation_chain_schema(
    operations: List[str], all_operations_dict: Dict[str, Dict]
) -> Tuple[Dict, Dict]:
    """Compile the complete schema for a chain of operations.

    Args:
        operations: List of operation names in the chain
        all_operations_dict: Dictionary of all operation definitions

    Returns:
        Tuple of (input_schema, output_schema) representing the complete schemas
    """
    def get_schemas(op_name):
        op_def = all_operations_dict.get(op_name)
        if not op_def:
            raise ValueError(f"Operation {op_name} not found")
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

    return _compile_chain_schema_core(
        operations=operations,
        get_schemas_fn=get_schemas,
        is_intermediate_fn=is_intermediate,
        handle_object_field_fn=handle_object_field,
        create_schema_fn=create_schema
    )

def build_operation_chains(yaml_config: Dict) -> Dict[str, Any]:
    """Build operation chains with compiled schemas from YAML configuration.

    Args:
        yaml_config: YAML configuration dictionary

    Returns:
        Updated configuration with compiled operation chains
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
    for name, chain_def in composite_ops.items():
        chain = chain_def.get('chain', '').split('.')
        alias = chain_def.get('alias', name)

        input_schema, output_schema = compile_operation_chain_schema(chain, all_ops_dict)

        # Create a new operation definition with the compiled schemas
        ops[name] = {
            'description': f"Compiled chain: {'.'.join(chain)}",
            'chain': chain,
            'input_schema': input_schema,
            'output_schema': output_schema,
            'alias': alias
        }

    return ops


def load_and_compile_operations(yaml_path: str) -> Dict[str, Any]:
    """Load operations from YAML file and compile operation chains.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Dictionary of operations with compiled schemas
    """
    import yaml

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return build_operation_chains(config)
