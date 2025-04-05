"""Schema compiler for analysis operations to maintain field propagation while minimizing schemas."""

from typing import Dict, List, Tuple, Any

from ..base import OpSchema


def compile_operation_chain_schema(
    operations: List[str], all_operations_dict: Dict[str, Dict]
) -> Tuple[OpSchema, OpSchema]:
    """Compile the complete schema for a chain of operations.

    Args:
        operations: List of operation names in the chain
        all_operations_dict: Dictionary of all operation definitions

    Returns:
        Tuple of (input_schema, output_schema) representing the complete schemas
    """
    input_fields: Dict = {}
    output_fields: Dict = {}
    intermediate_fields: Dict = {}

    for op_name in operations:
        op_def = all_operations_dict.get(op_name)
        if not op_def:
            raise ValueError(f"Operation {op_name} not found")

        # Add input fields that aren't already outputs or intermediates
        if 'input_schema' in op_def:
            for field_name, field_def in op_def['input_schema'].items():
                if field_name not in output_fields and field_name not in intermediate_fields:
                    input_fields[field_name] = field_def

        # Add output fields
        if 'output_schema' in op_def:
            for field_name, field_def in op_def['output_schema'].items():
                # Check if field is marked as intermediate only
                if field_def.get('intermediate_only', False):
                    intermediate_fields[field_name] = field_def
                else:
                    # Fix any 'object' type fields to use 'string' as the datasets_dtype
                    if field_def.get('datasets_dtype') == 'object':
                        field_def_copy = field_def.copy()
                        # Use string instead of object for PyArrow compatibility
                        field_def_copy['datasets_dtype'] = 'string'
                        output_fields[field_name] = field_def_copy
                    else:
                        output_fields[field_name] = field_def

    return input_fields, output_fields


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
