import os
import json
import traceback
import logging
import inspect
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Sequence

log = logging.getLogger(__name__)


IT_ANALYSIS_DUMP_DIR_NAME = "it_analysis_debug_dump"

class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with interpretune."""

def handle_exception_with_debug_dump(
    e: Exception,
    context_data: Union[Dict[str, Any], Sequence[Any]],
    operation_name: str = "operation",
    debug_dir_override: Optional[Union[str, Path]] = None
) -> None:
    """Handle an exception by creating a detailed debug dump file and re-raising the exception.

    Args:
        e: The caught exception
        context_data: Either a dictionary with context-specific debug information or
                     a sequence of variables to be introspected and serialized
        operation_name: Description of the operation that failed (for error messages)
        debug_dir_override: Optional custom path for debug directory

    Raises:
        The original exception after creating the debug dump
    """
    # Create debug dump file
    debug_dir = Path(debug_dir_override) if debug_dir_override else \
        Path(tempfile.gettempdir()) / IT_ANALYSIS_DUMP_DIR_NAME

    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dump_file = debug_dir / f"{operation_name}_error_{timestamp}.json"

    # Add exception information to debug data
    debug_info = {
        "error": str(e),
        "traceback": traceback.format_exc(),
    }

    # Process context_data based on its type
    if isinstance(context_data, dict):
        # If it's already a dictionary, use it directly
        debug_info.update(context_data)
    elif isinstance(context_data, (list, tuple)):
        # For sequences, introspect each item
        context_dict = {}

        # Try to get variable names from the caller's frame
        frame = inspect.currentframe()
        try:
            if frame and frame.f_back:
                # Get the line of code that called this function
                call_line = inspect.getframeinfo(frame.f_back).code_context[0].strip()
                # Try to extract the argument name for context_data
                if "context_data=" in call_line:
                    # Get the part after context_data=
                    context_part = call_line.split("context_data=")[1]
                    # Check if it's a tuple
                    if context_part.strip().startswith("("):
                        # Find the complete tuple by tracking parentheses
                        open_count = 0
                        tuple_str = ""
                        for char in context_part:
                            tuple_str += char
                            if char == '(':
                                open_count += 1
                            elif char == ')':
                                open_count -= 1
                                if open_count == 0:
                                    break

                        # Extract variable names from the tuple
                        var_names = [name.strip() for name in tuple_str.strip("()").split(",")]
                        if len(var_names) == len(context_data):
                            # We have matching variable names for each item in the tuple
                            for i, name in enumerate(var_names):
                                if i < len(context_data):
                                    context_dict[f"var_{i}_{name}"] = _introspect_variable(context_data[i])
                            # Skip the generic processing below
                            debug_info.update(context_dict)
        finally:
            del frame  # Avoid reference cycles

        # Generic sequence processing if we couldn't get variable names
        for i, item in enumerate(context_data):
            context_dict[f"var_{i}"] = _introspect_variable(item)

        debug_info.update(context_dict)
    else:
        # Handle single item
        debug_info["context"] = _introspect_variable(context_data)

    # Save debug info to file
    with open(dump_file, 'w') as f:
        json.dump(debug_info, f, indent=2, default=_json_serializer)

    log.error(f"{operation_name.capitalize()} failed: {e}. Debug info saved to {dump_file}")
    raise e

def _introspect_variable(var: Any) -> Dict[str, Any]:
    """Introspect a variable to create a detailed representation for debugging.

    Args:
        var: The variable to introspect

    Returns:
        A dictionary with detailed information about the variable
    """
    result = {
        "type": str(type(var).__name__),
    }

    # Handle different types appropriately
    if var is None:
        result["value"] = None
    elif isinstance(var, (str, int, float, bool)):
        # Simple scalar types
        result["value"] = var
    elif isinstance(var, (list, tuple)):
        # For sequences, include length and sample of items
        result["length"] = len(var)
        result["sample"] = var[:10] if len(var) > 10 else var
    elif isinstance(var, dict):
        # For dictionaries, include keys and sample of values
        result["keys"] = list(var.keys())
        if len(var) <= 10:
            result["content"] = var
        else:
            # Take a sample of items if dict is large
            sample = {k: var[k] for k in list(var.keys())[:10]}
            result["sample"] = sample
    elif hasattr(var, "__dict__"):
        # For objects with attributes
        result["class"] = var.__class__.__name__
        result["module"] = var.__class__.__module__

        # Get public attributes
        attrs = {}
        for attr_name in dir(var):
            if not attr_name.startswith("_"):
                try:
                    if not callable(getattr(var, attr_name, None)):
                        attr_value = getattr(var, attr_name)
                        attrs[attr_name] = str(attr_value)
                except Exception:
                    attrs[attr_name] = "<error getting attribute>"

        result["attributes"] = attrs
        result["repr"] = repr(var)
    else:
        # Fallback for other types
        result["repr"] = repr(var)

    return result

def _json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code."""
    try:
        return str(obj)
    except Exception:
        return "<non-serializable>"
