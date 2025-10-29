"""File patcher for analysis injection framework.

This module patches circuit_tracer files with analysis hooks using regex-based line matching for robustness.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import logging

from .config_parser import FileHook


class HookRegistry:
    """Global registry for managing analysis hook functions."""

    def __init__(self):
        self._hooks: Dict[str, Callable] = {}
        self._enabled: bool = False
        self._context: Dict[str, Any] = {}  # Shared context for all hooks

    def register(self, point_id: str, func: Callable) -> None:
        """Register an analysis function for a point ID."""
        self._hooks[point_id] = func

    def execute(self, point_id: str, local_vars: Dict[str, Any]) -> None:
        """Execute hook if enabled and registered."""
        if not self._enabled:
            return

        func = self._hooks.get(point_id)
        if func is not None:
            # Merge shared context with local vars
            merged_vars = {**local_vars, **self._context}
            try:
                func(merged_vars)
            except Exception as e:
                logging.getLogger("analysis_injection").warning(f"Exception in analysis hook '{point_id}': {e}")

    def enable(self) -> None:
        """Enable hook execution."""
        self._enabled = True

    def disable(self) -> None:
        """Disable hook execution."""
        self._enabled = False

    def set_context(self, **kwargs) -> None:
        """Set shared context available to all hooks.

        This is useful for passing config values like target_tokens.
        """
        self._context.update(kwargs)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get value from shared context."""
        return self._context.get(key, default)


# Global singleton
HOOK_REGISTRY = HookRegistry()


def get_analysis_vars(
    context_keys: list[str] | None = None, local_keys: list[str] | None = None, local_vars: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Get analysis variables from context and local scope.

    Args:
        context_keys: Keys to retrieve from HOOK_REGISTRY context
        local_keys: Keys to retrieve from local_vars
        local_vars: Local variables dict (usually from function locals())

    Returns:
        Dictionary containing requested variables
    """
    result = {}

    # Get context variables
    if context_keys:
        for key in context_keys:
            value = HOOK_REGISTRY.get_context(key)
            if value is not None:
                result[key] = value

    # Get local variables
    if local_keys and local_vars:
        for key in local_keys:
            if key in local_vars:
                result[key] = local_vars[key]

    return result


def find_line_by_regex(file_path: Path, regex_pattern: str) -> Optional[int]:
    """Find line number that matches regex pattern.

    Args:
        file_path: Path to file to search
        regex_pattern: Regex pattern to match

    Returns:
        Line number (0-indexed) if found, None otherwise
    """
    pattern = re.compile(regex_pattern)
    with open(file_path) as f:
        for line_num, line in enumerate(f):
            if pattern.search(line):
                # Debug: log match location
                try:
                    import logging

                    logging.getLogger("analysis_injection").info(
                        f"Regex matched in {file_path}: pattern={regex_pattern!r} at line {line_num + 1}"
                    )
                except Exception:
                    pass
                return line_num

    # If we didn't find a match, log a debug message
    try:
        import logging

        logging.getLogger("analysis_injection").warning(f"Regex not found in {file_path}: pattern={regex_pattern!r}")
    except Exception:
        pass

    return None


def patch_file_with_hooks(
    file_path: Path,
    hooks: List[FileHook],
    output_path: Optional[Path] = None,
) -> Path:
    """Patch a file by inserting hook calls at regex-matched lines.

    Args:
        file_path: Source file to patch
        hooks: List of FileHook objects for this file
        output_path: Where to write patched file (temp file if None)

    Returns:
        Path to patched file
    """
    # Read original file
    with open(file_path) as f:
        lines = f.readlines()

    # Find line numbers for each hook
    hook_lines = []
    for hook in hooks:
        line_num = find_line_by_regex(file_path, hook.regex_pattern)
        if line_num is None:
            # Log and continue so we can see which patterns failed without throwing
            import logging

            logging.getLogger("analysis_injection").warning(
                f"Pattern not found for point {hook.point_id} in {file_path}: {hook.regex_pattern}"
            )
            continue
        hook_lines.append((line_num, hook))

    # Sort by line number (reverse for correct insertion)
    hook_lines.sort(key=lambda x: x[0], reverse=True)

    # Insert hooks
    for line_num, hook in hook_lines:
        # Detect indentation of target line
        indent = len(lines[line_num]) - len(lines[line_num].lstrip())

        # Create hook call
        hook_call = f"{' ' * indent}HOOK_REGISTRY.execute('{hook.point_id}', locals())\n"

        # Insert after or before matched line
        if hook.insert_after:
            lines.insert(line_num + 1, hook_call)
        else:
            lines.insert(line_num, hook_call)

    # Add import at top of file (after any existing imports)
    # Use the new package path for the in-repo analysis_injection package located under utils
    import_line = "from it_examples.utils.analysis_injection.analysis_hook_patcher import HOOK_REGISTRY\n"

    # Find where to insert import (after last import or at beginning)
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ")):
            insert_pos = i + 1
        elif line.strip() and not line.strip().startswith("#"):
            break

    lines.insert(insert_pos, import_line)

    # Write patched file
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        output_path = Path(temp_file.name)
        temp_file.writelines(lines)
        temp_file.close()
    else:
        with open(output_path, "w") as f:
            f.writelines(lines)

    return output_path


def create_patched_module_loader(module_name: str, patched_file_path: Path) -> None:
    """Load patched module and install in sys.modules.

    Args:
        module_name: Full module name (e.g., 'circuit_tracer.attribution.attribute')
        patched_file_path: Path to patched Python file
    """
    spec = importlib.util.spec_from_file_location(module_name, patched_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {patched_file_path}")

    # Remove old module if it exists to force fresh load
    if module_name in sys.modules:
        del sys.modules[module_name]

    module = importlib.util.module_from_spec(spec)
    # Set the module in sys.modules BEFORE executing to handle circular imports
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Verify the module's __file__ attribute matches our patched file
    if hasattr(module, "__file__") and module.__file__ != str(patched_file_path):
        import logging

        logging.getLogger("analysis_injection").warning(
            f"Module {module_name} loaded but __file__ doesn't match: "
            f"expected {patched_file_path}, got {module.__file__}"
        )


def patch_target_package_files(
    file_hooks: Dict[str, FileHook],
    target_package_path: str | Path,
    target_package_name: str,
) -> Dict[str, Path]:
    """Patch target package files with hooks.

    Args:
        file_hooks: Dict mapping point_id to FileHook objects
        target_package_path: Root path to target package
        target_package_name: Name of the target package (e.g., 'circuit_tracer')

    Returns:
        Dict mapping module names to patched file paths
    """
    target_package_path = Path(target_package_path)

    # Group hooks by file
    hooks_by_file: Dict[Path, List[FileHook]] = {}
    for hook in file_hooks.values():
        if not hook.enabled:
            continue
        full_path = target_package_path / hook.file_path
        if full_path not in hooks_by_file:
            hooks_by_file[full_path] = []
        hooks_by_file[full_path].append(hook)

    # Patch each file
    patched_modules = {}
    for file_path, hooks in hooks_by_file.items():
        # Convert file path to module name
        rel_path = file_path.relative_to(target_package_path)
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_name = f"{target_package_name}." + ".".join(module_parts)

        # Patch file
        patched_path = patch_file_with_hooks(file_path, hooks)
        patched_modules[module_name] = patched_path

    return patched_modules


def install_patched_modules(patched_modules: Dict[str, Path]) -> None:
    """Install patched modules into sys.modules.

    Args:
        patched_modules: Dict mapping module names to patched file paths
    """
    for module_name, patched_path in patched_modules.items():
        create_patched_module_loader(module_name, patched_path)


def install_patched_modules_with_references(patched_modules: Dict[str, Path]) -> None:
    """Install patched modules and update all cross-references to patched functions.

    This function addresses the issue where modules that have already imported functions
    from patched modules still hold references to the original (unpatched) functions.
    After installing patched modules, it scans all loaded modules and updates any
    references to functions from patched modules.

    Args:
        patched_modules: Dict mapping module names to patched file paths
    """
    import types

    # First install the patched modules normally
    install_patched_modules(patched_modules)

    # Then update any existing references to functions from patched modules
    for module_name, patched_path in patched_modules.items():
        patched_module = sys.modules.get(module_name)
        if patched_module is None:
            continue

        # Find all modules that have imported functions from this patched module
        for importer_name, importer_module in list(sys.modules.items()):
            if importer_module is None or not hasattr(importer_module, "__dict__") or importer_module is patched_module:
                continue

            # Check each attribute in the importing module
            for attr_name, attr_value in list(importer_module.__dict__.items()):
                if (
                    isinstance(attr_value, types.FunctionType)
                    and getattr(attr_value, "__module__", None) == module_name
                ):
                    # Replace with the patched version if it exists
                    if hasattr(patched_module, attr_name):
                        patched_func = getattr(patched_module, attr_name)
                        setattr(importer_module, attr_name, patched_func)

                        # Log the update for debugging
                        try:
                            import logging

                            logging.getLogger("analysis_injection").debug(
                                f"Updated reference to {module_name}.{attr_name} in {importer_name}"
                            )
                        except Exception:
                            pass


def count_regex_matches(file_path: Path, regex_pattern: str) -> int:
    """Count number of lines in a file matching regex_pattern.

    Returns an integer count (0 if none found).
    """
    pattern = re.compile(regex_pattern)
    matches = 0
    with open(file_path) as f:
        for line in f:
            if pattern.search(line):
                matches += 1
    return matches


def validate_file_hooks(file_hooks: Dict[str, FileHook], target_package_path: str | Path) -> Dict[str, List[str]]:
    """Validate that each FileHook's regex matches exactly one line in the target file.

    Returns a dict with keys 'missing' and 'multiple', each a list of human-readable descriptions for hooks that failed
    validation.
    """
    target_package_path = Path(target_package_path)
    missing: List[str] = []
    multiple: List[str] = []

    for hook in file_hooks.values():
        if not hook.enabled:
            continue
        full_path = target_package_path / hook.file_path
        try:
            cnt = count_regex_matches(full_path, hook.regex_pattern)
        except FileNotFoundError:
            missing.append(f"MISSING FILE: {full_path} (for point {hook.point_id})")
            continue

        if cnt == 0:
            missing.append(f"{full_path} :: pattern={hook.regex_pattern!r} (point={hook.point_id})")
        elif cnt > 1:
            multiple.append(
                f"{full_path} :: pattern={hook.regex_pattern!r} matched {cnt} lines (point={hook.point_id})"
            )

    return {"missing": missing, "multiple": multiple}


def get_module_debug_info(module_name: str) -> Dict[str, Any]:
    """Get debug information about a module's patching status.

    Args:
        module_name: Full module name (e.g., 'circuit_tracer.attribution.attribute')

    Returns:
        Dict with debug information including:
        - loaded: Whether module is in sys.modules
        - file_path: Path to the loaded module file
        - has_hook_registry: Whether HOOK_REGISTRY is in module namespace
        - hook_call_count: Number of HOOK_REGISTRY.execute() calls in source
        - has_analysis_import: Whether analysis_injection import is present
    """
    import inspect

    info = {
        "module_name": module_name,
        "loaded": False,
        "file_path": None,
        "has_hook_registry": False,
        "hook_call_count": 0,
        "has_analysis_import": False,
        "functions_with_hooks": [],
    }

    if module_name not in sys.modules:
        return info

    info["loaded"] = True
    module = sys.modules[module_name]

    # Get file path
    if hasattr(module, "__file__") and module.__file__:
        info["file_path"] = module.__file__

        # Check source code
        try:
            with open(module.__file__) as f:
                source = f.read()
                info["hook_call_count"] = source.count("HOOK_REGISTRY.execute")
                info["has_analysis_import"] = "from it_examples.utils.analysis_injection" in source
        except Exception as e:
            info["read_error"] = str(e)

    # Check if HOOK_REGISTRY is in module namespace
    if hasattr(module, "HOOK_REGISTRY"):
        info["has_hook_registry"] = True

    # Check functions for hook calls in their source
    try:
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            try:
                source = inspect.getsource(obj)
                if "HOOK_REGISTRY.execute" in source:
                    info["functions_with_hooks"].append(name)
            except Exception:
                pass
    except Exception as e:
        info["inspect_error"] = str(e)

    return info


def verify_patching(patched_modules: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Verify that patched modules are correctly loaded and contain hooks.

    Args:
        patched_modules: Dict mapping module names to patched file paths

    Returns:
        Dict mapping module names to their debug info
    """
    verification = {}
    for module_name, patched_path in patched_modules.items():
        info = get_module_debug_info(module_name)
        info["expected_patched_path"] = str(patched_path)
        info["paths_match"] = info["file_path"] == str(patched_path) if info["file_path"] else False
        verification[module_name] = info

    return verification
