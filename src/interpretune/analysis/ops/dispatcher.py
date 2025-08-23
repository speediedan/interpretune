"""Dispatcher for analysis operations."""

from __future__ import annotations
from typing import Optional, Dict, NamedTuple, List, Tuple, Iterator, Callable, Union, Any
from pathlib import Path
from functools import wraps
from collections import defaultdict
import importlib
import yaml

import torch
from transformers import BatchEncoding

from interpretune.analysis import IT_ANALYSIS_CACHE, IT_ANALYSIS_OP_PATHS, IT_ANALYSIS_HUB_CACHE
from interpretune.analysis.ops.base import AnalysisOp, CompositeAnalysisOp, OpSchema, ColCfg
from interpretune.analysis.ops.auto_columns import apply_auto_columns
from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager, OpDef
from interpretune.analysis.ops.dynamic_module_utils import ensure_op_paths_in_syspath, get_function_from_dynamic_module
from interpretune.protocol import BaseAnalysisBatchProtocol
from interpretune.utils.logging import rank_zero_debug, rank_zero_warn


def _ensure_loaded(func):
    """Decorator to ensure operations are loaded before access."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._loaded:
            self.load_definitions()
        return func(self, *args, **kwargs)

    return wrapper


class DispatchContext(NamedTuple):
    """Context for dispatching operations."""

    pass  # We don't use context keys yet but may in the future


class AnalysisOpDispatcher:
    """Dispatcher for analysis operations with lazy loading and caching.

    This class handles loading operation definitions from YAML and dispatching them based on a given context. Operations
    are dynamically instantiated from their definitions when first accessed.
    """

    # TODO:
    #  - decide whether to make the dispatcher a singleton or not
    #  - decide whether to make the dispatcher thread-safe
    def __init__(self, yaml_paths: Optional[Union[Path, List[Path]]] = None, enable_hub_ops: bool = True):
        # Initialize yaml_paths
        self.yaml_paths = [Path(p.strip()) for p in IT_ANALYSIS_OP_PATHS]  # Start with op_paths

        # Always include the default built-in analysis ops yaml
        self.yaml_paths.append(Path(__file__).parent / "native_analysis_functions.yaml")

        # Handle user-provided yaml_paths
        if yaml_paths:  # otherwise use only use the default
            if isinstance(yaml_paths, (Path, str)):
                self.yaml_paths.append(Path(yaml_paths))
            else:
                # Handle list/iterable of paths (convert strings to Path objects)
                self.yaml_paths.extend(Path(p) for p in yaml_paths)
                assert all(isinstance(p, Path) for p in self.yaml_paths), (
                    "yaml_paths must be a Path, string, or a list of Paths/strings"
                )

        self.enable_hub_ops = enable_hub_ops
        self._op_definitions: Dict[str, OpDef] = {}
        self._dispatch_table = {}  # {op_name: {context: instantiated_op}}
        self._aliases = {}  # {alias: op_name}
        self._op_to_aliases = defaultdict(list)  # {op_name: [aliases]}
        self._loaded = False
        self._loading_in_progress = False
        # resolve op_paths from yaml_paths
        self.op_paths = []
        # Resolve op_paths from yaml_paths
        self._resolve_op_paths_from_yaml_paths()
        # Ensure op_paths are in sys.path
        ensure_op_paths_in_syspath(self.op_paths)
        self._cache_manager = OpDefinitionsCacheManager(IT_ANALYSIS_CACHE)

    def _normalize_op_name(self, name: str) -> str:
        # Normalize operation names for consistent lookup (case-insensitive, cross-platform)
        return name.replace("/", ".").replace("-", "_").lower()

    def _discover_yaml_files(self, paths: List[Path]) -> List[Path]:
        """Discover all YAML files from the given paths (files or directories)."""
        yaml_files = []
        for path in paths:
            if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
                yaml_files.append(path)
            elif path.is_dir():
                # Recursively find all YAML files in the directory
                yaml_files.extend(path.glob("**/*.yaml"))
                yaml_files.extend(path.glob("**/*.yml"))
        return sorted(set(yaml_files))  # Remove duplicates and sort for consistency

    def load_definitions(self) -> None:
        """Load operation definitions from YAML files."""
        if self._loaded or self._loading_in_progress:
            return

        self._loading_in_progress = True
        try:
            # Discover all YAML files from the configured paths
            yaml_files = self._discover_yaml_files(self.yaml_paths)
            rank_zero_debug(f"[DISPATCHER] Discovered {len(yaml_files)} local YAML files")

            # Add hub operations if enabled
            if self.enable_hub_ops:
                rank_zero_debug("[DISPATCHER] Hub ops enabled, adding hub YAML files")
                hub_yaml_files = self._cache_manager.add_hub_yaml_files()
                yaml_files.extend(hub_yaml_files or [])
                rank_zero_debug(f"[DISPATCHER] Total YAML files after hub: {len(yaml_files)}")
            else:
                rank_zero_debug("[DISPATCHER] Hub ops disabled")

            # Set up cache manager with discovered YAML files # TODO: might be able to remove since we already discover
            for yaml_file in yaml_files:
                if yaml_file not in [info.path for info in self._cache_manager._yaml_files]:
                    self._cache_manager.add_yaml_file(yaml_file)

            rank_zero_debug("[DISPATCHER] Attempting to load from cache")
            # Try to load from cache first
            cached_definitions = self._cache_manager.load_cache()
            if cached_definitions is not None:
                rank_zero_debug(f"[DISPATCHER] Cache HIT: Loaded {len(cached_definitions)} definitions from cache")
                self._op_definitions = cached_definitions
                self._set_default_hub_op_aliases()
            else:
                rank_zero_debug("[DISPATCHER] Cache MISS: Compiling from source")
                # Cache miss or invalid - load from YAML and compile
                rank_zero_debug("Cache miss or invalid, loading from YAML and compiling")
                self._load_from_yaml_and_compile(yaml_files)

            # Build aliases mapping
            self._populate_aliases_from_definitions()

            self._loaded = True
            rank_zero_debug(f"[DISPATCHER] Loaded {len(self._op_definitions)} operation definitions")

        except Exception as e:
            rank_zero_warn(f"Failed to load operation definitions: {e}")
            raise
        finally:
            self._loading_in_progress = False

    def _load_from_yaml_and_compile(self, yaml_files: List[Path]):
        """Load from YAML files and compile to cache."""
        # Load and merge all YAML files
        raw_definitions = {}
        composite_operations = {}

        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    yaml_content = yaml.safe_load(f)

                if not yaml_content:
                    rank_zero_debug(f"Empty YAML file: {yaml_file}")
                    continue

                # Apply namespace prefixes for hub operations
                namespaced_content = self._apply_hub_namespacing(yaml_content, yaml_file)

                # Separate composite operations from regular operations
                for key, value in namespaced_content.items():
                    if key == "composite_operations":
                        composite_operations.update(value)
                    else:
                        if key in raw_definitions:
                            rank_zero_debug(f"Operation '{key}' redefined in {yaml_file}, using latest definition")
                        raw_definitions[key] = value

            except Exception as e:
                rank_zero_debug(f"Failed to load YAML file {yaml_file}: {e}")
                # Continue processing other files rather than failing completely
                continue

        # Second pass: Compile schemas with required_ops dependencies
        self._compile_required_ops_schemas(raw_definitions)

        # Process composite operations with schema compilation
        if composite_operations:
            from interpretune.analysis.ops.compiler.schema_compiler import build_operation_compositions

            # Create a complete YAML structure for build_operation_compositions
            complete_yaml = raw_definitions.copy()
            complete_yaml["composite_operations"] = composite_operations

            # Apply schema compilation for composite operations
            compiled_ops = build_operation_compositions(complete_yaml)

            # Update definitions with compiled operation schemas
            for op_name, op_def in compiled_ops.items():
                if op_name not in raw_definitions:
                    raw_definitions[op_name] = op_def
                else:
                    # Update existing definition with compiled schemas
                    if "input_schema" in op_def:
                        raw_definitions[op_name]["input_schema"] = op_def["input_schema"]
                    if "output_schema" in op_def:
                        raw_definitions[op_name]["output_schema"] = op_def["output_schema"]

        # Convert raw definitions to OpDef objects
        self._convert_raw_definitions_to_opdefs(raw_definitions)
        self._set_default_hub_op_aliases()
        # Build aliases mapping
        self._populate_aliases_from_definitions()

        # Save to cache for next time
        self._cache_manager.save_cache(self._op_definitions)

        self._loaded = True

    def _compile_required_ops_schemas(self, definitions_to_compile: Dict[str, Dict]):
        """Compile schemas by recursively including required_ops dependencies."""
        from interpretune.analysis.ops.compiler.schema_compiler import compile_op_schema

        # TODO: consider moving this compilation to schema_compiler.py, we're keeping this here for now because
        #       applying auto-columns should not be part of schema_compiler.py
        # Compile all operations
        for op_name in list(definitions_to_compile.keys()):
            try:
                compile_op_schema(op_name, definitions_to_compile)
                # Apply optional auto-columns after compilation
                apply_auto_columns(definitions_to_compile[op_name])
            except ValueError as e:
                rank_zero_warn(f"Failed to compile operation '{op_name}': {e}")
                # Remove the operation if it fails to compile
                definitions_to_compile.pop(op_name, None)

    def _convert_raw_definitions_to_opdefs(self, raw_definitions: Dict[str, Dict]):
        """Convert raw dictionary definitions to OpDef objects."""
        for op_name, op_def in raw_definitions.items():
            op_name = self._normalize_op_name(op_name)
            # Convert schemas to OpSchema objects
            input_schema = self._convert_to_op_schema(op_def.get("input_schema", {}))
            output_schema = self._convert_to_op_schema(op_def.get("output_schema", {}))

            importable_params = op_def.get("importable_params", {})

            # Create OpDef
            op_def_obj = OpDef(
                name=op_name,
                description=op_def.get("description", ""),
                implementation=op_def.get("implementation", ""),
                input_schema=input_schema,
                output_schema=output_schema,
                aliases=op_def.get("aliases", []),
                importable_params=importable_params,
                normal_params=op_def.get("normal_params", {}),
                required_ops=op_def.get("required_ops", []),
                composition=op_def.get("composition", None),
            )

            self._op_definitions[op_name] = op_def_obj

    def _apply_hub_namespacing(self, yaml_content: Dict[str, Any], yaml_file: Path) -> Dict[str, Any]:
        """Apply hub namespacing to operations from hub files."""
        rank_zero_debug(f"[DISPATCHER] Processing yaml_file: {yaml_file}")

        # Get namespace for this file
        namespace = self._cache_manager.get_hub_namespace(yaml_file)
        rank_zero_debug(f"[DISPATCHER] Retrieved namespace: '{namespace}'")

        # If it's a top-level namespace (non-hub), return unchanged
        if "." not in namespace:
            rank_zero_debug(f"[DISPATCHER] No dots in namespace '{namespace}' - returning unchanged")
            return yaml_content

        rank_zero_debug(f"[DISPATCHER] Applying namespace '{namespace}' to operations: {list(yaml_content.keys())}")

        # Apply namespacing to hub operations
        namespaced_content = {}

        for op_name, op_config in yaml_content.items():
            if op_name == "composite_operations":
                # Handle composite operations separately - namespace the compositions
                namespaced_composites = {}
                for comp_name, comp_config in op_config.items():
                    namespaced_comp_name = f"{namespace}.{comp_name}"
                    namespaced_composites[namespaced_comp_name] = comp_config.copy()

                    # Also namespace any aliases
                    if "aliases" in comp_config:
                        namespaced_aliases = []
                        for alias in comp_config["aliases"]:
                            namespaced_aliases.append(f"{namespace}.{alias}")
                        namespaced_composites[namespaced_comp_name]["aliases"] = namespaced_aliases

                namespaced_content["composite_operations"] = namespaced_composites
                continue

            # Add namespace prefix to operation name
            namespaced_name = f"{namespace}.{op_name}"
            rank_zero_debug(f"[DISPATCHER] Namespacing '{op_name}' -> '{namespaced_name}'")
            namespaced_content[namespaced_name] = op_config.copy()

            # Also namespace any aliases
            if "aliases" in op_config:
                namespaced_aliases = []
                for alias in op_config["aliases"]:
                    namespaced_alias = f"{namespace}.{alias}"
                    rank_zero_debug(f"[DISPATCHER] Namespacing alias '{alias}' -> '{namespaced_alias}'")
                    namespaced_aliases.append(namespaced_alias)
                namespaced_content[namespaced_name]["aliases"] = namespaced_aliases

        rank_zero_debug(f"[DISPATCHER] Final namespaced operations: {list(namespaced_content.keys())}")
        return namespaced_content

    def _populate_aliases_from_definitions(self):
        """Build alias mappings from operation definitions."""
        # Clear existing mappings
        self._aliases.clear()
        self._op_to_aliases.clear()

        op_definitions = self._op_definitions.copy()

        for op_name, op_def in op_definitions.items():
            op_name_norm = self._normalize_op_name(op_name)
            # Build mapping for each alias
            for alias in op_def.aliases:
                alias_norm = self._normalize_op_name(alias)
                # Prevent self-referencing aliases
                if alias_norm == op_name_norm:
                    continue

                # Add alias reference to definitions if not already present (normally should be already present)
                if alias_norm not in self._op_definitions:
                    self._op_definitions[alias_norm] = op_def
                if self._op_definitions[alias_norm] == op_def:
                    self._aliases[alias_norm] = op_name_norm
                    self._op_to_aliases[op_name_norm].append(alias_norm)
                else:
                    rank_zero_warn(
                        f"The alias '{alias}' is already associated with different operation "
                        f"({self._op_definitions[alias_norm]}) so will not be added."
                    )
                # For namespaced operations, also add non-namespaced convenience alias mapping
                # This allows "test_hub_alias" to resolve to "testuser.test.test_op"
                if "." in op_name_norm:
                    # Extract the original (non-namespaced) alias
                    original_alias = (
                        alias_norm.split(".", 3)[-1] if alias_norm.count(".") >= 3 else alias_norm.split(".")[-1]
                    )
                    if original_alias in self._aliases:
                        # If the original alias already exists, ensure it points to the same op_name
                        if self._aliases[original_alias] != op_name_norm:
                            rank_zero_warn(
                                f"The name '{original_alias}' already exists for a different operation. "
                                f"The fully-qualified alias name ({alias_norm}) has been added as an alias "
                                f"for {op_name_norm}."
                            )
                    else:
                        if self._aliases and original_alias != op_name_norm:
                            self._aliases[original_alias] = op_name_norm
                            self._op_to_aliases[op_name_norm].append(alias_norm)

    @_ensure_loaded
    def list_operations(self) -> List[str]:
        """Get a list of all available operation names.

        Returns:
            List of operation names including both native and hub operations
        """
        return list(self._op_definitions.keys())

    @property
    @_ensure_loaded
    def registered_ops(self) -> Dict[str, OpDef]:
        """Get all registered operation definitions without instantiating them."""
        # TODO: return a generator here instead of a dict? May be better to provide a separate method for that
        return {name: op_def for name, op_def in self._op_definitions.items()}

    @_ensure_loaded
    def resolve_alias(self, op_alias: str) -> str | None:
        return self._aliases.get(op_alias, None)

    @_ensure_loaded
    def get_op_aliases(self, op_name: str) -> Dict[str, List[str]]:
        return self._op_to_aliases[op_name]

    @_ensure_loaded
    def get_all_aliases(self) -> Iterator[Tuple[str, str]]:
        """Get all registered operation aliases."""
        for alias, op_name in self._aliases.items():
            yield (alias, op_name)

    def _resolve_name_safe(self, op_name: str, visited: Optional[set] = None) -> str:
        """Safely resolve names with cycle detection."""
        if visited is None:
            visited = set()

        if op_name in visited:
            # Cycle detected, return the original name
            return op_name

        if op_name not in self._aliases:
            return op_name

        visited.add(op_name)
        resolved = self._resolve_name_safe(self._aliases[op_name], visited)
        visited.remove(op_name)

        return resolved

    def _set_default_hub_op_aliases(self) -> None:
        """Ensure operations are accessible both with and without namespaces."""
        # Use existing definitions if no raw definitions provided
        target_ops = self._op_definitions
        current_ops = dict(self._op_definitions)

        for op_name, op_def in current_ops.items():
            # If this is a namespaced operation, also add it without namespace
            if "." in op_name:
                # Extract the base name (last part after final dot)
                base_name = op_name.split(".")[-1]

                # Only add if there's no existing operation with that base name
                # and it's not a self-reference
                if base_name in target_ops:
                    # If the base name already exists, ensure it points to the same OpDef
                    if target_ops[base_name] != target_ops[op_name]:
                        rank_zero_warn(
                            f"Base name '{base_name}' already has an assigned op or alias so '{op_name}' "
                            "cannot be mapped to it. The fully-qualified name will need to be "
                            "used unless another alias is provided."
                        )
                else:
                    if base_name != op_name:
                        target_ops[base_name] = target_ops[op_name]

            # # Handle aliases - ensure they point to the same OpDef
            for alias in op_def.aliases:
                # Skip self-referencing aliases
                if alias == op_name:
                    continue

                if alias in target_ops:
                    # If alias already exists, ensure it points to the same OpDef
                    if target_ops[alias] != target_ops[op_name]:
                        rank_zero_warn(
                            f"Alias '{alias}' already has an assigned op or alias so the "
                            f"alias specified by '{op_name}' cannot be mapped to it"
                        )
                else:
                    target_ops[alias] = target_ops[op_name]

                # Extract base alias name if it's namespaced
                # This allows "test_hub_op" to resolve to "testuser.test.test_hub_op"
                if "." in alias:
                    base_alias = alias.split(".")[-1]
                    if base_alias in target_ops:
                        # If base alias already exists, ensure it points to the same OpDef
                        if target_ops[base_alias] != target_ops[op_name]:
                            rank_zero_warn(
                                f"Base alias '{base_alias}' already has an assigned op or alias so the alias"
                                f" specified by '{alias}' cannot be mapped to it. The fully-qualified "
                                " name will need to be used unless another alias is provided."
                            )
                    else:
                        if base_alias != op_name and base_alias != alias:
                            target_ops[base_alias] = target_ops[op_name]
        return target_ops

    def _import_callable(self, callable_path: str) -> Callable:
        """Import a callable from a path."""
        module_path, func_name = callable_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            imported_fn = getattr(module, func_name)
        except Exception as e:
            raise ValueError(
                f"Import of the specified function {func_name} from {module_path} (specified callable "
                f"path {callable_path}) failed with the following exception: {e}"
            )
        return imported_fn

    def _import_hub_callable(self, op_name: str, op_def: OpDef) -> Callable:
        """Import a callable from a hub path."""
        rank_zero_debug(f"Attempting dynamic loading for namespaced operation: {op_name}")

        # Extract repo name from the operation name and module/function from implementation field
        # Format of op_name: "repo_name.function_name" or "user.repo.function_name"
        parts = op_name.split(".")
        if len(parts) >= 3:
            # Take the first two parts as repo identifier
            repo_name = ".".join(parts[:2])
        else:
            raise ValueError(f"Invalid namespaced operation format: {op_name}. Expected 'user.repo.function_name'")

        # Extract module and function names from implementation field
        if not op_def.implementation:
            raise ValueError(f"No implementation specified for hub operation: {op_name}")

        implementation_parts = op_def.implementation.split(".")
        if len(implementation_parts) < 2:
            raise ValueError(f"Invalid implementation format: {op_def.implementation}. Expected 'module.function'")

        # Last part is function name, everything before is module path
        function_name = implementation_parts[-1]
        module_name = ".".join(implementation_parts[:-1])

        function_reference = f"{module_name}.{function_name}"

        implementation = get_function_from_dynamic_module(
            function_reference=function_reference,
            op_repo_name_or_path=repo_name,
            cache_dir=IT_ANALYSIS_HUB_CACHE,
        )
        rank_zero_debug(f"Successfully loaded dynamic operation: {op_name}")
        return implementation

    @staticmethod
    def _function_param_from_hub_module(param_path: str, implementation: Callable) -> Optional[Callable]:
        # Try to use the dynamically loaded module if module names match
        func_name = param_path.rsplit(".", 1)[-1]
        param_module = param_path.rsplit(".", 1)[0]
        imported_module_name = implementation.__module__.split(".")[-1]
        resolved_fn_param = None

        if param_module == imported_module_name:
            # Get the module object from the implementation function
            import sys

            module_obj = sys.modules.get(implementation.__module__)
            if module_obj is not None:
                resolved_fn_param = getattr(module_obj, func_name, None)
        return resolved_fn_param

    @_ensure_loaded
    def _instantiate_op(self, op_name: str) -> AnalysisOp:
        """Instantiate an operation from its definition."""
        op_def = self._op_definitions.get(op_name)
        if not op_def:
            raise ValueError(f"Unknown operation: {op_name}")

        # Handle composite operations
        if op_def.composition is not None:
            composition = op_def.composition
            # instantiate each operation in the composition
            ops = [self.get_op(op) for op in composition]
            op = CompositeAnalysisOp(ops, name=op_name, aliases=op_def.aliases)
            op.description = op_def.description
            op.input_schema = op_def.input_schema
            op.output_schema = op_def.output_schema
            return op

        # Check if this is a namespaced operation that needs dynamic loading
        if _is_hub_op := ("." in op_def.name and self.enable_hub_ops):
            implementation = self._import_hub_callable(op_def.name, op_def)
        else:
            # Handle regular operations
            implementation = self._import_callable(op_def.implementation)

        # Build impl_params from importable_params and normal_params
        impl_params = {}

        # Import any additional functions specified in importable_params
        for param_name, param_path in op_def.importable_params.items():
            resolved_fn_param = None
            if _is_hub_op:
                resolved_fn_param = AnalysisOpDispatcher._function_param_from_hub_module(param_path, implementation)
            if not resolved_fn_param:
                resolved_fn_param = self._import_callable(param_path)
            if resolved_fn_param is None:
                rank_zero_warn(
                    f"Importable parameter '{param_name}' in operation '{op_name}' could not be resolved: "
                    f"{param_path}. It will not be available in the operation."
                )
                continue
            impl_params[param_name] = resolved_fn_param

        # Add normal parameters
        impl_params.update(op_def.normal_params)

        op = AnalysisOp(
            name=op_name,
            description=op_def.description,
            output_schema=op_def.output_schema,
            input_schema=op_def.input_schema,
            aliases=op_def.aliases,
            impl_params=impl_params,
        )

        # Set the implementation
        op._impl = implementation

        return op

    def _is_lazy_op_handle(self, obj) -> bool:
        """Check if an object is a lazy operation handle (factory function)."""
        return callable(obj) and not isinstance(obj, AnalysisOp)

    def _convert_to_op_schema(self, schema_dict: Dict) -> OpSchema:
        """Convert a schema dictionary to an OpSchema object with ColCfg values."""
        result = {}
        for field_name, field_config in schema_dict.items():
            if isinstance(field_config, dict):
                result[field_name] = ColCfg(**field_config)
            elif isinstance(field_config, ColCfg):
                result[field_name] = field_config
        return OpSchema(result)

    @_ensure_loaded
    def get_op(
        self, op_name: str, context: Optional[DispatchContext] = None, lazy: bool = False
    ) -> AnalysisOp | Callable:
        """Get an operation by name, optionally instantiating it if needed.

        Args:
            op_name: Name of the operation to retrieve
            context: Optional context for operation dispatching
            lazy: If True, defer instantiation until the operation is actually used

        Returns:
            The requested operation or None if lazy=True and the op hasn't been instantiated yet
        """
        if context is None:
            context = DispatchContext()

        # Resolve names with cycle detection
        resolved_name = self._resolve_name_safe(op_name)

        # Check if operation exists
        if resolved_name not in self._op_definitions:
            raise ValueError(f"Unknown operation: {op_name}")

        # Get or create dispatch table entry for this operation
        if resolved_name not in self._dispatch_table:
            self._dispatch_table[resolved_name] = {}

        ctx_dict = self._dispatch_table[resolved_name]

        # Check if we already have an entry for this context
        if context in ctx_dict:
            existing = ctx_dict[context]
            if lazy:
                # For lazy requests, return whatever we have (factory or instance)
                return existing
            elif self._is_lazy_op_handle(existing):
                # We have a factory function but need an instance
                ctx_dict[context] = self._instantiate_op(resolved_name)
                return ctx_dict[context]
            else:
                # We already have an instantiated operation
                return existing

        # No entry for this context yet
        if lazy:
            # Store a factory function that will instantiate the op when needed
            ctx_dict[context] = lambda: self._instantiate_op(resolved_name)
        else:
            # Eagerly instantiate the operation
            ctx_dict[context] = self._instantiate_op(resolved_name)
        return ctx_dict[context]

    def _maybe_instantiate_op(self, op_ref, context: DispatchContext = DispatchContext()) -> AnalysisOp:
        """Ensure an operation is instantiated based on various reference types."""
        # If it's an OpWrapper, use its _ensure_instantiated method to get the actual op
        if hasattr(op_ref, "_ensure_instantiated") and callable(op_ref._ensure_instantiated):
            return op_ref._ensure_instantiated()  # This now returns the actual op, not the wrapper

        # If it's an AnalysisOp, get the op name
        if isinstance(op_ref, AnalysisOp):
            op_name = op_ref.name
        else:
            assert isinstance(op_ref, str), "op_ref must be an OpWrapper, AnalysisOp or a string"
            op_name = op_ref

        ctx_dict = self._dispatch_table.get(op_name, {})
        op = ctx_dict.get(context)

        # TODO: decide if we want to handle this edge case where the dispatch_table contains a factory function
        #       that was not added by OpWrapper, basically custom op lazy loading
        # Check if the stored value is a factory function or needs to be instantiated
        if callable(op) and not isinstance(op, AnalysisOp):
            # Instantiate the operation and update the dispatch table
            instantiated_op = op()
            ctx_dict[context] = instantiated_op
            return instantiated_op
        elif op is not None:
            return op
        else:
            # Try to get the op if it's not in the dispatch table
            return self.get_op(op_name, context)

    @_ensure_loaded
    def instantiate_all_ops(self) -> Dict[str, AnalysisOp]:
        """Get all operations as instantiated AnalysisOp objects."""
        instantiated_ops = {}

        # Only instantiate operations that are not aliases pointing to other operations
        for op_name in self._op_definitions:
            # Skip if this is an alias that points to a different operation
            if op_name in self._aliases and self._aliases[op_name] != op_name:
                continue

            try:
                op = self.get_op(op_name)
                if isinstance(op, AnalysisOp):
                    instantiated_ops[op_name] = op
            except Exception as e:
                rank_zero_warn(f"Failed to instantiate operation '{op_name}': {e}")
                continue

        return instantiated_ops

    @_ensure_loaded
    def compile_ops(
        self, op_names: str | List[str | AnalysisOp], name: Optional[str] = None, aliases: Optional[List[str]] = None
    ) -> CompositeAnalysisOp:
        """Create a composition of operations from a list of operation names."""
        # See NOTE [Composition and Compilation Limitations]
        # Support for dot-separated string format
        if isinstance(op_names, str):
            op_names = op_names.split(".")
        # If op_names is a list, split any string elements containing '.' into multiple op names
        elif isinstance(op_names, list):
            split_names = []
            for op_name in op_names:
                if isinstance(op_name, str) and "." in op_name:
                    split_names.extend(op_name.split("."))
                else:
                    split_names.append(op_name)
            op_names = split_names

        ops = [self.get_op(op_name) if isinstance(op_name, str) else op_name for op_name in op_names]
        return CompositeAnalysisOp(ops, name=name, aliases=aliases)

    def __call__(
        self,
        op_name: str,
        module: Optional[torch.nn.Module] = None,
        analysis_batch: Optional[BaseAnalysisBatchProtocol] = None,
        batch: Optional[BatchEncoding] = None,
        batch_idx: Optional[int] = None,
    ) -> BaseAnalysisBatchProtocol:
        """Call an operation by name."""
        # Support for dot-separated operation names (creating compositions on-demand)
        if "." in op_name:
            composite_op = self.compile_ops(op_name)
            return composite_op(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=batch_idx)

        # Get the operation, instantiating it if it's a factory function
        op = self.get_op(op_name)

        return op(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=batch_idx)

    def _resolve_op_paths_from_yaml_paths(self):
        """Resolve op_paths from yaml_paths.

        For directories in yaml_paths, add the yaml_path to op_paths. For yaml files, add the direct parent directory of
        the yaml file to op_paths.
        """
        for yaml_path in self.yaml_paths:
            yaml_path = Path(yaml_path).resolve()

            if yaml_path.is_dir():
                # Add directory to op_paths if not already present
                if yaml_path not in self.op_paths:
                    self.op_paths.append(yaml_path)
            elif yaml_path.is_file():
                # Add parent directory of yaml file to op_paths if not already present
                parent_dir = yaml_path.parent
                if parent_dir not in self.op_paths:
                    self.op_paths.append(parent_dir)


# Global dispatcher instance
DISPATCHER = AnalysisOpDispatcher()
