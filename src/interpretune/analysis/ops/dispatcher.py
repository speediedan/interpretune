"""Dispatcher for analysis operations."""
from __future__ import annotations
from typing import Optional, Dict, NamedTuple, List, Tuple, Iterator, Callable
from pathlib import Path
from functools import wraps
from collections import defaultdict
import importlib
import yaml

from transformers import BatchEncoding

from interpretune.analysis.ops.base import AnalysisOp, OpSchema, CompositeAnalysisOp, ColCfg
from interpretune.analysis.ops.auto_columns import apply_auto_columns
from interpretune.protocol import AnalysisBatchProtocol


class DispatchContext(NamedTuple):
    """Context for dispatching operations."""
    pass  # We don't use context keys yet but may in the future


class AnalysisOpDispatcher:
    """Dispatcher for analysis operations.

    This class handles loading operation definitions from YAML and dispatching them based on a given context. Operations
    are dynamically instantiated from their definitions when first accessed.
    """

    # TODO:
    #  - decide whether to make the dispatcher a singleton or not
    #  - decide whether to make the dispatcher thread-safe
    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = yaml_path or Path(__file__).parent / "native_analysis_functions.yaml"
        self._op_definitions = {}
        self._dispatch_table = {}  # {op_name: {context: instantiated_op}}
        self._aliases = {}  # {alias: op_name}
        self._op_to_aliases = defaultdict(list)  # {op_name: [aliases]}
        self._loaded = False
        self._loading_in_progress = False

    def load_definitions(self):
        """Load operation definitions from YAML."""
        if self._loaded or self._loading_in_progress:
            return

        try:
            self._loading_in_progress = True
            with open(self.yaml_path, "r") as f:
                yaml_content = yaml.safe_load(f)

            # First pass: Load individual operations without resolving required_ops
            for op_name, op_def in yaml_content.items():
                # Skip composite operations section
                if op_name == "composite_operations":
                    continue

                # Store the operation definition
                self._op_definitions[op_name] = op_def

                # Register alias if provided
                if "aliases" in op_def:
                    self._op_to_aliases[op_name] = op_def["aliases"]
                    for alias in op_def["aliases"]:
                        self._aliases[alias] = op_name
                        # TODO: incur added complexity of making this a weakref if we start scaling the number of ops
                        self._op_definitions[alias] = op_def

            # Second pass: Compile schemas with required_ops dependencies
            self._compile_required_ops_schemas()

            # Process composite operations with schema compilation
            if "composite_operations" in yaml_content:
                from interpretune.analysis.ops.compiler import build_operation_compositions

                # Apply schema compilation for composite operations
                compiled_ops = build_operation_compositions(yaml_content)

                # Update definitions with compiled operation schemas
                for op_name, op_def in compiled_ops.items():
                    if op_name not in self._op_definitions:
                        self._op_definitions[op_name] = op_def
                    else:
                        # Update existing definition with compiled schemas
                        if "input_schema" in op_def:
                            self._op_definitions[op_name]["input_schema"] = op_def["input_schema"]
                        if "output_schema" in op_def:
                            self._op_definitions[op_name]["output_schema"] = op_def["output_schema"]

                # Register aliases from composite operations
                for comp_name, comp_def in yaml_content["composite_operations"].items():
                    self._op_to_aliases[comp_name] = comp_def.get("aliases", [])
                    if "aliases" in comp_def:
                        for alias in comp_def["aliases"]:
                            self._aliases[alias] = comp_name
            self._loaded = True
        finally:
            self._loading_in_progress = False

    def _compile_required_ops_schemas(self):
        """Compile schemas by recursively including required_ops dependencies."""
        from interpretune.analysis.ops.compiler.schema_compiler import compile_op_schema

        # Track which operations have been compiled to avoid infinite recursion
        compiled = set()

        # Compile all operations
        for op_name in list(self._op_definitions.keys()):
            if op_name not in self._aliases:  # Skip aliases to avoid duplicates
                compile_op_schema(op_name, self._op_definitions, compiled=compiled)
                # Apply optional auto-columns after compilation
                apply_auto_columns(self._op_definitions[op_name])

    def _ensure_loaded(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not self._loaded:
                self.load_definitions()
            return method(self, *args, **kwargs)
        return wrapper

    @property
    @_ensure_loaded
    def registered_ops(self) -> Dict[str, dict]:
        """Get all registered operation definitions without instantiating them."""
        return self._op_definitions.copy()

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

    def _import_callable(self, callable_path: str) -> Callable:
        """Import a callable from a path."""
        module_path, func_name = callable_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    @_ensure_loaded
    def _instantiate_op(self, op_name: str) -> AnalysisOp:
        """Instantiate an operation from its definition."""
        op_def = self._op_definitions.get(op_name)
        if not op_def:
            raise ValueError(f"Unknown operation: {op_name}")

        # Handle composite operations
        if "composition" in op_def:
            composition = op_def["composition"]
            # instantiate each operation in the composition
            ops = [self.get_op(op) for op in composition]
            op = CompositeAnalysisOp(
                ops,
                name=op_name,
                aliases=op_def.get("aliases")
            )
            if "description" in op_def:
                op.description = op_def["description"]
            if "input_schema" in op_def:
                op.input_schema = self._convert_to_op_schema(op_def["input_schema"])
            if "output_schema" in op_def:
                op.output_schema = self._convert_to_op_schema(op_def["output_schema"])
            return op

        # Handle regular operations
        implementation = self._import_callable(op_def["implementation"])
        callables = {"implementation": implementation}

        # Import any additional functions specified in function_params
        if "function_params" in op_def:
            for param_name, param_path in op_def["function_params"].items():
                callables[param_name] = self._import_callable(param_path)

        # Convert schema dictionaries to OpSchema objects with ColCfg values
        input_schema = self._convert_to_op_schema(op_def.get("input_schema", {}))
        output_schema = self._convert_to_op_schema(op_def.get("output_schema", {}))
        aliases = op_def.get("aliases", None)
        op = AnalysisOp(
            name=op_name,
            description=op_def.get("description", ""),
            output_schema=output_schema,
            input_schema=input_schema,
            aliases=aliases,
            callables=callables
        )

        return op

    def _is_lazy_op_handle(self, op_handle) -> bool:
        """Check if the given operation handle is a lazy operation handle."""
        return callable(op_handle) and not isinstance(op_handle, AnalysisOp)

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
    def get_op(self, op_name: str, context: DispatchContext = DispatchContext(),
               lazy: bool = False) -> AnalysisOp | Callable:
        """Get an operation by name, optionally instantiating it if needed.

        Args:
            op_name: Name of the operation to retrieve
            context: Optional context for operation dispatching
            lazy: If True, defer instantiation until the operation is actually used

        Returns:
            The requested operation or None if lazy=True and the op hasn't been instantiated yet
        """
        if op_name not in self._op_definitions:
            raise ValueError(f"Unknown operation: {op_name}")
        if op_name in self._aliases:  # avoid duplicate entries in dispatch table
            return self.get_op(self._aliases[op_name], context, lazy)
        ctx_dict = self._dispatch_table.setdefault(op_name, {})
        if context not in ctx_dict or self._is_lazy_op_handle(ctx_dict[context]):
            if lazy:
                # Store a factory function that will instantiate the op when needed
                ctx_dict[context] = lambda: self._instantiate_op(op_name)
            else:
                # Eagerly instantiate the operation
                ctx_dict[context] = self._instantiate_op(op_name)

        return ctx_dict.get(context)

    def _maybe_instantiate_op(self, op_ref, context: DispatchContext = DispatchContext()) -> AnalysisOp:
        """Ensure an operation is instantiated based on various reference types."""
        # If it's an OpWrapper, use its _ensure_instantiated method to get the actual op
        if hasattr(op_ref, '_ensure_instantiated') and callable(op_ref._ensure_instantiated):
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
        """Instantiate all operations and return them as a dictionary."""
        result = {}
        for op_name in self._op_definitions:
            # Ensure operations are actually instantiated, not just factory functions
            op = self.get_op(op_name)
            result[op_name] = op
        return result

    @_ensure_loaded
    def compile_ops(self, op_names: str | List[str | AnalysisOp], name: Optional[str] = None,
                     aliases: Optional[List[str]] = None) -> CompositeAnalysisOp:
        """Create a composition of operations from a list of operation names."""
        # See NOTE [Composition and Compilation Limitations]
        # Support for dot-separated string format
        if isinstance(op_names, str):
            op_names = op_names.split('.')
        # If op_names is a list, split any string elements containing '.' into multiple op names
        elif isinstance(op_names, list):
            split_names = []
            for op_name in op_names:
                if isinstance(op_name, str) and '.' in op_name:
                    split_names.extend(op_name.split('.'))
                else:
                    split_names.append(op_name)
            op_names = split_names

        ops = [self.get_op(op_name) if isinstance(op_name, str) else op_name for op_name in op_names]
        return CompositeAnalysisOp(ops, name=name, aliases=aliases)

    def __call__(self, op_name: str, module, analysis_batch: Optional[AnalysisBatchProtocol],
                 batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Call an operation by name."""
        # Support for dot-separated operation names (creating compositions on-demand)
        if '.' in op_name:
            composite_op = self.compile_ops(op_name)
            return composite_op(
                module=module,
                analysis_batch=analysis_batch,
                batch=batch,
                batch_idx=batch_idx
            )

        # Get the operation, instantiating it if it's a factory function
        op = self.get_op(op_name)

        return op(
            module=module,
            analysis_batch=analysis_batch,
            batch=batch,
            batch_idx=batch_idx
        )


# Global dispatcher instance
DISPATCHER = AnalysisOpDispatcher()
