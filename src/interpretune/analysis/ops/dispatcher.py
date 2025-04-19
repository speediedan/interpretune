"""Dispatcher for analysis operations."""
from __future__ import annotations
from typing import Optional, Dict, NamedTuple, List, Tuple, Iterator, Callable
from pathlib import Path
import yaml
import importlib

from transformers import BatchEncoding

from interpretune.analysis.ops.base import AnalysisOp, OpSchema, ChainedAnalysisOp, ColCfg
from interpretune.protocol import AnalysisBatchProtocol


class DispatchContext(NamedTuple):
    """Context for dispatching operations."""
    pass  # We don't use context keys yet but may in the future


class AnalysisOpDispatcher:
    """Dispatcher for analysis operations.

    This class handles loading operation definitions from YAML and dispatching them based on a given context. Operations
    are dynamically instantiated from their definitions when first accessed.
    """

    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = yaml_path or Path(__file__).parent / "native_analysis_functions.yaml"
        self._op_definitions = {}
        self._dispatch_table = {}  # {op_name: {context: instantiated_op}}
        self._aliases = {}  # {alias: op_name}
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

            # Load individual operations
            for op_name, op_def in yaml_content.items():
                # Skip composite operations section
                if op_name == "composite_operations":
                    continue

                # Store the operation definition
                self._op_definitions[op_name] = op_def

                # Register alias if provided
                if "aliases" in op_def:
                    for alias in op_def["aliases"]:
                        self._aliases[alias] = op_name

            # Process composite operations with schema compilation
            if "composite_operations" in yaml_content:
                from interpretune.analysis.ops.compiler import build_operation_chains

                # Apply schema compilation for composite operations
                compiled_ops = build_operation_chains(yaml_content)

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
                    if "alias" in comp_def:
                        self._aliases[comp_def["alias"]] = comp_name

            self._loaded = True
        finally:
            self._loading_in_progress = False

    @property
    def registered_ops(self) -> Dict[str, dict]:
        """Get all registered operation definitions without instantiating them."""
        if not self._loaded:
            self.load_definitions()
        return self._op_definitions.copy()

    def get_op_aliases(self) -> Iterator[Tuple[str, str]]:
        """Get all registered operation aliases."""
        if not self._loaded:
            self.load_definitions()
        for alias, op_name in self._aliases.items():
            yield (alias, op_name)

    def get_by_alias(self, alias: str) -> Optional[AnalysisOp]:
        """Get an operation by its alias."""
        if not self._loaded:
            self.load_definitions()
        if alias in self._aliases:
            return self.get_op(self._aliases[alias])
        return None

    def _import_callable(self, callable_path: str) -> Callable:
        """Import a callable from a path."""
        module_path, func_name = callable_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, func_name)

    def _instantiate_op(self, op_name: str) -> AnalysisOp:
        """Instantiate an operation from its definition."""
        if not self._loaded:
            self.load_definitions()

        op_def = self._op_definitions.get(op_name)
        if not op_def:
            raise ValueError(f"Unknown operation: {op_name}")

        # Handle chained operations
        if "chain" in op_def:
            chain = op_def["chain"]
            if isinstance(chain, str):
                chain = chain.split(".")
            ops = [self.get_op(op) for op in chain]
            op = ChainedAnalysisOp(ops, alias=op_def.get("alias"))
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

        op = AnalysisOp(
            name=op_name,
            description=op_def.get("description", ""),
            output_schema=output_schema,
            input_schema=input_schema,
            active_alias=op_def.get("alias"),
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

    def get_op(self, op_name: str, context: DispatchContext = DispatchContext(), lazy: bool = False) -> AnalysisOp:
        """Get an operation by name, optionally instantiating it if needed.

        Args:
            op_name: Name of the operation to retrieve
            context: Optional context for operation dispatching
            lazy: If True, defer instantiation until the operation is actually used

        Returns:
            The requested operation or None if lazy=True and the op hasn't been instantiated yet
        """
        if not self._loaded:
            self.load_definitions()

        if op_name not in self._op_definitions:
            if op_name in self._aliases:
                return self.get_op(self._aliases[op_name], context, lazy)
            raise ValueError(f"Unknown operation: {op_name}")

        ctx_dict = self._dispatch_table.setdefault(op_name, {})
        if context not in ctx_dict or self._is_lazy_op_handle(ctx_dict[context]):
            if lazy:
                # Store a factory function that will instantiate the op when needed
                ctx_dict[context] = lambda: self._instantiate_op(op_name)
            else:
                # Eagerly instantiate the operation
                ctx_dict[context] = self._instantiate_op(op_name)

        return ctx_dict.get(context)

    def _maybe_instantiate_op(self, op_name: str, context: DispatchContext = DispatchContext()) -> AnalysisOp:
        """Ensure an operation is instantiated, creating it if it's a factory function."""
        ctx_dict = self._dispatch_table.get(op_name, {})
        op = ctx_dict.get(context)

        # Check if the stored value is a factory function
        if callable(op) and not isinstance(op, AnalysisOp):
            # Instantiate the operation and update the dispatch table
            instantiated_op = op()
            ctx_dict[context] = instantiated_op
            return instantiated_op
        return op

    def instantiate_all_ops(self) -> Dict[str, AnalysisOp]:
        """Instantiate all operations and return them as a dictionary."""
        if not self._loaded:
            self.load_definitions()

        result = {}
        for op_name in self._op_definitions:
            # Ensure operations are actually instantiated, not just factory functions
            op = self.get_op(op_name)
            if callable(op) and not isinstance(op, AnalysisOp):
                op = self._maybe_instantiate_op(op_name)
            result[op_name] = op
        return result

    def create_chain(self, op_names: List[str], alias: Optional[str] = None) -> ChainedAnalysisOp:
        """Create a chain of operations from a list of operation names."""
        if not self._loaded:
            self.load_definitions()

        # Support for dot-separated string format
        if len(op_names) == 1 and isinstance(op_names[0], str) and '.' in op_names[0]:
            op_names = op_names[0].split('.')

        ops = [self.get_op(op_name) for op_name in op_names]
        # Ensure all ops are instantiated, not just factory functions
        ops = [op if isinstance(op, AnalysisOp) else self._maybe_instantiate_op(op_name)
               for op_name, op in zip(op_names, ops)]
        return ChainedAnalysisOp(ops, alias=alias)

    def create_chain_from_ops(self, ops: List[AnalysisOp], alias: Optional[str] = None) -> ChainedAnalysisOp:
        """Create a chain from operation instances."""
        # Ensure all ops are properly instantiated
        ops = [op if isinstance(op, AnalysisOp) else self._maybe_instantiate_op(op)
              for op in ops]
        return ChainedAnalysisOp(ops, alias=alias)

    def create_chain_from_string(self, chain_str: str, alias: Optional[str] = None) -> ChainedAnalysisOp:
        """Create a chain of operations from a dot-separated string.

        Args:
            chain_str: Dot-separated string of operation names
            alias: Optional alias for the chain

        Returns:
            ChainedAnalysisOp instance
        """
        return self.create_chain(chain_str.split('.'), alias)

    def __call__(self, op_name: str, module, analysis_batch: Optional[AnalysisBatchProtocol],
                 batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Call an operation by name."""
        # Support for dot-separated operation names (creating chains on-demand)
        if '.' in op_name:
            chain_op = self.create_chain_from_string(op_name)
            return chain_op(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=batch_idx)

        # Get the operation, instantiating it if it's a factory function
        op = self.get_op(op_name)
        if op is None:
            raise ValueError(f"Unknown operation: {op_name}")

        # Check if the op is a factory function and instantiate it if needed
        if callable(op) and not isinstance(op, AnalysisOp):
            op = self._maybe_instantiate_op(op_name)

        return op(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=batch_idx)


# Global dispatcher instance
DISPATCHER = AnalysisOpDispatcher()
