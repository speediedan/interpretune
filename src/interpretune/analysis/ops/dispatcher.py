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
    def registered_ops(self) -> Dict[str, AnalysisOp]:
        """Get all registered operations as a dictionary."""
        return self.instantiate_all_ops()

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
        """Get an operation by name, optionally instantiating it if needed."""
        if not self._loaded:
            self.load_definitions()

        if op_name not in self._op_definitions:
            if op_name in self._aliases:
                return self.get_op(self._aliases[op_name], context, lazy)
            raise ValueError(f"Unknown operation: {op_name}")

        ctx_dict = self._dispatch_table.setdefault(op_name, {})
        if context not in ctx_dict and not lazy:
            # Instantiate the operation if it's not already instantiated for this context
            ctx_dict[context] = self._instantiate_op(op_name)

        return ctx_dict.get(context) if context in ctx_dict else None

    def instantiate_all_ops(self) -> Dict[str, AnalysisOp]:
        """Instantiate all operations and return them as a dictionary."""
        if not self._loaded:
            self.load_definitions()

        result = {}
        for op_name in self._op_definitions:
            result[op_name] = self.get_op(op_name)
        return result

    def create_chain(self, chain_str: str) -> ChainedAnalysisOp:
        """Create a chain of operations from a dot-separated string."""
        if not self._loaded:
            self.load_definitions()

        op_names = chain_str.split(".")
        ops = [self.get_op(op_name) for op_name in op_names]
        return ChainedAnalysisOp(ops)

    def create_chain_from_ops(self, ops: List[AnalysisOp]) -> ChainedAnalysisOp:
        """Create a chain of operations from a list of operations."""
        ".".join(op.name for op in ops)
        return ChainedAnalysisOp(ops)

    def __call__(self, op_name: str, module, analysis_batch: Optional[AnalysisBatchProtocol],
                 batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Call an operation by name."""
        op = self.get_op(op_name)
        if op is None:
            raise ValueError(f"Unknown operation: {op_name}")
        return op(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=batch_idx)


# Global dispatcher instance
DISPATCHER = AnalysisOpDispatcher()
