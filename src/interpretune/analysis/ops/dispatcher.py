"""Dispatcher for analysis operations."""
from __future__ import annotations
from typing import Optional, Dict, NamedTuple, List, Tuple, Iterator
from pathlib import Path
import yaml
import importlib
import inspect

from transformers import BatchEncoding

from interpretune.analysis.ops.base import AnalysisOp, OpSchema, ChainedAnalysisOp, ColCfg
from interpretune.protocol import AnalysisBatchProtocol


class DispatchContext(NamedTuple):
    """Context for dispatching operations."""
    ctx_name: str = "default"  # Default context name


class AnalysisOpDispatcher:
    """Dispatcher for analysis operations.

    This class handles loading operation definitions from YAML and dispatching them based on a given context.
    """

    def __init__(self, yaml_path: Optional[Path] = None):
        """Initialize the dispatcher.

        Args:
            yaml_path: Path to the YAML file containing operation definitions.
                If None, uses the default path.
        """
        self.yaml_path = yaml_path or (Path(__file__).parent / "native_analysis_functions.yaml")
        self._op_definitions: Dict[str, Dict] = {}
        self._dispatch_table: Dict[str, Dict[DispatchContext, AnalysisOp]] = {}
        self._aliases: Dict[str, str] = {}
        self._loaded = False
        self._loading_in_progress = False  # Flag to prevent recursion during loading

    def load_definitions(self):
        """Load operation definitions from YAML."""
        if self._loaded or self._loading_in_progress:
            return

        try:
            self._loading_in_progress = True

            with open(self.yaml_path, "r") as f:
                definitions = yaml.safe_load(f)

            # Load primitive operations
            for op_name, op_def in definitions.items():
                if op_name == "composite_operations":
                    continue

                self._op_definitions[op_name] = op_def

                # Register aliases
                for alias in op_def.get("aliases", []):
                    self._aliases[alias] = op_name

            # Load composite operations after primitives
            if "composite_operations" in definitions:
                for comp_name, comp_def in definitions["composite_operations"].items():
                    if "chain" in comp_def:
                        self._op_definitions[comp_name] = {
                            "chain": comp_def["chain"],
                            "description": f"Composite operation: {comp_def['chain']}",
                            "alias": comp_def.get("alias")
                        }
                        # Register alias if provided
                        if "alias" in comp_def:
                            self._aliases[comp_def["alias"]] = comp_name

            self._loaded = True
        finally:
            self._loading_in_progress = False

    def get_op_aliases(self) -> Iterator[Tuple[str, str]]:
        """Get all operation aliases without instantiating operations.

        Yields:
            Tuples of (operation name, alias)
        """
        if not self._loaded:
            self.load_definitions()

        # Yield registered aliases
        for alias, op_name in self._aliases.items():
            yield op_name, alias

        # Also yield each op as its own alias
        for op_name in self._op_definitions:
            yield op_name, op_name

    def _instantiate_op(self, op_name: str) -> AnalysisOp:
        """Instantiate an operation from its definition.

        Args:
            op_name: Name of the operation to instantiate.

        Returns:
            Instantiated operation.
        """
        if not self._loaded:
            self.load_definitions()

        if op_name not in self._op_definitions:
            # Try to resolve by alias
            if op_name in self._aliases:
                op_name = self._aliases[op_name]
            else:
                raise ValueError(f"Unknown operation: {op_name}")

        op_def = self._op_definitions[op_name]

        # Handle chained operations
        if "chain" in op_def:
            chain_ops = []
            for chain_op_name in op_def["chain"].split("."):
                # For chained ops, we need to check if it's already instantiated
                if chain_op_name in self._dispatch_table and DispatchContext() in self._dispatch_table[chain_op_name]:
                    chain_ops.append(self._dispatch_table[chain_op_name][DispatchContext()])
                else:
                    # Recursive instantiation for each operation in the chain
                    chain_op = self._instantiate_op(chain_op_name)
                    # Cache it to avoid recursive instantiation loops
                    if chain_op_name not in self._dispatch_table:
                        self._dispatch_table[chain_op_name] = {}
                    self._dispatch_table[chain_op_name][DispatchContext()] = chain_op
                    chain_ops.append(chain_op)

            # Pass the alias if it exists in the definition
            alias = op_def.get("alias")
            return ChainedAnalysisOp(chain_ops, alias=alias)

        # Create schema objects from definition
        output_schema = OpSchema({
            k: ColCfg(**v) for k, v in op_def.get("output_schema", {}).items()
        })

        input_schema = None
        if "input_schema" in op_def:
            input_schema = OpSchema({
                k: ColCfg(**v) for k, v in op_def["input_schema"].items()
            })

        # Instantiate the operation class
        class_path = op_def["class_path"]
        description = op_def.get("description", "")

        # Import the class and create an instance
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Check the signature to determine how to instantiate
        sig = inspect.signature(cls.__init__)
        params = {}

        if "name" in sig.parameters:
            params["name"] = op_name
        if "description" in sig.parameters:
            params["description"] = description
        if "output_schema" in sig.parameters:
            params["output_schema"] = output_schema
        if "input_schema" in sig.parameters:
            params["input_schema"] = input_schema

        # Add any additional parameters from the definition
        for param_name in sig.parameters:
            if param_name in op_def and param_name not in params and param_name != "self":
                params[param_name] = op_def[param_name]

        return cls(**params)

    def get_op(self, op_name: str, context: DispatchContext = DispatchContext(), lazy: bool = False) -> AnalysisOp:
        """Get an operation by name and context.

        Args:
            op_name: Name of the operation to get.
            context: Dispatch context.
            lazy: If True, return a callable factory function instead of materializing the operation.
                This avoids infinite recursion when importing modules.

        Returns:
            The operation instance or a factory function if lazy=True.
        """
        if not self._loaded:
            self.load_definitions()

        # Try to resolve by alias
        resolved_name = self._aliases.get(op_name, op_name)

        # Check if operation is already instantiated for this context
        if resolved_name in self._dispatch_table and context in self._dispatch_table[resolved_name]:
            return self._dispatch_table[resolved_name][context]

        if lazy:
            # Return a factory function that will materialize the op when called
            def lazy_factory():
                return self._instantiate_op(resolved_name)

            # Make it look like an AnalysisOp by adding essential attributes
            lazy_factory.name = resolved_name
            if resolved_name in self._op_definitions:
                lazy_factory.description = self._op_definitions[resolved_name].get("description", "")

            return lazy_factory

        # Instantiate the operation
        op = self._instantiate_op(resolved_name)

        # Cache the instantiated operation
        if resolved_name not in self._dispatch_table:
            self._dispatch_table[resolved_name] = {}
        self._dispatch_table[resolved_name][context] = op

        return op

    def instantiate_all_ops(self) -> Dict[str, AnalysisOp]:
        """Instantiate all operations and return them as a dictionary."""
        result = {}

        # First instantiate all primitive operations
        for op_name, op_alias in self.get_op_aliases():
            # Skip composite operations in the first pass
            if op_name in self._op_definitions and "chain" in self._op_definitions[op_name]:
                continue

            # Only instantiate if it doesn't already exist
            if op_name not in self._dispatch_table or DispatchContext() not in self._dispatch_table[op_name]:
                try:
                    op = self.get_op(op_name)
                    result[op_alias] = op
                except Exception as e:
                    # Log errors but continue
                    print(f"Warning: Failed to instantiate operation {op_name}: {e}")
                    continue

        # Then instantiate all composite operations (which may depend on primitives)
        for op_name, op_alias in self.get_op_aliases():
            # Only process composite operations in the second pass
            if op_name not in self._op_definitions or "chain" not in self._op_definitions[op_name]:
                continue

            # Only instantiate if it doesn't already exist
            if op_name not in self._dispatch_table or DispatchContext() not in self._dispatch_table[op_name]:
                try:
                    op = self.get_op(op_name)
                    result[op_alias] = op
                except Exception as e:
                    # Log errors but continue
                    print(f"Warning: Failed to instantiate composite operation {op_name}: {e}")
                    continue

        return result

    def __call__(self, op_name: str, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int,
                context: DispatchContext = DispatchContext()) -> AnalysisBatchProtocol:
        """Call an operation by name.

        Args:
            op_name: Name of the operation to call.
            module: Module to call the operation on.
            analysis_batch: Analysis batch to operate on.
            batch: Input batch.
            batch_idx: Batch index.
            context: Dispatch context.

        Returns:
            Result of the operation.
        """
        op = self.get_op(op_name, context)
        return op(module, analysis_batch, batch, batch_idx)

    def create_chain(self, op_chain_str: str,
                    context: DispatchContext = DispatchContext()) -> ChainedAnalysisOp:
        """Create a chained operation from a dot-separated string.

        Args:
            op_chain_str: String of dot-separated operation names.
            context: Dispatch context.

        Returns:
            A ChainedAnalysisOp instance.
        """
        ops = []
        for op_name in op_chain_str.split("."):
            ops.append(self.get_op(op_name, context))
        return ChainedAnalysisOp(ops)

    def create_chain_from_ops(self, ops: List[AnalysisOp], alias: Optional[str] = None) -> ChainedAnalysisOp:
        """Create a chained operation from a list of operations.

        Args:
            ops: List of operations.
            alias: Optional alias for the chain

        Returns:
            A ChainedAnalysisOp instance combining the operations.
        """
        # First validate that the operations can be chained
        self.validate_op_chain(ops)

        # Return a chained analysis op with the given operations
        return ChainedAnalysisOp(ops, alias=alias)

    def get_by_alias(self, alias: str,
                    context: DispatchContext = DispatchContext()) -> Optional[AnalysisOp]:
        """Get an operation by its alias.

        Args:
            alias: Alias of the operation to get.
            context: Dispatch context.

        Returns:
            Operation instance or None if not found.
        """
        if not self._loaded:
            self.load_definitions()

        if alias in self._aliases:
            return self.get_op(self._aliases[alias], context)
        return None

    def validate_op_chain(self, ops: List[AnalysisOp]) -> bool:
        """Validate that a chain of operations is compatible.

        Args:
            ops: List of operations to validate.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        if not ops:
            return False

        # For now, a simple validation: check that output schema of one op
        # has all required inputs for the next op
        for i in range(len(ops) - 1):
            current_op = ops[i]
            next_op = ops[i+1]

            if next_op.input_schema:
                required_inputs = {k for k, v in next_op.input_schema.items() if v.required}
                available_outputs = set(current_op.output_schema.keys())

                missing = required_inputs - available_outputs
                if missing:
                    raise ValueError(
                        f"Operation chain invalid: {next_op.name} requires inputs {missing} "
                        f"which are not provided by {current_op.name}"
                    )

        return True


# Global dispatcher instance
DISPATCHER = AnalysisOpDispatcher()
