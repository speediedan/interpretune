"""Base classes for analysis operations."""

from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Literal, Union, Optional, Any, Dict, Callable, Sequence
from dataclasses import dataclass, fields
from contextlib import contextmanager
import os

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from interpretune.protocol import BaseAnalysisBatchProtocol

# Module-level constants for default operation parameters
DEFAULT_OP_PARAMS = {"module": None, "analysis_batch": None, "batch": None, "batch_idx": None}

DEFAULT_OP_PARAM_NAMES = frozenset(DEFAULT_OP_PARAMS.keys())


def build_call_args(module, analysis_batch, batch, batch_idx, impl_params=None, **kwargs):
    """Build arguments for operation calls.

    Args:
        module: The module instance
        analysis_batch: The analysis batch
        batch: The input batch
        batch_idx: The batch index
        impl_params: Implementation-specific parameters
        **kwargs: Additional keyword arguments

    Returns:
        Dictionary of arguments for the operation call
    """
    args = {"module": module, "analysis_batch": analysis_batch, "batch": batch, "batch_idx": batch_idx}
    if impl_params:
        args.update(impl_params)
    args.update(kwargs)
    return args


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, name, value):
        super().__setitem__(name, value)

    def __delattr__(self, name):
        try:
            super().__delitem__(name)
        except KeyError as e:
            raise AttributeError(e)


class AnalysisBatch(AttrDict):
    def __getattr__(self, name):
        return super().__getattr__(name)

    def __eq__(self, other):
        """Compare AnalysisBatch objects, using torch.equal for tensor values.

        N.B. this comparison is not exhaustive and may not work for all objects/edge cases. Its semantics should be
        considered provisional and may change in the future as usage patterns/requirements become clearer.
        """
        if not isinstance(other, (AnalysisBatch, dict)):
            return False

        # Check if both have the same keys
        if set(self.keys()) != set(other.keys()):
            return False

        # Compare each value
        for key in self.keys():
            val1 = self[key]
            val2 = other[key]

            # Handle tensor comparison
            if hasattr(val1, "dtype") and hasattr(val1, "shape") and hasattr(val2, "dtype") and hasattr(val2, "shape"):
                # Both are tensor-like objects, use torch.equal
                try:
                    import torch

                    if torch.is_tensor(val1) and torch.is_tensor(val2):
                        if not torch.equal(val1, val2):
                            return False
                    else:
                        raise TypeError  # catch this to handle objects that are torch tensor-like but not torch tensors
                        # return False  # If they are not both tensors, return False
                except (RuntimeError, TypeError):
                    # Fallback to regular comparison if torch.equal fails
                    # For tensors, try element-wise comparison if possible
                    try:
                        comparison_result = val1 == val2
                        # Check if the result has an .all() method (actual tensor comparison)
                        if hasattr(comparison_result, "all"):
                            if not comparison_result.all():
                                return False
                        else:
                            # For non-tensor objects that return boolean directly
                            if not comparison_result:
                                return False
                    except (RuntimeError, TypeError, AttributeError):
                        # Final fallback for non-comparable objects
                        if val1 is not val2:
                            return False
            else:
                # Regular comparison for non-tensor values
                if val1 != val2:
                    return False

        return True

    def update(self, **kwargs):  # type: ignore[override]
        for key, value in kwargs.items():
            self[key] = value

    def to_cpu(self):
        """Detach and move all field tensors to CPU."""

        def maybe_detach(val, visited=None):
            if visited is None:
                visited = set()
            if id(val) in visited:
                return val
            visited.add(id(val))
            if isinstance(val, torch.Tensor):
                return val.detach().cpu()
            elif isinstance(val, dict):
                return {k: maybe_detach(v, visited) for k, v in val.items()}
            return val

        for key, value in list(self.items()):
            self[key] = maybe_detach(value)


DIM_VAR = Literal["batch_size", "max_answer_tokens", "num_classes", "vocab_size", "max_seq_len"]


@dataclass(frozen=True)
class ColCfg:
    """Configuration for a dataset column."""

    datasets_dtype: str  # Explicit datasets dtype string (e.g. "float32", "int64")
    required: bool = True
    dyn_dim: Optional[int] = None
    dyn_dim_ceil: Optional[DIM_VAR] = None  # helper for dynamic dimension handling in some contexts
    non_tensor: bool = False
    per_latent: bool = False
    per_sae_hook: bool = False  # For fields that have per-SAE hook subfields
    intermediate_only: bool = False  # Indicates column used in processing but not written to output
    connected_obj: Literal["analysis_store", "datamodule"] = "analysis_store"
    array_shape: tuple[Optional[Union[int, DIM_VAR]], ...] | None = None  # Shape with optional dimension variables
    sequence_type: bool = True  # Default to sequence type for most fields
    array_dtype: str | None = None  # Override for array fields, defaults to datasets_dtype

    def to_dict(self) -> dict:
        """Convert to JSON serializable dict."""
        result = {}
        for f in fields(ColCfg):
            val = getattr(self, f.name)
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ColCfg":
        """Create from dict representation."""
        # TODO: expected to add custom logic here
        return cls(**data)

    def __hash__(self) -> int:
        """Make ColCfg hashable by handling potentially unhashable components."""
        # Convert array_shape to a hashable representation if it contains unhashable elements
        hashable_shape = None
        if self.array_shape is not None:
            # Convert any unhashable elements in array_shape to their string representation
            hashable_shape = tuple(str(dim) if isinstance(dim, (list, dict)) else dim for dim in self.array_shape)

        # Include all other attributes in the hash
        return hash(
            (
                self.datasets_dtype,
                self.required,
                self.dyn_dim,
                self.non_tensor,
                self.per_latent,
                self.per_sae_hook,
                self.intermediate_only,
                self.connected_obj,
                hashable_shape,
                self.sequence_type,
                self.array_dtype,
            )
        )


class OpSchema(dict):
    """Schema defining column specifications for analysis operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate()

    def _validate(self):
        """Validate all values are ColCfg instances."""
        for val in self.values():
            if not isinstance(val, ColCfg):
                raise TypeError(f"Values must be ColCfg instances, got {type(val)}")

    def __hash__(self):  # type: ignore[override]
        """Make OpSchema hashable by using a frozenset of items."""
        return hash(frozenset((k, hash(v)) for k, v in sorted(self.items())))

    def __eq__(self, other):
        if not isinstance(other, OpSchema):
            return NotImplemented
        return frozenset(self.items()) == frozenset(other.items())


def wrap_summary(
    analysis_batch: BaseAnalysisBatchProtocol,
    batch: BatchEncoding,
    tokenizer: PreTrainedTokenizerBase | None = None,
    save_prompts: bool = False,
    save_tokens: bool = False,
    decode_kwargs: Optional[dict[str, Any]] = None,
) -> BaseAnalysisBatchProtocol:
    decode_kwargs = decode_kwargs or {}
    if save_prompts:
        assert batch["input"] is not None, "Input batch must contain 'input' field for decoding prompts"
        assert tokenizer is not None, "Tokenizer is required to decode prompts"
        analysis_batch.prompts = tokenizer.batch_decode(batch["input"], **decode_kwargs)
    elif hasattr(analysis_batch, "prompts"):
        del analysis_batch.prompts

    if save_tokens:
        assert batch["input"] is not None, "Input batch must contain 'input' field for saving tokens"
        analysis_batch.tokens = batch["input"].detach().cpu()
    elif hasattr(analysis_batch, "tokens"):
        del analysis_batch.tokens

    # TODO: we need to remove this cache clearing hardcoding to refer to the relevant schema configuration
    #       and enable serialization of these fields based on the schema (on a non-default basis) in the future
    for key in ["cache", "grad_cache"]:
        if hasattr(analysis_batch, key):
            setattr(analysis_batch, key, None)
    analysis_batch.to_cpu()
    return analysis_batch


# we use this simple helper function for pickling ops of both AnalysisOp and OpWrapper
def _reconstruct_op(cls, state):
    """Reconstruct an operation from its class and state dictionary."""
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


class AnalysisOp:
    """Base class for analysis operations."""

    def __init__(
        self,
        name: str,
        description: str,
        output_schema: OpSchema,
        input_schema: Optional[OpSchema] = None,
        aliases: Optional[Sequence[str]] = None,
        impl_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.output_schema = output_schema
        self.input_schema = input_schema
        self._ctx_key = None
        self._aliases = aliases  # Store aliases for the operation
        self._impl = None
        self.impl_params = impl_params or {}

    @property
    def ctx_key(self) -> str:
        """Return the context key if set, otherwise return the name."""
        return self._ctx_key if self._ctx_key is not None else self.name

    @contextmanager
    def active_ctx_key(self, ctx_key):
        """Context manager for temporarily setting the active context key.

        Args:
            ctx_key: The context key to set during the context execution
        """
        original_ctx_key = self._ctx_key
        try:
            self._ctx_key = ctx_key
            yield
        finally:
            self._ctx_key = original_ctx_key

    def _validate_input_schema(
        self, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch: Optional[BatchEncoding]
    ) -> None:
        """Validate that required inputs defined in input_schema exist in analysis_batch or batch."""
        if self.input_schema is None:
            return

        for key, col_cfg in self.input_schema.items():
            if not col_cfg.required:
                continue

            if col_cfg.connected_obj == "datamodule":
                # Check in batch for fields from datamodule
                # TODO: decide whether to allow this fallback behavior or require explicit mapping by the op definitions
                # We don't raise an error until we also check if it's already been processed and moved to analysis_batch
                if key not in batch and (
                    analysis_batch is None or not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None
                ):
                    raise ValueError(f"Missing required input '{key}' for {self.name} operation")
            else:  # 'analysis_store'
                # Check in analysis_batch for fields from previous operations
                if analysis_batch is None or not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
                    raise ValueError(f"Missing required analysis input '{key}' for {self.name} operation")

    @staticmethod
    def process_batch(
        analysis_batch: BaseAnalysisBatchProtocol,
        batch: BatchEncoding,
        output_schema: OpSchema,
        tokenizer: PreTrainedTokenizerBase | None = None,
        save_prompts: bool = False,
        save_tokens: bool = False,
        decode_kwargs: Optional[dict[str, Any]] = None,
    ) -> BaseAnalysisBatchProtocol:
        """Process analysis batch using provided output schema.

        This static method handles the common processing logic for analysis batches,
        including token handling and schema-based transformations.

        Args:
            analysis_batch: The analysis batch to process
            batch: The raw batch data
            output_schema: Schema defining the structure of the output
            tokenizer: Optional tokenizer for decoding prompts
            save_prompts: Whether to save prompts
            save_tokens: Whether to save tokens
            decode_kwargs: Additional keyword arguments for decoding

        Returns:
            Processed analysis batch
        """
        # First apply basic wrapping
        analysis_batch = wrap_summary(
            analysis_batch, batch, tokenizer, save_prompts, save_tokens, decode_kwargs=decode_kwargs
        )
        # TODO: it probably makes sense to add custom dataset builders to handle these transformations rather than
        #       doing it here for better separation of concerns and internal consistency/api symmetry (e.g. we
        #       have custom formatters for reading back these transformed columns and all serde logic should be
        #       encapsulated at the same level of abstraction)
        # Process column configurations
        for col_name, col_cfg in output_schema.items():
            if col_name not in analysis_batch.keys():
                continue

            # Handle dynamic dimension swapping
            if col_cfg.dyn_dim is not None:
                tensor = getattr(analysis_batch, col_name)
                if isinstance(tensor, torch.Tensor) and tensor.dim() > col_cfg.dyn_dim:
                    dims = list(range(tensor.dim()))
                    dims[0], dims[col_cfg.dyn_dim] = dims[col_cfg.dyn_dim], dims[0]
                    setattr(analysis_batch, col_name, tensor.permute(*dims))

            # Handle per_latent serialization only if the field actually contains per-latent data
            if col_cfg.per_latent:
                orig_dict = getattr(analysis_batch, col_name)
                if isinstance(orig_dict, dict):
                    serialized_dict = {}
                    for hook_name, latent_dict in orig_dict.items():
                        # Only serialize if the value is actually a dict mapping latents to tensors
                        if isinstance(latent_dict, dict) and any(
                            isinstance(v, torch.Tensor) for v in latent_dict.values()
                        ):
                            # Split into latents and their corresponding values
                            latents = sorted(latent_dict.keys())  # Sort for consistency
                            per_latent = [latent_dict[k] for k in latents]
                            serialized_dict[hook_name] = {"latents": latents, "per_latent": per_latent}
                        else:
                            # Keep the original value if it's not in the expected per-latent format
                            serialized_dict[hook_name] = latent_dict
                    setattr(analysis_batch, col_name, serialized_dict)

        return analysis_batch

    # TODO: Add a mode where save_batch does not apply dyn_dim serialization transformations? Would allow for
    # wrap_summary/latent transformations to be executed but enable manual dataset construction
    def save_batch(
        self,
        analysis_batch: BaseAnalysisBatchProtocol,
        batch: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase | None = None,
        save_prompts: bool = False,
        save_tokens: bool = False,
        decode_kwargs: Optional[dict[str, Any]] = None,
    ) -> BaseAnalysisBatchProtocol:
        """Save analysis batch using process_batch static method."""
        return self.process_batch(
            analysis_batch=analysis_batch,
            batch=batch,
            output_schema=self.output_schema,
            tokenizer=tokenizer,
            save_prompts=save_prompts,
            save_tokens=save_tokens,
            decode_kwargs=decode_kwargs,
        )

    def __eq__(self, other: object) -> bool:
        # Compare based on op name directly.
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, AnalysisOp):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        # Updated hash to use input_schema instead of description.
        return hash((self.name, self.output_schema, self.input_schema))

    def __repr__(self) -> str:
        """Detailed representation showing schema and input requirements."""
        return (
            f"AnalysisOp(name='{self.name}', "
            f"description='{self.description}', "
            f"output_schema={self.output_schema}, "
            f"input_schema={self.input_schema})"
        )

    def __str__(self) -> str:
        """Simple one-line description of the analysis operation."""
        return f"{self.name}: {self.description}"

    def __reduce__(self):
        # TODO: consider more robust serialization in the future
        return (_reconstruct_op, (self.__class__, self.__dict__.copy()))

    @property
    def impl(self) -> Optional[Callable]:
        """Get the implementation function."""
        return self._impl

    def _resolve_call_params(
        self, impl_func: Callable, module, analysis_batch, batch, batch_idx, **kwargs
    ) -> Dict[str, Any]:
        """Resolve parameters to pass to the implementation function using smart parameter detection."""
        import inspect

        # Use centralized parameter building
        available_defaults = build_call_args(
            module, analysis_batch, batch, batch_idx, impl_params=self.impl_params, **kwargs
        )

        try:
            sig = inspect.signature(impl_func)
        except (ValueError, TypeError):
            # If we can't get signature, fall back to passing all defaults
            return available_defaults

        call_args = {}

        # Only pass parameters that the function accepts
        for param_name, param_value in available_defaults.items():
            if param_name in sig.parameters:
                call_args[param_name] = param_value

        return call_args

    def _call_with_resolved_params(self, module, analysis_batch, batch, batch_idx, **kwargs):
        """Unified call method that handles parameter resolution."""
        if self._impl is None:
            raise NotImplementedError(f"Operation {self.name} has no implementation")

        # Use centralized parameter building
        all_params = build_call_args(module, analysis_batch, batch, batch_idx, impl_params=self.impl_params, **kwargs)

        # Resolve parameters for this specific implementation
        resolved_params = self._resolve_call_params(self._impl, **all_params)

        return self._impl(**resolved_params)

    def __call__(
        self,
        module: Optional[torch.nn.Module] = None,
        analysis_batch: Optional[BaseAnalysisBatchProtocol] = None,
        batch: Optional[BatchEncoding] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
    ) -> BaseAnalysisBatchProtocol:
        """Execute the operation using the configured implementation."""
        analysis_batch = analysis_batch or AnalysisBatch()

        # Validate input schema if provided
        if self.input_schema:
            self._validate_input_schema(analysis_batch, batch)

        # Use unified call interface
        result = self._call_with_resolved_params(module, analysis_batch, batch, batch_idx, **kwargs)
        return result


# NOTE: [Composition and Compilation Limitations]
#   - Currently only sequential composition and schema compilation is supported, but the intention
#     is to allow DAG of ops/schemas to be compiled in the future.


class CompositeAnalysisOp(AnalysisOp):
    """A composition of analysis operations to be executed."""

    def __init__(
        self,
        ops: Sequence[AnalysisOp],
        name: Optional[str] = None,
        aliases: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        # Create a name that combines all operation names
        self.composition_name = ".".join(op.name for op in ops)
        description = description or f"Composition of operations: {' → '.join(op.description for op in ops)}"
        self.name = name or self.composition_name

        # Import here to avoid circular imports
        from interpretune.analysis.ops.compiler.schema_compiler import jit_compile_composition_schema
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        # Check if alias exists in dispatcher's op_definitions
        if self.name in DISPATCHER._op_definitions:
            op_def = DISPATCHER._op_definitions[self.name]
            input_schema = op_def.input_schema
            output_schema = op_def.output_schema
        else:
            # Compile input and output schemas using the op definitions dictionary
            input_schema, output_schema = jit_compile_composition_schema(ops, DISPATCHER._op_definitions)  # type: ignore[arg-type]

        super().__init__(
            name=self.name,
            description=description,
            output_schema=output_schema,  # type: ignore[arg-type]
            input_schema=input_schema,  # type: ignore[arg-type]
            aliases=aliases,
            *args,
            **kwargs,
        )
        self.composition = ops

    def __call__(
        self,
        module: Optional[torch.nn.Module] = None,
        analysis_batch: Optional[BaseAnalysisBatchProtocol] = None,
        batch: Optional[BatchEncoding] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
    ) -> BaseAnalysisBatchProtocol:
        """Execute all operations in sequence with automatic parameter resolution."""
        current_batch = analysis_batch or AnalysisBatch()

        for op in self.composition:
            with op.active_ctx_key(self.name):
                # Use centralized parameter building and resolution
                current_batch = op._call_with_resolved_params(module, current_batch, batch, batch_idx, **kwargs)

        return current_batch


class OpWrapper:
    """A special wrapper for operations that ensures the op is instantiated when accessed directly or when
    attributes are accessed."""

    # Class variable to store module access info
    _target_module = None
    _debugger_identifier = os.environ.get("IT_ENABLE_LAZY_DEBUGGER", "")

    # Properties that can be accessed without instantiating the op
    _DIRECT_ACCESS_ATTRS = (
        "_op_name",
        "_instantiated_op",
        "_is_instantiated",
        "__str__",
        "__repr__",
        "__class__",
        "dispatcher",
        "_dispatcher",
        "__reduce__",
        "__reduce_ex__",
    )

    _DEBUG_OVERRIDE_ATTRS = ("__iter__", "__len__")

    def __init__(self, op_name):
        self._op_name = op_name
        self._instantiated_op = None
        self._is_instantiated = False  # Track instantiation status separately

        # Lazy import to avoid circular imports
        self._dispatcher = None

    @classmethod
    def initialize(cls, target_module):
        """Set the target module where operations will be registered."""
        cls._target_module = target_module

    @classmethod
    def register_operations(cls, module, dispatcher):
        """Register all operations from the dispatcher to the module as lazy OpWrapper instances.

        Args:
            module: The module where operations will be registered
            dispatcher: The operations dispatcher instance
        """
        # Set target module for the wrappers
        cls.initialize(module)

        # Set debugger identifier class variable
        cls._debugger_identifier = os.environ.get("IT_ENABLE_LAZY_DEBUGGER", "")

        # Register all operations with lazy getters
        for op_name in dispatcher.registered_ops:
            if dispatcher.resolve_alias(op_name) is not None:
                # Skip aliases
                continue
            # Use lazy=True to avoid instantiation until actual use
            _ = dispatcher.get_op(op_name, lazy=True)
            wrapper = cls(op_name)
            setattr(module, op_name, wrapper)
            for alias in dispatcher.get_op_aliases(op_name):
                setattr(module, alias, wrapper)

    @property
    def dispatcher(self):
        """Lazily load the dispatcher only when needed."""
        if self._dispatcher is None:
            from interpretune.analysis.ops.dispatcher import DISPATCHER

            self._dispatcher = DISPATCHER
        return self._dispatcher

    def _ensure_instantiated(self):
        """Make sure the operation is instantiated."""
        if not self._is_instantiated:
            # Get the op from the dispatcher
            op = self.dispatcher.get_op(self._op_name)
            self._instantiated_op = op
            self._is_instantiated = True

            # Update module attribute to replace wrapper with actual op if target module is set
            if self.__class__._target_module is not None:
                if hasattr(self.__class__._target_module, self._op_name):
                    setattr(self.__class__._target_module, self._op_name, self._instantiated_op)

                # Also update any aliases that point to this wrapper
                for alias in self.dispatcher.get_op_aliases(self._op_name):
                    if hasattr(self.__class__._target_module, alias):
                        setattr(self.__class__._target_module, alias, self._instantiated_op)

            return self._instantiated_op
        return self._instantiated_op

    def __getattribute__(self, name):
        """Override to monitor all attribute access."""
        # Allow direct access to critical properties without triggering instantiation
        if name in type(self)._DIRECT_ACCESS_ATTRS:
            return object.__getattribute__(self, name)

        # Special handling for attributes commonly checked by debuggers
        if type(self)._debugger_identifier and name in type(self)._DEBUG_OVERRIDE_ATTRS:
            import traceback

            stack = traceback.extract_stack()
            is_debugger_inspection = any(type(self)._debugger_identifier in frame.filename for frame in stack)
            if is_debugger_inspection:
                return None

        # Normal attribute access
        if name not in ("_ensure_instantiated", "__call__"):
            op = self._ensure_instantiated()
            return getattr(op, name)

        return object.__getattribute__(self, name)

    def __call__(self, *args, **kwargs):
        """When called as a function, instantiate and call the real op."""
        op = self._ensure_instantiated()
        if op is None:
            raise RuntimeError(f"Failed to instantiate operation '{self._op_name}'")
        return op(*args, **kwargs)

    def __getattr__(self, name):
        """Forward any attribute access to the instantiated op."""
        op = self._ensure_instantiated()
        return getattr(op, name)

    def __str__(self):
        """Return a string representation that indicates instantiation status."""
        if self._is_instantiated:
            return f"OpWrapper('{self._op_name}', instantiated)"
        return f"OpWrapper('{self._op_name}', not instantiated)"

    def __repr__(self):
        """Return a detailed representation useful for debugging."""
        if self._is_instantiated:
            instantiated_op_repr = repr(self._instantiated_op)
            return f"OpWrapper(name='{self._op_name}', instantiated=True, op={instantiated_op_repr})"
        return f"OpWrapper(name='{self._op_name}', instantiated=False)"

    def __reduce__(self):
        """Handle pickling by converting to the actual operation."""
        op = self._ensure_instantiated()
        if getattr(op, "__reduce__", None) and callable(op.__reduce__):
            return op.__reduce__()
        return (_reconstruct_op, (op.__class__, op.__dict__.copy()))


# Type alias for objects that behave like analysis operations (either actual AnalysisOp instances or OpWrapper proxies)
AnalysisOpLike = (OpWrapper, AnalysisOp)
