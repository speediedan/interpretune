"""Base classes for analysis operations."""
from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Literal, Union, Optional, Any, Dict, Callable
from dataclasses import dataclass, fields
import os

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from interpretune.protocol import AnalysisBatchProtocol


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
        # Remove protocol constraint - access any attributes
        return super().__getattr__(name)

    def update(self, **kwargs):
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


DIM_VAR = Literal['batch_size', 'max_answer_tokens', 'num_classes', 'vocab_size', 'max_seq_len']


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
    connected_obj: Literal['analysis_store', 'datamodule'] = 'analysis_store'
    array_shape: tuple[Optional[Union[int, DIM_VAR]], ...] | None = None  # Shape with optional dimension variables
    sequence_type: bool = True  # Default to sequence type for most fields
    array_dtype: str | None = None  # Override for array fields, defaults to datasets_dtype

    def to_dict(self) -> dict:
        """Convert to JSON serializable dict."""
        result = {}
        for f in fields(ColCfg):
            val = getattr(self, f.name)
            if isinstance(val, type):
                val = val.__name__
            result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'ColCfg':
        """Create from dict representation."""
        if 'dtype' in data and isinstance(data['dtype'], str):
            type_map = {'float32': float, 'int64': int, 'string': str}
            data['dtype'] = type_map.get(data['dtype'], eval(data['dtype']))
        return cls(**data)

    def __hash__(self) -> int:
        """Make ColCfg hashable by handling potentially unhashable components."""
        # Convert array_shape to a hashable representation if it contains unhashable elements
        hashable_shape = None
        if self.array_shape is not None:
            # Convert any unhashable elements in array_shape to their string representation
            hashable_shape = tuple(str(dim) if isinstance(dim, (list, dict)) else dim
                                   for dim in self.array_shape)

        # Include all other attributes in the hash
        return hash((
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
        ))


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

    def __hash__(self):
        """Make OpSchema hashable by using a frozenset of items."""
        return hash(frozenset((k, hash(v)) for k, v in sorted(self.items())))

    def __eq__(self, other):
        if not isinstance(other, OpSchema):
            return NotImplemented
        return frozenset(self.items()) == frozenset(other.items())


def wrap_summary(analysis_batch: AnalysisBatchProtocol, batch: BatchEncoding,
                 tokenizer: PreTrainedTokenizerBase | None = None,
                 save_prompts: bool = False, save_tokens: bool = False,
                 decode_kwargs: Optional[dict[str, Any]] = None) -> AnalysisBatchProtocol:
    decode_kwargs = decode_kwargs or {}
    if save_prompts:
        assert tokenizer is not None, "Tokenizer is required to decode prompts"
        analysis_batch.prompts = tokenizer.batch_decode(batch['input'], **decode_kwargs)
    elif hasattr(analysis_batch, 'prompts'):
        del analysis_batch.prompts

    if save_tokens:
        analysis_batch.tokens = batch['input'].detach().cpu()
    elif hasattr(analysis_batch, 'tokens'):
        del analysis_batch.tokens

    # TODO: we need to remove this cache clearing hardcoding to refer to the relevant schema configuration
    #       and enable serialization of these fields based on the schema (on a non-default basis) in the future
    for key in ['cache', 'grad_cache']:
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
    def __init__(self, name: str, description: str,
                 output_schema: OpSchema,
                 input_schema: Optional[OpSchema] = None,
                 active_alias: Optional[str] = None,
                 callables: Optional[Dict[str, Callable]] = None) -> None:
        self.name = name
        self.description = description
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.active_alias = active_alias  # Store the active alias when op is part of a chain
        self._impl = None
        self.callables = callables or {}  # Store functions for operation implementation

    @property
    def alias(self) -> str:
        """Return the active alias if set, otherwise return the name."""
        return self.active_alias if self.active_alias is not None else self.name

    def _validate_input_schema(self, analysis_batch: Optional[AnalysisBatchProtocol], batch: BatchEncoding) -> None:
        """Validate that required inputs defined in input_schema exist in analysis_batch or batch."""
        if self.input_schema is None:
            return

        for key, col_cfg in self.input_schema.items():
            if not col_cfg.required:
                continue

            if col_cfg.connected_obj == 'datamodule':
                # Check in batch for fields from datamodule
                # TODO: decide whether to allow this fallback behavior or require explicit mapping by the op definitions
                # We don't raise an error until we also check if it's already been processed and moved to analysis_batch
                if key not in batch and (analysis_batch is None or
                                          not hasattr(analysis_batch, key) or
                                          getattr(analysis_batch, key) is None):
                    raise ValueError(f"Missing required input '{key}' for {self.name} operation")
            else:  # 'analysis_store'
                # Check in analysis_batch for fields from previous operations
                if analysis_batch is None or not hasattr(analysis_batch, key) or getattr(analysis_batch, key) is None:
                    raise ValueError(f"Missing required analysis input '{key}' for {self.name} operation")

    @staticmethod
    def process_batch(analysis_batch: AnalysisBatchProtocol, batch: BatchEncoding,
                      output_schema: OpSchema, tokenizer: PreTrainedTokenizerBase | None = None,
                      save_prompts: bool = False, save_tokens: bool = False,
                      decode_kwargs: Optional[dict[str, Any]] = None) -> AnalysisBatchProtocol:
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
        analysis_batch = wrap_summary(analysis_batch, batch, tokenizer, save_prompts, save_tokens,
                         decode_kwargs=decode_kwargs)
        # TODO: it probably makes sense to add custom dataset builders to handle these transformations rather than
        #       doing it here for better separation of concerns and internal consistency/api symmetry (e.g. we
        #       have custom formatters for reading back these transformed columns and all serde logic should be
        #       encapsulated at the same level of abstraction)
        # Process column configurations
        for col_name, col_cfg in output_schema.items():
            if (col_name not in analysis_batch.keys()):
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
                        if isinstance(latent_dict, dict) and any(isinstance(v, torch.Tensor) for
                                                                 v in latent_dict.values()):
                            # Split into latents and their corresponding values
                            latents = sorted(latent_dict.keys())  # Sort for consistency
                            per_latent = [latent_dict[k] for k in latents]
                            serialized_dict[hook_name] = {
                                'latents': latents,
                                'per_latent': per_latent
                            }
                        else:
                            # Keep the original value if it's not in the expected per-latent format
                            serialized_dict[hook_name] = latent_dict
                    setattr(analysis_batch, col_name, serialized_dict)

        return analysis_batch

    # TODO: Add a mode where save_batch does not apply dyn_dim serialization transformations? Would allow for
    # wrap_summary/latent transformations to be executed but enable manual dataset construction
    def save_batch(self, analysis_batch: AnalysisBatchProtocol, batch: BatchEncoding,
                  tokenizer: PreTrainedTokenizerBase | None = None, save_prompts: bool = False,
                  save_tokens: bool = False, decode_kwargs: Optional[dict[str, Any]] = None) -> AnalysisBatchProtocol:
        """Save analysis batch using process_batch static method."""
        return self.process_batch(
            analysis_batch=analysis_batch,
            batch=batch,
            output_schema=self.output_schema,
            tokenizer=tokenizer,
            save_prompts=save_prompts,
            save_tokens=save_tokens,
            decode_kwargs=decode_kwargs
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

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Execute the operation using the configured implementation."""
        analysis_batch = analysis_batch or AnalysisBatch()
        # Validate input schema if provided
        if self.input_schema:
            self._validate_input_schema(analysis_batch, batch)

        # Use the implementation function from callables if available
        if "implementation" in self.callables:
            return self.callables["implementation"](module, analysis_batch, batch, batch_idx, **{
                k: v for k, v in self.callables.items() if k != "implementation"
            })

        # Fallback to direct invocation if no implementation is registered
        raise NotImplementedError(f"Operation {self.name} does not have an implementation function registered")


class ChainedAnalysisOp(AnalysisOp):
    """A chain of analysis operations to be executed in sequence."""

    def __init__(self, ops: list[AnalysisOp], alias: Optional[str] = None) -> None:
        # Create a name that combines all operation names
        name = '.'.join(op.name for op in ops)
        description = f"Chain of operations: {' → '.join(op.description for op in ops)}"

        # Import here to avoid circular imports
        from interpretune.analysis.ops.compiler.schema_compiler import jit_compile_chain_schema
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        # Check if alias exists in dispatcher's op_definitions
        if alias and alias in DISPATCHER._op_definitions:
            op_def = DISPATCHER._op_definitions[alias]
            input_schema = op_def['input_schema']
            output_schema = op_def['output_schema']
        else:
            # Compile input and output schemas using the op definitions dictionary
            input_schema, output_schema = jit_compile_chain_schema(
                ops, DISPATCHER._op_definitions
            )

        # Use provided alias or default to the generated name  # TODO: revisit drawbacks of this aliasing scheme
        active_alias = alias or name

        super().__init__(name=name, description=description,
                         output_schema=output_schema, input_schema=input_schema,
                         active_alias=active_alias)
        self.chain = ops

        # TODO: the active op alias should be set in the dispatcher or ITmodule state
        # Set active_alias on each op in the chain
        for op in self.chain:
            op.active_alias = active_alias

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Execute each operation in the chain."""
        result = analysis_batch or AnalysisBatch()
        for op in self.chain:
            result = op(module, result, batch, batch_idx)
        return result

class OpWrapper:
    """A special wrapper for operations that ensures the op is instantiated when accessed directly or when
    attributes are accessed."""

    # Class variable to store module access info
    _target_module = None
    _debugger_identifier = os.environ.get('IT_ENABLE_LAZY_DEBUGGER', '')

    @classmethod
    def initialize(cls, target_module):
        """Set the target module where operations will be registered."""
        cls._target_module = target_module

    def __init__(self, op_name):
        self._op_name = op_name
        self._instantiated_op = None
        self._is_instantiated = False  # Track instantiation status separately

        # Lazy import to avoid circular imports
        self._dispatcher = None

    @property
    def _get_dispatcher(self):
        """Lazily load the dispatcher only when needed."""
        if self._dispatcher is None:
            from interpretune.analysis.ops.dispatcher import DISPATCHER
            self._dispatcher = DISPATCHER
        return self._dispatcher

    def _ensure_instantiated(self):
        """Make sure the operation is instantiated."""
        if not self._is_instantiated:
            # Get the op from the dispatcher
            op = self._get_dispatcher.get_op(self._op_name)
            self._instantiated_op = op
            self._is_instantiated = True

            # Update module attribute to replace wrapper with actual op if target module is set
            if self.__class__._target_module is not None:
                if hasattr(self.__class__._target_module, self._op_name):
                    setattr(self.__class__._target_module, self._op_name, self._instantiated_op)

                # Also update any aliases that point to this wrapper
                for alias, op_name in list(self._get_dispatcher.get_op_aliases()):
                    if op_name == self._op_name and hasattr(self.__class__._target_module, alias):
                        if getattr(self.__class__._target_module, alias) is self:
                            setattr(self.__class__._target_module, alias, self._instantiated_op)

            # Return the instantiated op directly
            return self._instantiated_op
        return self._instantiated_op

    def __getattribute__(self, name):
        """Override to monitor all attribute access."""
        # Allow direct access to critical properties without triggering instantiation
        if name in ('_op_name', '_instantiated_op', '_is_instantiated', '__str__', '__repr__',
                   '__class__', '_get_dispatcher', '_dispatcher', '__reduce__', '__getstate__'):
            return object.__getattribute__(self, name)

        # Special handling for attributes commonly checked by debuggers
        # Only do stack analysis if DEBUGGER_IDENTIFIER is set and we're checking special attrs
        if self.__class__._debugger_identifier and name in ('__iter__', '__len__'):
            import traceback

            # Check for debugger-originated calls
            stack = traceback.extract_stack()
            is_debugger_inspection = any(self.__class__._debugger_identifier in frame.filename for frame in stack)

            if is_debugger_inspection:
                # Return None to make hasattr() return False during debugging
                return None

        # Normal attribute access
        if name not in ('_ensure_instantiated', '__call__'):
            op = self._ensure_instantiated()
            return getattr(op, name)

        return object.__getattribute__(self, name)

    def __call__(self, *args, **kwargs):
        """When called as a function, instantiate and call the real op."""
        op = self._ensure_instantiated()
        return op(*args, **kwargs)

    def __getattr__(self, name):
        """Forward any attribute access to the instantiated op."""
        op = self._ensure_instantiated()
        return getattr(op, name)

    # Override special methods to avoid instantiation
    def __str__(self):
        """Return a string representation that indicates instantiation status."""
        if self._is_instantiated:
            return f"OpWrapper('{self._op_name}', instantiated)"
        return f"OpWrapper('{self._op_name}', not instantiated)"

    def __repr__(self):
        """Return a detailed representation useful for debugging."""
        if self._is_instantiated:
            instantiated_op_repr = repr(self._instantiated_op) if self._is_instantiated else 'None'
            return (
                f"OpWrapper(name='{self._op_name}', instantiated=True, "
                f"op={instantiated_op_repr})"
            )
        return f"OpWrapper(name='{self._op_name}', instantiated=False)"

    # TODO: consider refactoring and simplifying this since we largely only use with pickling of OpWrapper during
    #       dataset fingerprinting operations (we don't really need the wrapper at that point it would seem)
    def __reduce__(self):
        """Handle pickling by converting to the actual operation.

        Instead of trying to pickle the wrapper, this returns information to rebuild the actual operation during
        unpickling.
        """
        # Ensure the operation is instantiated
        op = self._ensure_instantiated()

        # Forward to the operation's own state
        if hasattr(op, '__reduce__'):
            return op.__reduce__()

        # If the operation doesn't have __reduce__, create standard pickle data
        # This uses the operation's class and __dict__ to ensure proper reconstruction
        return (_reconstruct_op,(op.__class__, op.__dict__.copy()))

    def __getstate__(self):
        """Return state for pickling - delegate to the actual operation."""
        op = self._ensure_instantiated()
        if hasattr(op, '__getstate__'):
            return op.__getstate__()
        return op.__dict__.copy()
