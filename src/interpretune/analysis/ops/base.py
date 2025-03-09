"""Base classes for analysis operations."""
from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Literal, Union, Optional, Any
from dataclasses import dataclass, fields

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
    """Contains all analysis results for a single batch.

    Fields:
        logit_diffs: torch.Tensor | dict[str, dict[int, torch.Tensor]]  # [batch_size]
        answer_logits: torch.Tensor | dict[str, dict[int, torch.Tensor]]  # [batch_size, 1, num_classes]
        loss: torch.Tensor | dict[str, dict[int, torch.Tensor]]  # [batch_size]
        labels: torch.Tensor  # [batch_size]
        orig_labels: torch.Tensor  # [batch_size]
        preds: torch.Tensor | dict[str, dict[int, torch.Tensor]]  # [batch_size]
        cache: ActivationCacheProtocol
        grad_cache: ActivationCacheProtocol
        answer_indices: torch.Tensor  # [batch_size]
        alive_latents: dict[str, list[int]]
        correct_activations: dict[str, torch.Tensor]  # [batch_size, d_sae] (for each sae)
        attribution_values: dict[str, torch.Tensor]
        tokens: torch.Tensor
        prompts: list[str]
    """

    def __getattr__(self, name):
        if name not in AnalysisBatchProtocol.__annotations__:
            raise AttributeError(f"'{name}' is not a valid AnalysisBatch attribute")
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


DIM_VAR = Literal['batch_size', 'max_answer_tokens', 'num_classes']


@dataclass(frozen=True)
class ColCfg:
    """Configuration for a dataset column."""
    datasets_dtype: str  # Explicit datasets dtype string (e.g. "float32", "int64")
    required: bool = True
    dyn_dim: Optional[int] = None
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
    if save_prompts:
        assert tokenizer is not None, "Tokenizer is required to decode prompts"
        analysis_batch.prompts = tokenizer.batch_decode(batch['input'], **decode_kwargs)
    if save_tokens:
        analysis_batch.tokens = batch['input'].detach().cpu()
    for key in ['cache', 'grad_cache']:
        if hasattr(analysis_batch, key):
            setattr(analysis_batch, key, None)
    analysis_batch.to_cpu()
    return analysis_batch


class AnalysisOp:
    """Base class for analysis operations."""
    def __init__(self, name: str, description: str,
                 output_schema: OpSchema,
                 input_schema: Optional[OpSchema] = None,
                 active_alias: Optional[str] = None) -> None:
        self.name = name
        self.description = description
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.active_alias = active_alias  # Store the active alias when op is part of a chain
        self._impl = None

    @property
    def alias(self) -> str:
        """Return the active alias if set, otherwise return the name."""
        return self.active_alias if self.active_alias is not None else self.name

    @property
    def callable(self) -> CallableAnalysisOp:
        """Get callable wrapper for this operation."""
        if self._impl is None:
            self._impl = CallableAnalysisOp(self)
        return self._impl

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

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Default implementation - should be overridden by subclasses."""
        raise NotImplementedError(f"Operation {self.name} does not implement __call__")


class CallableAnalysisOp(AnalysisOp):
    """Wrapper to make AnalysisOp callable with the same interface as the operation."""
    def __init__(self, op: AnalysisOp):
        super().__init__(name=op.name,
                         description=op.description,
                         output_schema=op.output_schema,
                         input_schema=op.input_schema)
        self._op = op
        self.__name__ = op.name
        self.__module__ = "interpretune"
        self.__qualname__ = op.name

    def __call__(self, module, analysis_batch: Optional[AnalysisBatchProtocol],
                batch: BatchEncoding, batch_idx: int) -> AnalysisBatchProtocol:
        """Execute the analysis operation on the given batch by delegating to the op."""
        return self._op(module, analysis_batch, batch, batch_idx)

    def __str__(self) -> str:
        """String representation showing the operation name."""
        return f"{self.name}"

    def __repr__(self) -> str:
        """Detailed representation showing the wrapped operation."""
        return f"CallableAnalysisOp(op={self._op!r})"

    # Add missing method to make class compatible with modified dispatcher
    @property
    def callable(self):
        """Return self since this is already a callable wrapper."""
        return self


class ChainedAnalysisOp(AnalysisOp):
    """A chain of analysis operations to be executed in sequence."""

    def __init__(self, ops: list[AnalysisOp], alias: Optional[str] = None) -> None:
        # Create a name that combines all operation names
        name = '.'.join(op.name for op in ops)
        description = f"Chain of operations: {' → '.join(op.description for op in ops)}"

        # The output schema is the output schema of the last operation
        output_schema = ops[-1].output_schema
        # The input schema is the input schema of the first operation
        input_schema = ops[0].input_schema

        # Use provided alias or default to the generated name
        active_alias = alias or name

        super().__init__(name=name, description=description,
                         output_schema=output_schema, input_schema=input_schema,
                         active_alias=active_alias)
        self.chain = ops

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
