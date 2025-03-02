from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Callable, Literal, Union, Optional, Any
from dataclasses import dataclass, fields

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float

from interpretune.protocol import AnalysisBatchProtocol


def boolean_logits_to_avg_logit_diff(
    logits: Float[torch.Tensor, "batch seq 2"],
    target_indices: torch.Tensor,
    reduction: Literal["mean", "sum"] | None = None,
    keep_as_tensor: bool = True,
) -> list[float] | float:
    """Returns the avg logit diff on a set of prompts, with fixed s2 pos and stuff."""
    incorrect_indices = 1 - target_indices
    correct_logits = torch.gather(logits, 2, torch.reshape(target_indices, (-1,1,1))).squeeze()
    incorrect_logits = torch.gather(logits, 2, torch.reshape(incorrect_indices, (-1,1,1))).squeeze()
    logit_diff = correct_logits - incorrect_logits
    if reduction is not None:
        logit_diff = logit_diff.mean() if reduction == "mean" else logit_diff.sum()
    return logit_diff if keep_as_tensor else logit_diff.tolist()

def ablate_sae_latent(
    sae_acts: torch.Tensor,
    hook: HookPoint,  # required by transformer_lens.hook_points._HookFunctionProtocol
    latent_idx: int | None = None,
    seq_pos: torch.Tensor | None = None,  # batched
) -> torch.Tensor:
    """Ablate a particular latent at a particular sequence position.

    If either argument is None, we ablate at all latents / sequence positions.
    """
    sae_acts[torch.arange(sae_acts.size(0)), seq_pos, latent_idx] = 0.0
    return sae_acts

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

    # No need for __eq__ and __hash__ methods anymore as frozen dataclass implements them automatically

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

# Base schema for logit diff operations
LOGIT_DIFF_BASE_SCHEMA = OpSchema({
    'logit_diffs': ColCfg(datasets_dtype="float32"),  # Defaults to sequence_type=True
    'answer_logits': ColCfg(datasets_dtype="float32", sequence_type=False,
                           array_shape=('batch_size', 'max_answer_tokens', 'num_classes')),
    'loss': ColCfg(datasets_dtype="float32", sequence_type=False),  # Scalar value
    'preds': ColCfg(datasets_dtype="int64"),  # Defaults to sequence_type=True
    'labels': ColCfg(datasets_dtype="int64"),  # Defaults to sequence_type=True
    'orig_labels': ColCfg(datasets_dtype="int64"),  # Defaults to sequence_type=True
    'answer_indices': ColCfg(datasets_dtype="int64"),  # Defaults to sequence_type=True
    'tokens': ColCfg(datasets_dtype="int64", required=False, dyn_dim=1,
                     array_shape=(None, 'batch_size'), sequence_type=False),
    'prompts': ColCfg(datasets_dtype="string", required=False, non_tensor=True),
})

# Schema for SAE operations
SAE_SCHEMA = OpSchema({
    **LOGIT_DIFF_BASE_SCHEMA,
    'cache': ColCfg(datasets_dtype="object", non_tensor=True, intermediate_only=True),
    'alive_latents': ColCfg(datasets_dtype="int64", per_sae_hook=True, non_tensor=True, ),
    'correct_activations': ColCfg(datasets_dtype="float32", per_sae_hook=True),
})

# Base schema for attribution operations (without grad_cache)
ATTRIBUTION_BASE_SCHEMA = OpSchema({
    **SAE_SCHEMA,
    'attribution_values': ColCfg(datasets_dtype="float32", per_sae_hook=True),
})

# Complete attribution schema including grad_cache
ATTRIBUTION_SCHEMA = OpSchema({
    **ATTRIBUTION_BASE_SCHEMA,
    'grad_cache': ColCfg(datasets_dtype="object", non_tensor=True, intermediate_only=True),
})

# Schema for ablation operation outputs (extending attribution base schema with per_latent fields)
ABLATION_SCHEMA = OpSchema({
    **ATTRIBUTION_BASE_SCHEMA,
    'logit_diffs': ColCfg(datasets_dtype="float32", per_latent=True),
    'answer_logits': ColCfg(datasets_dtype="float32", per_latent=True, sequence_type=False,
                           array_shape=('batch_size', 'max_answer_tokens', 'num_classes')),
    'loss': ColCfg(datasets_dtype="float32", per_latent=True, sequence_type=False),
    'preds': ColCfg(datasets_dtype="int64", per_latent=True),
})

LOGIT_DIFF_BASE_INPUT_SCHEMA = OpSchema({
    'input': ColCfg(datasets_dtype="float32", connected_obj='datamodule'),
    'labels': ColCfg(datasets_dtype="int64", connected_obj='datamodule'),
})

ABLATION_INPUT_SCHEMA = OpSchema({
         **LOGIT_DIFF_BASE_INPUT_SCHEMA,
        'logit_diffs': ColCfg(datasets_dtype="float32"),
        'alive_latents': ColCfg(datasets_dtype="int64", per_sae_hook=True, non_tensor=True),
        'answer_indices': ColCfg(datasets_dtype="int64"),
})

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

class CallableAnalysisOp:
    """Wrapper to make AnalysisOp callable with the same interface as the operation."""
    def __init__(self, op: AnalysisOp):
        self._op = op
        self.__name__ = op.name
        self.__module__ = "interpretune"
        self.__qualname__ = op.name
        self.__doc__ = op.description

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the wrapped operation."""
        return getattr(self._op, name)

    def __call__(self, *args, **kwargs):
        # The actual implementation will be provided by the concrete operation class
        raise NotImplementedError(
            f"Callable interface for {self._op.name} has not been implemented"
        )

class AnalysisOp:
    """Base class for analysis operations."""
    def __init__(self, name: str, description: str,
                 output_schema: OpSchema,
                 input_schema: Optional[OpSchema] = None) -> None:
        self.name = name
        self.description = description
        self.output_schema = output_schema
        self.input_schema = input_schema
        self._callable = None

    @property
    def callable(self) -> CallableAnalysisOp:
        """Get callable wrapper for this operation."""
        if self._callable is None:
            self._callable = CallableAnalysisOp(self)
        return self._callable

    def save_batch(self, analysis_batch: AnalysisBatchProtocol, batch: BatchEncoding,
                  tokenizer: PreTrainedTokenizerBase | None = None, save_prompts: bool = False,
                  save_tokens: bool = False, decode_kwargs: Optional[dict[str, Any]] = None) -> AnalysisBatchProtocol:
        """Save analysis batch using default wrap_summary behavior."""
        analysis_batch = wrap_summary(analysis_batch, batch, tokenizer, save_prompts, save_tokens,
                          decode_kwargs=decode_kwargs)

        # Process column configurations
        for col_name, col_cfg in self.output_schema.items():
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

class LogitDiffsOp(AnalysisOp):
    """Analysis operation that computes logit differences."""
    def __init__(self, name: str, description: str,
                 output_schema: OpSchema,
                 input_schema: Optional[OpSchema] = None,
                 logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff) -> None:
        super().__init__(name, description, output_schema, input_schema)
        self.logit_diff_fn = logit_diff_fn

class AblationOp(LogitDiffsOp):
    """Analysis operation that performs ablation studies."""
    def __init__(self, name: str, description: str,
                 output_schema: OpSchema,
                 input_schema: Optional[OpSchema] = None,
                 logit_diff_fn: Callable = boolean_logits_to_avg_logit_diff,
                 ablate_latent_fn: Callable = ablate_sae_latent) -> None:
        super().__init__(name, description, output_schema, input_schema, logit_diff_fn)
        self.ablate_latent_fn = ablate_latent_fn

class AnalysisOps(dict):
    """Registry of available analysis operations."""
    def __init__(self):
        super().__init__()
        # TODO: move these native op aliases to be loaded from config similar to module_registry.yaml
        # TODO: make op registration lazy once/if we start to have a reasonable number of ops
        self.op_aliases = {
            'logit_diffs.sae': 'logit_diffs_sae',
            'logit_diffs.base': 'logit_diffs_base',
            'logit_diffs.attribution.ablation': 'logit_diffs_attr_ablation',
            'logit_diffs.attribution.grad_based': 'logit_diffs_attr_grad',
        }
        self._register_defaults()

    def register(self, op: AnalysisOp, alias: str | None = None) -> None:
        """Register a new analysis operation with optional alias."""
        self[op.name] = op
        if alias is not None:
            self.op_aliases[op.name] = alias

    def iter_aliased_ops(self):
        """Iterate through (op, alias) pairs for all registered ops with aliases."""
        for op_name, alias in self.op_aliases.items():
            op = self.get(op_name)
            if op is not None:
                yield op, alias

    def get_op(self, op_name: str) -> Optional[AnalysisOp]:
        """Get registered operation."""
        return self.get(op_name)

    def _register_defaults(self) -> None:
        """Register default analysis operations."""
        self.register(LogitDiffsOp(
            name='logit_diffs.base',
            description='Clean forward pass without SAE',
            output_schema=LOGIT_DIFF_BASE_SCHEMA,
            input_schema=LOGIT_DIFF_BASE_INPUT_SCHEMA,
        ))

        self.register(LogitDiffsOp(
            name='logit_diffs.sae',
            description='Clean forward pass with SAE',
            output_schema=SAE_SCHEMA,
            input_schema=LOGIT_DIFF_BASE_INPUT_SCHEMA,
        ))

        self.register(AblationOp(
            name='logit_diffs.attribution.ablation',
            description='Ablation analysis',
            output_schema=ABLATION_SCHEMA,
            input_schema=ABLATION_INPUT_SCHEMA,
        ))

        self.register(LogitDiffsOp(
            name='logit_diffs.attribution.grad_based',
            description='Attribution patching analysis',
            output_schema=ATTRIBUTION_SCHEMA,
            input_schema=LOGIT_DIFF_BASE_INPUT_SCHEMA,
        ))

# Global registry instance
ANALYSIS_OPS = AnalysisOps()
