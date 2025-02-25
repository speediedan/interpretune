from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Protocol, Union, Optional, NamedTuple, Any, Dict
from collections.abc import Callable, Sequence
from pathlib import Path
from dataclasses import dataclass, field, fields

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from sae_lens.config import HfDataset


NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str], str]]

class SAEFqn(NamedTuple):
    release: str
    sae_id: str

class ActivationCacheProtocol(Protocol):
    """Core activation cache protocol."""
    cache_dict: dict[str, torch.Tensor]
    has_batch_dim: bool
    has_embed: bool
    has_pos_embed: bool

    def __getitem__(self, key: str | tuple) -> torch.Tensor: ...
    def stack_activation(self, activation_name: str, layer: int = -1,
                        sublayer_type: str | None = None) -> torch.Tensor: ...

DEFAULT_DECODE_KWARGS = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}

class AnalysisBatchProtocol(Protocol):
    """Core analysis batch protocol defining attributes/methods required for analysis operations.

    Attributes:
        logit_diffs (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Per batch logit differences with shape [batch_size]
        answer_logits (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Model output logits with shape [batch_size, 1, num_classes]
        loss (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Loss values with shape [batch_size]
        labels (Optional[torch.Tensor]):
            Ground truth labels with shape [batch_size]
        orig_labels (Optional[torch.Tensor]):
            Original unmodified labels with shape [batch_size]
        preds (Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]):
            Model predictions with shape [batch_size]
        cache (Optional[ActivationCacheProtocol]):
            Forward pass activation cache
        grad_cache (Optional[ActivationCacheProtocol]):
            Backward pass gradient cache
        answer_indices (Optional[torch.Tensor]):
            Indices of answers with shape [batch_size]
        alive_latents (Optional[dict[str, list[int]]]):
            Active latent indices per SAE hook
        correct_activations (Optional[dict[str, torch.Tensor]]):
            SAE activations after corrections with shape [batch_size, d_sae] for each SAE
        attribution_values (Optional[dict[str, torch.Tensor]]):
            Attribution values per SAE hook
        tokens (Optional[torch.Tensor]):
            Input token IDs
        prompts (Optional[list[str]]):
            Text prompts
    """
    logit_diffs: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    answer_logits: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    loss: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    preds: Optional[torch.Tensor | dict[str, dict[int, torch.Tensor]]]
    labels: Optional[torch.Tensor]
    orig_labels: Optional[torch.Tensor]
    cache: Optional[ActivationCacheProtocol]
    grad_cache: Optional[ActivationCacheProtocol]
    answer_indices: Optional[torch.Tensor]
    alive_latents: Optional[dict[str, list[int]]]
    correct_activations: Optional[dict[str, torch.Tensor]]
    attribution_values: dict[str, torch.Tensor]
    tokens: Optional[torch.Tensor]
    prompts: Optional[list[str]]

    def update(self, **kwargs) -> None: ...
    def to_cpu(self) -> None: ...

class SAEDictProtocol(Protocol):
    """Protocol for SAE analysis dictionary operations."""
    def shapes(self) -> dict[str, torch.Size | list[torch.Size]]: ...
    def batch_join(self, across_saes: bool = False,
                  join_fn: Callable = torch.cat) -> SAEDictProtocol | list[torch.Tensor]: ...
    def apply_op_by_sae(self, operation: Callable | str,
                       *args, **kwargs) -> SAEDictProtocol: ...

class AnalysisStoreProtocol(Protocol):
    """Protocol verifying core analysis store functionality."""
    dataset: HfDataset
    streaming: bool
    cache_dir: str | None
    save_dir: Path
    dataset_trust_remote_code: bool
    stack_batches: bool
    split: str
    op_output_dataset_path: str | None

    def by_sae(self, field_name: str, stack_latents: bool = True) -> SAEDictProtocol: ...
    def __getattr__(self, name: str) -> list: ...
    def reset(self) -> None: ...

class AnalysisCfgProtocol(Protocol):
    """Protocol verifying core analysis configuration functionality."""
    analysis_store: AnalysisStoreProtocol
    op: AnalysisOp
    ablate_latent_fn: Callable
    logit_diff_fn: Callable
    fwd_hooks: list[tuple]
    bwd_hooks: list[tuple]
    cache_dict: dict
    names_filter: NamesFilter | None
    base_logit_diffs: list
    alive_latents: list[dict]
    answer_indices: list[torch.Tensor]
    # Save configuration fields
    prompts: bool
    tokens: bool
    decode_kwargs: dict

    def check_add_default_hooks(
        self, op: AnalysisOp,
        names_filter: str | Callable | None,
        cache_dict: dict | None
    ) -> tuple[list[tuple], list[tuple]]: ...

class SAEAnalysisProtocol(Protocol):
    """Protocol for SAE analysis components requiring a subset of SAEAnalysisMixin methods."""

    def construct_names_filter(
        self,
        target_layers: list[int],
        sae_hook_match_fn: Callable[[str, list[int] | None], bool]
    ) -> NamesFilter:
        ...

    def get_latents_and_indices(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        analysis_batch: AnalysisBatchProtocol | None = None,
        cache: AnalysisStoreProtocol | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        ...

    def run_with_ctx(
        self,
        analysis_batch: AnalysisBatchProtocol,
        batch: dict[str, Any],
        batch_idx: int,
        **kwargs: Any
    ) -> None:
        ...

    def loss_and_logit_diffs(
        self,
        analysis_batch: AnalysisBatchProtocol,
        batch: dict[str, Any],
        batch_idx: int
    ) -> None:
        ...

@dataclass
class ColCfg:
    """Defines Column-Specific IT Formatter configuration."""
    dyn_dim: Optional[int] = None
    non_tensor: bool = False
    per_latent: bool = False

    def to_dict(self) -> dict:
        """Convert to JSON serializable dict."""
        return {
            'dyn_dim': self.dyn_dim,
            'non_tensor': self.non_tensor,
            'per_latent': self.per_latent
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ColCfg':
        """Create from dict representation."""
        return cls(**data)

    def __eq__(self, other):
        if not isinstance(other, ColCfg):
            return NotImplemented
        return (self.dyn_dim == other.dyn_dim and
                self.non_tensor == other.non_tensor and
                self.per_latent == other.per_latent)

    def __hash__(self):
        return hash((self.dyn_dim, self.non_tensor, self.per_latent))

    def get_type(self) -> str:
        """Get the feature type for this column configuration."""
        return "sequence" if self.non_tensor else "array2d"

DEFAULT_COL_CFG = {
    'tokens': ColCfg(dyn_dim=1),
    'alive_latents': ColCfg(non_tensor=True)
}
ABLATION_COL_CFG = {
    **DEFAULT_COL_CFG,
    'loss': ColCfg(per_latent=True),
    'logit_diffs': ColCfg(per_latent=True),
    'preds': ColCfg(per_latent=True),
    'answer_logits': ColCfg(per_latent=True),
}

class TensorMeta(NamedTuple):
    """Simple metadata tuple for intermediate tensor results."""
    shape: torch.Size
    dtype: torch.dtype

@dataclass
class OpSchema:
    """Base schema for analysis operations.

    Similar to DTensor OpSchema, this tracks operation requirements and helps decide
    how to handle analysis strategies.

    Args:
        tensor_meta: Optional tensor metadata
        has_sae_fields: Whether op requires SAE field access
        has_attribution: Whether op computes attributions
        has_gradients: Whether op requires gradient computation
        per_latent_outputs: Whether op returns outputs per latent
        optional_prompts: Whether prompts field is optional
        optional_tokens: Whether tokens field is optional
    """
    tensor_meta: Optional[TensorMeta] = None
    has_sae_fields: bool = False
    has_attribution: bool = False
    has_correct_activations: bool = False
    per_latent_outputs: bool = False
    optional_prompts: bool = True
    optional_tokens: bool = True
    col_cfg: dict[str, ColCfg] = field(default_factory=lambda: DEFAULT_COL_CFG)

    def __post_init__(self):
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        """Cached hash computation that converts col_cfg to hashable form."""
        if self._hash is None:
            # Convert col_cfg to tuple of frozensets for hashing
            hashable_col_cfg = tuple(
                (k, (v.dyn_dim, v.non_tensor))
                for k, v in sorted(self.col_cfg.items())
            )
            fields_list = [getattr(self, f.name) for f in fields(self) if f.name != 'col_cfg']
            self._hash = hash(tuple(fields_list + [hashable_col_cfg]))
        return self._hash

@dataclass
class OpInputSchema:
    """Schema for operation input requirements."""
    required_keys: set[str] = field(default_factory=lambda: {'labels', 'input'})
    optional_keys: set[str] = field(default_factory=set)

    def validate(self, batch: Dict[str, Any]) -> bool:
        """Validate input batch against schema."""
        return all(k in batch for k in self.required_keys)

    # Added hash implementation to make OpInputSchema hashable
    def __hash__(self) -> int:
        return hash((frozenset(self.required_keys), frozenset(self.optional_keys)))

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
    def __init__(self, name: str, description: str, schema: Optional[OpSchema] = None,
                 input_schema: OpInputSchema = None) -> None:
        self.name = name
        self.description = description
        self.schema = schema or OpSchema()
        self.input_schema = input_schema or OpInputSchema()

    def save_batch(self, analysis_batch: AnalysisBatchProtocol, batch: BatchEncoding,
                  tokenizer: PreTrainedTokenizerBase | None = None, save_prompts: bool = False,
                  save_tokens: bool = False, decode_kwargs: Optional[dict[str, Any]] = None) -> AnalysisBatchProtocol:
        """Save analysis batch using default wrap_summary behavior."""
        analysis_batch = wrap_summary(analysis_batch, batch, tokenizer, save_prompts, save_tokens,
                          decode_kwargs=decode_kwargs)

        # Process column configurations
        for col_name, col_cfg in self.schema.col_cfg.items():
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
        return hash((self.name, self.schema, self.input_schema))

    def __repr__(self) -> str:
        """Detailed representation showing schema and input requirements."""
        return (
            f"AnalysisOp(name='{self.name}', "
            f"description='{self.description}', "
            f"schema={self.schema}, "
            f"input_schema={self.input_schema})"
        )

    def __str__(self) -> str:
        """Simple one-line description of the analysis operation."""
        return f"{self.name}: {self.description}"

class AnalysisOps(dict):
    """Registry of available analysis operations."""
    def __init__(self):
        super().__init__()
        self._register_defaults()

    def register(self, op: AnalysisOp) -> None:
        """Register a new analysis operation."""
        self[op.name] = op

    def get_op(self, op_name: str) -> Optional[AnalysisOp]:
        """Get registered operation."""
        return self.get(op_name)

    def _register_defaults(self) -> None:
        """Register default analysis operations."""
        self.register(AnalysisOp(
            name='clean_no_sae',
            description='Clean forward pass without SAE',
            schema=OpSchema(),
        ))

        self.register(AnalysisOp(
            name='clean_w_sae',
            description='Clean forward pass with SAE',
            schema=OpSchema(has_sae_fields=True, has_correct_activations=True),
        ))

        self.register(AnalysisOp(
            name='ablation',
            description='Ablation analysis',
            schema=OpSchema(
                has_sae_fields=True,
                has_attribution=True,
                per_latent_outputs=True,
                col_cfg=ABLATION_COL_CFG
            ),
        ))

        self.register(AnalysisOp(
            name='attr_patching',
            description='Attribution patching analysis',
            schema=OpSchema(
                has_sae_fields=True,
                has_attribution=True,
                has_correct_activations=True
            ),
        ))

# Global registry instance
ANALYSIS_OPS = AnalysisOps()
