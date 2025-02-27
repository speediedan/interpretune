from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Callable, NamedTuple, Union, Optional, Any, Protocol, Sequence
from pathlib import Path

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from sae_lens.config import HfDataset

NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str], str]]

class SAEFqn(NamedTuple):
    release: str
    sae_id: str

class AnalysisOpProtocol(Protocol):
    """Protocol defining required interface for analysis operations."""
    name: str
    description: str
    output_schema: dict
    input_schema: Optional[dict]

    def save_batch(self, analysis_batch: AnalysisBatchProtocol, batch: BatchEncoding,
                  tokenizer: PreTrainedTokenizerBase | None = None,
                  save_prompts: bool = False, save_tokens: bool = False,
                  decode_kwargs: Optional[dict] = None) -> AnalysisBatchProtocol: ...

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
    output_store: AnalysisStoreProtocol
    input_store: AnalysisStoreProtocol
    op: AnalysisOpProtocol
    fwd_hooks: list[tuple]
    bwd_hooks: list[tuple]
    cache_dict: dict
    names_filter: NamesFilter | None
    # Save configuration fields
    save_prompts: bool
    save_tokens: bool
    decode_kwargs: dict

    def check_add_default_hooks(
        self, op: AnalysisOpProtocol,
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

class ActivationCacheProtocol(Protocol):
    """Core activation cache protocol."""
    cache_dict: dict[str, torch.Tensor]
    has_batch_dim: bool
    has_embed: bool
    has_pos_embed: bool

    def __getitem__(self, key: str | tuple) -> torch.Tensor: ...
    def stack_activation(self, activation_name: str, layer: int = -1,
                        sublayer_type: str | None = None) -> torch.Tensor: ...

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
    attribution_values: Optional[dict[str, torch.Tensor]]
    tokens: Optional[torch.Tensor]
    prompts: Optional[list[str]]

    def update(self, **kwargs) -> None: ...
    def to_cpu(self) -> None: ...
