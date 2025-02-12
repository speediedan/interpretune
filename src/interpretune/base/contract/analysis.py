from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Protocol, Union, TypeAlias, Optional, NamedTuple
from collections.abc import Callable, Sequence

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from interpretune.base.config.shared import AnalysisMode


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

class AnalysisBatchProtocol(Protocol):
    """Core analysis batch attributes/methods protocol."""
    # TODO: reduce the number of attributes to the minimum required for any analysis mode
    logit_diffs: torch.Tensor | dict[str, dict[int, torch.Tensor]]
    labels: torch.Tensor
    tokens: torch.Tensor
    prompts: list[str]
    attribution_values: dict[str, torch.Tensor]
    answer_logits: torch.Tensor | dict[str, dict[int, torch.Tensor]]
    loss: torch.Tensor | dict[str, dict[int, torch.Tensor]]
    orig_labels: torch.Tensor
    preds: torch.Tensor | dict[str, dict[int, torch.Tensor]]
    cache: ActivationCacheProtocol
    grad_cache: ActivationCacheProtocol
    answer_indices: torch.Tensor
    correct_activations: dict[str, torch.Tensor]
    alive_latents: dict[str, list[int]]

    def update(self, **kwargs) -> None: ...
    def to_cpu(self) -> None: ...

class SaveCfgProtocol(Protocol):
    """Save configuration protocol."""
    prompts: bool
    tokens: bool
    cache: bool
    grad_cache: bool
    decode_kwargs: dict

    def wrap_summary(self, analysis_batch: AnalysisBatchProtocol,
                    batch: BatchEncoding,
                    tokenizer: PreTrainedTokenizerBase | None = None) -> None: ...

class SAEDictProtocol(Protocol):
    """Protocol for SAE analysis dictionary operations."""
    def shapes(self) -> dict[str, torch.Size | list[torch.Size]]: ...
    def batch_join(self, across_saes: bool = False,
                  join_fn: Callable = torch.cat) -> SAEDictProtocol | list[torch.Tensor]: ...
    def apply_op_by_sae(self, operation: Callable | str,
                       *args, **kwargs) -> SAEDictProtocol: ...

class AnalysisCacheProtocol(Protocol):
    """Protocol verifying core analysis cache functionality."""
    batches: list[AnalysisBatchProtocol]
    save_cfg: SaveCfgProtocol

    def save(self, analysis_batch: AnalysisBatchProtocol,
            batch: BatchEncoding,
            tokenizer: PreTrainedTokenizerBase | None = None) -> None: ...
    def by_sae(self, field_name: str, stack_latents: bool = True) -> SAEDictProtocol: ...
    def __getattr__(self, name: str) -> list: ...


class AnalysisCfgProtocol(Protocol):
    """Protocol verifying core analysis configuration functionality."""
    analysis_cache: AnalysisCacheProtocol
    mode: AnalysisMode
    ablate_latent_fn: Callable
    logit_diff_fn: Callable
    fwd_hooks: list[tuple]
    bwd_hooks: list[tuple]
    cache_dict: dict
    names_filter: NamesFilter | None
    base_logit_diffs: list
    alive_latents: list[dict]
    answer_indices: list[torch.Tensor]

    def check_add_default_hooks(
        self, mode: AnalysisMode,
        names_filter: str | Callable | None,
        cache_dict: dict | None
    ) -> tuple[list[tuple], list[tuple]]: ...

# TODO: create a set of protocol variants for each analysis mode
AnalysisCacheVariants: TypeAlias = AnalysisCacheProtocol
