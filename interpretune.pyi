"""Type stubs for Interpretune analysis operations."""
# This file is auto-generated. Do not modify directly.

from typing import Callable, Optional
import torch
from transformers import BatchEncoding
from interpretune.protocol import AnalysisBatchProtocol

# Basic operations

def ablation_attribution(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int,
    logit_diff_fn: Callable = "interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff",
    get_loss_preds_diffs: Callable = "interpretune.analysis.ops.definitions.get_loss_preds_diffs"
) -> AnalysisBatchProtocol:
    """Compute attribution values from ablation

Input Schema:
    answer_logits (float32)
    answer_indices (int64)
    alive_latents (int64)
    logit_diffs (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    tokens (int64)
    prompts (string)

Output Schema:
    attribution_values (float32)
    logit_diffs (float32)
    answer_logits (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    answer_indices (int64)
    tokens (int64)
    prompts (string)
    alive_latents (int64)
"""
    ...


def get_alive_latents(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Extract alive latents from cache

Output Schema:
    alive_latents (int64)
    answer_indices (int64)
    cache (object)
"""
    ...


def get_answer_indices(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Extract answer indices from batch

Output Schema:
    answer_indices (int64)
    tokens (int64)
"""
    ...


def gradient_attribution(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Compute attribution values from gradients

Input Schema:
    answer_indices (int64)
    logit_diffs (float32)
    grad_cache (object)

Output Schema:
    logit_diffs (float32)
    answer_logits (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    answer_indices (int64)
    tokens (int64)
    prompts (string)
    alive_latents (int64)
    attribution_values (float32)
    correct_activations (float32)
    grad_cache (object)
"""
    ...


def labels_to_ids(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Convert label strings to tensor IDs

Input Schema:
    labels (string)

Output Schema:
    labels (int64)
    orig_labels (int64)
"""
    ...


def logit_diffs(
    module: torch.nn.Module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int,
    logit_diff_fn: Callable = "interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff",
    get_loss_preds_diffs: Callable = "interpretune.analysis.ops.definitions.get_loss_preds_diffs"
) -> AnalysisBatchProtocol:
    """Clean forward pass for computing logit differences

Input Schema:
    input (float32)
    labels (int64)
    orig_labels (int64)

Output Schema:
    logit_diffs (float32)
    answer_logits (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    answer_indices (int64)
    tokens (int64)
    prompts (string)
"""
    ...


def logit_diffs_cache(
    module: torch.nn.Module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int,
    logit_diff_fn: Callable = "interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff",
    get_loss_preds_diffs: Callable = "interpretune.analysis.ops.definitions.get_loss_preds_diffs"
) -> AnalysisBatchProtocol:
    """Clean forward pass for computing logit differences including cache activations (chained only)

Input Schema:
    input (float32)
    labels (int64)
    orig_labels (int64)

Output Schema:
    logit_diffs (float32)
    answer_logits (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    answer_indices (int64)
    tokens (int64)
    prompts (string)
    cache (object)
"""
    ...


def model_ablation(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int,
    ablate_latent_fn: Callable = "interpretune.analysis.ops.definitions.ablate_sae_latent"
) -> AnalysisBatchProtocol:
    """Model ablation analysis

Input Schema:
    logit_diffs (float32)
    answer_logits (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    answer_indices (int64)
    tokens (int64)
    prompts (string)

Output Schema:
    answer_logits (float32)
    answer_indices (int64)
    alive_latents (int64)
    logit_diffs (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    tokens (int64)
    prompts (string)
"""
    ...


def model_cache_forward(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Model forward pass with cache

Output Schema:
    answer_logits (float32)
    answer_indices (int64)
    cache (object)
"""
    ...


def model_forward(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Basic model forward pass

Output Schema:
    answer_logits (float32)
    answer_indices (int64)
"""
    ...


def model_gradient(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int,
    logit_diff_fn: Callable = "interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff",
    get_loss_preds_diffs: Callable = "interpretune.analysis.ops.definitions.get_loss_preds_diffs"
) -> AnalysisBatchProtocol:
    """Model gradient-based attribution

Output Schema:
    answer_logits (float32)
    answer_indices (int64)
    loss (float32)
    logit_diffs (float32)
    preds (int64)
    grad_cache (object)
"""
    ...


def sae_correct_acts(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch: BatchEncoding,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Compute correct activations from SAE cache

Input Schema:
    logit_diffs (float32)
    answer_indices (int64)
    cache (object)

Output Schema:
    logit_diffs (float32)
    answer_logits (float32)
    loss (float32)
    preds (int64)
    labels (int64)
    orig_labels (int64)
    answer_indices (int64)
    tokens (int64)
    prompts (string)
    cache (object)
    alive_latents (int64)
    correct_activations (float32)
"""
    ...


# Composite operations

def logit_diffs_attr_ablation(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Chain of operations: labels_to_ids.model_cache_forward.logit_diffs_cache.model_ablation.ablation_attribution
    """
    ...


def logit_diffs_attr_grad(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Chain of operations: labels_to_ids.model_gradient.gradient_attribution
    """
    ...


def logit_diffs_base(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Chain of operations: labels_to_ids.model_forward.logit_diffs
    """
    ...


def logit_diffs_sae(
    module,
    analysis_batch: Optional[AnalysisBatchProtocol],
    batch,
    batch_idx: int
) -> AnalysisBatchProtocol:
    """Chain of operations: labels_to_ids.model_cache_forward.logit_diffs_cache.sae_correct_acts
    """
    ...
