"""Type stubs for Interpretune analysis operations."""
# This file is auto-generated. Do not modify directly.

from typing import Callable, Optional
import torch
from transformers import BatchEncoding
from interpretune.protocol import BaseAnalysisBatchProtocol, DefaultAnalysisBatchProtocol

# Basic operations

def ablation_attribution(
    module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    logit_diff_fn: Callable = ...,
    get_loss_preds_diffs: Callable = ...,
) -> DefaultAnalysisBatchProtocol:
    """Compute attribution values from ablation

    Input Schema:
        input (float32)
        answer_indices (int64)
        alive_latents (int64)
        logit_diffs (float32)
        answer_logits (float32)
        label_ids (int64)
        orig_labels (int64)

    Output Schema:
        attribution_values (float32)
        logit_diffs (float32)
        answer_logits (float32)
        loss (float32)
        preds (int64)

    Function parameter defaults (from YAML):
        logit_diff_fn: interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.ops.definitions.get_loss_preds_diffs
    """
    ...

def get_alive_latents(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Extract alive latents from cache

    Input Schema:
        cache (object)
        answer_indices (int64)

    Output Schema:
        alive_latents (int64)
    """
    ...

def get_answer_indices(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Extract answer indices from batch

    Input Schema:
        input (int64)
        answer_indices (int64)

    Output Schema:
        answer_indices (int64)
    """
    ...

def gradient_attribution(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Compute attribution values from gradients

    Input Schema:
        input (float32)
        answer_indices (int64)
        logit_diffs (float32)
        grad_cache (object)

    Output Schema:
        attribution_values (float32)
        correct_activations (float32)
        prompts (string)
    """
    ...

def labels_to_ids(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding
) -> DefaultAnalysisBatchProtocol:
    """Convert label strings to tensor IDs

    Input Schema:
        labels (string)

    Output Schema:
        label_ids (int64)
        orig_labels (int64)
    """
    ...

def logit_diffs(
    module: torch.nn.Module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    logit_diff_fn: Callable = ...,
    get_loss_preds_diffs: Callable = ...,
) -> DefaultAnalysisBatchProtocol:
    """Clean forward pass for computing logit differences

    Input Schema:
        input (float32)
        label_ids (int64)
        orig_labels (int64)
        answer_logits (float32)
        answer_indices (int64)

    Output Schema:
        loss (float32)
        logit_diffs (float32)
        preds (int64)
        answer_logits (float32)

    Function parameter defaults (from YAML):
        logit_diff_fn: interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.ops.definitions.get_loss_preds_diffs
    """
    ...

def logit_diffs_cache(
    module: torch.nn.Module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    logit_diff_fn: Callable = ...,
    get_loss_preds_diffs: Callable = ...,
) -> DefaultAnalysisBatchProtocol:
    """Clean forward pass for computing logit differences including cache activations (composition only)

    Input Schema:
        input (float32)
        answer_logits (float32)
        answer_indices (int64)
        label_ids (int64)
        orig_labels (int64)
        cache (object)

    Output Schema:
        loss (float32)
        logit_diffs (float32)
        preds (int64)
        answer_logits (float32)

    Function parameter defaults (from YAML):
        logit_diff_fn: interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.ops.definitions.get_loss_preds_diffs
    """
    ...

def model_ablation(
    module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    batch_idx: int,
    ablate_latent_fn: Callable = ...,
) -> DefaultAnalysisBatchProtocol:
    """Model ablation analysis

    Input Schema:
        input (int64)
        alive_latents (int64)
        answer_indices (int64)

    Output Schema:
        answer_logits (float32)

    Function parameter defaults (from YAML):
        ablate_latent_fn: interpretune.analysis.ops.definitions.ablate_sae_latent
    """
    ...

def model_cache_forward(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Model forward pass with cache

    Input Schema:
        input (int64)

    Output Schema:
        answer_logits (float32)
        cache (object)
        prompts (string)
    """
    ...

model_forward_cache = model_cache_forward

def model_forward(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Basic model forward pass

    Input Schema:
        input (int64)
        answer_indices (int64)

    Output Schema:
        answer_logits (float32)
        prompts (string)
    """
    ...

def model_gradient(
    module,
    analysis_batch: DefaultAnalysisBatchProtocol,
    batch: BatchEncoding,
    batch_idx: int,
    logit_diff_fn: Callable = ...,
    get_loss_preds_diffs: Callable = ...,
) -> DefaultAnalysisBatchProtocol:
    """Model gradient-based attribution

    Input Schema:
        input (int64)
        label_ids (int64)
        orig_labels (int64)

    Output Schema:
        answer_logits (float32)
        answer_indices (int64)
        loss (float32)
        logit_diffs (float32)
        preds (int64)
        grad_cache (object)
        prompts (string)

    Function parameter defaults (from YAML):
        logit_diff_fn: interpretune.analysis.ops.definitions.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.ops.definitions.get_loss_preds_diffs
    """
    ...

def sae_correct_acts(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Compute correct activations from SAE cache

    Input Schema:
        logit_diffs (float32)
        answer_indices (int64)
        cache (object)

    Output Schema:
        correct_activations (float32)
    """
    ...

# Composite operations

def logit_diffs_attr_ablation(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_cache_forward.logit_diffs_cache.model_ablation.ablation_attribution
    """
    ...

def logit_diffs_attr_grad(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_gradient.gradient_attribution
    """
    ...

def logit_diffs_base(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_forward.logit_diffs
    """
    ...

def logit_diffs_sae(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_cache_forward.logit_diffs_cache.sae_correct_acts
    """
    ...
