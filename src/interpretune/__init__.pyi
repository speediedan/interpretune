"""Type stubs for Interpretune analysis operations."""
# This file is auto-generated. Do not modify directly.

from typing import Callable, Optional
import torch
from transformers import BatchEncoding
from interpretune.protocol import BaseAnalysisBatchProtocol, DefaultAnalysisBatchProtocol

# Main module exports - added for static analysis
# These imports resolve pyright 'unknown import symbol' errors caused by the complex import hook
# mechanism used for analysis operations.
from interpretune.base.datamodules import ITDataModule as ITDataModule
from interpretune.base.components.mixins import MemProfilerHooks as MemProfilerHooks
from interpretune.analysis.ops import AnalysisBatch as AnalysisBatch
from interpretune.analysis import (
    AnalysisStore as AnalysisStore,
    DISPATCHER as DISPATCHER,
    LatentAnalysisTargets as LatentAnalysisTargets,
)
from interpretune.config import (
    ITLensConfig as ITLensConfig,
    SAELensConfig as SAELensConfig,
    PromptConfig as PromptConfig,
    ITDataModuleConfig as ITDataModuleConfig,
    ITConfig as ITConfig,
    GenerativeClassificationConfig as GenerativeClassificationConfig,
    BaseGenerationConfig as BaseGenerationConfig,
    HFGenerationConfig as HFGenerationConfig,
    SAELensFromPretrainedConfig as SAELensFromPretrainedConfig,
    AnalysisCfg as AnalysisCfg,
)
from interpretune.session import ITSessionConfig as ITSessionConfig, ITSession as ITSession
from interpretune.runners import AnalysisRunner as AnalysisRunner
from interpretune.utils import rank_zero_warn as rank_zero_warn, sanitize_input_name as sanitize_input_name
from interpretune.protocol import STEP_OUTPUT as STEP_OUTPUT

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

def compute_attribution_graph(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Generate an attribution graph with circuit-tracer

    Input Schema:
        concept_direction (float32)

    Output Schema:
        input_string (string)
        adjacency_matrix (float32)
        active_features (int64)
        selected_features (int64)
        activation_values (float32)
        logit_target_ids (int64)
        logit_target_tokens (string)
        logit_probabilities (float32)
        input_tokens (int64)
        graph_cfg_json (string)
        graph_scan_json (string)
        graph_vocab_size (int64)
        graph_metadata (string)
    """
    ...

ct_graph = compute_attribution_graph

def concept_direction(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Aggregate latent concept examples into a normalized concept direction vector

    Input Schema:
        concept_latent_state (float32)
        concept_group_id (int64)
        concept_group_name (string)
        concept_example_weight (float32)
        concept_group_a_name (string)
        concept_group_b_name (string)
        concept_label (string)
        concept_group_a (string)
        concept_group_b (string)
        concept_direction_mode (string)

    Output Schema:
        concept_direction (float32)
        concept_label (string)
        concept_group_a_token_ids (int64)
        concept_group_b_token_ids (int64)
        concept_group_a_name (string)
        concept_group_b_name (string)
        concept_direction_mode (string)
    """
    ...

semantic_direction = concept_direction

def direct_concept_direction_intervention(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Add a scaled concept-direction vector to the residual stream and return pre/post intervention logits

    Input Schema:
        concept_direction (float32)
        concept_cache_key (string)
        direction_scale_factor (float32)
        logit_target_ids (int64)

    Output Schema:
        pre_intervention_logits (float32)
        post_intervention_logits (float32)
        logit_diff (float32)
    """
    ...

direction_intervention = direct_concept_direction_intervention

def extract_concept_latent_examples(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Filter and annotate latent rows for concept-direction aggregation

    Input Schema:
        concept_latent_state (float32)
        cache (object)
        answer_indices (int64)
        orig_labels (int64)
        logit_diffs (float32)
        concept_group_a_label_ids (int64)
        concept_group_b_label_ids (int64)
        concept_group_a_name (string)
        concept_group_b_name (string)
        concept_cache_key (string)
        concept_correct_only (int64)
        concept_weight_by_logit_diff (int64)

    Output Schema:
        concept_latent_state (float32)
        concept_group_id (int64)
        concept_group_name (string)
        concept_example_logit_diff (float32)
        concept_example_weight (float32)
        concept_cache_key (string)
        concept_group_a_name (string)
        concept_group_b_name (string)
    """
    ...

concept_latent_examples = extract_concept_latent_examples

def extract_concept_latent_state(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Extract per-example latent rows from the configured cache key

    Input Schema:
        cache (object)
        answer_indices (int64)
        concept_cache_key (string)

    Output Schema:
        concept_latent_state (float32)
        concept_cache_key (string)
    """
    ...

concept_latent_state_from_cache = extract_concept_latent_state

def extract_top_features(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, top_n: int | None = None, **kwargs
) -> AnalysisBatch:
    """Extract top-N influential features from an attribution graph

    Input Schema:
        active_features (int64)
        activation_values (float32)

    Output Schema:
        top_feature_ids (int64)
        top_feature_scores (float32)
        top_feature_activation_values (float32)
    """
    ...

ct_top_features = extract_top_features

def feature_intervention_forward(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Run feature interventions and return pre/post intervention outputs

    Input Schema:
        top_feature_ids (int64)
        top_feature_scores (float32)
        top_feature_activation_values (float32)

    Output Schema:
        intervention_config (string)
        intervention_specs_json (string)
        intervention_layers (int64)
        intervention_positions (int64)
        intervention_feature_ids (int64)
        intervention_values (float32)
        pre_intervention_logits (float32)
        post_intervention_logits (float32)
        logit_diff (float32)
    """
    ...

ct_feature_intervention = feature_intervention_forward

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

def graph_node_influence(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Compute node influence scores for an attribution graph

    Input Schema:
        adjacency_matrix (float32)
        active_features (int64)
        selected_features (int64)
        logit_target_ids (int64)
        logit_target_tokens (string)
        logit_probabilities (float32)
        input_string (string)
        input_tokens (int64)
        activation_values (float32)
        graph_cfg_json (string)
        graph_scan_json (string)
        graph_vocab_size (int64)

    Output Schema:
        node_influence_scores (float32)
        node_feature_ids (int64)
    """
    ...

ct_node_influence = graph_node_influence

def graph_prune(module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs) -> AnalysisBatch:
    """Prune a circuit-tracer attribution graph

    Input Schema:
        input_string (string)
        adjacency_matrix (float32)
        active_features (int64)
        selected_features (int64)
        activation_values (float32)
        logit_target_ids (int64)
        logit_target_tokens (string)
        logit_probabilities (float32)
        input_tokens (int64)
        graph_cfg_json (string)
        graph_scan_json (string)
        graph_vocab_size (int64)

    Output Schema:
        input_string (string)
        adjacency_matrix (float32)
        active_features (int64)
        selected_features (int64)
        activation_values (float32)
        logit_target_ids (int64)
        logit_target_tokens (string)
        logit_probabilities (float32)
        input_tokens (int64)
        graph_cfg_json (string)
        graph_scan_json (string)
        graph_vocab_size (int64)
        graph_metadata (string)
    """
    ...

ct_graph_prune = graph_prune

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

def model_fwd(
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

model_forward = model_fwd

def model_fwd_w_cache(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Model forward pass with activation caching (no latent model hooks)

    Input Schema:
        input (int64)

    Output Schema:
        answer_logits (float32)
        cache (object)
        prompts (string)
    """
    ...

def model_fwd_w_cache_latent_models(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Model forward pass with activation caching and latent model (SAE) hooks

    Input Schema:
        input (int64)

    Output Schema:
        answer_logits (float32)
        cache (object)
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

def attribution_from_concept(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    concept_direction.compute_attribution_graph.graph_node_influence.extract_top_features

    Concept direction through graph attribution and top-feature extraction
    """
    ...

def intervention_from_concept(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    concept_direction.compute_attribution_graph.graph_node_influence.extract_top_features.feature_intervention_forward

    Full analysis-level concept attribution and intervention pipeline
    """
    ...

def intervention_from_features(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    feature_intervention_forward

    Feature intervention from extracted top features
    """
    ...

def logit_diffs_attr_ablation(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_fwd_w_cache_latent_models.logit_diffs_cache.model_ablation.ablation_attribution
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
    labels_to_ids.model_fwd.logit_diffs
    """
    ...

def logit_diffs_sae(
    module, analysis_batch: Optional[BaseAnalysisBatchProtocol], batch, batch_idx: int
) -> BaseAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_fwd_w_cache_latent_models.logit_diffs_cache.sae_correct_acts
    """
    ...
