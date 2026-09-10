"""Type stubs for Interpretune analysis operations."""
# This file is auto-generated. Do not modify directly.

from typing import Callable, Optional
import torch
from transformers import BatchEncoding

# The public surface, DERIVED from `interpretune.__all__` rather than listed here.
# A .pyi shadows the module it names, so every public name absent from this file is invisible
# to a type checker even though it exists and is exported.
from importlib.metadata import version as version
from interpretune.adapter_registry import ADAPTER_REGISTRY as ADAPTER_REGISTRY
from interpretune.adapters.core import ITModule as ITModule
from interpretune.adapters.lightning import (
    LightningDataModule as LightningDataModule,
    LightningModule as LightningModule,
)
from interpretune.adapters.registration import CompositionRegistry as CompositionRegistry
from interpretune.adapters.sae_lens.adapter import (
    SAELensNNsightModule as SAELensNNsightModule,
    SAELensTLModule as SAELensTLModule,
)
from interpretune.adapters.transformer_lens.adapter import ITLensModule as ITLensModule
from interpretune.analysis import (
    AnalysisBatch as AnalysisBatch,
    AnalysisStore as AnalysisStore,
    DISPATCHER as DISPATCHER,
    LatentAnalysisTargets as LatentAnalysisTargets,
)
from interpretune.base import (
    ITCLI as ITCLI,
    ITDataModule as ITDataModule,
    IT_BASE as IT_BASE,
    MemProfilerHooks as MemProfilerHooks,
    it_init as it_init,
    it_session_end as it_session_end,
)
from interpretune.config import (
    AnalysisArtifactCfg as AnalysisArtifactCfg,
    AnalysisCfg as AnalysisCfg,
    AnalysisRunnerCfg as AnalysisRunnerCfg,
    AutoCompConfig as AutoCompConfig,
    BaseGenerationConfig as BaseGenerationConfig,
    ChatTemplatePromptConfig as ChatTemplatePromptConfig,
    CircuitTracerConfig as CircuitTracerConfig,
    CoreGenerationConfig as CoreGenerationConfig,
    GenerativeClassificationConfig as GenerativeClassificationConfig,
    HFFromPretrainedConfig as HFFromPretrainedConfig,
    HFGenerationConfig as HFGenerationConfig,
    ITConfig as ITConfig,
    ITDataModuleConfig as ITDataModuleConfig,
    ITLensConfig as ITLensConfig,
    ITLensFromPretrainedNoProcessingConfig as ITLensFromPretrainedNoProcessingConfig,
    ITSerializableCfg as ITSerializableCfg,
    ITSharedConfig as ITSharedConfig,
    PromptConfig as PromptConfig,
    SAELensConfig as SAELensConfig,
    SAELensFromPretrainedConfig as SAELensFromPretrainedConfig,
    TLensGenerationConfig as TLensGenerationConfig,
)
from interpretune.extensions import (
    DebugGeneration as DebugGeneration,
    DebugLMConfig as DebugLMConfig,
    MemProfiler as MemProfiler,
    MemProfilerCfg as MemProfilerCfg,
    NeuronpediaConfig as NeuronpediaConfig,
    NeuronpediaIntegration as NeuronpediaIntegration,
)
from interpretune.protocol import (
    Adapter as Adapter,
    AllPhases as AllPhases,
    AllSteps as AllSteps,
    AnalysisOpProtocol as AnalysisOpProtocol,
    AnalysisStoreProtocol as AnalysisStoreProtocol,
    BaseAnalysisBatchProtocol as BaseAnalysisBatchProtocol,
    CorePhases as CorePhases,
    CoreSteps as CoreSteps,
    DefaultAnalysisBatchProtocol as DefaultAnalysisBatchProtocol,
    ITDataModuleProtocol as ITDataModuleProtocol,
    ITModuleProtocol as ITModuleProtocol,
    STEP_OUTPUT as STEP_OUTPUT,
)
from interpretune.registry import (
    ModuleRegistry as ModuleRegistry,
    RegKeyType as RegKeyType,
    RegisteredCfg as RegisteredCfg,
    apply_defaults as apply_defaults,
    gen_module_registry as gen_module_registry,
    instantiate_and_register as instantiate_and_register,
    it_cfg_factory as it_cfg_factory,
)
from interpretune.runners import (
    AnalysisRunner as AnalysisRunner,
    SessionRunner as SessionRunner,
)
from interpretune.session import (
    ITSession as ITSession,
    ITSessionConfig as ITSessionConfig,
)
from interpretune.utils import (
    MisconfigurationException as MisconfigurationException,
    move_data_to_device as move_data_to_device,
    rank_zero_info as rank_zero_info,
    rank_zero_warn as rank_zero_warn,
    sanitize_input_name as sanitize_input_name,
    to_device as to_device,
)

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
        logit_diff_fn: interpretune.analysis.optools.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.optools.get_loss_preds_diffs
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

def extract_concept_latent_examples(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Filter and annotate latent rows for concept-direction aggregation

    Input Schema:
        concept_latent_state (float32)
        cache (object)
        answer_indices (int64)
        context_token_indices (int64)
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
        concept_context_indices (int64)
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
        context_token_indices (int64)
        concept_cache_key (string)

    Output Schema:
        concept_latent_state (float32)
        concept_cache_key (string)
        context_token_indices (int64)
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
        node_influence_scores (float32)
        node_signed_influence_scores (float32)
        node_logit_diff_gradient_scores (float32)

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
        active_features (int64)
        activation_values (float32)
        node_influence_scores (float32)
        node_signed_influence_scores (float32)
        node_logit_diff_gradient_scores (float32)

    Output Schema:
        intervention_config (string)
        intervention_specs_json (string)
        feature_intervention_dict_json (string)
        intervention_layers (int64)
        intervention_positions (int64)
        intervention_feature_ids (int64)
        intervention_values (float32)
        intervention_base_values (float32)
        intervention_score_values (float32)
        intervention_scale_factors (float32)
        pre_intervention_logits (float32)
        post_intervention_logits (float32)
        intervention_activation_cache (object)
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
        node_signed_influence_scores (float32)
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

def jlens_concept_probe(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Cosine of activations against concept tokens' readout-faithful J-lens directions

    Input Schema:
        cache (object)
        jlens_concept_token_ids (int64) (required)
        jlens_apply_final_norm (bool)
        jlens_layer (int64)
        jlens_layer_percentile (float32)
        jlens_cache_key (string)
        jlens_positions (int64)
        jlens_repo_id (string)
        jlens_model_id (string)
        jlens_lens_path (string)

    Output Schema:
        jlens_concept_cosine (float32)
        jlens_concept_token_ids (int64)
        jlens_layer (int64)
        jlens_positions (int64)
        jlens_provenance (object)
        jlens_basis (string)
    """
    ...

jacobian_lens_concept_probe = jlens_concept_probe

def jlens_read(module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs) -> AnalysisBatch:
    """Rank vocabulary tokens by the Jacobian-lens readout at a fitted layer and chosen positions

    Input Schema:
        cache (object)
        jlens_layer (int64)
        jlens_layer_percentile (float32)
        jlens_cache_key (string)
        jlens_positions (int64)
        jlens_top_k (int64)
        jlens_include_rms_scale (bool)
        jlens_repo_id (string)
        jlens_model_id (string)
        jlens_lens_path (string)

    Output Schema:
        jlens_top_token_ids (int64)
        jlens_top_token_scores (float32)
        jlens_top_token_strings (object)
        jlens_layer (int64)
        jlens_positions (int64)
        jlens_include_rms_scale (bool)
        jlens_provenance (object)
    """
    ...

jacobian_lens_read = jlens_read

def jlens_sparse_inventory(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Sparse nonnegative decomposition of an activation over the J-lens dictionary, with its residual

    Input Schema:
        jlens_apply_final_norm (bool)
        cache (object)
        jlens_inventory_k (int64)
        jlens_layer (int64)
        jlens_layer_percentile (float32)
        jlens_cache_key (string)
        jlens_positions (int64)
        jlens_repo_id (string)
        jlens_model_id (string)
        jlens_lens_path (string)

    Output Schema:
        jlens_inventory_token_ids (object)
        jlens_inventory_coefficients (object)
        jlens_inventory_residual_ratio (float32)
        jlens_layer (int64)
        jlens_positions (int64)
        jlens_provenance (object)
        jlens_basis (string)
    """
    ...

jacobian_lens_sparse_inventory = jlens_sparse_inventory

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

def latent_correct_acts(
    module, analysis_batch: DefaultAnalysisBatchProtocol, batch: BatchEncoding, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Compute correct activations from the latent-model activation cache

    Input Schema:
        logit_diffs (float32)
        answer_indices (int64)
        cache (object)

    Output Schema:
        correct_activations (float32)
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
        logit_diff_fn: interpretune.analysis.optools.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.optools.get_loss_preds_diffs
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
        logit_diff_fn: interpretune.analysis.optools.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.optools.get_loss_preds_diffs
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
        ablate_latent_fn: interpretune.analysis.ops.bundled.sae.sae_ops.ablate_sae_latent
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

def model_fwd_intervention(
    module, analysis_batch: AnalysisBatch, batch: BatchEncoding, batch_idx: int, **kwargs
) -> AnalysisBatch:
    """Apply generalized hook-point interventions and return pre/post intervention logits

    Input Schema:
        interventions (object)
        interventions_json (string)
        intervention_hook_pattern (string)
        intervention_tensor (float32)
        intervention_tensors_json (string)
        intervention_mode (string)
        intervention_use_intervention_tensor_as_basis (bool)
        intervention_scale_factor (float32)
        intervention_position_scope (string)
        use_latent_models (bool)
        concept_direction (float32)
        concept_cache_key (string)
        direction_scale_factor (float32)
        logit_target_ids (int64)

    Output Schema:
        pre_intervention_logits (float32)
        post_intervention_logits (float32)
        intervention_position_effect (float32)
        logit_diff (float32)
    """
    ...

direction_intervention = model_fwd_intervention
direct_concept_direction_intervention = model_fwd_intervention

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
        logit_diff_fn: interpretune.analysis.optools.boolean_logits_to_avg_logit_diff
        get_loss_preds_diffs: interpretune.analysis.optools.get_loss_preds_diffs
    """
    ...

# Composite operations

def attribution_from_concept(
    module, analysis_batch: Optional[DefaultAnalysisBatchProtocol], batch, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Composition of operations:
    concept_direction.compute_attribution_graph.graph_node_influence.extract_top_features

    Concept direction through graph attribution and top-feature extraction
    """
    ...

def intervention_from_concept(
    module, analysis_batch: Optional[DefaultAnalysisBatchProtocol], batch, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Composition of operations:
    concept_direction.compute_attribution_graph.graph_node_influence.extract_top_features.feature_intervention_forward

    Full analysis-level concept attribution and intervention pipeline
    """
    ...

def intervention_from_features(
    module, analysis_batch: Optional[DefaultAnalysisBatchProtocol], batch, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Composition of operations:
    feature_intervention_forward

    Feature intervention from extracted top features
    """
    ...

def logit_diffs_attr_ablation(
    module, analysis_batch: Optional[DefaultAnalysisBatchProtocol], batch, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_fwd_w_cache_latent_models.logit_diffs_cache.model_ablation.ablation_attribution
    """
    ...

def logit_diffs_attr_grad(
    module, analysis_batch: Optional[DefaultAnalysisBatchProtocol], batch, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_gradient.gradient_attribution
    """
    ...

def logit_diffs_base(
    module, analysis_batch: Optional[DefaultAnalysisBatchProtocol], batch, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_fwd.logit_diffs
    """
    ...

def logit_diffs_latent(
    module, analysis_batch: Optional[DefaultAnalysisBatchProtocol], batch, batch_idx: int
) -> DefaultAnalysisBatchProtocol:
    """Composition of operations:
    labels_to_ids.model_fwd_w_cache_latent_models.logit_diffs_cache.latent_correct_acts
    """
    ...
