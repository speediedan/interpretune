"""Bundled circuit-tracer op family: attribution graphs, feature ranking, and feature interventions.

Backend specifics stay behind the analysis-backend seam (:mod:`interpretune.analysis.backends`);
this module is self-contained modulo the sanctioned op-authoring surfaces. See the custom ops
composition guide.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import BatchEncoding

from interpretune.analysis.backends import (
    FeatureSelectionSpec,
    _apply_optional_feature_sign_filter,
    _augment_feature_rows_for_selection,
    _select_top_feature_indices,
    apply_feature_selection_filter,
    require_analysis_backend,
)
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.optools import (
    last_token_logits,
    mean_target_logit_delta,
    resolve_feature_score_source,
)


def compute_attribution_graph_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Generate and decompose a circuit-tracer attribution graph."""
    analysis_backend = require_analysis_backend(module)
    prompt = kwargs.pop("prompt", None) or analysis_backend.resolve_prompt(module, analysis_batch, batch)
    concept_direction = analysis_batch.get("concept_direction")
    concept_label = analysis_batch.get("concept_label")
    concept_group_a_token_ids = analysis_batch.get("concept_group_a_token_ids")
    concept_group_b_token_ids = analysis_batch.get("concept_group_b_token_ids")
    concept_direction_mode = analysis_batch.get("concept_direction_mode")
    if concept_direction is not None and "attribution_targets" not in kwargs:
        kwargs["attribution_targets"] = analysis_backend.build_concept_attribution_targets(
            module,
            prompt,
            concept_direction,
            concept_label,
            concept_group_a_token_ids=concept_group_a_token_ids,
            concept_group_b_token_ids=concept_group_b_token_ids,
            concept_direction_mode=concept_direction_mode,
        )

    # Only forward kwargs consumed by graph construction. Composite pipeline
    # calls also carry downstream stage params such as top_n and
    # intervention_scale_factor that must remain available for later ops.
    attribution_graph_kwargs = {
        name: kwargs[name]
        for name in (
            "attribution_targets",
            "max_n_logits",
            "desired_logit_prob",
            "batch_size",
            "max_feature_nodes",
            "offload",
            "verbose",
            "update_interval",
        )
        if name in kwargs
    }

    graph = module.generate_attribution_graph(prompt, **attribution_graph_kwargs)
    extra_metadata: dict[str, Any] = {}
    extra_metadata["batch_idx"] = batch_idx
    if concept_label is not None:
        extra_metadata["concept_label"] = concept_label
    analysis_batch.update(**analysis_backend.decompose_graph(graph, extra_metadata=extra_metadata))

    # Resolve virtual logit_target_ids from concept-direction graphs.
    # Circuit-tracer assigns virtual IDs (>= vocab_size) to custom concept targets.
    # Replace them with the real concept group token IDs so downstream ops can index logits.
    logit_target_ids = getattr(analysis_batch, "logit_target_ids", None)
    if logit_target_ids is not None and concept_direction is not None:
        ids_tensor = torch.as_tensor(logit_target_ids, dtype=torch.long).reshape(-1)
        vocab_size = getattr(analysis_batch, "graph_vocab_size", None)
        if vocab_size is not None and (ids_tensor >= int(vocab_size)).any():
            real_ids = list(concept_group_a_token_ids or []) + list(concept_group_b_token_ids or [])
            if not real_ids:
                raise ValueError(
                    "logit_target_ids contain virtual IDs (>= vocab_size) but no concept group "
                    "token IDs are available for resolution. Provide concept_group_a_token_ids / "
                    "concept_group_b_token_ids or explicit logit_target_ids."
                )
            analysis_batch.update(logit_target_ids=torch.tensor(real_ids, dtype=torch.long))

    return analysis_batch


def extract_top_features_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    top_n: int | None = None,
    **kwargs,
) -> AnalysisBatch:
    """Extract the top scoring features from analysis-batch feature rows.

    An optional ``feature_selection`` kwarg (:class:`FeatureSelectionSpec`) pre-filters
    ``active_features`` rows before score sorting.  The filter uses **OR** semantics —
    a row is kept if it matches *any* criterion in the spec.
    """
    feature_selection: FeatureSelectionSpec | None = kwargs.get("feature_selection", None)

    active_features = torch.as_tensor(getattr(analysis_batch, "active_features", []), dtype=torch.long)
    selected_features = torch.as_tensor(getattr(analysis_batch, "selected_features", []), dtype=torch.long)
    activation_values = getattr(analysis_batch, "activation_values", None)
    activation_tensor = (
        None if activation_values is None else torch.as_tensor(activation_values, dtype=torch.float32).reshape(-1)
    )
    if active_features.numel() == 0:
        analysis_batch.update(
            top_feature_ids=torch.empty((0, 3), dtype=torch.long),
            top_feature_scores=torch.empty((0,), dtype=torch.float32),
        )
        return analysis_batch
    active_features = active_features.reshape(-1, 3)

    score_source = resolve_feature_score_source(
        kwargs.get("score_source") or (feature_selection.score_source if feature_selection else None)
    )
    score_values = getattr(analysis_batch, score_source, None) if score_source else None
    if score_source is not None and score_values is None:
        raise ValueError(f"extract_top_features score_source '{score_source}' is not present on analysis_batch")
    if score_values is None:
        score_values = getattr(analysis_batch, "node_influence_scores", None)
    if score_values is None:
        score_values = getattr(analysis_batch, "activation_values", None)
    scores = torch.as_tensor(score_values, dtype=torch.float32)
    if scores.dim() > 1:
        scores = scores.reshape(-1)

    feature_rows = active_features
    aligned_activation_values = None
    if selected_features.numel() > 0 and selected_features.shape[0] == scores.shape[0]:
        feature_rows = require_analysis_backend(module).select_feature_rows(active_features, selected_features)
        if activation_tensor is not None and activation_tensor.shape[0] == active_features.shape[0]:
            aligned_activation_values = activation_tensor.index_select(0, selected_features.reshape(-1))
        elif activation_tensor is not None and activation_tensor.shape[0] == selected_features.shape[0]:
            aligned_activation_values = activation_tensor
    elif active_features.shape[0] != scores.shape[0]:
        raise ValueError(
            "extract_top_features requires active_features to match score length directly or via selected_features"
        )
    elif activation_tensor is not None and activation_tensor.shape[0] == active_features.shape[0]:
        aligned_activation_values = activation_tensor

    if feature_selection is not None:
        feature_rows, scores, aligned_activation_values = _augment_feature_rows_for_selection(
            feature_rows,
            scores,
            aligned_activation_values,
            feature_selection,
        )

    # ---- apply optional pre-filter before score ranking ----
    if feature_selection is not None:
        sel_mask = apply_feature_selection_filter(feature_rows, feature_selection)
        if sel_mask.any():
            sel_idx = sel_mask.nonzero(as_tuple=False).reshape(-1)
            feature_rows = feature_rows.index_select(0, sel_idx)
            scores = scores.index_select(0, sel_idx)
            if aligned_activation_values is not None:
                aligned_activation_values = aligned_activation_values.index_select(0, sel_idx)

        feature_rows, scores, aligned_activation_values = _apply_optional_feature_sign_filter(
            feature_rows,
            scores,
            aligned_activation_values,
            feature_selection,
        )

    rank_by_abs = bool(kwargs.get("rank_by_abs", feature_selection.rank_by_abs if feature_selection else False))
    top_indices = _select_top_feature_indices(
        feature_rows,
        scores,
        top_n,
        feature_selection,
        rank_scores=scores.abs() if rank_by_abs else None,
    )
    update_payload: dict[str, Any] = {
        "top_feature_ids": feature_rows.index_select(0, top_indices).detach().cpu(),
        "top_feature_scores": scores.index_select(0, top_indices).detach().cpu(),
    }
    if aligned_activation_values is not None:
        update_payload["top_feature_activation_values"] = (
            aligned_activation_values.index_select(0, top_indices).detach().cpu()
        )
    analysis_batch.update(**update_payload)
    return analysis_batch


def graph_prune_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Prune a structured circuit-tracer graph and refresh decomposed outputs."""
    analysis_backend = require_analysis_backend(module)
    graph = analysis_backend.hydrate_graph_from_batch(analysis_batch)
    pruned_graph = analysis_backend.build_pruned_graph(
        graph,
        node_threshold=float(kwargs.get("node_threshold", 0.8)),
        edge_threshold=float(kwargs.get("edge_threshold", 0.98)),
    )
    analysis_batch.update(
        **analysis_backend.decompose_graph(pruned_graph, extra_metadata={"batch_idx": batch_idx, "pruned": True})
    )
    return analysis_batch


def graph_node_influence_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Compute feature-node influence scores from a structured graph."""
    analysis_backend = require_analysis_backend(module)
    graph = analysis_backend.hydrate_graph_from_batch(analysis_batch)
    node_scores, node_feature_ids = analysis_backend.compute_node_influence_scores(graph)
    update_payload: dict[str, Any] = {
        "node_influence_scores": node_scores,
        "node_feature_ids": node_feature_ids,
    }
    signed_score_fn = getattr(analysis_backend, "compute_signed_node_influence_scores", None)
    if callable(signed_score_fn):
        signed_scores = signed_score_fn(graph)
        update_payload["node_signed_influence_scores"] = signed_scores
    analysis_batch.update(
        **update_payload,
    )
    return analysis_batch


def feature_intervention_forward_impl(
    module,
    analysis_batch: AnalysisBatch,
    batch: BatchEncoding,
    batch_idx: int,
    **kwargs,
) -> AnalysisBatch:
    """Run circuit-tracer feature interventions against the module replacement model.

    This op currently implements forward-only intervention analysis and stores both the circuit-tracer tuple payload and
    a canonical feature-target InterventionDict summary for AnalysisStore consumers.
    """
    analysis_backend = require_analysis_backend(module)

    replacement_model = getattr(module, "replacement_model", None)
    if replacement_model is None:
        raise ValueError("feature_intervention_forward requires module.replacement_model")

    prompt = kwargs.pop("prompt", None)
    if prompt is None:
        prompt = analysis_backend.resolve_prompt(module, analysis_batch, batch)
    # Canonicalize to token ids before handing the prompt to the replacement model. `get_activations`
    # and `feature_intervention` below accept a raw `str`, but the backend then tokenizes it with
    # `add_special_tokens=True` -- so a prompt that already carries an explicit BOS (as our chat-
    # templated prompts do) is silently given a SECOND one. Every position index then refers to a
    # different token than the attribution graph's, and interventions appear to change activations at
    # positions the intervention cannot causally reach. `ensure_tokenized` is idempotent for tensor/
    # list inputs, so this is a no-op when the caller already passed token ids.
    ensure_tokenized = getattr(replacement_model, "ensure_tokenized", None)
    if callable(ensure_tokenized):
        prompt = ensure_tokenized(prompt)
    settings = analysis_backend.resolve_feature_intervention_settings(module, kwargs)
    if (
        getattr(analysis_batch, "top_feature_ids", None) is None
        and getattr(analysis_batch, "active_features", None) is not None
    ):
        selection_kwargs = {
            name: kwargs[name] for name in ("feature_selection", "score_source", "rank_by_abs") if name in kwargs
        }
        analysis_batch = extract_top_features_impl(
            module,
            analysis_batch,
            batch,
            batch_idx,
            top_n=kwargs.get("top_n"),
            **selection_kwargs,
        )
    feature_rows = analysis_batch.require(
        "top_feature_ids",
        message="feature_intervention_forward requires top_feature_ids in analysis_batch or scoped inputs",
    )
    feature_scores = analysis_batch.get("top_feature_scores")
    feature_activation_values = analysis_batch.get("top_feature_activation_values")
    target_ids = analysis_batch.get("logit_target_ids")

    # If no explicit logit_target_ids, try to resolve from concept group token IDs
    if target_ids is None:
        concept_a_ids = analysis_batch.get("concept_group_a_token_ids")
        concept_b_ids = analysis_batch.get("concept_group_b_token_ids")
        real_ids = list(concept_a_ids or []) + list(concept_b_ids or [])
        if real_ids:
            target_ids = torch.tensor(real_ids, dtype=torch.long)

    intervention_inputs = {
        "top_feature_ids": feature_rows,
        "top_feature_scores": feature_scores,
        "top_feature_activation_values": feature_activation_values,
        "logit_target_ids": target_ids,
    }
    analysis_batch.update(
        **{
            key: value
            for key, value in intervention_inputs.items()
            if value is not None and getattr(analysis_batch, key, None) is None
        }
    )
    interventions, intervention_payload = analysis_backend.build_feature_interventions(intervention_inputs, settings)

    pre_logits_raw, _ = replacement_model.get_activations(prompt)
    pre_logits = last_token_logits(pre_logits_raw)
    intervention_activation_cache = None

    if interventions:
        post_logits_raw, intervention_activation_cache = replacement_model.feature_intervention(
            prompt,
            interventions,
            **analysis_backend.feature_intervention_call_kwargs(settings),
        )
        post_logits = last_token_logits(post_logits_raw)
    else:
        post_logits = pre_logits.clone()

    target_ids_tensor = None
    if target_ids is not None:
        target_ids_tensor = torch.as_tensor(target_ids, dtype=torch.long).reshape(-1)
    logit_diff = mean_target_logit_delta(pre_logits, post_logits, target_ids_tensor)

    analysis_batch.update(
        **intervention_payload,
        pre_intervention_logits=pre_logits,
        post_intervention_logits=post_logits,
        logit_diff=logit_diff.detach().cpu(),
    )
    if intervention_activation_cache is not None:
        analysis_batch.update(intervention_activation_cache=intervention_activation_cache)
    return analysis_batch
