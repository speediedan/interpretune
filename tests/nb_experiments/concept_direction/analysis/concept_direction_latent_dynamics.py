from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, cast

import torch

import interpretune as it
from interpretune.analysis.backends import require_analysis_backend
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.helpers import _flatten_concept_store_rows

from tests.nb_experiments.concept_direction.concept_direction import (
    NotebookHarnessConfig,
    build_notebook_harness_config,
    construct_concept_pair_analysis_inputs,
    execute_concept_latent_extraction_ops,
    experiment_session,
    get_key_token_ids_and_labels,
    maybe_zero_softcap,
    phase_run_name,
    render_prompt,
    resolve_target_tokens,
)
from tests.nb_experiments.nb_harness_utils import (
    _extract_top_features_with_optional_filter,
    _serialize_constrained_feature_selection,
    _serialize_intervention_call_kwargs,
    _summarize_graph_input_tokens,
    resolve_feature_dashboard_metadata,
    tensor_to_cpu,
)
from tests.nb_experiments.concept_direction.analysis.concept_direction_analysis import (
    build_concept_direction_stage_artifact,
    capture_context_enhanced_extraction_snapshot,
    compute_concept_direction_geometry,
)
from tests.nb_experiments.concept_direction.analysis.latent_state_projection import (
    ProjectionMethod,
    UmapBackendPreference,
    project_embeddings,
)


_BLOCK_LAYER_PATTERN = re.compile(r"blocks\.(\d+)\.")
_STAGE_ORDER = {
    "target_token_embed_difference_direction": 10,
    "embed_mean_difference_direction": 20,
    "answer_state_mean_difference_direction": 30,
    "context_state_mean_difference_direction": 40,
    "projected_context_state_mean_difference_direction": 50,
    "paired_residual_mean_direction_unweighted": 60,
    "store_context_enhanced_paired_rejection_unnormalized_direction": 70,
    "store_context_enhanced_paired_rejection_direction": 80,
    "store_context_enhanced_paired_rejection_reconstruction_direction": 90,
    "embed_paired_rejection_direction": 100,
    "store_answer_position_paired_rejection_direction": 110,
}


def _last_token_id(tokenizer: Any, token: str) -> int:
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if not encoded:
        raise ValueError(f"Unable to tokenize {token!r}")
    return int(encoded[-1])


def _resolve_group_token_ids(cfg: NotebookHarnessConfig, tokenizer: Any) -> tuple[list[int], list[int]]:
    group_a_ids = [_last_token_id(tokenizer, token) for token in cfg.concept_pair.group_a_tokens]
    group_b_ids = [_last_token_id(tokenizer, token) for token in cfg.concept_pair.group_b_tokens]
    return group_a_ids, group_b_ids


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _tensor_summary(value: torch.Tensor, *, include_values_below: int | None = None) -> dict[str, Any]:
    tensor = torch.as_tensor(value).detach().cpu().float().reshape(-1)
    summary = {
        "shape": list(tensor.shape),
        "norm": float(torch.linalg.vector_norm(tensor).item()),
        "mean_abs": float(tensor.abs().mean().item()) if tensor.numel() else 0.0,
        "max_abs": float(tensor.abs().max().item()) if tensor.numel() else 0.0,
        "sha256": _tensor_sha256(tensor),
    }
    if include_values_below is not None and tensor.numel() <= include_values_below:
        summary["values"] = tensor.tolist()
    return summary


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    flat_vector = torch.as_tensor(vector, dtype=torch.float32).detach().cpu().reshape(-1)
    norm = torch.linalg.vector_norm(flat_vector)
    if not torch.isfinite(norm) or norm.item() <= 0:
        return flat_vector
    return flat_vector / norm


def _stage_geometry_from_vector(
    direction: torch.Tensor,
    *,
    direction_mode: str,
    reference_geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    geometry = {
        key: value
        for key, value in (reference_geometry or {}).items()
        if key not in {"raw_direction", "normalized_direction", "raw_direction_summary", "normalized_direction_summary"}
    }
    raw_direction = torch.as_tensor(direction, dtype=torch.float32).detach().cpu().reshape(-1)
    normalized_direction = _normalize_vector(raw_direction)
    geometry.update(
        {
            "direction_mode": direction_mode,
            "raw_direction_summary": _tensor_summary(raw_direction, include_values_below=64),
            "normalized_direction_summary": _tensor_summary(normalized_direction, include_values_below=64),
        }
    )
    return geometry


def _target_only_embed_direction(
    embed_weight: torch.Tensor,
    target_token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    target_states = embed_weight[torch.tensor(target_token_ids, dtype=torch.long)]
    if target_states.shape[0] == 1:
        raw_direction = target_states[0].detach().cpu().float()
    elif target_states.shape[0] >= 2:
        raw_direction = (target_states[0] - target_states[1]).detach().cpu().float()
    else:
        raise ValueError("target_only_embed_direction requires at least one target token")
    return raw_direction, _normalize_vector(raw_direction)


def _weighted_group_mean_direction(
    states: torch.Tensor,
    group_ids: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_a_mask = group_ids == 0
    group_b_mask = group_ids == 1
    if not bool(group_a_mask.any().item()) or not bool(group_b_mask.any().item()):
        raise ValueError("Weighted mean difference requires both group-a and group-b rows")

    group_a_mean = torch.sum(states[group_a_mask] * weights[group_a_mask].unsqueeze(-1), dim=0) / weights[
        group_a_mask
    ].sum().clamp_min(1e-12)
    group_b_mean = torch.sum(states[group_b_mask] * weights[group_b_mask].unsqueeze(-1), dim=0) / weights[
        group_b_mask
    ].sum().clamp_min(1e-12)
    raw_direction = group_a_mean - group_b_mean
    raw_norm = torch.linalg.vector_norm(raw_direction)
    normalized_direction = raw_direction / raw_norm.clamp_min(1e-12)
    return raw_direction.detach().cpu(), normalized_direction.detach().cpu()


def _paired_rejection_payload(
    states: torch.Tensor,
    group_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    group_names: list[str],
) -> dict[str, Any]:
    group_a_mask = group_ids == 0
    group_b_mask = group_ids == 1
    group_a_states = states[group_a_mask]
    group_b_states = states[group_b_mask]
    group_a_weights = weights[group_a_mask]
    group_b_weights = weights[group_b_mask]

    if group_a_states.shape[0] != group_b_states.shape[0]:
        raise ValueError("paired_rejection stage inspection requires equal group-a and group-b counts")

    residual_rows: list[torch.Tensor] = []
    pair_weight_rows: list[torch.Tensor] = []
    residual_records: list[dict[str, Any]] = []
    group_a_names = [name for name in group_names if name == group_names[0]]
    group_b_names = [name for name in group_names if name != group_names[0]]

    for pair_index, (state_a, state_b, weight_a, weight_b) in enumerate(
        zip(group_a_states, group_b_states, group_a_weights, group_b_weights, strict=True)
    ):
        denom = torch.dot(state_b, state_b).clamp_min(1e-12)
        projection = (torch.dot(state_a, state_b) / denom) * state_b
        residual = state_a - projection
        pair_weight = (weight_a + weight_b) / 2
        residual_rows.append(residual)
        pair_weight_rows.append(pair_weight)
        residual_records.append(
            {
                "pair_index": int(pair_index),
                "group_a_name": group_a_names[pair_index] if pair_index < len(group_a_names) else "group_a",
                "group_b_name": group_b_names[pair_index] if pair_index < len(group_b_names) else "group_b",
                "group_a_norm": float(torch.linalg.vector_norm(state_a).item()),
                "group_b_norm": float(torch.linalg.vector_norm(state_b).item()),
                "pair_weight": float(pair_weight.item()),
                "projection_norm": float(torch.linalg.vector_norm(projection).item()),
                "residual_norm": float(torch.linalg.vector_norm(residual).item()),
                "group_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        state_a.reshape(1, -1),
                        state_b.reshape(1, -1),
                    ).item()
                ),
            }
        )

    residual_tensor = torch.stack(residual_rows).detach().cpu()
    pair_weight_tensor = torch.stack(pair_weight_rows).detach().cpu()
    unweighted_mean_direction = residual_tensor.mean(dim=0)
    raw_direction = torch.sum(
        residual_tensor * pair_weight_tensor.unsqueeze(-1),
        dim=0,
    ) / pair_weight_tensor.sum().clamp_min(1e-12)
    normalized_direction = raw_direction / torch.linalg.vector_norm(raw_direction).clamp_min(1e-12)
    return {
        "residuals": residual_tensor,
        "pair_weights": pair_weight_tensor,
        "unweighted_mean_direction": unweighted_mean_direction.detach().cpu(),
        "raw_direction": raw_direction.detach().cpu(),
        "normalized_direction": normalized_direction.detach().cpu(),
        "records": residual_records,
    }


def _serialize_projection_records(
    matrix: torch.Tensor,
    row_metadata: list[dict[str, Any]],
    *,
    method: ProjectionMethod,
    projection_kwargs: Mapping[str, Any],
    umap_backend_preference: UmapBackendPreference,
) -> dict[str, Any]:
    result = project_embeddings(
        matrix,
        method=method,
        umap_backend_preference=umap_backend_preference,
        **projection_kwargs,
    )
    records: list[dict[str, Any]] = []
    for coordinates, metadata in zip(result.coordinates, row_metadata, strict=True):
        record = dict(metadata)
        record["x"] = float(coordinates[0])
        record["y"] = float(coordinates[1])
        records.append(record)
    return {
        "method": result.method,
        "backend": result.backend,
        "metadata": result.metadata,
        "records": records,
    }


def _stage_order(stage_name: str) -> int:
    return _STAGE_ORDER.get(stage_name, 999)


def _feature_metadata_key(layer: int, feature_id: int) -> tuple[int, int]:
    return (int(layer), int(feature_id))


def _serialize_key_tokens(cfg: NotebookHarnessConfig, tokenizer: Any) -> list[dict[str, Any]]:
    key_token_ids, key_token_labels = get_key_token_ids_and_labels(cfg, tokenizer)
    return [
        {
            "token_id": int(token_id),
            "token_label": str(token_label),
        }
        for token_id, token_label in zip(key_token_ids, key_token_labels, strict=True)
    ]


def _handle_hook_name(handle: Any) -> str:
    try:
        hook_name = handle.cfg.metadata.hook_name
    except AttributeError:
        layer_idx = getattr(handle, "layer_idx", None)
        return "" if layer_idx is None else f"blocks.{int(layer_idx)}.transcoder"
    return "" if hook_name is None else str(hook_name)


def _resolve_handle_layer(handle: Any) -> int | None:
    layer_idx = getattr(handle, "layer_idx", None)
    if layer_idx is not None:
        return int(layer_idx)
    hook_name = _handle_hook_name(handle)
    match = _BLOCK_LAYER_PATTERN.search(hook_name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_feature_handles(module: Any) -> dict[int, Any]:
    handles_by_layer: dict[int, Any] = {}
    for handle in getattr(module, "sae_handles", []) or []:
        layer = _resolve_handle_layer(handle)
        if layer is None or layer in handles_by_layer:
            continue
        handles_by_layer[layer] = handle

    transcoders = getattr(getattr(module, "model", None), "transcoders", None)
    transcoder_container = getattr(transcoders, "_module", transcoders)
    for handle in getattr(transcoder_container, "transcoders", []) or []:
        layer = _resolve_handle_layer(handle)
        if layer is None or layer in handles_by_layer:
            continue
        handles_by_layer[layer] = handle
    return handles_by_layer


def _extract_feature_vector(handle: Any, *, feature_id: int, vector_type: str) -> torch.Tensor:
    if vector_type == "encoder":
        weight = torch.as_tensor(handle.W_enc).detach().cpu().float()
        if weight.dim() != 2:
            raise ValueError(f"Encoder weight must be rank-2, got shape {tuple(weight.shape)}")
        if getattr(handle, "layer_idx", None) is not None:
            if feature_id >= weight.shape[0]:
                raise ValueError(f"Encoder weight does not expose feature row {feature_id}")
            return weight[feature_id].reshape(-1)
        if feature_id < weight.shape[1]:
            return weight[:, feature_id].reshape(-1)
        if feature_id < weight.shape[0]:
            return weight[feature_id].reshape(-1)
        raise ValueError(f"Encoder weight does not expose feature {feature_id}")
    if vector_type == "decoder":
        weight = torch.as_tensor(handle.W_dec).detach().cpu().float()
        if weight.dim() != 2 or feature_id >= weight.shape[0]:
            raise ValueError(f"Decoder weight does not expose feature row {feature_id}")
        return weight[feature_id].reshape(-1)
    raise ValueError(f"Unsupported vector_type: {vector_type}")


def _build_feature_vector_payload(
    module: Any,
    stage_graphs: Mapping[str, Mapping[str, Any]],
    *,
    feature_metadata: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    handles_by_layer = _resolve_feature_handles(module)
    records: list[dict[str, Any]] = []
    vector_rows: list[torch.Tensor] = []

    for stage_name, payload in stage_graphs.items():
        feature_details = cast(list[dict[str, Any]], payload.get("top_feature_details", []))
        for rank, feature_detail in enumerate(feature_details, start=1):
            layer = int(feature_detail["layer"])
            position = int(feature_detail["position"])
            feature_id = int(feature_detail["feature_id"])
            feature_score = float(feature_detail["score"])
            handle = handles_by_layer.get(layer)
            if handle is None:
                continue
            metadata = feature_metadata.get(_feature_metadata_key(layer, feature_id), {})
            for vector_type in ("encoder", "decoder"):
                vector = _extract_feature_vector(handle, feature_id=feature_id, vector_type=vector_type)
                vector_rows.append(vector)
                records.append(
                    {
                        "stage_name": stage_name,
                        "stage_rank": int(payload.get("stage_rank", _stage_order(stage_name))),
                        "rank": int(rank),
                        "layer": int(layer),
                        "position": int(position),
                        "feature_id": int(feature_id),
                        "score": float(feature_score),
                        "feature_url": metadata.get("feature_url"),
                        "feature_explanation": metadata.get("feature_explanation"),
                        "vector_type": vector_type,
                        "hook_name": _handle_hook_name(handle),
                        "hover_label": (f"{stage_name}::{vector_type}::L{layer}:P{position}:F{feature_id}:R{rank}"),
                    }
                )

    if not vector_rows:
        return records, torch.empty((0, 0), dtype=torch.float32)
    return records, torch.stack(vector_rows)


def _build_stage_graph_bundle(
    module: Any,
    cfg: NotebookHarnessConfig,
    *,
    rendered_prompt: str,
    direction: torch.Tensor,
    stage_name: str,
    top_n: int,
) -> dict[str, Any]:
    tokenizer = module.replacement_model.tokenizer
    group_a_ids, group_b_ids = _resolve_group_token_ids(cfg, tokenizer)
    key_tokens = _serialize_key_tokens(cfg, tokenizer)
    target_ids, _ = resolve_target_tokens(cfg, tokenizer)
    analysis_backend = require_analysis_backend(module)
    attribution_targets = analysis_backend.build_concept_attribution_targets(
        module,
        rendered_prompt,
        direction,
        cfg.analysis_concept_label,
        concept_group_a_token_ids=group_a_ids,
        concept_group_b_token_ids=group_b_ids,
        concept_direction_mode=cfg.analysis_direction_mode_name,
    )
    graph_call_kwargs = {
        "attribution_targets": attribution_targets,
        "max_n_logits": 1,
        "desired_logit_prob": 1.0,
        "batch_size": cfg.batch_size,
        "max_feature_nodes": cfg.max_feature_nodes,
        "offload": "cpu",
        "verbose": False,
    }
    serialized_graph_call_kwargs = _serialize_intervention_call_kwargs(
        {
            **graph_call_kwargs,
            "attribution_targets": [
                {
                    "kind": type(target).__name__,
                    "token_str": getattr(target, "token_str", None),
                    "prob": None if getattr(target, "prob", None) is None else float(target.prob),
                }
                for target in cast(list[Any], list(attribution_targets or []))
            ],
        }
    )
    analysis_batch = AnalysisBatch(
        prompts=[rendered_prompt],
        concept_direction=direction,
        concept_label=cfg.analysis_concept_label,
        concept_direction_mode=cfg.analysis_direction_mode_name,
        concept_group_a_token_ids=group_a_ids,
        concept_group_b_token_ids=group_b_ids,
        logit_target_ids=torch.tensor(target_ids, dtype=torch.long),
    )
    graph_result = cast(
        Any,
        it.compute_attribution_graph(
            module,
            analysis_batch,
            batch=cast(Any, None),
            batch_idx=0,
            **graph_call_kwargs,
        ),
    )
    influence_result = cast(Any, it.graph_node_influence(module, graph_result, batch=cast(Any, None), batch_idx=0))
    top_payload = dict(cast(Any, graph_result))
    top_payload.update(dict(cast(Any, influence_result)))
    top_features_result, applied_rows = _extract_top_features_with_optional_filter(
        module,
        cfg,
        top_payload,
        top_n=top_n,
    )
    top_feature_ids = torch.as_tensor(top_features_result.top_feature_ids, dtype=torch.long).reshape(-1, 3)
    top_feature_scores = torch.as_tensor(top_features_result.top_feature_scores, dtype=torch.float32).reshape(-1)
    return {
        "stage_name": stage_name,
        "top_feature_rows": [[int(value) for value in row.tolist()] for row in top_feature_ids],
        "top_feature_scores": [float(value) for value in top_feature_scores.tolist()],
        "key_tokens": key_tokens,
        "requested_feature_selection": _serialize_constrained_feature_selection(cfg.constrained_feature_selection_refs),
        "applied_feature_selection_rows": [list(row) for row in applied_rows],
        "graph_call_kwargs": serialized_graph_call_kwargs,
        "graph_input_tokens": _summarize_graph_input_tokens(
            tokenizer,
            rendered_prompt,
            cfg.prompt_render_mode,
            graph_result.input_tokens,
        ),
    }


def _build_stage_vector_payload(
    *,
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    stage_name: str,
    direction: torch.Tensor,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    group_a_ids, group_b_ids = _resolve_group_token_ids(cfg, tokenizer)
    return build_concept_direction_stage_artifact(
        path_label=stage_name,
        direction_mode=str(geometry.get("direction_mode", cfg.analysis_direction_mode_name)),
        direction=direction,
        tokenizer=tokenizer,
        group_a_token_ids=group_a_ids,
        group_b_token_ids=group_b_ids,
        geometry=geometry,
    )


def run_latent_dynamics_analysis(
    config_path: str | Path,
    *,
    projection_method: ProjectionMethod = "umap",
    stage_top_n: int | None = None,
) -> dict[str, Any]:
    cfg, should_cleanup_work_root, resolved_payload = build_notebook_harness_config(config_path)
    latent_dynamics_cfg = cast(Mapping[str, Any], resolved_payload.get("LATENT_DYNAMICS", {}))
    projection_cfg = cast(Mapping[str, Any], latent_dynamics_cfg.get("projection", {}))
    umap_backend_preference = cast(
        UmapBackendPreference,
        str(projection_cfg.get("backend_preference", "cpu")).strip().lower(),
    )
    projection_kwargs = {
        "n_components": int(projection_cfg.get("n_components", 2)),
        "n_neighbors": int(projection_cfg.get("n_neighbors", 10)),
        "min_dist": float(projection_cfg.get("min_dist", 0.1)),
        "metric": str(projection_cfg.get("metric", "cosine")),
        "random_state": int(projection_cfg.get("random_state", 17)),
    }
    effective_stage_top_n = int(latent_dynamics_cfg.get("stage_top_n", stage_top_n or cfg.top_n))

    try:
        with experiment_session(
            cfg.work_root,
            phase_run_name(cfg, "latent_dynamics"),
            **cfg.session_kwargs,
        ) as (_, module, tokenizer):
            model_backend = getattr(module, "_model_backend", None)
            if model_backend is None:
                raise ValueError("experiment session module must expose _model_backend")
            device = next(module.model.parameters()).device
            (target_a_id, target_b_id), target_tokens = resolve_target_tokens(cfg, tokenizer)
            rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)

            with maybe_zero_softcap(module, cfg):
                (
                    cached_batches,
                    answer_indices,
                    context_token_indices,
                    orig_labels,
                    logit_diffs,
                    prediction_info,
                    prompt_examples,
                ) = construct_concept_pair_analysis_inputs(
                    cfg,
                    module,
                    tokenizer,
                    model_backend,
                    device,
                    target_a_id,
                    target_b_id,
                )

                store_plain_batches = execute_concept_latent_extraction_ops(
                    module,
                    cfg,
                    cached_batches,
                    answer_indices,
                    context_token_indices,
                    orig_labels,
                    logit_diffs,
                    len(prompt_examples),
                    extraction_mode="answer_position_state",
                )
                store_context_batches = execute_concept_latent_extraction_ops(
                    module,
                    cfg,
                    cached_batches,
                    answer_indices,
                    context_token_indices,
                    orig_labels,
                    logit_diffs,
                    len(prompt_examples),
                    extraction_mode=cfg.store_latent_extraction_mode,
                )

                plain_aggregated_batch = cast(Any, store_plain_batches[-1])
                plain_latent_rows = cast(Any, plain_aggregated_batch.concept_latent_state_rows)
                plain_group_id_rows = cast(Any, plain_aggregated_batch.concept_group_id_rows)
                plain_group_name_rows = cast(Any, plain_aggregated_batch.concept_group_name_rows)
                plain_weight_rows = cast(Any, plain_aggregated_batch.concept_example_weight_rows)
                plain_states, plain_group_ids, plain_weights, _ = _flatten_concept_store_rows(
                    plain_latent_rows,
                    plain_group_id_rows,
                    plain_group_name_rows,
                    plain_weight_rows,
                )
                context_aggregated_batch = cast(Any, store_context_batches[-1])
                context_latent_rows = cast(Any, context_aggregated_batch.concept_latent_state_rows)
                context_group_id_rows = cast(Any, context_aggregated_batch.concept_group_id_rows)
                context_group_name_rows = cast(Any, context_aggregated_batch.concept_group_name_rows)
                context_weight_rows = cast(Any, context_aggregated_batch.concept_example_weight_rows)
                context_states_final, context_group_ids, context_weights, context_group_names = (
                    _flatten_concept_store_rows(
                        context_latent_rows,
                        context_group_id_rows,
                        context_group_name_rows,
                        context_weight_rows,
                    )
                )

                plain_geometry = compute_concept_direction_geometry(
                    plain_states,
                    plain_group_ids,
                    direction_mode=cfg.analysis_direction_mode_name,
                    example_weights=plain_weights,
                )
                context_geometry = compute_concept_direction_geometry(
                    context_states_final,
                    context_group_ids,
                    direction_mode=cfg.analysis_direction_mode_name,
                    example_weights=context_weights,
                )

                store_plain_direction = cast(
                    Any,
                    it.concept_direction(
                        module,
                        plain_aggregated_batch,
                        batch=cast(Any, None),
                        batch_idx=0,
                    ),
                )
                store_context_direction = cast(
                    Any,
                    it.concept_direction(
                        module,
                        context_aggregated_batch,
                        batch=cast(Any, None),
                        batch_idx=0,
                    ),
                )
                embed_direction = cast(
                    Any,
                    it.concept_direction(
                        module,
                        AnalysisBatch(
                            concept_group_a=list(cfg.concept_pair.group_a_tokens),
                            concept_group_b=list(cfg.concept_pair.group_b_tokens),
                            concept_label=cfg.concept_pair.concept_label,
                            concept_direction_mode=cfg.analysis_direction_mode_name,
                        ),
                        batch=cast(Any, None),
                        batch_idx=0,
                    ),
                )

                context_snapshots = [
                    capture_context_enhanced_extraction_snapshot(
                        AnalysisBatch(
                            cache=cached_batch,
                            answer_indices=answer_index,
                            context_token_indices=context_index,
                            concept_cache_key=cfg.store_concept_cache_key,
                            orig_labels=group_label,
                            logit_diffs=logit_diff,
                            use_answer_state_as_basis=cfg.use_answer_state_as_basis,
                        ),
                        context_scale=cfg.context_enhanced_scale,
                        use_answer_state_as_basis=cfg.use_answer_state_as_basis,
                    )
                    for cached_batch, answer_index, context_index, group_label, logit_diff in zip(
                        cached_batches,
                        answer_indices,
                        context_token_indices,
                        orig_labels,
                        logit_diffs,
                        strict=True,
                    )
                ]

                answer_state_matrix = torch.cat([snapshot.answer_states for snapshot in context_snapshots], dim=0)
                context_state_matrix = torch.cat([snapshot.context_states for snapshot in context_snapshots], dim=0)
                projected_state_matrix = torch.cat([snapshot.projected_states for snapshot in context_snapshots], dim=0)
                final_state_matrix = torch.cat([snapshot.final_latent_states for snapshot in context_snapshots], dim=0)
                _, normalized_answer_direction = _weighted_group_mean_direction(
                    answer_state_matrix,
                    context_group_ids,
                    context_weights,
                )
                _, normalized_context_direction = _weighted_group_mean_direction(
                    context_state_matrix,
                    context_group_ids,
                    context_weights,
                )
                _, normalized_projected_direction = _weighted_group_mean_direction(
                    projected_state_matrix,
                    context_group_ids,
                    context_weights,
                )
                paired_residual_payload = _paired_rejection_payload(
                    final_state_matrix,
                    context_group_ids,
                    context_weights,
                    group_names=context_group_names,
                )

                stage_vectors: dict[str, dict[str, Any]] = {}
                stage_graphs: dict[str, dict[str, Any]] = {}

                def _register_stage(
                    stage_name: str,
                    direction_tensor: torch.Tensor,
                    geometry: Mapping[str, Any],
                    *,
                    stage_kind: str = "direction",
                ) -> None:
                    stage_vector = tensor_to_cpu(direction_tensor.reshape(-1))
                    stage_rank = _stage_order(stage_name)
                    stage_payload = _build_stage_vector_payload(
                        cfg=cfg,
                        tokenizer=tokenizer,
                        stage_name=stage_name,
                        direction=stage_vector,
                        geometry=geometry,
                    )
                    stage_payload["stage_rank"] = int(stage_rank)
                    stage_payload["stage_kind"] = stage_kind
                    stage_payload["direction_vector"] = stage_vector.tolist()
                    stage_vectors[stage_name] = stage_payload
                    stage_graph_bundle = _build_stage_graph_bundle(
                        module,
                        cfg,
                        rendered_prompt=rendered_prompt,
                        direction=stage_vector,
                        stage_name=stage_name,
                        top_n=effective_stage_top_n,
                    )
                    stage_graph_bundle["stage_rank"] = int(stage_rank)
                    stage_graph_bundle["stage_kind"] = stage_kind
                    stage_graphs[stage_name] = stage_graph_bundle

                group_a_ids, group_b_ids = _resolve_group_token_ids(cfg, tokenizer)
                embed_weight = module.model.get_input_embeddings().weight.detach().cpu().float()
                target_embed_states = embed_weight[torch.tensor([target_a_id, target_b_id], dtype=torch.long)]
                target_embed_group_ids = torch.tensor([0, 1], dtype=torch.long)
                target_embed_geometry_base = compute_concept_direction_geometry(
                    target_embed_states,
                    target_embed_group_ids,
                    direction_mode="mean_difference",
                )
                target_embed_raw_direction, target_only_embed_direction = _target_only_embed_direction(
                    embed_weight,
                    [int(target_a_id), int(target_b_id)],
                )
                target_embed_geometry = _stage_geometry_from_vector(
                    target_embed_raw_direction,
                    direction_mode="mean_difference",
                    reference_geometry=target_embed_geometry_base,
                )
                embed_group_a = embed_weight[torch.tensor(group_a_ids, dtype=torch.long)]
                embed_group_b = embed_weight[torch.tensor(group_b_ids, dtype=torch.long)]
                embed_states = torch.cat([embed_group_a, embed_group_b], dim=0)
                embed_group_ids = torch.tensor(
                    [0] * int(embed_group_a.shape[0]) + [1] * int(embed_group_b.shape[0]),
                    dtype=torch.long,
                )
                embed_group_weights = torch.ones((embed_states.shape[0],), dtype=torch.float32)
                group_embed_raw_direction, group_mean_embed_direction = _weighted_group_mean_direction(
                    embed_states,
                    embed_group_ids,
                    embed_group_weights,
                )
                embed_geometry = compute_concept_direction_geometry(
                    embed_states,
                    embed_group_ids,
                    direction_mode=cfg.analysis_direction_mode_name,
                )
                group_embed_geometry = _stage_geometry_from_vector(
                    group_embed_raw_direction,
                    direction_mode="mean_difference",
                    reference_geometry=compute_concept_direction_geometry(
                        embed_states,
                        embed_group_ids,
                        direction_mode="mean_difference",
                    ),
                )
                raw_answer_direction, normalized_answer_direction = _weighted_group_mean_direction(
                    answer_state_matrix,
                    context_group_ids,
                    context_weights,
                )
                answer_geometry = compute_concept_direction_geometry(
                    answer_state_matrix,
                    context_group_ids,
                    direction_mode="mean_difference",
                    example_weights=context_weights,
                )
                answer_geometry = _stage_geometry_from_vector(
                    raw_answer_direction,
                    direction_mode="mean_difference",
                    reference_geometry=answer_geometry,
                )
                raw_context_direction, normalized_context_direction = _weighted_group_mean_direction(
                    context_state_matrix,
                    context_group_ids,
                    context_weights,
                )
                context_state_geometry = compute_concept_direction_geometry(
                    context_state_matrix,
                    context_group_ids,
                    direction_mode="mean_difference",
                    example_weights=context_weights,
                )
                context_state_geometry = _stage_geometry_from_vector(
                    raw_context_direction,
                    direction_mode="mean_difference",
                    reference_geometry=context_state_geometry,
                )
                raw_projected_direction, normalized_projected_direction = _weighted_group_mean_direction(
                    projected_state_matrix,
                    context_group_ids,
                    context_weights,
                )
                projected_geometry = compute_concept_direction_geometry(
                    projected_state_matrix,
                    context_group_ids,
                    direction_mode="mean_difference",
                    example_weights=context_weights,
                )
                projected_geometry = _stage_geometry_from_vector(
                    raw_projected_direction,
                    direction_mode="mean_difference",
                    reference_geometry=projected_geometry,
                )
                paired_geometry = {**context_geometry, "direction_mode": cfg.analysis_direction_mode_name}
                mean_raw_paired_residual_direction = cast(
                    torch.Tensor,
                    paired_residual_payload["unweighted_mean_direction"],
                )
                mean_raw_paired_residual_geometry = _stage_geometry_from_vector(
                    mean_raw_paired_residual_direction,
                    direction_mode="paired_rejection_unweighted",
                    reference_geometry=paired_geometry,
                )
                unnormalized_direction_vector = cast(torch.Tensor, context_geometry["raw_direction"])
                unnormalized_direction_geometry = _stage_geometry_from_vector(
                    unnormalized_direction_vector,
                    direction_mode=cfg.analysis_direction_mode_name,
                    reference_geometry=paired_geometry,
                )
                normalized_direction_vector = tensor_to_cpu(store_context_direction.concept_direction)
                normalized_direction_geometry = _stage_geometry_from_vector(
                    normalized_direction_vector,
                    direction_mode=cfg.analysis_direction_mode_name,
                    reference_geometry=paired_geometry,
                )

                _register_stage(
                    "target_token_embed_difference_direction",
                    target_only_embed_direction,
                    target_embed_geometry,
                )
                _register_stage(
                    "embed_mean_difference_direction",
                    group_mean_embed_direction,
                    group_embed_geometry,
                )
                _register_stage(
                    "answer_state_mean_difference_direction",
                    raw_answer_direction,
                    answer_geometry,
                )
                _register_stage(
                    "context_state_mean_difference_direction",
                    raw_context_direction,
                    context_state_geometry,
                )
                _register_stage(
                    "projected_context_state_mean_difference_direction",
                    raw_projected_direction,
                    projected_geometry,
                )
                _register_stage(
                    "paired_residual_mean_direction_unweighted",
                    mean_raw_paired_residual_direction,
                    mean_raw_paired_residual_geometry,
                )
                _register_stage(
                    "store_context_enhanced_paired_rejection_unnormalized_direction",
                    unnormalized_direction_vector,
                    unnormalized_direction_geometry,
                )
                _register_stage(
                    "store_context_enhanced_paired_rejection_direction",
                    normalized_direction_vector,
                    normalized_direction_geometry,
                )
                _register_stage(
                    "store_context_enhanced_paired_rejection_reconstruction_direction",
                    cast(torch.Tensor, paired_residual_payload["normalized_direction"]),
                    paired_geometry,
                )
                _register_stage(
                    "embed_paired_rejection_direction",
                    tensor_to_cpu(embed_direction.concept_direction),
                    embed_geometry,
                )
                _register_stage(
                    "store_answer_position_paired_rejection_direction",
                    tensor_to_cpu(store_plain_direction.concept_direction),
                    plain_geometry,
                )

                feature_metadata = resolve_feature_dashboard_metadata(
                    model_id=cfg.neuronpedia_model,
                    source_set=cfg.neuronpedia_set,
                    feature_rows=(payload.get("top_feature_rows", []) for payload in stage_graphs.values()),
                    base_url=cfg.neuronpedia_base_url,
                    preferred_type_name=cfg.local_explanation_type_name,
                    timeout_seconds=min(cfg.local_explanation_timeout_seconds, 15),
                )
                for payload in stage_graphs.values():
                    top_feature_rows = cast(list[list[int]], payload.get("top_feature_rows", []))
                    top_feature_scores = cast(list[float], payload.get("top_feature_scores", []))
                    top_feature_details: list[dict[str, Any]] = []
                    for feature_row, feature_score in zip(top_feature_rows, top_feature_scores, strict=True):
                        layer, position, feature_id = (int(value) for value in feature_row)
                        metadata = dict(feature_metadata.get(_feature_metadata_key(layer, feature_id), {}))
                        top_feature_details.append(
                            {
                                "layer": layer,
                                "position": position,
                                "feature_id": feature_id,
                                "score": float(feature_score),
                                **metadata,
                            }
                        )
                    payload["top_feature_details"] = top_feature_details

                feature_vector_rows, feature_vector_matrix = _build_feature_vector_payload(
                    module,
                    stage_graphs,
                    feature_metadata=feature_metadata,
                )

            example_rows: list[dict[str, Any]] = []
            trajectory_matrix_rows: list[torch.Tensor] = []
            trajectory_metadata: list[dict[str, Any]] = []
            target_token_a, target_token_b = (str(target_tokens[0]), str(target_tokens[1]))
            for index, (prompt_example, weight, logit_diff, snapshot) in enumerate(
                zip(prompt_examples, context_weights, logit_diffs, context_snapshots, strict=True)
            ):
                prompt_alignment = cast(dict[str, Any], prompt_example["prompt_alignment_artifact"])
                expected_answer = str(prompt_example["expected_answer"])
                logit_diff_value = float(torch.as_tensor(logit_diff, dtype=torch.float32).reshape(-1)[0].item())
                expected_answer_margin = logit_diff_value
                if expected_answer.casefold() == target_token_b.casefold():
                    expected_answer_margin = -logit_diff_value
                example_rows.append(
                    {
                        "example_index": int(index),
                        "group_name": str(prompt_example["group_name"]),
                        "probe_surface_text": str(prompt_example["probe_surface_text"]),
                        "expected_answer": expected_answer,
                        "example_weight": float(weight.item()),
                        "logit_diff": logit_diff_value,
                        "logit_diff_label": f"{target_token_a} - {target_token_b}",
                        "expected_answer_logit_margin": expected_answer_margin,
                        "context_token_source": prompt_alignment.get("context_token_source"),
                        "answer_token_text": prompt_alignment.get("cache_answer_token_text"),
                        "context_token_text": prompt_alignment.get("context_token_text"),
                        "answer_state_norm": float(torch.linalg.vector_norm(snapshot.answer_states[0]).item()),
                        "context_state_norm": float(torch.linalg.vector_norm(snapshot.context_states[0]).item()),
                        "projected_context_state_norm": float(
                            torch.linalg.vector_norm(snapshot.projected_states[0]).item()
                        ),
                        "selected_store_state_norm": float(
                            torch.linalg.vector_norm(snapshot.final_latent_states[0]).item()
                        ),
                        "selected_store_state_source": (
                            "projected_context_state"
                            if prompt_example["context_token_index"] is not None
                            and int(prompt_example["context_token_index"]) >= 0
                            else "answer_state_fallback"
                        ),
                        "projected_selected_state_delta_norm": float(
                            torch.linalg.vector_norm(
                                snapshot.final_latent_states[0] - snapshot.projected_states[0]
                            ).item()
                        ),
                        "answer_context_cosine": float(
                            torch.nn.functional.cosine_similarity(
                                snapshot.answer_states[0].reshape(1, -1),
                                snapshot.context_states[0].reshape(1, -1),
                            ).item()
                        ),
                    }
                )
                for stage_name, stage_tensor in (
                    ("answer_state", snapshot.answer_states[0]),
                    ("context_state", snapshot.context_states[0]),
                    ("projected_context_state", snapshot.projected_states[0]),
                    ("selected_store_state", snapshot.final_latent_states[0]),
                ):
                    trajectory_matrix_rows.append(stage_tensor.detach().cpu())
                    trajectory_metadata.append(
                        {
                            "example_index": int(index),
                            "stage_name": stage_name,
                            "projection_group": "latent_state_evolution",
                            "group_name": str(prompt_example["group_name"]),
                            "probe_surface_text": str(prompt_example["probe_surface_text"]),
                            "expected_answer": expected_answer,
                            "expected_answer_logit_margin": expected_answer_margin,
                            "selected_store_state_source": (
                                "projected_context_state"
                                if prompt_example["context_token_index"] is not None
                                and int(prompt_example["context_token_index"]) >= 0
                                else "answer_state_fallback"
                            ),
                            "hover_label": (
                                f"example={index} | stage={stage_name} | group={prompt_example['group_name']} | "
                                f"probe={prompt_example['probe_surface_text']}"
                            ),
                        }
                    )

            trajectory_matrix = torch.stack(trajectory_matrix_rows) if trajectory_matrix_rows else torch.empty((0, 0))
            direction_metadata = [
                {
                    "stage_name": stage_name,
                    "stage_rank": payload.get("stage_rank"),
                    "stage_kind": payload.get("stage_kind"),
                    "projection_group": "direction_vector",
                    "hover_label": stage_name,
                }
                for stage_name, payload in stage_vectors.items()
            ]

            empty_projection = {
                "method": str(projection_method),
                "backend": "not-run",
                "metadata": {},
                "records": [],
            }

            trajectory_projection = (
                _serialize_projection_records(
                    trajectory_matrix,
                    trajectory_metadata,
                    method=projection_method,
                    projection_kwargs=projection_kwargs,
                    umap_backend_preference=umap_backend_preference,
                )
                if trajectory_matrix.numel()
                else dict(empty_projection)
            )
            direction_matrix = torch.stack(
                [
                    torch.as_tensor(payload["direction_vector"], dtype=torch.float32)
                    for payload in stage_vectors.values()
                ]
            )
            direction_projection = (
                _serialize_projection_records(
                    direction_matrix,
                    direction_metadata,
                    method=projection_method,
                    projection_kwargs=projection_kwargs,
                    umap_backend_preference=umap_backend_preference,
                )
                if direction_matrix.numel()
                else dict(empty_projection)
            )
            feature_vector_projection = (
                _serialize_projection_records(
                    feature_vector_matrix,
                    feature_vector_rows,
                    method=projection_method,
                    projection_kwargs=projection_kwargs,
                    umap_backend_preference=umap_backend_preference,
                )
                if feature_vector_matrix.numel()
                else dict(empty_projection)
            )
            aggregate_matrix_rows = list(trajectory_matrix_rows)
            aggregate_matrix_rows.extend(
                torch.as_tensor(payload["direction_vector"], dtype=torch.float32) for payload in stage_vectors.values()
            )
            aggregate_matrix_rows.extend(vector_row for vector_row in feature_vector_matrix)
            aggregate_metadata = [{**metadata, "aggregate_kind": "latent_state"} for metadata in trajectory_metadata]
            aggregate_metadata.extend(
                {**metadata, "aggregate_kind": "direction_vector"} for metadata in direction_metadata
            )
            aggregate_metadata.extend(
                {**metadata, "aggregate_kind": f"feature_{metadata['vector_type']}"} for metadata in feature_vector_rows
            )
            aggregate_projection = (
                _serialize_projection_records(
                    torch.stack(aggregate_matrix_rows),
                    aggregate_metadata,
                    method=projection_method,
                    projection_kwargs=projection_kwargs,
                    umap_backend_preference=umap_backend_preference,
                )
                if aggregate_matrix_rows
                else dict(empty_projection)
            )

            return {
                "config": {
                    "experiment_name": cfg.experiment_name,
                    "config_name": cfg.experiment_config_name,
                    "config_path": str(Path(config_path).expanduser().resolve()),
                    "prompt": cfg.prompt,
                    "rendered_prompt": rendered_prompt,
                    "prompt_render_mode": cfg.prompt_render_mode,
                    "calibration_surface": "orange" if "orange" in cfg.experiment_name else "concept_pair",
                    "target_tokens": list(target_tokens),
                    "target_token_ids": [int(target_a_id), int(target_b_id)],
                    "top_n": int(cfg.top_n),
                    "stage_top_n": int(effective_stage_top_n),
                    "concept_direction_mode": cfg.analysis_direction_mode_name,
                    "store_latent_extraction_mode": cfg.store_latent_extraction_mode,
                    "context_enhanced_scale": float(cfg.context_enhanced_scale),
                    "use_answer_state_as_basis": bool(cfg.use_answer_state_as_basis),
                    "projection_backend_preference": umap_backend_preference,
                    "resolved_key_tokens": _serialize_key_tokens(cfg, tokenizer),
                    "constrained_feature_selection": _serialize_constrained_feature_selection(
                        cfg.constrained_feature_selection_refs
                    ),
                },
                "prediction_info": prediction_info,
                "examples": example_rows,
                "context_snapshots": [snapshot.to_dict() for snapshot in context_snapshots],
                "pair_residuals": cast(list[dict[str, Any]], paired_residual_payload["records"]),
                "stage_vectors": stage_vectors,
                "stage_graphs": stage_graphs,
                "feature_vectors": feature_vector_rows,
                "trajectory_projection": trajectory_projection,
                "direction_projection": direction_projection,
                "feature_vector_projection": feature_vector_projection,
                "aggregate_projection": aggregate_projection,
            }
    finally:
        if should_cleanup_work_root:
            shutil.rmtree(Path(cfg.work_root), ignore_errors=True)


def build_latent_dynamics_frames(report: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd  # type: ignore[import-untyped]

    def _sorted_frame(rows: list[dict[str, Any]], sort_columns: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(sort_columns)

    stage_feature_rows: list[dict[str, Any]] = []
    for stage_name, payload in cast(dict[str, dict[str, Any]], report.get("stage_graphs", {})).items():
        feature_rows = cast(list[list[int]], payload.get("top_feature_rows", []))
        feature_scores = cast(list[float], payload.get("top_feature_scores", []))
        for rank, (feature_row, feature_score) in enumerate(zip(feature_rows, feature_scores, strict=True), start=1):
            feature_metadata = next(
                (
                    detail
                    for detail in cast(list[dict[str, Any]], payload.get("top_feature_details", []))
                    if int(detail["layer"]) == int(feature_row[0]) and int(detail["feature_id"]) == int(feature_row[2])
                ),
                {},
            )
            stage_feature_rows.append(
                {
                    "stage_name": stage_name,
                    "stage_rank": payload.get("stage_rank"),
                    "stage_kind": payload.get("stage_kind"),
                    "rank": int(rank),
                    "layer": int(feature_row[0]),
                    "position": int(feature_row[1]),
                    "feature_id": int(feature_row[2]),
                    "score": float(feature_score),
                    "feature_url": feature_metadata.get("feature_url"),
                    "feature_explanation": feature_metadata.get("feature_explanation"),
                    "key_tokens": payload.get("key_tokens", []),
                }
            )

    stage_summary_rows: list[dict[str, Any]] = []
    for stage_name, payload in cast(dict[str, dict[str, Any]], report.get("stage_vectors", {})).items():
        geometry = cast(dict[str, Any], payload.get("geometry", {}))
        normalized_direction = cast(dict[str, Any], geometry.get("normalized_direction", {}))
        raw_direction = cast(dict[str, Any], geometry.get("raw_direction", {}))
        stage_graph_payload = cast(dict[str, Any], report.get("stage_graphs", {})).get(stage_name, {})
        stage_summary_rows.append(
            {
                "stage_name": stage_name,
                "stage_rank": payload.get("stage_rank"),
                "stage_kind": payload.get("stage_kind"),
                "direction_norm": float(cast(dict[str, Any], payload.get("direction", {})).get("norm", 0.0)),
                "raw_direction_norm": raw_direction.get("norm"),
                "latent_row_count": geometry.get("latent_row_count"),
                "group_a_weight_sum": cast(dict[str, Any], geometry.get("group_weight_sums", {})).get("group_a"),
                "group_b_weight_sum": cast(dict[str, Any], geometry.get("group_weight_sums", {})).get("group_b"),
                "top_feature_count": len(cast(list[list[int]], stage_graph_payload.get("top_feature_rows", []))),
                "normalized_direction_sha": normalized_direction.get("sha256"),
            }
        )

    stage_key_token_rows: list[dict[str, Any]] = []
    for stage_name, payload in cast(dict[str, dict[str, Any]], report.get("stage_graphs", {})).items():
        for key_token in cast(list[dict[str, Any]], payload.get("key_tokens", [])):
            stage_key_token_rows.append(
                {
                    "stage_name": stage_name,
                    "stage_rank": payload.get("stage_rank"),
                    "token_id": key_token.get("token_id"),
                    "token_label": key_token.get("token_label"),
                }
            )

    return {
        "config": pd.DataFrame([cast(dict[str, Any], report.get("config", {}))]),
        "examples": pd.DataFrame(cast(list[dict[str, Any]], report.get("examples", []))),
        "pair_residuals": pd.DataFrame(cast(list[dict[str, Any]], report.get("pair_residuals", []))),
        "stage_summary": _sorted_frame(stage_summary_rows, ["stage_rank"]),
        "stage_features": _sorted_frame(stage_feature_rows, ["stage_rank", "rank"]),
        "stage_key_tokens": _sorted_frame(stage_key_token_rows, ["stage_rank", "token_label"]),
        "feature_vectors": _sorted_frame(
            cast(list[dict[str, Any]], report.get("feature_vectors", [])),
            ["stage_rank", "rank", "vector_type"],
        ),
        "trajectory_projection": pd.DataFrame(
            cast(dict[str, Any], report.get("trajectory_projection", {})).get("records", [])
        ),
        "direction_projection": pd.DataFrame(
            cast(dict[str, Any], report.get("direction_projection", {})).get("records", [])
        ),
        "feature_vector_projection": pd.DataFrame(
            cast(dict[str, Any], report.get("feature_vector_projection", {})).get("records", [])
        ),
        "aggregate_projection": pd.DataFrame(
            cast(dict[str, Any], report.get("aggregate_projection", {})).get("records", [])
        ),
    }


__all__ = ["build_latent_dynamics_frames", "run_latent_dynamics_analysis"]
