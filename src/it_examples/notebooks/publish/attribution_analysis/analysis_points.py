"""Default analysis point functions for the attribution analysis notebook."""

from __future__ import annotations

from typing import Any, Dict

import torch

from it_examples.utils.analysis_injection.analysis_hook_patcher import HOOK_REGISTRY, get_analysis_vars
from it_examples.utils.analysis_injection.orchestrator import (
    analysis_log_point,
    VarAnnotate,
    format_per_token,
    get_caller_context,
)
from it_examples.utils.example_helpers import collect_shapes
from it_examples.utils.raw_graph_analysis import get_topk_2nd_order_adjacency

# because we use analysis injection helper functions (get_analysis_vars) to update our local analysis point state:
# pyright: reportUndefinedVariable=false
# ruff: noqa: F821


def ap_compute_attribution_end(local_vars: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}

    # Collect shapes from attribution data
    collect_shapes(
        data,
        local_vars,
        [
            VarAnnotate("attribution_data['reconstruction']", var_value="", annotation="n_layers, n_pos, d_model"),
            VarAnnotate("attribution_data['decoder_vecs']", var_value="", annotation="num_active_features, d_model"),
        ],
    )
    analysis_log_point("after attribution component computation", data)


def ap_precomputation_phase_end(local_vars: Dict[str, Any]) -> None:
    # Use dict directly for cleaner access
    v = get_analysis_vars(context_keys=["target_token_analysis"], local_keys=["ctx"], local_vars=local_vars)
    v["target_token_analysis"].act_matrix = v["ctx"].activation_matrix
    target_logits = v["ctx"].logits[0, -1, v["target_token_analysis"].token_ids]
    data = {
        "target_token_ids": v["target_token_analysis"].token_ids,
        "target_logits": target_logits,
        "target_tokens": v["target_token_analysis"].tokens,
    }
    analysis_log_point("after precomputation phase", data)


def ap_forward_pass_end(local_vars: Dict[str, Any]) -> None:
    # Use dict directly for cleaner access
    v = get_analysis_vars(context_keys=["target_token_analysis"], local_keys=["ctx", "model"], local_vars=local_vars)

    resid_final = v["ctx"]._resid_activations[-1][-1, -1, :].detach()
    target_logits = resid_final @ v["model"].W_U[:, v["target_token_analysis"].token_ids]
    data = {
        "target_token_ids": v["target_token_analysis"].token_ids,
        "target_logits": target_logits,
        "target_tokens": v["target_token_analysis"].tokens,
    }
    analysis_log_point("after forward pass", data)


def ap_build_input_vectors_end(local_vars: Dict[str, Any]) -> None:
    # Use dict directly for cleaner access
    v = get_analysis_vars(
        context_keys=["target_token_analysis"],
        local_keys=[
            "logit_idx",
            "logit_p",
            "total_nodes",
            "total_active_feats",
            "max_feature_nodes",
            "edge_matrix",
            "row_to_node_index",
            "logit_vecs",
            "n_layers",
            "n_pos",
        ],
        local_vars=local_vars,
    )
    tta = v["target_token_analysis"]
    tta.update_logit_info(v["logit_idx"], v["logit_p"])
    max_n_logits = len(v["logit_idx"])

    HOOK_REGISTRY.set_context(
        max_feature_nodes=v["max_feature_nodes"],
        total_nodes=v["total_nodes"],
        logit_idx=v["logit_idx"],
        logit_p=v["logit_p"],
        n_pos=v["n_pos"],
        max_n_logits=max_n_logits,
    )
    data = {
        "logit_idx": v["logit_idx"],
        "logit_p": VarAnnotate("logit_p", var_value=v["logit_p"], annotation="non-demeaned logits probabilities"),
        "target_tokens": tta.tokens,
        "target_logit_indices": tta.logit_indices,
        "target_logit_p": tta.logit_probabilities,
        "logit_cumulative_prob": float(v["logit_p"].sum().item()),
        "total_nodes": v["total_nodes"],
        "max_feature_nodes": v["max_feature_nodes"],
        "total_active_feats": v["total_active_feats"],
        "n_logits": len(v["logit_idx"]),
        "n_layers": v["n_layers"],
        "n_pos": v["n_pos"],
        "max_n_logits": max_n_logits,
        "edge_matrix.shape": v["edge_matrix"].shape,
        "row_to_node_index.shape": v["row_to_node_index"].shape if v["row_to_node_index"] is not None else None,
        "logit_vecs.shape": v["logit_vecs"].shape,
    }
    analysis_log_point("after building input vectors w/ target logits", data)


def ap_compute_logit_attribution_end(local_vars: Dict[str, Any]) -> None:
    # Use dict directly for cleaner access
    v = get_analysis_vars(
        context_keys=["target_token_analysis"],
        local_keys=["edge_matrix", "logit_offset", "n_logits", "max_feature_nodes", "ctx"],
        local_vars=local_vars,
    )

    logit_section = v["edge_matrix"][: v["n_logits"], : v["max_feature_nodes"]]
    logit_section_zeros = torch.where(logit_section == 0)[1].unique()
    data = {}
    collect_shapes(data, local_vars, ["edge_matrix", "row_to_node_index"])
    data["logit_section.shape"] = logit_section.shape
    logit_zeros_min, logit_zeros_max = logit_section_zeros.min().item(), logit_section_zeros.max().item()
    data["logit_section_zeros_info"] = VarAnnotate(
        "logit_section_zeros_info",
        var_value=(logit_section_zeros.shape, logit_zeros_min, logit_zeros_max),
        annotation="logit section zeros shape, min node id, max node id",
    )
    inspect_logit_section_zero_boundary = [
        (logit_zeros_min - 1),
        logit_zeros_min,
        logit_zeros_max,
        (logit_zeros_max + 1),
    ]
    data["logit_section_zero_feature_range"] = VarAnnotate(
        "logit_section_zero_feature_range",
        var_value=v["target_token_analysis"].nodes_to_features(inspect_logit_section_zero_boundary),
        annotation="In the final layer, only the final position has non-zero feature attributions.",
    )
    analysis_log_point("after logit attribution", data)


def ap_compute_feature_attributions_end(local_vars: Dict[str, Any]) -> None:
    # Use dict directly for cleaner access
    v = get_analysis_vars(
        context_keys=["target_token_analysis"],
        local_keys=["n_visited", "max_feature_nodes", "logit_p", "edge_matrix", "ctx"],
        local_vars=local_vars,
    )
    tta = v["target_token_analysis"]
    top_init_edge_vals, top_init_edge_indices = torch.topk(
        v["edge_matrix"][tta.logit_indices.to(v["edge_matrix"].device), :], 5
    )
    top_init_edge_indices.squeeze_()
    top_init_edge_vals.squeeze_()
    tta.top_init_edge_vals = top_init_edge_vals.cpu()
    tta.top_init_edge_indices = top_init_edge_indices.cpu()
    tta.top_init_edge_features = tta.nodes_to_features(tta.top_init_edge_indices)

    data = {
        "target_tokens": tta.tokens,
        "target_logit_p": tta.logit_probabilities,
        "top_init_edge_indices": tta.top_init_edge_indices,
        "top_init_edge_features": tta.top_init_edge_features,
        "top_init_edge_vals": tta.top_init_edge_vals,
        "features_processed": v["n_visited"],
        "max_features": v["max_feature_nodes"],
        "progress": (
            f"{100 * v['n_visited'] / v['max_feature_nodes']:.1f}%"
            if v["max_feature_nodes"] and v["n_visited"] is not None
            else None
        ),
    }
    collect_shapes(
        data,
        local_vars,
        [
            VarAnnotate(
                "edge_matrix",
                var_value="",
                annotation=(
                    "( (n_logits + n_feature_nodes),  n_feature_nodes + n_error_nodes + n_token_nodes + n_logits"
                ),
            )
        ],
    )
    analysis_log_point("after feature attribution", data)


def ap_graph_creation_start(local_vars: Dict[str, Any]) -> None:
    # Use dict directly for cleaner access
    v = get_analysis_vars(
        context_keys=["target_token_analysis"],
        local_keys=["full_edge_matrix", "edge_matrix", "n_logits", "selected_features", "ctx"],
        local_vars=local_vars,
    )
    tta = v["target_token_analysis"]
    full_edge_matrix = v["full_edge_matrix"]
    n_logits = v["n_logits"]
    tta.reorg_logit_indices = (v["edge_matrix"].shape[0] - n_logits) + tta.logit_indices
    tta.graph_logit_indices = (full_edge_matrix.shape[0] - n_logits) + tta.logit_indices
    data: Dict[str, Any] = {}
    collect_shapes(data, local_vars, ["full_edge_matrix", "edge_matrix"])
    pre_normalized_logit_node_sum = full_edge_matrix[tta.graph_logit_indices.to(full_edge_matrix.device), :].sum(1)
    data["pre_normalized_logit_node_sum"] = VarAnnotate(
        "pre_normalized_logit_node_sum",
        var_value=pre_normalized_logit_node_sum,
        annotation="Sum of all source nodes for our target logits before normalization",
    )
    errvecs_shape = v["ctx"].error_vectors.shape
    n_total_nodes = len(v["selected_features"]) + (errvecs_shape[0] * errvecs_shape[1]) + errvecs_shape[1] + n_logits
    data["n_total_nodes"] = VarAnnotate(
        "n_total_nodes", var_value=n_total_nodes, annotation="len(selected_features):(n_layers * n_pos):n_pos:n_logits"
    )

    second_order_vals, second_order_indices, first_order_vals, first_order_indices = get_topk_2nd_order_adjacency(
        5,
        len(v["selected_features"]),
        tta.logit_indices.to(full_edge_matrix.device),
        10,
        adjacency_matrix=full_edge_matrix,
    )

    # Format per-token tensors
    per_token_tensors = [
        (first_order_indices, "pre_prune_1st_order_idxs"),
        (first_order_vals, "pre_prune_1st_order_vals"),
        (second_order_indices, "pre_prune_2nd_order_idxs"),
        (second_order_vals, "pre_prune_2nd_order_vals"),
    ]
    for var, attr in per_token_tensors:
        setattr(tta, attr, format_per_token(var, tta.tokens))
        data[attr] = getattr(tta, attr)

    analysis_log_point("Graph packaging complete", data)


def ap_node_compute_influence_init(local_vars: Dict[str, Any]) -> None:
    """Collect initial current_influence vector in compute_influence."""
    # Check call stack to determine context
    context = get_caller_context(
        candidate_ctxs=["compute_node_influence", "compute_edge_influence"], target_ctx="compute_node_influence"
    )

    # Only collect data for node influence context initially
    if context != "compute_node_influence":
        return

    v = get_analysis_vars(
        context_keys=["target_token_analysis"],
        local_keys=["A", "logit_weights", "current_influence", "max_n_logits"],
        local_vars=local_vars,
    )

    # Initialize trace dict with nested structure
    trace_dict = {"node": []}
    HOOK_REGISTRY.set_context(ap_node_compute_influence_trace=trace_dict)

    # Append initial influence vector
    trace_dict["node"].append(v["current_influence"].clone().detach().cpu())

    data = {
        "trace_dict": trace_dict,
        "iteration": 0,
        "context": "node",
        "current_influence_shape": v["current_influence"].shape,
    }

    tta = v["target_token_analysis"]
    logit_weights = v["logit_weights"]
    A = v["A"]
    max_n_logits = v["max_n_logits"]
    current_influence = v["current_influence"]

    data["normalized_tgt_logit_nodes"] = VarAnnotate(
        "normalized_tgt_logit_nodes",
        var_value=A[tta.graph_logit_indices, :].sum(dim=1),
        annotation="Verify normalization: `A[tta.graph_logit_indices, :].sum(dim=1)`",
    )

    second_order_vals, second_order_indices, first_order_vals, first_order_indices = get_topk_2nd_order_adjacency(
        5, v["max_n_logits"], tta.logit_indices.to(A.device), 10, adjacency_matrix=A
    )

    #  We don't display the second_order tensors since logit compute influence calculation isn't directly calculating
    #  feature-feature influences
    per_token_tensors = [
        (first_order_indices, "nodeinf_normed_1st_order_idxs"),
        (first_order_vals, "nodeinf_normed_1st_order_vals"),
        # (second_order_indices, "nodeinf_normed_2nd_order_idxs"), (second_order_vals, "nodeinf_normed_2nd_order_vals")
    ]
    for var, attr in per_token_tensors:
        setattr(tta, attr, format_per_token(var, tta.tokens))
        data[attr] = getattr(tta, attr)
    data["logit_weights"] = VarAnnotate(
        "logit_weights",
        var_value=logit_weights[-max_n_logits:],
        annotation="logit probabilites are injected as logit_weights to compute the initial influence vector",
    )
    data["initial_influence_feat_0"] = VarAnnotate(
        "initial_influence_feat_0",
        var_value=logit_weights[-max_n_logits:].dot(A[-max_n_logits:, 0]),
        annotation=(
            "Validate the initial element of our initial influence vector\n"
            "`logit_weights[-max_n_logits:].dot(A[-max_n_logits:, 0])`"
        ),
    )
    data["current_influence_feat_0"] = VarAnnotate(
        "current_influence_feat_0",
        var_value=current_influence[0].clone().cpu(),
        annotation=(
            "The first element of our current influence vector `current_influence[0]` consists of the "
            "weighted sum of the direct influences for each source feature on the logit probs"
        ),
    )

    analysis_log_point("After initial compute_influence computation (node context)", data)


def ap_node_compute_influence(local_vars: Dict[str, Any]) -> None:
    """Collect current_influence vectors after each iteration of compute_influence."""
    # Check call stack to determine context
    context = get_caller_context(
        candidate_ctxs=["compute_node_influence", "compute_edge_influence"], target_ctx="compute_node_influence"
    )

    # Only collect data for node influence context
    if context != "compute_node_influence":
        return

    v = get_analysis_vars(context_keys=[], local_keys=["current_influence"], local_vars=local_vars)

    # Get the trace dict (should already be initialized)
    trace_dict = HOOK_REGISTRY.get_context("ap_node_compute_influence_trace", {"node": []})

    # Append current influence vector
    if "node" not in trace_dict:
        trace_dict["node"] = []
    trace_dict["node"].append(v["current_influence"].clone().detach().cpu())

    data = {
        "trace_dict": trace_dict,
        "iteration": len(trace_dict["node"]) - 1,
        "context": "node",
        "current_influence_shape": v["current_influence"].shape,
    }

    analysis_log_point("After compute_influence iteration (node context)", data)


def ap_graph_prune_node_influence_end(local_vars: Dict[str, Any]) -> None:
    v = get_analysis_vars(
        local_keys=["node_influence", "node_mask", "node_threshold", "pruned_matrix", "n_logits", "n_tokens"],
        local_vars=local_vars,
    )
    pruned_matrix = v["pruned_matrix"]
    pruned_matrix_nonzero = pruned_matrix.count_nonzero().item()
    node_mask = v["node_mask"]
    node_mask_nonzero = node_mask.count_nonzero().item()

    data = {
        "n_logits": v["n_logits"],
        "n_tokens": v["n_tokens"],
        "node_mask_nonzero": VarAnnotate(
            "node_mask_nonzero",
            var_value=node_mask_nonzero,
            annotation=(
                "Number of non-zero elements in the node mask, includes token and logit nodes "
                "plus minimum required nodes to meet specified node threshold"
            ),
        ),
        "node_threshold": v["node_threshold"],
        "pruned_matrix_nonzero": pruned_matrix_nonzero,
    }
    retained_nodes = node_mask_nonzero - v["n_logits"] - v["n_tokens"]
    data["node_mask_nonlogit_nontoken_nonzero"] = VarAnnotate(
        "node_mask_nonlogit_nontoken_nonzero",
        var_value=retained_nodes,
        annotation="We always keep token and logit nodes, so we subtract them from the count",
    )
    data["percent_of_nodes_kept"] = VarAnnotate(
        "percent_of_nodes_kept",
        var_value=f"{round(retained_nodes / node_mask.shape[0] * 100, 2)}%",
        annotation="The percentage of nodes kept after pruning",
    )
    data["percent_of_elements_kept"] = VarAnnotate(
        "percent_of_elements_kept",
        var_value=f"{round(pruned_matrix_nonzero / pruned_matrix.numel() * 100, 2)}%",
        annotation=(
            "The percentage of non-zero elements kept in the pruned adjacency matrix "
            "after applying our node_mask to both dimensions"
        ),
    )
    analysis_log_point("After node_influence threshold pruning applied", data)


def ap_graph_prune_edge_influence_post_norm(local_vars: Dict[str, Any]) -> None:
    v = get_analysis_vars(
        context_keys=["target_token_analysis", "n_pos"],
        local_keys=["edge_scores", "normalized_pruned", "pruned_influence", "pruned_matrix", "max_n_logits"],
        local_vars=local_vars,
    )
    tta, edge_scores, pruned_matrix, pruned_influence, normalized_pruned = (
        v["target_token_analysis"],
        v["edge_scores"],
        v["pruned_matrix"],
        v["pruned_influence"],
        v["normalized_pruned"],
    )
    nontoken_nonlogit_offset = v["n_pos"] + v["max_n_logits"]
    data = {}

    collect_shapes(
        data,
        local_vars,
        [
            VarAnnotate("edge_scores", var_value="", annotation="total_nodes, total_nodes"),
            VarAnnotate("normalized_pruned", var_value="", annotation="total_nodes, total_nodes"),
            VarAnnotate("pruned_influence", var_value="", annotation="total_nodes"),
        ],
    )
    # total influence from pruned influence for each of the top initial edges (top feature nodes influencing logits)
    pruned_influence_top_init_feature_vals = pruned_influence[tta.top_init_edge_indices.to(pruned_influence.device)]
    # top 5 non-token, non-logit feature logit prob influences from pruned influence
    pruned_influence_top_nontoken_nonlogit_feature_vals, pruned_influence_top_nontoken_nonlogit_feature_idxs = (
        torch.topk(pruned_influence[0:-nontoken_nonlogit_offset], k=5)
    )
    # top individual edge prob for each target token logit from edge scores
    top_target_edge_vals, top_target_edge_idxs = torch.topk(edge_scores[tta.graph_logit_indices, :], k=5)
    # overall influence for each of the top features associated with top target logit edges
    pruned_influence_top_features_pruned = pruned_influence[top_target_edge_idxs]

    max_logit_max_feature_idx = edge_scores[-v["max_n_logits"], :].argmax()
    total_pruned_influence_max_logit_max_feature = pruned_influence[max_logit_max_feature_idx]
    total_logit_prob_influence_max_logit_max_feature = edge_scores[
        -v["max_n_logits"] :, max_logit_max_feature_idx
    ].sum()
    total_nonlogit_influence_max_logit_max_feature = edge_scores[: -v["max_n_logits"], max_logit_max_feature_idx].sum()
    prenormed_top_features = pruned_matrix[tta.top_init_edge_indices.to(pruned_matrix.device), :].sum(2)
    postnormed_top_features = normalized_pruned[tta.top_init_edge_indices.to(normalized_pruned.device), :].sum(2)
    target_token_logit_prob = edge_scores[tta.graph_logit_indices, :].sum(1)

    per_token_tensors = [
        (top_target_edge_vals, "top_target_edge_vals"),
        (top_target_edge_idxs, "top_target_edge_idxs"),
        (target_token_logit_prob, "target_token_logit_prob"),
    ]
    for var, attr in per_token_tensors:
        setattr(tta, attr, format_per_token(var, tta.tokens))
        data[attr] = getattr(tta, attr)

    # Loop over k values to compute top k edge logit probabilities and percentages
    ks = [1, 10, 100, 1000]
    for k in ks:
        top_k_values = torch.topk(edge_scores[tta.graph_logit_indices, :], k=k)[0]
        if k == 1:
            top_k_logit_prob = top_k_values.squeeze(-1)  # Shape: (n_tokens,)
        else:
            top_k_logit_prob = top_k_values.sum(1)  # Sum over top k for each token

        top_k_percent_logit_prob = top_k_logit_prob / target_token_logit_prob

        attr_name = f"top_{k}_edge{'s' if k > 1 else ''}_percent_logit_prob"
        setattr(tta, attr_name, format_per_token(top_k_percent_logit_prob, tta.tokens))
        data[attr_name] = getattr(tta, attr_name)

    data["prenormed_top_features"] = VarAnnotate(
        "prenormed_top_features",
        var_value=prenormed_top_features,
        annotation="Sum of pre-normalization pruned matrix values for our top initial edge indices",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["postnormed_top_features"] = VarAnnotate(
        "postnormed_top_features",
        var_value=postnormed_top_features,
        annotation=(
            "Verify normalization of the post-normalization pruned matrix values for our top initial edge indices"
        ),
        format_tensor_kwargs={"float_precision": 3},
    )
    data["max_logit_max_feature_idx"] = VarAnnotate(
        "max_logit_max_feature_idx",
        var_value=max_logit_max_feature_idx,
        annotation="Index of the feature with the maximum edge score for our top logit",
    )
    data["total_pruned_influence_max_logit_max_feature"] = VarAnnotate(
        "total_pruned_influence_max_logit_max_feature",
        var_value=total_pruned_influence_max_logit_max_feature,
        annotation="Total pruned influence for the feature with max edge score to our top logit",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["total_logit_prob_influence_max_logit_max_feature"] = VarAnnotate(
        "total_logit_prob_influence_max_logit_max_feature",
        var_value=total_logit_prob_influence_max_logit_max_feature,
        annotation="Sum of edge scores to all logits from the feature with max edge score for our top logit",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["total_nonlogit_influence_max_logit_max_feature"] = VarAnnotate(
        "total_nonlogit_influence_max_logit_max_feature",
        var_value=total_nonlogit_influence_max_logit_max_feature,
        annotation="Sum of edge scores to non-logit nodes from the feature with max edge score for our top logit.",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["pruned_influence_top_init_feature_vals"] = VarAnnotate(
        "pruned_influence_top_init_feature_vals",
        var_value=pruned_influence_top_init_feature_vals,
        annotation=(
            "Total influence from pruned influence for each of the top initial edges "
            "(top feature nodes influencing logits)"
        ),
        format_tensor_kwargs={"float_precision": 3},
    )
    data["pruned_influence_top_nontoken_nonlogit_feature_vals"] = VarAnnotate(
        "pruned_influence_top_nontoken_nonlogit_feature_vals",
        var_value=pruned_influence_top_nontoken_nonlogit_feature_vals,
        annotation="Top 5 non-token, non-logit feature total influences from pruned influence",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["pruned_influence_top_nontoken_nonlogit_feature_idxs"] = VarAnnotate(
        "pruned_influence_top_nontoken_nonlogit_feature_idxs",
        var_value=pruned_influence_top_nontoken_nonlogit_feature_idxs,
        annotation="Indices of top 5 non-token, non-logit feature total influences from pruned influence",
    )
    data["pruned_influence_top_features_pruned"] = VarAnnotate(
        "pruned_influence_top_features_pruned",
        var_value=pruned_influence_top_features_pruned,
        annotation="Overall influence for each of the top features associated with top target logit edges",
        format_tensor_kwargs={"float_precision": 3},
    )
    analysis_log_point("After edge influence calculation", data)


def ap_graph_prune_edge_influence_pre_mask(local_vars: Dict[str, Any]) -> None:
    v = get_analysis_vars(
        # context_keys=["target_token_analysis", "n_pos"],
        local_keys=["edge_mask", "node_mask", "logit_weights", "edge_scores", "n_logits"],
        local_vars=local_vars,
    )
    edge_mask, node_mask, edge_scores, logit_weights, n_logits = (
        v["edge_mask"],
        v["node_mask"],
        v["edge_scores"],
        v["logit_weights"],
        v["n_logits"],
    )
    data = {}
    data["edge_mask_numel"] = VarAnnotate(
        "edge_mask_numel", var_value=edge_mask.numel(), annotation="Number of elements in the edge mask"
    )
    data["edge_mask_count_nonzero"] = VarAnnotate(
        "edge_mask_count_nonzero",
        var_value=edge_mask.count_nonzero(),
        annotation="Number of non-zero elements in the edge mask",
    )
    data["edge_mask_percent_nonzero"] = VarAnnotate(
        "edge_mask_percent_nonzero",
        var_value=f"{(edge_mask.count_nonzero().item() / edge_mask.numel()) * 100:.2f}%",
        annotation="Percentage of non-zero elements in the edge mask",
    )
    data["node_mask_numel"] = VarAnnotate(
        "node_mask_numel", var_value=node_mask.numel(), annotation="Number of elements in the node mask"
    )
    data["node_mask_count_nonzero"] = VarAnnotate(
        "node_mask_count_nonzero",
        var_value=node_mask.count_nonzero(),
        annotation="Number of non-zero elements in the node mask",
    )
    data["node_mask_percent_nonzero"] = VarAnnotate(
        "node_mask_percent_nonzero",
        var_value=f"{(node_mask.count_nonzero().item() / node_mask.numel()) * 100:.2f}%",
        annotation="Percentage of non-zero elements in the node mask",
    )
    data["sum_edge_score_attribution_to_target_logits"] = VarAnnotate(
        "sum_edge_score_attribution_to_target_logits",
        var_value=edge_scores[-n_logits:, :].sum(),
        annotation="Sum of edge score attribution to target logits, `edge_scores[-n_logits:, :].sum()`",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["sum_logit_probabilities"] = VarAnnotate(
        "sum_logit_probabilities",
        var_value=logit_weights[-n_logits:].sum(),
        annotation=(
            "Sum of logit probabilities for target logits, `logit_weights[-n_logits:].sum()`, "
            "should match aggregate edge score attribution"
        ),
        format_tensor_kwargs={"float_precision": 3},
    )
    analysis_log_point("After edge influence calculation", data)


def ap_graph_prune_edge_influence_end(local_vars: Dict[str, Any]) -> None:
    v = get_analysis_vars(
        context_keys=["target_token_analysis"],
        local_keys=[
            "sorted_scores",
            "cumulative_scores",
            "sorted_indices",
            "final_scores",
            "edge_mask",
            "node_mask",
        ],
        local_vars=local_vars,
    )
    sorted_scores = v["sorted_scores"]
    cumulative_scores = v["cumulative_scores"]
    sorted_indices = v["sorted_indices"]
    final_scores = v["final_scores"]
    edge_mask = v["edge_mask"]
    node_mask = v["node_mask"]

    data = {}
    data["cumulative_scores_first_10"] = VarAnnotate(
        "cumulative_scores_first_10",
        var_value=cumulative_scores[:10],
        annotation="First 10 values of cumulative_scores",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["cumulative_scores_last_10"] = VarAnnotate(
        "cumulative_scores_last_10",
        var_value=cumulative_scores[-10:],
        annotation="Last 10 values of cumulative_scores",
    )
    data["cumulative_scores_30th_feature"] = VarAnnotate(
        "cumulative_scores_30th_feature",
        var_value=cumulative_scores[29],
        annotation="Cumulative score of the 30th feature",
        format_tensor_kwargs={"float_precision": 3},
    )
    contributors_mask = torch.where(cumulative_scores < 1, 1, 0)
    data["contributors_mask_sum"] = VarAnnotate(
        "contributors_mask_sum",
        var_value=contributors_mask.sum(dim=0),
        annotation="Number of non-zero contributors in cumulative_scores (< 1)",
    )
    data["sorted_indices_first_10"] = VarAnnotate(
        "sorted_indices_first_10",
        var_value=sorted_indices[:10],
        annotation="First 10 values of sorted_indices",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["sorted_scores_first_10"] = VarAnnotate(
        "sorted_scores_first_10",
        var_value=sorted_scores[:10],
        annotation="First 10 values of sorted_scores (from node_influence)",
        format_tensor_kwargs={"float_precision": 3},
    )
    data["final_scores_first_10"] = VarAnnotate(
        "final_scores_first_10",
        var_value=final_scores[:10],
        annotation=(
            "The returned final_scores reflects the CUMULATIVE node influences of each feature "
            "up to that feature number (create_nodes requires the cumulative_scores form of features "
            "for graph construction)"
        ),
        format_tensor_kwargs={"float_precision": 3},
    )
    data["node_mask_count_nonzero"] = VarAnnotate(
        "node_mask_count_nonzero",
        var_value=node_mask.count_nonzero(),
        annotation="Number of non-zero elements in node_mask",
    )
    data["node_mask_sparsity"] = VarAnnotate(
        "node_mask_sparsity",
        var_value=f"{(node_mask.count_nonzero().item() / node_mask.numel()) * 100:.2f}%",
        annotation="Final sparsity of node_mask (percentage of non-zero elements)",
    )
    data["edge_mask_count_nonzero"] = VarAnnotate(
        "edge_mask_count_nonzero",
        var_value=edge_mask.count_nonzero(),
        annotation="Number of non-zero elements in edge_mask",
    )
    data["edge_mask_sparsity"] = VarAnnotate(
        "edge_mask_sparsity",
        var_value=f"{(edge_mask.count_nonzero().item() / edge_mask.numel()) * 100:.2f}%",
        annotation="Final sparsity of edge_mask (percentage of non-zero elements)",
    )
    analysis_log_point("After edge influence pruning end", data)


AP_FUNCTIONS = {
    "ap_compute_attribution_end": ap_compute_attribution_end,
    "ap_precomputation_phase_end": ap_precomputation_phase_end,
    "ap_forward_pass_end": ap_forward_pass_end,
    "ap_build_input_vectors_end": ap_build_input_vectors_end,
    "ap_compute_logit_attribution_end": ap_compute_logit_attribution_end,
    "ap_compute_feature_attributions_end": ap_compute_feature_attributions_end,
    "ap_graph_creation_start": ap_graph_creation_start,
    "ap_node_compute_influence_init": ap_node_compute_influence_init,
    "ap_node_compute_influence": ap_node_compute_influence,
    "ap_graph_prune_node_influence_end": ap_graph_prune_node_influence_end,
    "ap_graph_prune_edge_influence_post_norm": ap_graph_prune_edge_influence_post_norm,
    "ap_graph_prune_edge_influence_pre_mask": ap_graph_prune_edge_influence_pre_mask,
    "ap_graph_prune_edge_influence_end": ap_graph_prune_edge_influence_end,
}

__all__ = ["AP_FUNCTIONS"]
