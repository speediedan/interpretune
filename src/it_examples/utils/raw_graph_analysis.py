import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from circuit_tracer.frontend.graph_models import Node

if TYPE_CHECKING:
    from circuit_tracer.graph import Graph


# **TODO** save/parameterize the below vectorized 2nd order adjacency_matrix influence sampling function
# for later use in utils
# example:
# top_graph_2nd_order_vals, top_graph_2nd_order_indices = get_topk_2nd_order_adjacency(
#     5, 6374, torch.tensor((0,5)), 10, None, full_edge_matrix, None
# )


def get_topk_2nd_order_adjacency(
    k: int,
    limit_node: int,
    inspect_logit_idxs: torch.Tensor,
    n_logits: int = 10,
    graph: Optional["Graph"] = None,
    adjacency_matrix: Optional[torch.Tensor] = None,
    adj_matrix_name: str = "adjacency_matrix",
):
    """Returns the top-k 2nd order adjacency values and indices for non-error, non-logit nodes.

    Args:
        k (int): Number of top values to select per row.
        limit_node (int): Only consider nodes with indices < limit_node.
        top_adj_edge_postnorm_indices (torch.Tensor): Indices of top adjacency edges (shape: [n_rows, k]).
        graph (Graph, optional): Graph object containing adjacency matrices.
        adjacency_matrix (torch.Tensor, optional): Adjacency matrix to use directly.
        adj_matrix_name (str): Name of the adjacency matrix attribute in graph.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (topk_values, topk_indices), both of shape [n_rows, k, k]
    """
    if adjacency_matrix is None:
        if graph is None:
            raise ValueError("Either 'graph' or 'adjacency_matrix' must be provided.")
        adjacency_matrix = getattr(graph, adj_matrix_name)
    adj_logit_idxs = (adjacency_matrix.shape[0] - n_logits) + inspect_logit_idxs
    first_order_values, first_order_indices = torch.topk(adjacency_matrix[adj_logit_idxs, :], k)
    adj_mask = first_order_indices < limit_node
    safe_indices = first_order_indices.clone()
    safe_indices[~adj_mask] = 0  # Replace invalid indices with a dummy row (e.g., 0)
    selected_rows = adjacency_matrix[safe_indices]
    selected_rows[~adj_mask] = float("-inf")  # Mask out invalid rows
    flat_rows = selected_rows.view(-1, adjacency_matrix.shape[1])
    topk_values, topk_indices = torch.topk(flat_rows, k, dim=1)
    topk_values = topk_values.view(*selected_rows.shape[:2], k)
    topk_indices = topk_indices.view(*selected_rows.shape[:2], k)
    return topk_values, topk_indices, first_order_values, first_order_indices


# version of compute_influence that lets us trace the approximate neumann series convergence


def compute_influence_trace_and_save(
    A: torch.Tensor, logit_weights: torch.Tensor, pt_path: str, max_iter: int = 1000, trace_total: bool = False
):
    """Computes influence per iteration, saves adjacency matrix, logit_weights, and trace to a .pt file."""
    current_influence = logit_weights @ A
    influence = current_influence
    trace = [current_influence.clone()]
    iterations = 0
    while current_influence.any():
        if iterations >= max_iter:
            raise RuntimeError(f"Influence computation failed to converge after {iterations} iterations")
        current_influence = current_influence @ A
        influence += current_influence
        trace.append(current_influence.clone() if not trace_total else influence.clone())
        iterations += 1
    # Save to .pt file
    torch.save(
        {
            "adjacency_matrix": A,
            "logit_weights": logit_weights,
            "trace": torch.stack(trace),
        },
        pt_path,
    )
    return trace


# plot summary stats of a tensor where each row contains a distribution of a convergence process


def tensor_distribution_summary_stats(tensor: torch.Tensor):
    """Returns summary statistics (mean, std, min, max, median, nonzero count) for each row in a tensor. Each row
    is assumed to represent a distribution from a given iteration.

    Args:
        tensor (torch.Tensor): shape (num_iterations, num_values)

    Returns:
        List[dict]: List of dicts containing summary stats for each iteration.
    """
    stats = []
    for i in range(tensor.shape[0]):
        row = tensor[i]
        stats.append(
            {
                "iteration": i,
                "mean": row.mean().item(),
                "std": row.std().item(),
                "min": row.min().item(),
                "max": row.max().item(),
                "median": row.median().item(),
                "nonzero_count": (row != 0).sum().item(),
                "total_count": row.numel(),
            }
        )
    return stats


def plot_ridgeline_convergence(
    data: torch.Tensor,
    stats: Optional[List[Dict[str, Any]]] = None,
    title: str = "Convergence Distribution Ridgeline Plot",
):
    """Plot ridgeline distributions for convergence data using Plotly.

    Args:
        data: Tensor of shape [n_iterations_to_convergence, n_total_nodes] containing the convergence data
        stats: List of dictionaries containing summary statistics for each iteration
        title: Title for the plot
    """
    if not stats:
        stats = tensor_distribution_summary_stats(data)
    if data.shape[0] != len(stats):
        raise ValueError(f"Data has {data.shape[0]} rows but {len(stats)} stat entries")

    fig = go.Figure()

    # Color scale for iterations
    colors = px.colors.sequential.Viridis
    color_scale = np.linspace(0, 1, len(stats))

    # Calculate y-positions for ridgeline effect
    y_spacing = 1.0
    y_positions = np.arange(len(stats)) * y_spacing

    # Process each iteration
    for i, (row_data, stat) in enumerate(zip(data, stats)):
        # Convert to numpy and filter out zeros for better visualization
        values = row_data.cpu().numpy() if torch.is_tensor(row_data) else row_data
        nonzero_values = values[values > 0]

        if len(nonzero_values) == 0:
            # Handle the case where all values are zero (like iteration 27)
            continue

        # Use log scale for better visualization of wide range
        log_values = np.log10(nonzero_values + 1e-40)  # Add small epsilon to handle zeros

        # Create histogram to get density
        counts, bin_edges = np.histogram(log_values, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize density for ridgeline effect (scale down for visibility)
        normalized_density = counts / np.max(counts) * 0.8 if np.max(counts) > 0 else counts

        # Create y-coordinates for this distribution
        y_coords = y_positions[i] + normalized_density

        # Add the distribution curve
        color_idx = int(color_scale[i] * (len(colors) - 1))
        color = colors[color_idx]

        # Fill area under curve
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([bin_centers, bin_centers[::-1]]),
                y=np.concatenate([y_coords, np.full(len(bin_centers), y_positions[i])]),
                fill="toself",
                fillcolor=color,
                opacity=0.6,
                line=dict(color=color, width=1),
                name=f"Iteration {stat['iteration']}",
                hovertemplate=(
                    f"Iteration {stat['iteration']}<br>"
                    + f"Mean: {stat['mean']:.2e}<br>"
                    + f"Max: {stat['max']:.2e}<br>"
                    + f"Nonzero: {stat['nonzero_count']}/{stat['total_count']}<br>"
                    + "Log10(Value): %{x:.2f}<br>"
                    + "<extra></extra>"
                ),
            )
        )

        # Add baseline
        fig.add_trace(
            go.Scatter(
                x=[bin_centers.min(), bin_centers.max()],
                y=[y_positions[i], y_positions[i]],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(title="Log10(Value + ε)", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(
            title="Iteration",
            tickmode="array",
            tickvals=y_positions,
            ticktext=[f"Iter {i}" for i in range(len(stats))],
            showgrid=False,
        ),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        width=1200,
        height=800,
        plot_bgcolor="white",
    )

    # Add annotations for key statistics (mean and max for first several iterations)
    for i, stat in enumerate(stats[:8]):  # Show stats for first 8 iterations
        if stat["nonzero_count"] > 0:
            # Mean annotation
            fig.add_annotation(
                x=np.log10(stat["mean"] + 1e-40),
                y=y_positions[i] + 0.3,
                text=f"μ={stat['mean']:.1e}",
                showarrow=False,
                font=dict(size=8, color="blue"),
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1,
                opacity=0.9,
            )

            # Max annotation
            fig.add_annotation(
                x=np.log10(stat["max"] + 1e-40),
                y=y_positions[i] + 0.6,
                text=f"max={stat['max']:.1e}",
                showarrow=False,
                font=dict(size=8, color="red"),
                bgcolor="white",
                bordercolor="red",
                borderwidth=1,
                opacity=0.9,
            )

    return fig


# TODO: convert these manual examples to usage docstring w/ tests or remove
# # example usage of neumann series convergence tracing

# trace = compute_influence_trace_and_save(
#     A, logit_weights, '/tmp/adjmat_logit_trace_neumann_convergence.pt'
# )  # by default, traces only marginal updates (the diminishing updates of each order's update)
# trace = compute_influence_trace_and_save(
#     A, logit_weights, '/tmp/adjmat_logit_trace_neumann_convergence.pt', trace_total=True
# )  # trace the total influence matrix evolution (not the diminishing updates of each order's update)

# # within ipynb notebook
# # target neumann series convergence trace
# trace_path = './adjmat_logit_trace_neumann_convergence.pt'

# data = torch.load(trace_path, map_location="cpu")
# trace = data["trace"]  # shape: (num_iters, ...)

# fig = plot_ridgeline_convergence(data=trace, stats=None, title="Neumann Series Convergence Trace Ridgeline Plot")
# fig.show()


# convenience function to unpack variables in a deserialized pytorch dictionary to local scope
# this is useful for debugging and inspecting the contents of a deserialized graph
# e.g. `raw_graph_data['adjacency_matrix']` will be available as `adjacency_matrix`
def unpack_objs_from_pt_dict(tensor_dict):
    for key, value in tensor_dict.items():
        locals()[key] = value
    return locals()


def load_graph_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_topk_edges_for_node_range(node_range: tuple, adjacency_matrix: torch.Tensor, topk: int = 5, dim: int = 0):
    """Returns the topk edge values and indices for the specified node_range along the given dimension.

    Args:
        node_range (tuple): (start_idx, end_idx) specifying the node indices.
        adjacency_matrix (torch.Tensor): The adjacency matrix to analyze.
        topk (int, optional): Number of top edges to return. Defaults to 5.
        dim (int, optional): Dimension along which to compute topk. Defaults to 0.

    Returns:
        topk_vals (torch.Tensor): Topk edge values for each node in the range.
        topk_indices (torch.Tensor): Indices of the topk edges for each node in the range.
    """
    idxs = range(node_range[0], node_range[1])
    selected = adjacency_matrix[idxs, :] if dim == 0 else adjacency_matrix[:, idxs]
    topk_vals, topk_indices = torch.topk(selected, topk, dim=1 if dim == 0 else 0)
    return topk_vals, topk_indices


def get_logit_indices_for_tokens(
    graph,
    token_ids: Optional[torch.Tensor] = None,
    token_strings: Optional[list] = None,
    tokenizer=None,
):
    """Given a tensor of token ids or a list of token strings (with tokenizer), return the corresponding indices in
    the adjacency matrix for the logit nodes of those tokens.

    Args:
        graph: The graph object containing logit_tokens and adjacency_matrix.
        token_ids (torch.Tensor, optional): Tensor of token ids to inspect.
        token_strings (list, optional): List of token strings to inspect.
        tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer to convert strings to ids.

    Returns:
        torch.Tensor: Indices in the adjacency matrix for the logit nodes of the specified tokens.
    """
    if token_strings is not None and tokenizer is not None:
        inspect_ids = torch.tensor(tokenizer.convert_tokens_to_ids(token_strings), device=graph.logit_tokens.device)
    elif token_ids is not None:
        inspect_ids = token_ids.to(graph.logit_tokens.device)
    else:
        raise ValueError("Either token_ids or (token_strings and tokenizer) must be provided.")

    lmask = (graph.logit_tokens.unsqueeze(1) == inspect_ids).any(dim=1)
    indices = torch.nonzero(lmask, as_tuple=False).cpu().squeeze()
    if indices.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)
    adj_offset = graph.adjacency_matrix.shape[0] - len(graph.logit_tokens)
    final_logit_idxs = adj_offset + indices
    return final_logit_idxs, indices


def generate_topk_node_mapping(graph, node_mask, topk_feats_to_translate=None, cumulative_scores=None, tokenizer=None):
    """Returns a dict mapping node_idx to node_id string for the specified number of features, and also returns the
    calculated node index ranges for feature, error, token, and logit nodes.

    If topk_feats_to_translate is None, all features are included.
    """
    node_ids = {}
    n_features = len(graph.selected_features)
    layers = graph.cfg.n_layers
    n_pos = graph.n_pos

    # Calculate index boundaries for node types
    feature_start_idx = 0
    feature_end_idx = n_features
    error_start_idx = feature_end_idx
    error_end_idx = error_start_idx + (layers * n_pos)
    token_start_idx = error_end_idx
    token_end_idx = token_start_idx + n_pos
    logit_start_idx = token_end_idx
    # If cumulative_scores is provided, use its length for logit_end_idx, else infer from graph
    if cumulative_scores is not None:
        logit_end_idx = len(cumulative_scores)
    elif hasattr(graph, "logit_tokens"):
        logit_end_idx = logit_start_idx + len(graph.logit_tokens)
    else:
        logit_end_idx = logit_start_idx

    node_ranges = {
        "feature_nodes": (feature_start_idx, feature_end_idx),
        "error_nodes": (error_start_idx, error_end_idx),
        "token_nodes": (token_start_idx, token_end_idx),
        "logit_nodes": (logit_start_idx, logit_end_idx),
    }

    # Determine how many features to process
    if topk_feats_to_translate is None or topk_feats_to_translate > n_features:
        topk = n_features
    else:
        topk = topk_feats_to_translate

    # Get node indices from node_mask
    node_indices = node_mask.nonzero().squeeze().tolist()
    if isinstance(node_indices, int):
        node_indices = [node_indices]

    count = 0
    for node_idx in node_indices:
        if node_idx in range(feature_start_idx, feature_end_idx):
            layer, pos, feat_idx = graph.active_features[graph.selected_features[node_idx]].tolist()
            node_id = f"{layer}_{feat_idx}_{pos}"
            node_ids[node_idx] = node_id
            count += 1
            if count >= topk:
                break
        elif node_idx in range(error_start_idx, error_end_idx):
            layer, pos = divmod(node_idx - error_start_idx, n_pos)
            node_id = Node.error_node(layer, pos).node_id
            node_ids[node_idx] = node_id
        elif node_idx in range(token_start_idx, token_end_idx):
            pos = node_idx - token_start_idx
            vocab_idx = graph.input_tokens[pos]
            node_id = f"E_{vocab_idx}_{pos}"
            node_ids[node_idx] = node_id
        elif node_idx in range(logit_start_idx, logit_end_idx):
            pos = node_idx - logit_start_idx
            vocab_idx = graph.logit_tokens[pos]
            layer = str(layers + 1)
            node_id = f"{layer}_{vocab_idx}_{pos}"
            node_ids[node_idx] = node_id
    return node_ids, node_ranges


def get_node_ids_for_adj_matrix_indices(adj_indices, node_mapping):
    """Returns the node_ids for all adjacency_matrix target nodes provided.

    Args:
        adj_indices (Union[list, torch.Tensor]): Indices of nodes in the adjacency matrix.
        node_mapping (dict): Mapping from node index to node_id.

    Returns:
        list or tuple: List of node_ids if input is 1D, or tuple of lists (one per dim 0 entry)
            if input is multi-dimensional.
    """
    if isinstance(adj_indices, torch.Tensor):
        if adj_indices.dim() == 1:
            indices = adj_indices.tolist()
            return [node_mapping.get(idx, None) for idx in indices]
        elif adj_indices.dim() > 1:
            # Return a tuple of lists, one for each dim 0 entry
            return tuple([node_mapping.get(idx, None) for idx in row.tolist()] for row in adj_indices)
    elif isinstance(adj_indices, (list, tuple, int)):
        # Convert to list if it's a single integer
        if isinstance(adj_indices, int):
            adj_indices = [adj_indices]
        # Assume 1D list or tuple
        return [node_mapping.get(idx, None) for idx in adj_indices]
    else:
        raise TypeError("adj_indices must be an int, 1D list/tuple or torch.Tensor")


@dataclass
class RawGraphOverview:
    first_order_values: torch.Tensor
    first_order_indices: torch.Tensor
    second_order_values: torch.Tensor
    second_order_indices: torch.Tensor
    node_ranges: dict
    node_mapping: dict
    adj_matrix_target_logit_idxs: torch.Tensor
    target_logit_vec_idxs: torch.Tensor
    extra: Optional[Dict[str, Any]] = None

    def node_ids_for(self, idxs):
        """Given indices (tensor or list), return corresponding node_ids using the node_mapping."""
        return get_node_ids_for_adj_matrix_indices(idxs, self.node_mapping)

    def idxs_to_node_ids(self, idxs):
        return get_node_ids_for_adj_matrix_indices(idxs, self.node_mapping)

    @property
    def first_order_node_ids(self):
        return self.node_ids_for(self.first_order_indices)

    @property
    def second_order_node_ids(self):
        return self.node_ids_for(self.second_order_indices)

    @property
    def adj_matrix_target_logit_node_ids(self):
        return self.node_ids_for(self.adj_matrix_target_logit_idxs)


def gen_raw_graph_overview(
    k: int,
    target_token_ids: torch.Tensor,
    graph: Optional["Graph"] = None,
    adjacency_matrix: Optional[torch.Tensor] = None,
    adj_matrix_name: str = "adjacency_matrix",
    node_ranges: Optional[dict] = None,
    node_mapping: Optional[dict] = None,
    node_mask: Optional[torch.Tensor] = None,
):
    """Returns the top-k 2nd order adjacency values and indices for non-error, non-logit nodes, as well as the
    first order adjacency values and indices for the specified logit nodes. Also returns node_ranges, node_mapping,
    and adj_matrix_target_logit_idxs.

    Args:
        k (int): Number of top values to select per row (second order).
        target_token_ids (torch.Tensor): Token ids of logit nodes to inspect.
        graph (Graph, optional): Graph object containing adjacency matrices.
        adjacency_matrix (torch.Tensor, optional): Adjacency matrix to use directly.
        adj_matrix_name (str): Name of the adjacency matrix attribute in graph.
        node_ranges (dict, optional): Node ranges dict as returned by generate_topk_node_mapping.
        node_mapping (dict, optional): Node mapping dict as returned by generate_topk_node_mapping.
        node_mask (torch.Tensor, optional): Mask of active nodes to use for mapping.

    Returns:
        RawGraphOverview: Dataclass containing all relevant outputs.
    """
    if adjacency_matrix is None:
        if graph is None:
            raise ValueError("Either 'graph' or 'adjacency_matrix' must be provided.")
        adjacency_matrix = getattr(graph, adj_matrix_name)

    if node_ranges is None or node_mapping is None:
        # Use provided node_mask or default to all nodes
        if node_mask is None:
            node_mask = torch.ones(adjacency_matrix.shape[0], dtype=torch.bool)
        node_mapping, node_ranges = generate_topk_node_mapping(graph, node_mask)

    limit_node = node_ranges["feature_nodes"][1]

    # Get logit indices for those tokens
    adj_matrix_target_logit_idxs, target_logit_vec_idxs = get_logit_indices_for_tokens(
        graph, token_ids=target_token_ids
    )

    # First order: get topk edges for logit nodes
    first_order_values, first_order_indices = get_topk_edges_for_node_range(
        node_ranges["logit_nodes"], adjacency_matrix
    )
    # Select only the inspected logit nodes
    first_order_values = first_order_values[target_logit_vec_idxs]
    first_order_indices = first_order_indices[target_logit_vec_idxs]

    # Second order: for each topk first order edge, get its topk outgoing edges (if in feature_nodes)
    adj_mask = first_order_indices < limit_node
    safe_indices = first_order_indices.clone()
    safe_indices[~adj_mask] = 0  # Replace invalid indices with a dummy row (e.g., 0)
    selected_rows = adjacency_matrix[safe_indices]
    selected_rows[~adj_mask] = float("-inf")  # Mask out invalid rows
    flat_rows = selected_rows.view(-1, adjacency_matrix.shape[1])
    second_order_values, second_order_indices = torch.topk(flat_rows, k, dim=1)
    second_order_values = second_order_values.view(*selected_rows.shape[:2], k)
    second_order_indices = second_order_indices.view(*selected_rows.shape[:2], k)

    return RawGraphOverview(
        first_order_values=first_order_values,
        first_order_indices=first_order_indices,
        second_order_values=second_order_values,
        second_order_indices=second_order_indices,
        node_ranges=node_ranges,
        node_mapping=node_mapping,
        adj_matrix_target_logit_idxs=adj_matrix_target_logit_idxs,
        target_logit_vec_idxs=target_logit_vec_idxs,
    )


# NOTE: For a complete example of using these functions with Circuit Tracer graphs,
# see raw_graph_analysis_example_incomplete.py (TODO: requires local demo data to be made public)
