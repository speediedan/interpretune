import os
import json
from pathlib import Path

import torch

from circuit_tracer.frontend.graph_models import Node
from circuit_tracer.graph import Graph, prune_graph
from dataclasses import dataclass
from typing import Optional, Dict, Any


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

def get_topk_edges_for_node_range(
    node_range: tuple,
    adjacency_matrix: torch.Tensor,
    topk: int = 5,
    dim: int = 0
):
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
    token_ids: torch.Tensor = None,
    token_strings: list = None,
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

def generate_topk_node_mapping(
    graph, node_mask, topk_feats_to_translate=None, cumulative_scores=None, tokenizer=None
):
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
            return tuple(
                [node_mapping.get(idx, None) for idx in row.tolist()]
                for row in adj_indices
            )
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
    graph: "Graph" = None,
    adjacency_matrix: torch.Tensor = None,
    adj_matrix_name: str = "adjacency_matrix",
    node_ranges: dict = None,
    node_mapping: dict = None,
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
    selected_rows[~adj_mask] = float('-inf')  # Mask out invalid rows
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


node_threshold = 0.8
edge_threshold = 0.98
OS_HOME = os.environ.get("HOME")


local_np_graph_data = Path(OS_HOME) / "repos" / "local_np_graph_data"
target_example_dir = 'circuit_tracer_demo_specific_gradient_flow_attribution_example_orig'
target_example_raw_data_file = 'attribution_graph_ex_0.pt'
target_example_graph_file = 'ex-0.json'
raw_graph_inspect = local_np_graph_data / target_example_dir / target_example_raw_data_file
raw_graph_data = torch.load(raw_graph_inspect, weights_only=False, map_location="cpu")
graph_json_path = local_np_graph_data / target_example_dir / target_example_graph_file
graph_dict = load_graph_json(graph_json_path)
locals().update(unpack_objs_from_pt_dict(raw_graph_data))

graph = Graph.from_pt(raw_graph_inspect)
device = "cuda" if torch.cuda.is_available() else "cpu"
graph.to(device)
node_mask, edge_mask, cumulative_scores = (
    el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold)
)
graph.to("cpu")



# Examining logit node edges in the adjacency matrix directly
# Set our target_token_ids either by str (with tokenizer) or manually
target_token_ids = torch.tensor([26865, 22605], device=graph.logit_tokens.device)

# Generate our raw graph overview
raw_graph_overview = gen_raw_graph_overview(k=5, target_token_ids=target_token_ids, graph=graph, node_mask=node_mask)

# Explore as desired
# raw_graph_overview.first_order_node_ids
# (
#     ['20_15589_7', 'E_26865_6', '0_24_7', '21_5943_7', '23_12237_7'],
#     ['E_26865_6', '20_15589_7', '21_5943_7', '14_2268_6', '16_25_6']
# )
# raw_graph_overview.first_order_values
# tensor([[6.0000, 5.9062, 3.6719, 3.5000, 2.8594],
#         [9.6875, 5.5000, 3.8906, 2.8281, 2.7812]])
# raw_graph_overview.idxs_to_node_ids(6588)
# ['E_26865_6']

# Or indvidually analyze the adjacency matrix
# generate our node mapping and ranges
node_mapping, node_ranges = generate_topk_node_mapping(graph, node_mask)

# Get our topk edges for a given node range
topk_logit_vals, topk_logit_indices = get_topk_edges_for_node_range(node_ranges["logit_nodes"], graph.adjacency_matrix)

# Get our target logit indices into both the adjacency matrix and our logit_probabilities/logit_tokens vector
adj_matrix_target_logit_idxs, target_logit_vec_idxs = get_logit_indices_for_tokens(graph, target_token_ids)

# Gather our target logit topk edge values using the full adj_matrix logit indices
target_topk_logit_vals = torch.gather(graph.adjacency_matrix[adj_matrix_target_logit_idxs], 1,
                                      topk_logit_indices[target_logit_vec_idxs])

# Get node_ids for the target logit indices in the adjacency matrix
node_ids_for_target_logit_nodes = get_node_ids_for_adj_matrix_indices(adj_matrix_target_logit_idxs, node_mapping)

# Example output:
# node_ids_for_target_logit_nodes
# ['27_22605_0', '27_26865_5']

# Get the node_ids for the topk edges for our target logit nodes
node_ids_for_topk_edges_of_target_logit_nodes = get_node_ids_for_adj_matrix_indices(
    topk_logit_indices[target_logit_vec_idxs],
    node_mapping
)

# Example output:
# node_ids_for_topk_edges_of_target_logit_nodes[0]
# ['20_15589_7', 'E_26865_6', '0_24_7', '21_5943_7', '23_12237_7']
# node_ids_for_topk_edges_of_target_logit_nodes[1]
# ['E_26865_6', '20_15589_7', '21_5943_7', '14_2268_6', '16_25_6']
