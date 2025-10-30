"""
Raw Graph Analysis Example - INCOMPLETE

TODO: This example demonstrates manual raw graph analysis using local artifacts that are not yet
publicly available. Complete this example with publicly accessible demo data or parametrize it
for user-provided graph data.

The code below shows how to use the raw_graph_analysis utility functions to inspect
Circuit Tracer graphs manually, but requires local data files that don't exist in CI/testing environments.
"""

import os
from pathlib import Path

import torch

from circuit_tracer.graph import Graph, prune_graph
from it_examples.utils.raw_graph_analysis import (
    generate_topk_node_mapping,
    gen_raw_graph_overview,
    get_logit_indices_for_tokens,
    get_node_ids_for_adj_matrix_indices,
    get_topk_edges_for_node_range,
    load_graph_json,
    unpack_objs_from_pt_dict,
)

# TODO: Replace these hardcoded paths with configurable parameters or publicly available demo data
node_threshold = 0.8
edge_threshold = 0.98
OS_HOME = os.environ.get("HOME")
if OS_HOME is None:
    raise RuntimeError("HOME environment variable is not set")

local_it_demo_graph_data = Path(OS_HOME) / "repos" / "local_it_demo_graph_data"
target_example_dir = "ct_attribution_analysis_example"
target_example_raw_data_file = "it_circuit_tracer_compute_specific_logits_demo_1_20250929_102245.pt"
target_example_graph_file = "it_circuit_tracer_compute_specific_logits_demo_1_20250929_102245.json"
raw_graph_inspect = local_it_demo_graph_data / target_example_dir / target_example_raw_data_file
raw_graph_data = torch.load(raw_graph_inspect, weights_only=False, map_location="cpu")
graph_json_path = local_it_demo_graph_data / target_example_dir / target_example_graph_file
graph_dict = load_graph_json(graph_json_path)
locals().update(unpack_objs_from_pt_dict(raw_graph_data))

graph = Graph.from_pt(str(raw_graph_inspect))
device = "cuda" if torch.cuda.is_available() else "cpu"
graph.to(device)
node_mask, edge_mask, cumulative_scores = (el.cpu() for el in prune_graph(graph, node_threshold, edge_threshold))
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
target_topk_logit_vals = torch.gather(
    graph.adjacency_matrix[adj_matrix_target_logit_idxs], 1, topk_logit_indices[target_logit_vec_idxs]
)

# Get node_ids for the target logit indices in the adjacency matrix
node_ids_for_target_logit_nodes = get_node_ids_for_adj_matrix_indices(adj_matrix_target_logit_idxs, node_mapping)

# Example output:
# node_ids_for_target_logit_nodes
# ['27_22605_0', '27_26865_5']

# Get the node_ids for the topk edges for our target logit nodes
node_ids_for_topk_edges_of_target_logit_nodes = get_node_ids_for_adj_matrix_indices(
    topk_logit_indices[target_logit_vec_idxs], node_mapping
)

# Example output:
# node_ids_for_topk_edges_of_target_logit_nodes[0]
# ['20_15589_7', 'E_26865_6', '0_24_7', '21_5943_7', '23_12237_7']
# node_ids_for_topk_edges_of_target_logit_nodes[1]
# ['E_26865_6', '20_15589_7', '21_5943_7', '14_2268_6', '16_25_6']
