# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared example helper functions for it_examples."""

import os
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

import torch

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Import VarAnnotate from orchestrator (renamed from VarInspect)
from it_examples.utils.analysis_injection.orchestrator import VarAnnotate


@dataclass
class TargetTokenAnalysis:
    """Encapsulated data for target tokens in attribution analysis.

    All fields are aligned by token index - e.g., tokens[0], token_ids[0], etc.,
    correspond to the same token.
    """

    # Core token info - at least one of tokens or token_ids must be provided
    tokens: Optional[List[str]] = None  # e.g., ['▁Dallas', '▁Austin']
    token_ids: Optional["torch.Tensor"] = None  # e.g., tensor([26865, 22605])

    # Optional tokenizer for conversion between tokens and token_ids
    tokenizer: Optional[Any] = None  # Tokenizer handle for conversion

    # Default device for tensor operations
    default_device: Optional[str] = None  # e.g., 'cuda', 'cpu'

    # Activation matrix for feature conversion
    act_matrix: Optional["torch.Tensor"] = None  # Activation matrix for nodes_to_features conversion

    # Runtime-derived logit info (torch.Tensor for operations)
    logit_indices: Optional["torch.Tensor"] = None  # e.g., tensor([0, 1]) - indices into logit arrays
    logit_probabilities: Optional["torch.Tensor"] = None  # e.g., tensor([0.298, 0.456])

    # Edge matrix info (torch.Tensor for operations)
    edge_matrix_indices: Optional["torch.Tensor"] = None  # e.g., tensor([7358, 8921]) - indices into edge_matrix

    # Optional additional analysis fields
    top_init_edge_vals: Optional["torch.Tensor"] = None  # Shape: (n_tokens, k) - e.g., top 5 vals per token
    top_init_edge_indices: Optional["torch.Tensor"] = None  # Shape: (n_tokens, k) - e.g., top 5 indices per token

    def __post_init__(self):
        """Validate and initialize fields."""
        # Validate that at least one of tokens or token_ids is provided
        if self.tokens is None and self.token_ids is None:
            raise ValueError("At least one of 'tokens' or 'token_ids' must be provided")

        # If tokenizer is available and we have one but not the other, convert
        if self.tokenizer is not None:
            if self.tokens is None and self.token_ids is not None:
                # token_ids might already be a tensor, convert to list for tokenizer
                token_ids_list = self.token_ids.tolist() if isinstance(self.token_ids, torch.Tensor) else self.token_ids
                self.tokens = self.tokenizer.convert_ids_to_tokens(token_ids_list)
            elif self.token_ids is None and self.tokens is not None:
                token_ids_list = self.tokenizer.convert_tokens_to_ids(self.tokens)
                self.token_ids = torch.tensor(token_ids_list, device=self.default_device)
        else:
            # No tokenizer, both must be provided
            if self.tokens is None or self.token_ids is None:
                raise ValueError("Both 'tokens' and 'token_ids' must be provided when no tokenizer is available")
            # Convert token_ids to tensor if it's a list
            if not isinstance(self.token_ids, torch.Tensor):
                self.token_ids = torch.tensor(self.token_ids, device=self.default_device)

        # Ensure token_ids is a tensor
        if self.token_ids is not None and not isinstance(self.token_ids, torch.Tensor):
            self.token_ids = torch.tensor(self.token_ids, device=self.default_device)

        # Validate alignment of all provided tensor fields
        tensor_fields = [self.logit_indices, self.logit_probabilities, self.edge_matrix_indices]
        tensor_lengths = [len(field) for field in tensor_fields if field is not None]
        if tensor_lengths and len(set(tensor_lengths)) != 1:
            raise ValueError(f"All tensor fields must have the same length, got {tensor_lengths}")

        # Validate tensor fields align with tokens/token_ids
        # At this point, both tokens and token_ids should be set due to __post_init__ logic
        assert self.tokens is not None and self.token_ids is not None
        n_tokens = len(self.tokens)
        if self.logit_indices is not None and len(self.logit_indices) != n_tokens:
            raise ValueError(f"logit_indices length {len(self.logit_indices)} must match tokens length {n_tokens}")
        if self.logit_probabilities is not None and len(self.logit_probabilities) != n_tokens:
            raise ValueError(
                f"logit_probabilities length {len(self.logit_probabilities)} must match tokens length {n_tokens}"
            )
        if self.edge_matrix_indices is not None and len(self.edge_matrix_indices) != n_tokens:
            raise ValueError(
                f"edge_matrix_indices length {len(self.edge_matrix_indices)} must match tokens length {n_tokens}"
            )

        if self.top_init_edge_vals is not None and self.top_init_edge_vals.shape[0] != n_tokens:
            raise ValueError("top_init_edge_vals must align with number of tokens")

    def update_logit_info(self, logit_idx: "torch.Tensor", logit_p: "torch.Tensor") -> None:
        """Update logit_indices and logit_probabilities based on target token_ids alignment.

        Args:
            logit_idx: Tensor of logit indices (e.g., tensor([22605, 573, ...]))
            logit_p: Tensor of corresponding logit probabilities (e.g., tensor([0.2988, 0.1250, ...]))

        This ensures tokens, token_ids, logit_indices, and logit_probabilities are all aligned
        by descending logit_probability order.
        """
        if torch is None:
            raise ImportError("torch is required for update_logit_info method")

        if self.token_ids is None:
            raise ValueError("token_ids must be set to update logit info")

        # Find indices where logit_idx matches our target token_ids
        target_token_ids_tensor = self.token_ids.to(logit_idx.device)
        mask = (logit_idx.unsqueeze(1) == target_token_ids_tensor).any(dim=1)
        matching_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if len(matching_indices) == 0:
            raise ValueError("No matching token_ids found in logit_idx")

        # Get the corresponding probabilities and sort by descending probability
        matching_probs = logit_p[matching_indices]
        sorted_indices = torch.argsort(matching_probs, descending=True)

        # Update the fields in descending probability order
        self.logit_indices = matching_indices[sorted_indices]
        self.logit_probabilities = matching_probs[sorted_indices]

        # Move tensors to default_device if set
        if self.default_device is not None:
            self.logit_indices = self.logit_indices.to(self.default_device)
            self.logit_probabilities = self.logit_probabilities.to(self.default_device)

        # Reorder tokens and token_ids to match the logit probability order
        # Get the original positions of the matching tokens in our token_ids list
        matching_token_ids = logit_idx[matching_indices]
        sorted_token_indices = []
        token_ids_list = self.token_ids.tolist()
        for token_id in matching_token_ids[sorted_indices]:
            sorted_token_indices.append(token_ids_list.index(int(token_id.item())))

        self.tokens = [self.tokens[i] for i in sorted_token_indices]  # type: ignore
        reordered_token_ids = [token_ids_list[i] for i in sorted_token_indices]
        self.token_ids = torch.tensor(reordered_token_ids, device=self.default_device)

    def get_token_info(self, index: int) -> dict:
        """Get all info for a specific token as a dict."""
        info = {
            "token": self.tokens[index] if self.tokens else None,
            "token_id": self.token_ids[index] if self.token_ids else None,
        }
        if self.logit_indices is not None:
            info["logit_index"] = self.logit_indices[index].item()
        if self.logit_probabilities is not None:
            info["logit_probability"] = self.logit_probabilities[index].item()
        if self.edge_matrix_indices is not None:
            info["edge_matrix_index"] = self.edge_matrix_indices[index].item()
        if self.top_init_edge_vals is not None:
            info["top_edge_vals"] = self.top_init_edge_vals[index]
        if self.top_init_edge_indices is not None:
            info["top_edge_indices"] = self.top_init_edge_indices[index]
        return info

    def to_dataframe(self):
        """Convert to a pandas DataFrame for tabular inspection (requires pandas)."""
        import pandas as pd

        data = {
            "token": self.tokens,
            "token_id": self.token_ids.tolist() if isinstance(self.token_ids, torch.Tensor) else self.token_ids,
        }
        if self.logit_indices is not None:
            data["logit_index"] = self.logit_indices.tolist()
        if self.logit_probabilities is not None:
            data["logit_probability"] = self.logit_probabilities.tolist()
        if self.edge_matrix_indices is not None:
            data["edge_matrix_index"] = self.edge_matrix_indices.tolist()
        if self.top_init_edge_vals is not None:
            data["top_edge_vals"] = [row.tolist() for row in self.top_init_edge_vals]
        if self.top_init_edge_indices is not None:
            data["top_edge_indices"] = [row.tolist() for row in self.top_init_edge_indices]
        return pd.DataFrame(data)

    def nodes_to_features(
        self,
        target_nodes: Union[list[int], list[list[int]], "torch.Tensor"],
        act_matrix: Optional["torch.Tensor"] = None,
        feats_only: bool = False,
    ) -> Any:
        """Convert target nodes to feature tuples using the activation matrix.

        Args:
            target_nodes: List or tensor of node indices to convert. Can be:
                - 1D: [node1, node2, ...] or tensor([node1, node2, ...])
                - 2D: [[node1, node2, ...], [node3, node4, ...], ...] or
                      tensor([[node1, node2, ...], [node3, node4, ...], ...])
            act_matrix: Activation matrix to use. If None, uses self.act_matrix
            feats_only: If True, return only the feature tuples.
                       If False, return dict mapping indices to feature tuples.

        Returns:
            For 1D input:
                feats_only=False: {node_idx: (feat1, feat2, ...)}
                feats_only=True: [(feat1, feat2, ...), ...]
            For 2D input:
                feats_only=False: ({node_idx: (feat1, feat2, ...)}, {node_idx: (feat1, feat2, ...)}, ...)
                feats_only=True: ([(feat1, feat2, ...), ...], [(feat1, feat2, ...), ...], ...)
        """
        if act_matrix is None:
            assert self.act_matrix is not None, "act_matrix must be provided or set on the instance"
            act_matrix = self.act_matrix

        # Normalize input to tensor
        if isinstance(target_nodes, list):
            target_tensor = (
                torch.tensor(target_nodes)
                if len(target_nodes) > 0 and isinstance(target_nodes[0], list)
                else torch.tensor(target_nodes)
            )
        elif isinstance(target_nodes, torch.Tensor):
            target_tensor = target_nodes
        else:
            raise ValueError(f"target_nodes must be list or tensor, got {type(target_nodes)}")

        is_2d = target_tensor.dim() == 2

        def _safe_nodes_to_features(nodes_tensor):
            """Convert nodes tensor to dict of node:feature pairs with bounds checking."""
            # Check which indices are valid (within sparse tensor bounds)
            max_idx = act_matrix._nnz()
            valid_mask = nodes_tensor < max_idx

            # Get features for valid indices only
            valid_indices = nodes_tensor[valid_mask]
            node_to_feature = {}

            if len(valid_indices) > 0:
                # Get all valid features at once
                valid_features = act_matrix.indices().T[valid_indices]
                valid_features_tuples = [tuple(row.tolist()) for row in valid_features]

                # Create mapping from node index to feature tuple
                valid_node_to_feature = {
                    node_idx.item(): feature for node_idx, feature in zip(valid_indices, valid_features_tuples)
                }

                # Build result dict in input order
                for node_idx in nodes_tensor:
                    node_val = node_idx.item()
                    node_to_feature[node_val] = valid_node_to_feature.get(node_val, "non-feature node")
            else:
                # No valid indices, all are non-feature nodes
                for node_idx in nodes_tensor:
                    node_to_feature[node_idx.item()] = "non-feature node"

            return node_to_feature

        if is_2d:
            # Handle 2D input - return tuple of dicts
            results = []
            for i in range(target_tensor.shape[0]):
                row_tensor = target_tensor[i]
                row_dict = _safe_nodes_to_features(row_tensor)
                results.append(row_dict)

            if feats_only:
                return tuple(list(result.values()) for result in results)
            else:
                return tuple(results)
        else:
            # Handle 1D input
            result_dict = _safe_nodes_to_features(target_tensor)

            if feats_only:
                return list(result_dict.values())
            else:
                return result_dict


EnvVarSpec = Tuple[str, Union[str, Callable[[str], bool]]]


def validate_env_vars(env_specs: Sequence[EnvVarSpec]) -> bool:
    """Validate environment variables based on specified modes.

    Args:
        env_specs: Sequence of (env_var_name, mode) tuples where mode can be:
            - 'present': Variable must be present in os.environ (allows empty string)
            - 'non-empty': Variable must be present and not empty
            - callable: A lambda/function that takes the env value and returns bool

    Returns:
        True if all validations pass, False otherwise
    """
    for env_var, mode in env_specs:
        value = os.environ.get(env_var)
        if mode == "present":
            if value is None:
                return False
        elif mode == "non-empty":
            if not value:
                return False
        elif callable(mode):
            if value is None or not mode(value):
                return False
        else:
            raise ValueError(f"Invalid mode for {env_var}: {mode}")
    return True


def required_os_env(env_path: str | Path | None = None, env_reqs: Sequence[EnvVarSpec] | None = None) -> bool:
    """Load environment variables from .env file and validate required env vars.

    Args:
        env_path: Optional path to .env file. If None, load_dotenv auto-discovers.
        env_reqs: Optional sequence of (env_var, mode) tuples to validate.

    Returns:
        True if loading succeeds and all validations pass (or no reqs), False otherwise.
    """
    logger = logging.getLogger(__name__)

    if load_dotenv is None:
        logger.debug("dotenv not available, skipping env loading")
    else:
        loaded = False
        if env_path is not None:
            p = Path(env_path)
            if p.exists():
                loaded = load_dotenv(str(p))
            else:
                # Fall back to auto-discover
                loaded = load_dotenv()
        else:
            loaded = load_dotenv()

        if loaded:
            logger.debug("load_dotenv found and loaded a .env file")
        else:
            logger.debug("load_dotenv did not find a .env file")

    # Validate required env vars if provided
    if env_reqs:
        return validate_env_vars(env_reqs)
    return True


def collect_shapes(data: dict, local_vars: dict, var_inspects: Sequence[Union[str, VarAnnotate]]) -> dict:
    """Collect shape information from specified variables/attributes in local_vars context.

    Args:
        data: Dictionary to update with shape information
        local_vars: Local variables context to evaluate expressions in
        var_inspects: Sequence of variable inspection specifications. Each element can be:
            - str: Variable reference (e.g., "ctx.activation_matrix")
            - VarAnnotate: VarAnnotate dataclass with var_ref and annotation

    Returns:
        Updated data dictionary with shape information added

    Keys in data will be in format "{var_ref}.shape"
    Values will be VarAnnotate objects with output populated
    """
    for var_inspect in var_inspects:
        if isinstance(var_inspect, str):
            inspect_spec = VarAnnotate(var_ref=var_inspect)
        elif isinstance(var_inspect, VarAnnotate):
            inspect_spec = var_inspect
        else:
            continue  # Skip invalid specs

        key = f"{inspect_spec.var_ref}.shape"

        try:
            # Evaluate the variable reference in the local_vars context
            obj = eval(inspect_spec.var_ref, {"__builtins__": {}}, local_vars)

            # Check if the object has a shape attribute
            if hasattr(obj, "shape"):
                shape_tuple = tuple(obj.shape)
                inspect_spec.output = str(shape_tuple)
            else:
                inspect_spec.output = "not found or does not have shape attribute"

        except (NameError, AttributeError, KeyError, TypeError):
            inspect_spec.output = "not found or does not have shape attribute"

        # Store the VarAnnotate object in data
        data[key] = inspect_spec

    return data
