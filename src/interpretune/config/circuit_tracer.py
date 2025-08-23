from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass
import torch

from interpretune.config.shared import ITSerializableCfg
from interpretune.config.transformer_lens import ITLensConfig


@dataclass(kw_only=True)
class CircuitTracerConfig(ITSerializableCfg):
    """Configuration for Circuit Tracer functionality.

    This configuration extends ITLensConfig with circuit tracing specific parameters for generating attribution graphs
    using ReplacementModel and transcoders.
    """

    # Model and transcoder settings
    """Model name to use for circuit tracing. If None, uses the base model name."""
    model_name: Optional[str] = None
    """Transcoder set to use.

    Can be 'gemma', 'llama', or path to custom config.
    """
    transcoder_set: str = "gemma"
    """Data type for model and transcoders."""
    dtype: torch.dtype = torch.bfloat16

    # Attribution parameters
    """Maximum number of logit nodes to attribute from."""
    max_n_logits: int = 10
    """Cumulative probability threshold for top logits."""
    desired_logit_prob: float = 0.95
    """Batch size for backward passes during attribution."""
    batch_size: int = 256
    """Maximum number of feature nodes to include in attribution."""
    max_feature_nodes: Optional[int] = None
    """Memory optimization option ('cpu', 'disk', or None)."""
    offload: Optional[str] = None
    """Whether to display detailed progress information."""
    verbose: bool = True

    # Graph visualization settings
    """Default threshold for node pruning in visualization."""
    default_node_threshold: float = 0.8
    """Default threshold for edge pruning in visualization."""
    default_edge_threshold: float = 0.98

    # Output settings
    """Whether to automatically save generated graphs."""
    save_graphs: bool = True
    """Directory to save attribution graphs.If None, uses analysis output directory."""
    graph_output_dir: Optional[str] = None

    # Interpretune CT enhancement settings
    """ Specific tokens to analyze, will use tokens associated with top `max_n_logits` if `None`."""
    analysis_target_tokens: Optional[List[str]] = None
    """A tensor of pre-tokenized target token IDs for analysis or a module attribute to be used as a source for
    them."""
    target_token_ids: Optional[List[int] | torch.Tensor | str] = None
    """Whether to prepare graphs for Neuronpedia graph storage and analysis."""
    use_neuronpedia: bool = False


# Add configuration mixins that extend existing configs
@dataclass(kw_only=True)
class CircuitTracerITLensConfig(ITLensConfig):
    """ITLens configuration with Circuit Tracer support."""

    circuit_tracer_cfg: Optional[CircuitTracerConfig] = None
