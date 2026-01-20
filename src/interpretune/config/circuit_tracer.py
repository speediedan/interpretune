from __future__ import annotations
from dataclasses import dataclass
import os
import torch

from interpretune.config.shared import ITSerializableCfg


@dataclass(kw_only=True)
class CircuitTracerConfig(ITSerializableCfg):
    """Configuration for Circuit Tracer functionality.

    This configuration extends ITLensConfig with circuit tracing specific parameters for generating attribution graphs
    using ReplacementModel and transcoders.
    """

    """Backend to use for attribution.

    Options:
        - 'transformerlens': Use TransformerLens/HookedTransformer backend (default)
        - 'nnsight': Use NNsight/LanguageModel backend
    """
    backend: str = "transformerlens"

    # Model and transcoder settings
    """Model name to use for attribution. If None, uses the base model name."""
    model_name: str | None = None
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
    max_feature_nodes: int | None = None
    """Memory optimization option ('cpu', 'disk', or None)."""
    offload: str | None = None
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
    graph_output_dir: str | None = None

    # Interpretune CT enhancement settings
    """ Specific tokens to analyze, will use tokens associated with top `max_n_logits` if `None`."""
    analysis_target_tokens: list[str] | None = None
    """A tensor of pre-tokenized target token IDs for analysis or a module attribute to be used as a source for
    them."""
    target_token_ids: list[int] | torch.Tensor | str | None = None
    """Whether to prepare graphs for Neuronpedia graph storage and analysis."""
    use_neuronpedia: bool = False

    # NNsight backend-specific settings
    """Whether to use remote execution for NNsight backend.

    Only applicable when backend='nnsight'. Enables running models on NNsight's remote infrastructure.
    """
    nnsight_remote: bool = False
    """API key for NNsight remote execution.

    Only applicable when backend='nnsight' and nnsight_remote=True.
    """
    nnsight_api_key: str | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Resolve NNsight API key from environment if not explicitly set
        if self.nnsight_api_key is None and self.nnsight_remote:
            self.nnsight_api_key = os.environ.get("NNSIGHT_API_KEY")

        # Validate backend selection
        valid_backends = ["transformerlens", "nnsight"]
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend '{self.backend}'. Must be one of {valid_backends}")

        # Warn if NNsight-specific settings are configured but backend is not NNsight
        if self.backend != "nnsight":
            if self.nnsight_remote:
                import warnings

                warnings.warn(
                    "nnsight_remote=True but backend is not 'nnsight'. This setting will be ignored.", UserWarning
                )
            if self.nnsight_api_key is not None:
                import warnings

                warnings.warn(
                    "nnsight_api_key is set but backend is not 'nnsight'. This setting will be ignored.", UserWarning
                )
