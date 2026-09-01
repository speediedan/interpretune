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
    """Lazily load transcoder encoder weights from disk on access.

    When ``None`` (default), auto-enabled when ``offload='cpu'`` to avoid OOM during transcoder
    loading for large transcoder widths (e.g. 262k with 4B+ models).
    """
    lazy_encoder: bool | None = None
    """Lazily load transcoder decoder weights from disk on access (default ``True``)."""
    lazy_decoder: bool = True
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

    # Analysis-level feature intervention settings
    """Base scale factor applied to each constructed intervention value."""
    intervention_scale_factor: float = 1.0
    """Scale each feature by ``abs(score) / max(abs(score))`` before applying ``intervention_scale_factor``."""
    intervention_max_influence_norm_scale: bool = False
    """Use ``top_feature_scores`` sign to choose intervention direction when scores are available."""
    intervention_sign_aware_scale: bool = True
    """Optional constant value for all interventions.

    When ``None``, the op uses per-feature values from ``top_feature_scores``.
    """
    intervention_value: float | None = None
    """Source for per-feature intervention values when ``intervention_value`` is unset."""
    intervention_value_source: str = "top_feature_scores"
    """Optional explicit constrained layer list passed to circuit-tracer intervention APIs."""
    intervention_constrained_layers: list[int] | None = None
    """Optional passthrough for circuit-tracer attention freezing during intervention."""
    intervention_freeze_attention: bool | None = None
    """Optional passthrough controlling activation-function application during intervention."""
    intervention_apply_activation_function: bool | None = None
    """Whether to request sparse intervention activations from circuit-tracer."""
    intervention_sparse: bool = False
    """Whether to request intervention activations alongside logits from circuit-tracer."""
    intervention_return_activations: bool = False

    # NNsight backend-specific settings
    """Whether to use remote execution for NNsight backend.

    Only applicable when backend='nnsight'. Enables running models on NNsight's remote infrastructure.
    """
    nnsight_remote: bool = False
    """API key for NNsight remote execution.

    Only applicable when backend='nnsight' and nnsight_remote=True.
    """
    ndif_api_key: str | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Resolve NDIF API key from environment if not explicitly set
        if self.ndif_api_key is None and self.nnsight_remote:
            self.ndif_api_key = os.environ.get("NDIF_API_KEY")

        # Validate backend selection
        valid_backends = ["transformerlens", "nnsight"]
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend '{self.backend}'. Must be one of {valid_backends}")

        valid_intervention_value_sources = ["top_feature_scores", "top_feature_activation_values", "constant"]
        if self.intervention_value_source not in valid_intervention_value_sources:
            raise ValueError(
                "Invalid intervention_value_source "
                f"'{self.intervention_value_source}'. Must be one of {valid_intervention_value_sources}"
            )

        if self.intervention_value_source == "constant" and self.intervention_value is None:
            raise ValueError("intervention_value must be set when intervention_value_source='constant'")

        # Warn if NNsight-specific settings are configured but backend is not NNsight
        if self.backend != "nnsight":
            if self.nnsight_remote:
                import warnings

                warnings.warn(
                    "nnsight_remote=True but backend is not 'nnsight'. This setting will be ignored.", UserWarning
                )
            if self.ndif_api_key is not None:
                import warnings

                warnings.warn(
                    "ndif_api_key is set but backend is not 'nnsight'. This setting will be ignored.", UserWarning
                )
