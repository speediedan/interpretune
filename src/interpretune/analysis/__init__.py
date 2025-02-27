"""Analysis submodule."""
from datasets.formatting import _register_formatter

from interpretune.analysis.ops import ANALYSIS_OPS
from interpretune.analysis.formatters import ITAnalysisFormatter
from interpretune.analysis.core import (
    AnalysisStore,
    AnalysisBatch,
    SAEAnalysisTargets,
    LatentMetrics,
    ActivationSumm,
    PredSumm,
    compute_correct,
    base_vs_sae_logit_diffs,
    latent_metrics_scatter,
)
from interpretune.analysis.protocol import (
    SAEFqn,
    AnalysisBatchProtocol,
    AnalysisOpProtocol,
    AnalysisStoreProtocol,
    AnalysisCfgProtocol,
)

# Register the custom formatter
_register_formatter(ITAnalysisFormatter, "interpretune", aliases=["it", "itanalysis"])

__all__ = [
    # Core formatters and registries
    "ANALYSIS_OPS",
    "ITAnalysisFormatter",

    # Core analysis classes
    "AnalysisStore",
    "AnalysisBatch",
    "SAEAnalysisTargets",

    # Metric containers
    "LatentMetrics",
    "ActivationSumm",
    "PredSumm",

    # Protocol definitions
    "SAEFqn",
    "AnalysisBatchProtocol",
    "AnalysisOpProtocol",
    "AnalysisStoreProtocol",
    "AnalysisCfgProtocol",

    # Utility functions
    "compute_correct",
    "base_vs_sae_logit_diffs",
    "latent_metrics_scatter",
]
