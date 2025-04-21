"""Analysis submodule."""
from datasets.formatting import _register_formatter


from interpretune.analysis.ops import ColCfg, OpSchema, AnalysisBatch, DISPATCHER, AnalysisOp
from interpretune.analysis.core import (AnalysisStore, SAEAnalysisTargets, SAEAnalysisDict, LatentMetrics,
                                        ActivationSumm, PredSumm, compute_correct, base_vs_sae_logit_diffs,
                                        schema_to_features, latent_metrics_scatter, _make_simple_cache_hook,
                                        resolve_names_filter)
from interpretune.analysis.formatters import ITAnalysisFormatter
from interpretune.protocol import (SAEFqn, AnalysisBatchProtocol, AnalysisOpProtocol, AnalysisStoreProtocol,
                                   AnalysisCfgProtocol)


# Register the custom formatter
_register_formatter(ITAnalysisFormatter, "interpretune", aliases=["it", "itanalysis"])

__all__ = [
    # Core Analysis Classes
    "AnalysisStore",
    "SAEAnalysisTargets",
    "schema_to_features",
    "_make_simple_cache_hook",
    "resolve_names_filter",
    "SAEAnalysisDict",

    # Metric Containers
    "LatentMetrics",
    "ActivationSumm",
    "PredSumm",

    # Utility Functions
    "compute_correct",
    "base_vs_sae_logit_diffs",
    "latent_metrics_scatter",

    # Analysis Operations
    "DISPATCHER",
    "AnalysisOp",
    "ColCfg",
    "OpSchema",
    "AnalysisBatch",

    # Formatters
    "ITAnalysisFormatter",

    # Protocol Definitions
    "SAEFqn",
    "AnalysisBatchProtocol",
    "AnalysisOpProtocol",
    "AnalysisStoreProtocol",
    "AnalysisCfgProtocol",
]
