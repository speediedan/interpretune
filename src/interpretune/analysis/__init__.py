"""Analysis submodule."""
from datasets.formatting import _register_formatter

from interpretune.analysis.ops import ANALYSIS_OPS, AnalysisOp, ColCfg
from interpretune.analysis.core import (AnalysisStore, AnalysisBatch, SAEAnalysisTargets, LatentMetrics, ActivationSumm,
                                        PredSumm, compute_correct, base_vs_sae_logit_diffs, schema_to_features,
                                        latent_metrics_scatter, _make_simple_cache_hook, resolve_names_filter)
from interpretune.analysis.formatters import ITAnalysisFormatter
from interpretune.protocol import (SAEFqn, AnalysisBatchProtocol, AnalysisOpProtocol, AnalysisStoreProtocol,
                                   AnalysisCfgProtocol)


# Register the custom formatter
_register_formatter(ITAnalysisFormatter, "interpretune", aliases=["it", "itanalysis"])

__all__ = [
    # Core Analysis Classes
    "AnalysisStore",         # from .core
    "AnalysisBatch",         # from .core
    "SAEAnalysisTargets",    # from .core
    "schema_to_features",    # from .core
    "_make_simple_cache_hook", # from .core
    "resolve_names_filter",  # from .core

    # Metric Containers
    "LatentMetrics",         # from .core
    "ActivationSumm",        # from .core
    "PredSumm",              # from .core

    # Utility Functions
    "compute_correct",       # from .core
    "base_vs_sae_logit_diffs", # from .core
    "latent_metrics_scatter", # from .core

    # Analysis Operations
    "ANALYSIS_OPS",          # from .ops
    "AnalysisOp",            # from .ops
    "ColCfg",                # from .ops

    # Formatters
    "ITAnalysisFormatter",   # from .formatters

    # Protocol Definitions
    "SAEFqn",                # from .protocol
    "AnalysisBatchProtocol", # from .protocol
    "AnalysisOpProtocol",    # from .protocol
    "AnalysisStoreProtocol", # from .protocol
    "AnalysisCfgProtocol",   # from .protocol
]
