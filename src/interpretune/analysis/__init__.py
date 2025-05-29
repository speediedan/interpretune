"""Analysis submodule."""
import os
from pathlib import Path
from datasets.formatting import _register_formatter
from datasets.config import HF_CACHE_HOME  # we expect to leverage the Hugging Face cache system for analysis artifacts
# we ignore these for the entire file so that we can set the cache directory before importing analysis components
# ruff: noqa: E402
IT_ANALYSIS_CACHE_DIR_NAME = "interpretune"
DEFAULT_IT_ANALYSIS_CACHE = os.path.join(HF_CACHE_HOME, IT_ANALYSIS_CACHE_DIR_NAME)
IT_ANALYSIS_CACHE = Path(os.getenv("IT_ANALYSIS_CACHE_DIR", DEFAULT_IT_ANALYSIS_CACHE))

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
    "IT_ANALYSIS_CACHE",

    # Formatters
    "ITAnalysisFormatter",

    # Protocol Definitions
    "SAEFqn",
    "AnalysisBatchProtocol",
    "AnalysisOpProtocol",
    "AnalysisStoreProtocol",
    "AnalysisCfgProtocol",
]
