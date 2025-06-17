"""Analysis submodule."""
import os
from pathlib import Path
from datasets.formatting import _register_formatter
from datasets.config import HF_CACHE_HOME  # we expect to leverage the Hugging Face cache system for analysis artifacts
# we ignore these for the entire file so that we can set the cache directory before importing analysis components
# ruff: noqa: E402
IT_ANALYSIS_CACHE_DIR_NAME = "interpretune"
DEFAULT_IT_ANALYSIS_CACHE = os.path.join(HF_CACHE_HOME, IT_ANALYSIS_CACHE_DIR_NAME)
IT_ANALYSIS_CACHE = Path(os.getenv("IT_ANALYSIS_CACHE", DEFAULT_IT_ANALYSIS_CACHE))

IT_MODULES_CACHE = os.getenv("IT_MODULES_CACHE", os.path.join(IT_ANALYSIS_CACHE, "modules"))
IT_DYNAMIC_MODULE_NAME = "interpretune_modules"

# Hub cache configuration
IT_TRUST_REMOTE_CODE_ENV = os.getenv("IT_TRUST_REMOTE_CODE", "true")  # TODO: make this None by default before release
IT_TRUST_REMOTE_CODE = IT_TRUST_REMOTE_CODE_ENV.lower() in ("true", "1", "yes") \
    if IT_TRUST_REMOTE_CODE_ENV is not None else None

IT_ANALYSIS_HUB_CACHE_DIR_NAME = "interpretune_ops"
DEFAULT_IT_ANALYSIS_HUB_CACHE = Path(os.path.join(HF_CACHE_HOME, "hub")) / IT_ANALYSIS_HUB_CACHE_DIR_NAME
IT_ANALYSIS_HUB_CACHE = Path(os.getenv("IT_ANALYSIS_HUB_CACHE", DEFAULT_IT_ANALYSIS_HUB_CACHE))

# Environment variable for additional op definition paths
IT_ANALYSIS_OP_PATHS = os.getenv("IT_ANALYSIS_OP_PATHS", "").split(":") if os.getenv("IT_ANALYSIS_OP_PATHS") else []

from interpretune.analysis.ops import ColCfg, OpSchema, AnalysisBatch, DISPATCHER, AnalysisOp
from interpretune.analysis.ops.hub_manager import HubAnalysisOpManager
from interpretune.analysis.core import (AnalysisStore, SAEAnalysisTargets, SAEAnalysisDict, LatentMetrics,
                                        ActivationSumm, PredSumm, compute_correct, base_vs_sae_logit_diffs,
                                        schema_to_features, latent_metrics_scatter, _make_simple_cache_hook,
                                        resolve_names_filter)
from interpretune.analysis.formatters import ITAnalysisFormatter
from interpretune.protocol import (SAEFqn, DefaultAnalysisBatchProtocol, BaseAnalysisBatchProtocol, AnalysisOpProtocol,
                                   AnalysisStoreProtocol, AnalysisCfgProtocol)
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
    "HubAnalysisOpManager",

    # Hub Configuration
    "IT_ANALYSIS_HUB_CACHE",
    "IT_ANALYSIS_OP_PATHS",

    # Formatters
    "ITAnalysisFormatter",

    # Protocol Definitions
    "SAEFqn",
    "DefaultAnalysisBatchProtocol",
    "BaseAnalysisBatchProtocol",
    "AnalysisOpProtocol",
    "AnalysisStoreProtocol",
    "AnalysisCfgProtocol",
]
