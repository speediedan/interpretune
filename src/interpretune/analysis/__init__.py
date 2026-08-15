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

# Trust for hub-resident code is resolved at call time by interpretune.hub.trust — never captured
# into a constant here, so opting in from inside a running session (a notebook cell) works.

# op-collection hub cache now defined by the unified hub layer (interpretune.hub.cache); the env-var
# override and on-disk default are unchanged
from interpretune.hub.cache import IT_ANALYSIS_HUB_CACHE

# Environment variable for additional op definition paths
IT_ANALYSIS_OP_PATHS = os.getenv("IT_ANALYSIS_OP_PATHS", "").split(":") if os.getenv("IT_ANALYSIS_OP_PATHS") else []

from interpretune.analysis.ops import ColCfg, OpSchema, AnalysisBatch, DISPATCHER, AnalysisOp, OpWrapper, AnalysisOpLike
from interpretune.analysis.execution import AnalysisInputs, execute_analysis_op, execute_analysis_step
from interpretune.hub.manager import HubAnalysisOpManager
from interpretune.analysis.backends import (
    AnalysisBackend,
    AnalysisBackendCapability,
    BackendCapability,
    ModuleCapabilities,
)
from interpretune.analysis.core import (
    AnalysisStore,
    analysis_store_from_batches,
    LatentAnalysisTargets,
    LatentAnalysisDict,
    LatentMetrics,
    ActivationSumm,
    PredSumm,
    compute_correct,
    base_vs_sae_logit_diffs,
    schema_to_features,
    latent_metrics_scatter,
    _make_simple_cache_hook,
    resolve_names_filter,
)
from interpretune.analysis.formatters import ITAnalysisFormatter
from interpretune.protocol import (
    LatentModelFqn,
    DefaultAnalysisBatchProtocol,
    BaseAnalysisBatchProtocol,
    AnalysisOpProtocol,
    AnalysisStoreProtocol,
    AnalysisCfgProtocol,
)

# Register the custom formatter
_register_formatter(ITAnalysisFormatter, "interpretune", aliases=["it", "itanalysis"])

__all__ = [
    # Core Analysis Classes
    "AnalysisStore",
    "analysis_store_from_batches",
    "LatentAnalysisTargets",
    "schema_to_features",
    "_make_simple_cache_hook",
    "resolve_names_filter",
    "LatentAnalysisDict",
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
    "OpWrapper",
    "AnalysisOpLike",
    "ColCfg",
    "OpSchema",
    "AnalysisBatch",
    "AnalysisInputs",
    "IT_ANALYSIS_CACHE",
    "HubAnalysisOpManager",
    "execute_analysis_op",
    "execute_analysis_step",
    # Hub Configuration
    "IT_ANALYSIS_HUB_CACHE",
    "IT_ANALYSIS_OP_PATHS",
    # Formatters
    "ITAnalysisFormatter",
    "BackendCapability",
    "AnalysisBackendCapability",
    "AnalysisBackend",
    "ModuleCapabilities",
    # Protocol Definitions
    "LatentModelFqn",
    "DefaultAnalysisBatchProtocol",
    "BaseAnalysisBatchProtocol",
    "AnalysisOpProtocol",
    "AnalysisStoreProtocol",
    "AnalysisCfgProtocol",
]
