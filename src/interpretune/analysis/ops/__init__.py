"""Operation definitions and infrastructure for analysis operations."""
from interpretune.analysis.ops.base import OpSchema, ColCfg, AnalysisBatch, AnalysisOp, ChainedAnalysisOp
from interpretune.analysis.ops.dispatcher import DISPATCHER

# Expose the dispatcher as an object that can be called directly
dispatcher = DISPATCHER

# Ensure all definitions are loaded immediately
DISPATCHER.load_definitions()

__all__ = [
    "OpSchema",
    "ColCfg",
    "AnalysisBatch",
    "AnalysisOp",
    "ChainedAnalysisOp",
    "DISPATCHER",
    "dispatcher",
]
