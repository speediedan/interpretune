"""Operation definitions and infrastructure for analysis operations."""
from interpretune.analysis.ops.base import OpSchema, ColCfg, AnalysisBatch, AnalysisOp, ChainedAnalysisOp
from interpretune.analysis.ops.dispatcher import DISPATCHER

# Expose the dispatcher as an object that can be called directly
dispatcher = DISPATCHER

# Ensure all definitions are loaded immediately and instantiate all operations
DISPATCHER.load_definitions()
# Ensure all operations (including composite ones) are available immediately
DISPATCHER.instantiate_all_ops()

__all__ = [
    "OpSchema",
    "ColCfg",
    "AnalysisBatch",
    "AnalysisOp",
    "ChainedAnalysisOp",
    "DISPATCHER",
    "dispatcher",
]
