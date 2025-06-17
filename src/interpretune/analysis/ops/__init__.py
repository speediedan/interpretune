"""Operation definitions and infrastructure for analysis operations."""
from interpretune.analysis.ops.base import OpSchema, ColCfg, AnalysisBatch, AnalysisOp, CompositeAnalysisOp
from interpretune.analysis.ops.dispatcher import DISPATCHER


__all__ = [
    "OpSchema",
    "ColCfg",
    "AnalysisBatch",
    "AnalysisOp",
    "CompositeAnalysisOp",
    "DISPATCHER",
    ]
