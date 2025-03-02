"""
Interpretune
=====================

The interpretune package provides analysis tools for exploring model interpretability.
"""
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

from interpretune.__about__ import *  # noqa: F401, F403
from interpretune.protocol import ITModuleProtocol, ITDataModuleProtocol


# In PyTorch 2.0+, setting this variable will force `torch.cuda.is_available()` and `torch.cuda.device_count()`
# to use an NVML-based implementation that doesn't poison forks.
# https://github.com/pytorch/pytorch/issues/83973
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

class _AnalysisImportHook(MetaPathFinder):
    """MetaPathFinder that exposes analysis ops in the top-level interpretune namespace only when analysis module
    is imported."""
    def find_spec(self, fullname, path, target=None):
        if fullname == "interpretune.analysis":
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Remove ourselves temporarily from sys.meta_path to avoid infinite recursion
        sys.meta_path.remove(self)
        try:
            import interpretune.analysis as analysis
            current_module = sys.modules["interpretune"]
            for op, alias in analysis.ANALYSIS_OPS.iter_aliased_ops():
                setattr(current_module, alias, op.callable)
            return analysis
        finally:
            sys.meta_path.insert(0, self)

# Register our import hook to handle interpretune.analysis imports
sys.meta_path.insert(0, _AnalysisImportHook())

# we need to defer all imports that depend on the analysis module until after the import hook is registered
from interpretune.session import ITSession, ITSessionConfig  # noqa: E402
from interpretune.runners import SessionRunner, AnalysisRunner  # noqa: E402
from interpretune.base import ITDataModule, BaseITModule, ProfilerHooksMixin  # noqa: E402

__all__ = [
    # Session Module
    "ITSession",
    "ITSessionConfig",

    # Runners
    "SessionRunner",
    "AnalysisRunner",

    # Protocol Module
    "ITModuleProtocol",
    "ITDataModuleProtocol",

    # Base Modules
    "ITDataModule",
    "BaseITModule",

    # Base Components
    "ProfilerHooksMixin",
]
