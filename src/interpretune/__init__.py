"""
Interpretune
=====================

The interpretune package provides analysis tools for exploring model interpretability.
"""
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

# we ignore these for the entire file due to our import hook dependency
# ruff: noqa: E402
# Setting this variable will force `torch.cuda.is_available()` and `torch.cuda.device_count()`
# to use an NVML-based implementation that doesn't poison forks.
# https://github.com/pytorch/pytorch/issues/83973
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

from interpretune.__about__ import *  # noqa: F401, F403
from interpretune.protocol import (ITModuleProtocol, ITDataModuleProtocol, Adapter, STEP_OUTPUT,
                                   CorePhases, CoreSteps, AllPhases, AllSteps, AnalysisStoreProtocol,
                                   AnalysisBatchProtocol, AnalysisOpProtocol)


class _AnalysisImportHook(MetaPathFinder):
    """MetaPathFinder that exposes analysis ops in the top-level interpretune namespace when analysis module is
    imported."""
    def find_spec(self, fullname, path, target=None):
        if fullname == "interpretune.analysis":
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        sys.meta_path.remove(self)  # Remove ourselves temporarily from sys.meta_path to avoid infinite recursion
        try:
            import interpretune.analysis
            from interpretune.analysis.ops.base import OpWrapper
            # Register available op definitions with OpWrappers (definitions only, implementations lazily instantiated)
            OpWrapper.register_operations(sys.modules["interpretune"], interpretune.analysis.DISPATCHER)
            return sys.modules["interpretune.analysis"]
        finally:
            sys.meta_path.insert(0, self)

# Register our import hook to handle interpretune.analysis imports
sys.meta_path.insert(0, _AnalysisImportHook())

# allow import of core objects from all second-level IT modules via interpretune.x import y
from interpretune.adapters import (ITModule, LightningDataModule, LightningModule, ITLensModule, SAELensModule,
                                   ADAPTER_REGISTRY, CompositionRegistry)
from interpretune.analysis import AnalysisStore, AnalysisBatch, DISPATCHER, SAEAnalysisTargets
from interpretune.config import (ITConfig, ITDataModuleConfig, AnalysisCfg, AnalysisRunnerCfg,
                                 AnalysisArtifactCfg, ITLensConfig, SAELensConfig, ITSharedConfig, PromptConfig,
                                 AutoCompConfig, HFFromPretrainedConfig, ITLensFromPretrainedNoProcessingConfig,
                                 TLensGenerationConfig, GenerativeClassificationConfig, SAELensFromPretrainedConfig,
                                 ITSerializableCfg, BaseGenerationConfig, HFGenerationConfig, CoreGenerationConfig)
from interpretune.extensions import MemProfiler, MemProfilerCfg, DebugGeneration, DebugLMConfig
from interpretune.utils import (MisconfigurationException, rank_zero_info, rank_zero_warn, to_device,
                                move_data_to_device, sanitize_input_name)

# we need to defer all imports that depend on the analysis module until after the import hook is registered
from interpretune.session import ITSession, ITSessionConfig
from interpretune.runners import SessionRunner, AnalysisRunner
from interpretune.base import ITDataModule, MemProfilerHooks, ITCLI, it_init, IT_BASE, it_session_end


__all__ = [
    # Protocol Module
    "ITModuleProtocol",
    "ITDataModuleProtocol",
    "Adapter",
    "STEP_OUTPUT",
    "CorePhases",
    "CoreSteps",
    "AllPhases",
    "AllSteps",
    "AnalysisStoreProtocol",
    "AnalysisBatchProtocol",
    "AnalysisOpProtocol",

    # Adapters Module
    "ITModule",
    "LightningDataModule",
    "LightningModule",
    "ITLensModule",
    "SAELensModule",
    "ADAPTER_REGISTRY",
    "CompositionRegistry",

    # Analysis Module
    "AnalysisStore",
    "AnalysisBatch",
    "DISPATCHER",
    "SAEAnalysisTargets",

    # Config Module
    "ITConfig",
    "ITDataModuleConfig",
    "AnalysisCfg",
    "AnalysisRunnerCfg",
    "AnalysisArtifactCfg",
    "ITLensConfig",
    "SAELensConfig",
    "ITSharedConfig",
    "PromptConfig",
    "AutoCompConfig",
    "HFFromPretrainedConfig",
    "ITLensFromPretrainedNoProcessingConfig",
    "TLensGenerationConfig",
    "GenerativeClassificationConfig",
    "SAELensFromPretrainedConfig",
    "ITSerializableCfg",
    "HFGenerationConfig",
    "BaseGenerationConfig",
    "CoreGenerationConfig",

    # Extensions Module
    "MemProfiler",
    "MemProfilerCfg",
    "DebugGeneration",
    "DebugLMConfig",

    # Utils Module
    "MisconfigurationException",
    "rank_zero_info",
    "rank_zero_warn",
    "to_device",
    "move_data_to_device",
    "sanitize_input_name",

    # Session Module
    "ITSession",
    "ITSessionConfig",

    # Runners Module
    "SessionRunner",
    "AnalysisRunner",

    # Base Module
    "ITDataModule",
    "MemProfilerHooks",
    "ITCLI",
    "it_init",
    "IT_BASE",
    "it_session_end",
]
