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
# In PyTorch 2.0+, setting this variable will force `torch.cuda.is_available()` and `torch.cuda.device_count()`
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

        # Remove ourselves temporarily from sys.meta_path to avoid infinite recursion
        sys.meta_path.remove(self)
        try:
            # First, import just the dispatcher to avoid circular imports
            import importlib
            dispatcher_module = importlib.import_module("interpretune.analysis.ops.dispatcher")

            # Load operation definitions but don't instantiate yet
            dispatcher_module.DISPATCHER.load_definitions()

            # Then import the rest of the analysis module
            import interpretune.analysis

            # Now register all operations directly to the module
            current_module = sys.modules["interpretune"]

            # Register utility functions
            setattr(current_module, "create_op_chain", interpretune.analysis.DISPATCHER.create_chain)
            setattr(current_module, "create_op_chain_from_ops", interpretune.analysis.DISPATCHER.create_chain_from_ops)

            # Import OpWrapper class from base.py
            from interpretune.analysis.ops.base import OpWrapper

            # Initialize OpWrapper with the current module
            OpWrapper.initialize(current_module)

            # Set debugger identifier class variable directly
            OpWrapper._debugger_identifier = os.environ.get('IT_ENABLE_LAZY_DEBUGGER', '')

            # Register all operations with lazy getters
            for op_name in interpretune.analysis.DISPATCHER._op_definitions:
                # Use lazy=True to avoid instantiation until actual use
                interpretune.analysis.DISPATCHER.get_op(op_name, lazy=True)
                # Create a wrapper that will instantiate the op when accessed
                setattr(current_module, op_name, OpWrapper(op_name))

            # Register all aliases as well
            for op_alias, op_name in interpretune.analysis.DISPATCHER.get_op_aliases():
                if not hasattr(current_module, op_alias):
                    # Create the wrapper for aliases as well
                    setattr(current_module, op_alias, OpWrapper(op_name))

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
