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

from interpretune.__about__ import __version__ as version  # noqa: E402
from interpretune.protocol import (
    ITModuleProtocol,
    ITDataModuleProtocol,
    Adapter,
    STEP_OUTPUT,
    CorePhases,
    CoreSteps,
    AllPhases,
    AllSteps,
    AnalysisStoreProtocol,
    DefaultAnalysisBatchProtocol,
    BaseAnalysisBatchProtocol,
    AnalysisOpProtocol,
)


class _AnalysisImportHook(MetaPathFinder):
    """MetaPathFinder that exposes analysis ops in the top-level interpretune namespace when analysis module is
    imported."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "interpretune.analysis":
            return ModuleSpec(fullname, self, is_package=True)  # type: ignore[arg-type]
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

_LAZY_MODULE_ATTRS = {
    # adapters
    "ITModule": "interpretune.adapters.ITModule",
    "LightningDataModule": "interpretune.adapters.LightningDataModule",
    "LightningModule": "interpretune.adapters.LightningModule",
    "ITLensModule": "interpretune.adapters.ITLensModule",
    "SAELensModule": "interpretune.adapters.SAELensModule",
    "ADAPTER_REGISTRY": "interpretune.adapters.ADAPTER_REGISTRY",
    "CompositionRegistry": "interpretune.adapters.CompositionRegistry",
    # analysis
    "AnalysisStore": "interpretune.analysis.AnalysisStore",
    "AnalysisBatch": "interpretune.analysis.AnalysisBatch",
    "DISPATCHER": "interpretune.analysis.DISPATCHER",
    "SAEAnalysisTargets": "interpretune.analysis.SAEAnalysisTargets",
    # config
    "ITConfig": "interpretune.config.ITConfig",
    "ITDataModuleConfig": "interpretune.config.ITDataModuleConfig",
    "AnalysisCfg": "interpretune.config.AnalysisCfg",
    "AnalysisRunnerCfg": "interpretune.config.AnalysisRunnerCfg",
    "AnalysisArtifactCfg": "interpretune.config.AnalysisArtifactCfg",
    "ITLensConfig": "interpretune.config.ITLensConfig",
    "SAELensConfig": "interpretune.config.SAELensConfig",
    "CircuitTracerConfig": "interpretune.config.CircuitTracerConfig",
    "CircuitTracerITLensConfig": "interpretune.config.CircuitTracerITLensConfig",
    "ITSharedConfig": "interpretune.config.ITSharedConfig",
    "PromptConfig": "interpretune.config.PromptConfig",
    "AutoCompConfig": "interpretune.config.AutoCompConfig",
    "HFFromPretrainedConfig": "interpretune.config.HFFromPretrainedConfig",
    "ITLensFromPretrainedNoProcessingConfig": "interpretune.config.ITLensFromPretrainedNoProcessingConfig",
    "TLensGenerationConfig": "interpretune.config.TLensGenerationConfig",
    "GenerativeClassificationConfig": "interpretune.config.GenerativeClassificationConfig",
    "SAELensFromPretrainedConfig": "interpretune.config.SAELensFromPretrainedConfig",
    "ITSerializableCfg": "interpretune.config.ITSerializableCfg",
    "BaseGenerationConfig": "interpretune.config.BaseGenerationConfig",
    "HFGenerationConfig": "interpretune.config.HFGenerationConfig",
    "CoreGenerationConfig": "interpretune.config.CoreGenerationConfig",
    # extensions
    "MemProfiler": "interpretune.extensions.MemProfiler",
    "MemProfilerCfg": "interpretune.extensions.MemProfilerCfg",
    "DebugGeneration": "interpretune.extensions.DebugGeneration",
    "DebugLMConfig": "interpretune.extensions.DebugLMConfig",
    "NeuronpediaIntegration": "interpretune.extensions.NeuronpediaIntegration",
    "NeuronpediaConfig": "interpretune.extensions.NeuronpediaConfig",
    # utils
    "MisconfigurationException": "interpretune.utils.MisconfigurationException",
    "rank_zero_info": "interpretune.utils.rank_zero_info",
    "rank_zero_warn": "interpretune.utils.rank_zero_warn",
    "to_device": "interpretune.utils.to_device",
    "move_data_to_device": "interpretune.utils.move_data_to_device",
    "sanitize_input_name": "interpretune.utils.sanitize_input_name",
    # session / runners / base / registry
    "ITSession": "interpretune.session.ITSession",
    "ITSessionConfig": "interpretune.session.ITSessionConfig",
    "SessionRunner": "interpretune.runners.SessionRunner",
    "AnalysisRunner": "interpretune.runners.AnalysisRunner",
    "ITDataModule": "interpretune.base.ITDataModule",
    "MemProfilerHooks": "interpretune.base.MemProfilerHooks",
    "ITCLI": "interpretune.base.ITCLI",
    "it_init": "interpretune.base.it_init",
    "IT_BASE": "interpretune.base.IT_BASE",
    "it_session_end": "interpretune.base.it_session_end",
    "ModuleRegistry": "interpretune.registry.ModuleRegistry",
    "RegisteredCfg": "interpretune.registry.RegisteredCfg",
    "RegKeyType": "interpretune.registry.RegKeyType",
    "it_cfg_factory": "interpretune.registry.it_cfg_factory",
    "gen_module_registry": "interpretune.registry.gen_module_registry",
    "instantiate_and_register": "interpretune.registry.instantiate_and_register",
    "apply_defaults": "interpretune.registry.apply_defaults",
}


def __getattr__(name: str):
    """Lazily import heavy submodules/objects on attribute access to keep `import interpretune` fast.

    This uses PEP 562 (__getattr__ on packages).
    """
    if name in _LAZY_MODULE_ATTRS:
        module_path = _LAZY_MODULE_ATTRS[name]
        module_name, attr = module_path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[attr])
        val = getattr(module, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_MODULE_ATTRS.keys()))


__all__ = [
    # About Module
    "version",
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
    "DefaultAnalysisBatchProtocol",
    "BaseAnalysisBatchProtocol",
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
    "CircuitTracerConfig",
    "CircuitTracerITLensConfig",
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
    "NeuronpediaIntegration",
    "NeuronpediaConfig",
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
    # Registry Components
    "ModuleRegistry",
    "RegisteredCfg",
    "RegKeyType",
    "it_cfg_factory",
    "gen_module_registry",
    "instantiate_and_register",
    "apply_defaults",
]


# Static typing helpers: import names during type-checking so tools/IDEs can resolve symbols.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import for type-checkers only
    from interpretune.adapters import (
        ITModule,
        LightningDataModule,
        LightningModule,
        ITLensModule,
        SAELensModule,
        ADAPTER_REGISTRY,
        CompositionRegistry,
    )
    from interpretune.analysis import AnalysisStore, AnalysisBatch, DISPATCHER, SAEAnalysisTargets
    from interpretune.config import (
        ITConfig,
        ITDataModuleConfig,
        AnalysisCfg,
        AnalysisRunnerCfg,
        AnalysisArtifactCfg,
        ITLensConfig,
        SAELensConfig,
        CircuitTracerConfig,
        CircuitTracerITLensConfig,
        ITSharedConfig,
        PromptConfig,
        AutoCompConfig,
        HFFromPretrainedConfig,
        ITLensFromPretrainedNoProcessingConfig,
        TLensGenerationConfig,
        GenerativeClassificationConfig,
        SAELensFromPretrainedConfig,
        ITSerializableCfg,
        BaseGenerationConfig,
        HFGenerationConfig,
        CoreGenerationConfig,
    )
    from interpretune.extensions import (
        MemProfiler,
        MemProfilerCfg,
        DebugGeneration,
        DebugLMConfig,
        NeuronpediaIntegration,
        NeuronpediaConfig,
    )
    from interpretune.utils import (
        MisconfigurationException,
        rank_zero_info,
        rank_zero_warn,
        to_device,
        move_data_to_device,
        sanitize_input_name,
    )
    from interpretune.session import ITSession, ITSessionConfig
    from interpretune.runners import SessionRunner, AnalysisRunner
    from interpretune.base import ITDataModule, MemProfilerHooks, ITCLI, it_init, IT_BASE, it_session_end
    from interpretune.registry import (
        ModuleRegistry,
        RegisteredCfg,
        RegKeyType,
        it_cfg_factory,
        gen_module_registry,
        instantiate_and_register,
        apply_defaults,
    )
