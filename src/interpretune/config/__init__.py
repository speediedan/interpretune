from interpretune.config.shared import (
    ITSerializableCfg,
    ITSharedConfig,
    AutoCompConf,
    AutoCompConfig,
    search_candidate_subclass_attrs,
)
from interpretune.config.datamodule import ITDataModuleConfig, PromptConfig
from interpretune.config.extensions import ExtensionConf, ITExtension, ITExtensionsConfigMixin
from interpretune.config.mixins import (
    GenerativeClassificationConfig,
    HFFromPretrainedConfig,
    HFGenerationConfig,
    BaseGenerationConfig,
    CoreGenerationConfig,
)
from interpretune.config.module import ITConfig, ITState
from interpretune.config.transformer_lens import (
    ITLensBridgeConfig,
    ITLensConfig,
    ITLensCustomConfig,
    ITLensFromPretrainedConfig,
    ITLensFromPretrainedNoProcessingConfig,
    TLensGenerationConfig,
    TLConfigInitMixin,
    ITLensCfgTypes,
    ITLensCfg,
)
from interpretune.config.sae_lens import (
    SAEConfig,
    SAECfgType,
    SAELensFromPretrainedConfig,
    SAELensCustomConfig,
    SAELensConfig,
)

# Conditionally import circuit_tracer configs
try:
    from interpretune.config.circuit_tracer import CircuitTracerConfig

    _circuit_tracer_config_available = True
except ImportError:
    # circuit_tracer not available, define placeholder classes
    CircuitTracerConfig = None
    _circuit_tracer_config_available = False

# Conditionally import nnsight configs
try:
    from interpretune.config.nnsight import NNsightConfig, ITNNsightConfig, NNsightCfg, NNsightCfgTypes

    _nnsight_config_available = True
except ImportError:
    # nnsight not available, define placeholder classes
    NNsightConfig = None
    ITNNsightConfig = None
    _nnsight_config_available = False

from interpretune.config.analysis import AnalysisCfg, AnalysisArtifactCfg
from interpretune.config.runner import SessionRunnerCfg, AnalysisRunnerCfg, init_analysis_dirs, init_analysis_cfgs


__all__ = [
    # from interpretune.config.shared
    "ITSerializableCfg",
    "ITSharedConfig",
    "AutoCompConf",
    "AutoCompConfig",
    "search_candidate_subclass_attrs",
    # from interpretune.config.datamodule
    "ITDataModuleConfig",
    "PromptConfig",
    # from interpretune.config.extensions
    "ExtensionConf",
    "ITExtension",
    "ITExtensionsConfigMixin",
    # from interpretune.config.mixins
    "GenerativeClassificationConfig",
    "HFFromPretrainedConfig",
    "HFGenerationConfig",
    "BaseGenerationConfig",
    "CoreGenerationConfig",
    # from interpretune.config.module
    "ITConfig",
    "ITState",
    # from interpretune.config.transformer_lens
    "ITLensBridgeConfig",
    "ITLensConfig",
    "ITLensCustomConfig",
    "ITLensFromPretrainedConfig",
    "ITLensFromPretrainedNoProcessingConfig",
    "TLensGenerationConfig",
    "TLConfigInitMixin",
    "ITLensCfg",
    "ITLensCfgTypes",
    # from interpretune.config.sae_lens
    "SAEConfig",
    "SAECfgType",
    "SAELensFromPretrainedConfig",
    "SAELensCustomConfig",
    "SAELensConfig",
    # from interpretune.config.nnsight
    "NNsightConfig",
    "NNsightCfg",
    "NNsightCfgTypes",
    "ITNNsightConfig",
    # from interpretune.config.analysis
    "AnalysisCfg",
    "AnalysisArtifactCfg",
    # from interpretune.config.runner
    "SessionRunnerCfg",
    "AnalysisRunnerCfg",
    "init_analysis_dirs",
    "init_analysis_cfgs",
]

# Add circuit_tracer configs only if available
if _circuit_tracer_config_available:
    __all__.extend(
        [
            "CircuitTracerConfig",
        ]
    )

# Add nnsight configs only if available
if _nnsight_config_available:
    __all__.extend(
        [
            "NNsightConfig",
            "ITNNsightConfig",
            "NNsightCfg",
            "NNsightCfgTypes",
        ]
    )
