from typing import TYPE_CHECKING

from interpretune.config.shared import (
    ITSerializableCfg,
    ITSharedConfig,
    AutoCompConf,
    AutoCompConfig,
    search_candidate_subclass_attrs,
)
from interpretune.config.datamodule import ChatTemplatePromptConfig, ITDataModuleConfig, PromptConfig
from interpretune.config.extensions import ExtensionConf, ITExtension, ITExtensionsConfigMixin
from interpretune.config.mixins import (
    GenerativeClassificationConfig,
    HFFromPretrainedConfig,
    HFGenerationConfig,
    BaseGenerationConfig,
    CoreGenerationConfig,
)
from interpretune.config.module import ITConfig, ITState

from interpretune.config.analysis import AnalysisCfg, AnalysisArtifactCfg
from interpretune.config.runner import SessionRunnerCfg, AnalysisRunnerCfg, init_analysis_dirs, init_analysis_cfgs

# ADAPTER CONFIGS RESOLVE LAZILY, and this is the point of the per-adapter package layout (#401).
# Each adapter's config now lives with its adapter (`interpretune.adapters.<name>.config`), so a
# BUNDLED adapter's config has exactly the standing a hub-delivered one does. These names stay
# importable from here because they always have been, but importing them is no longer a condition of
# importing `interpretune.config`: `transformer_lens` and `sae_lens` were previously eager, which made
# two optional-extra frameworks de facto hard requirements of the core config package.
_ADAPTER_CONFIG_EXPORTS = {
    "ITLensBridgeConfig": "transformer_lens",
    "ITLensCfg": "transformer_lens",
    "ITLensCfgTypes": "transformer_lens",
    "ITLensConfig": "transformer_lens",
    "ITLensCustomConfig": "transformer_lens",
    "ITLensFromPretrainedConfig": "transformer_lens",
    "ITLensFromPretrainedNoProcessingConfig": "transformer_lens",
    "TLConfigInitMixin": "transformer_lens",
    "TLensGenerationConfig": "transformer_lens",
    "SAECfgType": "sae_lens",
    "SAEConfig": "sae_lens",
    "SAELensConfig": "sae_lens",
    "SAELensCustomConfig": "sae_lens",
    "SAELensFromPretrainedConfig": "sae_lens",
    "ITNNsightConfig": "nnsight",
    "NNsightCfg": "nnsight",
    "NNsightCfgTypes": "nnsight",
    "NNsightConfig": "nnsight",
    "CircuitTracerConfig": "circuit_tracer",
}

if TYPE_CHECKING:
    from interpretune.adapters.circuit_tracer.config import CircuitTracerConfig as CircuitTracerConfig
    from interpretune.adapters.nnsight.config import (
        ITNNsightConfig as ITNNsightConfig,
        NNsightCfg as NNsightCfg,
        NNsightCfgTypes as NNsightCfgTypes,
        NNsightConfig as NNsightConfig,
    )
    from interpretune.adapters.sae_lens.config import (
        SAECfgType as SAECfgType,
        SAEConfig as SAEConfig,
        SAELensConfig as SAELensConfig,
        SAELensCustomConfig as SAELensCustomConfig,
        SAELensFromPretrainedConfig as SAELensFromPretrainedConfig,
    )
    from interpretune.adapters.transformer_lens.config import (
        ITLensBridgeConfig as ITLensBridgeConfig,
        ITLensCfg as ITLensCfg,
        ITLensCfgTypes as ITLensCfgTypes,
        ITLensConfig as ITLensConfig,
        ITLensCustomConfig as ITLensCustomConfig,
        ITLensFromPretrainedConfig as ITLensFromPretrainedConfig,
        ITLensFromPretrainedNoProcessingConfig as ITLensFromPretrainedNoProcessingConfig,
        TLConfigInitMixin as TLConfigInitMixin,
        TLensGenerationConfig as TLensGenerationConfig,
    )


def __getattr__(name: str):
    """Resolve an adapter's config class from its adapter package on first use."""
    adapter = _ADAPTER_CONFIG_EXPORTS.get(name)
    if adapter is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(f"interpretune.adapters.{adapter}.config")
    except ImportError as e:
        raise AttributeError(
            f"{name!r} is provided by the {adapter!r} adapter, whose package is not importable here: {e}. "
            f"Install the relevant extra to use it."
        ) from e
    return getattr(module, name)


__all__ = [
    # from interpretune.config.shared
    "ITSerializableCfg",
    "ITSharedConfig",
    "AutoCompConf",
    "AutoCompConfig",
    "search_candidate_subclass_attrs",
    # from interpretune.config.datamodule
    "ITDataModuleConfig",
    "ChatTemplatePromptConfig",
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
    # from interpretune.adapters.transformer_lens.config
    "ITLensBridgeConfig",
    "ITLensConfig",
    "ITLensCustomConfig",
    "ITLensFromPretrainedConfig",
    "ITLensFromPretrainedNoProcessingConfig",
    "TLensGenerationConfig",
    "TLConfigInitMixin",
    "ITLensCfg",
    "ITLensCfgTypes",
    # from interpretune.adapters.sae_lens.config
    "SAEConfig",
    "SAECfgType",
    "SAELensFromPretrainedConfig",
    "SAELensCustomConfig",
    "SAELensConfig",
    # from interpretune.adapters.nnsight.config
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

# CircuitTracerConfig is exported on the same lazy footing as every other adapter config. It was
# previously conditional on an import probe, which made its presence in `__all__` depend on what
# happened to be installed -- a different contract from the rest of this module for no reason a caller
# could see.
__all__.append("CircuitTracerConfig")
