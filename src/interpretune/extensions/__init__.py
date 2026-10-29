"""Bundled extensions, resolved lazily.

This package's names are resolved via PEP 562 ``__getattr__`` rather than eager imports: importing
any single extension submodule executes this ``__init__``, so an eager sibling import here would
make every bundled extension a hard import-time requirement of all the others. Concretely, the
neuronpedia extension pulls in the full analysis stack (sae_lens et al.), which a bare interpretune
install does not carry. The configuration classes live in ``interpretune.config.extension_cfgs``, which
``import interpretune.config`` reads without importing any extension (#624).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Redundant aliases mark these as re-exports (ruff F401) without a runtime import.
    from interpretune.extensions.debug_generation import (
        DebugGeneration as DebugGeneration,
        DebugLMConfig as DebugLMConfig,
    )
    from interpretune.extensions.memprofiler import (
        MemProfiler as MemProfiler,
        MemProfilerCfg as MemProfilerCfg,
        MemProfilerHooks as MemProfilerHooks,
        MemProfilerFuncs as MemProfilerFuncs,
        MemProfilerSchedule as MemProfilerSchedule,
        DefaultMemHooks as DefaultMemHooks,
    )
    from interpretune.extensions.neuronpedia import (
        NeuronpediaIntegration as NeuronpediaIntegration,
        NeuronpediaConfig as NeuronpediaConfig,
    )

_CFGS = "interpretune.config.extension_cfgs"
_EXPORT_MODULES = {
    "DebugGeneration": "interpretune.extensions.debug_generation",
    "DebugLMConfig": _CFGS,
    "MemProfiler": "interpretune.extensions.memprofiler",
    "MemProfilerCfg": _CFGS,
    "MemProfilerHooks": _CFGS,
    "MemProfilerFuncs": _CFGS,
    "MemProfilerSchedule": _CFGS,
    "DefaultMemHooks": _CFGS,
    "NeuronpediaIntegration": "interpretune.extensions.neuronpedia",
    "NeuronpediaConfig": _CFGS,
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name in _EXPORT_MODULES:
        import importlib

        module = importlib.import_module(_EXPORT_MODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
