"""Bundled extensions, resolved lazily.

This package's names are resolved via PEP 562 ``__getattr__`` rather than eager imports: importing
any single extension submodule executes this ``__init__``, so an eager sibling import here would
make every bundled extension a hard import-time requirement of all the others. Concretely, the
neuronpedia extension pulls in the full analysis stack (sae_lens et al.), which a bare interpretune
install does not carry -- and bundled-extension detection (config/extensions.py) imports
``debug_generation`` on the ``import interpretune.config`` path (#403).
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

_EXPORT_MODULES = {
    "DebugGeneration": "debug_generation",
    "DebugLMConfig": "debug_generation",
    "MemProfiler": "memprofiler",
    "MemProfilerCfg": "memprofiler",
    "MemProfilerHooks": "memprofiler",
    "MemProfilerFuncs": "memprofiler",
    "MemProfilerSchedule": "memprofiler",
    "DefaultMemHooks": "memprofiler",
    "NeuronpediaIntegration": "neuronpedia",
    "NeuronpediaConfig": "neuronpedia",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name in _EXPORT_MODULES:
        import importlib

        module = importlib.import_module(f"interpretune.extensions.{_EXPORT_MODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
