"""The circuit-tracer adapter: its module composition, its config, and (where it has one) its backend.

Everything an adapter owns lives in ONE package, so a bundled adapter is structurally identical to a
hub-delivered one (interpretune#401). Nothing here is privileged over a component that arrives from the
Hub: same layout, same registration path, same discovery seam.

Exports resolve LAZILY. Several submodules import their framework at module level, and this package sits
on paths that must stay importable without it, so a name is resolved only when it is actually used.
"""

from typing import TYPE_CHECKING

#: What this adapter needs before its implementation module can be imported, in the same vocabulary a hub
#: component's manifest uses (``interpretune`` / ``adapters`` / ``pip``). One predicate, two sources: a hub
#: component declares in its manifest, a bundled adapter declares here.
#:
#: THIS MUST STAY EAGER -- a module-level constant, never routed through the lazy ``__getattr__`` map below.
#: The whole point is that registration can read it to decide WHETHER to import ``.adapter``; resolving it
#: lazily would import that submodule to answer the question, which needs the very dependency being tested
#: for. Tidying it into the map would look like consistency and would silently restore the bootstrap
#: problem, so a test asserts this is readable with circuit-tracer absent.
__it_requires__ = {"pip": ["circuit-tracer"]}

if TYPE_CHECKING:
    from interpretune.adapters.circuit_tracer.registry import (
        CT_BACKEND_REGISTRY as CT_BACKEND_REGISTRY,
        CT_MODEL_BACKEND_FACTORIES as CT_MODEL_BACKEND_FACTORIES,
    )
    from interpretune.adapters.circuit_tracer.adapter import (
        BaseCircuitTracerModule as BaseCircuitTracerModule,
        CircuitTracerAdapter as CircuitTracerAdapter,
        CircuitTracerAnalysisMixin as CircuitTracerAnalysisMixin,
        CircuitTracerAttributeMixin as CircuitTracerAttributeMixin,
        CircuitTracerModule as CircuitTracerModule,
        CircuitTracerNNsightModule as CircuitTracerNNsightModule,
        CircuitTracerNNsightModuleMixin as CircuitTracerNNsightModuleMixin,
        CircuitTracerTLModule as CircuitTracerTLModule,
        CircuitTracerTLModuleMixin as CircuitTracerTLModuleMixin,
        InstantiatedGraph as InstantiatedGraph,
        ReplacementModelType as ReplacementModelType,
    )
    from interpretune.adapters.circuit_tracer.backends import (
        CircuitTracerAnalysisBackend as CircuitTracerAnalysisBackend,
        DEFAULT_CT_ANALYSIS_BACKEND as DEFAULT_CT_ANALYSIS_BACKEND,
    )
    from interpretune.adapters.circuit_tracer.config import (
        CircuitTracerConfig as CircuitTracerConfig,
    )

# public name -> the submodule that defines it
_EXPORT_MODULES = {
    "BaseCircuitTracerModule": "adapter",
    "CT_BACKEND_REGISTRY": "registry",
    "CT_MODEL_BACKEND_FACTORIES": "registry",
    "CircuitTracerAdapter": "adapter",
    "CircuitTracerAnalysisBackend": "backends",
    "CircuitTracerAnalysisMixin": "adapter",
    "CircuitTracerAttributeMixin": "adapter",
    "CircuitTracerConfig": "config",
    "CircuitTracerModule": "adapter",
    "CircuitTracerNNsightModule": "adapter",
    "CircuitTracerNNsightModuleMixin": "adapter",
    "CircuitTracerTLModule": "adapter",
    "CircuitTracerTLModuleMixin": "adapter",
    "DEFAULT_CT_ANALYSIS_BACKEND": "backends",
    "InstantiatedGraph": "adapter",
    "ReplacementModelType": "adapter",
}

__all__ = [
    "BaseCircuitTracerModule",
    "CT_BACKEND_REGISTRY",
    "CT_MODEL_BACKEND_FACTORIES",
    "CircuitTracerAdapter",
    "CircuitTracerAnalysisBackend",
    "CircuitTracerAnalysisMixin",
    "CircuitTracerAttributeMixin",
    "CircuitTracerConfig",
    "CircuitTracerModule",
    "CircuitTracerNNsightModule",
    "CircuitTracerNNsightModuleMixin",
    "CircuitTracerTLModule",
    "CircuitTracerTLModuleMixin",
    "DEFAULT_CT_ANALYSIS_BACKEND",
    "InstantiatedGraph",
    "ReplacementModelType",
]


# Submodules searched, in order, for a public name the map above does not carry. The map is generated
# from what each submodule DEFINES; a submodule may also re-export a name it imported (a strategy
# adapter, a replacement model), and those were reachable on the pre-package module. Preserving that is
# what makes this motion a motion.
_FALLBACK_SUBMODULES = ("adapter", "backends", "config")


def __getattr__(name: str):
    import importlib

    sub = _EXPORT_MODULES.get(name)
    if sub is not None:
        return getattr(importlib.import_module(f"{__name__}.{sub}"), name)
    if not name.startswith("_"):
        for candidate in _FALLBACK_SUBMODULES:
            try:
                module = importlib.import_module(f"{__name__}.{candidate}")
            except ImportError:
                continue
            if hasattr(module, name):
                return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Report the lazy exports too, so `dir()` and introspective discovery see the real surface."""
    return sorted(set(globals()) | set(_EXPORT_MODULES))
