"""The SAELens adapter: its module composition, its config, and (where it has one) its backend.

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
#: Registration reads it to decide WHETHER to import the implementation submodule; resolving it lazily would
#: import that submodule to answer the question, which needs the very dependency being tested for.
__it_requires__ = {"pip": ["sae-lens"]}

if TYPE_CHECKING:
    from interpretune.adapters.sae_lens.adapter import (
        BaseSAELensModule as BaseSAELensModule,
        InstantiatedSAE as InstantiatedSAE,
        SAELensAdapter as SAELensAdapter,
        SAELensAnalysisMixin as SAELensAnalysisMixin,
        SAELensAttributeMixin as SAELensAttributeMixin,
        SAELensNNsightModule as SAELensNNsightModule,
        SAELensNNsightModuleMixin as SAELensNNsightModuleMixin,
        SAELensTLModule as SAELensTLModule,
        SAELensTLModuleMixin as SAELensTLModuleMixin,
    )
    from interpretune.adapters.sae_lens.config import (
        SAECfgType as SAECfgType,
        SAELensConfig as SAELensConfig,
        SAELensCustomConfig as SAELensCustomConfig,
        SAELensFromPretrainedConfig as SAELensFromPretrainedConfig,
    )

# public name -> the submodule that defines it
_EXPORT_MODULES = {
    "BaseSAELensModule": "adapter",
    "InstantiatedSAE": "adapter",
    "SAECfgType": "config",
    "SAELensAdapter": "adapter",
    "SAELensAnalysisMixin": "adapter",
    "SAELensAttributeMixin": "adapter",
    "SAELensConfig": "config",
    "SAELensCustomConfig": "config",
    "SAELensFromPretrainedConfig": "config",
    "SAELensNNsightModule": "adapter",
    "SAELensNNsightModuleMixin": "adapter",
    "SAELensTLModule": "adapter",
    "SAELensTLModuleMixin": "adapter",
}

__all__ = [
    "BaseSAELensModule",
    "InstantiatedSAE",
    "SAECfgType",
    "SAELensAdapter",
    "SAELensAnalysisMixin",
    "SAELensAttributeMixin",
    "SAELensConfig",
    "SAELensCustomConfig",
    "SAELensFromPretrainedConfig",
    "SAELensNNsightModule",
    "SAELensNNsightModuleMixin",
    "SAELensTLModule",
    "SAELensTLModuleMixin",
]


# Submodules searched, in order, for a public name the map above does not carry. The map is generated
# from what each submodule DEFINES; a submodule may also re-export a name it imported (a strategy
# adapter, a replacement model), and those were reachable on the pre-package module. Preserving that is
# what makes this motion a motion.
_FALLBACK_SUBMODULES = ("adapter", "config")


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
