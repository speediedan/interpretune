"""The NNsight adapter: its module composition, its config, and (where it has one) its backend.

Everything an adapter owns lives in ONE package, so a bundled adapter is structurally identical to a
hub-delivered one (interpretune#401). Nothing here is privileged over a component that arrives from the
Hub: same layout, same registration path, same discovery seam.

Exports resolve LAZILY. Several submodules import their framework at module level, and this package sits
on paths that must stay importable without it, so a name is resolved only when it is actually used.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interpretune.adapters.nnsight.adapter import (
        BaseNNsightModule as BaseNNsightModule,
        NNsightAdapter as NNsightAdapter,
        NNsightAttributeMixin as NNsightAttributeMixin,
        NNsightModule as NNsightModule,
    )
    from interpretune.adapters.nnsight.backends import (
        NNsightActivationCacheAdapter as NNsightActivationCacheAdapter,
        NNsightModelBackend as NNsightModelBackend,
        get_default_configs_per_pass as get_default_configs_per_pass,
    )
    from interpretune.adapters.nnsight.config import (
        ITNNsightConfig as ITNNsightConfig,
        NNsightCfg as NNsightCfg,
        NNsightCfgTypes as NNsightCfgTypes,
        NNsightConfig as NNsightConfig,
    )

# public name -> the submodule that defines it
_EXPORT_MODULES = {
    "BaseNNsightModule": "adapter",
    "ITNNsightConfig": "config",
    "NNsightActivationCacheAdapter": "backends",
    "NNsightAdapter": "adapter",
    "NNsightAttributeMixin": "adapter",
    "NNsightCfg": "config",
    "NNsightCfgTypes": "config",
    "NNsightConfig": "config",
    "NNsightModelBackend": "backends",
    "NNsightModule": "adapter",
    "get_default_configs_per_pass": "backends",
}

__all__ = [
    "BaseNNsightModule",
    "ITNNsightConfig",
    "NNsightActivationCacheAdapter",
    "NNsightAdapter",
    "NNsightAttributeMixin",
    "NNsightCfg",
    "NNsightCfgTypes",
    "NNsightConfig",
    "NNsightModelBackend",
    "NNsightModule",
    "get_default_configs_per_pass",
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
