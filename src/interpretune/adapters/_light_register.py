"""Lightweight adapter registration utilities.

This module performs the minimal imports necessary to populate the
`ADAPTER_REGISTRY` by calling `register_adapter_ctx` on adapter classes.
It intentionally avoids importing heavy runtime dependencies to enable
adapter implementation modules to be written (optionally) to be safe to import at
module-level when optional heavy dependencies are guarded by TYPE_CHECKING
or local imports.

Keep this module small and dependency-free so it can be imported at
package initialization time to ensure the registry is populated for
runtime consumers that expect registrations to exist.
"""

from importlib import import_module
from types import ModuleType
from typing import Iterable


def _import_adapter_module(module_path: str) -> ModuleType:
    """Import an adapter module and return the module object.

    We rely on adapter modules to avoid importing heavy third-party dependencies at module import time (they use
    TYPE_CHECKING/local imports). If a module import raises, we surface a warning but do not fail hard so the rest of
    registration can continue.
    """
    try:
        return import_module(module_path)
    except Exception:
        # Import failures here are non-fatal for the import-time
        # registration pass; leave the registry empty for adapters that
        # could not be imported and allow them to register lazily later.
        return None  # type: ignore[return-value]


def register_all_adapters(registry) -> None:
    """Call `register_adapter_ctx` on each known adapter class.

    This function imports adapter implementation modules and invokes the registration classmethod on matching classes.
    The set of modules is initially explicit and small but may switch to an entrypoint-based discovery mechanism in the
    future.
    """
    # TODO: consider making this auto-discoverable via entrypoints
    adapter_modules: Iterable[str] = (
        "interpretune.adapters.core",
        "interpretune.adapters.lightning",
        "interpretune.adapters.sae_lens",
        "interpretune.adapters.transformer_lens",
        "interpretune.adapters.circuit_tracer",
    )

    for mod_path in adapter_modules:
        mod = _import_adapter_module(mod_path)
        if mod is None:
            continue
        # Each adapter module defines one or more adapter classes that implement
        # `register_adapter_ctx`; find them and call the registration method.
        # Avoid importing heavy symbols here; rely on module-level classes.
        try:
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                # We intentionally check for callability and the presence of
                # the registration method rather than concrete types to keep
                # this pass lightweight.
                if hasattr(attr, "register_adapter_ctx") and callable(getattr(attr, "register_adapter_ctx")):
                    try:
                        attr.register_adapter_ctx(registry)
                    except Exception:
                        # Don't let a single adapter registration failure stop the rest.
                        continue
        except Exception:
            # Defensive: if introspection on the module fails, skip it.
            continue
