"""Small lazy import helpers used by tests to avoid importing heavy optional dependencies during pytest collection.

Usage patterns in tests:
    from tests._lazy_imports import lazy_import, lazy_attr
    torch = lazy_import("torch")
    ActivationCache = lazy_attr("transformer_lens.ActivationCache")

This module intentionally keeps implementation tiny and dependency-free.
"""

from __future__ import annotations

import importlib
import types
from typing import Any
import sys


class _LazyModule(types.ModuleType):
    def __init__(self, module_name: str):
        super().__init__(module_name)
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
            # copy attributes for faster subsequent access
            self.__dict__.update(self._module.__dict__)
        return self._module

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - lightweight runtime helper
        mod = self._load()
        return getattr(mod, item)


class _LazyAttr:
    """Proxy for a single attribute referenced by dotted path.

    Example: lazy_attr('torch.testing.assert_close') returns a callable proxy that
    resolves and dispatches to the real object on first call or attribute access.
    """

    def __init__(self, dotted_path: str):
        self._dotted_path = dotted_path
        self._resolved = False
        self._obj = None

    def _resolve(self):
        if not self._resolved:
            module_path, attr = self._dotted_path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            self._obj = getattr(mod, attr)
            self._resolved = True
            # Best-effort: replace this proxy with the real object in any module
            # that still references the proxy in its globals. This handles the
            # common pattern "X = lazy_attr('a.b.C')" used in tests.
            try:
                for mod in list(sys.modules.values()):
                    if mod is None:
                        continue
                    gd = getattr(mod, "__dict__", None)
                    if not isinstance(gd, dict):
                        continue
                    for name, val in list(gd.items()):
                        if val is self:
                            gd[name] = self._obj
            except Exception:
                # best-effort only; failing to replace is non-fatal
                pass
        return self._obj

    def __call__(self, *args, **kwargs):  # pragma: no cover - dispatch helper
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, item: str):  # pragma: no cover
        return getattr(self._resolve(), item)


def lazy_import(module_name: str) -> types.ModuleType:
    """Return a module-like proxy that imports on-demand."""
    return _LazyModule(module_name)


def lazy_attr(dotted_path: str) -> Any:
    """Return a proxy for a single attribute (callable/class/etc) that resolves lazily."""
    return _LazyAttr(dotted_path)
