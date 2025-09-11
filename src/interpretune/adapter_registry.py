"""Adapter registry with lazy initialization.

We intentionally defer the light-weight adapter registration pass until the registry is first accessed. This mirrors the
lazy-loading pattern used by the example module registry and avoids importing optional heavy dependencies at package
import time while still ensuring the registry is populated when it's needed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from interpretune.adapters.registration import CompositionRegistry

_logger = logging.getLogger(__name__)


class LazyCompositionRegistry:
    """Lazy-initializing wrapper around `CompositionRegistry`.

    The underlying `CompositionRegistry` is created and populated by calling
    the light-weight registration helper `register_all_adapters` on first
    access.
    """

    def __init__(self) -> None:
        self._registry: Optional[CompositionRegistry] = None
        self._lock = threading.RLock()

    def _ensure_initialized(self) -> None:
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    registry = CompositionRegistry()
                    try:
                        # Import the lightweight registration helper lazily so
                        # we don't pull heavy optional deps at import-time.
                        from interpretune.adapters._light_register import register_all_adapters

                        register_all_adapters(registry)
                    except Exception:  # pragma: no cover - defensive logging
                        _logger.exception("Light adapter registration failed during lazy init")
                    self._registry = registry

    @property
    def registry(self) -> CompositionRegistry:
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry

    # Delegate commonly used methods to the underlying registry. Using
    # __getattr__ keeps the wrapper small while still exposing the full API.
    def __getattr__(self, name: str) -> Any:
        # Called only when attribute not found on the wrapper itself.
        self._ensure_initialized()
        return getattr(self._registry, name)  # type: ignore[arg-type]

    def __repr__(self) -> str:  # pragma: no cover - convenience
        # we use self._registry here to avoid triggering initialization with repr
        return f"LazyCompositionRegistry(initialized={self._registry is not None})"


# Expose singleton instance expected by the rest of the codebase.
ADAPTER_REGISTRY = LazyCompositionRegistry()
