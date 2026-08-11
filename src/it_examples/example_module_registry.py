"""The example module registry: a thin assembly over the centralized hydration machinery.

The generic pieces live in core (hub design v3 §11.2 centralization): :class:`~interpretune.registry.ModuleHydrator`
and :class:`~interpretune.registry.LazyModuleRegistry` in ``interpretune.registry``; manifest parsing, key
derivation/parity, and config-file loading in ``interpretune.hub.manifest``. This module only points them at the
in-tree ``examples/`` publish sources and re-exports the loading helpers its consumers use.

FLIP PENDING (4c, sequenced last): once notebooks resolve via the cache-backed registry (local-publish bridge /
hub fetch), this module is deleted outright — in-tree ``examples/`` trees become publish sources ONLY, never a
load path.
"""

from __future__ import annotations

from pathlib import Path

from interpretune.hub.manifest import (  # noqa: F401  (re-exported for existing consumers)
    derive_config_key,
    iter_component_manifests,
    load_config_file,
)
from interpretune.registry import LazyModuleRegistry, instantiate_and_register

DEFAULT_REGISTRY_ROOT = Path(__file__).parent / "examples"

# NOTE: the former example_{datamodule,itmodule}_defaults are MATERIALIZED into the published
# configuration files (per-config self-containment, hub design v3 ruling) — published bodies are
# complete; no load-time defaults injection exists on the example path. Test-entry defaults live with
# the test registry (tests/module_registry.py).
default_experiment_tag = "test_itmodule"


def example_register_func(target_registry):
    """Registration callable for example configuration bodies (self-contained; no defaults injection)."""
    from functools import partial

    return partial(instantiate_and_register, target_registry=target_registry)


def make_example_registry(registry_root: Path | None = None) -> LazyModuleRegistry:
    """Assemble a lazy example registry over a component tree (defaults to the in-tree publish sources)."""
    return LazyModuleRegistry(
        registry_root=registry_root or DEFAULT_REGISTRY_ROOT, register_func_factory=example_register_func
    )


# Create lazy-loading instance over the in-tree publish sources (flip pending — see module docstring)
MODULE_EXAMPLE_REGISTRY = make_example_registry()
