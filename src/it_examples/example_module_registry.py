"""Per-key lazy example module registry backed by decomposed component trees.

Example definitions live in ``src/it_examples/examples/<task>/`` mirroring the Hub component-repo layout
(interpretune#1): an ``it_component.yaml`` manifest indexing self-contained configuration files under
``configs/<key>.yaml``, where ``<key>`` is derived as ``<task_variant>.<model>.<composition>[.<descriptor>]``
(composition = ``+``-joined adapters in the composition registry's canonical order).

Resolution is per-key (interpretune#236): ``MODULE_EXAMPLE_REGISTRY.get(<key>)`` parses the small manifests, then
constructs ONLY the requested entry — a broken sibling configuration cannot fail a caller that never touches it.
Tuple/composition lookups (a test-suite pattern) fall back to hydrating all entries.
"""

from __future__ import annotations

from pathlib import Path
from functools import partial
import threading
from typing import Any, Callable, Iterator, Tuple, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from interpretune.registry import ModuleRegistry

DEFAULT_REGISTRY_ROOT = Path(__file__).parent / "examples"

# Export commonly used defaults to maintain API compatibility
default_experiment_tag = "test_itmodule"
example_datamodule_defaults = dict(prepare_data_map_cfg={"batched": True})
example_itmodule_defaults = dict(
    optimizer_init={
        "class_path": "torch.optim.AdamW",
        "init_args": {"weight_decay": 1.0e-06, "eps": 1.0e-07, "lr": 3.0e-05},
    },
    lr_scheduler_init={
        "class_path": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "init_args": {"T_0": 1, "T_mult": 2, "eta_min": 1.0e-06},
    },
)


def derive_config_key(cfg: dict) -> str:
    """Derive the canonical configuration key (canonical implementation: ``interpretune.hub.manifest``)."""
    from interpretune.hub.manifest import derive_config_key as _derive

    return _derive(cfg)


def example_register_func(target_registry):
    """Registration callable applying the example config defaults into ``target_registry``."""
    from interpretune.registry import instantiate_and_register, apply_defaults

    return partial(
        instantiate_and_register,
        target_registry=target_registry,
        itdm_cfg_defaults_fn=partial(apply_defaults, defaults=example_datamodule_defaults),
        it_cfg_defaults_fn=partial(apply_defaults, defaults=example_itmodule_defaults),
    )


def iter_component_manifests(registry_root: Path | None = None) -> Iterator[tuple[Path, dict]]:
    """Yield ``(component_dir, parsed_manifest)`` for every component tree under ``registry_root``."""
    root = registry_root or DEFAULT_REGISTRY_ROOT
    if not root.is_dir():
        return
    for manifest_path in sorted(root.glob("*/it_component.yaml")):
        with open(manifest_path, encoding="utf-8") as fh:
            yield manifest_path.parent, yaml.safe_load(fh)


def load_config_file(config_path: Path, expected_key: str | None = None) -> tuple[str, dict]:
    """Load one configuration file, parity-checking filename == manifest key == derived-from-fields."""
    from interpretune.hub.manifest import check_config_key_parity

    with open(config_path, encoding="utf-8") as fh:
        body = yaml.safe_load(fh)
    return check_config_key_parity(config_path, body, expected_key=expected_key), body


class ExampleRegistryHydrator:
    """Per-key hydration of example configuration files into a ``ModuleRegistry``."""

    def __init__(self, registry_root: Path | None = None):
        self.registry_root = registry_root or DEFAULT_REGISTRY_ROOT
        self._index: dict[str, Path] | None = None
        self._hydrated: set[str] = set()
        self._registry = None
        self._lock = threading.RLock()

    @property
    def index(self) -> dict[str, Path]:
        """Key → configuration-file map assembled from the (small) component manifests; no entry construction."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    index: dict[str, Path] = {}
                    for component_dir, manifest in iter_component_manifests(self.registry_root):
                        for key, rel in (manifest.get("module", {}).get("configs") or {}).items():
                            index[key] = component_dir / rel
                    self._index = index
        return self._index

    @property
    def registry(self) -> "ModuleRegistry":
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    from interpretune.registry import ModuleRegistry

                    self._registry = ModuleRegistry()
        return self._registry

    def _register_func(self):
        return example_register_func(self.registry)

    def hydrate(self, key: str) -> bool:
        """Construct and register exactly one entry.

        Returns False if the key is not in any manifest.
        """
        if key in self._hydrated:
            return True
        if key not in self.index:
            return False
        with self._lock:
            if key in self._hydrated:
                return True
            parity_key, body = load_config_file(self.index[key], expected_key=key)
            self._register_func()(parity_key, body)
            self._hydrated.add(key)
        return True

    def hydrate_all(self) -> None:
        """Hydrate every indexed entry.

        Bulk hydration suppresses per-entry config-normalization feedback (`ITInstantiationFeedbackWarning`) just
        as the pre-decomposition bulk loader did — callers listing or tuple-resolving the registry did not ask for
        any single entry's feedback. Single-key `hydrate()` stays verbose: per-key construction is exactly the
        "directly instantiating a config" case where feedback should surface (interpretune#236).
        """
        import warnings

        from interpretune.utils import ITInstantiationFeedbackWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ITInstantiationFeedbackWarning)
            for key in self.index:
                self.hydrate(key)


class LazyModuleRegistry:
    """Mapping-style facade over per-key lazy hydration; ``builder`` swaps in an eager registry (tests)."""

    def __init__(self, builder: Callable[[], "ModuleRegistry"] | None = None, registry_root: Path | None = None):
        self._builder = builder
        self._built = None
        self._hydrator = None if builder is not None else ExampleRegistryHydrator(registry_root)
        self._lock = threading.RLock()

    @property
    def registry(self) -> "ModuleRegistry":
        """The underlying registry.

        Builder mode constructs eagerly; hydrator mode does NOT hydrate here.
        """
        if self._builder is not None:
            if self._built is None:
                with self._lock:
                    if self._built is None:
                        self._built = self._builder()
            return self._built
        assert self._hydrator is not None
        return self._hydrator.registry

    def _resolve(self, key) -> None:
        """Hydrate what a lookup needs: one entry for a known string key, everything for tuple/protocol keys."""
        if self._hydrator is None:
            return
        if isinstance(key, str):
            if not self._hydrator.hydrate(key):
                # unknown string key: hydrate all so the raised KeyError lists every available entry
                self._hydrator.hydrate_all()
        else:
            self._hydrator.hydrate_all()

    def get(self, target: Tuple | str | Any, default: Any = None) -> Any:
        self._resolve(target)
        return self.registry.get(target, default)

    def register(self, *args, **kwargs):
        return self.registry.register(*args, **kwargs)

    def __getitem__(self, key):
        self._resolve(key)
        return self.registry[key]

    def __setitem__(self, key, value):
        self.registry[key] = value

    def __contains__(self, key):
        if self._hydrator is not None and isinstance(key, str) and key in self._hydrator.index:
            return True
        self._resolve(key)
        return key in self.registry

    def _full_registry(self) -> "ModuleRegistry":
        if self._hydrator is not None:
            self._hydrator.hydrate_all()
        return self.registry

    def keys(self):
        return self._full_registry().keys()

    def values(self):
        return self._full_registry().values()

    def items(self):
        return self._full_registry().items()

    def __len__(self):
        return len(self._full_registry())

    def __str__(self):
        return str(self._full_registry())

    def __repr__(self):
        return repr(self._full_registry())

    # Forward other common methods
    def available_keys(self, *args, **kwargs):
        return self._full_registry().available_keys(*args, **kwargs)

    def available_keys_feedback(self, *args, **kwargs):
        return self._full_registry().available_keys_feedback(*args, **kwargs)

    def available_compositions(self, *args, **kwargs):
        return self._full_registry().available_compositions(*args, **kwargs)

    def remove(self, *args, **kwargs):
        return self.registry.remove(*args, **kwargs)


# Create lazy-loading instance
MODULE_EXAMPLE_REGISTRY = LazyModuleRegistry()
