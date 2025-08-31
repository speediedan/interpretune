"""Lazy-loading example module registry to improve test collection performance."""

from pathlib import Path
from functools import partial
import threading
from typing import Any, Union, Tuple

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


class LazyModuleRegistry:
    """Lazy loading wrapper for ModuleRegistry that defers initialization until first access."""

    def __init__(self):
        self._registry = None
        self._lock = threading.Lock()

    def _ensure_initialized(self):
        """Ensure the underlying registry is initialized (thread-safe)."""
        if self._registry is None:
            with self._lock:
                if self._registry is None:  # Double-check locking
                    self._registry = self._create_registry()

    def _create_registry(self):
        """Create the actual MODULE_EXAMPLE_REGISTRY with all the imports."""
        from interpretune.base import IT_BASE
        from interpretune.registry import ModuleRegistry, gen_module_registry, instantiate_and_register, apply_defaults
        from tests.modules import TestITDataModule, TestITModule

        DEFAULT_TEST_DATAMODULE = TestITDataModule
        DEFAULT_TEST_MODULE = TestITModule
        DEFAULT_MODULE_EXAMPLE_REGISTRY_PATH = Path(IT_BASE) / "example_module_registry.yaml"

        registry = ModuleRegistry()

        # Register Test/Example Module Configs
        itdm_cfg_defaults = partial(apply_defaults, defaults=example_datamodule_defaults)
        it_cfg_defaults = partial(apply_defaults, defaults=example_itmodule_defaults)

        example_instantiate_and_register = partial(
            instantiate_and_register,
            datamodule_cls=DEFAULT_TEST_DATAMODULE,
            module_cls=DEFAULT_TEST_MODULE,
            target_registry=registry,
            itdm_cfg_defaults_fn=itdm_cfg_defaults,
            it_cfg_defaults_fn=it_cfg_defaults,
        )

        gen_module_registry(
            yaml_reg_path=DEFAULT_MODULE_EXAMPLE_REGISTRY_PATH, register_func=example_instantiate_and_register
        )

        return registry

    def get(self, target: Union[Tuple, str, Any], default: Any = None) -> Any:
        """Get item from registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.get(target, default)

    def register(self, *args, **kwargs):
        """Register item in registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.register(*args, **kwargs)

    def __getitem__(self, key):
        """Get item from registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry[key]

    def __setitem__(self, key, value):
        """Set item in registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        self._registry[key] = value

    def __contains__(self, key):
        """Check if key is in registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return key in self._registry

    def keys(self):
        """Get keys from registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.keys()

    def values(self):
        """Get values from registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.values()

    def items(self):
        """Get items from registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.items()

    def __len__(self):
        """Get length of registry, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return len(self._registry)

    def __str__(self):
        """String representation, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return str(self._registry)

    def __repr__(self):
        """Repr, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return repr(self._registry)

    # Forward other common methods
    def available_keys(self, *args, **kwargs):
        """Available keys, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.available_keys(*args, **kwargs)

    def available_keys_feedback(self, *args, **kwargs):
        """Available keys feedback, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.available_keys_feedback(*args, **kwargs)

    def available_compositions(self, *args, **kwargs):
        """Available compositions, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.available_compositions(*args, **kwargs)

    def remove(self, *args, **kwargs):
        """Remove item, initializing if needed."""
        self._ensure_initialized()
        assert self._registry is not None
        return self._registry.remove(*args, **kwargs)


# Create lazy-loading instance
MODULE_EXAMPLE_REGISTRY = LazyModuleRegistry()
