"""Test-owned module registry (interpretune#1 / #236 workstream, lane 3d).

Owns the pytest-scale entries split out of ``src/it_examples/example_module_registry.yaml`` — the ones whose
datamodule/module classes live in ``tests.modules`` and therefore cannot ship. Registers BOTH YAMLs (test entries
first, then the example entries) into one registry so test parametrization keeps seeing the full original key set;
the example entries all declare shipping classes explicitly, so the test-class defaults below never leak into them.

``it_examples`` must never import from this module (that is the dependency 3d severed) — the import direction is
tests -> src only.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from it_examples.example_module_registry import (
    LazyModuleRegistry,
    example_datamodule_defaults,
    example_itmodule_defaults,
    iter_component_manifests,
    load_config_file,
)

TEST_MODULE_REGISTRY_PATH = Path(__file__).parent / "module_registry.yaml"


def _create_test_registry():
    """Build a registry holding the test entries plus the example entries (test-class defaults for the former)."""
    from interpretune.registry import ModuleRegistry, gen_module_registry, instantiate_and_register, apply_defaults
    from tests.modules import TestITDataModule, TestITModule

    registry = ModuleRegistry()

    itdm_cfg_defaults = partial(apply_defaults, defaults=example_datamodule_defaults)
    it_cfg_defaults = partial(apply_defaults, defaults=example_itmodule_defaults)

    test_instantiate_and_register = partial(
        instantiate_and_register,
        datamodule_cls=TestITDataModule,
        module_cls=TestITModule,
        target_registry=registry,
        itdm_cfg_defaults_fn=itdm_cfg_defaults,
        it_cfg_defaults_fn=it_cfg_defaults,
    )

    gen_module_registry(yaml_reg_path=TEST_MODULE_REGISTRY_PATH, register_func=test_instantiate_and_register)
    # example entries register second (from the decomposed component trees) so a key collision would
    # resolve toward the shipping definition
    for component_dir, manifest in iter_component_manifests():
        for key, rel in (manifest.get("module", {}).get("configs") or {}).items():
            parity_key, body = load_config_file(component_dir / rel, expected_key=key)
            test_instantiate_and_register(parity_key, body)

    return registry


TEST_MODULE_REGISTRY = LazyModuleRegistry(builder=_create_test_registry)
