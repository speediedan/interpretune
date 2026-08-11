"""Test-owned module registry (interpretune#1 / #236 workstream, lane 3d).

Owns the pytest-scale entries split out of the former monolithic example registry — the ones whose
datamodule/module classes live in ``tests.modules`` and therefore cannot ship. Registers BOTH YAMLs (test entries
first, then the example entries) into one registry so test parametrization keeps seeing the full original key set;
the example entries all declare shipping classes explicitly, so the test-class defaults below never leak into them.

``it_examples`` must never import from this module (that is the dependency 3d severed) — the import direction is
tests -> src only.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from interpretune.registry import LazyModuleRegistry

# Test-entry config defaults (formerly it_examples' example_*_defaults): published example configs
# are self-contained, so these now exist ONLY for the pytest-scale entries in this registry.
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

TEST_MODULE_REGISTRY_PATH = Path(__file__).parent / "test_module_registry.yaml"


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

    # test keys use the SAME derived grammar as hub configuration keys (umbrella design §11.8):
    # parity-check every mapping key against derive_config_key over the entry's structured fields
    import yaml
    from interpretune.hub.manifest import derive_config_key

    with open(TEST_MODULE_REGISTRY_PATH, encoding="utf-8") as fh:
        for key, body in yaml.safe_load(fh).items():
            derived = derive_config_key(body)
            if derived != key:
                raise ValueError(f"Test-registry key parity violation: {key!r} vs derived {derived!r}")
    # seeds FIRST: test entries composing published prompt definitions (compose_ref) hydrate during
    # gen_module_registry and resolve cache-only, so the components cache must already be seeded
    from interpretune.hub.components import resolve_component_config, resolve_component_manifest
    from it_examples.seeds import SEED_COMPONENTS, ensure_local_seeds

    ensure_local_seeds()
    gen_module_registry(yaml_reg_path=TEST_MODULE_REGISTRY_PATH, register_func=test_instantiate_and_register)
    # example entries register second (so a key collision would resolve toward the shipping definition),
    # resolved from the COMPONENTS CACHE via the local-publish bridge — post-flip, in-tree examples/
    # trees are publish sources only; the loader reads exclusively from the cache, and the publish
    # machinery itself is exercised on every test run (co-evolution atomicity)
    for repo_id in SEED_COMPONENTS:
        manifest, _, _ = resolve_component_manifest(repo_id)
        for key in manifest.get("module", {}).get("configs") or {}:
            parity_key, body = resolve_component_config(repo_id, key)
            test_instantiate_and_register(parity_key, body)

    return registry


TEST_MODULE_REGISTRY = LazyModuleRegistry(builder=_create_test_registry)


def make_component_tree_registry(registry_root: Path) -> LazyModuleRegistry:
    """Hydrator-mode registry over a component tree (TEST-ONLY: exercises the core hydration machinery).

    The runtime loader never reads component trees directly post-flip; tests use this to exercise per-key hydration,
    parity enforcement, and broken-sibling isolation against fixture trees.
    """
    from interpretune.registry import instantiate_and_register

    return LazyModuleRegistry(
        registry_root=registry_root,
        register_func_factory=lambda reg: partial(instantiate_and_register, target_registry=reg),
    )
