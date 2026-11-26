# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

from interpretune.registry import ModuleRegistry, RegisteredCfg, RegKeyType, it_cfg_factory
from tests.module_registry import TEST_MODULE_REGISTRY
from interpretune.protocol import Adapter
from tests.base_defaults import default_test_task
from base_defaults import BaseCfg


class TestClassModuleRegistration:
    @dataclass(kw_only=True)
    class DegenRegKeyObj:
        device_type: str = "cpu"
        model_cfg_key: str = "rte"

    def test_it_cfg_factory_direct(self):
        from interpretune.config import ITConfig

        cfg = TEST_MODULE_REGISTRY.get("rte.cust.core")
        orig_cfg = cfg.module_cfg.__dict__
        for k in ("entailment_mapping_indices", "entailment_mapping"):
            del orig_cfg[k]
        cfg = it_cfg_factory(cfg=orig_cfg)
        assert isinstance(cfg, ITConfig)

    def test_register_module(self, capsys):
        registry = ModuleRegistry()
        registry.register(
            reg_key="cust.test_example",
            model_src_key="cust",
            model_cfg_key=default_test_task,
            adapter_combinations=(Adapter.core,),
            registered_cfg=MagicMock(spec=RegisteredCfg),
            description="Testing example registration by both key and composition",
        )
        registry.register(
            reg_key="cust.test_example.transformer_lens",
            model_src_key="cust",
            model_cfg_key=default_test_task,
            adapter_combinations=[
                (Adapter.core, Adapter.transformer_lens),
                (Adapter.lightning, Adapter.transformer_lens),
            ],
            registered_cfg=MagicMock(spec=RegisteredCfg),
            cfg_dict={"orig": "cfg"},
            description="Testing example registration",
        )

        registry.get(("cust", default_test_task, (Adapter.core,)))
        registry.get("cust.test_example.transformer_lens")

        available_set = registry.available_compositions(adapter_filter="core")
        assert available_set == {
            ("cust", "rte", (Adapter.core,)),
            ("cust", "rte", (Adapter.core, Adapter.transformer_lens)),
        }
        assert registry["cust.test_example.transformer_lens"]["cfg_dict"] == {"orig": "cfg"}

        assert str(registry).startswith("Registered Modules: dict_keys(['cust.test_example', ('c")
        superset = registry.available_compositions()
        assert superset == available_set | {("cust", "rte", (Adapter.lightning, Adapter.transformer_lens))}

        registry.get(BaseCfg(model_src_key="cust", model_cfg_key="rte", adapter_ctx=(Adapter.core,)))

        with pytest.raises(AssertionError, match="Non-string/non-tuple keys must be `RegKeyQueryable`"):
            registry.get(TestClassModuleRegistration.DegenRegKeyObj(model_cfg_key="rte"))

        with pytest.raises(KeyError, match="was not found in the registry"):
            registry.get(("cust", default_test_task, (Adapter.lightning, Adapter.sae_lens)))

        with pytest.raises(KeyError, match="was not found in the registry"):
            registry.get("cust.oops_dont_exist.transformer_lens")

        expected_composed_available = {
            RegKeyType.STRING: "cust.test_example                   Testing",
            RegKeyType.TUPLE: "('cust', 'rte', (<Adapter.core: 'core'>,)) ",
            RegKeyType.COMBO: "e'>, <Adapter.transformer_lens: 'transformer_lens'>))            Testing",
        }

        for key_type in RegKeyType:
            registry.available_keys(key_type=key_type)
            captured = capsys.readouterr()
            assert expected_composed_available[key_type] in captured.out.strip()

        registry.available_keys()
        captured = capsys.readouterr()
        assert expected_composed_available[RegKeyType.STRING] in captured.out.strip()

        registry.remove(("cust", default_test_task, (Adapter.core,)))
        with pytest.raises(KeyError, match="was not found in the registry"):
            registry.get(("cust", default_test_task, (Adapter.core,)))


class TestComponentTreeRegistryLaziness:
    """Per-key hydration guarantees for the core hydrator machinery.

    Post-flip, no runtime registry reads the in-tree example trees — these tests exercise the CORE
    ``ModuleHydrator``/``LazyModuleRegistry`` machinery against a copy of the warmed rte snapshot as fixture
    data (via the test-only ``make_component_tree_registry`` helper).
    """

    GOOD_KEY = "rte_demo.gemma2.circuit_tracer+nnsight"
    BROKEN_KEY = "rte_demo.gpt2.sae_lens"

    @pytest.fixture()
    def component_root(self, tmp_path):
        """A component tree copied from the warmed rte snapshot: tests below may mutate it freely."""
        from tests.rte_component import rte_component_tree

        return rte_component_tree(tmp_path / "registry")

    @pytest.fixture()
    def broken_sibling_root(self, component_root):
        """The component tree with one configuration's class_path made unimportable."""
        root = component_root
        broken = root / "rte" / "configs" / f"{self.BROKEN_KEY}.yaml"
        broken.write_text(
            broken.read_text(encoding="utf-8").replace("rte_boolq.RTEBoolqDataModule", "no.such.module.Nope"),
            encoding="utf-8",
        )
        return root

    def test_get_hydrates_only_requested_entry(self, component_root):
        from tests.module_registry import make_component_tree_registry

        registry = make_component_tree_registry(component_root)
        cfg = registry.get(self.GOOD_KEY)
        assert cfg is not None
        assert registry._hydrator._hydrated == {self.GOOD_KEY}
        # membership checks must not construct entries either
        assert self.BROKEN_KEY in registry
        assert registry._hydrator._hydrated == {self.GOOD_KEY}

    def test_broken_sibling_does_not_break_other_entries(self, broken_sibling_root):
        from tests.module_registry import make_component_tree_registry

        registry = make_component_tree_registry(broken_sibling_root)
        assert registry.get(self.GOOD_KEY) is not None  # must not raise
        with pytest.raises(ModuleNotFoundError):
            registry.get(self.BROKEN_KEY)

    def test_key_parity_enforced(self, component_root):
        """Filename == manifest key == derived-from-fields, or the loader refuses."""
        from tests.module_registry import make_component_tree_registry

        root = component_root
        drifted = root / "rte" / "configs" / f"{self.GOOD_KEY}.yaml"
        # change a structured field without renaming the file: derived key no longer matches the stem
        drifted.write_text(
            drifted.read_text(encoding="utf-8").replace("task_variant: rte_demo", "task_variant: rte"), encoding="utf-8"
        )
        registry = make_component_tree_registry(root)
        with pytest.raises(ValueError, match="parity violation"):
            registry.get(self.GOOD_KEY)

    def test_all_seed_keys_resolve_from_cache(self, tmp_path):
        """Every seed configuration resolves CACHE-ONLY post-flip (bridge -> resolve -> hydrate).

        Both seeds are Hub-resident: the RTE experiment's published module keys hydrate from the warmed
        default cache at the pinned self-contained revision, and the prompt definitions take the
        local-publish bridge roundtrip from their warmed snapshot (a dev checkout without either
        warmed component fails here naming the fetch).
        """
        from interpretune.hub.components import resolve_component_manifest
        from interpretune.hub.promptconfigs import resolve_prompt_config_class
        from interpretune.testing.conformance.inputs import SEED_REPO, SEED_REVISION
        import interpretune as it
        from tests.prompt_configs_component import PROMPT_CONFIGS_REPO, publish_prompt_configs

        cache = tmp_path / "components"
        publish_prompt_configs(cache)
        manifest, _, _ = resolve_component_manifest(PROMPT_CONFIGS_REPO, cache_dir=cache)
        definitions = sorted(manifest["promptconfigs"]["definitions"])
        assert definitions == ["GemmaPromptConfig", "Llama3PromptConfig"]
        for name in definitions:
            assert resolve_prompt_config_class(f"{PROMPT_CONFIGS_REPO}#{name}", cache_dir=cache).__name__ == name
        rte_manifest, _, _ = resolve_component_manifest(SEED_REPO, revision=SEED_REVISION)
        rte_keys = sorted((rte_manifest.get("module") or {}).get("configs") or {})
        assert len(rte_keys) == 16  # rte_demo (8), rte_ft (5), rte_sparse (2), rte_latent (1)
        for key in rte_keys:
            assert it.hub.load(SEED_REPO, key, revision=SEED_REVISION) is not None
