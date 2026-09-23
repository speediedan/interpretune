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
from dataclasses import dataclass

import pytest

from interpretune.base import ITDataModule, BaseITModule
from interpretune.protocol import Adapter
from interpretune.adapters import CompositionRegistry, ITModule, LightningAdapter, LightningModule


class TestClassRegistration:
    def test_canonicalize_composition(self):
        TEST_ADAPTER_REGISTRY = CompositionRegistry()
        adapter_tuple_cc_result = TEST_ADAPTER_REGISTRY.canonicalize_composition(
            (
                "transformer_lens",
                Adapter.core,
            )
        )
        str_list_cc_result = TEST_ADAPTER_REGISTRY.canonicalize_composition(["core", "transformer_lens", Adapter.core])
        assert adapter_tuple_cc_result == str_list_cc_result == (Adapter.core, Adapter.transformer_lens)

    def test_register_adapter(self):
        TEST_ADAPTER_REGISTRY = CompositionRegistry()
        TEST_ADAPTER_REGISTRY.register(
            Adapter.core,
            component_key="module",
            adapter_combination=(Adapter.core,),
            composition_classes=(ITModule,),
            description="core adapter to be used with native PyTorch",
        )
        with pytest.raises(KeyError, match="was not found in the registry. Available valid compositions"):
            TEST_ADAPTER_REGISTRY.get((Adapter.lightning,))
        TEST_ADAPTER_REGISTRY.register(
            Adapter.core,
            component_key="datamodule",
            adapter_combination=(Adapter.core,),
            composition_classes=(ITDataModule,),
            description="core adapter to be used with native PyTorch",
        )
        TEST_ADAPTER_REGISTRY.register(
            Adapter.core,
            component_key="module",
            adapter_combination=(Adapter.lightning,),
            composition_classes=(LightningAdapter, BaseITModule, LightningModule),
            description="lighting adapter",
        )
        available_set = TEST_ADAPTER_REGISTRY.available_compositions(adapter_filter="core")
        assert available_set == {("module", Adapter.core), ("datamodule", Adapter.core)}
        with pytest.warns(UserWarning, match="The following adapter names"):
            _ = TEST_ADAPTER_REGISTRY.available_compositions(adapter_filter=[Adapter.core, "oops"])
        TEST_ADAPTER_REGISTRY.remove(("datamodule", Adapter.core))
        TEST_ADAPTER_REGISTRY.remove(("module", Adapter.lightning))
        assert TEST_ADAPTER_REGISTRY.available_compositions() == {("module", Adapter.core)}
        assert str(TEST_ADAPTER_REGISTRY)


def test_lazy_adapter_registry_initializes(caplog):
    """Ensure the lazy adapter registry can be accessed without raising.

    The test triggers lazy initialization by calling a read-only method. The expectation is that initialization
    completes (possibly leaving an empty registry if adapters couldn't be imported) and the call returns a set.
    """
    from interpretune.adapter_registry import ADAPTER_REGISTRY

    # Should not raise and should return a set (possibly empty)
    comps = ADAPTER_REGISTRY.available_compositions()
    assert isinstance(comps, set)


class TestRegistryBackedConfigDiscovery:
    """Every adapter's config class, bundled or hub-delivered, reaches auto-composition only through the registry.

    A hub-delivered adapter runs from a revision-scoped synthetic module, so no import path can name it. Bundled
    adapters therefore register the same way rather than being found by a path derived from their name, which would be a
    route only the bundled set could ever take.
    """

    @staticmethod
    def _cfg_cls():
        from interpretune.config.module import ITConfig

        @dataclass(kw_only=True)
        class _RegisteredCfg(ITConfig):
            pass

        return _RegisteredCfg

    def test_a_registered_class_comes_back(self):
        registry = CompositionRegistry()
        cls = self._cfg_cls()
        registry.register_module_cfg_class(Adapter.circuit_tracer, cls)
        assert registry.module_cfg_class(Adapter.circuit_tracer) is cls

    def test_an_adapter_that_registered_nothing_answers_none(self):
        # The common case by far, and it has to be None rather than a guess: every bundled-only session
        # takes this path for every adapter.
        assert CompositionRegistry().module_cfg_class(Adapter.circuit_tracer) is None

    def test_the_adapter_may_be_named_by_string(self):
        registry = CompositionRegistry()
        cls = self._cfg_cls()
        registry.register_module_cfg_class("circuit_tracer", cls)
        assert registry.module_cfg_class("circuit_tracer") is cls
        assert registry.module_cfg_class(Adapter.circuit_tracer) is cls

    def test_registering_the_same_class_twice_is_accepted(self):
        # A component reloaded in one session re-runs its entrypoint; that must not be an error.
        registry = CompositionRegistry()
        cls = self._cfg_cls()
        registry.register_module_cfg_class(Adapter.circuit_tracer, cls)
        registry.register_module_cfg_class(Adapter.circuit_tracer, cls)
        assert registry.module_cfg_class(Adapter.circuit_tracer) is cls

    def test_a_second_different_class_for_one_adapter_is_refused_by_name(self):
        # Silently keeping one would make composition depend on registration order, which a caller can
        # neither see nor control.
        registry = CompositionRegistry()
        first, second = self._cfg_cls(), self._cfg_cls()
        registry.register_module_cfg_class(Adapter.circuit_tracer, first)
        with pytest.raises(ValueError, match="already registered module cfg class"):
            registry.register_module_cfg_class(Adapter.circuit_tracer, second)
        assert registry.module_cfg_class(Adapter.circuit_tracer) is first

    def test_discovery_returns_a_registered_class(self, monkeypatch):
        """The integration: a class reachable by no import path is still discovered.

        Carries its own positive control. `circuit_tracer` defines no ``ITConfig`` subclass, so it is
        absent from discovery before the registration -- asserting that first is what makes the second
        assertion evidence rather than a statement about an adapter that was already there.
        """
        from interpretune.config.module import ITConfig
        from interpretune.config import shared

        before, _ = shared.find_adapter_subclasses(ITConfig)
        assert Adapter.circuit_tracer not in before, (
            "positive control failed: circuit_tracer was already discoverable, so this test cannot show "
            "that the registry is what makes the class reachable"
        )

        cls = self._cfg_cls()
        monkeypatch.setattr(shared, "_registered_cfg_classes", lambda space: {Adapter.circuit_tracer: cls})
        after, _ = shared.find_adapter_subclasses(ITConfig)
        assert after.get(Adapter.circuit_tracer) is cls

    def test_bundled_adapters_register_their_config_class(self):
        """Bundled adapters take the hub route: each registers its module config class in ``register_adapter_ctx``."""
        from interpretune.adapter_registry import ADAPTER_REGISTRY
        from interpretune.adapters.nnsight.config import ITNNsightConfig
        from interpretune.adapters.sae_lens.config import SAELensConfig
        from interpretune.adapters.transformer_lens.config import ITLensConfig

        registered = {a: ADAPTER_REGISTRY.module_cfg_class(a) for a in Adapter.__members__.values()}
        expected = {
            Adapter.transformer_lens: ITLensConfig,
            Adapter.sae_lens: SAELensConfig,
            Adapter.nnsight: ITNNsightConfig,
        }
        assert {a: c for a, c in registered.items() if c is not None} == expected

    def test_discovery_reads_only_the_registry(self, monkeypatch):
        """With the registry emptied, a bundled adapter is not discoverable either: there is no second route.

        The first assertion is the positive control: nnsight IS discoverable through the live registry, so its absence
        afterwards is caused by the emptied registry rather than by nnsight never having been reachable.
        """
        from interpretune.config.module import ITConfig
        from interpretune.config import shared

        found, _ = shared.find_adapter_subclasses(ITConfig)
        assert Adapter.nnsight in found, "positive control failed: nnsight is not discoverable through the registry"

        monkeypatch.setattr(shared, "_registered_cfg_classes", lambda space: {})
        assert shared.find_adapter_subclasses(ITConfig) == ({}, {})
