"""#432: a hub-delivered adapter's classes are nameable from YAML through a stable, unversioned import path."""

from __future__ import annotations

import importlib
import sys

import pytest

from tests.core.test_hub_adapters import FIXTURE_ADAPTER, REGISTERS_DECLARED, _write_component  # noqa: F401
from tests.core.test_hub_adapters import restore_adapter_enum  # noqa: F401  (fixture, used by name)


@pytest.fixture()
def loaded(tmp_path, monkeypatch, restore_adapter_enum):  # noqa: F811
    from interpretune.adapters.registration import CompositionRegistry
    from interpretune.hub.adapters import load_hub_adapter
    from interpretune.hub.components import local_publish
    from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

    component = _write_component(
        tmp_path, REGISTERS_DECLARED + "\n    class FixtureAdapterConfig:\n        stable = True\n"
    )
    cache = tmp_path / "components"
    local_publish(component, "org/fixture-adapter", cache_dir=cache)
    monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
    load_hub_adapter("org/fixture-adapter", cache_dir=cache, registry=CompositionRegistry())
    return cache


class TestStableImportPath:
    def test_the_alias_is_importable_and_is_the_loaded_module(self, loaded):
        from interpretune.hub.adapters import loaded_adapter_module, stable_module_name

        alias = stable_module_name("org/fixture-adapter")
        assert alias == "it_hub_adapters.org__fixture_adapter"
        assert importlib.import_module(alias) is loaded_adapter_module("org/fixture-adapter", cache_dir=loaded)
        assert importlib.import_module("it_hub_adapters").org__fixture_adapter is importlib.import_module(alias)

    def test_a_yaml_class_path_resolves_through_the_alias(self, loaded):
        from interpretune.utils.import_utils import instantiate_class

        cls = instantiate_class(
            init={"class_path": "it_hub_adapters.org__fixture_adapter.FixtureAdapterConfig"}, import_only=True
        )
        assert cls.stable is True
        assert instantiate_class(
            init={"class_path": "it_hub_adapters.org__fixture_adapter.FixtureAdapterConfig"}
        ).stable

    def test_the_alias_does_not_exist_before_a_load(self):
        from interpretune.hub.adapters import stable_module_name

        name = stable_module_name("org/never-loaded-anywhere")
        assert name not in sys.modules
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)

    def test_the_alias_follows_the_most_recent_load(self, tmp_path, monkeypatch, restore_adapter_enum):  # noqa: F811
        """Two revisions in one process: the revision-scoped names both survive, the alias names the latest."""
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter, stable_module_name
        from interpretune.hub.components import local_publish
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        cache = tmp_path / "components"
        first = _write_component(tmp_path / "one", REGISTERS_DECLARED + "\n    MARK = 1\n")
        local_publish(first, "org/fixture-adapter", cache_dir=cache)
        load_hub_adapter("org/fixture-adapter", cache_dir=cache, registry=CompositionRegistry())
        assert importlib.import_module(stable_module_name("org/fixture-adapter")).MARK == 1
        second = _write_component(tmp_path / "two", REGISTERS_DECLARED + "\n    MARK = 2\n")
        local_publish(second, "org/fixture-adapter", cache_dir=cache)
        load_hub_adapter("org/fixture-adapter", cache_dir=cache, registry=CompositionRegistry())
        assert importlib.import_module(stable_module_name("org/fixture-adapter")).MARK == 2
