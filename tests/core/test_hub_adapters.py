"""The ``adapters`` component kind (#125): manifest schema, the enum extension, and the load path.

Tested against a trivial in-test fixture component rather than a real adapter, deliberately: the rails and the passenger
have to be able to fail independently, or a broken adapter and broken rails present identically.
"""

from __future__ import annotations

import textwrap

import pytest

FIXTURE_ADAPTER = "fixture_adapter"


@pytest.fixture()
def restore_adapter_enum():
    """Undo dynamic ``Adapter`` members a test adds; the enum is process-global state."""
    from interpretune.protocol import Adapter
    from interpretune.adapters import registration

    names = set(Adapter._member_map_)
    yield
    for name in set(Adapter._member_map_) - names:
        member = Adapter._member_map_.pop(name)
        Adapter._member_names_.remove(name)
        Adapter._value2member_map_.pop(member.value, None)
        type.__delattr__(Adapter, name)
        registration._DYNAMIC_ADAPTERS.pop(name, None)


def _write_component(root, entrypoint_body: str, declares=(FIXTURE_ADAPTER,), name="fixture_adapter_entry.py"):
    """Materialize a minimal adapters component dir and return it."""
    import yaml

    component = root / "component"
    component.mkdir(parents=True, exist_ok=True)
    (component / "it_component.yaml").write_text(
        yaml.safe_dump(
            {
                "it_schema_version": 1,
                "kinds": ["adapters"],
                "adapters": {
                    "entrypoint": name,
                    "declares": list(declares),
                    "compositions": [{"component": "module", "adapters": ["core", *declares]}],
                },
            }
        ),
        encoding="utf-8",
    )
    (component / name).write_text(textwrap.dedent(entrypoint_body), encoding="utf-8")
    return component


REGISTERS_DECLARED = '''
    class FixtureAdapterModule:
        """A stand-in for a real adapter mixin: the rails do not care what it composes."""

    class FixtureAdapter:
        @classmethod
        def register_adapter_ctx(cls, adapter_ctx_registry) -> None:
            adapter_ctx_registry.register(
                lead_adapter="fixture_adapter",
                component_key="module",
                adapter_combination=("core", "fixture_adapter"),
                composition_classes=(FixtureAdapterModule,),
                description="fixture adapter for the #125 rails",
            )
'''

REGISTERS_NOTHING = """
    class FixtureAdapter:
        pass
"""


class TestAdaptersManifestSchema:
    @pytest.mark.parametrize(
        ("manifest", "match"),
        [
            ({"it_schema_version": 1, "kinds": ["adapters"]}, "requires an `adapters.entrypoint`"),
            (
                {"it_schema_version": 1, "kinds": ["adapters"], "adapters": {"entrypoint": "e.py"}},
                "adapters.declares",
            ),
            (
                {"it_schema_version": 1, "kinds": ["adapters"], "adapters": {"entrypoint": "e.py", "declares": []}},
                "adapters.declares",
            ),
            (
                {
                    "it_schema_version": 1,
                    "kinds": ["adapters"],
                    "adapters": {"entrypoint": "e.py", "declares": ["not an identifier"]},
                },
                "adapters.declares",
            ),
            (
                {
                    "it_schema_version": 1,
                    "kinds": ["adapters"],
                    "adapters": {"entrypoint": "e.py", "declares": ["ok"], "compositions": [{"component": "module"}]},
                },
                "compositions",
            ),
        ],
        ids=["no-section", "no-declares", "empty-declares", "bad-identifier", "bad-composition"],
    )
    def test_validation_failure_modes(self, manifest, match):
        from interpretune.hub.manifest import ComponentManifestError, validate_component_manifest

        with pytest.raises(ComponentManifestError, match=match):
            validate_component_manifest(manifest)

    def test_valid_manifest_and_card_tag(self, tmp_path):
        from interpretune.hub.cards import generate_component_card
        from interpretune.hub.manifest import load_component_manifest

        component = _write_component(tmp_path, REGISTERS_DECLARED)
        manifest = load_component_manifest(component / "it_component.yaml")
        card = generate_component_card(manifest, "speediedan/fixture-adapter")
        assert "interpretune-adapters" in card.data.tags


class TestDynamicAdapterEnum:
    def test_adds_a_member_usable_as_an_adapter(self, restore_adapter_enum):
        from interpretune.adapters.registration import register_dynamic_adapter
        from interpretune.protocol import Adapter

        member = register_dynamic_adapter(FIXTURE_ADAPTER, source="org/fixture")
        assert Adapter[FIXTURE_ADAPTER] is member
        assert Adapter(FIXTURE_ADAPTER) is member
        assert getattr(Adapter, FIXTURE_ADAPTER) is member
        assert member.value == FIXTURE_ADAPTER

    def test_is_idempotent_for_one_source(self, restore_adapter_enum):
        from interpretune.adapters.registration import register_dynamic_adapter

        first = register_dynamic_adapter(FIXTURE_ADAPTER, source="org/fixture")
        assert register_dynamic_adapter(FIXTURE_ADAPTER, source="org/fixture") is first

    def test_refuses_to_shadow_a_builtin_adapter(self, restore_adapter_enum):
        from interpretune.adapters.registration import DynamicAdapterError, register_dynamic_adapter

        with pytest.raises(DynamicAdapterError, match="built-in"):
            register_dynamic_adapter("core", source="org/fixture")

    def test_refuses_a_name_another_component_took(self, restore_adapter_enum):
        from interpretune.adapters.registration import DynamicAdapterError, register_dynamic_adapter

        register_dynamic_adapter(FIXTURE_ADAPTER, source="org/fixture")
        with pytest.raises(DynamicAdapterError, match="already registered by"):
            register_dynamic_adapter(FIXTURE_ADAPTER, source="other/component")

    def test_composition_keys_canonicalize_with_a_dynamic_member(self, restore_adapter_enum):
        from interpretune.adapters.registration import CompositionRegistry, register_dynamic_adapter

        register_dynamic_adapter(FIXTURE_ADAPTER, source="org/fixture")
        registry = CompositionRegistry()
        assert registry.canonicalize_composition(("core", FIXTURE_ADAPTER)) == registry.canonicalize_composition(
            (FIXTURE_ADAPTER, "core")
        )


class TestHubAdapterLoad:
    @staticmethod
    def _publish(tmp_path, body, **kwargs):
        from interpretune.hub.components import local_publish

        component = _write_component(tmp_path, body, **kwargs)
        cache = tmp_path / "components"
        local_publish(component, "org/fixture-adapter", cache_dir=cache)
        return cache

    def test_refuses_without_trust_optin(self, tmp_path, monkeypatch, restore_adapter_enum):
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR, RemoteCodeNotTrustedError

        cache = self._publish(tmp_path, REGISTERS_DECLARED)
        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        with pytest.raises(RemoteCodeNotTrustedError, match="composes into the session MRO"):
            load_hub_adapter("org/fixture-adapter", cache_dir=cache, registry=CompositionRegistry())

    def test_loads_and_registers_the_declared_composition(self, tmp_path, monkeypatch, restore_adapter_enum):
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR
        from interpretune.protocol import Adapter

        cache = self._publish(tmp_path, REGISTERS_DECLARED)
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        registry = CompositionRegistry()
        members = load_hub_adapter("org/fixture-adapter", cache_dir=cache, registry=registry).members

        assert [m.name for m in members] == [FIXTURE_ADAPTER]
        key = ("module",) + registry.canonicalize_composition((Adapter.core, members[0]))
        assert len(registry.get(key)) == 1

    def test_manifest_only_snapshot_names_the_materializing_verb(self, tmp_path, monkeypatch, restore_adapter_enum):
        """The state a key-less pull used to leave: manifest cached, entrypoint never fetched. The refusal must
        name the verb that fixes it, not only the causes it resembles."""
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import AdapterComponentError, load_hub_adapter
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

        from interpretune.hub.components import local_publish

        # its own repo id: the entrypoint module name is revision-scoped, and the fixture's revision is
        # already imported by the other tests in this process, which would satisfy the sys.modules lookup
        cache = tmp_path / "components"
        local_publish(_write_component(tmp_path, REGISTERS_DECLARED), "org/manifest-only", cache_dir=cache)
        for entry in (cache / "models--org--manifest-only" / "snapshots").rglob("*.py"):
            entry.unlink()
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        with pytest.raises(
            AdapterComponentError, match=r"not present in the snapshot.*hub\.pull\('org/manifest-only'\)"
        ):
            load_hub_adapter("org/manifest-only", cache_dir=cache, registry=CompositionRegistry())

    def test_entrypoint_registering_nothing_is_an_error(self, tmp_path, monkeypatch, restore_adapter_enum):
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import AdapterComponentError, load_hub_adapter
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

        cache = self._publish(tmp_path, REGISTERS_NOTHING)
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        with pytest.raises(AdapterComponentError, match="registered no compositions"):
            load_hub_adapter("org/fixture-adapter", cache_dir=cache, registry=CompositionRegistry())

    def test_loaded_module_is_reachable_without_reconstructing_its_name(
        self, tmp_path, monkeypatch, restore_adapter_enum
    ):
        """An adapter's non-registered surface (a seam, capability helpers) must be reachable.

        The alternative is callers rebuilding the revision-scoped module name themselves, which hardcodes a naming
        scheme that belongs to the loader.
        """
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter, loaded_adapter_module
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

        cache = self._publish(tmp_path, REGISTERS_DECLARED)
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        load_hub_adapter("org/fixture-adapter", cache_dir=cache, registry=CompositionRegistry())

        module = loaded_adapter_module("org/fixture-adapter", cache_dir=cache)
        assert hasattr(module, "FixtureAdapter")

    def test_reaching_an_unloaded_module_raises_rather_than_importing_it(self, tmp_path):
        """The accessor resolves; it must never become a second, ungated execution path.

        Published under its own repo id rather than reusing the fixture's. ``local_publish`` is
        content-addressed and ``sys.modules`` persists for the whole session, so a component another
        test in this class already loaded is genuinely loaded here too -- and the assertion would pass
        or fail on test ORDER rather than on behaviour.
        """
        from interpretune.hub.components import local_publish
        from interpretune.hub.adapters import AdapterComponentError, loaded_adapter_module

        component = _write_component(tmp_path, REGISTERS_DECLARED, declares=("never_loaded_adapter",))
        cache = tmp_path / "components"
        local_publish(component, "org/never-loaded", cache_dir=cache)

        with pytest.raises(AdapterComponentError, match="has not been loaded"):
            loaded_adapter_module("org/never-loaded", cache_dir=cache)

    def test_component_without_the_adapters_kind_is_rejected(self, tmp_path, monkeypatch):
        import yaml

        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import AdapterComponentError, load_hub_adapter
        from interpretune.hub.components import local_publish
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

        component = tmp_path / "component"
        component.mkdir()
        (component / "it_component.yaml").write_text(
            yaml.safe_dump(
                {
                    "it_schema_version": 1,
                    "kinds": ["promptconfigs"],
                    "promptconfigs": {"entrypoint": "e.py", "definitions": {"X": {}}},
                }
            ),
            encoding="utf-8",
        )
        (component / "e.py").write_text("class X: pass\n", encoding="utf-8")
        cache = tmp_path / "components"
        local_publish(component, "org/not-an-adapter", cache_dir=cache)
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        with pytest.raises(AdapterComponentError, match="publishes no adapters"):
            load_hub_adapter("org/not-an-adapter", cache_dir=cache, registry=CompositionRegistry())


CONDITIONAL_COMPOSITIONS = REGISTERS_DECLARED


class TestSkippedCompositionsAreVisible:
    """A declared composition this environment cannot support is reported where an UNCONFIGURED consumer sees it,
    and returned, so the skip is never silent (#465)."""

    @staticmethod
    def _conditional_component(tmp_path):
        import yaml

        component = _write_component(tmp_path, REGISTERS_DECLARED)
        manifest = yaml.safe_load((component / "it_component.yaml").read_text())
        manifest["adapters"]["compositions"].append(
            {
                "component": "datamodule",
                "adapters": ["core", FIXTURE_ADAPTER],
                "requires": {"pip": ["a-package-nobody-has"]},
            }
        )
        (component / "it_component.yaml").write_text(yaml.safe_dump(manifest))
        return component

    def test_the_load_returns_the_skip_report(self, tmp_path, monkeypatch, restore_adapter_enum):
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import HubAdapterLoad, load_hub_adapter
        from interpretune.hub.components import local_publish
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR
        from interpretune.utils.logging import UnavailableCompositionWarning

        cache = tmp_path / "components"
        local_publish(self._conditional_component(tmp_path), "org/conditional", cache_dir=cache)
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        with pytest.warns(UnavailableCompositionWarning, match="1 of 2 declared composition"):
            loaded = load_hub_adapter("org/conditional", cache_dir=cache, registry=CompositionRegistry())
        assert isinstance(loaded, HubAdapterLoad)
        assert [m.name for m in loaded.members] == [FIXTURE_ADAPTER]
        assert len(loaded.skipped) == 1
        key, reason = loaded.skipped[0]
        assert FIXTURE_ADAPTER in key and "datamodule" in key and "a-package-nobody-has" in reason
        with pytest.raises(TypeError):
            iter(loaded)  # the record is not a list; an old-style caller fails loudly rather than silently

    def test_the_report_reaches_a_consumer_with_default_logging(self, tmp_path):
        """Measured against what a plain script receives: no logging configured, no warnings filter, stderr.

        `caplog` at INFO would capture the record and pass against the bug; a subprocess is the only honest
        instrument for "an unconfigured consumer sees it".
        """
        import os
        import subprocess
        import sys

        from interpretune.hub.components import local_publish
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

        cache = tmp_path / "components"
        local_publish(self._conditional_component(tmp_path), "org/conditional", cache_dir=cache)
        script = (
            "from pathlib import Path\n"
            "from interpretune.adapters.registration import CompositionRegistry\n"
            "from interpretune.hub.adapters import load_hub_adapter\n"
            f"load_hub_adapter('org/conditional', cache_dir=Path({str(cache)!r}), registry=CompositionRegistry())\n"
            "print('LOAD COMPLETE')\n"
        )
        env = {**os.environ, IT_TRUST_REMOTE_CODE_ENV_VAR: "1"}
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180, env=env)
        assert result.returncode == 0, result.stderr[-2000:]
        assert "LOAD COMPLETE" in result.stdout
        assert (
            "a-package-nobody-has" in result.stderr and "unavailable here rather than nonexistent" in result.stderr
        ), "the skip report did not reach a consumer with default logging:\n" + result.stderr[-2000:]
