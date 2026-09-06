"""#454: the unstated rails invariants a component author cannot discover, now stated and checked."""

from __future__ import annotations

import pytest
import yaml

from interpretune.hub.manifest import UnpairedCompositionWarning, validate_component_manifest
from tests.core.test_hub_adapters import FIXTURE_ADAPTER, _write_component
from tests.core.test_hub_adapters import restore_adapter_enum  # noqa: F401  (fixture, used by name)


REGISTERS_BOTH = '''
    class FixtureAdapterModule:
        """A stand-in for a real adapter mixin."""

    class FixtureAdapter:
        @classmethod
        def register_adapter_ctx(cls, adapter_ctx_registry) -> None:
            for component_key in ("module", "datamodule"):
                adapter_ctx_registry.register(
                    lead_adapter="fixture_adapter",
                    component_key=component_key,
                    adapter_combination=("core", "fixture_adapter"),
                    composition_classes=(FixtureAdapterModule,),
                    description="fixture adapter registering both slots",
                )
'''


def _manifest(compositions):
    return {
        "it_schema_version": 1,
        "kinds": ["adapters"],
        "adapters": {"entrypoint": "e.py", "declares": ["x"], "compositions": compositions},
    }


class TestPairingIsWarnedAtValidation:
    def test_a_module_without_its_datamodule_counterpart_is_named(self):
        with pytest.warns(UnpairedCompositionWarning, match=r"core\+x declares only \['module'\].*datamodule path"):
            validate_component_manifest(_manifest([{"component": "module", "adapters": ["core", "x"]}]), source="t")

    def test_a_paired_set_is_silent_and_order_insensitive(self, recwarn):
        validate_component_manifest(
            _manifest(
                [
                    {"component": "module", "adapters": ["core", "x"]},
                    {"component": "datamodule", "adapters": ["x", "core"]},
                ]
            ),
            source="t",
        )
        assert not [w for w in recwarn if issubclass(w.category, UnpairedCompositionWarning)]

    def test_it_warns_rather_than_refuses(self):
        """A caller who only builds modules is coherent; the pairing is a habit of every bundled adapter, not a
        rule."""
        with pytest.warns(UnpairedCompositionWarning):
            manifest = validate_component_manifest(
                _manifest([{"component": "datamodule", "adapters": ["core", "x"]}]), source="t"
            )
        assert manifest["adapters"]["compositions"]


class TestUnderstatingIsRefusedAtLoad:
    def test_a_registration_the_manifest_never_declared_is_refused(self, tmp_path, monkeypatch, restore_adapter_enum):  # noqa: F811
        """The card renders the DECLARED list; an undeclared registration would make it claim less than the
        truth."""
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import AdapterComponentError, load_hub_adapter
        from interpretune.hub.components import local_publish
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

        component = _write_component(tmp_path, REGISTERS_BOTH)
        manifest_path = component / "it_component.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        # the entrypoint registers module AND datamodule for (core, fixture_adapter); declare only one of them
        manifest["adapters"]["compositions"] = [{"component": "module", "adapters": ["core", FIXTURE_ADAPTER]}]
        manifest_path.write_text(yaml.safe_dump(manifest))
        cache = tmp_path / "components"
        with pytest.warns(UnpairedCompositionWarning):
            local_publish(component, "org/understated", cache_dir=cache)
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        with pytest.warns(UnpairedCompositionWarning):
            with pytest.raises(
                AdapterComponentError, match="registered composition\\(s\\) the manifest does not declare"
            ):
                load_hub_adapter("org/understated", cache_dir=cache, registry=CompositionRegistry())


class TestTheModelHandleContractIsStated:
    def test_the_backend_protocol_says_what_model_is(self):
        from interpretune.analysis.backends.protocols import ModelBackendCore

        doc = " ".join((ModelBackendCore.__doc__ or "").split())  # wrapping is the formatter's business
        assert "module.model" in doc and "must carry its own handle" in doc
