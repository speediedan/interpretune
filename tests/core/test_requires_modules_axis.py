"""#472: a composition can declare the modules it imports, so a distribution present at a checkout that lacks the
module is skipped-and-reported rather than registered-and-broken."""

from __future__ import annotations

import pytest

from interpretune.hub.manifest import ComponentManifestError, validate_component_manifest
from interpretune.utils.requirements import requirement_status


class TestModulesAxis:
    def test_a_present_module_is_met_and_an_absent_one_names_itself(self):
        assert requirement_status({"modules": ["json", "importlib.util"]}, "t") == []
        unmet = requirement_status({"modules": ["json", "no_such_pkg_xyz.sub"]}, "t")
        assert [u.kind for u in unmet] == ["modules"] and "no_such_pkg_xyz.sub" in unmet[0].message
        assert "not present" in unmet[0].message

    def test_a_missing_submodule_of_a_present_package_is_unmet(self):
        """The motivating case: the distribution is there, one module of it is not."""
        unmet = requirement_status({"pip": ["packaging"], "modules": ["packaging.no_such_module_here"]}, "t")
        assert [u.kind for u in unmet] == ["modules"]

    def test_evaluation_order_keeps_modules_before_pip(self):
        unmet = requirement_status({"pip": ["a-package-nobody-has"], "modules": ["no_such_pkg_xyz"]}, "t")
        assert [u.kind for u in unmet] == ["modules", "pip"]

    def test_nothing_is_imported_to_answer(self, tmp_path, monkeypatch):
        """find_spec, not import: a module with an import-time side effect is located without running it."""
        (tmp_path / "side_effect_probe.py").write_text("raise RuntimeError('imported')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        assert requirement_status({"modules": ["side_effect_probe"]}, "t") == []


class TestManifestValidationOfTheAxis:
    def _manifest(self, requires, *, per_composition=False):
        m = {
            "it_schema_version": 1,
            "kinds": ["adapters"],
            "adapters": {
                "entrypoint": "e.py",
                "declares": ["x"],
                "compositions": [{"component": "module", "adapters": ["core", "x"]}],
            },
        }
        if per_composition:
            m["adapters"]["compositions"][0]["requires"] = requires
        else:
            m["requires"] = requires
        return m

    @pytest.mark.parametrize("per_composition", [False, True])
    def test_dotted_importable_names_are_accepted(self, per_composition):
        validate_component_manifest(
            self._manifest(
                {"pip": ["circuit-tracer"], "modules": ["circuit_tracer.replacement_model_interp_engine"]},
                per_composition=per_composition,
            ),
            source="t",
        )

    @pytest.mark.parametrize("per_composition", [False, True])
    def test_a_non_identifier_entry_is_refused(self, per_composition):
        with pytest.raises(ComponentManifestError, match="dotted importable names"):
            validate_component_manifest(
                self._manifest({"modules": ["circuit-tracer"]}, per_composition=per_composition), source="t"
            )

    def test_an_unknown_axis_is_ignored_for_forward_compatibility(self):
        """A manifest written for a newer interpretune may carry an axis this one does not evaluate."""
        validate_component_manifest(self._manifest({"module": ["x"], "modules": ["json"]}), source="t")
        assert requirement_status({"module": ["x"]}, "t") == []

    def test_a_list_axis_must_be_a_list_of_strings(self):
        with pytest.raises(ComponentManifestError, match="must be a list of non-empty strings"):
            validate_component_manifest(self._manifest({"modules": "circuit_tracer"}), source="t")


class TestACompositionSkipsRatherThanBreaks:
    def test_the_composition_is_partitioned_out_with_the_module_named(self):
        from interpretune.hub.adapters import _partition_declared_compositions

        manifest = {
            "adapters": {
                "compositions": [
                    {"component": "module", "adapters": ["core", "x"]},
                    {
                        "component": "module",
                        "adapters": ["core", "x", "circuit_tracer"],
                        "requires": {"pip": ["packaging"], "modules": ["packaging.no_such_module_here"]},
                    },
                ]
            }
        }
        satisfiable, unsupported = _partition_declared_compositions(manifest, source="t")
        assert len(satisfiable) == 1 and len(unsupported) == 1
        key, reason = unsupported[0]
        assert "circuit_tracer" in key and "packaging.no_such_module_here" in reason
