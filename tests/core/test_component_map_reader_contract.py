"""The component-map reader contract (the evolution policy's reader half).

Every case here is a decision, pinned so it stays one: the readable window, what an unknown key of each class does, why
an unknown kind is refused, and that every version inside the window has a frozen document.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from interpretune.analysis.points.component_map import (
    COMPONENT_MAP_SCHEMA_MIN_READABLE,
    COMPONENT_MAP_SCHEMA_VERSION,
    ComponentMapDeprecationWarning,
    component_map_for,
    load_component_map_file,
    register,
)

FIXTURES = Path(__file__).parent / "fixtures"
ROWS = {
    "embed": {"module": "transformer.wte", "kind": "embed"},
    "blocks.{i}": {"module": "transformer.h.{i}", "kind": "block"},
    "blocks.{i}.attn": {"module": "transformer.h.{i}.attn", "kind": "attn"},
    "unembed": {"module": "lm_head", "kind": "unembed"},
}


def _write(tmp_path: Path, doc: dict, name: str = "map.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _doc(**overrides) -> dict:
    doc = {"schema_version": COMPONENT_MAP_SCHEMA_VERSION, "architecture": "ToyLMHeadModel", "components": dict(ROWS)}
    doc.update(overrides)
    return doc


class TestReadableWindow:
    def test_the_current_version_reads(self, tmp_path):
        assert load_component_map_file(_write(tmp_path, _doc())).schema_version == COMPONENT_MAP_SCHEMA_VERSION

    def test_a_newer_version_is_refused_and_says_upgrade(self, tmp_path):
        """Before this rule the reader accepted ANY integer, so a schema-7 document loaded as if it were schema
        1."""
        path = _write(tmp_path, _doc(schema_version=COMPONENT_MAP_SCHEMA_VERSION + 1))
        with pytest.raises(ValueError, match="written by a newer interpretune.*Upgrade interpretune"):
            load_component_map_file(path)

    def test_an_older_than_floor_version_is_refused_and_says_republish(self, tmp_path):
        path = _write(tmp_path, _doc(schema_version=COMPONENT_MAP_SCHEMA_MIN_READABLE - 1))
        with pytest.raises(ValueError, match="older than the minimum readable schema.*Re-publish"):
            load_component_map_file(path)

    @pytest.mark.parametrize("bad", [None, "1", 1.0, True])
    def test_a_malformed_version_is_told_apart_from_an_unsupported_one(self, tmp_path, bad):
        doc = _doc()
        if bad is None:
            del doc["schema_version"]
        else:
            doc["schema_version"] = bad
        with pytest.raises(ValueError, match="lacks an integer `schema_version`"):
            load_component_map_file(_write(tmp_path, doc))

    @pytest.mark.parametrize("version", range(COMPONENT_MAP_SCHEMA_MIN_READABLE, COMPONENT_MAP_SCHEMA_VERSION + 1))
    def test_every_readable_version_has_a_frozen_fixture(self, version):
        """Derived from the window, so a bump adds a failing case until its fixture exists (the artifact envelope's
        own mechanism for making deferred work fire at the bump rather than after it)."""
        fixture = FIXTURES / f"component_map_schema{version}.yaml"
        assert fixture.is_file(), (
            f"schema {version} is inside the readable window [{COMPONENT_MAP_SCHEMA_MIN_READABLE}, "
            f"{COMPONENT_MAP_SCHEMA_VERSION}] but has no frozen document at tests/core/fixtures/{fixture.name}. "
            "Freeze one copied verbatim from a bundled document written at that version, in the same change."
        )
        cmap = load_component_map_file(fixture)
        assert cmap.schema_version == version
        assert cmap.kind_of("blocks.{i}.attn", None) == "attn"


class TestUnknownKeys:
    """Two classes of key, two rules.

    Descriptive keys are ignored; the applicability class arrives only with a schema bump, which the window refuses
    first, so within the window ignoring is safe.
    """

    def test_an_unknown_top_level_key_is_ignored(self, tmp_path):
        cmap = load_component_map_file(_write(tmp_path, _doc(a_field_from_a_later_release={"anything": [1, 2]})))
        assert cmap.architecture == "ToyLMHeadModel"

    def test_an_unknown_row_key_is_ignored(self, tmp_path):
        rows = dict(ROWS)
        rows["blocks.{i}.attn"] = {**rows["blocks.{i}.attn"], "annotation_from_later": "tolerated"}
        cmap = load_component_map_file(_write(tmp_path, _doc(components=rows)))
        assert cmap.kind_of("blocks.{i}.attn", None) == "attn"

    def test_an_unknown_property_is_ignored_and_a_mistyped_known_one_is_refused(self, tmp_path):
        cmap = load_component_map_file(
            _write(tmp_path, _doc(properties={"sandwich_norms": False, "later_property": 3}))
        )
        assert cmap.properties["later_property"] == 3 and cmap.sandwich_norms is False
        with pytest.raises(ValueError, match="property 'sandwich_norms' must be bool"):
            load_component_map_file(_write(tmp_path, _doc(properties={"sandwich_norms": "yes"})))

    def test_a_schema_1_document_spells_properties_as_facts_and_still_reads(self, tmp_path):
        doc = _doc(schema_version=1, facts={"sandwich_norms": True})
        cmap = load_component_map_file(_write(tmp_path, doc))
        assert cmap.sandwich_norms is True and cmap.facts is cmap.properties

    def test_a_schema_2_document_using_the_old_spelling_is_told_the_new_one(self, tmp_path):
        with pytest.raises(ValueError, match="renamed to `properties`"):
            load_component_map_file(_write(tmp_path, _doc(facts={"sandwich_norms": True})))

    def test_an_unknown_kind_is_refused_with_the_reason(self, tmp_path):
        rows = dict(ROWS)
        rows["blocks.{i}.mixer"] = {"module": "backbone.layers.{i}.mixer", "kind": "mixer"}
        with pytest.raises(ValueError, match="unknown component kind 'mixer'.*refused rather than defaulted"):
            load_component_map_file(_write(tmp_path, _doc(components=rows)))


@pytest.fixture
def isolated_registry():
    """Snapshot the process-wide map registry so a test's registration does not leak into the next."""
    from interpretune.analysis.points import component_map as cm

    cm._load_bundled()
    before = dict(cm._REGISTRY)
    yield
    cm._REGISTRY.clear()
    cm._REGISTRY.update(before)


class TestRetirement:
    def test_a_retired_map_loads_with_a_warning_naming_the_replacement(self, tmp_path):
        path = _write(tmp_path, _doc(deprecated_since="0.9.0", replacement="ToyLMHeadModelV2"))
        with pytest.warns(ComponentMapDeprecationWarning, match="deprecated since 0.9.0; use 'ToyLMHeadModelV2'"):
            cmap = load_component_map_file(path)
        assert cmap.deprecated_since == "0.9.0"

    def test_strict_lookup_refuses_a_retired_map(self, tmp_path, isolated_registry):
        path = _write(tmp_path, _doc(architecture="RetiredToyModel", deprecated_since="0.9.0", replacement="Toy2"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComponentMapDeprecationWarning)
            register(load_component_map_file(path))
        assert component_map_for("RetiredToyModel").deprecated_since == "0.9.0"
        with pytest.raises(ValueError, match="deprecated since 0.9.0; use 'Toy2' \\(refused in strict mode\\)"):
            component_map_for("RetiredToyModel", strict=True)

    def test_no_bundled_map_is_retired(self):
        from interpretune.analysis.points.component_map import known_architectures

        assert all(component_map_for(a).deprecated_since is None for a in known_architectures())
