"""Snapshot-entrypoint class resolution for module/datamodule payloads (#497 groundwork).

A payload ``class_path`` normally imports from the environment. Once an experiment moves out of
the tree, its payloads address the component's own entrypoint file instead, which is importable
only from the cached snapshot. These tests prove the hub-aware variant against a synthetic
fixture component (local-published, so hermetic and offline): snapshot resolution, in-tree
precedence, named refusal, missing-file guidance, and the trust gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ENTRYPOINT_TEMPLATE = '''\
class FixtureWidget:
    """Standalone widget: no imports, so the snapshot import cannot drag the tree in."""

    def __init__(self, tag: str = {tag!r}) -> None:
        self.tag = tag


class FixtureGadget:
    """Second class, proving attribute selection inside the snapshot module."""

    def __init__(self, tag: str = "g") -> None:
        self.tag = tag
'''

MODULE_CONFIG_KEY = "fixture.tiny.core"


def _write_fixture_component(root: Path, tag: str = "w") -> Path:
    """A minimal publishable tree: one module config, one datamodule entry, one entrypoint file."""
    (root / "configs").mkdir(parents=True)
    (root / "fixture_entry.py").write_text(ENTRYPOINT_TEMPLATE.format(tag=tag), encoding="utf-8")
    manifest = {
        "it_schema_version": 1,
        "kinds": ["module", "datamodule"],
        "module": {
            "entrypoint": "fixture_entry.py",
            "task": {"name": "fixture", "description": "synthetic entrypoint fixture"},
            "configs": {MODULE_CONFIG_KEY: f"configs/{MODULE_CONFIG_KEY}.yaml"},
        },
        "datamodules": {
            "fixture_dm": {"entrypoint": "fixture_entry.py", "config": "configs/dm.yaml"},
        },
    }
    (root / "it_component.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    config_body = {
        "task_variant": "fixture",
        "model": "tiny",
        "composition": ["core"],
        "module_cfg": {"class_path": "fixture_entry.FixtureWidget"},
    }
    (root / "configs" / f"{MODULE_CONFIG_KEY}.yaml").write_text(yaml.safe_dump(config_body), encoding="utf-8")
    (root / "configs" / "dm.yaml").write_text(yaml.safe_dump({}), encoding="utf-8")
    return root


@pytest.fixture()
def entrypoint_cache(tmp_path):
    """The fixture component local-published into an isolated components cache."""
    from interpretune.hub.components import local_publish

    component = _write_fixture_component(tmp_path / "component")
    cache = tmp_path / "cache"
    local_publish(component, "someorg/fixture", entrypoint_src=component / "fixture_entry.py", cache_dir=cache)
    return cache


@pytest.fixture()
def unexecuted_entrypoint_cache(tmp_path):
    """Same shape under a distinct revision, so no earlier test has executed its module.

    Synthetic module names incorporate the revision: reusing the default tag would resolve to the
    already-executed module and pass the trust gate without challenging it (the behavior the
    promptconfigs suite pins deliberately, which is exactly what this control must not rely on).
    """
    from interpretune.hub.components import local_publish

    component = _write_fixture_component(tmp_path / "component", tag="unexecuted")
    cache = tmp_path / "cache"
    local_publish(component, "someorg/fixture", entrypoint_src=component / "fixture_entry.py", cache_dir=cache)
    return cache


def test_entrypoint_stem_class_resolves_from_the_snapshot(entrypoint_cache):
    """Proof (i): a stem-addressed class comes from the snapshot, not the environment."""
    import inspect

    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    cls = instantiate_hub_aware_class(
        {"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=entrypoint_cache
    )
    assert cls.__name__ == "FixtureWidget"
    assert cls.__module__.startswith("it_hub_components.someorg__fixture.")
    assert Path(inspect.getfile(cls)).is_relative_to(entrypoint_cache)
    assert cls(tag="t").tag == "t"


def test_environment_import_wins_when_both_resolve(entrypoint_cache, monkeypatch):
    """Proof (ii): in-tree precedence is deterministic; the snapshot is the fallback, not a shadow."""
    import sys
    import types

    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    fake = types.ModuleType("fixture_entry")
    fake.FixtureWidget = type("FixtureWidget", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fixture_entry", fake)
    cls = instantiate_hub_aware_class(
        {"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=entrypoint_cache
    )
    assert cls is fake.FixtureWidget


def test_undeclared_stem_refuses_naming_both_attempts(entrypoint_cache):
    """Proof (iii): a miss is a named refusal, not a bare ImportError and not a guess."""
    from interpretune.hub.entrypoints import instantiate_hub_aware_class
    from interpretune.utils.exceptions import MisconfigurationException

    with pytest.raises(MisconfigurationException, match="no cached component"):
        instantiate_hub_aware_class({"class_path": "nope_missing.Widget"}, import_only=True, cache_dir=entrypoint_cache)


def test_multi_segment_stems_keep_the_existing_failure(entrypoint_cache):
    """Snapshot stems are top-level modules; a dotted path was never consultable, so its error stands."""
    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    with pytest.raises(ModuleNotFoundError):
        instantiate_hub_aware_class(
            {"class_path": "nope_missing.sub.Widget"}, import_only=True, cache_dir=entrypoint_cache
        )


def test_declared_but_absent_entrypoint_names_the_pull(entrypoint_cache):
    """Proof (iv): a manifest-only snapshot fails with the fetch that fixes it."""
    import sys

    from interpretune.hub.components import resolve_component_manifest
    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    _, snapshot, revision = resolve_component_manifest("someorg/fixture", cache_dir=entrypoint_cache)
    (snapshot / "fixture_entry.py").unlink()
    module_name = f"it_hub_components.someorg__fixture.{revision}"
    sys.modules.pop(module_name, None)
    with pytest.raises(FileNotFoundError, match="materialize the entrypoint"):
        instantiate_hub_aware_class(
            {"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=entrypoint_cache
        )


def test_declared_payloads_include_entrypoints_and_datamodule_configs():
    """Key-less pull materializes what cache-only loaders read whole: entrypoints plus dm configs.

    Module configs stay per-key (pinned by the hookmaps test); entrypoints and datamodule
    standalone configs cannot, since snapshot class resolution and datamodule resolution read
    them whole from the snapshot.
    """
    from interpretune.hub.components import declared_component_payloads

    manifest = {
        "kinds": ["module", "datamodule"],
        "module": {"entrypoint": "entry.py", "configs": {"k": "configs/k.yaml"}},
        "datamodules": {"dm": {"entrypoint": "entry.py", "config": "configs/dm.yaml"}},
    }
    assert declared_component_payloads(manifest) == ["entry.py", "configs/dm.yaml"]


def test_explicit_cache_dir_does_not_bind_the_default_cache(tmp_path, monkeypatch):
    """Cache isolation: an explicit cache scopes resolution; a same-stem default never leaks in."""
    from interpretune.hub import components
    from interpretune.hub.components import local_publish
    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    real = tmp_path / "real" / "cache"
    local_publish(
        _write_fixture_component(tmp_path / "real" / "component", tag="real"),
        "someorg/fixture",
        entrypoint_src=tmp_path / "real" / "component" / "fixture_entry.py",
        cache_dir=real,
    )
    decoy = tmp_path / "decoy" / "cache"
    local_publish(
        _write_fixture_component(tmp_path / "decoy" / "component", tag="decoy"),
        "someorg/fixture",
        entrypoint_src=tmp_path / "decoy" / "component" / "fixture_entry.py",
        cache_dir=decoy,
    )
    monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", decoy)
    cls = instantiate_hub_aware_class({"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=real)
    assert cls().tag == "real"


def test_default_cache_honors_the_established_patch_pattern(tmp_path, monkeypatch):
    """Without an explicit cache, the default binding applies (including the test-suite patch)."""
    from interpretune.hub import components
    from interpretune.hub.components import local_publish
    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    cache = tmp_path / "cache"
    local_publish(
        _write_fixture_component(tmp_path / "component"),
        "someorg/fixture",
        entrypoint_src=tmp_path / "component" / "fixture_entry.py",
        cache_dir=cache,
    )
    monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", cache)
    cls = instantiate_hub_aware_class({"class_path": "fixture_entry.FixtureWidget"}, import_only=True)
    assert cls().tag == "w"


def test_snapshot_functions_locate_by_reference(tmp_path):
    """Parent packages are bound, so serializers resolve Hub functions instead of pickling by value.

    Stuffing only the full dotted name into ``sys.modules`` leaves parent-attribute traversal
    (dill's function location, ``mock.patch`` string targets) unable to reach the module, and
    serializers fall back to pickling Hub-defined functions by value with their whole globals.
    Uses its own tag: synthetic names are content hashes, so a shared tag would resolve to
    another test's already-imported module instead of exercising this path.
    """
    import sys

    from dill._dill import _locate_function

    from interpretune.hub.components import local_publish
    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    component = _write_fixture_component(tmp_path / "component", tag="locate")
    cache = tmp_path / "cache"
    local_publish(component, "someorg/fixture", entrypoint_src=component / "fixture_entry.py", cache_dir=cache)
    cls = instantiate_hub_aware_class({"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=cache)
    assert cls.__module__.startswith("it_hub_components.")
    parent_name, _, _ = cls.__module__.rpartition(".")
    assert getattr(sys.modules[parent_name.rpartition(".")[0]], parent_name.rpartition(".")[2]) is not None
    assert _locate_function(cls.__init__, None) is True


def test_snapshot_execution_needs_the_trust_opt_in(unexecuted_entrypoint_cache, monkeypatch):
    """Proof (v): the gate fires before exec, with this path's own wording pinned."""
    from interpretune.hub.entrypoints import instantiate_hub_aware_class
    from interpretune.hub.trust import RemoteCodeNotTrustedError

    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "0")
    with pytest.raises(RemoteCodeNotTrustedError, match="component entrypoint"):
        instantiate_hub_aware_class(
            {"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=unexecuted_entrypoint_cache
        )


def test_finder_reimports_an_evicted_snapshot_module(tmp_path, monkeypatch):
    """A fresh process (spawn worker) restores the module from disk: evict, reimport, identical file."""
    import importlib
    import inspect
    import sys

    from interpretune.hub import components
    from interpretune.hub.components import local_publish
    from interpretune.hub.entrypoints import instantiate_hub_aware_class

    # Unique tag: synthetic names are content hashes, so sharing the default tag with other
    # tests would resolve to their already-imported module instead of exercising the finder.
    component = _write_fixture_component(tmp_path / "component", tag="reimport")
    cache = tmp_path / "cache"
    local_publish(component, "someorg/fixture", entrypoint_src=component / "fixture_entry.py", cache_dir=cache)
    monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", cache)
    cls = instantiate_hub_aware_class({"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=cache)
    name, expected_file = cls.__module__, inspect.getfile(cls)
    assert name.startswith("it_hub_components.")
    evicted = sys.modules.pop(name)
    try:
        restored = importlib.import_module(name)
    finally:
        sys.modules.setdefault(name, evicted)
    assert inspect.getfile(restored) == expected_file
    assert restored.FixtureWidget.__name__ == "FixtureWidget"


def test_finder_ignores_ordinary_and_unknown_names(entrypoint_cache):
    """Only it_hub_components stems consult the cache; unknown stems miss without network."""
    from interpretune.hub.entrypoints import _HubSnapshotFinder

    finder = _HubSnapshotFinder()
    assert finder.find_spec("os", None) is None
    assert finder.find_spec("nope.missing", None) is None
    assert finder.find_spec("it_hub_components.nope__none.abc123", None) is None


def test_finder_install_is_idempotent():
    """Repeated installs add no duplicate finders."""
    import sys

    from interpretune.hub.entrypoints import install_hub_module_finder

    install_hub_module_finder()
    install_hub_module_finder()
    assert sum(type(finder).__name__ == "_HubSnapshotFinder" for finder in sys.meta_path) == 1
