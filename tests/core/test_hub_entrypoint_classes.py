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

ENTRYPOINT_SRC = '''\
class FixtureWidget:
    """Standalone widget: no imports, so the snapshot import cannot drag the tree in."""

    def __init__(self, tag: str = "w") -> None:
        self.tag = tag


class FixtureGadget:
    """Second class, proving attribute selection inside the snapshot module."""

    def __init__(self, tag: str = "g") -> None:
        self.tag = tag
'''

MODULE_CONFIG_KEY = "fixture.tiny.core"


def _write_fixture_component(root: Path) -> Path:
    """A minimal publishable tree: one module config, one datamodule entry, one entrypoint file."""
    (root / "configs").mkdir(parents=True)
    (root / "fixture_entry.py").write_text(ENTRYPOINT_SRC, encoding="utf-8")
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


def test_snapshot_execution_needs_the_trust_opt_in(entrypoint_cache, monkeypatch):
    """Proof (v): the gate fires before exec, with this path's own wording pinned."""
    from interpretune.hub.entrypoints import instantiate_hub_aware_class
    from interpretune.hub.trust import RemoteCodeNotTrustedError

    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "0")
    with pytest.raises(RemoteCodeNotTrustedError, match="component entrypoint"):
        instantiate_hub_aware_class(
            {"class_path": "fixture_entry.FixtureWidget"}, import_only=True, cache_dir=entrypoint_cache
        )
