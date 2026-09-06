"""#435 / #436: the published artifact is the manifest's allowlist plus declared extras, and the Hub tree is made
to MATCH the staged tree rather than merely receive it.

Before this, the only way to publish a collection's tests was a hand-push, which carried whatever else sat in the
working directory (a `.pytest_cache/` tree reached the first published collection that way), and a renamed entrypoint
left its old name live beside the new one because an upload never deletes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from interpretune.hub.manifest import ComponentManifestError, validate_component_manifest


def _manifest(**overrides):
    base = {"it_schema_version": 1, "kinds": ["ops"], "ops": {"files": ["ops.yaml"]}}
    base.update(overrides)
    return base


def _component(tmp_path: Path, extra_files) -> Path:
    root = tmp_path / "component"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "__pycache__").mkdir()
    (root / ".pytest_cache").mkdir()
    (root / "it_component.yaml").write_text(yaml.safe_dump(_manifest(extra_files=extra_files)))
    (root / "ops.yaml").write_text("ops: {}\n")
    (root / "tests" / "test_ops.py").write_text("def test_ok(): pass\n")
    (root / "tests" / "__pycache__" / "test_ops.cpython-313.pyc").write_bytes(b"\x00")
    (root / ".pytest_cache" / "CACHEDIR.TAG").write_text("tag\n")
    (root / "NOTES.md").write_text("notes\n")
    return root


class TestExtraFilesDeclaration:
    def test_shape_is_a_list_of_relative_paths(self):
        with pytest.raises(ComponentManifestError, match="`extra_files` must be a list"):
            validate_component_manifest(_manifest(extra_files="tests"), source="t")
        for bad in ("/etc/passwd", "../sibling", "it_component.yaml"):
            with pytest.raises(ComponentManifestError, match="relative path inside the component directory"):
                validate_component_manifest(_manifest(extra_files=[bad]), source="t")
        validate_component_manifest(_manifest(extra_files=["tests", "NOTES.md"]), source="t")

    def test_declared_extras_are_staged_and_caches_are_not(self, tmp_path):
        from interpretune.hub.publish import build_component_tree

        root = _component(tmp_path, ["tests", "NOTES.md"])
        out = tmp_path / "out"
        build_component_tree(root, out)
        built = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        assert built == ["NOTES.md", "it_component.yaml", "ops.yaml", "tests/test_ops.py"]

    def test_undeclared_files_are_not_published(self, tmp_path):
        """The allowlist stays strict: an undeclared file in the working directory never reaches the tree."""
        from interpretune.hub.publish import build_component_tree

        root = _component(tmp_path, ["tests"])
        out = tmp_path / "out"
        build_component_tree(root, out)
        assert not (out / "NOTES.md").exists() and not (out / ".pytest_cache").exists()

    def test_a_missing_or_escaping_extra_is_refused(self, tmp_path):
        from interpretune.hub.publish import build_component_tree

        root = _component(tmp_path, ["absent.md"])
        with pytest.raises(FileNotFoundError, match="extra file 'absent.md'"):
            build_component_tree(root, tmp_path / "out")
        # the manifest validator refuses `..`; a symlink is the other way out, and the builder resolves it
        root = _component(tmp_path / "second", ["escape"])
        (root / "escape").symlink_to(tmp_path)
        with pytest.raises(ValueError, match="resolves outside"):
            build_component_tree(root, tmp_path / "out2")

    def test_the_card_lists_the_extras(self):
        from interpretune.hub.cards import generate_component_card

        card = str(generate_component_card(_manifest(extra_files=["tests"]), "org/coll"))
        assert "## Supplementary files" in card and "`tests`" in card and "arrived out of band" in card

    def test_local_publish_carries_the_extras(self, tmp_path):
        from interpretune.hub.components import local_publish, resolve_component_manifest

        root = _component(tmp_path, ["tests"])
        cache = tmp_path / "cache"
        local_publish(root, "org/coll", cache_dir=cache)
        _, snapshot, _ = resolve_component_manifest("org/coll", cache_dir=cache)
        assert (snapshot / "tests" / "test_ops.py").is_file()


class TestPublishedTreeMatchesStaged:
    @staticmethod
    def _manager(remote_files, exists=True):
        from interpretune.hub.manager import COMPONENT_KIND, ITHubResourceManager

        api = Mock()
        commit = Mock()
        commit.oid = "new-sha"
        api.upload_folder.return_value = commit
        if exists:
            api.repo_info.return_value = Mock(sha="old-sha")
        else:
            from tests.core.test_hub_manager import _create_mock_repository_not_found_error

            api.repo_info.side_effect = _create_mock_repository_not_found_error("absent")
        api.list_repo_files.return_value = list(remote_files)
        manager = ITHubResourceManager(kind=COMPONENT_KIND)
        manager.api = api
        return manager, api

    def test_stale_remote_files_are_reported_and_removed(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "it_component.yaml").write_text("x")
        (staged / "adapter.py").write_text("x")
        remote = [".gitattributes", "it_component.yaml", "interp_engine_adapter.py", ".pytest_cache/CACHEDIR.TAG"]
        manager, api = self._manager(remote)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert manager.upload(staged, "org/adapter", match_staged=True) == "new-sha"
        kwargs = api.upload_folder.call_args.kwargs
        assert kwargs["delete_patterns"] == [".pytest_cache/CACHEDIR.TAG", "interp_engine_adapter.py"]
        assert ".gitattributes" not in kwargs["delete_patterns"], "Hub-managed files are never stale"
        text = " ".join(str(w.message) for w in caught)
        assert "interp_engine_adapter.py" in text and ".pytest_cache/CACHEDIR.TAG" in text
        assert "matches the staged one" in text

    def test_a_matching_remote_deletes_nothing_and_warns_nothing(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "it_component.yaml").write_text("x")
        manager, api = self._manager([".gitattributes", "it_component.yaml"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager.upload(staged, "org/adapter", match_staged=True)
        assert api.upload_folder.call_args.kwargs["delete_patterns"] is None
        assert not [w for w in caught if "removes" in str(w.message)]

    def test_a_new_repo_needs_no_listing(self, tmp_path):
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "it_component.yaml").write_text("x")
        manager, api = self._manager([], exists=False)
        manager.upload(staged, "org/adapter", match_staged=True)
        api.list_repo_files.assert_not_called()
        api.create_repo.assert_called_once()

    def test_exact_paths_are_escaped_as_patterns(self):
        from interpretune.hub.manager import fnmatch_escape
        import fnmatch

        weird = "tests/[old]_ops?.py"
        assert fnmatch.fnmatch(weird, fnmatch_escape(weird))
        assert not fnmatch.fnmatch("tests/o_opsx.py", fnmatch_escape(weird))

    def test_caches_are_never_uploaded(self, tmp_path):
        from interpretune.hub.manager import UPLOAD_IGNORE_PATTERNS, HubAnalysisOpManager

        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "ops.yaml").write_text("x")
        _, api = self._manager([], exists=False)
        manager = HubAnalysisOpManager(cache_dir=tmp_path / "cache")
        manager.api = api
        manager.upload_ops(staged, "org/ops")
        assert api.upload_folder.call_args.kwargs["ignore_patterns"] == list(UPLOAD_IGNORE_PATTERNS)
        assert ".pytest_cache/*" in UPLOAD_IGNORE_PATTERNS and "__pycache__/*" in UPLOAD_IGNORE_PATTERNS

    def test_publish_component_asks_for_a_matching_tree(self, tmp_path, monkeypatch):
        from interpretune.hub import publish as publish_mod

        root = _component(tmp_path, ["tests"])
        calls = {}

        class _Manager:
            def __init__(self, kind, token=None):
                calls["kind"] = kind.name

            def upload(self, local_dir, repo_id, **kwargs):
                calls["kwargs"] = kwargs
                calls["staged"] = sorted(
                    p.relative_to(local_dir).as_posix() for p in Path(local_dir).rglob("*") if p.is_file()
                )
                return "sha"

        monkeypatch.setattr("interpretune.hub.manager.ITHubResourceManager", _Manager)
        assert publish_mod.publish_component(root, "org/coll") == "sha"
        assert calls["kwargs"]["match_staged"] is True and calls["kind"] == "component"
        assert calls["staged"] == ["README.md", "it_component.yaml", "ops.yaml", "tests/test_ops.py"]
