"""Manifest-routed op-collection discovery and the revision-pinned pull (#266 Phase 3, item 1).

Op discovery used to glob every ``*.yaml`` in a cached snapshot. It is now routed through
``it_component.yaml``'s ``ops.files`` list, which is a contract change with three consequences worth pinning
down: a repo's non-op YAML is no longer op input, a collection's ``requires:`` window is enforced before its
ops load, and the manifest is genuinely read first (the ``countDownloads`` anchor registration PR A claims).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager
from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError
from interpretune.hub.opcollections import (
    OpCollectionManifestError,
    declared_op_files,
    resolve_cached_op_files,
)

OP_YAML = """my_hub_op:
  description: fixture op
  implementation: interpretune.analysis.ops.bundled.core.core_ops.model_fwd_impl
  input_schema: {}
  output_schema: {}
"""


def _snapshot(root: Path, repo: str = "models--username--repo", manifest: str | None = None, **files: str) -> Path:
    """Build one HF-layout cached snapshot; ``manifest=None`` writes no ``it_component.yaml`` at all."""
    snapshot = root / repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    if manifest is not None:
        (snapshot / "it_component.yaml").write_text(manifest)
    for name, body in files.items():
        (snapshot / name.replace("__", ".")).write_text(body)
    return snapshot


OPS_MANIFEST = "it_schema_version: 1\nkinds: [ops]\nops:\n  files: [ops.yaml]\n"


class TestDeclaredOpFiles:
    def test_ops_kind_declares_its_files(self):
        assert declared_op_files({"kinds": ["ops"], "ops": {"files": ["a.yaml", "b.yaml"]}}) == ["a.yaml", "b.yaml"]

    def test_component_without_ops_kind_declares_none(self):
        """Not a failure: a repo may publish modules or prompt configs and no ops."""
        assert declared_op_files({"kinds": ["module"], "module": {"configs": {}}}) == []

    def test_empty_files_list_is_a_failure(self):
        with pytest.raises(OpCollectionManifestError, match="empty `ops.files`"):
            declared_op_files({"kinds": ["ops"], "ops": {"files": []}}, source="fixture")


class TestResolveCachedOpFiles:
    def test_declared_files_resolve_within_the_snapshot(self, tmp_path):
        snapshot = _snapshot(tmp_path, manifest=OPS_MANIFEST, ops__yaml=OP_YAML)
        assert resolve_cached_op_files(snapshot, source="fixture") == [snapshot / "ops.yaml"]

    def test_undeclared_yaml_is_not_op_input(self, tmp_path):
        """The reason routing exists: a README fixture or config sample must not reach the op compiler."""
        snapshot = _snapshot(
            tmp_path, manifest=OPS_MANIFEST, ops__yaml=OP_YAML, some_config__yaml="not: an op definition\n"
        )
        resolved = resolve_cached_op_files(snapshot, source="fixture")
        assert [p.name for p in resolved] == ["ops.yaml"]

    def test_missing_manifest_names_the_contract(self, tmp_path):
        snapshot = _snapshot(tmp_path, manifest=None, ops__yaml=OP_YAML)
        with pytest.raises(OpCollectionManifestError, match="no it_component.yaml"):
            resolve_cached_op_files(snapshot, source="fixture")

    def test_declared_file_that_is_absent_is_a_failure(self, tmp_path):
        snapshot = _snapshot(tmp_path, manifest=OPS_MANIFEST)
        with pytest.raises(OpCollectionManifestError, match="not present in the snapshot"):
            resolve_cached_op_files(snapshot, source="fixture")

    def test_requires_window_is_enforced_before_ops_load(self, tmp_path):
        manifest = OPS_MANIFEST + 'requires:\n  interpretune: ">=99"\n'
        snapshot = _snapshot(tmp_path, manifest=manifest, ops__yaml=OP_YAML)
        from interpretune.hub.components import ComponentRequirementError

        with pytest.raises(ComponentRequirementError, match=">=99"):
            resolve_cached_op_files(snapshot, source="fixture")

    def test_satisfiable_requires_window_resolves(self, tmp_path):
        manifest = OPS_MANIFEST + 'requires:\n  interpretune: ">=0.1.0.dev0"\n'
        snapshot = _snapshot(tmp_path, manifest=manifest, ops__yaml=OP_YAML)
        assert [p.name for p in resolve_cached_op_files(snapshot, source="fixture")] == ["ops.yaml"]

    def test_manifest_never_lists_itself(self, tmp_path):
        """Feeding the manifest to the op compiler drops every op in the process, bundled included.

        Raised by the shared manifest validator (the base ``ComponentManifestError``), not by the ops-kind
        resolution here: the rule belongs to the schema, and this asserts resolution surfaces it.
        """
        from interpretune.hub.manifest import ComponentManifestError

        manifest = "it_schema_version: 1\nkinds: [ops]\nops:\n  files: [it_component.yaml]\n"
        snapshot = _snapshot(tmp_path, manifest=manifest)
        with pytest.raises(ComponentManifestError, match="must not list it_component.yaml"):
            resolve_cached_op_files(snapshot, source="fixture")


class TestDiscoveryFailsSoftPerCollection:
    """A malformed third-party collection must not deny a session the ops it does have."""

    def _discover(self, tmp_path, **snapshot_kwargs):
        hub_cache = tmp_path / "hub"
        _snapshot(hub_cache, **snapshot_kwargs)
        cache_manager = OpDefinitionsCacheManager(tmp_path / "cache")
        with (
            patch("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache),
            patch("interpretune.analysis.IT_ANALYSIS_OP_PATHS", []),
        ):
            return cache_manager.discover_hub_yaml_files()

    def test_missing_manifest_warns_and_contributes_nothing(self, tmp_path):
        with pytest.warns(UserWarning, match="Skipping op collection"):
            assert self._discover(tmp_path, manifest=None, ops__yaml=OP_YAML) == []

    def test_incompatible_collection_warns_and_contributes_nothing(self, tmp_path):
        manifest = OPS_MANIFEST + 'requires:\n  interpretune: ">=99"\n'
        with pytest.warns(UserWarning, match="Skipping op collection"):
            assert self._discover(tmp_path, manifest=manifest, ops__yaml=OP_YAML) == []

    def test_strict_load_raises_instead_of_warning(self, tmp_path, monkeypatch):
        """Regression guard: discovery's own fail-soft wrapper must not swallow the strict-mode raise."""
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        with pytest.raises(OpLoadError, match="Skipping op collection"):
            self._discover(tmp_path, manifest=None, ops__yaml=OP_YAML)

    def test_component_declaring_no_ops_is_quiet(self, tmp_path):
        """A module-kind repo in the cache is not a malformed op collection, so it must not warn."""
        import warnings

        manifest = "it_schema_version: 1\nkinds: [module]\nmodule:\n  configs: {}\n"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._discover(tmp_path, manifest=manifest) == []


class TestPullIsManifestFirstAndPinned:
    """The ``countDownloads: path:"it_component.yaml"`` contract, and single-revision coherence."""

    def _run_pull(self, tmp_path, manifest_body: str, revision: str | None = None):
        from interpretune.hub import opcollections

        calls: list[tuple[str, str | None]] = []
        snapshot = tmp_path / "models--u--r" / "snapshots" / ("d" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "it_component.yaml").write_text(manifest_body)
        (snapshot / "ops.yaml").write_text(OP_YAML)

        def fake_download(repo_id, filename, revision=None, **kwargs):
            calls.append((filename, revision))
            return str(snapshot / filename)

        # `pull_op_collection` imports hf_hub_download inside the function body, so the module-level
        # huggingface_hub attribute is the binding it resolves through.
        with patch("huggingface_hub.hf_hub_download", fake_download):
            paths, commit = opcollections.pull_op_collection("u/r", revision=revision, cache_dir=tmp_path)
        return calls, paths, commit

    def test_manifest_is_fetched_first_and_op_files_are_pinned_to_its_commit(self, tmp_path):
        calls, paths, commit = self._run_pull(tmp_path, OPS_MANIFEST, revision="main")

        assert calls[0][0] == "it_component.yaml", "the manifest must be the first fetch (download-count anchor)"
        assert calls[0][1] == "main", "the manifest fetch honors the caller's requested revision"
        assert commit == "d" * 40
        op_fetches = [c for c in calls if c[0] == "ops.yaml"]
        assert op_fetches and all(rev == commit for _, rev in op_fetches), (
            "op files must be fetched at the commit the manifest resolved to, never at a moving ref"
        )
        assert [p.name for p in paths] == ["ops.yaml"]

    def test_pulling_a_repo_that_declares_no_ops_is_an_error(self, tmp_path):
        manifest = "it_schema_version: 1\nkinds: [module]\nmodule:\n  configs: {}\n"
        with pytest.raises(OpCollectionManifestError, match="no `ops` kind"):
            self._run_pull(tmp_path, manifest)
