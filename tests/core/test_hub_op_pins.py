"""Build 849's silent half (#334): a pin must bind EXECUTION, not just the download.

The trust posture tells users to pin a revision "so trusted code cannot change under you", and
``pull_ops(revision=...)`` always honored the pin for the fetch — but discovery ignored pins
entirely, so pinning ``A`` and letting anyone republish ``B`` meant the next session that resolved
``main`` would scan, trust-gate, compile and execute ``B``. These tests fabricate exactly that
republish state (pinned snapshot beside a newer ``refs/main``) and hold discovery to the pin.

Binding is strict by design: a pinned revision that is evicted (or cached without its manifest) is
refused with the restore/release gesture, never silently substituted — substitution IS the bug this
closes, just wearing a different justification.
"""

from __future__ import annotations

import json

import pytest

from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager
from interpretune.analysis.ops.compiler.load_policy import OpLoadError
from interpretune.hub.pins import clear_op_pin, list_op_pins, op_pins_dir, read_op_pin, record_op_pin

PINNED_SHA = "a" * 40
REPUBLISHED_SHA = "b" * 40
EVICTED_SHA = "c" * 40

MANIFEST = "it_schema_version: 1\nkinds:\n- ops\nops:\n  files:\n  - fake_ops.yaml\n"
OPS_YAML = """collection:
  name: fake_pinned_repo
  version: 0.0.1

fake_pinned_probe_op:
  description: minimal op for pin-binding tests
  implementation: fake_defs.fake_impl
  input_schema:
    orig_labels:
      datasets_dtype: int64
      required: false
  output_schema:
    preds:
      datasets_dtype: int64
"""
IMPL = "def fake_impl(module, analysis_batch, batch, batch_idx, **kw):\n    return analysis_batch\n"


def _write_snapshot(repo_dir, sha, complete=True):
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "fake_defs.py").write_text(IMPL)
    if complete:
        (snap / "it_component.yaml").write_text(MANIFEST)
        (snap / "fake_ops.yaml").write_text(OPS_YAML)
    return snap


@pytest.fixture
def republished_hub_cache(tmp_path):
    """A pinned snapshot beside a newer complete refs/main: the exact state a republish leaves."""
    cache = tmp_path / "hub_cache"  # a subdir, so the `<cache>_pins` sidecar stays inside tmp_path
    repo = cache / "models--pinuser--pinned_repo"
    _write_snapshot(repo, PINNED_SHA)
    _write_snapshot(repo, REPUBLISHED_SHA)
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text(REPUBLISHED_SHA)
    return cache


def _discover(cache_root, tmp_path, monkeypatch):
    import interpretune.analysis as ia

    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    monkeypatch.setattr(ia, "IT_ANALYSIS_HUB_CACHE", str(cache_root))
    manager = OpDefinitionsCacheManager(tmp_path / "opcache")
    return manager, manager.discover_hub_yaml_files()


class TestPinBindsDiscovery:
    def test_republish_cannot_move_a_pinned_environment(self, republished_hub_cache, tmp_path, monkeypatch, recwarn):
        record_op_pin("pinuser/pinned_repo", PINNED_SHA, "v0.0.1", cache_root=republished_hub_cache)
        _, yaml_files = _discover(republished_hub_cache, tmp_path, monkeypatch)
        assert any(PINNED_SHA[:12] in str(p) for p in yaml_files), (
            "discovery loaded the republished refs/main revision over the pin -- the #334 gap"
        )
        assert not any(REPUBLISHED_SHA[:12] in str(p) for p in yaml_files)

    def test_without_a_pin_refs_main_still_wins(self, republished_hub_cache, tmp_path, monkeypatch):
        _, yaml_files = _discover(republished_hub_cache, tmp_path, monkeypatch)
        assert any(REPUBLISHED_SHA[:12] in str(p) for p in yaml_files), (
            "the unpinned default (refs/main, manifest-complete) must be preserved"
        )

    def test_evicted_pin_is_refused_not_substituted(self, republished_hub_cache, tmp_path, monkeypatch, recwarn):
        record_op_pin("pinuser/pinned_repo", EVICTED_SHA, EVICTED_SHA, cache_root=republished_hub_cache)
        _, yaml_files = _discover(republished_hub_cache, tmp_path, monkeypatch)
        assert yaml_files == [], "an evicted pinned revision must refuse, never substitute refs/main"
        warned = " ".join(str(w.message) for w in recwarn.list)
        assert "no longer cached" in warned and EVICTED_SHA[:12] in warned
        assert "unpin_ops" in warned and "pull_ops" in warned, "the refusal must carry the restore/release gesture"

    def test_evicted_pin_raises_under_strict_loading(self, republished_hub_cache, tmp_path, monkeypatch):
        record_op_pin("pinuser/pinned_repo", EVICTED_SHA, EVICTED_SHA, cache_root=republished_hub_cache)
        monkeypatch.setenv("IT_STRICT_OP_LOAD", "1")
        with pytest.raises(OpLoadError, match="no longer cached"):
            _discover(republished_hub_cache, tmp_path, monkeypatch)

    def test_manifestless_pinned_snapshot_is_refused(self, tmp_path, monkeypatch, recwarn):
        cache = tmp_path / "cache"
        repo = cache / "models--pinuser--pinned_repo"
        _write_snapshot(repo, PINNED_SHA, complete=False)
        _write_snapshot(repo, REPUBLISHED_SHA)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(REPUBLISHED_SHA)
        record_op_pin("pinuser/pinned_repo", PINNED_SHA, "v0.0.1", cache_root=cache)
        _, yaml_files = _discover(cache, tmp_path, monkeypatch)
        assert yaml_files == []
        warned = " ".join(str(w.message) for w in recwarn.list)
        assert "cached without its manifest" in warned

    def test_refusal_is_scoped_to_the_pinned_repo(self, republished_hub_cache, tmp_path, monkeypatch, recwarn):
        other = republished_hub_cache / "models--other--healthy_repo"
        _write_snapshot(other, REPUBLISHED_SHA)
        (other / "refs").mkdir()
        (other / "refs" / "main").write_text(REPUBLISHED_SHA)
        record_op_pin("pinuser/pinned_repo", EVICTED_SHA, EVICTED_SHA, cache_root=republished_hub_cache)
        _, yaml_files = _discover(republished_hub_cache, tmp_path, monkeypatch)
        assert any("healthy_repo" in str(p) for p in yaml_files), (
            "one repo's pin refusal must not deny the session its other collections"
        )


class TestPinRecordLifecycle:
    def test_record_read_clear_roundtrip(self, tmp_path):
        path = record_op_pin("u/r", PINNED_SHA, "v1.0.0", cache_root=tmp_path / "cache")
        assert path.is_file() and op_pins_dir(tmp_path / "cache").name == "cache_pins"
        pin = read_op_pin("u/r", cache_root=tmp_path / "cache")
        assert pin is not None and pin["commit"] == PINNED_SHA and pin["requested_revision"] == "v1.0.0"
        assert clear_op_pin("u/r", cache_root=tmp_path / "cache") is True
        assert read_op_pin("u/r", cache_root=tmp_path / "cache") is None
        assert clear_op_pin("u/r", cache_root=tmp_path / "cache") is False

    def test_repin_overwrites(self, tmp_path):
        record_op_pin("u/r", PINNED_SHA, "v1", cache_root=tmp_path / "cache")
        record_op_pin("u/r", REPUBLISHED_SHA, "v2", cache_root=tmp_path / "cache")
        pin = read_op_pin("u/r", cache_root=tmp_path / "cache")
        assert pin is not None and pin["commit"] == REPUBLISHED_SHA, "re-pulling at a new revision IS the update verb"

    def test_malformed_record_degrades_to_unpinned_with_a_warning(self, tmp_path, recwarn):
        path = record_op_pin("u/r", PINNED_SHA, "v1", cache_root=tmp_path / "cache")
        path.write_text("{not json")
        assert read_op_pin("u/r", cache_root=tmp_path / "cache") is None
        assert any("malformed" in str(w.message) for w in recwarn.list)
        path.write_text(json.dumps({"repo_id": "u/r"}))  # parseable but no commit
        assert read_op_pin("u/r", cache_root=tmp_path / "cache") is None

    def test_list_includes_pins_for_uncached_repos(self, tmp_path):
        record_op_pin("u/gone", PINNED_SHA, "v1", cache_root=tmp_path / "cache")
        record_op_pin("u/here", REPUBLISHED_SHA, "v2", cache_root=tmp_path / "cache")
        assert set(list_op_pins(cache_root=tmp_path / "cache")) == {"u/gone", "u/here"}


class TestPullSidePinSemantics:
    def test_explicit_revision_records_a_pin(self, tmp_path):
        from interpretune.hub.opcollections import _record_or_report_pin

        _record_or_report_pin("u/r", "v1.2.3", PINNED_SHA, str(tmp_path / "cache"))
        pin = read_op_pin("u/r", cache_root=tmp_path / "cache")
        assert pin is not None and pin["commit"] == PINNED_SHA and pin["requested_revision"] == "v1.2.3"

    @pytest.mark.parametrize("revision", [None, "main"])
    def test_the_moving_default_is_not_a_pin(self, tmp_path, revision):
        from interpretune.hub.opcollections import _record_or_report_pin

        _record_or_report_pin("u/r", revision, PINNED_SHA, str(tmp_path / "cache"))
        assert read_op_pin("u/r", cache_root=tmp_path / "cache") is None, (
            "freezing a user who explicitly asked for `main` would be the opposite surprise"
        )

    def test_unpinned_pull_over_a_pin_says_the_pin_still_governs(self, tmp_path, recwarn):
        from interpretune.hub.opcollections import _record_or_report_pin

        record_op_pin("u/r", PINNED_SHA, "v1", cache_root=tmp_path / "cache")
        _record_or_report_pin("u/r", None, REPUBLISHED_SHA, str(tmp_path / "cache"))
        warned = " ".join(str(w.message) for w in recwarn.list)
        assert "keeps loading the pin" in warned and "unpin_ops" in warned
        pin = read_op_pin("u/r", cache_root=tmp_path / "cache")
        assert pin is not None and pin["commit"] == PINNED_SHA, "an unpinned pull must not move the pin"


class TestProvenanceReportsThePin:
    def test_cached_op_revision_prefers_the_pin_over_refs_main(self, tmp_path, monkeypatch):
        import interpretune.analysis.ops.dispatcher as dispatcher_module
        from interpretune.analysis.ops.dispatcher import _cached_op_revision

        cache = tmp_path / "hub"
        refs = cache / "models--pinuser--pinned_repo" / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(REPUBLISHED_SHA)
        monkeypatch.setattr(dispatcher_module, "IT_ANALYSIS_HUB_CACHE", cache)
        assert _cached_op_revision("hub:pinuser.pinned_repo") == REPUBLISHED_SHA
        record_op_pin("pinuser/pinned_repo", PINNED_SHA, "v1", cache_root=cache)
        assert _cached_op_revision("hub:pinuser.pinned_repo") == PINNED_SHA, (
            "a pinned session must report the revision it actually loads, not where `main` moved"
        )

    def test_loaded_paths_beat_the_filesystem_answer(self, republished_hub_cache, tmp_path, monkeypatch):
        record_op_pin("pinuser/pinned_repo", PINNED_SHA, "v1", cache_root=republished_hub_cache)
        manager, yaml_files = _discover(republished_hub_cache, tmp_path, monkeypatch)
        for yaml_file in yaml_files:
            manager.add_yaml_file(yaml_file)
        assert manager.hub_commit_for_namespace("pinuser.pinned_repo") == PINNED_SHA
        assert manager.hub_commit_for_namespace("nobody.nothing") is None


class TestApiVerbs:
    def test_op_pins_annotates_cachedness_and_unpin_releases(self, republished_hub_cache, monkeypatch):
        import interpretune.analysis as ia
        from interpretune.hub.api import op_pins, unpin_ops

        monkeypatch.setattr(ia, "IT_ANALYSIS_HUB_CACHE", str(republished_hub_cache))
        record_op_pin("pinuser/pinned_repo", PINNED_SHA, "v1", cache_root=republished_hub_cache)
        record_op_pin("pinuser/evicted_repo", EVICTED_SHA, "v9", cache_root=republished_hub_cache)
        pins = op_pins(cache_dir=republished_hub_cache)
        assert pins["pinuser/pinned_repo"]["cached"] is True
        assert pins["pinuser/evicted_repo"]["cached"] is False, (
            "cached: False is the 'discovery is refusing this collection' flag"
        )
        assert unpin_ops("pinuser/evicted_repo", cache_dir=republished_hub_cache, reload=False) is True
        assert unpin_ops("pinuser/evicted_repo", cache_dir=republished_hub_cache, reload=False) is False
        assert set(op_pins(cache_dir=republished_hub_cache)) == {"pinuser/pinned_repo"}
