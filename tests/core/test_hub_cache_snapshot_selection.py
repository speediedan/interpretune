"""Build 849's crash half: discovery must not route through a manifest-less refs/main snapshot.

huggingface_hub materializes one snapshot dir per resolved revision containing only the files
actually fetched at it. A revision-pinned pull therefore leaves a COMPLETE pinned snapshot beside
whatever partial snapshot any single-file fetch at `main` created -- and `refs/main` points at the
partial one. Routing discovery by refs alone then skips the entire collection ("no it_component.yaml
in the cached snapshot") while the pull reports success: measured as #327's build-849 failure, where
the demo pulled the collection and `it.jlens_patch_intervention` raised one line later.

The fixture fabricates exactly that cache state; huggingface_hub's scanner works on any conforming
layout, so no network and no real repo is involved. The SILENT half (a complete newer main snapshot
winning over the revision a user pinned) is #334, deliberately not solved here.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager

PARTIAL_SHA = "a" * 40
COMPLETE_SHA = "b" * 40

MANIFEST = "it_schema_version: 1\nkinds:\n- ops\nops:\n  files:\n  - fake_ops.yaml\n"
OPS_YAML = """collection:
  name: fake_partial_repo
  version: 0.0.1

fake_partial_probe_op:
  description: minimal op for snapshot-selection tests
  implementation: fake_defs.fake_impl
  input_schema:
    orig_labels:
      datasets_dtype: int64
      required: false
  output_schema:
    preds:
      datasets_dtype: int64
"""


@pytest.fixture
def poisoned_hub_cache(tmp_path):
    """Refs/main -> a partial snapshot (impl file only); a complete manifested snapshot beside it."""
    repo = tmp_path / "models--fakeuser--fake_partial_repo"
    partial = repo / "snapshots" / PARTIAL_SHA
    complete = repo / "snapshots" / COMPLETE_SHA
    partial.mkdir(parents=True)
    complete.mkdir(parents=True)
    (partial / "fake_defs.py").write_text(
        "def fake_impl(module, analysis_batch, batch, batch_idx, **kw):\n    return analysis_batch\n"
    )
    (complete / "it_component.yaml").write_text(MANIFEST)
    (complete / "fake_ops.yaml").write_text(OPS_YAML)
    (complete / "fake_defs.py").write_text(
        "def fake_impl(module, analysis_batch, batch, batch_idx, **kw):\n    return analysis_batch\n"
    )
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text(PARTIAL_SHA)
    return tmp_path


def test_discovery_falls_back_to_the_manifested_snapshot(poisoned_hub_cache, tmp_path, monkeypatch, recwarn):
    import interpretune.analysis as ia

    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    monkeypatch.setattr(ia, "IT_ANALYSIS_HUB_CACHE", str(poisoned_hub_cache))
    manager = OpDefinitionsCacheManager(tmp_path / "opcache")
    yaml_files = manager.discover_hub_yaml_files()
    names = [p.name for p in yaml_files]
    assert "fake_ops.yaml" in names, (
        "discovery routed through the partial refs/main snapshot and skipped the collection -- "
        "the build-849 failure mode"
    )
    assert any(COMPLETE_SHA[:12] in str(p) for p in yaml_files)
    warned = " ".join(str(w.message) for w in recwarn.list)
    assert "partial fetch" in warned and PARTIAL_SHA[:12] in warned, (
        "the mismatch must be SAID, not silently papered over"
    )


def test_refs_main_still_wins_when_it_is_complete(poisoned_hub_cache, tmp_path, monkeypatch):
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    # promote the partial snapshot to complete: main now legitimately wins
    partial = poisoned_hub_cache / "models--fakeuser--fake_partial_repo" / "snapshots" / PARTIAL_SHA
    (partial / "it_component.yaml").write_text(MANIFEST)
    (partial / "fake_ops.yaml").write_text(OPS_YAML)
    import interpretune.analysis as ia

    monkeypatch.setattr(ia, "IT_ANALYSIS_HUB_CACHE", str(poisoned_hub_cache))
    manager = OpDefinitionsCacheManager(tmp_path / "opcache")
    yaml_files = manager.discover_hub_yaml_files()
    assert any(PARTIAL_SHA[:12] in str(p) for p in yaml_files), "a complete refs/main snapshot must keep priority"
