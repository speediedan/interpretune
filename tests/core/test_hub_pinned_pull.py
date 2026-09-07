"""#490: a revision-pinned pull into a clean cache must leave a component the loader can address.

hf_hub_download writes `refs/main` only for an unpinned fetch, and resolution read nothing else, so following the
trust posture's own advice (pin a revision) on a clean machine produced a complete snapshot the loader reported as
never cached. A pinned pull now records its pin where resolution looks, resolution prefers the pin over `main`,
an explicit revision addresses any cached snapshot, and the cached-but-unaddressable state is named as such.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from interpretune.hub.components import (
    COMPONENT_PIN_REF,
    cached_component_revisions,
    read_component_pin,
    resolve_component_manifest,
)

SHA_A = "a" * 40
SHA_B = "b" * 40
REPO = "org/pinned"


def _fake_hub(cache: Path, commits: dict[str, dict], *, main: str | None):
    """A stand-in for hf_hub_download: writes the snapshot for the requested revision, and `refs/main` only
    for an unpinned fetch, exactly as huggingface_hub does."""

    def download(repo_id, filename, revision=None, cache_dir=None, token=None, **_):
        commit = main if revision in (None, "main") else revision
        assert commit in commits, f"fake hub has no revision {commit!r}"
        repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
        snap = repo_dir / "snapshots" / commit
        snap.mkdir(parents=True, exist_ok=True)
        (snap / filename).write_text(yaml.safe_dump(commits[commit]))
        if revision in (None, "main"):
            (repo_dir / "refs").mkdir(exist_ok=True)
            (repo_dir / "refs" / "main").write_text(commit)
        return str(snap / filename)

    return download


MANIFEST_A = {"it_schema_version": 1, "kinds": ["ops"], "ops": {"files": ["a.yaml"]}}
MANIFEST_B = {"it_schema_version": 1, "kinds": ["ops"], "ops": {"files": ["b.yaml"]}}


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    from interpretune.hub import components

    cache = tmp_path / "cache"
    monkeypatch.setattr(
        components, "hf_hub_download", _fake_hub(cache, {SHA_A: MANIFEST_A, SHA_B: MANIFEST_B}, main=SHA_B)
    )
    return cache


class TestPinnedPullIsAddressable:
    def test_a_pinned_pull_into_a_clean_cache_loads(self, hub):
        import interpretune as it

        manifest, commit = it.hub.pull(REPO, revision=SHA_A, cache_dir=hub)
        assert commit == SHA_A and manifest["ops"]["files"] == ["a.yaml"]
        assert not (hub / "models--org--pinned" / "refs" / "main").exists(), (
            "the fake mirrors HF: no main ref on a pinned fetch"
        )
        assert read_component_pin(REPO, cache_dir=hub) == SHA_A
        resolved, _, revision = resolve_component_manifest(REPO, cache_dir=hub)
        assert revision == SHA_A and resolved["ops"]["files"] == ["a.yaml"]

    def test_an_unpinned_pull_resolves_through_main_as_before(self, hub):
        import interpretune as it

        _, commit = it.hub.pull(REPO, cache_dir=hub)
        assert commit == SHA_B and read_component_pin(REPO, cache_dir=hub) is None
        assert resolve_component_manifest(REPO, cache_dir=hub)[2] == SHA_B

    def test_the_pin_beats_a_later_main_until_released(self, hub):
        import interpretune as it

        it.hub.pull(REPO, revision=SHA_A, cache_dir=hub)
        it.hub.pull(REPO, cache_dir=hub)  # a later unpinned fetch moves refs/main to B
        assert (hub / "models--org--pinned" / "refs" / "main").read_text() == SHA_B
        assert resolve_component_manifest(REPO, cache_dir=hub)[2] == SHA_A, (
            "a republish must not change a pinned environment"
        )
        assert it.hub.unpin(REPO, cache_dir=hub) is True
        assert resolve_component_manifest(REPO, cache_dir=hub)[2] == SHA_B
        assert it.hub.unpin(REPO, cache_dir=hub) is False

    def test_an_explicit_revision_addresses_any_cached_snapshot(self, hub):
        import interpretune as it

        it.hub.pull(REPO, cache_dir=hub)
        it.hub.pull(REPO, revision=SHA_A, cache_dir=hub)
        assert sorted(cached_component_revisions(REPO, cache_dir=hub)) == [SHA_A, SHA_B]
        assert resolve_component_manifest(REPO, cache_dir=hub, revision=SHA_B)[2] == SHA_B
        with pytest.raises(KeyError, match=r"no cached revision 'c+'.*pull\('org/pinned', revision='c+'\)"):
            resolve_component_manifest(REPO, cache_dir=hub, revision="c" * 40)

    def test_cached_but_unaddressable_is_named_with_both_fixes(self, hub):
        """The state older caches are in: a snapshot with no ref of any kind."""
        import interpretune as it

        it.hub.pull(REPO, revision=SHA_A, cache_dir=hub)
        (hub / "models--org--pinned" / "refs" / COMPONENT_PIN_REF).unlink()
        with pytest.raises(
            KeyError, match=r"cached .* but no revision is addressed.*revision='a+'.*pull\('org/pinned'\)"
        ):
            resolve_component_manifest(REPO, cache_dir=hub)

    def test_nothing_cached_still_says_so(self, hub):
        with pytest.raises(KeyError, match="is not in the local cache"):
            resolve_component_manifest(REPO, cache_dir=hub)

    def test_load_hub_adapter_and_load_take_the_revision(self, tmp_path, monkeypatch):
        """The consumer verbs carry the same switch, so a caller who pinned can say so at load time too."""
        import inspect

        from interpretune.hub.adapters import load_hub_adapter
        from interpretune.hub.api import load

        assert "revision" in inspect.signature(load_hub_adapter).parameters
        assert "revision" in inspect.signature(load).parameters
