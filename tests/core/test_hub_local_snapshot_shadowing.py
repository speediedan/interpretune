"""#437: a stale ``local_publish`` snapshot must not shadow the Hub silently.

Resolution is cache-only by design and stays so. What changes is that a local snapshot resolving for a repo whose cache
ALSO holds a Hub revision is named, and a caller verifying a publish can refuse it outright.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from interpretune.hub.components import (
    LocalSnapshotShadowsHubError,
    LocalSnapshotWarning,
    is_local_revision,
    local_publish,
    resolve_component_manifest,
)

RTE = Path(__file__).parent.parent.parent / "src" / "it_examples" / "examples" / "rte"
RTE_ENTRYPOINT = RTE.parent.parent / "experiments" / "rte_boolq.py"


def _hub_snapshot(cache: Path, repo_id: str, sha: str, *, make_main: bool) -> Path:
    """A sha-shaped snapshot in HF layout, as a Hub download leaves it."""
    repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "it_component.yaml").write_text(
        yaml.safe_dump({"it_schema_version": 1, "kinds": ["ops"], "ops": {"files": ["x.yaml"]}})
    )
    if make_main:
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs" / "main").write_text(sha)
    return snap


def _resolved_silently(repo_id, cache):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifest, _snapshot, revision = resolve_component_manifest(repo_id, cache_dir=cache)
    shadow = [w for w in caught if issubclass(w.category, LocalSnapshotWarning)]
    return revision, shadow


class TestLocalSnapshotResolution:
    def test_a_local_only_repo_resolves_silently(self, tmp_path):
        """The in-tree seeds: nothing is shadowed, so nothing is said."""
        cache = tmp_path / "cache"
        local_publish(RTE, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        revision, shadow = _resolved_silently("speediedan/rte", cache)
        assert is_local_revision(revision) and not shadow

    def test_a_local_snapshot_over_a_hub_revision_is_named(self, tmp_path):
        cache = tmp_path / "cache"
        _hub_snapshot(cache, "speediedan/rte", "a" * 40, make_main=True)
        local_publish(RTE, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)  # refs/main now local
        with pytest.warns(
            LocalSnapshotWarning, match=r"shadows the cached Hub revision.*aaaaaaaaaaaa.*the Hub was not consulted"
        ):
            _, _, revision = resolve_component_manifest("speediedan/rte", cache_dir=cache)
        assert is_local_revision(revision)

    def test_a_hub_revision_resolves_silently_even_beside_local_snapshots(self, tmp_path):
        cache = tmp_path / "cache"
        local_publish(RTE, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        _hub_snapshot(cache, "speediedan/rte", "b" * 40, make_main=True)  # a later pull moved refs/main to the Hub
        revision, shadow = _resolved_silently("speediedan/rte", cache)
        assert revision == "b" * 40 and not shadow

    def test_require_hub_refuses_a_local_snapshot_naming_the_fetch(self, tmp_path):
        cache = tmp_path / "cache"
        _hub_snapshot(cache, "speediedan/rte", "c" * 40, make_main=False)
        local_publish(RTE, "speediedan/rte", entrypoint_src=RTE_ENTRYPOINT, cache_dir=cache)
        with pytest.raises(
            LocalSnapshotShadowsHubError, match=r"cccccccccccc.*interpretune.hub.pull\('speediedan/rte'\)"
        ):
            resolve_component_manifest("speediedan/rte", cache_dir=cache, require_hub=True)
        # and the verbs a verifier reaches for carry the same switch
        import interpretune as it

        with pytest.raises(LocalSnapshotShadowsHubError):
            it.hub.load("speediedan/rte", "rte_demo.gpt2.sae_lens", cache_dir=cache, require_hub=True)

    def test_require_hub_accepts_a_hub_revision(self, tmp_path):
        cache = tmp_path / "cache"
        _hub_snapshot(cache, "org/thing", "d" * 40, make_main=True)
        _, _, revision = resolve_component_manifest("org/thing", cache_dir=cache, require_hub=True)
        assert revision == "d" * 40
