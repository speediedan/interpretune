"""Seed components: materialize the in-tree publish sources into the local components cache.

Post-flip (hub design v3 §11.2), the in-tree ``examples/`` trees are publish sources ONLY — the loader
reads exclusively from the components cache. This module is the local-publish bridge's entry point for
the seeds that ship with ``it_examples``: one idempotent, offline call materializes them into the cache
in HF layout, after which ``it.hub.load(...)`` / ``ITSession.from_hub(...)`` resolve them exactly as
they would a hub-fetched component (and the publish machinery itself is exercised on every run).
"""

from __future__ import annotations

from pathlib import Path

_PKG_ROOT = Path(__file__).parent
_EXAMPLES_ROOT = _PKG_ROOT / "examples"

# repo_id -> (component publish source, module entrypoint source). Seed repos live under the
# maintainer org by convention; users publish their OWN components under their own orgs.
SEED_COMPONENTS: dict[str, tuple[Path, Path]] = {
    "speediedan/rte": (_EXAMPLES_ROOT / "rte", _PKG_ROOT / "experiments" / "rte_boolq.py"),
}


def ensure_local_seeds(cache_dir: Path | None = None) -> dict[str, str]:
    """Local-publish every in-tree seed component into the components cache (offline, idempotent).

    Returns ``{repo_id: revision}`` — content-derived pseudo-revisions, so unchanged trees are no-ops.
    """
    from interpretune.hub.components import local_publish

    return {
        repo_id: local_publish(component_dir, repo_id, entrypoint_src=entrypoint, cache_dir=cache_dir)
        for repo_id, (component_dir, entrypoint) in SEED_COMPONENTS.items()
    }
