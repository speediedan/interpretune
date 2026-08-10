"""Cache roots and HF-cache introspection shared by every interpretune Hub resource kind.

Layout (hub-integration design, ACCEPTED v2 §7): one ``IT_CACHE`` root (default
``$HF_CACHE_HOME/interpretune``) with per-purpose subtrees. Existing environment variables remain authoritative
as subpath overrides — ``IT_ANALYSIS_HUB_CACHE`` keeps its historical default so op-collection caches do not
move on disk.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from datasets.config import HF_CACHE_HOME  # canonical HF cache root, shared with the analysis caches
from huggingface_hub.utils._cache_manager import CachedRepoInfo, CachedRevisionInfo, scan_cache_dir

IT_CACHE = Path(os.getenv("IT_CACHE", os.path.join(HF_CACHE_HOME, "interpretune")))

# component repos (modules/datamodules/ops/adapters) fetched from the Hub
IT_COMPONENTS_HUB_CACHE = Path(os.getenv("IT_COMPONENTS_HUB_CACHE", IT_CACHE / "hub" / "components"))
# data artifact repos (analysis stores, dashboards) — reserved for #124 consumers
IT_ARTIFACTS_HUB_CACHE = Path(os.getenv("IT_ARTIFACTS_HUB_CACHE", IT_CACHE / "hub" / "artifacts"))

# historical op-collection hub cache location (subpath override preserved; default unchanged on disk)
IT_ANALYSIS_HUB_CACHE_DIR_NAME = "interpretune_ops"
DEFAULT_IT_ANALYSIS_HUB_CACHE = Path(os.path.join(HF_CACHE_HOME, "hub")) / IT_ANALYSIS_HUB_CACHE_DIR_NAME
IT_ANALYSIS_HUB_CACHE = Path(os.getenv("IT_ANALYSIS_HUB_CACHE", DEFAULT_IT_ANALYSIS_HUB_CACHE))

_HUB_REPO_DIR_RE = re.compile(r"models--([^-]+(?:-[^-]+)*)--([^-]+(?:-[^-]+)*)$")


def _get_latest_revision(repo: CachedRepoInfo) -> CachedRevisionInfo | None:
    """Get the latest cached revision for a repository, preferring the ``main`` ref."""
    if not repo.revisions:
        return None
    main_revision = repo.refs.get("main")
    if main_revision is not None:
        return main_revision
    return max(repo.revisions, key=lambda rev: rev.last_modified)


def parse_hub_cache_path(path: Path, cache_root: Path) -> tuple[bool, str]:
    """Determine whether ``path`` lives in ``cache_root``'s HF cache layout and extract its namespace.

    Returns ``(is_hub_file, namespace)`` where the namespace is ``<user>.<repo>`` derived from the
    ``models--user--repo`` cache directory name (empty string when ``path`` is not a hub-cache file).
    """
    try:
        relative = path.resolve().relative_to(Path(cache_root).resolve())
    except ValueError:
        return False, ""
    for part in relative.parts:
        if part.startswith("models--"):
            if "snapshots" in relative.parts or "blobs" in relative.parts:
                match = _HUB_REPO_DIR_RE.match(part)
                if match:
                    user, repo = match.groups()
                    return True, f"{user}.{repo}"
            break
    return False, ""


def scan_cached_repos(cache_root: Path) -> list[CachedRepoInfo]:
    """Scan an HF-layout cache dir, returning model-repo entries (empty when the dir does not exist)."""
    root = Path(cache_root)
    if not root.exists():
        return []
    return [repo for repo in scan_cache_dir(root).repos if repo.repo_type == "model"]
