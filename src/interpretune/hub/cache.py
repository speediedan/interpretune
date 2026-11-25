"""Cache roots and HF-cache introspection shared by every interpretune Hub resource kind.

Layout (hub-integration design, ACCEPTED v2 §7): one ``IT_CACHE`` root (default
``$HF_CACHE_HOME/interpretune``) with per-purpose subtrees. Existing environment variables remain authoritative
as subpath overrides — ``IT_ANALYSIS_HUB_CACHE`` keeps its historical default so op-collection caches do not
move on disk.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from datasets.config import HF_CACHE_HOME  # canonical HF cache root, shared with the analysis caches
from huggingface_hub.utils._cache_manager import CachedRepoInfo, CachedRevisionInfo, scan_cache_dir

IT_CACHE = Path(os.getenv("IT_CACHE", os.path.join(HF_CACHE_HOME, "interpretune")))

# component repos (modules/datamodules/ops/adapters) fetched from the Hub
IT_COMPONENTS_HUB_CACHE = Path(os.getenv("IT_COMPONENTS_HUB_CACHE", IT_CACHE / "hub" / "components"))
# data artifact repos (analysis stores, dashboards)
IT_ARTIFACTS_HUB_CACHE = Path(os.getenv("IT_ARTIFACTS_HUB_CACHE", IT_CACHE / "hub" / "artifacts"))

# historical op-collection hub cache location (subpath override preserved; default unchanged on disk)
IT_ANALYSIS_HUB_CACHE_DIR_NAME = "interpretune_ops"
DEFAULT_IT_ANALYSIS_HUB_CACHE = Path(os.path.join(HF_CACHE_HOME, "hub")) / IT_ANALYSIS_HUB_CACHE_DIR_NAME
IT_ANALYSIS_HUB_CACHE = Path(os.getenv("IT_ANALYSIS_HUB_CACHE", DEFAULT_IT_ANALYSIS_HUB_CACHE))


def _root_setting(override_var: str, default_setting: str) -> str:
    return override_var if override_var in os.environ else default_setting


# What set each root, named in a refusal: the subpath override when one is set, otherwise the variable it derives from.
IT_CACHE_SETTING = "IT_CACHE" if "IT_CACHE" in os.environ else "IT_CACHE (unset, so its default under HF_HOME)"
IT_COMPONENTS_HUB_CACHE_SETTING = _root_setting("IT_COMPONENTS_HUB_CACHE", IT_CACHE_SETTING)
IT_ARTIFACTS_HUB_CACHE_SETTING = _root_setting("IT_ARTIFACTS_HUB_CACHE", IT_CACHE_SETTING)
IT_ANALYSIS_HUB_CACHE_SETTING = _root_setting(
    "IT_ANALYSIS_HUB_CACHE", "IT_ANALYSIS_HUB_CACHE (unset, so its default under HF_HOME)"
)

_writable_roots: set[str] = set()


def ensure_writable_cache_root(root: Path | str, setting: str) -> Path:
    """Create ``root`` if needed and prove it writable, once per process, refusing by name if it is not.

    Run at the first hub-cache use rather than at import, so a session that never touches the Hub never pays for it.
    Only writability is probed: what a run will download is unknowable here, so a full disk mid-download stays the
    downloader's error. The probe file has a unique name, so processes sharing a root (CI legs) cannot collide.
    """
    root = Path(root)
    key = os.fspath(root)
    if key in _writable_roots:
        return root
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".it-write-probe-", dir=root)
        os.close(fd)
        os.unlink(probe)
    except OSError as exc:
        from interpretune.utils.exceptions import MisconfigurationException

        variable = setting.split(" ", 1)[0]
        raise MisconfigurationException(
            f"interpretune's hub cache root {key} (set by {setting}) is not a writable directory: {exc}. Point"
            f" {variable} at a directory you can write to, or make this one writable. Set it before importing"
            " interpretune: the cache roots are read once, at import."
        ) from exc
    _writable_roots.add(key)
    return root


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
