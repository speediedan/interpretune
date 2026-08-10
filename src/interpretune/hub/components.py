"""Manifest-first fetch of interpretune component repos.

The standard resolution path fetches ``it_component.yaml`` (a few KB), reads its ``configs:`` index, then
fetches only the requested configuration file — pinning every follow-up fetch to the commit the manifest came
from, so a repo updated mid-resolution cannot hand back files from two different revisions. This also makes
``countDownloads: path:"it_component.yaml"`` correct by construction: every resolution starts at the manifest.

No function here is called implicitly by registry lookups — local resolution never touches the network; Hub
components are fetched only by these explicit calls.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from interpretune.hub.cache import IT_COMPONENTS_HUB_CACHE
from interpretune.hub.manifest import IT_COMPONENT_MANIFEST, check_config_key_parity, validate_component_manifest

try:  # version telemetry for hf_hub_download (design §8); interpretune may be running from a raw checkout
    from importlib.metadata import version as _pkg_version

    _IT_VERSION: str | None = _pkg_version("interpretune")
except Exception:  # pragma: no cover - metadata unavailable in exotic layouts
    _IT_VERSION = None

_TELEMETRY = {"library_name": "interpretune", "library_version": _IT_VERSION}


def _snapshot_revision(downloaded_path: Path) -> str:
    """Extract the resolved commit hash from an hf-cache download path (``.../snapshots/<commit>/...``)."""
    parts = Path(downloaded_path).parts
    return parts[parts.index("snapshots") + 1]


def pull_component_manifest(
    repo_id: str, revision: str | None = None, cache_dir: Path | None = None, token: str | None = None
) -> tuple[dict, str]:
    """Fetch and validate a component repo's manifest; returns ``(manifest, resolved_commit)``."""
    path = hf_hub_download(
        repo_id,
        IT_COMPONENT_MANIFEST,
        revision=revision,
        cache_dir=str(cache_dir or IT_COMPONENTS_HUB_CACHE),
        token=token,
        **_TELEMETRY,
    )
    manifest = validate_component_manifest(yaml.safe_load(Path(path).read_text()), source=f"{repo_id}@{revision}")
    return manifest, _snapshot_revision(Path(path))


def pull_component_config(
    repo_id: str, key: str, revision: str | None = None, cache_dir: Path | None = None, token: str | None = None
) -> tuple[str, dict]:
    """Manifest-first fetch of ONE configuration by key; returns ``(canonical_key, config_body)``.

    The configuration fetch is pinned to the commit the manifest resolved to, and the loader parity-check (filename ==
    manifest key == derived-from-fields) runs on the fetched file before it is returned.
    """
    manifest, commit = pull_component_manifest(repo_id, revision=revision, cache_dir=cache_dir, token=token)
    configs = (manifest.get("module") or {}).get("configs") or {}
    if key not in configs:
        raise KeyError(
            f"{repo_id} declares no configuration {key!r}. Available: {sorted(configs)} "
            f"(manifest revision {commit[:12]})"
        )
    cfg_path = Path(
        hf_hub_download(
            repo_id,
            configs[key],
            revision=commit,  # pinned: partial materialization stays single-revision coherent
            cache_dir=str(cache_dir or IT_COMPONENTS_HUB_CACHE),
            token=token,
            **_TELEMETRY,
        )
    )
    body = yaml.safe_load(cfg_path.read_text())
    return check_config_key_parity(cfg_path, body, expected_key=key), body


def register_component_config(
    repo_id: str, key: str, revision: str | None = None, cache_dir: Path | None = None, token: str | None = None
) -> str:
    """Fetch one configuration and register it under its hub-namespaced key ``<org>.<repo>.<key>``.

    Returns the namespaced registry key. Bare-key aliasing (collision-aware) is dispatcher-style follow-up work; the
    namespaced form is always registered and always unambiguous.
    """
    from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY, example_register_func

    canonical_key, body = pull_component_config(repo_id, key, revision=revision, cache_dir=cache_dir, token=token)
    namespaced = f"{repo_id.replace('/', '.')}.{canonical_key}"
    example_register_func(MODULE_EXAMPLE_REGISTRY.registry)(namespaced, body)
    return namespaced
