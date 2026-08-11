"""The ratified hub verb surface (hub-integration design 5e): ``it.hub.pull`` / ``it.hub.load`` / ``it.hub.push``.

Convenience wrappers over the manifest-first component machinery (`.components`, `.publish`) returning
hydrated :class:`~interpretune.registry.RegisteredCfg` tuples — the shape notebooks and experiment scripts
unpack directly. ``pull`` is the ONLY verb here that touches the network; ``load`` is cache-only by design
(the no-implicit-network invariant), and ``ITSession.from_hub`` composes either with session construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from interpretune.registry import RegisteredCfg


def _hydrate_component_body(key: str, body: dict) -> RegisteredCfg:
    """Hydrate a fetched configuration body through the one-door loader into a ``RegisteredCfg``."""
    from interpretune.config.loading import load_session_cfg

    loaded = load_session_cfg(body, expected_key=key)
    return RegisteredCfg(
        loaded.datamodule_cfg,
        loaded.module_cfg,
        loaded.datamodule_cls,  # type: ignore[arg-type]  # loader resolves str/None to concrete classes
        loaded.module_cls,  # type: ignore[arg-type]
    )


def pull(
    repo_id: str,
    key: str | None = None,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> Any:
    """Explicitly fetch a component from the Hub (network; manifest-first, revision-pinned).

    With ``key``, fetches that configuration and returns its hydrated
    :class:`~interpretune.registry.RegisteredCfg` — ``datamodule_cfg, module_cfg, datamodule_cls,
    module_cls = it.hub.pull("org/repo", "<key>")``. Without ``key``, returns
    ``(manifest, resolved_commit)`` so callers can inspect the available configurations first.
    """
    from interpretune.hub.components import pull_component_config, pull_component_manifest

    if key is None:
        return pull_component_manifest(repo_id, revision=revision, cache_dir=cache_dir, token=token)
    canonical, body = pull_component_config(repo_id, key, revision=revision, cache_dir=cache_dir, token=token)
    return _hydrate_component_body(canonical, body)


def load(repo_id: str, key: str, *, cache_dir: Path | None = None) -> RegisteredCfg:
    """Cache-only hydration of one configuration — never touches the network.

    The component must already be in the local components cache, via an explicit :func:`pull` or the
    local-publish bridge (e.g. ``it_examples.seeds.ensure_local_seeds`` for the in-tree seeds); an
    uncached component raises with the exact fetch command.
    """
    from interpretune.hub.components import resolve_component_config

    canonical, body = resolve_component_config(repo_id, key, cache_dir=cache_dir)
    return _hydrate_component_body(canonical, body)


def push(
    component_dir: Path,
    repo_id: str,
    *,
    entrypoint_src: Path | None = None,
    private: bool = False,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Publish a component tree to the Hub (build + parity checks + generated card); returns the commit."""
    from interpretune.hub.publish import publish_component

    return publish_component(
        component_dir,
        repo_id,
        entrypoint_src=entrypoint_src,
        private=private,
        token=token,
        commit_message=commit_message,
    )
