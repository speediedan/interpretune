"""The ratified hub verb surface (hub-integration design 5e): ``it.hub.pull`` / ``it.hub.load`` / ``it.hub.push``.

Convenience wrappers over the manifest-first component machinery (`.components`, `.publish`) returning
hydrated :class:`~interpretune.registry.RegisteredCfg` tuples — the shape notebooks and experiment scripts
unpack directly. ``pull`` is the ONLY verb here that touches the network; ``load`` is cache-only by design
(the no-implicit-network invariant), and ``ITSession.from_hub`` composes either with session construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from interpretune.registry import RegisteredCfg

if TYPE_CHECKING:
    from interpretune.registry import RegisteredDataModuleCfg


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


def pull_ops(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
    reload: bool = True,
) -> tuple[list[Path], str]:
    """Explicitly fetch an op collection from the Hub (network; manifest-first, revision-pinned).

    Returns ``(op_yaml_paths, resolved_commit)``. The ops kind needs its own verb rather than a ``key``-less
    :func:`pull`: op collections live in their own cache (``IT_ANALYSIS_HUB_CACHE``, which is what op
    discovery scans), so a collection fetched through the component verb would land where the dispatcher
    never looks.

    With ``reload`` (default), the dispatcher re-reads its op sources so the fetched collection is usable
    immediately. Without it the ops are on disk but a session that already loaded definitions will not see
    them, and using one raises ``Unknown operation`` -- pass ``reload=False`` only when fetching several
    collections before paying for one reload. Reloading re-runs the trust gate and every collection's
    compatibility window; it does not import any op implementation, which still happens lazily.
    """
    from interpretune.hub.opcollections import pull_op_collection

    result = pull_op_collection(repo_id, revision=revision, cache_dir=cache_dir, token=token)
    if reload:
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.reload_definitions()
    return result


def load(repo_id: str, key: str, *, cache_dir: Path | None = None) -> RegisteredCfg:
    """Cache-only hydration of one configuration — never touches the network.

    The component must already be in the local components cache, via an explicit :func:`pull` or the
    local-publish bridge (e.g. ``it_examples.seeds.ensure_local_seeds`` for the in-tree seeds); an
    uncached component raises with the exact fetch command.
    """
    from interpretune.hub.components import resolve_component_config

    canonical, body = resolve_component_config(repo_id, key, cache_dir=cache_dir)
    return _hydrate_component_body(canonical, body)


def load_datamodule(repo_id: str, name: str, *, cache_dir: Path | None = None) -> "RegisteredDataModuleCfg":
    """Cache-only hydration of one STANDALONE datamodule entry (#128) — never touches the network.

    The datamodule-only half of the two-path contract: the returned pair has no module coupling, and the
    payload's own ``shared_config`` is the only shared configuration applied. Same cache discipline as
    :func:`load` — an uncached component raises with the exact fetch command.
    """
    from interpretune.config.loading import load_datamodule_cfg
    from interpretune.hub.components import resolve_datamodule_config
    from interpretune.registry import DEFAULT_DATAMODULE, RegisteredDataModuleCfg

    body = resolve_datamodule_config(repo_id, name, cache_dir=cache_dir)
    dm_cfg, dm_cls = load_datamodule_cfg(body)
    return RegisteredDataModuleCfg(dm_cfg, dm_cls or DEFAULT_DATAMODULE)


def prefer_ops(*repo_ids: str, replace: bool = False) -> list[str]:
    """Resolve BARE op names to these collections' ops; returns the active precedence list.

    Bundled ops win bare names by default, which is what makes a session work offline and identically for
    everyone. Opting into a hub collection's copy of an op is therefore explicit and per-namespace, never a
    side effect of pulling one::

        it.hub.pull_ops("speediedan/concept_direction_ops")   # fetch (network)
        it.hub.prefer_ops("speediedan/concept_direction_ops") # and now `concept_direction` means theirs

    Fully-qualified names are unaffected: they always address exactly what they name, in both directions.
    Call with no arguments to clear the opt-in. ``IT_OP_PRECEDENCE`` is the env parity for scripted runs.
    """
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    return DISPATCHER.prefer_ops(*repo_ids, replace=replace)


def op_info(op_name: str):
    """Report which collection an op name resolves to now, and what the alternatives are.

    Returns an :class:`~interpretune.analysis.ops.dispatcher.OpResolution` -- provenance, declared collection
    identity, cached revision for hub collections, the other definitions sharing the bare name, and the
    precedence that decided between them. Printing it is the intended use.
    """
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    return DISPATCHER.op_info(op_name)


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
