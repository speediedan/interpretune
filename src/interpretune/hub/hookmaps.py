"""The ``hookmaps`` component kind: per-architecture component maps published as hub DATA.

A component map says where each bridge component of one architecture lives in its module tree. The documents are
venngram's (``venngram.maps.schema``), read through venngram's reader, so any schema venngram reads is accepted; the
vocabulary they serve is ``docs/activation_point_vocabulary.md``. Interpretune bundles no map: it reads venngram's
registry, and a ``hookmaps`` component is the override channel, registering its maps in process ahead of the
registries, for an architecture the registry does not serve or to republish one so a consumer can check the two agree.

Kind contract, mirroring ``datamodule`` rather than ``adapters``: the payload is YAML that parses into a
frozen dataclass, nothing executes, so the trust gate is not consulted. The publisher parses every declared
document (a broken map is refused at publish); the loader is cache-only; ``it.hub.pull`` /
``it.hub.pull_hookmaps`` are the verbs that download.
"""

from __future__ import annotations

from pathlib import Path

from interpretune.hub.manifest import ComponentManifestError, IT_COMPONENT_MANIFEST


class HookMapComponentError(ComponentManifestError):
    """A ``hookmaps`` component's documents cannot be registered as declared."""


def declared_hookmap_files(manifest: dict, source: str = "<manifest>") -> list[str]:
    """The manifest's declared component-map documents, or a refusal naming the missing kind."""
    if "hookmaps" not in (manifest.get("kinds") or []):
        raise HookMapComponentError(
            f"{source}: component does not declare the `hookmaps` kind (kinds: {manifest.get('kinds')!r}); "
            "there are no component maps to load."
        )
    return list((manifest.get("hookmaps") or {}).get("files") or [])


def resolve_cached_hookmap_files(repo_id: str, cache_dir: Path | None = None) -> tuple[list[Path], str]:
    """CACHE-ONLY: the snapshot paths of a cached component's declared documents, plus the revision."""
    from interpretune.hub.components import enforce_component_requires, resolve_component_manifest

    manifest, snapshot, revision = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    source = f"{repo_id}@{revision[:12]}"
    enforce_component_requires(manifest, source=source)
    paths = []
    for rel in declared_hookmap_files(manifest, source):
        path = snapshot / rel
        if not path.is_file():
            raise HookMapComponentError(
                f"{source}: manifest declares hookmaps document {rel!r}, which is not present in the snapshot. "
                f"A manifest-only fetch leaves exactly this state: run interpretune.hub.pull_hookmaps({repo_id!r}) "
                "to materialize the documents."
            )
        paths.append(path)
    return paths, revision


def _comparable(cmap) -> tuple:
    """What two maps for one architecture must share to be the same map: per row its module, kind, ``optional`` and
    layers, the properties, and the declared kinds and stacks (a kind's output rule and a stack's template decide how a
    row resolves as much as the row does). A row's ``layout`` and the document's provenance are not compared."""
    rows = {name: (e.module, e.kind, e.optional, e.layers) for name, e in cmap.components.items()}
    return rows, dict(cmap.properties), dict(cmap.kinds), dict(cmap.stacks)


def load_hub_hookmaps(repo_id: str, cache_dir: Path | None = None, *, replace: bool = False) -> list:
    """Parse and register every document of ONE cached ``hookmaps`` component; returns the venngram maps.

    Registration goes through venngram's in-process registry (``venngram.maps.register``), which every lookup reads
    ahead of the registries, so a hub map is indistinguishable from a registry one to every consumer
    (`component_map_for`, the point resolver, the derived hook mapping, a tunable view). The collision rule: an
    architecture that already has a map (registered in process, or indexed by a registry) must be mapped IDENTICALLY
    or the load is refused naming both sources; ``replace=True`` makes the shadowing deliberate. Silent replacement
    is the one thing this kind must not do, since a map is what every activation-point address for that
    architecture resolves through.
    """
    from dataclasses import replace as _dc_replace

    from venngram.maps import component_map_for, load_component_map, register

    paths, revision = resolve_cached_hookmap_files(repo_id, cache_dir=cache_dir)
    loaded = []
    for path in paths:
        cmap = _dc_replace(load_component_map(path), source=f"{repo_id}@{revision[:12]}:{path.name}")
        try:
            existing = component_map_for(cmap.architecture)
        except KeyError:
            existing = None
        if existing is not None and not replace:
            if _comparable(existing) != _comparable(cmap):
                raise HookMapComponentError(
                    f"{cmap.source}: map for {cmap.architecture!r} disagrees with the one already served from "
                    f"{existing.source!r}. Two sources for one architecture must agree; pass replace=True to shadow "
                    "the served map deliberately."
                )
            loaded.append(existing)
            continue
        register(cmap)
        loaded.append(cmap)
    return loaded


def pull_hookmap_files(
    repo_id: str, revision: str | None = None, cache_dir: Path | None = None, token: str | None = None
) -> tuple[list[Path], str]:
    """Manifest-first fetch of a ``hookmaps`` component's documents, pinned to the manifest's commit."""
    from interpretune.hub.components import pull_component_manifest, pull_component_payloads

    manifest, commit = pull_component_manifest(repo_id, revision=revision, cache_dir=cache_dir, token=token)
    declared = declared_hookmap_files(manifest, source=f"{repo_id}@{commit[:12]}")
    if IT_COMPONENT_MANIFEST in declared:  # unreachable via validate_component_manifest; guards hand-built ones
        raise HookMapComponentError(f"{repo_id}: `hookmaps.files` lists the manifest itself.")
    paths = pull_component_payloads(
        repo_id, {"hookmaps": {"files": declared}}, commit, cache_dir=cache_dir, token=token
    )
    return paths, commit


__all__ = [
    "HookMapComponentError",
    "declared_hookmap_files",
    "load_hub_hookmaps",
    "pull_hookmap_files",
    "resolve_cached_hookmap_files",
]
