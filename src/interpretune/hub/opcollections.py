"""Manifest-first resolution for the ``ops`` component kind (op collections).

The ops kind reaches manifest-first parity with the module kind here (``.components``): a collection's
``it_component.yaml`` is read FIRST and declares, through ``ops.files``, exactly which YAMLs in the repo are
op definitions. Every follow-up fetch is pinned to the commit the manifest resolved to, so a repo updated
mid-resolution cannot hand back op definitions from two different revisions.

This replaces a blind ``*.yaml`` glob over the cached snapshot, and the difference is a contract change rather
than an optimization:

- **A repo's non-op YAML is no longer op input.** The glob fed every YAML to the op compiler, so a README
  fixture, a config sample or a CI workflow copied into the repo became a load failure with global blast
  radius. Declaring the op files is the only way a collection can carry anything else.
- **``countDownloads: path:"it_component.yaml"`` becomes true for ops too.** Registration PR A claims the
  manifest anchors download counting for every kind; under the glob no ops resolution read the manifest at
  all, so the claim was inaccurate for this kind specifically.
- **A collection's ``requires:`` window is enforced before its ops load**, using the same machinery
  components use, so an incompatible collection is skipped rather than compiled into a session that cannot
  run it.

The caller supplies the failure policy: functions here raise, and op discovery routes those raises through
``op_load_failure`` so one malformed collection warns (or, under ``IT_STRICT_OP_LOAD=1``, raises) instead of
taking down every other op in the process.
"""

from __future__ import annotations

from pathlib import Path

from interpretune.hub.cache import IT_ANALYSIS_HUB_CACHE
from interpretune.hub.manifest import IT_COMPONENT_MANIFEST, ComponentManifestError, load_component_manifest


class OpCollectionManifestError(ComponentManifestError):
    """An op collection's manifest is absent, declares no ops, or declares op files that are not present."""


def declared_op_files(manifest: dict, source: str = "<collection>") -> list[str]:
    """Repo-relative op-definition YAML paths a validated manifest declares.

    Returns an empty list when the manifest declares no ``ops`` kind: a component repo may legitimately carry
    modules or prompt configs and no ops at all, which is not a failure.
    """
    if "ops" not in (manifest.get("kinds") or []):
        return []
    files = (manifest.get("ops") or {}).get("files") or []
    if not files:  # unreachable via validate_component_manifest; guards hand-built manifests
        raise OpCollectionManifestError(f"{source}: kind `ops` declares an empty `ops.files` list")
    return [str(f) for f in files]


def resolve_cached_op_files(snapshot_dir: Path, source: str) -> list[Path]:
    """Manifest-routed op YAMLs for one CACHED collection snapshot; never touches the network.

    Raises when the snapshot has no manifest (the well-formed-ops-repo contract), when the manifest is
    malformed, when its ``requires:`` window is unsatisfied, or when a declared op file is missing. Returns an
    empty list for a cached component that simply declares no ops.
    """
    from interpretune.hub.components import enforce_component_requires

    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / IT_COMPONENT_MANIFEST
    if not manifest_path.is_file():
        raise OpCollectionManifestError(
            f"{source}: no {IT_COMPONENT_MANIFEST} in the cached snapshot, so no op definitions can be "
            "identified. Op discovery is manifest-routed: a well-formed ops repo declares its op YAMLs in "
            f"`ops.files` with `kinds: [ops]`. Re-publish the collection with a manifest (or, if this repo is "
            "not an op collection, it does not belong in the analysis-ops cache)."
        )
    manifest = load_component_manifest(manifest_path)
    enforce_component_requires(manifest, source=source)
    resolved = []
    for rel in declared_op_files(manifest, source=source):
        path = snapshot_dir / rel
        if not path.is_file():
            raise OpCollectionManifestError(
                f"{source}: manifest declares op file {rel!r}, which is not present in the snapshot. "
                "A partial download or a manifest listing a path that was never published will do this."
            )
        resolved.append(path)
    return resolved


def pull_op_collection(
    repo_id: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> tuple[list[Path], str]:
    """Manifest-first fetch of ONE op collection's definition files; returns ``(paths, resolved_commit)``.

    The manifest is fetched first and every declared op file is then fetched at the commit the manifest
    resolved to (never at a moving ref), mirroring :func:`~interpretune.hub.components.pull_component_config`.
    Fetching a collection does not load its ops: the dispatcher picks them up from the cache on its next
    load, so this stays an explicit network call with no compile side effects.
    """
    import yaml
    from huggingface_hub import hf_hub_download

    from interpretune.hub.components import _TELEMETRY, _snapshot_revision, enforce_component_requires
    from interpretune.hub.manifest import validate_component_manifest

    root = str(cache_dir or IT_ANALYSIS_HUB_CACHE)
    manifest_path = Path(
        hf_hub_download(repo_id, IT_COMPONENT_MANIFEST, revision=revision, cache_dir=root, token=token, **_TELEMETRY)
    )
    commit = _snapshot_revision(manifest_path)
    manifest = validate_component_manifest(
        yaml.safe_load(manifest_path.read_text(encoding="utf-8")), source=f"{repo_id}@{commit[:12]}"
    )
    enforce_component_requires(manifest, source=f"{repo_id}@{commit[:12]}")
    files = declared_op_files(manifest, source=f"{repo_id}@{commit[:12]}")
    if not files:
        raise OpCollectionManifestError(
            f"{repo_id}@{commit[:12]} declares kinds {manifest.get('kinds')!r} and no `ops` kind, so it "
            "publishes no op definitions."
        )
    paths = [
        Path(
            hf_hub_download(
                repo_id,
                rel,
                revision=commit,  # pinned: a collection materializes from exactly one revision
                cache_dir=root,
                token=token,
                **_TELEMETRY,
            )
        )
        for rel in files
    ]
    # An op collection's YAMLs name Python implementations inside the repo; the impl modules have to be
    # local before the compiler resolves them, and they are not enumerated by the manifest.
    _pull_collection_impl_modules(repo_id, commit, root, token, paths)
    _record_or_report_pin(repo_id, revision, commit, root)
    return paths, commit


def _record_or_report_pin(repo_id: str, revision: str | None, commit: str, cache_root: str) -> None:
    """Make a pinned pull durable, or say when an unpinned pull will not move what discovery loads.

    ``revision=None`` and ``revision="main"`` both mean "the moving default" and record nothing —
    freezing a user who explicitly asked for ``main`` would be the opposite surprise. Every other
    explicit revision (commit, tag, branch) pins the RESOLVED commit: op discovery loads that
    revision from now on, beating ``refs/main``, until the pin is moved (re-pull at another
    revision) or released (``it.hub.unpin_ops``). See ``interpretune.hub.pins`` (#334).
    """
    from interpretune.hub.pins import read_op_pin, record_op_pin

    if revision is not None and revision != "main":
        record_op_pin(repo_id, commit, requested_revision=revision, cache_root=cache_root)
        return
    pin = read_op_pin(repo_id, cache_root=cache_root)
    if pin is not None and pin["commit"] != commit:
        from interpretune.utils.logging import rank_zero_warn

        rank_zero_warn(
            f"{repo_id!r} resolved `main` to {commit[:12]}, but the collection is pinned to "
            f"{pin['commit'][:12]} and op discovery keeps loading the pin. Move the pin with "
            f"it.hub.pull_ops({repo_id!r}, revision={commit!r}) or release it with "
            f"it.hub.unpin_ops({repo_id!r})."
        )


def _pull_collection_impl_modules(
    repo_id: str, commit: str, cache_dir: str, token: str | None, op_yaml_paths: list[Path]
) -> None:
    """Fetch the in-repo Python modules a collection's op YAMLs point their implementations at.

    Resolved from the declared op YAMLs rather than from a manifest field: the implementation reference
    already lives in the op definition, so a second declaration could disagree with it.
    """
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    from interpretune.hub.components import _TELEMETRY

    modules: set[str] = set()
    for path in op_yaml_paths:
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(body, dict):
            continue
        for op in body.values():
            impl = op.get("implementation") if isinstance(op, dict) else None
            if isinstance(impl, str) and "." in impl:
                # `pkg.module.func` -> `pkg/module.py`; only in-repo relative modules can be fetched
                modules.add(impl.rsplit(".", 1)[0].replace(".", "/") + ".py")
    for rel in sorted(modules):
        try:
            hf_hub_download(repo_id, rel, revision=commit, cache_dir=cache_dir, token=token, **_TELEMETRY)
        except EntryNotFoundError:
            # An implementation living in an installed package (interpretune's own, say) is normal and
            # resolvable by import; only a repo-relative module needs fetching.
            continue


__all__ = ["OpCollectionManifestError", "declared_op_files", "pull_op_collection", "resolve_cached_op_files"]
