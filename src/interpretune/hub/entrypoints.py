"""Snapshot-entrypoint class resolution for module/datamodule payloads.

A payload ``class_path`` normally imports from the environment
(:func:`interpretune.utils.import_utils.instantiate_class`). Once an experiment moves out of the
tree, its payloads address the component's own entrypoint file instead (for example
``rte_boolq.RTEBoolqModule``), which is importable only from the cached snapshot. This module is
the one place that bridges the two: try the environment first, and only on failure consult the
cached-component manifests for an entrypoint stem match.

The snapshot import itself (trust gate at execution, revision-scoped synthetic name, missing-file
error naming the pull) is shared with the promptconfigs entrypoint path, which delegates here, so
there is exactly one guarded helper rather than two.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

from interpretune.utils.exceptions import MisconfigurationException

_HUB_MODULE_PREFIX = "it_hub_components."


def synthetic_entrypoint_name(repo_id: str, revision: str) -> str:
    """Revision-scoped module name for a snapshot entrypoint.

    Incorporates org, repo AND revision so definitions from different cached revisions of the same
    repo can never collide in ``sys.modules`` as repos update.
    """
    sanitized = repo_id.replace("/", "__").replace("-", "_").replace(".", "_")
    return f"it_hub_components.{sanitized}.{revision}"


def import_snapshot_entrypoint(
    repo_id: str, entrypoint: str, *, cache_dir: Path | None = None, what: str
) -> ModuleType:
    """Import a cached component's entrypoint file under its revision-scoped module name.

    Cache-only: the component must already be cached via an explicit pull or the local-publish
    bridge. The trust gate belongs here, at the point of execution: this is where interpretune
    runs Python that came from a hub repo. ``what`` names the entrypoint kind in the refusal
    (each caller states its own context, so refusal wording stays pinned per path).
    """
    from interpretune.hub.components import resolve_component_manifest
    from interpretune.hub.trust import ensure_remote_code_trusted

    _, snapshot, revision = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    module_name = synthetic_entrypoint_name(repo_id, revision)
    if module_name in sys.modules:
        return sys.modules[module_name]
    ensure_remote_code_trusted(repo_id, what=what)
    if not (snapshot / entrypoint).is_file():
        raise FileNotFoundError(
            f"{repo_id}@{revision[:12]}: manifest declares entrypoint {entrypoint!r}, which is not "
            f"present in the snapshot. A manifest-only fetch leaves exactly this state: run "
            f"interpretune.hub.pull({repo_id!r}) to materialize the entrypoint (loading never downloads)."
        )
    spec = importlib.util.spec_from_file_location(module_name, snapshot / entrypoint)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _ensure_parent_packages(module_name)
    return module


def _ensure_parent_packages(fullname: str) -> None:
    """Create the namespace parents of a directly-registered synthetic module and bind the child.

    A normal import sets each parent attribute as it loads each level; stuffing only the full
    dotted name into ``sys.modules`` skips that, so parent-attribute traversal (dill's function
    location, ``mock.patch`` string targets) cannot reach the module and serializers fall back to
    pickling by value. Bindings follow the standard rule (latest import wins); identity checks at
    use sites still distinguish revisions, so a stale binding degrades to the old fallback rather
    than a silent wrong module.
    """
    parts = fullname.split(".")
    for depth in range(1, len(parts)):
        parent_name = ".".join(parts[:depth])
        if parent_name not in sys.modules:
            parent = ModuleType(parent_name)
            parent.__path__ = []
            sys.modules[parent_name] = parent
    # Bind unconditionally, not only when unset: after an eviction and re-import,
    # ``sys.modules`` holds the new module while a stale parent attribute still points at the
    # old one, and then attribute traversal (dill location, ``mock.patch`` targets) and
    # ``sys.modules`` reads reach different objects with no error. The binding always agrees
    # with ``sys.modules`` this way; identity checks at use sites still tell revisions apart.
    for depth in range(1, len(parts)):
        parent = sys.modules[".".join(parts[:depth])]
        child = sys.modules.get(".".join(parts[: depth + 1]))
        if child is not None:
            setattr(parent, parts[depth], child)


class _HubSnapshotFinder(MetaPathFinder):
    """Resolve revision-scoped snapshot modules from the on-disk components cache.

    Snapshot entrypoints execute under synthetic names (``it_hub_components.<repo>.<rev>``) that no
    ``sys.path`` entry provides, so a fresh process -- a spawn worker unpickling a Hub-loaded class,
    a notebook kernel that never ran the loader -- cannot import them back by reference. Without a
    finder that import fails, and serializers fall back to pickling Hub classes by value, which the
    revision scope exists to make meaningless. The finder closes the loop: same machine, same cache,
    same classes, with the trust gate checked before any execution, exactly as on the loader path.

    Only the default components cache is consulted (plus whatever ``IT_COMPONENTS_HUB_CACHE`` names
    in the importing process, since workers inherit the environment). Cache-only throughout: never
    touches the network. Ambiguous stems refuse rather than guess, mirroring
    :func:`find_entrypoint_owner`.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "it_hub_components" or (fullname.startswith(_HUB_MODULE_PREFIX) and fullname.count(".") < 2):
            return ModuleSpec(fullname, loader=None, is_package=True)
        if not fullname.startswith(_HUB_MODULE_PREFIX):
            return None
        from interpretune.hub.cache import scan_cached_repos
        from interpretune.hub.components import IT_COMPONENTS_HUB_CACHE as COMPONENTS_DEFAULT
        from interpretune.hub.components import cached_component_revisions, resolve_component_manifest
        from interpretune.hub.trust import ensure_remote_code_trusted

        cache_default = Path(COMPONENTS_DEFAULT)
        for repo in sorted(scan_cached_repos(cache_default), key=lambda r: r.repo_id):
            for rev in cached_component_revisions(repo.repo_id):
                if synthetic_entrypoint_name(repo.repo_id, rev) != fullname:
                    continue
                manifest, snapshot, _ = resolve_component_manifest(repo.repo_id, revision=rev)
                rels = []
                module_section = manifest.get("module") or {}
                if module_section.get("entrypoint"):
                    rels.append(module_section["entrypoint"])
                for entry in (manifest.get("datamodules") or {}).values():
                    if isinstance(entry, dict) and entry.get("entrypoint"):
                        rels.append(entry["entrypoint"])
                rels = sorted(set(rels))
                if len(rels) != 1:
                    return None
                ensure_remote_code_trusted(repo.repo_id, what=f"the component entrypoint {rels[0]!r}")
                return importlib.util.spec_from_file_location(fullname, snapshot / rels[0])
        return None


_FINDER_INSTALLED = False


def install_hub_module_finder() -> None:
    """Append the snapshot finder to ``sys.meta_path`` (idempotent, microsecond cost).

    Called at ``interpretune`` import time so fresh worker processes inherit it through the normal
    import chain: anything that can import interpretune can restore a Hub-loaded class by reference.
    Appended last, so it only ever sees names every other finder declined.
    """
    global _FINDER_INSTALLED
    if _FINDER_INSTALLED or any(isinstance(finder, _HubSnapshotFinder) for finder in sys.meta_path):
        return
    sys.meta_path.append(_HubSnapshotFinder())
    _FINDER_INSTALLED = True


def find_entrypoint_owner(stem: str, *, cache_dir: Path | None = None) -> tuple[str, str] | None:
    """The ``(repo_id, entrypoint)`` whose module/datamodule entrypoint stem is ``stem``.

    Returns None when no cached component declares it. Refuses (rather than guessing) when more
    than one does: deterministic alphabetical choice would silently bind whichever repo sorts
    first. Manifests that fail to read are skipped; a component too corrupt to declare its
    entrypoints cannot supply them, and the caller reports the miss by name either way.
    """
    from interpretune.hub.cache import scan_cached_repos
    from interpretune.hub.components import IT_COMPONENTS_HUB_CACHE, resolve_component_manifest

    owners: list[tuple[str, str]] = []
    root = Path(cache_dir) if cache_dir is not None else Path(IT_COMPONENTS_HUB_CACHE)
    for repo in sorted(scan_cached_repos(root), key=lambda r: r.repo_id):
        try:
            manifest, _, _ = resolve_component_manifest(repo.repo_id, cache_dir=cache_dir)
        except Exception:
            continue
        rels = []
        module_section = manifest.get("module") or {}
        if module_section.get("entrypoint"):
            rels.append(module_section["entrypoint"])
        for entry in (manifest.get("datamodules") or {}).values():
            if isinstance(entry, dict) and entry.get("entrypoint"):
                rels.append(entry["entrypoint"])
        for rel in rels:
            if Path(rel).stem == stem and (repo.repo_id, rel) not in owners:
                owners.append((repo.repo_id, rel))
    if len(owners) > 1:
        claimed = ", ".join(sorted(f"{repo} ({rel})" for repo, rel in owners))
        raise MisconfigurationException(
            f"Entrypoint stem {stem!r} is declared by more than one cached component: {claimed}. "
            "Resolving by alphabetical order would silently bind whichever repo sorts first, so this "
            "refuses instead; disambiguate by importing the intended component explicitly."
        )
    return owners[0] if owners else None


def instantiate_hub_aware_class(
    init: dict[str, Any],
    args: Any | tuple[Any, ...] | None = None,
    import_only: bool = False,
    *,
    cache_dir: Path | None = None,
) -> Any:
    """Like :func:`interpretune.utils.import_utils.instantiate_class`, plus snapshot resolution.

    For a single-segment module stem that a cached component owns, the snapshot resolves and the
    environment is not consulted: the pin says which code runs, and an environment-first order
    would let any importable same-named module silently shadow the pinned revision. The
    environment is consulted only when no cached component declares the stem (existing behavior
    and errors preserved exactly there), and when both resolve to different files the call is
    refused naming both, rather than binding whichever the import order favors. Multi-segment
    stems never consult the map. AttributeError on an otherwise importable module likewise
    stands, since the map is irrelevant to it.
    """
    from interpretune.utils import instantiate_class

    class_path = init.get("class_path", None) if isinstance(init, dict) else None
    if not isinstance(class_path, str) or "." not in class_path:
        return instantiate_class(init, args, import_only=import_only)
    module_part, _, class_name = class_path.rpartition(".")
    if "." in module_part:
        return instantiate_class(init, args, import_only=import_only)
    owner = find_entrypoint_owner(module_part, cache_dir=cache_dir)
    if owner is None:
        return instantiate_class(init, args, import_only=import_only)
    repo_id, entrypoint = owner
    module = import_snapshot_entrypoint(
        repo_id, entrypoint, cache_dir=cache_dir, what=f"the component entrypoint {entrypoint!r}"
    )
    try:
        args_class = getattr(module, class_name)
    except AttributeError:
        raise MisconfigurationException(
            f"Could not resolve {class_path!r}: cached component {repo_id!r} declares entrypoint "
            f"{entrypoint!r}, but it defines no attribute {class_name!r}."
        ) from None
    _refuse_environment_divergence(module_part, repo_id, entrypoint, cache_dir=cache_dir)
    if import_only:
        return args_class
    if args and not isinstance(args, tuple):
        args = (args,)
    kwargs = init.get("init_args", {})
    return args_class(**kwargs) if not args else args_class(*args, **kwargs)


def _refuse_environment_divergence(stem: str, repo_id: str, entrypoint: str, *, cache_dir: Path | None = None) -> None:
    """Refuse when the environment resolves an owned stem to a different file than the snapshot.

    An older installed copy, a sibling checkout, or a stale tree on ``PYTHONPATH`` would otherwise
    shadow the pinned revision with no error. Same file (hardlink, symlink, or shared checkout)
    is not divergence and proceeds.
    """
    import importlib
    import os

    try:
        env_module = importlib.import_module(stem)
    except ImportError:
        return
    env_file = getattr(env_module, "__file__", None)
    if env_file is None:
        return
    snapshot_file = _snapshot_entrypoint_file(repo_id, entrypoint, cache_dir=cache_dir)
    if snapshot_file is None:
        return
    try:
        same = os.path.samefile(env_file, snapshot_file)
    except OSError:
        return
    if not same:
        raise MisconfigurationException(
            f"Entrypoint stem {stem!r} resolves in the environment to {env_file} and in the "
            f"components cache to {snapshot_file} (component {repo_id!r}): two different files "
            "claim one name, so binding either silently would shadow the other. Remove or rename "
            "one of them rather than relying on import order."
        )


def _snapshot_entrypoint_file(repo_id: str, entrypoint: str, *, cache_dir: Path | None = None) -> Path | None:
    """The on-disk entrypoint file a cached component declares, or None when absent."""
    from interpretune.hub.components import resolve_component_manifest

    try:
        _, snapshot, _ = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    except Exception:
        return None
    candidate = snapshot / entrypoint
    return candidate if candidate.is_file() else None
