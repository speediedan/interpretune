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
from pathlib import Path
from types import ModuleType
from typing import Any

from interpretune.utils.exceptions import MisconfigurationException


def synthetic_entrypoint_name(repo_id: str, revision: str) -> str:
    """Revision-scoped module name for a snapshot entrypoint.

    Incorporates org, repo AND revision so definitions from different cached revisions of the same
    repo can never collide in ``sys.modules`` as repos update.
    """
    sanitized = repo_id.replace("/", "__").replace("-", "_").replace(".", "_")
    return f"it_hub_components.{sanitized}.{revision}"


def import_snapshot_entrypoint(repo_id: str, entrypoint: str, *, cache_dir: Path | None = None) -> ModuleType:
    """Import a cached component's entrypoint file under its revision-scoped module name.

    Cache-only: the component must already be cached via an explicit pull or the local-publish
    bridge. The trust gate belongs here, at the point of execution: this is where interpretune
    runs Python that came from a hub repo.
    """
    from interpretune.hub.components import resolve_component_manifest
    from interpretune.hub.trust import ensure_remote_code_trusted

    _, snapshot, revision = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    module_name = synthetic_entrypoint_name(repo_id, revision)
    if module_name in sys.modules:
        return sys.modules[module_name]
    ensure_remote_code_trusted(repo_id, what=f"the component entrypoint {entrypoint!r}")
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
    return module


def find_entrypoint_owner(stem: str, *, cache_dir: Path | None = None) -> tuple[str, str] | None:
    """The ``(repo_id, entrypoint)`` whose module/datamodule entrypoint stem is ``stem``.

    Returns None when no cached component declares it. Refuses (rather than guessing) when more
    than one does: deterministic alphabetical choice would silently bind whichever repo sorts
    first. Manifests that fail to read are skipped; a component too corrupt to declare its
    entrypoints cannot supply them, and the caller reports the miss by name either way.
    """
    from interpretune.hub.cache import IT_COMPONENTS_HUB_CACHE, scan_cached_repos
    from interpretune.hub.components import resolve_component_manifest

    owners: list[tuple[str, str]] = []
    for repo in sorted(scan_cached_repos(Path(cache_dir or IT_COMPONENTS_HUB_CACHE)), key=lambda r: r.repo_id):
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
    """Like :func:`interpretune.utils.import_utils.instantiate_class`, plus snapshot fallback.

    The environment import runs first and wins whenever it succeeds, so behavior is unchanged for every resolvable path
    and the transition is deterministic. Only an ImportError on a single-segment module stem consults the cached-
    component entrypoint map; multi-segment stems never do (their failure stands as the existing error), and a stem no
    component declares fails naming both the import error and the consulted stems. AttributeError on an otherwise
    importable module likewise stands, since the map is irrelevant to it.
    """
    from interpretune.utils import instantiate_class

    try:
        return instantiate_class(init, args, import_only=import_only)
    except (ImportError, AttributeError) as exc:
        first_error = exc
    class_path = init.get("class_path", None) if isinstance(init, dict) else None
    if not isinstance(class_path, str) or "." not in class_path:
        raise first_error
    module_part, _, class_name = class_path.rpartition(".")
    if "." in module_part:
        raise first_error
    owner = find_entrypoint_owner(module_part, cache_dir=cache_dir)
    if owner is None:
        raise MisconfigurationException(
            f"Could not resolve {class_path!r}: the environment import failed ({first_error}), and no "
            f"cached component declares a module/datamodule entrypoint named {module_part!r}. If this "
            "names a hub component, cache it first with an explicit it.hub.pull(...)."
        ) from first_error
    repo_id, entrypoint = owner
    module = import_snapshot_entrypoint(repo_id, entrypoint, cache_dir=cache_dir)
    try:
        args_class = getattr(module, class_name)
    except AttributeError:
        raise MisconfigurationException(
            f"Could not resolve {class_path!r}: cached component {repo_id!r} declares entrypoint "
            f"{entrypoint!r}, but it defines no attribute {class_name!r}."
        ) from None
    if import_only:
        return args_class
    if args and not isinstance(args, tuple):
        args = (args,)
    kwargs = init.get("init_args", {})
    return args_class(**kwargs) if not args else args_class(*args, **kwargs)
