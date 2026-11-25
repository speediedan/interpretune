"""Loading a component's helper modules: the ``helpers`` kind (``it.hub.load_helpers``).

A component can ship Python that is neither its entrypoint nor an op collection: the analysis and plotting helpers one
experiment's notebook uses. This is the gated way to import them. It is cache-only, checks the trust decision before
anything executes, and registers modules under a package scoped by repository and revision, so two components (or two
revisions of one) never collide in ``sys.modules`` and nothing is added to ``sys.path``.

Modules one component declares can import each other relatively (``from .other import thing``). Only declared names
resolve: a file the manifest does not list is not importable this way.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Mapping
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

#: Parent package of every loaded helper module: ``it_hub_helpers.<org>__<repo>.<revision>.<name>``.
HUB_HELPERS_PACKAGE = "it_hub_helpers"


class HelperModuleError(ValueError):
    """A helper module was asked for that its component does not declare."""


class _HelperPackage(NamedTuple):
    """What one component revision's package resolves its submodules from."""

    repo_id: str
    revision: str
    snapshot: Path
    modules: Mapping[str, str]


_PACKAGES: dict[str, _HelperPackage] = {}


def _declared(repo_id: str, revision: str, modules: Mapping[str, str], name: str) -> str:
    """The declared path of helper module ``name``, or a refusal listing what the component declares."""
    if name not in modules:
        listed = ", ".join(sorted(modules)) or "none (the manifest has no kind `helpers`)"
        raise HelperModuleError(
            f"{repo_id}@{revision[:12]} declares no helper module {name!r}. Declared helper modules: {listed}."
        )
    return modules[name]


class _HelperModuleFinder(MetaPathFinder):
    """Resolve submodules of the registered helper packages to their declared files, and nothing else."""

    def find_spec(self, fullname, path=None, target=None):
        package, _, name = fullname.rpartition(".")
        entry = _PACKAGES.get(package)
        if entry is None or name not in entry.modules:
            return None
        from interpretune.hub.components import _cached_declared_file
        from interpretune.hub.trust import ensure_remote_code_trusted

        rel = entry.modules[name]
        # checked here rather than only in load_helpers, so a relative import between declared modules is gated too
        ensure_remote_code_trusted(entry.repo_id, what=f"the helper module {rel!r}")
        file = _cached_declared_file(
            entry.snapshot,
            rel,
            repo_id=entry.repo_id,
            what=f"helper module {name!r}",
            fetch=f"interpretune.hub.pull({entry.repo_id!r}, revision={entry.revision!r})",
        )
        return importlib.util.spec_from_file_location(fullname, file)


_FINDER = _HelperModuleFinder()


def _register_package(repo_id: str, revision: str, snapshot: Path, modules: Mapping[str, str]) -> str:
    """Register the revision-scoped package the component's helper modules import under, and return its name."""
    sanitized = repo_id.replace("/", "__").replace("-", "_").replace(".", "_")
    package = f"{HUB_HELPERS_PACKAGE}.{sanitized}.{revision}"
    registered = _PACKAGES.get(package)
    if registered is not None and registered.repo_id != repo_id:
        raise HelperModuleError(
            f"{repo_id}@{revision[:12]} would load under {package!r}, where {registered.repo_id} is already registered "
            "in this process: the two repository names differ only in punctuation. Load them in separate processes."
        )
    # a revision identifies its content, so a second cache directory holding it only moves where files are read from
    _PACKAGES[package] = _HelperPackage(repo_id, revision, snapshot, dict(modules))
    if _FINDER not in sys.meta_path:
        sys.meta_path.append(_FINDER)
    parts = package.split(".")
    for depth in range(1, len(parts) + 1):
        parent = ".".join(parts[:depth])
        if parent not in sys.modules:
            module = ModuleType(parent)
            module.__path__ = []  # a package with no search path: submodules come from the finder only
            module.__spec__ = ModuleSpec(parent, loader=None, is_package=True)
            sys.modules[parent] = module
            if depth > 1:
                setattr(sys.modules[".".join(parts[: depth - 1])], parts[depth - 1], module)
    return package


def load_helpers(repo_id: str, name: str, *, revision: str | None = None, cache_dir: Path | None = None) -> ModuleType:
    """Import helper module ``name`` of a cached component and return it (cache-only, never downloads).

    Fetch the component first with ``it.hub.pull(repo_id, revision=...)``, which executes nothing, and opt in with
    ``IT_TRUST_REMOTE_CODE=1``: importing a helper module runs the publisher's code in this process. ``revision`` is
    resolved as for ``it.hub.load``: the one given, else the recorded pin, else the cached ``main``.

    Raises:
        HelperModuleError: the component declares no helper module of that name.
        RemoteCodeNotTrustedError: remote code is not trusted in this session.
        ComponentFileNotCachedError: the declared file is not in the cached snapshot.
    """
    from interpretune.hub.components import resolve_component_manifest
    from interpretune.hub.trust import ensure_remote_code_trusted

    manifest, snapshot, resolved = resolve_component_manifest(repo_id, cache_dir=cache_dir, revision=revision)
    modules = (manifest.get("helpers") or {}).get("modules") or {}
    rel = _declared(repo_id, resolved, modules, name)
    # consulted on every call, so an opt-out set later in the session refuses an already imported module too
    ensure_remote_code_trusted(repo_id, what=f"the helper module {rel!r}")
    package = _register_package(repo_id, resolved, snapshot, modules)
    return importlib.import_module(f"{package}.{name}")
