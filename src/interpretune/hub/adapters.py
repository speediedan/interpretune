"""Loading the ``adapters`` component kind: hub-delivered execution adapters (interpretune#125).

Adapters are the rarest hub artifact and the riskiest, because unlike ops and prompt configs they
compose into the MRO of the module a session runs. They therefore ride the SAME trust gate as every
other hub-resident code path (#255) with no bypass, and they raise rather than warn on refusal: an
adapter that silently fails to load leaves a session whose composition key resolves to something the
user did not ask for, which is worse than not starting.

Three properties this path has deliberately:

- **Manifest-declared surface.** ``adapters.declares`` names the ``Adapter`` members the component
  adds, so the loader knows what the entrypoint was supposed to register BEFORE executing it and can
  say so afterwards when it did not. Discovering the surface by running the code and seeing what
  appeared is how a component gets to register more than it advertised.
- **Cache-only resolution**, like every other component kind: an adapter is loaded from a snapshot an
  explicit ``it.hub.pull`` (or the local-publish bridge) put there, never fetched implicitly.
- **Revision-scoped module names**, so two cached revisions of one component cannot collide in
  ``sys.modules`` as the repo updates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from interpretune.hub.manifest import ComponentManifestError

if TYPE_CHECKING:
    from interpretune.protocol import Adapter


class AdapterComponentError(ComponentManifestError):
    """A component's ``adapters`` kind is absent, malformed, or did not register what it declared."""


def declared_adapters(manifest: dict, source: str = "<component>") -> list[str]:
    """Adapter names a validated manifest declares; empty when the component carries no ``adapters`` kind."""
    if "adapters" not in (manifest.get("kinds") or []):
        return []
    declares = (manifest.get("adapters") or {}).get("declares") or []
    if not declares:  # unreachable via validate_component_manifest; guards hand-built manifests
        raise AdapterComponentError(f"{source}: kind `adapters` declares an empty `adapters.declares` list")
    return [str(d) for d in declares]


def _import_adapter_entrypoint(repo_id: str, snapshot: Path, revision: str, entrypoint: str) -> ModuleType:
    """Import the entrypoint from a cached snapshot under a revision-scoped synthetic module name."""
    sanitized = repo_id.replace("/", "__").replace("-", "_").replace(".", "_")
    module_name = f"it_hub_adapters.{sanitized}.{revision}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = snapshot / entrypoint
    if not path.is_file():
        raise AdapterComponentError(
            f"{repo_id}@{revision[:12]}: manifest declares adapters entrypoint {entrypoint!r}, which is not "
            "present in the snapshot. A partial download, or a manifest naming a path that was never "
            "published, will do this."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # a half-executed module in sys.modules would be returned intact by the next call
        sys.modules.pop(module_name, None)
        raise
    return module


def loaded_adapter_module(repo_id: str, cache_dir: Path | None = None) -> ModuleType:
    """The entrypoint module a previously-loaded adapter component was imported from.

    Adapters can expose more than the compositions they register -- a translation seam, capability
    helpers -- and a caller that has loaded one should be able to reach those without reconstructing
    the revision-scoped module name the loader synthesised. Reaching into ``sys.modules`` by hand is
    the alternative, and it hardcodes a naming scheme that is this module's business.

    Raises if the component has not been loaded in this process: this resolves an already-imported
    module and deliberately executes nothing, so it needs no trust gate of its own.
    """
    from interpretune.hub.components import resolve_component_manifest

    _, _, revision = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    sanitized = repo_id.replace("/", "__").replace("-", "_").replace(".", "_")
    module_name = f"it_hub_adapters.{sanitized}.{revision}"
    module = sys.modules.get(module_name)
    if module is None:
        raise AdapterComponentError(
            f"{repo_id} has not been loaded in this process (looked for {module_name!r}). Call "
            "`load_hub_adapter` first; this accessor deliberately executes nothing itself."
        )
    return module


def load_hub_adapter(repo_id: str, cache_dir: Path | None = None, registry=None) -> list[Adapter]:
    """Load ONE cached adapter component: trust gate, enum extension, entrypoint, registration.

    Returns the :class:`~interpretune.protocol.Adapter` members the component contributed. Idempotent:
    reloading the same cached revision returns the same members without re-executing the entrypoint.

    Raises rather than degrading. Op discovery can drop one bad collection and still give a working
    session; a requested adapter cannot be dropped, because the composition the caller asked for is
    what would silently change.
    """
    from interpretune.adapters import ADAPTER_REGISTRY
    from interpretune.adapters.registration import AdapterProtocol, register_dynamic_adapter
    from interpretune.hub.components import enforce_component_requires, resolve_component_manifest
    from interpretune.hub.trust import ensure_remote_code_trusted

    registry = ADAPTER_REGISTRY if registry is None else registry
    manifest, snapshot, revision = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    source = f"{repo_id}@{revision[:12]}"
    names = declared_adapters(manifest, source=source)
    if not names:
        raise AdapterComponentError(
            f"{source} declares kinds {manifest.get('kinds')!r} and no `adapters` kind, so it publishes no adapters."
        )
    enforce_component_requires(manifest, source=source)
    entrypoint = (manifest.get("adapters") or {}).get("entrypoint")
    if not entrypoint:  # unreachable via validate_component_manifest; guards hand-built manifests
        raise AdapterComponentError(
            f"{source}: kind `adapters` declares no `adapters.entrypoint`, so there is nothing to load."
        )
    # The gate belongs HERE, at the point of execution, and it raises: an adapter composes into the MRO,
    # so "loaded fewer things" is not a degraded success for this kind (#255, #125).
    ensure_remote_code_trusted(
        repo_id, what=f"the adapter entrypoint {entrypoint!r} (it composes into the session MRO)"
    )

    members = [register_dynamic_adapter(name, source=repo_id) for name in names]
    before = set(registry.keys())
    module = _import_adapter_entrypoint(repo_id, snapshot, revision, entrypoint)
    for _, member in vars(module).items():
        if isinstance(member, type) and isinstance(member, AdapterProtocol) and hasattr(member, "register_adapter_ctx"):
            member.register_adapter_ctx(registry)
    added = set(registry.keys()) - before
    unregistered = [m for m in members if not any(m in key for key in added)]
    if unregistered and added:
        raise AdapterComponentError(
            f"{source}: entrypoint {entrypoint!r} registered {len(added)} composition(s), none of which use "
            f"the declared adapter(s) {[m.name for m in unregistered]!r}. The manifest advertises the adapter "
            "surface a component adds; code that registers something else is a mismatch worth failing on."
        )
    if not added:
        raise AdapterComponentError(
            f"{source}: entrypoint {entrypoint!r} registered no compositions. An adapters component whose "
            f"entrypoint registers nothing declares {names!r} and delivers nothing."
        )
    return members
