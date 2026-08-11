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


class ComponentRequirementError(ImportError):
    """The current environment cannot satisfy a component manifest's ``requires:`` block."""


def enforce_component_requires(manifest: dict, source: str = "<component>") -> None:
    """Enforce a manifest's ``requires:`` block against the current environment (fail informatively).

    Three failure modes, each with its own message: an unsatisfied ``interpretune`` version specifier,
    an adapter name this interpretune does not know (usually means a newer interpretune is required —
    NOT that every listed adapter's backend must be importable, since configs compose subsets), and a
    ``pip`` requirement that is not installed / does not satisfy its specifier.
    """
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    req = manifest.get("requires") or {}
    it_spec = req.get("interpretune")
    if it_spec:
        try:
            it_version = _pkg_version("interpretune")
        except Exception:
            from interpretune import __version__ as it_version  # raw-checkout fallback
        if not SpecifierSet(str(it_spec)).contains(it_version, prereleases=True):
            raise ComponentRequirementError(
                f"{source}: requires interpretune {it_spec!r} but {it_version!r} is installed. "
                "(If this version looks wrong for your checkout, stale packaging metadata — e.g. an old "
                "src/*.egg-info directory — can shadow the real installation when src/ is on sys.path.)"
            )
    from interpretune.protocol import Adapter

    for name in req.get("adapters") or []:
        if name not in Adapter.__members__:
            raise ComponentRequirementError(
                f"{source}: requires adapter {name!r}, which this interpretune does not provide "
                f"(known: {sorted(Adapter.__members__)}) — a newer interpretune release may be required."
            )
    for entry in req.get("pip") or []:
        r = Requirement(str(entry))
        try:
            installed = _pkg_version(r.name)
        except Exception:
            raise ComponentRequirementError(
                f"{source}: requires pip package {entry!r}, which is not installed."
            ) from None
        if r.specifier and not r.specifier.contains(installed, prereleases=True):
            raise ComponentRequirementError(f"{source}: requires {entry!r} but {r.name} {installed!r} is installed.")


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
    manifest = validate_component_manifest(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), source=f"{repo_id}@{revision}"
    )
    return manifest, _snapshot_revision(Path(path))


def pull_component_config(
    repo_id: str, key: str, revision: str | None = None, cache_dir: Path | None = None, token: str | None = None
) -> tuple[str, dict]:
    """Manifest-first fetch of ONE configuration by key; returns ``(canonical_key, config_body)``.

    The configuration fetch is pinned to the commit the manifest resolved to, and the loader parity-check (filename ==
    manifest key == derived-from-fields) runs on the fetched file before it is returned.
    """
    manifest, commit = pull_component_manifest(repo_id, revision=revision, cache_dir=cache_dir, token=token)
    enforce_component_requires(manifest, source=f"{repo_id}@{commit[:12]}")
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
    body = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return check_config_key_parity(cfg_path, body, expected_key=key), body


def register_component_config(
    repo_id: str,
    key: str,
    target_registry=None,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
    alias_bare_key: bool = True,
) -> str:
    """Fetch one configuration and register it under its hub-namespaced key ``<org>.<repo>.<key>``.

    Returns the namespaced registry key — always registered, always unambiguous. With ``alias_bare_key``
    (default), the bare canonical key is additionally registered as a collision-aware alias: an existing
    bare-key entry is KEPT (never silently overridden) with a warning pointing at the namespaced form.
    """
    import copy
    from functools import partial

    from interpretune.registry import MODULE_REGISTRY, instantiate_and_register
    from interpretune.utils.logging import rank_zero_warn

    if target_registry is None:
        target_registry = MODULE_REGISTRY
    canonical_key, body = pull_component_config(repo_id, key, revision=revision, cache_dir=cache_dir, token=token)
    namespaced = f"{repo_id.replace('/', '.')}.{canonical_key}"
    register = partial(instantiate_and_register, target_registry=target_registry)
    register(namespaced, copy.deepcopy(body))
    if alias_bare_key:
        if canonical_key in target_registry:
            rank_zero_warn(
                f"Bare key {canonical_key!r} is already registered; keeping the existing entry. "
                f"Use the namespaced key {namespaced!r} to address this component unambiguously."
            )
        else:
            register(canonical_key, copy.deepcopy(body))
    return namespaced


def resolve_component_manifest(repo_id: str, cache_dir: Path | None = None) -> tuple[dict, Path, str]:
    """CACHE-ONLY manifest read: returns ``(manifest, snapshot_dir, revision)``; never touches the network.

    Reads the repo's cached ``refs/main`` revision from the components cache — whether it got there via
    an explicit hub fetch or the local-publish bridge. Raises with the explicit fetch command when the
    component is not cached (the no-implicit-network invariant, design §3.2).
    """
    root = Path(cache_dir or IT_COMPONENTS_HUB_CACHE)
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        raise KeyError(
            f"Component {repo_id!r} is not in the local cache ({root}). Fetch it explicitly first: "
            f"interpretune.hub.pull({repo_id!r}) — local resolution never performs implicit network access."
        )
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    manifest = validate_component_manifest(
        yaml.safe_load((snapshot / IT_COMPONENT_MANIFEST).read_text(encoding="utf-8")), source=f"{repo_id}@cache"
    )
    return manifest, snapshot, revision


def resolve_component_config(repo_id: str, key: str, cache_dir: Path | None = None) -> tuple[str, dict]:
    """CACHE-ONLY resolution of one configuration: never touches the network (design invariant §3.2)."""
    manifest, snapshot, _ = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    enforce_component_requires(manifest, source=f"{repo_id}@cache")
    configs = (manifest.get("module") or {}).get("configs") or {}
    if key not in configs:
        raise KeyError(f"{repo_id} (cached) declares no configuration {key!r}. Available: {sorted(configs)}")
    cfg_path = snapshot / configs[key]
    body = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return check_config_key_parity(cfg_path, body, expected_key=key), body


def local_publish(
    component_dir: Path, repo_id: str, entrypoint_src: Path | None = None, cache_dir: Path | None = None
) -> str:
    """The LOCAL-PUBLISH BRIDGE (design v3 §11.2): materialize an in-tree component into the cache.

    Builds the publishable tree exactly as a hub publish would (same builder, same parity checks, card
    included) and installs it into the components cache in HF layout under a content-derived
    pseudo-revision, updating ``refs/main``. CI and dev then resolve seeds from the cache with no
    network — the loader's cache-only invariant holds, the CURRENT tree is what gets tested
    (co-evolution atomicity), and the publish machinery itself is exercised on every run. Idempotent:
    unchanged content maps to the same revision.
    """
    import hashlib
    import shutil
    import tempfile

    from interpretune.hub.cards import generate_component_card
    from interpretune.hub.publish import build_component_tree

    root = Path(cache_dir or IT_COMPONENTS_HUB_CACHE)
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp) / "build"
        manifest = build_component_tree(Path(component_dir), build, entrypoint_src=entrypoint_src)
        generate_component_card(manifest, repo_id).save(build / "README.md")
        digest = hashlib.sha256()
        for f in sorted(p for p in build.rglob("*") if p.is_file()):
            digest.update(f.relative_to(build).as_posix().encode())
            digest.update(f.read_bytes())
        revision = f"local{digest.hexdigest()[:35]}"  # 40 chars total: matches sha-shaped revision dirs
        repo_dir = root / f"models--{repo_id.replace('/', '--')}"
        snapshot = repo_dir / "snapshots" / revision
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(build, snapshot)
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    return revision
