"""Manifest-first fetch of interpretune component repos.

The standard resolution path fetches ``it_component.yaml`` (a few KB), reads its ``configs:`` index, then
fetches only the requested configuration file — pinning every follow-up fetch to the commit the manifest came
from, so a repo updated mid-resolution cannot hand back files from two different revisions. This also makes
``countDownloads: path:"it_component.yaml"`` correct by construction: every resolution starts at the manifest.

No function here is called implicitly by registry lookups — local resolution never touches the network; Hub
components are fetched only by these explicit calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from interpretune.hub.cache import IT_COMPONENTS_HUB_CACHE
from interpretune.hub.manifest import (
    IT_COMPONENT_MANIFEST,
    ComponentManifestError,
    check_config_key_parity,
    validate_component_manifest,
)


# The predicate lives in `interpretune.utils.requirements`, NOT here: it is shared by the hub and bundled
# paths, and a predicate shared by both cannot sit inside one of them. The bundled registration pass
# imports only the standard library, so reaching into the hub subsystem to answer "is this available here"
# would invert the layering. (Layering, not import cost -- huggingface_hub and yaml are already resident
# after `import interpretune`, via transformers and jsonargparse.)
from interpretune.utils.requirements import (
    UnmetRequirement as UnmetRequirement,
    installed_version as _installed_version,
    requirement_status as requirement_status,
)


# version telemetry for hf_hub_download (design §8); interpretune may be running from a raw checkout
_TELEMETRY = {"library_name": "interpretune", "library_version": _installed_version("interpretune")}


class ComponentRequirementError(ImportError):
    """The current environment cannot satisfy a component manifest's ``requires:`` block."""


def enforce_component_requires(manifest: dict, source: str = "<component>") -> None:
    """Enforce a manifest's ``requires:`` block against the current environment (fail informatively).

    Three failure modes, each with its own message: an unsatisfied ``interpretune`` version specifier,
    an adapter name this interpretune does not know (usually means a newer interpretune is required —
    NOT that every listed adapter's backend must be importable, since configs compose subsets), and a
    ``pip`` requirement that is not installed / does not satisfy its specifier.

    Disposition only — :func:`requirement_status` evaluates, and raising ``unmet[0]`` preserves this
    function's pre-split behaviour exactly.
    """
    if unmet := requirement_status(manifest.get("requires") or {}, source):
        raise ComponentRequirementError(unmet[0].message)


def _snapshot_revision(downloaded_path: Path) -> str:
    """Extract the resolved commit hash from an hf-cache download path (``.../snapshots/<commit>/...``)."""
    parts = Path(downloaded_path).parts
    return parts[parts.index("snapshots") + 1]


#: Revisions written by :func:`local_publish` carry this prefix; Hub commits are bare hex.
LOCAL_REVISION_PREFIX = "local"


def is_local_revision(revision: str | None) -> bool:
    """Whether a cached revision came from the local-publish bridge rather than the Hub."""
    return bool(revision) and str(revision).startswith(LOCAL_REVISION_PREFIX)


def describe_revision(revision: str | None) -> str:
    """A cached revision as a reader should see it: ``local publish 2931c895...`` or ``f13f4770...``.

    A local-publish pseudo-revision is 40 characters and sha-shaped, so printed bare it reads as a commit; the
    prefix does its job only for a reader who knows to look for it. Every place a revision is rendered for a
    person goes through this, so the origin travels with the identifier.
    """
    if revision is None:
        return "none"
    short = revision[:12]
    return f"local publish {short[len(LOCAL_REVISION_PREFIX) :] or short}" if is_local_revision(revision) else short


class HubUnreachableError(LookupError):
    """The Hub answered 404 for a repo: absent, OR not visible to the token in use.

    HF returns 404 rather than 403 for a private repo a token cannot see, and ``whoami`` naming the repo's owner
    is not evidence either way (a token scoped to a subset of the owner's repos still answers ``whoami`` with the
    owner). So "404 while authenticated as the owner" reads as conclusive absence and is not. The message says
    both readings, because acting on the wrong one once republished a live component as though into a fresh repo.
    """


def _explain_404(repo_id: str, exc: BaseException | None = None) -> HubUnreachableError:
    return HubUnreachableError(
        f"The Hub returned 404 for {repo_id!r}: the repo is absent, OR it is private and not visible to the token "
        "in use. HF answers 404 (not 403) for a private repo a token cannot see, and `whoami` naming the owner does "
        "not distinguish the two. Check the token's repo scope before treating this as absence; a cached snapshot "
        "of this repo, if any, is not evidence of Hub presence either (interpretune.hub.hub_presence)."
    )


def pull_component_manifest(
    repo_id: str, revision: str | None = None, cache_dir: Path | None = None, token: str | None = None
) -> tuple[dict, str]:
    """Fetch and validate a component repo's manifest; returns ``(manifest, resolved_commit)``.

    A 404 is re-raised as :class:`HubUnreachableError`, whose message carries both readings (absent, or not
    visible to this token), since the bare error reads as conclusive absence and is not.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id,
            IT_COMPONENT_MANIFEST,
            revision=revision,
            cache_dir=str(cache_dir or IT_COMPONENTS_HUB_CACHE),
            token=token,
            **_TELEMETRY,
        )
    except RepositoryNotFoundError as exc:
        raise _explain_404(repo_id, exc) from exc
    manifest = validate_component_manifest(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), source=f"{repo_id}@{revision}"
    )
    commit = _snapshot_revision(Path(path))
    if revision and revision != "main":
        # A pinned fetch writes a complete snapshot that nothing could address: hf_hub_download writes
        # `refs/main` only for an unpinned fetch, and resolution read nothing else, so following the trust
        # posture's own advice (pin a revision) on a clean machine produced a component the loader reported as
        # never cached while the snapshot sat beside the message. The pin is recorded where resolution looks.
        record_component_pin(repo_id, commit, requested_revision=revision, cache_dir=cache_dir)
    return manifest, commit


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


def declared_component_payloads(manifest: dict) -> list[str]:
    """The repo-relative files a cached component needs beyond its manifest for its NON-configuration kinds.

    ``pull_component_config`` fetches one configuration by key and op collections have their own cache and
    verb, so those are not listed. What IS listed is every payload a cache-only loader reads whole: the
    adapters entrypoint (``load_hub_adapter``), the promptconfigs entrypoint (``import_cached_entrypoint``)
    and the hookmaps documents (``load_hub_hookmaps``). A manifest-only pull of such a component used to
    leave a snapshot those loaders could not complete, with an error blaming a partial download.
    """
    rels: list[str] = []
    for section in ("adapters", "promptconfigs"):
        entrypoint = (manifest.get(section) or {}).get("entrypoint")
        if entrypoint:
            rels.append(entrypoint)
    rels.extend((manifest.get("hookmaps") or {}).get("files") or [])
    return rels


def pull_component_payloads(
    repo_id: str, manifest: dict, commit: str, cache_dir: Path | None = None, token: str | None = None
) -> list[Path]:
    """Materialize a validated manifest's declared payloads into the same snapshot, pinned to ``commit``."""
    paths = []
    for rel in declared_component_payloads(manifest):
        paths.append(
            Path(
                hf_hub_download(
                    repo_id,
                    rel,
                    revision=commit,  # pinned: the snapshot stays single-revision coherent
                    cache_dir=str(cache_dir or IT_COMPONENTS_HUB_CACHE),
                    token=token,
                    **_TELEMETRY,
                )
            )
        )
    return paths


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


class LocalSnapshotWarning(UserWarning):
    """A local-publish snapshot resolved for a repo whose cache also holds a Hub revision, so the Hub is
    shadowed."""


class LocalSnapshotShadowsHubError(LookupError):
    """``require_hub=True`` and the cached ``refs/main`` is a local-publish snapshot."""


def _hub_snapshots(repo_dir: Path) -> list[str]:
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(p.name for p in snapshots.iterdir() if p.is_dir() and not is_local_revision(p.name))


#: The ref a revision-pinned fetch records, inside the repo's HF cache layout beside `refs/main`. Resolution
#: prefers it, so a pinned environment keeps executing what it pinned even after an unpinned fetch moves `main`.
COMPONENT_PIN_REF = "it-pinned"


def _repo_dir(repo_id: str, cache_dir: Path | None) -> Path:
    return Path(cache_dir or IT_COMPONENTS_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def record_component_pin(repo_id: str, commit: str, *, requested_revision: str, cache_dir: Path | None = None) -> Path:
    """Write the pin marker for one component repo; re-pulling at another revision moves it."""
    if not commit:
        raise ValueError(f"refusing to record an empty commit for {repo_id!r}")
    refs = _repo_dir(repo_id, cache_dir) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    path = refs / COMPONENT_PIN_REF
    path.write_text(commit, encoding="utf-8")
    return path


def read_component_pin(repo_id: str, cache_dir: Path | None = None) -> str | None:
    """The pinned commit for a component repo, or ``None`` when unpinned."""
    path = _repo_dir(repo_id, cache_dir) / "refs" / COMPONENT_PIN_REF
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def clear_component_pin(repo_id: str, cache_dir: Path | None = None) -> bool:
    """Release a component pin; returns whether one existed.

    Resolution then reads ``refs/main`` again.
    """
    path = _repo_dir(repo_id, cache_dir) / "refs" / COMPONENT_PIN_REF
    if not path.is_file():
        return False
    path.unlink()
    return True


def cached_component_revisions(repo_id: str, cache_dir: Path | None = None) -> list[str]:
    """Every snapshot of a repo in the cache that carries a manifest, addressed or not."""
    snapshots = _repo_dir(repo_id, cache_dir) / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(p.name for p in snapshots.iterdir() if (p / IT_COMPONENT_MANIFEST).is_file())


def resolve_component_manifest(
    repo_id: str, cache_dir: Path | None = None, *, revision: str | None = None, require_hub: bool = False
) -> tuple[dict, Path, str]:
    """CACHE-ONLY manifest read: returns ``(manifest, snapshot_dir, revision)``; never touches the network.

    Reads the repo's cached ``refs/main`` revision from the components cache — whether it got there via
    an explicit hub fetch or the local-publish bridge. Raises with the explicit fetch command when the
    component is not cached (the no-implicit-network invariant, design §3.2).

    A local-publish snapshot and a Hub revision are indistinguishable to a caller at the moment it matters, and
    the local one wins whenever it was written last: a verification of a publish once loaded days-old code that
    way and reported success. So when ``refs/main`` is a local snapshot AND the cache also holds a Hub revision
    of the same repo, the resolution says so (a :class:`LocalSnapshotWarning` naming both), and with
    ``require_hub=True`` it refuses instead, for callers that are verifying what the Hub serves. A repo that
    exists only locally (the in-tree seeds) resolves silently, as before.
    """
    root = Path(cache_dir or IT_COMPONENTS_HUB_CACHE)
    repo_dir = _repo_dir(repo_id, cache_dir)
    cached = cached_component_revisions(repo_id, cache_dir)
    if revision is not None:
        # An explicit revision: the caller knows what it wants; it must be in the cache, and nothing is fetched.
        # A short sha is the natural thing to pin with and `pull` accepts one (the Hub resolves it), so `load`
        # accepts any unambiguous prefix of a cached revision too, as git does; the snapshot directory carries
        # the full sha, and an exact-match comparison rejected the very string a pull had just succeeded with.
        matches = [r for r in cached if r == revision or r.startswith(revision)]
        if len(matches) > 1:
            raise KeyError(
                f"Component {repo_id!r}: revision prefix {revision!r} is ambiguous among cached snapshots "
                f"{matches}; give more characters."
            )
        if not matches:
            raise KeyError(
                f"Component {repo_id!r} has no cached revision matching {revision!r} ({root}); cached: "
                f"{cached or 'none'}. Fetch it explicitly: interpretune.hub.pull({repo_id!r}, "
                f"revision={revision!r}) — local resolution never performs implicit network access."
            )
        revision = matches[0]
    else:
        pinned = read_component_pin(repo_id, cache_dir)
        ref = repo_dir / "refs" / "main"
        if pinned and pinned in cached:
            revision = pinned  # the pin beats `main`: a republish cannot change what a pinned environment loads
        elif ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
        elif cached:
            # Cached but unaddressable: a snapshot written by something that recorded no ref (a pinned fetch
            # before pins were recorded, or a raw hf_hub_download). Saying "not in the cache" here sent a
            # reader hunting for a download that had succeeded; the message names what is on disk and the
            # two ways to address it.
            raise KeyError(
                f"Component {repo_id!r} is cached ({root}) but no revision is addressed: cached snapshot(s) "
                f"{[r[:12] for r in cached]}, no `refs/main` and no pin. Resolve one explicitly with "
                f"revision={cached[0]!r}, or fetch unpinned once (interpretune.hub.pull({repo_id!r})) to set "
                "`refs/main`; a pinned pull now records its pin, so this state comes from older caches."
            )
        else:
            raise KeyError(
                f"Component {repo_id!r} is not in the local cache ({root}). Fetch it explicitly first: "
                f"interpretune.hub.pull({repo_id!r}) — local resolution never performs implicit network access."
            )
    if is_local_revision(revision):
        hub_revisions = _hub_snapshots(repo_dir)
        if require_hub:
            raise LocalSnapshotShadowsHubError(
                f"{repo_id!r} resolves to the local-publish snapshot {revision[:12]} (cache-only; the Hub was not "
                f"consulted), and require_hub=True. Cached Hub revisions: {[r[:12] for r in hub_revisions] or 'none'}. "
                f"Fetch the Hub revision explicitly: interpretune.hub.pull({repo_id!r})."
            )
        if hub_revisions:
            from interpretune.utils.logging import rank_zero_warn

            rank_zero_warn(
                f"{repo_id!r} resolved to the local-publish snapshot {revision[:12]}, which shadows the cached Hub "
                f"revision(s) {[r[:12] for r in hub_revisions]}; the Hub was not consulted (resolution is cache-only). "
                f"If you meant the published artifact, run interpretune.hub.pull({repo_id!r}) or resolve with "
                "require_hub=True.",
                category=LocalSnapshotWarning,
            )
    snapshot = repo_dir / "snapshots" / revision
    manifest = validate_component_manifest(
        yaml.safe_load((snapshot / IT_COMPONENT_MANIFEST).read_text(encoding="utf-8")), source=f"{repo_id}@cache"
    )
    return manifest, snapshot, revision


def resolve_component_config(
    repo_id: str, key: str, cache_dir: Path | None = None, *, revision: str | None = None, require_hub: bool = False
) -> tuple[str, dict]:
    """CACHE-ONLY resolution of one configuration: never touches the network (design invariant §3.2)."""
    manifest, snapshot, _ = resolve_component_manifest(
        repo_id, cache_dir=cache_dir, revision=revision, require_hub=require_hub
    )
    enforce_component_requires(manifest, source=f"{repo_id}@cache")
    configs = (manifest.get("module") or {}).get("configs") or {}
    if key not in configs:
        raise KeyError(f"{repo_id} (cached) declares no configuration {key!r}. Available: {sorted(configs)}")
    cfg_path = snapshot / configs[key]
    body = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return check_config_key_parity(cfg_path, body, expected_key=key), body


def resolve_datamodule_config(repo_id: str, name: str, cache_dir: Path | None = None) -> dict:
    """CACHE-ONLY resolution of one datamodule entry's standalone payload (#128).

    Datamodule entries are named by the manifest's ``datamodules`` index rather than by derived
    configuration keys, so there is no key-parity check here -- the name IS the address. The returned
    body is the standalone-consumption payload only: module configurations inline their own
    ``datamodule_cfg`` wholesale and never read this payload (strictly two-path, no merge semantics).
    """
    manifest, snapshot, _ = resolve_component_manifest(repo_id, cache_dir=cache_dir)
    enforce_component_requires(manifest, source=f"{repo_id}@cache")
    entries = manifest.get("datamodules") or {}
    if name not in entries:
        raise KeyError(f"{repo_id} (cached) declares no datamodule {name!r}. Available: {sorted(entries)}")
    body = yaml.safe_load((snapshot / entries[name]["config"]).read_text(encoding="utf-8"))
    if not isinstance(body, dict) or "datamodule_cfg" not in body:
        raise ComponentManifestError(
            f"{repo_id}#{name}: standalone datamodule payload must carry a `datamodule_cfg` mapping."
        )
    if "module_cfg" in body or "registered_cfg" in body:
        raise ComponentManifestError(
            f"{repo_id}#{name}: a standalone datamodule payload must not carry module configuration "
            "(`module_cfg`/`registered_cfg`) -- it is the datamodule-only half of the two-path contract."
        )
    return body


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


@dataclass(frozen=True)
class HubPresence:
    """The answer to "is this repo (and revision) on the Hub, as seen by this token?", asked EXPLICITLY.

    Resolution never asks it (the cache-only invariant is right and stays), so a cached snapshot of a repo that
    was deleted, renamed, or is merely invisible to the current token keeps loading; this is the verb for a
    caller who wants to know. ``reachable`` is False on a 404, which means absent OR not visible to the token,
    and ``detail`` says so.
    """

    repo_id: str
    reachable: bool
    revision: str | None
    revision_present: bool | None
    detail: str


def hub_presence(repo_id: str, revision: str | None = None, token: str | None = None) -> HubPresence:
    """Ask the Hub whether ``repo_id`` (and ``revision``, when given) is reachable with this token.

    Network.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    if is_local_revision(revision):
        return HubPresence(
            repo_id,
            False,
            revision,
            False,
            f"{describe_revision(revision)} is a local-publish snapshot; it was never on the Hub",
        )
    api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id, revision=revision)
    except RevisionNotFoundError:
        return HubPresence(
            repo_id, True, revision, False, f"repo reachable; revision {describe_revision(revision)} is not on the Hub"
        )
    except RepositoryNotFoundError:
        return HubPresence(repo_id, False, revision, None, str(_explain_404(repo_id, None)))
    return HubPresence(
        repo_id, True, revision, True if revision else None, f"reachable; Hub head {str(getattr(info, 'sha', ''))[:12]}"
    )
