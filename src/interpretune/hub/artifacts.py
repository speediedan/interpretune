"""Artifact repos for AnalysisStores: the ``it_artifact.json`` envelope + push/pull/describe.

The data half of the hub design (§5): analysis results published as DATASET repos with a
machine-written envelope, so a second user can obtain attribution-graph steering results without
the compute. Local storage stays Arrow (R9) — parquet is strictly the interchange format on the
hub side, produced by ``Dataset.push_to_hub`` and consumed by ``load_dataset``; the interpretune
formatter re-attaches on pull from the envelope's serialized ``col_cfg``.

Envelope posture follows the dashboards manifest donor: DESCRIPTIVE, not authoritative — the
dataset files are the truth, the envelope restates identity/provenance so a reader can decide
whether the artifact is the one they want. The ``identity`` block is written ONCE at first publish
and preserved verbatim on every re-push (the never-rewritten rule, mechanized); ``provenance`` is
refreshed freely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interpretune.hub.cache import IT_ARTIFACTS_HUB_CACHE

IT_ARTIFACT_ENVELOPE = "it_artifact.json"

#: The schema this code WRITES. A single integer major version, deliberately: additive optional
#: fields never bump it (readers ignore unknown keys by contract), so the only thing a version
#: change communicates is "a reader that does not understand this cannot be correct". See
#: `docs/analysis_store_serialization.md` for the policy and the rejected major.minor alternative.
ARTIFACT_SCHEMA_VERSION = 1

#: The oldest schema this code READS. Hub artifacts outlive the code that wrote them, so the
#: reader accepts a WINDOW, never a single version; raising this floor retires published
#: artifacts and is a documented, deliberate act.
ARTIFACT_SCHEMA_MIN_READABLE = 1

ANALYSIS_STORE_KIND = "analysis-store"


class ArtifactEnvelopeError(ValueError):
    """An artifact envelope is missing, malformed, or declares an unsupported schema/kind."""


def _validate_schema_version(envelope: dict, source: str) -> None:
    """Enforce the readable-window rule, distinguishing too-new from too-old from malformed.

    The three cases need three different actions from the reader, so they get three messages: too new means upgrade
    interpretune, too old means the artifact predates the supported floor and needs a re-publish, malformed means the
    envelope is not one of ours at all.
    """
    schema = envelope.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise ArtifactEnvelopeError(
            f"{source}: artifact schema {schema!r} is not an integer version — the `schema` header is "
            "mandatory and identifies the envelope as an interpretune artifact."
        )
    if schema > ARTIFACT_SCHEMA_VERSION:
        raise ArtifactEnvelopeError(
            f"{source}: artifact schema {schema} was written by a newer Interpretune than this one "
            f"(this build reads {ARTIFACT_SCHEMA_MIN_READABLE}-{ARTIFACT_SCHEMA_VERSION}). Upgrade "
            "interpretune to read it; older readers cannot safely guess what a newer schema means."
        )
    if schema < ARTIFACT_SCHEMA_MIN_READABLE:
        raise ArtifactEnvelopeError(
            f"{source}: artifact schema {schema} is older than the minimum readable schema "
            f"({ARTIFACT_SCHEMA_MIN_READABLE}). Re-publish the artifact with a current interpretune "
            "(push_analysis_store) to bring its envelope forward."
        )


def validate_artifact_envelope(envelope: Any, source: str = "<envelope>") -> dict:
    """Validate the coarse shape of a parsed artifact envelope, returning it on success.

    Unknown top-level keys are deliberately tolerated: that tolerance is what lets additive
    envelope fields ship without a schema bump, so it is part of the contract rather than laxity.
    """
    if not isinstance(envelope, dict):
        raise ArtifactEnvelopeError(f"{source}: envelope must be a mapping, got {type(envelope).__name__}")
    _validate_schema_version(envelope, source)
    if envelope.get("artifact_kind") != ANALYSIS_STORE_KIND:
        raise ArtifactEnvelopeError(
            f"{source}: unsupported artifact_kind {envelope.get('artifact_kind')!r} "
            f"(supported: {ANALYSIS_STORE_KIND!r})."
        )
    identity = envelope.get("identity")
    if not isinstance(identity, dict) or not identity.get("store_id"):
        raise ArtifactEnvelopeError(f"{source}: `identity.store_id` is mandatory (and never rewritten).")
    if not isinstance(envelope.get("interpretune"), dict):
        raise ArtifactEnvelopeError(f"{source}: the `interpretune` block (col_cfg carrier) is mandatory.")
    return envelope


def _serialize_col_cfg(store) -> dict[str, dict]:
    """Serialize the store's per-column ``ColCfg`` mapping for the envelope (JSON-native)."""
    col_cfg = (store.it_format_kwargs or {}).get("col_cfg") or {}
    out = {}
    for name, cfg in col_cfg.items():
        out[name] = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    return out


def _analysis_backend_name(store) -> str | None:
    """The PORTABLE backend reference for the envelope: a registry name, never an instance."""
    backend = (store.it_format_kwargs or {}).get("analysis_backend")
    if backend is None:
        return None
    if isinstance(backend, str):
        return backend
    from interpretune.analysis.backends import ANALYSIS_BACKEND_REGISTRY

    for name, registered in ANALYSIS_BACKEND_REGISTRY.items():
        if registered is backend:
            return name
    raise ArtifactEnvelopeError(
        f"The store's analysis_backend ({type(backend).__name__}) is not a registered named backend — "
        "register it via interpretune.analysis.backends.register_analysis_backend so artifacts can "
        "reference it portably."
    )


def content_fingerprint(store) -> str:
    """Deterministic sha256 over the store's Arrow record batches (change-detection, push-time cost).

    Recorded in PROVENANCE (freely rewritten), never in identity — consumers get change detection
    without identity churn (umbrella ruling). Deliberately independent of ``datasets``' own
    ``_fingerprint`` (whose randomness is #183, out of scope here).
    """
    import hashlib

    digest = hashlib.sha256()
    table = getattr(store.dataset.data, "table", store.dataset.data)
    for batch in table.to_batches():
        for column in batch.columns:
            for buf in column.buffers():
                if buf is not None:
                    digest.update(buf)
    return digest.hexdigest()


def build_analysis_store_envelope(
    store,
    identity: dict | None = None,
    provenance: dict | None = None,
    interchange: dict | None = None,
) -> dict:
    """Build the envelope for an AnalysisStore (descriptive; dataset files remain the truth).

    ``identity`` is normally supplied by :func:`push_analysis_store` (preserved from an existing
    envelope on re-push, minted once otherwise); ``provenance`` entries are merged over the
    defaults derived here.
    """
    import uuid
    from datetime import datetime, timezone

    from interpretune.hub.components import _installed_version

    dataset = store.dataset
    if dataset is None:
        raise ArtifactEnvelopeError("Cannot build an envelope for an AnalysisStore with no dataset loaded.")
    features = getattr(dataset, "features", None) or {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    identity = dict(identity) if identity else {"store_id": str(uuid.uuid4()), "created_utc": now}
    prov = {
        "interpretune_version": _installed_version("interpretune"),
        "content_fingerprint": content_fingerprint(store),
    }
    prov.update(provenance or {})
    artifacts: dict[str, Any] = {
        "split": store.split,
        "num_rows": getattr(dataset, "num_rows", None),
        "columns": sorted(features.keys()),
        # interchange details recorded from day one (umbrella rider): format now, row-group /
        # page-index detail joins post-MVP with the streaming guards — no schema bump needed then
        "interchange": {"format": "parquet", "local_format": "arrow", **(interchange or {})},
    }
    envelope = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": ANALYSIS_STORE_KIND,
        "generated_utc": now,
        "identity": identity,
        "provenance": prov,
        "artifacts": artifacts,
        "interpretune": {"col_cfg": _serialize_col_cfg(store), "analysis_backend": _analysis_backend_name(store)},
    }
    # writer-side guard: never emit an envelope this build could not itself read back. Catches a
    # schema bumped past the reader's window and a caller-supplied `identity` that would publish
    # unreadable — both cheap here, both expensive once they are on the hub.
    return validate_artifact_envelope(envelope, source="<writer>")


def push_analysis_store(
    store,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    provenance: dict | None = None,
    commit_message: str | None = None,
) -> str:
    """Publish an AnalysisStore as a dataset repo: parquet interchange + envelope + generated card.

    Returns the resulting repo revision (commit sha). On re-push to an existing artifact repo, the
    envelope's ``identity`` block is read back and preserved verbatim — identity survives
    mutation; only ``provenance``/``artifacts`` refresh.
    """
    import io

    from huggingface_hub import HfApi

    from interpretune.hub.cards import generate_artifact_card

    if store.dataset is None:
        raise ArtifactEnvelopeError("Cannot push an AnalysisStore with no dataset loaded.")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    identity = None
    try:
        existing = api.hf_hub_download(repo_id, IT_ARTIFACT_ENVELOPE, repo_type="dataset")
        identity = validate_artifact_envelope(
            json.loads(Path(existing).read_text(encoding="utf-8")), source=f"{repo_id} (existing)"
        )["identity"]
    except ArtifactEnvelopeError:
        raise  # a malformed EXISTING envelope must fail loudly, not be silently re-minted
    except Exception:
        identity = None  # first publish (no envelope yet)

    # parquet interchange: push_to_hub converts per-shard; the LOCAL store remains Arrow (R9)
    store.dataset.push_to_hub(repo_id, split=store.split, private=private, token=token, commit_message=commit_message)
    parquet_files = sorted(f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet"))
    envelope = build_analysis_store_envelope(
        store, identity=identity, provenance=provenance, interchange={"files": parquet_files}
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(envelope, indent=1, sort_keys=True).encode("utf-8")),
        path_in_repo=IT_ARTIFACT_ENVELOPE,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or "Update it_artifact.json envelope",
    )
    card = generate_artifact_card(envelope, repo_id)
    api.upload_file(
        path_or_fileobj=io.BytesIO(str(card).encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or "Update generated artifact card",
    )
    sha = api.repo_info(repo_id, repo_type="dataset").sha
    assert sha is not None  # a repo we just committed to always has a head revision
    return sha


def pull_analysis_store(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
):
    """Fetch an analysis-store artifact and hydrate it: envelope-first, formatter re-attached.

    Revision-pinned like component resolution: the envelope resolves first, and the dataset files
    are fetched from the SAME commit. Returns a ready ``AnalysisStore`` whose interpretune
    formatter is re-attached from the envelope's serialized ``col_cfg`` — steering-capable graph
    hydration then flows through the normal ``AnalysisBackend`` seam with no pipeline re-run.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download

    from interpretune.analysis import AnalysisStore
    from interpretune.hub.components import _TELEMETRY

    root = str(cache_dir or IT_ARTIFACTS_HUB_CACHE)
    env_path = hf_hub_download(
        repo_id, IT_ARTIFACT_ENVELOPE, repo_type="dataset", revision=revision, cache_dir=root, token=token, **_TELEMETRY
    )
    envelope = validate_artifact_envelope(
        json.loads(Path(env_path).read_text(encoding="utf-8")), source=f"{repo_id}@{revision or 'main'}"
    )
    commit = Path(env_path).parts[Path(env_path).parts.index("snapshots") + 1]
    snapshot = snapshot_download(
        repo_id, repo_type="dataset", revision=commit, cache_dir=root, token=token, **_TELEMETRY
    )
    split = envelope["artifacts"].get("split") or "validation"
    dataset = load_dataset(str(snapshot), split=split)
    # the formatter contract is SERIALIZED col_cfg (OpSchemaExt runs ColCfg.from_dict itself), so
    # the envelope block passes straight through as format kwargs; a named analysis backend
    # re-attaches the hydration seam (graph consumers get Graph objects, not primitive columns)
    it_block = envelope["interpretune"]
    format_kwargs: dict[str, Any] = {}
    if it_block.get("col_cfg"):
        format_kwargs["col_cfg"] = it_block["col_cfg"]
    if it_block.get("analysis_backend"):
        from interpretune.analysis.backends import resolve_analysis_backend

        format_kwargs["analysis_backend"] = resolve_analysis_backend(it_block["analysis_backend"])
    store = AnalysisStore(dataset=dataset, split=split, it_format_kwargs=format_kwargs or None)
    return store


def describe_analysis_store(repo_id: str, *, cache_dir: Path | None = None) -> dict:
    """CACHE-ONLY description of an artifact repo: the envelope, without touching the network."""
    root = Path(cache_dir or IT_ARTIFACTS_HUB_CACHE)
    repo_dir = root / f"datasets--{repo_id.replace('/', '--')}"
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        raise KeyError(
            f"Artifact {repo_id!r} is not in the local cache ({root}). Fetch it explicitly first: "
            f"interpretune.hub.pull_analysis_store({repo_id!r}) — local resolution never performs "
            "implicit network access."
        )
    snapshot = repo_dir / "snapshots" / ref.read_text(encoding="utf-8").strip()
    return validate_artifact_envelope(
        json.loads((snapshot / IT_ARTIFACT_ENVELOPE).read_text(encoding="utf-8")), source=f"{repo_id}@cache"
    )
