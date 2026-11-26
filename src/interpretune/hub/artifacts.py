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

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from interpretune.hub.cache import IT_ARTIFACTS_HUB_CACHE, IT_ARTIFACTS_HUB_CACHE_SETTING, ensure_writable_cache_root

IT_ARTIFACT_ENVELOPE = "it_artifact.json"

#: The schema this code WRITES. A single integer major version, deliberately: additive optional
#: fields never bump it (readers ignore unknown keys by contract), so the only thing a version
#: change communicates is "a reader that does not understand this cannot be correct". See
#: `docs/analysis_store_serialization.md` for the policy and the rejected major.minor alternative.
ARTIFACT_SCHEMA_VERSION = 3

#: The oldest schema this code READS. Hub artifacts outlive the code that wrote them, so the
#: reader accepts a WINDOW, never a single version; raising this floor retires published
#: artifacts and is a documented, deliberate act.
ARTIFACT_SCHEMA_MIN_READABLE = 1

ANALYSIS_STORE_KIND = "analysis-store"

#: The uncompressed byte target for a published store's Parquet row groups (``datasets`` sizes row groups from
#: ``datasets.config.MAX_ROW_GROUP_SIZE``, 100MB by default, and ``push_to_hub`` takes no row-group argument). Measured
#: on a 142 MB store of float tensor columns, streamed with projection: smaller groups make a row cheaper to reach (the
#: first row of every column cost 9.5 MB at 4MB, 34 MB at 16MB and the whole 143 MB file at 100MB), while each group
#: costs at least one range request per projected column and a fallback dictionary page per float column (a scan of two
#: scalar columns took 38 requests at 4MB, 11 at 16MB and 4 at 100MB; the file grew 17% at 4MB and 6% at 16MB). 16MB
#: balances the two. A store whose rows are each larger than the target gets one row per group whatever it is.
ANALYSIS_STORE_ROW_GROUP_SIZE = "16MB"

#: ``HfFileSystem``'s read-ahead block for a streamed pull. Its 5 MiB default is fetched on every range request, so a
#: projected read of two small columns cost 5.4 MB of transfer where 64 KiB blocks cost 197 KB.
STREAMING_BLOCK_SIZE = 64 * 1024

#: The estimated first-row cost above which a streamed pull warns. A scan reads one row group ahead, so reaching the
#: first row fetches about two row groups of the projected columns, and a store pushed before the row-group target
#: (a file under 100 MB is then one group) costs its whole file. 64 MB is about two groups at the 16 MB target.
STREAMED_FIRST_ROW_WARN_BYTES = 64 * 1024 * 1024

#: The first schema whose envelopes must carry a ``subject`` block. Version 2 made it mandatory, since a store says
#: nothing usable about what it was computed on without it; a version-1 envelope is still read, with no subject.
SUBJECT_REQUIRED_FROM_SCHEMA = 2


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
    if envelope["schema"] >= SUBJECT_REQUIRED_FROM_SCHEMA:
        from interpretune.hub.subject import StoreSubject, StoreSubjectError

        try:
            StoreSubject.from_dict(envelope.get("subject"), source=source)
        except StoreSubjectError as e:
            raise ArtifactEnvelopeError(str(e)) from None
    _validate_card_block(envelope.get("card"), source)
    return envelope


def _validate_card_block(card: Any, source: str) -> None:
    """The optional ``card`` block holds the publisher's card text; refuse one whose shape this build cannot
    render."""
    if card is None:
        return
    if not isinstance(card, dict) or not isinstance(card.get("summary"), str):
        raise ArtifactEnvelopeError(f"{source}: the `card` block must be a mapping whose `summary` is a string.")


def _serialize_col_cfg(store) -> dict[str, dict]:
    """Serialize the store's per-column ``ColCfg`` mapping for the envelope (JSON-native)."""
    col_cfg = (store.it_format_kwargs or {}).get("col_cfg") or {}
    out = {}
    for name, cfg in col_cfg.items():
        out[name] = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    return out


def _resolve_protocol_path(path: str, source: str = "<envelope>"):
    """Resolve a protocol recorded in an envelope, refusing any module a reader has not already trusted.

    A store is data, but this one field names code: importing it runs that module's top level. So only two kinds of
    module are resolved. ``interpretune``'s own, and a hub component module already in ``sys.modules``, which is
    there only because a trusted load executed it (see ``interpretune.hub.trust``), so resolving it imports nothing
    new. Anything else, or a path that does not name a ``BaseAnalysisBatchProtocol`` subclass, is refused by name:
    serving the default protocol in its place would hand back a store whose attribute surface silently differs from
    the one it was written with.
    """
    import sys

    from interpretune.hub.entrypoints import _HUB_MODULE_PREFIX
    from interpretune.protocol import BaseAnalysisBatchProtocol

    module_path, _, attr = path.rpartition(".")
    if module_path.split(".")[0] == "interpretune":
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ArtifactEnvelopeError(f"{source}: protocol_cls {path!r} does not import: {e}") from e
    elif module_path.startswith(_HUB_MODULE_PREFIX) and module_path in sys.modules:
        module = sys.modules[module_path]
    else:
        raise ArtifactEnvelopeError(
            f"{source}: protocol_cls {path!r} names a module outside interpretune and the hub components this process"
            " has loaded through the remote-code gate, so it is not imported. Load the component that defines it"
            " first (it.hub.load, under IT_TRUST_REMOTE_CODE), then pull the store again; or pull it with"
            " protocol='schema' to read its columns without importing that module."
        )
    protocol_cls = getattr(module, attr, None)
    # MRO rather than issubclass: the base is a non-runtime_checkable Protocol, so issubclass raises.
    if not isinstance(protocol_cls, type) or BaseAnalysisBatchProtocol not in protocol_cls.__mro__:
        raise ArtifactEnvelopeError(f"{source}: protocol_cls {path!r} is not a BaseAnalysisBatchProtocol subclass.")
    return protocol_cls


def _protocol_cls_entry(store) -> dict[str, Any]:
    """The store's batch protocol as a PORTABLE import path, omitted when it is the default.

    Recorded at write time rather than reconstructed on pull, for the same reason op provenance is:
    what a name resolves to later is not what produced the store. Mirrors
    `_analysis_backend_name` -- a portable reference rather than an instance, resolved on the way back.

    Omitted for the default, so an envelope only carries the key when it says something. A reader seeing
    no key gets the default, which is what they would have got anyway.
    """
    from interpretune.protocol import DefaultAnalysisBatchProtocol, batch_protocol_fields

    protocol_cls = getattr(store, "_protocol_cls", None)
    if protocol_cls is None or protocol_cls is DefaultAnalysisBatchProtocol:
        return {}
    if (recorded := getattr(protocol_cls, "recorded_protocol", None)) is not None:
        # a schema-only read re-published: carry the original path and fields forward rather than the stand-in's
        return {"protocol_cls": recorded, "protocol_fields": sorted(batch_protocol_fields(protocol_cls))}
    return {
        "protocol_cls": f"{protocol_cls.__module__}.{protocol_cls.__qualname__}",
        # the attribute surface as data, so a reader who does not import the protocol still gets the same names
        "protocol_fields": sorted(batch_protocol_fields(protocol_cls)),
    }


def _schema_only_protocol(declared: str, fields: list[str], fields_from: str) -> type:
    """A stand-in for an envelope's protocol, built from field names alone, so nothing the envelope names is
    imported.

    A store's protocol decides only which names get attribute access; rows decode through ``col_cfg``. The stand-in
    declares ``fields`` and carries ``recorded_protocol`` (the path the envelope names) and ``fields_from``
    (``"envelope"`` when the writer recorded the names, ``"columns"`` when an older envelope did not and the store's
    own columns stand in, so a declared field the store does not hold raises rather than reading as None).
    """
    import types

    from interpretune.protocol import BaseAnalysisBatchProtocol

    def _body(ns: dict) -> None:
        ns["__annotations__"] = dict.fromkeys(fields, "Any")
        ns["__module__"] = __name__
        ns["recorded_protocol"] = declared
        ns["fields_from"] = fields_from

    return types.new_class(f"SchemaOnly[{declared}]", (BaseAnalysisBatchProtocol,), {}, _body)


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


def _store_subject(store, subject=None):
    """The subject to publish: the caller's, else the one stamped at write time, else a refusal naming the fix."""
    subject = subject if subject is not None else getattr(store, "subject", None)
    if subject is None:
        raise ArtifactEnvelopeError(
            "The store records no subject (the model, tokenizer, component map and dictionaries its columns were"
            " computed on): it was not written by an op on a module, or the module stated no model id. Pass"
            " `subject=StoreSubject(model_id=...)` (interpretune.hub.subject) to say what it was computed on."
        )
    return subject


def build_analysis_store_envelope(
    store,
    identity: dict | None = None,
    provenance: dict | None = None,
    interchange: dict | None = None,
    subject=None,
    card_summary: str | None = None,
) -> dict:
    """Build the envelope for an AnalysisStore (descriptive; dataset files remain the truth).

    ``identity`` is normally supplied by :func:`push_analysis_store` (preserved from an existing
    envelope on re-push, minted once otherwise); ``provenance`` entries are merged over the
    defaults derived here. ``subject`` (a :class:`~interpretune.hub.subject.StoreSubject`) defaults to the one
    the store was stamped with when an op wrote it; a store with neither is refused by name. ``card_summary`` is
    the publisher's card text (Markdown), recorded as the ``card`` block the generated card renders under its title.
    """
    import uuid
    from datetime import datetime, timezone

    from interpretune.hub.components import _installed_version

    _refuse_streamed(store, "build an envelope for")
    dataset = store.dataset
    if dataset is None:
        raise ArtifactEnvelopeError("Cannot build an envelope for an AnalysisStore with no dataset loaded.")
    subject = _store_subject(store, subject)
    features = getattr(dataset, "features", None) or {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    identity = dict(identity) if identity else {"store_id": str(uuid.uuid4()), "created_utc": now}
    prov = {
        "interpretune_version": _installed_version("interpretune"),
        "content_fingerprint": content_fingerprint(store),
    }
    # Op-collection provenance is READ from the store's write-time stamp, not reconstructed here.
    # `interpretune_version` alone identifies a BUNDLED op's contract, because the package that shipped the
    # op is the package that pinned it; a hub op's contract lives in its own collection at its own revision,
    # so without this the envelope cannot say what produced its columns.
    op_collections = [record.to_dict() for record in getattr(store, "op_provenance", ()) or ()]
    if op_collections:
        prov["op_collections"] = op_collections
    # caller-supplied provenance still wins: an explicit record beats a derived one, and a store loaded from
    # disk carries no stamp of its own to lose
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
        "subject": subject.to_dict(),
        "provenance": prov,
        "artifacts": artifacts,
        "interpretune": {
            "col_cfg": _serialize_col_cfg(store),
            "analysis_backend": _analysis_backend_name(store),
            **_protocol_cls_entry(store),
        },
    }
    if card_summary:
        envelope["card"] = {"summary": card_summary}
    # writer-side guard: never emit an envelope this build could not itself read back. Catches a
    # schema bumped past the reader's window and a caller-supplied `identity` that would publish
    # unreadable — both cheap here, both expensive once they are on the hub.
    return validate_artifact_envelope(envelope, source="<writer>")


def _row_group_bytes(row_group_size: int | str) -> int:
    """``row_group_size`` in bytes, refusing a size ``datasets`` would not parse or that is not positive."""
    from datasets.utils.py_utils import convert_file_size_to_int

    try:
        size = convert_file_size_to_int(row_group_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"row_group_size={row_group_size!r} is not a size: pass bytes as an int, or a string such as '16MB'."
        ) from exc
    if size <= 0:
        raise ValueError(f"row_group_size={row_group_size!r} must be a positive size.")
    return size


@contextmanager
def _row_group_target(row_group_size: int | str):
    """Size the row groups ``Dataset.push_to_hub`` writes inside this block.

    ``push_to_hub`` takes no row-group argument and reads ``datasets.config.MAX_ROW_GROUP_SIZE`` when it computes its
    writer batch, so the target is set there for the push and restored after it, also when the push raises.
    """
    import datasets.config

    previous = datasets.config.MAX_ROW_GROUP_SIZE
    datasets.config.MAX_ROW_GROUP_SIZE = row_group_size
    try:
        yield
    finally:
        datasets.config.MAX_ROW_GROUP_SIZE = previous


def _refuse_streamed(store, action: str) -> None:
    """A streamed store has no table to publish; say so rather than report it as empty."""
    if getattr(store, "stream", None) is not None:
        raise ArtifactEnvelopeError(
            f"Cannot {action} a streamed AnalysisStore: a stream has no table to write. Pull the store without"
            " `streaming=True` (or build a Dataset from its rows) and publish that."
        )


def push_analysis_store(
    store,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    provenance: dict | None = None,
    commit_message: str | None = None,
    skip_checks: frozenset[str] | set[str] | tuple[str, ...] = (),
    add_checks: frozenset[str] | set[str] | tuple[str, ...] = (),
    subject=None,
    row_group_size: int | str = ANALYSIS_STORE_ROW_GROUP_SIZE,
    card_summary: str | None = None,
) -> str:
    """Publish an AnalysisStore as a dataset repo: parquet interchange + envelope + generated card.

    Returns the resulting repo revision (commit sha). On re-push to an existing artifact repo, the
    envelope's ``identity`` block is read back and preserved verbatim — identity survives
    mutation; only ``provenance``/``artifacts`` refresh.

    The card is generated from the envelope: the store's subject, its provenance (``provenance`` fields included) and
    ``card_summary``, the publisher's own Markdown, rendered under the title (say what the rows are and where their
    inputs came from). The summary is recorded in the envelope, so a re-push that passes none keeps the published one;
    pass ``card_summary=""`` to drop it. Text edited into the card on the Hub is replaced by the next push.

    The Parquet is written by ``Dataset.push_to_hub``, which page-indexes every file from ``datasets`` 4.1.0 (the
    floor), so no pre-upload check reads it. After the upload the ``range-read`` check reads one published file back
    through ``HfFileSystem`` and raises ``PublishedArtifactCheckError`` if it does not decode: the artifact is then
    already live. ``skip_checks`` / ``add_checks`` name checks to skip or turn on (unknown names are refused before
    any network call), and skips are recorded in the envelope's ``provenance``. ``subject`` overrides the one the
    store was stamped with; a store with neither is refused before any network call.

    ``row_group_size`` is the uncompressed byte target for the Parquet row groups (bytes, or a string such as
    ``"16MB"``), recorded in the envelope as ``artifacts.interchange.row_group_target_bytes``. The default,
    :data:`ANALYSIS_STORE_ROW_GROUP_SIZE`, is measured for streamed, projected reads; a streamed store is refused.
    """
    import io

    from huggingface_hub import HfApi

    from interpretune.hub.cards import generate_artifact_card
    from interpretune.utils.publish_checks import (
        ANALYSIS_STORE,
        POST_UPLOAD,
        PRE_UPLOAD,
        PublishTarget,
        run_publish_checks,
        validate_check_names,
    )

    validate_check_names(ANALYSIS_STORE, skip_checks, add_checks)
    row_group_bytes = _row_group_bytes(row_group_size)
    _refuse_streamed(store, "push")
    if store.dataset is None:
        raise ArtifactEnvelopeError("Cannot push an AnalysisStore with no dataset loaded.")
    subject = _store_subject(store, subject)  # refused before any network call
    if card_summary is not None and not isinstance(card_summary, str):
        raise TypeError(f"card_summary must be Markdown text (a str), got {type(card_summary).__name__}.")
    target = PublishTarget(kind=ANALYSIS_STORE, destination=repo_id, store="dataset", token=token)
    run_publish_checks(target, PRE_UPLOAD, skip_checks=skip_checks, add_checks=add_checks)
    skipped = sorted(set(skip_checks))
    if skipped:
        provenance = {**(provenance or {}), "publish_checks_skipped": skipped}
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    identity, published_summary = None, None
    try:
        existing = api.hf_hub_download(repo_id, IT_ARTIFACT_ENVELOPE, repo_type="dataset")
        existing_envelope = validate_artifact_envelope(
            json.loads(Path(existing).read_text(encoding="utf-8")), source=f"{repo_id} (existing)"
        )
        identity = existing_envelope["identity"]
        published_summary = (existing_envelope.get("card") or {}).get("summary")
    except ArtifactEnvelopeError:
        raise  # a malformed EXISTING envelope must fail loudly, not be silently re-minted
    except Exception:
        identity = None  # first publish (no envelope yet)
    if card_summary is None:
        card_summary = published_summary  # the card text survives a re-push that does not restate it

    # parquet interchange: push_to_hub converts per-shard; the LOCAL store remains Arrow (R9)
    with _row_group_target(row_group_bytes):
        store.dataset.push_to_hub(
            repo_id, split=store.split, private=private, token=token, commit_message=commit_message
        )
    parquet_files = sorted(f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet"))
    envelope = build_analysis_store_envelope(
        store,
        identity=identity,
        provenance=provenance,
        interchange={"files": parquet_files, "row_group_target_bytes": row_group_bytes},
        subject=subject,
        card_summary=card_summary,
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
    published = PublishTarget(
        kind=ANALYSIS_STORE,
        destination=repo_id,
        store="dataset",
        remote_files=tuple(f"hf://datasets/{repo_id}@{sha}/{name}" for name in parquet_files[:1]),
        token=token,
    )
    run_publish_checks(published, POST_UPLOAD, skip_checks=skip_checks, add_checks=add_checks)
    return sha


def _published_parquet_urls(repo_id: str, commit: str, envelope: dict) -> list[str]:
    """``hf://`` paths of the Parquet files the envelope lists, at ``commit``; refused when it lists none.

    The listed files are read rather than a path convention, so a reader streams exactly what the writer published.
    """
    files = (envelope["artifacts"].get("interchange") or {}).get("files")
    if not files:
        raise ArtifactEnvelopeError(
            f"{repo_id}@{commit}: the envelope lists no Parquet files (artifacts.interchange.files), so there is"
            " nothing named to stream. Pull it without `streaming=True`, or re-publish it with a current interpretune."
        )
    return [f"hf://datasets/{repo_id}@{commit}/{name}" for name in files]


def _streamed_first_row_bytes(url: str, columns: list[str] | None, storage_options: dict) -> tuple[int, int]:
    """Estimate what reaching the first row of ``url`` fetches, from its footer alone: ``(bytes, row groups)``.

    One footer read (a single block of :data:`STREAMING_BLOCK_SIZE` for a footer that fits it) gives each row group's
    compressed chunk sizes for the projected ``columns``. The scan reads one group ahead, so the estimate is the
    largest group's projected bytes, twice when the file has more than one group.
    """
    import fsspec
    import pyarrow.parquet as pq

    protocol = url.split("://", 1)[0] if "://" in url else "file"
    options = {**storage_options.get(protocol, {}), "block_size": STREAMING_BLOCK_SIZE}
    fs, path = fsspec.core.url_to_fs(url, **options)
    with fs.open(path, "rb", block_size=STREAMING_BLOCK_SIZE) as handle:
        metadata = pq.ParquetFile(handle).metadata
    wanted = None if columns is None else set(columns)
    largest = 0
    for g in range(metadata.num_row_groups):
        group = metadata.row_group(g)
        chunks = (group.column(c) for c in range(group.num_columns))
        projected = sum(
            chunk.total_compressed_size
            for chunk in chunks
            if wanted is None or chunk.path_in_schema.split(".", 1)[0] in wanted
        )
        largest = max(largest, projected)
    return largest * min(2, metadata.num_row_groups), metadata.num_row_groups


def _warn_on_streamed_read_cost(
    source: str, files: list[str], urls: list[str], columns: list[str] | None, storage_options: dict
) -> None:
    """Warn when reaching a streamed store's first row is estimated to fetch more than
    :data:`STREAMED_FIRST_ROW_WARN_BYTES`.

    The read is correct, only slow, so this warns rather than refuses. A listed file that cannot be found is refused,
    naming the envelope's commit and file list, which the raw ``FileNotFoundError`` does not.
    """
    worst = (0, 0, "")
    for name, url in zip(files, urls):
        try:
            estimate, groups = _streamed_first_row_bytes(url, columns, storage_options)
        except FileNotFoundError as e:
            raise ArtifactEnvelopeError(
                f"{source}: the envelope lists {name!r}, which is not in the repository at that commit (it lists"
                f" {files}). Re-publish the store, or pull the commit that holds those files."
            ) from e
        worst = max(worst, (estimate, groups, name))
    estimate, groups, name = worst
    if estimate > STREAMED_FIRST_ROW_WARN_BYTES:
        from interpretune.utils import rank_zero_warn

        projected = f"columns {columns}" if columns is not None else "every column"
        rank_zero_warn(
            f"{source}: reaching the first row of this streamed store is estimated to fetch about"
            f" {estimate / 1e6:.0f} MB ({projected}; {name} has {groups} row group(s), and a scan reads one ahead)."
            " Pass `columns=` to read only what you need, or re-publish it with push_analysis_store, which writes"
            f" row groups of {ANALYSIS_STORE_ROW_GROUP_SIZE}."
        )


def _stream_parquet(
    urls: list[str], split: str, *, columns: list[str] | None, storage_options: dict, token: str | None
):
    """A ``datasets`` stream over ``urls``, reading only ``columns`` and only the row groups a read reaches.

    ``storage_options`` is keyed by filesystem protocol (``{"hf": {"block_size": ...}}``); ``HfFileSystem`` takes the
    block size in its constructor, which is how a nested key reaches it and a flat one does not.

    pyarrow pre-buffers every row group's projected column chunks when a scan starts, so by default reading one row
    fetches the whole file however its row groups are sized; ``pre_buffer=False`` lets a read stop where it stops.
    """
    import pyarrow.dataset as pads
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files={split: urls},
        split=split,
        streaming=True,
        columns=columns,
        token=token,
        storage_options=storage_options,
        fragment_scan_options=pads.ParquetFragmentScanOptions(pre_buffer=False),
    )


def pull_analysis_store(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
    streaming: bool = False,
    columns: list[str] | None = None,
    block_size: int = STREAMING_BLOCK_SIZE,
    protocol: Literal["resolve", "schema"] = "resolve",
):
    """Fetch an analysis-store artifact and hydrate it: envelope-first, formatter re-attached.

    Revision-pinned like component resolution: the envelope resolves first, and the dataset files
    are fetched from the SAME commit. Returns a ready ``AnalysisStore`` whose interpretune
    formatter is re-attached from the envelope's serialized ``col_cfg`` — steering-capable graph
    hydration then flows through the normal ``AnalysisBackend`` seam with no pipeline re-run.

    ``streaming=True`` downloads nothing up front: the store streams the Parquet files the envelope lists, at the
    envelope's commit, through range requests of ``block_size`` bytes, reading only ``columns`` when they are named.
    It can be iterated and read by column but not indexed by position (see ``AnalysisStore``). ``columns`` projects a
    streamed read, so it is refused without ``streaming=True``. A streamed pull first reads each listed file's footer
    and warns when reaching the first row is estimated to cost more than :data:`STREAMED_FIRST_ROW_WARN_BYTES`.

    ``protocol`` says what to do with the batch protocol the envelope records. ``"resolve"`` (the default) uses the
    class itself: interpretune's own, or one from a hub component this process loaded through the remote-code gate,
    and refuses any other, since importing it runs its module. ``"schema"`` imports nothing: the store gets a stand-in
    declaring the field names the envelope records (see ``_schema_only_protocol``), which gives the same attribute
    access, and every column reads and decodes as in a resolved pull. The protocol's own methods, if it defines any,
    are what a schema read gives up.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download

    from interpretune.analysis import AnalysisStore
    from interpretune.hub.components import _TELEMETRY

    if protocol not in ("resolve", "schema"):
        raise ValueError(f"protocol={protocol!r} is not a protocol mode: pass 'resolve' (the default) or 'schema'.")
    if columns is not None and not streaming:
        raise ValueError(
            "columns= projects a streamed read, and a full pull downloads every column regardless. Pass"
            " `streaming=True` with it, or pull the whole store and call `select_columns` on it."
        )
    root = str(cache_dir or ensure_writable_cache_root(IT_ARTIFACTS_HUB_CACHE, IT_ARTIFACTS_HUB_CACHE_SETTING))
    env_path = hf_hub_download(
        repo_id, IT_ARTIFACT_ENVELOPE, repo_type="dataset", revision=revision, cache_dir=root, token=token, **_TELEMETRY
    )
    envelope = validate_artifact_envelope(
        json.loads(Path(env_path).read_text(encoding="utf-8")), source=f"{repo_id}@{revision or 'main'}"
    )
    commit = Path(env_path).parts[Path(env_path).parts.index("snapshots") + 1]
    split = envelope["artifacts"].get("split") or "validation"
    if streaming:
        urls = _published_parquet_urls(repo_id, commit, envelope)
        hf_options = {"hf": {"block_size": block_size, "token": token}}
        files = list(envelope["artifacts"]["interchange"]["files"])
        _warn_on_streamed_read_cost(f"{repo_id}@{commit}", files, urls, columns, hf_options)
        dataset = _stream_parquet(urls, split, columns=columns, storage_options=hf_options, token=token)
    else:
        snapshot = snapshot_download(
            repo_id, repo_type="dataset", revision=commit, cache_dir=root, token=token, **_TELEMETRY
        )
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
    store_kwargs: dict[str, Any] = {}
    if (declared := it_block.get("protocol_cls")) is not None:
        if protocol == "schema":
            recorded = it_block.get("protocol_fields")
            fields = recorded if recorded is not None else list(dataset.column_names or [])
            store_kwargs["protocol_cls"] = _schema_only_protocol(
                declared, fields, "envelope" if recorded is not None else "columns"
            )
        else:
            store_kwargs["protocol_cls"] = _resolve_protocol_path(declared, source=repo_id)
    store = AnalysisStore(dataset=dataset, split=split, it_format_kwargs=format_kwargs or None, **store_kwargs)
    if (record := envelope.get("subject")) is not None:  # absent only from a version-1 envelope
        from interpretune.hub.subject import StoreSubject

        store.subject = StoreSubject.from_dict(record, source=repo_id)
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
