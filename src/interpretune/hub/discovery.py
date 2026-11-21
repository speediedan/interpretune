"""Which publishers a session trusts as sources of shared analysis, and finding what they published.

Trust is per publisher: a Hub namespace (a user or an organization) whose analysis stores and dashboard corpora a
session will discover. One list governs both kinds, so a publisher trusted for one is trusted for the other. Discovery
reads only what each artifact says about itself (a store's ``it_artifact.json``, a corpus's ``dashboards.json``),
never a repo or bucket name.

The list resolves in order: the ``publishers=`` keyword, then the ``IT_TRUSTED_PUBLISHERS`` environment variable
(comma-separated, read at call time like ``IT_TRUST_REMOTE_CODE``), then :data:`DEFAULT_TRUSTED_PUBLISHERS`.
Discovering a publisher's artifacts runs none of its code: a pulled store still resolves ``protocol_cls`` only from
interpretune or a component loaded through the remote-code gate.
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interpretune.hub.subject import StoreSubject

log = logging.getLogger(__name__)

IT_TRUSTED_PUBLISHERS_ENV_VAR = "IT_TRUSTED_PUBLISHERS"

#: The project's own namespace, trusted when neither the keyword nor the environment variable names publishers.
DEFAULT_TRUSTED_PUBLISHERS: tuple[str, ...] = ("speediedan",)

#: The Hub tag every pushed analysis store carries (``generate_artifact_card``): ``<library>-<artifact kind>``.
ANALYSIS_STORE_TAG = "interpretune-analysis-store"

# A Hub namespace: letters, digits, '-', '_' and '.', not starting or ending with a separator. A slash means a repo id
# was passed where a namespace belongs, which would silently match nothing.
_NAMESPACE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


class UntrustedPublisherSpecError(ValueError):
    """A trust list names something that is not a Hub namespace."""


class StoreSchemaSkippedWarning(UserWarning):
    """A published store was left out of a listing because this interpretune cannot read its envelope."""


def trusted_publishers(publishers: Iterable[str] | str | None = None) -> tuple[str, ...]:
    """The publishers this call trusts: ``publishers`` if given, else ``IT_TRUSTED_PUBLISHERS``, else the default.

    The environment variable is read here, at call time, so setting it in a running notebook takes effect. Each entry
    must be a bare Hub namespace (``speediedan``, not ``speediedan/some-repo``); anything else is refused by name
    rather than matched against nothing. Duplicates are dropped, keeping the first occurrence.
    """
    if publishers is None:
        raw = os.environ.get(IT_TRUSTED_PUBLISHERS_ENV_VAR)
        if raw is None:
            return DEFAULT_TRUSTED_PUBLISHERS
        source = f"${IT_TRUSTED_PUBLISHERS_ENV_VAR}"
        entries = [p.strip() for p in raw.split(",")]
    else:
        source = "publishers="
        entries = [publishers] if isinstance(publishers, str) else list(publishers)
    resolved: list[str] = []
    for entry in entries:
        if not isinstance(entry, str) or not _NAMESPACE.match(entry):
            raise UntrustedPublisherSpecError(
                f"{source} names {entry!r}, which is not a Hub namespace: a trusted publisher is a user or "
                "organization name such as 'speediedan', without a repo part."
            )
        if entry not in resolved:
            resolved.append(entry)
    if not resolved:
        raise UntrustedPublisherSpecError(
            f"{source} names no publishers. Name at least one, or unset {IT_TRUSTED_PUBLISHERS_ENV_VAR} to use the "
            f"default {list(DEFAULT_TRUSTED_PUBLISHERS)}."
        )
    return tuple(resolved)


@dataclass(frozen=True)
class StoreListing:
    """One discoverable analysis store, described by its own envelope at a pinned commit.

    ``revision`` is the commit the envelope was read at: pass it to ``pull_analysis_store`` to get exactly the store
    this row describes. ``subject`` is None only for a schema-1 store, which predates the subject block.
    """

    repo_id: str
    publisher: str
    revision: str
    schema: int
    store_id: str
    subject: StoreSubject | None
    generated_by: str | None
    num_rows: int | None
    columns: tuple[str, ...]
    private: bool | None


def _read_envelope(repo_id: str, revision: str, token: str | bool | None, cache_dir: Path | None) -> Any:
    from huggingface_hub import hf_hub_download

    from interpretune.hub.artifacts import IT_ARTIFACT_ENVELOPE
    from interpretune.hub.cache import (
        IT_ARTIFACTS_HUB_CACHE,
        IT_ARTIFACTS_HUB_CACHE_SETTING,
        ensure_writable_cache_root,
    )

    root = str(cache_dir or ensure_writable_cache_root(IT_ARTIFACTS_HUB_CACHE, IT_ARTIFACTS_HUB_CACHE_SETTING))
    path = hf_hub_download(
        repo_id, IT_ARTIFACT_ENVELOPE, repo_type="dataset", revision=revision, cache_dir=root, token=token
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _listing(repo_id: str, publisher: str, revision: str, private: bool | None, envelope: dict) -> StoreListing:
    from interpretune.hub.subject import StoreSubject

    source = f"{repo_id}@{revision[:8]}"
    record = envelope.get("subject")
    artifacts = envelope.get("artifacts") or {}
    return StoreListing(
        repo_id=repo_id,
        publisher=publisher,
        revision=revision,
        schema=envelope["schema"],
        store_id=envelope["identity"]["store_id"],
        subject=None if record is None else StoreSubject.from_dict(record, source=source),
        generated_by=(envelope.get("provenance") or {}).get("generated_by"),
        num_rows=artifacts.get("num_rows"),
        columns=tuple(artifacts.get("columns") or ()),
        private=private,
    )


def find_analysis_stores(
    model_id: str | None = None,
    *,
    publishers: Iterable[str] | str | None = None,
    token: str | bool | None = None,
    cache_dir: Path | None = None,
) -> list[StoreListing]:
    """The analysis stores the trusted publishers have pushed, optionally only those computed on ``model_id``.

    Lists each publisher's datasets tagged ``interpretune-analysis-store`` and reads each one's ``it_artifact.json``
    at the commit the listing reports (a few KB each; no Parquet moves). The ``model_id`` match is on the store's
    ``subject``, so a schema-1 store, which has none, is left out of a filtered listing.

    A tagged dataset whose envelope this interpretune cannot read (written by a newer interpretune, below the readable
    floor, missing or malformed) is left out with a :class:`StoreSchemaSkippedWarning` naming it and why, so one
    unreadable store never hides the rest. A failure to reach the Hub raises. ``token`` reaches private stores the
    token can read.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError

    from interpretune.hub.artifacts import ArtifactEnvelopeError, validate_artifact_envelope

    api = HfApi(token=token)
    found: list[StoreListing] = []
    for publisher in trusted_publishers(publishers):
        for info in api.list_datasets(author=publisher, filter=ANALYSIS_STORE_TAG, token=token):
            revision = info.sha or api.dataset_info(info.id, token=token).sha
            if revision is None:
                raise LookupError(f"the Hub reported no head commit for {info.id!r}, so its envelope cannot be pinned")
            source = f"{info.id}@{revision[:8]}"
            try:
                envelope = validate_artifact_envelope(_read_envelope(info.id, revision, token, cache_dir), source)
            except ArtifactEnvelopeError as e:
                warnings.warn(f"left {info.id!r} out of the listing: {e}", StoreSchemaSkippedWarning, stacklevel=2)
                continue
            except (EntryNotFoundError, json.JSONDecodeError) as e:
                # tagged, but no parsable envelope at that commit; a network or auth failure still raises, since a
                # listing short by an unreachable store would read as complete
                warnings.warn(
                    f"left {info.id!r} out of the listing: its it_artifact.json at {revision[:8]} could not be read "
                    f"({type(e).__name__}: {e})",
                    StoreSchemaSkippedWarning,
                    stacklevel=2,
                )
                continue
            listing = _listing(info.id, publisher, revision, getattr(info, "private", None), envelope)
            if model_id is not None and (listing.subject is None or listing.subject.model_id != model_id):
                log.debug("skipping %s: computed on %s, not %s", source, listing.subject, model_id)
                continue
            found.append(listing)
    return found
