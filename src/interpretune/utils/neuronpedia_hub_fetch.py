"""Fetch a published dashboard corpus and import it into a local Neuronpedia.

The orchestration lives here rather than in ``scripts/fetch_dashboards_from_hub.py`` so callers other
than the CLI can use it -- notably the example notebooks, which need dashboards present before they
can select or steer features and should not make a user go and read a guide first.

:func:`dashboards_present` is the cheap, parameterized probe: ask whether a specific (model, source
set) is already populated, in one query, without embedding SQL at the call site.
:func:`ensure_dashboards` is that probe plus the fetch, so a notebook cell can be a single call whose
cost is one query in the common case where the dashboards are already there.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interpretune.utils.neuronpedia_dashboard_hub import download_dashboard_run
from interpretune.utils.neuronpedia_db_utils import _redact_connection_url
from interpretune.utils.neuronpedia_source_conflicts import (
    ConflictPolicy,
    describe_source_set,
)

__all__ = [
    "CORPUS_MANIFEST",
    "DEFAULT_CORPUS_COLLECTIONS",
    "CorpusListing",
    "CorpusPlan",
    "DashboardDatabaseUnreachableError",
    "dashboards_present",
    "default_corpus_dest",
    "ensure_dashboards",
    "fetch_dashboards",
    "find_dashboard_corpora",
    "read_remote_manifest",
    "resolve_bucket_for",
    "resolve_corpus_config",
]

log = logging.getLogger(__name__)

#: Self-description carried at the root of every published corpus.
CORPUS_MANIFEST = "dashboards.json"

CONFIG_DIR = Path(__file__).resolve().parents[2] / "it_examples/neuronpedia_dashboard"

#: The Hub collections a client reads to discover published corpora: the trust list. Each item is a bucket whose own
#: ``dashboards.json`` says what it holds, so a collection names corpora and never describes them. It defaults to the
#: project's collection; pass ``collections=`` to read another publisher's.
DEFAULT_CORPUS_COLLECTIONS: tuple[str, ...] = ("speediedan/neuronpedia-dashboard-corpora-6af900009df4b4831ab24069",)


@dataclass(frozen=True)
class CorpusListing:
    """One discoverable corpus, described by its own manifest, enough to choose it before any GiB moves."""

    bucket: str
    collection: str
    model_id: str | None
    source_set_id: str | None
    prompt_corpus: str | None
    n_prompts: int | None
    n_tokens: int | None
    layers: int
    page_index: bool | None


def _prompt_corpus_label(corpus: dict[str, Any]) -> str | None:
    """``<hf_dataset_path>`` or ``<hf_dataset_path>:<config>``, the name a caller selects a corpus by."""
    path = corpus.get("hf_dataset_path")
    if not path:
        return None
    config = corpus.get("hf_dataset_config_name")
    return f"{path}:{config}" if config else path


@dataclass(frozen=True)
class CorpusPlan:
    """What a fetch would do, resolved from the corpus's own manifest."""

    bucket: str
    model_id: str
    source_set_id: str
    n_prompts: int | None
    n_tokens: int | None
    layers: int
    page_index: bool | None
    config: Path
    run_dir: Path


def default_corpus_dest() -> Path:
    """Where corpora land when the caller does not say.

    Mirrors the pipeline's artifact-root resolution so a downloaded corpus sits beside locally generated ones instead of
    in a second place nobody remembers.
    """
    if root := os.environ.get("IT_NP_CACHE"):
        return Path(root) / "hub_downloads"
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache/huggingface")
    return Path(hf_home) / "interpretune/neuronpedia/hub_downloads"


class DashboardDatabaseUnreachableError(ConnectionError):
    """The local Neuronpedia database could not be queried, so whether its dashboards are present is unknown."""


def _probe_dashboards(db_url: str, *, model_id: str, source_set_id: str, min_sources: int) -> bool:
    """:func:`dashboards_present`, raising :class:`DashboardDatabaseUnreachableError` when it cannot ask."""
    try:
        occupancy = describe_source_set(db_url, model_id=model_id, source_set_id=source_set_id)
    except Exception as exc:
        raise DashboardDatabaseUnreachableError(
            f"Cannot query the local Neuronpedia database at {_redact_connection_url(db_url)} to check for the"
            f" {model_id}/{source_set_id} dashboards: {exc}. Start the database, or fix the URL, and retry. Downloading"
            " the corpus would not help: it is imported into this same database."
        ) from exc
    return occupancy.source_count >= min_sources and occupancy.neuron_count > 0


def dashboards_present(
    db_url: str,
    *,
    model_id: str,
    source_set_id: str,
    min_sources: int = 1,
) -> bool:
    """Is this (model, source set) already populated in the local Neuronpedia?

    The parameterized alternative to an ad-hoc SQL snippet at each call site. ``min_sources`` guards
    against treating a PARTIAL import as present: pass the layer count you require (26 for the
    published corpora) and a half-finished import reports False rather than sending a caller on to
    work with missing layers.

    Returns False rather than raising when the database cannot be reached, so a caller can decide
    whether that is fatal. :func:`ensure_dashboards` does not take that reading, since only an absent
    corpus is fixed by a download.
    """
    try:
        return _probe_dashboards(db_url, model_id=model_id, source_set_id=source_set_id, min_sources=min_sources)
    except DashboardDatabaseUnreachableError as exc:
        log.warning("could not probe %s/%s (%s); treating as absent", model_id, source_set_id, exc.__cause__)
        return False


def find_dashboard_corpora(
    model_id: str | None = None,
    source_set_id: str | None = None,
    *,
    collections: tuple[str, ...] | list[str] = DEFAULT_CORPUS_COLLECTIONS,
    token: str | bool | None = None,
    collection_token: str | bool | None = False,
) -> list[CorpusListing]:
    """The published corpora the ``collections`` list, optionally only those serving a (model, source set).

    Reads each collection's bucket items and each bucket's ``dashboards.json`` (about 3 KB, with ``token``); nothing
    else is downloaded. The match is on what the corpus says about itself, never on the bucket's name.

    Collections are read anonymously unless ``collection_token`` is given (for a private collection): the Hub refuses
    a fine-grained token without ``collection.read`` even on a public collection, and the default one is public.
    """
    from huggingface_hub import HfApi

    found: list[CorpusListing] = []
    for slug in collections:
        for item in HfApi(token=collection_token).get_collection(slug, token=collection_token).items:
            if item.item_type != "bucket":
                log.debug(
                    "skipping %s item %s of %s: corpora are discovered as buckets", item.item_type, item.item_id, slug
                )
                continue
            manifest = read_remote_manifest(item.item_id, token)
            corpus = manifest.get("prompt_corpus") or {}
            listing = CorpusListing(
                bucket=item.item_id,
                collection=slug,
                model_id=(manifest.get("model") or {}).get("name"),
                source_set_id=(manifest.get("source_set") or {}).get("id"),
                prompt_corpus=_prompt_corpus_label(corpus),
                n_prompts=corpus.get("n_prompts"),
                n_tokens=corpus.get("n_tokens_in_prompt"),
                layers=len((manifest.get("layers") or {}).get("generated") or []),
                page_index=(manifest.get("artifacts") or {}).get("page_index"),
            )
            if model_id is not None and listing.model_id != model_id:
                continue
            if source_set_id is not None and listing.source_set_id != source_set_id:
                continue
            found.append(listing)
    return found


def resolve_bucket_for(
    model_id: str,
    source_set_id: str,
    *,
    prompt_corpus: str | None = None,
    collections: tuple[str, ...] | list[str] = DEFAULT_CORPUS_COLLECTIONS,
    token: str | bool | None = None,
    collection_token: str | bool | None = False,
) -> str:
    """The published bucket serving a (model, source set), chosen from the ``collections``.

    One prompt corpus per source set is not guaranteed, so several matches are refused unless ``prompt_corpus``
    (``<hf_dataset_path>`` or ``<hf_dataset_path>:<config>``) picks one, and no match is refused with what the
    collections do serve.
    """
    discover = {"collections": collections, "token": token, "collection_token": collection_token}
    matches = find_dashboard_corpora(model_id, source_set_id, **discover)
    if prompt_corpus is not None:
        matches = [m for m in matches if prompt_corpus in (m.prompt_corpus, (m.prompt_corpus or "").split(":")[0])]
    if len(matches) == 1:
        return matches[0].bucket
    if not matches:
        served = sorted(
            {f"{c.model_id}/{c.source_set_id} ({c.prompt_corpus})" for c in find_dashboard_corpora(**discover)}
        )
        wanted = f"{model_id!r}/{source_set_id!r}" + (f" with prompt corpus {prompt_corpus!r}" if prompt_corpus else "")
        raise LookupError(
            f"no corpus in {list(collections)} serves {wanted}. They serve: {', '.join(served) or 'nothing'}. Pass"
            " bucket= explicitly, another collections=, or generate the dashboards locally."
        )
    options = ", ".join(f"{m.prompt_corpus} ({m.bucket})" for m in matches)
    raise LookupError(
        f"several corpora serve {model_id!r}/{source_set_id!r}: {options}. Pass prompt_corpus= to choose one."
    )


def read_remote_manifest(bucket: str, token: str | bool | None = None) -> dict[str, Any]:
    """Fetch just ``dashboards.json`` (~3 KB) so a plan is known before any GiB moves."""
    import json
    import tempfile

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / CORPUS_MANIFEST
        HfApi(token=token).download_bucket_files(bucket, [(CORPUS_MANIFEST, target)], token=token)
        return json.loads(target.read_text(encoding="utf-8"))


def resolve_corpus_config(manifest: dict[str, Any], explicit: Path | None = None) -> Path:
    """Pick the committed config whose model + source set match the corpus.

    Matched on what the corpus says about itself rather than on the bucket name: bucket names are
    free text, and a renamed bucket would otherwise select the wrong import settings.
    """
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"config {explicit} does not exist")
        return Path(explicit)

    # Resolved through the pipeline's own loader, not yaml.safe_load: these configs use EXTENDS, and
    # the inherited keys are exactly the ones matched on (the RTE config declares its source set but
    # inherits model_name from its base, so a raw read matches nothing).
    from interpretune.utils.neuronpedia_dashboard_pipeline import load_dashboard_pipeline_config_payload

    want_model = (manifest.get("model") or {}).get("name")
    want_set = (manifest.get("source_set") or {}).get("id")
    want_prompts = (manifest.get("prompt_corpus") or {}).get("n_prompts")

    resolved: dict[Path, dict] = {}
    for candidate in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            payload = load_dashboard_pipeline_config_payload(candidate) or {}
        except Exception as exc:  # a malformed sibling must not block a valid match
            log.debug("skipping %s: %s", candidate, exc)
            continue
        resolved[candidate] = payload.get("pipeline") or {}

    matches = [
        c
        for c, p in resolved.items()
        if p.get("model_name") == want_model and p.get("neuronpedia_source_set_id") == want_set
    ]
    if not matches:
        raise LookupError(f"no committed config matches model={want_model!r} source_set={want_set!r} in {CONFIG_DIR}")
    if len(matches) > 1:
        # Several configs can share a source set (different prompt counts); the corpus knows which.
        narrowed = [c for c in matches if resolved[c].get("n_prompts_total") == want_prompts]
        matches = narrowed or matches
    if len(matches) > 1:
        raise LookupError(f"several configs match this corpus; pass one explicitly: {matches}")
    return matches[0]


def plan_fetch(
    bucket: str,
    *,
    dest: Path | None = None,
    config: Path | None = None,
    token: str | None = None,
) -> CorpusPlan:
    """Resolve everything a fetch needs, without downloading the corpus."""
    manifest = read_remote_manifest(bucket, token)
    model_id = (manifest.get("model") or {}).get("name") or "?"
    source_set_id = (manifest.get("source_set") or {}).get("id") or "?"
    corpus = manifest.get("prompt_corpus") or {}
    dest_parent = (Path(dest) if dest else default_corpus_dest()).expanduser()
    return CorpusPlan(
        bucket=bucket,
        model_id=model_id,
        source_set_id=source_set_id,
        n_prompts=corpus.get("n_prompts"),
        n_tokens=corpus.get("n_tokens_in_prompt"),
        layers=len((manifest.get("layers") or {}).get("generated") or []),
        page_index=(manifest.get("artifacts") or {}).get("page_index"),
        config=resolve_corpus_config(manifest, config),
        # The corpus lands under its OWN identity regardless of conflict policy: a rename changes
        # where rows go in the database, not what the corpus is, and making the download path depend
        # on database contents would put the same corpus in different directories on different hosts.
        run_dir=dest_parent / f"{model_id}_{source_set_id}",
    )


def build_import_command(
    plan: CorpusPlan,
    *,
    db_url: str | None,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    rename_suffix: str | None = None,
    allow_explanation_loss: bool = False,
) -> list[str]:
    """The import invocation, with the conflict policy passed THROUGH to the pipeline.

    The pipeline owns conflict resolution -- it needs identical behaviour for locally generated
    corpora -- so nothing is decided here that it would decide differently.
    """
    command = [
        sys.executable,
        "-m",
        "interpretune.utils.neuronpedia_dashboard_pipeline",
        f"--config={plan.config}",
        "--import-only-local-db",
        f"--run-root={plan.run_dir.parent}",
    ]
    if rename_suffix:
        command.append(f"--rename-suffix={rename_suffix}")
    elif policy is ConflictPolicy.RENAME:
        command.append("--autosuffix-on-exists")
    elif policy is ConflictPolicy.OVERWRITE:
        command.append("--overwrite-existing")
    if allow_explanation_loss:
        command.append("--allow-explanation-loss")
    if db_url:
        command.append(f"--local-db-url={db_url}")
    return command


def fetch_dashboards(
    bucket: str,
    *,
    db_url: str | None = None,
    dest: Path | None = None,
    config: Path | None = None,
    token: str | None = None,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    rename_suffix: str | None = None,
    allow_explanation_loss: bool = False,
    download: bool = True,
    import_to_db: bool = True,
) -> CorpusPlan:
    """Download a published corpus and import it. Returns the plan that was executed.

    Raises ``subprocess.CalledProcessError`` if the import fails, so a caller that ignores the return
    value still cannot mistake a failed import for a completed one.
    """
    plan = plan_fetch(bucket, dest=dest, config=config, token=token)
    log.info("corpus     : %s / %s", plan.model_id, plan.source_set_id)
    log.info("prompts    : %s x %s tokens, %s layers", plan.n_prompts, plan.n_tokens, plan.layers)
    log.info("destination: %s", plan.run_dir)

    if download:
        plan.run_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s ...", bucket)
        download_dashboard_run(bucket, plan.run_dir, store="bucket", token=token)

    if import_to_db:
        command = build_import_command(
            plan,
            db_url=db_url,
            policy=policy,
            rename_suffix=rename_suffix,
            allow_explanation_loss=allow_explanation_loss,
        )
        log.info("importing:\n  %s", " ".join(command))
        subprocess.run(command, check=True)
    return plan


def ensure_dashboards(
    db_url: str,
    *,
    model_id: str,
    source_set_id: str,
    bucket: str | None = None,
    prompt_corpus: str | None = None,
    min_sources: int = 1,
    dest: Path | None = None,
    token: str | None = None,
    allow_fetch: bool = True,
    **fetch_kwargs: Any,
) -> bool:
    """Make sure a (model, source set) is populated locally, downloading it if it is not.

    Returns True when the dashboards were already present, False when they had to be fetched. Cheap
    in the common case: one count query, no network.

    ``allow_fetch=False`` turns this into an assertion -- useful where an automatic multi-GiB
    download would be a surprise (CI, for instance) -- raising with the command to run instead.

    A database that cannot be queried raises :class:`DashboardDatabaseUnreachableError` rather than
    reading as absent: a download cannot fix it, and starting one hides the connection failure behind
    a long transfer into a database that will then refuse the import.
    """
    if _probe_dashboards(db_url, model_id=model_id, source_set_id=source_set_id, min_sources=min_sources):
        log.info("dashboards for %s/%s already present", model_id, source_set_id)
        return True

    bucket = bucket or resolve_bucket_for(model_id, source_set_id, prompt_corpus=prompt_corpus, token=token)
    if not allow_fetch:
        raise RuntimeError(
            f"no dashboards for {model_id}/{source_set_id} in the local database, and allow_fetch is off.\n"
            f"Fetch them with:\n"
            f"  python scripts/fetch_dashboards_from_hub.py --bucket {bucket} --local-db-url <url>"
        )

    log.info("dashboards for %s/%s are absent; fetching %s", model_id, source_set_id, bucket)
    fetch_dashboards(bucket, db_url=db_url, dest=dest, token=token, **fetch_kwargs)
    return False
