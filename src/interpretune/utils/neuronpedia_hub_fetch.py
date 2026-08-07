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
from interpretune.utils.neuronpedia_source_conflicts import (
    ConflictPolicy,
    describe_source_set,
)

__all__ = [
    "CORPUS_MANIFEST",
    "KNOWN_DASHBOARD_BUCKETS",
    "CorpusPlan",
    "dashboards_present",
    "default_corpus_dest",
    "ensure_dashboards",
    "fetch_dashboards",
    "read_remote_manifest",
    "resolve_bucket_for",
    "resolve_corpus_config",
]

log = logging.getLogger(__name__)

#: Self-description carried at the root of every published corpus.
CORPUS_MANIFEST = "dashboards.json"

CONFIG_DIR = Path(__file__).resolve().parents[2] / "it_examples/config/neuronpedia_dashboard"

# TEMPORARY STUB -- a hardcoded stand-in for a dashboard repository that does not exist yet.
#
# A client-side registry of published corpora is an anti-pattern and is not what this should be. It
# cannot see a corpus published after the release that shipped it, every publisher would need a patch
# merged here to be discoverable, and the mapping duplicates -- with no mechanism keeping it honest --
# metadata each corpus already carries about itself in ``dashboards.json``.
#
# It exists because there is currently no way to ASK the Hub "which published corpus serves
# (model, source set)?". Buckets are not indexed by content, bucket names are free text, and guessing
# one from a naming convention would download a corpus that imports into the requested source set
# while containing something else entirely -- a wrong answer that looks exactly like a right one. So
# the mapping is stated explicitly and the two entries below are the two corpora the example notebooks
# need; anything else must pass ``bucket=``.
#
# Standardized hub-based discovery replaces this wholesale -- corpora self-register (a dataset-repo
# collection, a tag convention, or an index artifact) and ``resolve_bucket_for`` becomes a query.
#
# TRACKED IN: docs/source/roadmap.md, "Standardized hub-based dashboard discovery" (under
# "Following: hub-resident, streamable dashboards"), which records the requirements a replacement
# must meet and the candidate mechanisms; docs/neuronpedia_dashboard_pipeline.md carries the same
# under "Wave 2 requirement: standardized hub-based dashboard discovery". A Wave 2 issue will be
# opened to track the work formally -- until then those sections ARE the tracking record, so update
# them alongside any change here.
#
# Treat any addition here as a stopgap, not as the extension point.
KNOWN_DASHBOARD_BUCKETS: dict[tuple[str, str], str] = {
    ("gemma-3-1b-it", "gemmascope-2-transcoder-16k"): (
        "speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__monology__dashboards"
    ),
    ("gemma-3-1b-it", "gemmascope-2-transcoder-16k-rte"): (
        "speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__rte__dashboards"
    ),
}


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
    whether that is fatal; the subsequent import reports the connection failure properly.
    """
    try:
        occupancy = describe_source_set(db_url, model_id=model_id, source_set_id=source_set_id)
    except Exception as exc:
        log.warning("could not probe %s/%s (%s); treating as absent", model_id, source_set_id, exc)
        return False
    return occupancy.source_count >= min_sources and occupancy.neuron_count > 0


def resolve_bucket_for(model_id: str, source_set_id: str) -> str:
    """The published bucket serving a (model, source set), or a loud failure.

    Backed by :data:`KNOWN_DASHBOARD_BUCKETS`, a temporary hardcoded stub; see the note there. This
    becomes a Hub query once standardized dashboard discovery exists, so callers should treat the
    lookup as authoritative and the table behind it as not.
    """
    try:
        return KNOWN_DASHBOARD_BUCKETS[(model_id, source_set_id)]
    except KeyError:
        known = ", ".join(f"{m}/{s}" for m, s in sorted(KNOWN_DASHBOARD_BUCKETS))
        raise LookupError(
            f"no published corpus is registered for {model_id!r}/{source_set_id!r}. "
            f"Known: {known}. Pass bucket= explicitly, or generate the dashboards locally."
        ) from None


def read_remote_manifest(bucket: str, token: str | None = None) -> dict[str, Any]:
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
    """
    if dashboards_present(db_url, model_id=model_id, source_set_id=source_set_id, min_sources=min_sources):
        log.info("dashboards for %s/%s already present", model_id, source_set_id)
        return True

    bucket = bucket or resolve_bucket_for(model_id, source_set_id)
    if not allow_fetch:
        raise RuntimeError(
            f"no dashboards for {model_id}/{source_set_id} in the local database, and allow_fetch is off.\n"
            f"Fetch them with:\n"
            f"  python scripts/fetch_dashboards_from_hub.py --bucket {bucket} --local-db-url <url>"
        )

    log.info("dashboards for %s/%s are absent; fetching %s", model_id, source_set_id, bucket)
    fetch_dashboards(bucket, db_url=db_url, dest=dest, token=token, **fetch_kwargs)
    return False
