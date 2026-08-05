#!/usr/bin/env python3
"""Download a published dashboard corpus and import it into a local Neuronpedia, in one command.

    python scripts/fetch_dashboards_from_hub.py --bucket <ns>/<bucket> --rename-existing

Everything else is derived. The corpus carries ``dashboards.json`` (model, source set, prompt corpus,
layers) and ``source_ids.json`` (identity), so the bucket id is enough to pick the matching committed
config, choose a destination, and import under the right source ids. Directories are created as
needed; ``--dest`` overrides where they go.

WHY THIS EXISTS RATHER THAN A LIST OF STEPS IN THE DOCS
-------------------------------------------------------
The manual sequence was download, then work out which config matches, then remember that
``--run-root`` points at the PARENT of the download, then know that importing over an occupied source
set is a silent no-op. Four chances to get it wrong, and the last one fails by *appearing to
succeed*. Here the failure modes are checked before any bytes move.

CONFLICTS ARE REFUSED BY DEFAULT
--------------------------------
Downloading and generating locally populate the SAME source set. The importer uses
``ON CONFLICT DO NOTHING``, so a second import reports success and writes nothing. This refuses up
front and offers both ways forward:

    --rename-existing     keep both; import under ``<set>__hub``
    --overwrite-existing  delete the resident rows first (refuses if explanations would cascade,
                          unless --allow-explanation-loss)

See :mod:`interpretune.utils.neuronpedia_source_conflicts` for why ``rename`` renames the incoming
corpus rather than the resident one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from interpretune.utils.neuronpedia_dashboard_hub import download_dashboard_run
from interpretune.utils.neuronpedia_source_conflicts import (
    DEFAULT_RENAME_SUFFIX,
    ConflictPolicy,
    describe_source_set,
    render_conflict_report,
    suffix_source_set_id,
)

CORPUS_MANIFEST = "dashboards.json"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "src/it_examples/config/neuronpedia_dashboard"
log = logging.getLogger("fetch_dashboards")


def _default_dest() -> Path:
    """Where corpora land when the caller does not say.

    Mirrors the pipeline's own artifact-root resolution so a downloaded corpus sits beside locally generated ones
    instead of in a second place nobody remembers.
    """
    if root := os.environ.get("IT_NP_CACHE"):
        return Path(root) / "hub_downloads"
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache/huggingface")
    return Path(hf_home) / "interpretune/neuronpedia/hub_downloads"


def _read_remote_manifest(bucket: str, token: str | None) -> dict:
    """Fetch just ``dashboards.json`` (~3 KB) so the plan is known before any GiB moves."""
    import tempfile

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / CORPUS_MANIFEST
        HfApi(token=token).download_bucket_files(bucket, [(CORPUS_MANIFEST, target)], token=token)
        return json.loads(target.read_text(encoding="utf-8"))


def _resolve_config(manifest: dict, explicit: Path | None) -> Path:
    """Pick the committed config whose model + source set match the corpus.

    Matching on what the corpus says about itself, rather than on the bucket name, is deliberate: bucket names are free
    text and a renamed bucket would silently select the wrong import settings.
    """
    if explicit:
        if not explicit.is_file():
            raise SystemExit(f"--config {explicit} does not exist")
        return explicit

    # Resolve through the pipeline's own loader, not yaml.safe_load: these configs use EXTENDS, and
    # the inherited keys are exactly the ones being matched on (the RTE config declares its source set
    # but inherits model_name from its base, so a raw read matches nothing).
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
        raise SystemExit(
            f"no committed config matches model={want_model!r} source_set={want_set!r}.\n"
            f"Looked in {CONFIG_DIR}. Pass --config explicitly."
        )
    if len(matches) > 1:
        # Several configs can share a source set (different prompt counts); the corpus knows which.
        narrowed = [c for c in matches if resolved[c].get("n_prompts_total") == want_prompts]
        matches = narrowed or matches
    if len(matches) > 1:
        listed = "\n  ".join(str(m) for m in matches)
        raise SystemExit(f"several configs match this corpus; pass --config to choose:\n  {listed}")
    return matches[0]


def _import_command(*, config: Path, run_root: Path, args: argparse.Namespace) -> list[str]:
    """Build the import invocation, passing the conflict policy THROUGH to the pipeline.

    The pipeline owns conflict resolution (it needs the same behaviour for locally generated corpora), so this does not
    decide anything the pipeline would decide differently. The pre-flight below is only an early, read-only warning so a
    refusal does not arrive after a multi-GiB download.
    """
    command = [
        sys.executable,
        "-m",
        "interpretune.utils.neuronpedia_dashboard_pipeline",
        f"--config={config}",
        "--import-only-local-db",
        f"--run-root={run_root}",
    ]
    if args.rename_existing:
        command.append("--rename-existing")
        command.append(f"--source-set-rename-suffix={args.rename_suffix}")
    elif args.overwrite_existing:
        command.append("--overwrite-existing")
    if args.allow_explanation_loss:
        command.append("--allow-explanation-loss")
    if args.local_db_url:
        command.append(f"--local-db-url={args.local_db_url}")
    return command


def _summarize(db_url: str, *, model_id: str, source_set_id: str) -> str:
    occupancy = describe_source_set(db_url, model_id=model_id, source_set_id=source_set_id)
    return (
        f"IMPORT SUMMARY  model={model_id} source_set={source_set_id}\n"
        f"  sources      : {occupancy.source_count}\n"
        f"  neurons      : {occupancy.neuron_count}\n"
        f"  explanations : {occupancy.explanation_count}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, help="Published corpus, e.g. <namespace>/<bucket-name>.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Parent directory for the corpus (created if absent). Defaults to $IT_NP_CACHE/hub_downloads, "
        "else $HF_HOME/interpretune/neuronpedia/hub_downloads.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Override the auto-selected import config.")
    parser.add_argument(
        "--local-db-url", default=None, help="Target Postgres. Defaults to the pipeline's own resolution."
    )
    parser.add_argument("--token", default=None, help="Hub token. Not needed for public buckets.")

    conflict = parser.add_mutually_exclusive_group()
    conflict.add_argument(
        "--rename-existing",
        action="store_true",
        help="On conflict, keep the existing set and import this corpus under <set>__<suffix>.",
    )
    conflict.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="On conflict, delete the resident rows for this set, then import.",
    )
    parser.add_argument("--rename-suffix", default=DEFAULT_RENAME_SUFFIX, help=f"Default: {DEFAULT_RENAME_SUFFIX}.")
    parser.add_argument(
        "--allow-explanation-loss",
        action="store_true",
        help="Permit --overwrite-existing to cascade away explanations, which no corpus can regenerate.",
    )

    parser.add_argument("--download-only", action="store_true", help="Fetch the corpus; skip the import.")
    parser.add_argument("--skip-download", action="store_true", help="Import an already-downloaded corpus.")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan; change nothing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = _read_remote_manifest(args.bucket, args.token)
    model_id = (manifest.get("model") or {}).get("name") or "?"
    source_set_id = (manifest.get("source_set") or {}).get("id") or "?"
    corpus = manifest.get("prompt_corpus") or {}
    layers = (manifest.get("layers") or {}).get("generated") or []

    log.info("corpus       : %s / %s", model_id, source_set_id)
    log.info("prompts      : %s x %s tokens", corpus.get("n_prompts", "?"), corpus.get("n_tokens_in_prompt", "?"))
    log.info("layers       : %s", len(layers))
    log.info("page index   : %s", (manifest.get("artifacts") or {}).get("page_index"))

    config = _resolve_config(manifest, args.config)
    log.info("config       : %s", config)

    policy = ConflictPolicy.ERROR
    if args.rename_existing:
        policy = ConflictPolicy.RENAME
    elif args.overwrite_existing:
        policy = ConflictPolicy.OVERWRITE

    db_url = args.local_db_url
    effective_set_id = source_set_id
    if db_url:
        # PRE-FLIGHT ONLY, and read-only: refuse before spending a multi-GiB download on an import
        # that would be rejected. The pipeline repeats this check authoritatively at import time.
        try:
            occupancy = describe_source_set(db_url, model_id=model_id, source_set_id=source_set_id)
        except Exception as exc:  # an unreachable DB is the import's problem to report, not this one
            log.warning("could not pre-check the target source set (%s); continuing", exc)
            occupancy = None
        if occupancy is not None and occupancy.occupied:
            if policy is ConflictPolicy.ERROR:
                print(
                    f"\nREFUSING TO IMPORT:\n{render_conflict_report(occupancy, rename_suffix=args.rename_suffix)}",
                    file=sys.stderr,
                )
                return 3
            if policy is ConflictPolicy.RENAME:
                effective_set_id = suffix_source_set_id(source_set_id, args.rename_suffix)
                log.info("conflict     : existing %r kept; importing as %r", source_set_id, effective_set_id)
            elif occupancy.explanation_count and not args.allow_explanation_loss:
                # The pipeline would refuse for this reason anyway; catching it here is the whole
                # point of a pre-flight, since otherwise the refusal lands after a multi-GiB download.
                print(
                    f"\nREFUSING TO IMPORT:\n--overwrite-existing would delete "
                    f"{occupancy.neuron_count} neurons from {source_set_id!r}, cascading away "
                    f"{occupancy.explanation_count} explanations. Activations can be regenerated "
                    "from a corpus; explanations cannot. Pass --allow-explanation-loss to proceed, "
                    "or --rename-existing to keep both.",
                    file=sys.stderr,
                )
                return 3
            else:
                log.info("conflict     : %r will be REPLACED (%s neurons)", source_set_id, occupancy.neuron_count)
    elif policy is ConflictPolicy.RENAME:
        effective_set_id = suffix_source_set_id(source_set_id, args.rename_suffix)

    # The corpus lands under its OWN identity regardless of policy: a rename changes where rows go in
    # the database, not what the corpus is, and making the download path depend on database contents
    # would put the same corpus in different directories on different machines.
    dest_parent = (args.dest or _default_dest()).expanduser()
    run_dir = dest_parent / f"{model_id}_{source_set_id}"
    log.info("destination  : %s", run_dir)

    command = _import_command(config=config, run_root=dest_parent, args=args)

    if args.dry_run:
        log.info("\nDRY RUN -- nothing downloaded, nothing imported.")
        log.info("would import with:\n  %s", " ".join(command))
        return 0

    if not args.skip_download:
        dest_parent.mkdir(parents=True, exist_ok=True)
        log.info("\ndownloading %s ...", args.bucket)
        download_dashboard_run(args.bucket, run_dir, store="bucket", token=args.token)
        log.info("downloaded to %s", run_dir)

    if args.download_only:
        log.info("\n--download-only: skipping import.")
        return 0

    log.info("\nimporting:\n  %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"\nimport failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    # Verification is part of the command, not a follow-up the caller has to remember: an import that
    # wrote nothing and an import that wrote everything both exit 0 under ON CONFLICT DO NOTHING.
    if db_url:
        log.info("\n%s", _summarize(db_url, model_id=model_id, source_set_id=effective_set_id))
    else:
        log.info("\n(no --local-db-url given; skipping the post-import summary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
