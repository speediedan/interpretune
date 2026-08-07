#!/usr/bin/env python3
"""Download a published dashboard corpus and import it into a local Neuronpedia, in one command.

    python scripts/fetch_dashboards_from_hub.py --bucket <ns>/<bucket> --autosuffix-on-exists

Everything else is derived. The corpus carries ``dashboards.json`` (model, source set, prompt corpus,
layers) and ``source_ids.json`` (identity), so the bucket id is enough to pick the matching committed
config, choose a destination, and import under the right source ids. Directories are created as
needed; ``--dest`` overrides where they go.

CONFLICTS ARE REFUSED BY DEFAULT
--------------------------------
Downloading and generating locally populate the SAME source set. The importer uses
``ON CONFLICT DO NOTHING``, so a second import reports success and writes nothing. This refuses up
front and offers both ways forward:

    --autosuffix-on-exists  keep both; import under ``<set>__<UTC timestamp>``
    --rename-suffix <name>  keep both, under a name you choose
    --overwrite-existing    delete the resident rows first (refuses if explanations would cascade,
                            unless --allow-explanation-loss)

See :mod:`interpretune.utils.neuronpedia_source_conflicts` for why ``rename`` renames the incoming
corpus rather than the resident one.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from interpretune.utils.neuronpedia_hub_fetch import (
    build_import_command,
    plan_fetch,
)
from interpretune.utils.neuronpedia_dashboard_hub import download_dashboard_run
from interpretune.utils.neuronpedia_source_conflicts import (
    ConflictPolicy,
    describe_source_set,
    generate_autosuffix,
    render_conflict_report,
    suffix_source_set_id,
)

log = logging.getLogger("fetch_dashboards")


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
        "--autosuffix-on-exists",
        action="store_true",
        help="On conflict, keep the existing set and import this corpus under <set>__<UTC timestamp>. "
        "Never refuses over the generated name.",
    )
    conflict.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="On conflict, delete the resident rows for this set, then import.",
    )
    parser.add_argument(
        "--rename-suffix",
        default=None,
        help="Import under <set>__<this> instead of a generated timestamp. Implies the rename "
        "policy. A collision on a name you chose is an error rather than something routed around.",
    )
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

    plan = plan_fetch(args.bucket, dest=args.dest, config=args.config, token=args.token)
    log.info("corpus       : %s / %s", plan.model_id, plan.source_set_id)
    log.info("prompts      : %s x %s tokens", plan.n_prompts, plan.n_tokens)
    log.info("layers       : %s", plan.layers)
    log.info("page index   : %s", plan.page_index)
    log.info("config       : %s", plan.config)

    policy = ConflictPolicy.ERROR
    # An explicit --rename-suffix selects the rename policy on its own, matching the pipeline.
    # Requiring both flags would let `--rename-suffix foo` silently refuse on a collision instead.
    if args.autosuffix_on_exists or args.rename_suffix:
        policy = ConflictPolicy.RENAME
    elif args.overwrite_existing:
        policy = ConflictPolicy.OVERWRITE

    db_url = args.local_db_url
    effective_set_id = plan.source_set_id
    if db_url:
        # PRE-FLIGHT ONLY, and read-only: refuse before spending a multi-GiB download on an import
        # that would be rejected. The pipeline repeats this check authoritatively at import time.
        try:
            occupancy = describe_source_set(db_url, model_id=plan.model_id, source_set_id=plan.source_set_id)
        except Exception as exc:  # an unreachable DB is the import's problem to report, not this one
            log.warning("could not pre-check the target source set (%s); continuing", exc)
            occupancy = None
        if occupancy is not None and occupancy.occupied:
            if policy is ConflictPolicy.ERROR:
                print(f"\nREFUSING TO IMPORT:\n{render_conflict_report(occupancy)}", file=sys.stderr)
                return 3
            if policy is ConflictPolicy.RENAME:
                effective_set_id = suffix_source_set_id(plan.source_set_id, args.rename_suffix or generate_autosuffix())
                log.info(
                    "conflict     : existing %r kept; importing as %r%s",
                    plan.source_set_id,
                    effective_set_id,
                    "" if args.rename_suffix else " (the pipeline regenerates its own stamp)",
                )
            elif occupancy.explanation_count and not args.allow_explanation_loss:
                # The pipeline would refuse for this reason anyway; catching it here is the whole
                # point of a pre-flight, since otherwise the refusal lands after a multi-GiB download.
                print(
                    f"\nREFUSING TO IMPORT:\n--overwrite-existing would delete "
                    f"{occupancy.neuron_count} neurons from {plan.source_set_id!r}, cascading away "
                    f"{occupancy.explanation_count} explanations. Activations can be regenerated "
                    "from a corpus; explanations cannot. Pass --allow-explanation-loss to proceed, "
                    "or --autosuffix-on-exists to keep both.",
                    file=sys.stderr,
                )
                return 3
            else:
                log.info("conflict     : %r will be REPLACED (%s neurons)", plan.source_set_id, occupancy.neuron_count)
    elif policy is ConflictPolicy.RENAME:
        effective_set_id = suffix_source_set_id(plan.source_set_id, args.rename_suffix or generate_autosuffix())

    log.info("destination  : %s", plan.run_dir)
    command = build_import_command(
        plan,
        db_url=db_url,
        policy=policy,
        rename_suffix=args.rename_suffix,
        allow_explanation_loss=args.allow_explanation_loss,
    )

    if args.dry_run:
        log.info("\nDRY RUN -- nothing downloaded, nothing imported.")
        log.info("would import with:\n  %s", " ".join(command))
        return 0

    if not args.skip_download:
        plan.run_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info("\ndownloading %s ...", args.bucket)
        download_dashboard_run(args.bucket, plan.run_dir, store="bucket", token=args.token)
        log.info("downloaded to %s", plan.run_dir)

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
        log.info("\n%s", _summarize(db_url, model_id=plan.model_id, source_set_id=effective_set_id))
    else:
        log.info("\n(no --local-db-url given; skipping the post-import summary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
