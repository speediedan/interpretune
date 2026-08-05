#!/usr/bin/env python3
"""CLI wrapper for publishing a columnar dashboard run to a Hugging Face bucket or dataset repo.

All logic lives in :mod:`interpretune.utils.neuronpedia_dashboard_hub`; this file is argument
parsing and reporting only. See that module's docstring for what is excluded and why the page-index
check is fatal by default.

Usage:
    # inspect what would be uploaded (no network writes)
    python scripts/publish_dashboards_to_hub.py --run-root <dir> --repo-id <user>/<name> --dry-run

    # publish to a bucket (default backend) under an explicit prefix
    python scripts/publish_dashboards_to_hub.py --run-root <dir> --repo-id <user>/<name> \
        --revision 24576-prompts

    # publish to a versioned dataset repo instead
    python scripts/publish_dashboards_to_hub.py --run-root <dir> --repo-id <user>/<name> \
        --store dataset --private

Exit codes: 1 bad arguments/run root, 2 missing page index, 3 unspecified bucket prefix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from interpretune.utils.neuronpedia_dashboard_hub import (
    BUCKET_ROOT,
    COPY_ROWS_STEM,
    DASHBOARD_STORES,
    DASHBOARDS_TOKEN_ENV_VAR,
    DEFAULT_STORE,
    UNSPECIFIED,
    BucketPrefixUnspecifiedError,
    MissingPageIndexError,
    build_publish_plan,
    format_publish_plan,
    publish_dashboard_run,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, help="Dashboard run directory (contains layer_*/).")
    parser.add_argument("--repo-id", required=True, help="Target Hub repo/bucket, e.g. user/my-dashboards.")
    parser.add_argument(
        "--store",
        choices=sorted(DASHBOARD_STORES),
        default=DEFAULT_STORE,
        help=f"Transport backend (default: {DEFAULT_STORE}). 'bucket' is Xet-backed and S3-reachable; "
        "'dataset' is git-backed and the right choice when a corpus must be pinned.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset branch, or bucket path PREFIX. Buckets require this or --bucket-root: they "
        "have no revisions and no history, so an unstated prefix silently shares the root with "
        "every other corpus and a later sync overwrites in place.",
    )
    parser.add_argument(
        "--bucket-root",
        action="store_true",
        help="Publish at the bucket root deliberately (the explicit alternative to --revision).",
    )
    parser.add_argument("--path-in-repo", default="", help="Subdirectory within the repo.")
    parser.add_argument(
        "--include-copy-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Publish {COPY_ROWS_STEM}.parquet (default: on). Excluding it saves ~45% but the "
        "corpus becomes UNIMPORTABLE: the per-batch manifests declare the table and the importer "
        "raises when it is absent rather than falling back. Only exclude for consumers that read "
        "the parquet directly and never import.",
    )
    parser.add_argument(
        "--allow-manifest-inconsistency",
        action="store_true",
        help="Publish even when the manifests reference an excluded table. Refused by default.",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo private if it does not exist.")
    parser.add_argument("--commit-message", default="Publish columnar dashboard artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan; make no network writes.")
    parser.add_argument(
        "--token",
        default=None,
        help=f"Hub token. Defaults to ${DASHBOARDS_TOKEN_ENV_VAR} (process env, then the repo .env), "
        "then the ambient huggingface-cli credential.",
    )
    parser.add_argument(
        "--allow-missing-page-index",
        action="store_true",
        help="Publish even if the corpus has no Parquet page index. Refused by default because the "
        "only fix after upload is to regenerate the corpus and upload it again.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        plan = build_publish_plan(Path(args.run_root), args.include_copy_rows)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_publish_plan(plan, repo_id=args.repo_id))

    if args.dry_run:
        print("\nDRY RUN -- nothing uploaded.")
        return 0

    if args.bucket_root and args.revision is not None:
        print("--bucket-root and --revision are mutually exclusive.", file=sys.stderr)
        return 1
    # Leave `revision` UNSPECIFIED when the user named neither, so the library raises its own
    # explanation rather than this wrapper guessing which backend cares.
    revision = BUCKET_ROOT if args.bucket_root else (args.revision if args.revision is not None else UNSPECIFIED)

    try:
        publish_dashboard_run(
            plan.run_root,
            args.repo_id,
            include_copy_rows=args.include_copy_rows,
            private=args.private,
            revision=revision,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
            require_page_index=not args.allow_missing_page_index,
            require_manifest_consistency=not args.allow_manifest_inconsistency,
            token=args.token,
            store=args.store,
        )
    except MissingPageIndexError as exc:
        print(f"\nREFUSING TO PUBLISH: {exc}", file=sys.stderr)
        return 2
    except BucketPrefixUnspecifiedError as exc:
        print(f"\nREFUSING TO PUBLISH: {exc}", file=sys.stderr)
        return 3

    kind = "buckets" if args.store == "bucket" else "datasets"
    print(f"\nuploaded to https://huggingface.co/{kind}/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
