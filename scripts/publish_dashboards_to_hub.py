#!/usr/bin/env python3
"""CLI wrapper for publishing a columnar dashboard run to a Hugging Face dataset repo.

All logic lives in :mod:`interpretune.utils.neuronpedia_dashboard_hub`; this file is argument
parsing and reporting only. See that module's docstring for what is excluded and why the page-index
check is fatal by default.

Usage:
    # inspect what would be uploaded (no network writes)
    python scripts/publish_dashboards_to_hub.py --run-root <dir> --repo-id <user>/<name> --dry-run

    # publish
    python scripts/publish_dashboards_to_hub.py --run-root <dir> --repo-id <user>/<name> --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from interpretune.utils.neuronpedia_dashboard_hub import (
    COPY_ROWS_STEM,
    DASHBOARDS_TOKEN_ENV_VAR,
    MissingPageIndexError,
    build_publish_plan,
    format_publish_plan,
    publish_dashboard_run,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, help="Dashboard run directory (contains layer_*/).")
    parser.add_argument("--repo-id", required=True, help="Target Hub dataset repo, e.g. user/my-dashboards.")
    parser.add_argument("--revision", default=None, help="Branch/revision to upload to (default: main).")
    parser.add_argument("--path-in-repo", default="", help="Subdirectory within the repo.")
    parser.add_argument("--include-copy-rows", action="store_true", help=f"Also upload {COPY_ROWS_STEM}.parquet.")
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

    try:
        publish_dashboard_run(
            plan.run_root,
            args.repo_id,
            include_copy_rows=args.include_copy_rows,
            private=args.private,
            revision=args.revision,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
            require_page_index=not args.allow_missing_page_index,
            token=args.token,
        )
    except MissingPageIndexError as exc:
        print(f"\nREFUSING TO PUBLISH: {exc}", file=sys.stderr)
        return 2

    print(f"\nuploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
