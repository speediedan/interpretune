#!/usr/bin/env python3
"""Publish a generated columnar dashboard run to a Hugging Face dataset repo.

Why the Hub rather than a file share: the artifacts are already Parquet, so a Hub dataset tree loads
with ``load_dataset(...)`` and streams row groups over HTTP with no bespoke reader. That streaming
path is the whole premise of the Wave 2 "shareable dashboards" direction; a zip on a drive cannot do
it. See ``columnar_formats_and_import_paths.md`` for the format analysis behind this.

WHAT IS EXCLUDED BY DEFAULT, and why it matters
-----------------------------------------------
``activation_copy_rows.parquet`` is a **pre-flattened duplicate** of ``activation_rows.parquet``,
carried so the Postgres binary-COPY import path has nothing left to transform. It is derived data: a
consumer who is not importing into Postgres does not need it, and one who is can flatten locally.

Measured on the reference 16k run (gemma-3-1b-it / gemmascope-2-transcoder-16k, 26 layers):
excluding it takes the upload from 6.30 GiB / 1014 files to 3.35 GiB / 910 files -- 47% of the
payload dropped with no information lost. Its twin ``activation_rows`` is 2.88 GiB of what remains,
so the two together are ~93% of the run.

Pass ``--include-copy-rows`` to publish it anyway (e.g. a revision aimed at local-DB importers).

STREAMING READINESS
-------------------
Files written without a Parquet page index cannot be range-read efficiently, so a consumer streaming
one row group still pays to fetch whole row groups. Enabling ``write_page_index`` at GENERATION time
is nearly free (~0.01% size) but cannot be retrofitted without rewriting the files -- i.e. without
regenerating and re-uploading the corpus. This script therefore REPORTS page-index coverage and warns
when it is absent, so an un-streamable upload is a deliberate choice rather than a surprise.

Usage:
    # inspect what would be uploaded (no network writes)
    python scripts/publish_dashboards_to_hub.py --run-root <dir> --repo-id <user>/<name> --dry-run

    # publish
    python scripts/publish_dashboards_to_hub.py --run-root <dir> --repo-id <user>/<name> --private
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Derived duplicate of activation_rows, only needed by the Postgres COPY import lane.
COPY_ROWS_STEM = "activation_copy_rows"
# Run logs carry absolute host paths, pids and timings -- noise for a consumer, and a small leak of
# local layout. Excluded always; the manifests already record what a reader needs.
EXCLUDED_SUFFIXES = {".log"}
GIB = 1024**3


def collect(run_root: Path, include_copy_rows: bool) -> tuple[list[Path], list[Path]]:
    """Split the run's files into (upload, skip)."""
    upload: list[Path] = []
    skip: list[Path] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            skip.append(path)
        elif not include_copy_rows and path.stem == COPY_ROWS_STEM:
            skip.append(path)
        else:
            upload.append(path)
    return upload, skip


def summarize(paths: list[Path]) -> dict[str, tuple[int, int]]:
    """Bytes and file count per table stem."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in paths:
        entry = agg[p.stem if p.suffix == ".parquet" else p.name]
        entry[0] += p.stat().st_size
        entry[1] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def page_index_coverage(paths: list[Path], sample: int = 12) -> tuple[int, int]:
    """How many sampled parquet files carry a page index?

    (with, checked)
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return (0, 0)
    parquet = [p for p in paths if p.suffix == ".parquet"][:sample]
    with_index = 0
    for p in parquet:
        try:
            col = pq.ParquetFile(p).metadata.row_group(0).column(0)
            if getattr(col, "has_offset_index", False):
                with_index += 1
        except Exception:  # unreadable footer should not abort a dry run
            continue
    return (with_index, len(parquet))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, help="Dashboard run directory (contains layer_*/).")
    ap.add_argument(
        "--repo-id", required=True, help="Target Hub dataset repo, e.g. speediedan/gemma-3-1b-it-dashboards."
    )
    ap.add_argument("--revision", default=None, help="Branch/revision to upload to (default: main).")
    ap.add_argument("--path-in-repo", default="", help="Subdirectory within the repo.")
    ap.add_argument("--include-copy-rows", action="store_true", help=f"Also upload {COPY_ROWS_STEM}.parquet files.")
    ap.add_argument("--private", action="store_true", help="Create the repo private if it does not exist.")
    ap.add_argument("--commit-message", default="Publish columnar dashboard artifacts")
    ap.add_argument("--dry-run", action="store_true", help="Report the plan; make no network writes.")
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.is_dir():
        print(f"run-root is not a directory: {run_root}", file=sys.stderr)
        return 1

    upload, skip = collect(run_root, args.include_copy_rows)
    if not upload:
        print(f"no files found under {run_root}", file=sys.stderr)
        return 1

    up_bytes = sum(p.stat().st_size for p in upload)
    skip_bytes = sum(p.stat().st_size for p in skip)

    print(f"run root : {run_root}")
    print(
        f"repo     : {args.repo_id}{'/' + args.path_in_repo if args.path_in_repo else ''}"
        f"{' @' + args.revision if args.revision else ''}"
    )
    print(f"upload   : {len(upload):>5} files  {up_bytes / GIB:6.2f} GiB")
    if skip:
        # Break the exclusions down by reason. A single total attributed to copy-rows would
        # misstate the saving now that logs are dropped too.
        copy_rows = [p for p in skip if p.stem == COPY_ROWS_STEM]
        copy_bytes = sum(p.stat().st_size for p in copy_rows)
        other = len(skip) - len(copy_rows)
        parts = []
        if copy_rows:
            parts.append(
                f"{len(copy_rows)} {COPY_ROWS_STEM} = {copy_bytes / (up_bytes + skip_bytes) * 100:.0f}% of the run"
            )
        if other:
            parts.append(f"{other} log/other")
        # Only advertise the flag when it would actually change the outcome.
        hint = "  -- pass --include-copy-rows to keep it" if copy_rows else ""
        print(f"excluded : {len(skip):>5} files  {skip_bytes / GIB:6.2f} GiB  ({'; '.join(parts)}){hint}")
    print("\nby table:")
    for stem, (nbytes, count) in sorted(summarize(upload).items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {stem:<28} {nbytes / GIB:6.3f} GiB  ({count} files)")

    with_idx, checked = page_index_coverage(upload)
    if checked:
        print(f"\npage index: {with_idx}/{checked} sampled parquet files")
        if with_idx == 0:
            print("  WARNING: no page index. Readers cannot range-read individual pages, so HTTP")
            print("  streaming falls back to whole row groups. This CANNOT be fixed after upload without")
            print("  regenerating the corpus -- enable write_page_index at generation time first.")

    largest = max(upload, key=lambda p: p.stat().st_size)
    print(f"\nlargest file: {largest.stat().st_size / (1024**2):.1f} MiB  {largest.relative_to(run_root)}")

    if args.dry_run:
        print("\nDRY RUN -- nothing uploaded.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    ignore = ["**/*.log"]
    if not args.include_copy_rows:
        ignore.append(f"**/{COPY_ROWS_STEM}.parquet")
    api.upload_folder(
        folder_path=str(run_root),
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        path_in_repo=args.path_in_repo,
        ignore_patterns=ignore,
        commit_message=args.commit_message,
    )
    print(f"\nuploaded to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
