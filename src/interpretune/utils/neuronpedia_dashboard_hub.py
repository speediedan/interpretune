"""Publish a generated columnar dashboard run to a Hugging Face dataset repo.

Why the Hub rather than a file share: the artifacts are already Parquet, so a Hub dataset tree loads
with ``load_dataset(...)`` and streams row groups over HTTP with no bespoke reader. That streaming
path is the premise of the shareable-dashboards direction; a zip on a drive cannot do it.

WHAT IS EXCLUDED BY DEFAULT
---------------------------
``activation_copy_rows.parquet`` is a pre-flattened duplicate of ``activation_rows.parquet``, carried
so the Postgres binary-COPY import lane has nothing left to transform. It is derived data: a consumer
who is not importing into Postgres does not need it, and one who is can flatten locally. Measured on
the reference 16k run, excluding it drops 47% of the payload with no information lost.

Run logs are always excluded -- they carry absolute host paths, pids and timings, which are noise to a
consumer and a small leak of local layout.

STREAMING READINESS (why :func:`page_index_coverage` exists)
------------------------------------------------------------
Parquet files written without a page index cannot be range-read at page granularity, so a consumer
streaming one row group still pays to fetch whole row groups -- which removes the reason for choosing
the Hub over a file share. ``write_page_index`` costs ~0.01% in size but only at GENERATION time; it
cannot be retrofitted without rewriting every file, i.e. regenerating and re-uploading the corpus.

:func:`publish_dashboard_run` therefore REFUSES by default to upload a corpus lacking a page index.
That is deliberate: the failure mode it prevents is discovering the gap after a multi-GiB upload,
when the only remedy is to regenerate and upload again. Pass ``require_page_index=False`` to override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "COPY_ROWS_STEM",
    "EXCLUDED_SUFFIXES",
    "DashboardPublishPlan",
    "PageIndexCoverage",
    "MissingPageIndexError",
    "collect_publish_files",
    "summarize_by_table",
    "page_index_coverage",
    "build_publish_plan",
    "format_publish_plan",
    "publish_ignore_patterns",
    "publish_dashboard_run",
]

log = logging.getLogger(__name__)

#: Derived duplicate of ``activation_rows``; only the Postgres COPY import lane needs it.
COPY_ROWS_STEM = "activation_copy_rows"

#: Always excluded. Run logs carry absolute host paths, pids and timings.
EXCLUDED_SUFFIXES = frozenset({".log"})

#: Parquet files inspected when estimating page-index coverage. The corpus is homogeneous -- every
#: file comes from the same writer configuration -- so a sample answers the question that matters
#: ("was write_page_index on for this run?") without opening ~900 footers.
DEFAULT_PAGE_INDEX_SAMPLE = 12

_GIB = 1024**3


class MissingPageIndexError(RuntimeError):
    """Raised when a corpus without a Parquet page index would be uploaded.

    Not a warning, because the remedy after the fact is to regenerate the corpus and upload it again.
    """


@dataclass(frozen=True)
class PageIndexCoverage:
    """How many sampled Parquet files carry a page index."""

    with_index: int
    checked: int

    @property
    def complete(self) -> bool:
        """True when every sampled file carries a page index."""
        return self.checked > 0 and self.with_index == self.checked

    @property
    def absent(self) -> bool:
        """True when files were checked and none carried a page index."""
        return self.checked > 0 and self.with_index == 0


@dataclass(frozen=True)
class DashboardPublishPlan:
    """What a publish would upload, and what it would leave behind."""

    run_root: Path
    upload: list[Path]
    skip: list[Path]
    page_index: PageIndexCoverage
    by_table: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def upload_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.upload)

    @property
    def skip_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.skip)

    @property
    def copy_rows_skipped(self) -> list[Path]:
        return [p for p in self.skip if p.stem == COPY_ROWS_STEM]

    @property
    def largest(self) -> Path | None:
        return max(self.upload, key=lambda p: p.stat().st_size) if self.upload else None


def collect_publish_files(run_root: Path, include_copy_rows: bool = False) -> tuple[list[Path], list[Path]]:
    """Split a run's files into ``(upload, skip)``.

    Hidden files are ignored entirely: they are neither uploaded nor reported as a deliberate
    exclusion, since they are editor/OS droppings rather than run output.
    """
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


def summarize_by_table(paths: list[Path]) -> dict[str, tuple[int, int]]:
    """Aggregate ``(bytes, file_count)`` per table.

    Parquet files group by stem (``activation_rows`` across every layer/batch); everything else
    groups by full name, so ``manifest.json`` and ``tokens_4096.pt`` stay legible.
    """
    agg: dict[str, list[int]] = {}
    for p in paths:
        key = p.stem if p.suffix == ".parquet" else p.name
        entry = agg.setdefault(key, [0, 0])
        entry[0] += p.stat().st_size
        entry[1] += 1
    return {k: (v[0], v[1]) for k, v in agg.items()}


def page_index_coverage(paths: list[Path], sample: int = DEFAULT_PAGE_INDEX_SAMPLE) -> PageIndexCoverage:
    """Report how many of the first ``sample`` Parquet files carry a page index.

    Returns ``checked == 0`` when pyarrow is unavailable or no Parquet files are present, which
    callers must treat as "unknown" rather than "absent" -- refusing an upload on a failed import
    would be a false positive.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow ships with datasets
        log.debug("pyarrow unavailable; skipping page-index inspection")
        return PageIndexCoverage(0, 0)

    parquet = [p for p in paths if p.suffix == ".parquet"][:sample]
    with_index = 0
    for p in parquet:
        try:
            column = pq.ParquetFile(p).metadata.row_group(0).column(0)
            if getattr(column, "has_offset_index", False):
                with_index += 1
        except Exception as exc:  # an unreadable footer must not abort a dry run
            log.debug("could not read parquet footer for %s: %s", p, exc)
    return PageIndexCoverage(with_index, len(parquet))


def build_publish_plan(
    run_root: Path,
    include_copy_rows: bool = False,
    sample: int = DEFAULT_PAGE_INDEX_SAMPLE,
) -> DashboardPublishPlan:
    """Inspect ``run_root`` and describe what publishing it would do.

    Makes no network calls.
    """
    run_root = Path(run_root).expanduser().resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(f"run root is not a directory: {run_root}")
    upload, skip = collect_publish_files(run_root, include_copy_rows)
    if not upload:
        raise FileNotFoundError(f"no publishable files found under {run_root}")
    return DashboardPublishPlan(
        run_root=run_root,
        upload=upload,
        skip=skip,
        page_index=page_index_coverage(upload, sample),
        by_table=summarize_by_table(upload),
    )


def format_publish_plan(plan: DashboardPublishPlan, repo_id: str = "", max_tables: int = 12) -> str:
    """Render a plan as the operator-facing report.

    Pure; no I/O beyond ``stat``.
    """
    total = plan.upload_bytes + plan.skip_bytes
    lines = [f"run root : {plan.run_root}"]
    if repo_id:
        lines.append(f"repo     : {repo_id}")
    lines.append(f"upload   : {len(plan.upload):>5} files  {plan.upload_bytes / _GIB:6.2f} GiB")

    if plan.skip:
        # Break exclusions down by reason -- a single total attributed to copy-rows would misstate
        # the saving now that logs are dropped too.
        copy_rows = plan.copy_rows_skipped
        parts = []
        if copy_rows:
            copy_bytes = sum(p.stat().st_size for p in copy_rows)
            pct = (copy_bytes / total * 100) if total else 0.0
            parts.append(f"{len(copy_rows)} {COPY_ROWS_STEM} = {pct:.0f}% of the run")
        if other := len(plan.skip) - len(copy_rows):
            parts.append(f"{other} log/other")
        # Only advertise the flag when it would actually change the outcome.
        hint = "  -- pass include_copy_rows to keep it" if copy_rows else ""
        lines.append(
            f"excluded : {len(plan.skip):>5} files  {plan.skip_bytes / _GIB:6.2f} GiB  ({'; '.join(parts)}){hint}"
        )

    lines.append("")
    lines.append("by table:")
    ranked = sorted(plan.by_table.items(), key=lambda kv: -kv[1][0])[:max_tables]
    for stem, (nbytes, count) in ranked:
        lines.append(f"  {stem:<28} {nbytes / _GIB:6.3f} GiB  ({count} files)")

    coverage = plan.page_index
    if coverage.checked:
        lines.append("")
        lines.append(f"page index: {coverage.with_index}/{coverage.checked} sampled parquet files")
        if coverage.absent:
            lines.append("  WARNING: no page index. Readers cannot range-read individual pages, so HTTP")
            lines.append("  streaming falls back to whole row groups. This CANNOT be fixed after upload")
            lines.append("  without regenerating the corpus -- enable write_page_index at generation time.")

    if (largest := plan.largest) is not None:
        size_mib = largest.stat().st_size / (1024**2)
        lines.append("")
        lines.append(f"largest file: {size_mib:.1f} MiB  {largest.relative_to(plan.run_root)}")
    return "\n".join(lines)


def publish_ignore_patterns(include_copy_rows: bool = False) -> list[str]:
    """The ``ignore_patterns`` handed to ``upload_folder``.

    Kept in lockstep with :func:`collect_publish_files` -- the plan a user reviews and the upload
    they get must exclude the same things, and the two are enforced by different mechanisms.

    The pattern shapes are load-bearing. ``huggingface_hub`` matches with :mod:`fnmatch` against
    repo-relative paths, where ``*`` spans ``/`` but a leading ``**/`` requires at least one
    separator. So ``**/*.log`` silently fails to match a log at the ROOT of the run -- exactly where
    a run log lands. Logs therefore use bare ``*.log``, and the copy-rows table lists both the
    root-level and nested forms rather than ``*activation_copy_rows.parquet``, which would
    over-match unrelated files with that suffix.
    """
    patterns = [f"*{suffix}" for suffix in sorted(EXCLUDED_SUFFIXES)]
    if not include_copy_rows:
        patterns += [f"{COPY_ROWS_STEM}.parquet", f"**/{COPY_ROWS_STEM}.parquet"]
    return patterns


def publish_dashboard_run(
    run_root: Path,
    repo_id: str,
    *,
    include_copy_rows: bool = False,
    private: bool = False,
    revision: str | None = None,
    path_in_repo: str = "",
    commit_message: str = "Publish columnar dashboard artifacts",
    require_page_index: bool = True,
    token: str | None = None,
) -> DashboardPublishPlan:
    """Upload a dashboard run to a Hub dataset repo. Returns the plan that was executed.

    Raises :class:`MissingPageIndexError` when the corpus carries no page index and
    ``require_page_index`` is set -- see the module docstring for why that is fatal rather than a
    warning. Unknown coverage (no pyarrow, no Parquet files) does not trip the check.
    """
    plan = build_publish_plan(run_root, include_copy_rows)

    if require_page_index and plan.page_index.absent:
        raise MissingPageIndexError(
            f"{plan.run_root} has no Parquet page index "
            f"({plan.page_index.with_index}/{plan.page_index.checked} sampled files). "
            "Uploading it would ship a corpus that cannot be range-read, and the fix requires "
            "regenerating and re-uploading. Enable write_page_index at generation time, or pass "
            "require_page_index=False to publish anyway."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(plan.run_root),
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=path_in_repo,
        ignore_patterns=publish_ignore_patterns(include_copy_rows),
        commit_message=commit_message,
    )
    return plan
