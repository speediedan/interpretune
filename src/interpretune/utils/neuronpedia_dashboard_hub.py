"""Publish a generated columnar dashboard run to a Hugging Face dataset repo.

Why the Hub rather than a file share: the artifacts are already Parquet, so a Hub dataset tree loads
with ``load_dataset(...)`` and streams row groups over HTTP with no bespoke reader. That streaming
path is the premise of the shareable-dashboards direction; a zip on a drive cannot do it.

COPY-ROWS ARE PUBLISHED BY DEFAULT, AND THAT IS NOT NEGOTIABLE FOR IMPORTABLE CORPORA
------------------------------------------------------------------------------------
``activation_copy_rows.parquet`` is a pre-flattened duplicate of ``activation_rows.parquet``, carried
so the Postgres binary-COPY import lane has nothing left to transform. It looks like derived data a
consumer could regenerate, and excluding it saves ~45% of the payload.

**Excluding it produces a corpus that cannot be imported at all.** Each
``feature_batch_*/manifest.json`` declares the table, and the importer resolves the activation table
from that manifest and RAISES when the preferred entry is absent -- it does not fall back to
``activation_rows``. Verified the hard way: a corpus published without it downloaded byte-identically
(910/910 files, sha256 clean) and then failed on layer 0 of the first import.

So the default is to include it, and :func:`publish_dashboard_run` refuses to publish a corpus whose
manifests reference a table the publish would drop. Exclusion remains available for a consumer who
reads the parquet directly (DuckDB, pandas) and never imports -- pass
``include_copy_rows=False, require_manifest_consistency=False`` to say so deliberately.

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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "COPY_ROWS_STEM",
    "EXCLUDED_SUFFIXES",
    "DashboardPublishPlan",
    "PageIndexCoverage",
    "MissingPageIndexError",
    "ManifestReferencesExcludedTableError",
    "BucketPrefixUnspecifiedError",
    "BUCKET_ROOT",
    "UNSPECIFIED",
    "CORPUS_MANIFEST_FILE",
    "corpus_manifest_summary",
    "manifest_referenced_tables",
    "collect_publish_files",
    "summarize_by_table",
    "page_index_coverage",
    "build_publish_plan",
    "format_publish_plan",
    "publish_ignore_patterns",
    "publish_dashboard_run",
    "resolve_dashboards_token",
    "DASHBOARDS_TOKEN_ENV_VAR",
    "DashboardStore",
    "HubDatasetStore",
    "HubBucketStore",
    "DASHBOARD_STORES",
    "DEFAULT_STORE",
    "get_dashboard_store",
    "download_dashboard_run",
    "TruncatedDownloadError",
]

log = logging.getLogger(__name__)

#: Derived duplicate of ``activation_rows``; only the Postgres COPY import lane needs it.
COPY_ROWS_STEM = "activation_copy_rows"

#: Always excluded. Run logs carry absolute host paths, pids and timings.
EXCLUDED_SUFFIXES = frozenset({".log"})

#: Token for the dashboards dataset/bucket, kept separate from ``HF_TOKEN`` on purpose: it is scoped
#: to just these artifacts, and reusing the ambient token name would silently widen or narrow every
#: other Hub operation in the process depending on which won.
DASHBOARDS_TOKEN_ENV_VAR = "HF_HUB_DASHBOARDS_TOKEN"

#: Corpus-level self-description written by the pipeline at the run root. Duplicated rather than
#: imported: pulling it from the pipeline module would drag sae_dashboard and torch into what is
#: otherwise a lightweight publisher. ``TestCorpusManifest`` asserts the two constants agree, so
#: drift is a test failure rather than a silently-missing file in a published corpus.
CORPUS_MANIFEST_FILE = "dashboards.json"

#: Parquet files inspected when estimating page-index coverage. The corpus is homogeneous -- every
#: file comes from the same writer configuration -- so a sample answers the question that matters
#: ("was write_page_index on for this run?") without opening ~900 footers.
DEFAULT_PAGE_INDEX_SAMPLE = 12

_GIB = 1024**3


class ManifestReferencesExcludedTableError(RuntimeError):
    """Raised when an excluded table is still referenced by the corpus's own manifests.

    Not a warning. The importer resolves the activation table from ``manifest.json`` and RAISES when
    the preferred entry is absent -- it does not fall back -- so such a corpus cannot be imported at
    all. Publishing it produces something that downloads cleanly and then fails on first use.
    """


class MissingPageIndexError(RuntimeError):
    """Raised when a corpus without a Parquet page index would be uploaded.

    Not a warning, because the remedy after the fact is to regenerate the corpus and upload it again.
    """


class TruncatedDownloadError(RuntimeError):
    """Raised when a downloaded corpus file does not match the size the bucket reports.

    The cause seen in practice is ``sync_bucket`` writing into an existing local file without
    truncating it, so a re-sync over a LONGER previous version leaves stale trailing bytes. For
    Parquet that is silent: the footer lives at the end of the file, so the file still opens as
    "a parquet file" and fails only when its metadata is deserialized.

    Raised rather than repaired because the safe repair -- delete the destination and re-download --
    throws away a multi-GiB download the caller may not want redone implicitly.
    """


class BucketPrefixUnspecifiedError(RuntimeError):
    """Raised when a bucket publish does not say where in the bucket it should land.

    Buckets are mutable and unversioned, so two corpora that differ only in a dimension absent from
    the bucket NAME -- prompt count is the obvious one -- silently interleave at the root, and
    ``sync`` overwrites in place with no history to recover from. A dataset repo would express the
    difference as a revision; a bucket can only express it as a path prefix.

    So the prefix is not defaulted. Pass one, or pass :data:`BUCKET_ROOT` to say the root is
    genuinely intended.
    """


class _Unspecified:
    """Distinguishes "the caller expressed no opinion" from "the caller chose the root"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSPECIFIED"


#: Sentinel default for ``revision``; see :class:`BucketPrefixUnspecifiedError`.
UNSPECIFIED: Any = _Unspecified()

#: Explicit "publish at the bucket root". Named so the intent is legible at the call site and in the
#: shell -- an empty string passed positionally reads like an oversight, which is what this replaces.
BUCKET_ROOT = ""


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


def resolve_dashboards_token(env_file: Path | None = None) -> str | None:
    """Find the dashboards-scoped Hub token, or None to fall back to the ambient credential.

    Resolution order: the process environment, then ``env_file`` (default: the repo ``.env``). The
    environment wins so a one-off override does not require editing a file, and returning None
    rather than raising keeps the ambient ``huggingface-cli login`` token working for anyone who has
    not adopted the scoped one.
    """
    token = os.environ.get(DASHBOARDS_TOKEN_ENV_VAR)
    if token:
        return token.strip() or None

    if env_file is None:
        candidate = Path(__file__).resolve().parents[3] / ".env"
        env_file = candidate if candidate.exists() else None
    if env_file is None or not Path(env_file).exists():
        return None
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        log.debug("python-dotenv unavailable; cannot read %s", env_file)
        return None
    value = dotenv_values(Path(env_file)).get(DASHBOARDS_TOKEN_ENV_VAR)
    return value.strip() or None if value else None


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


def manifest_referenced_tables(run_root: Path, sample: int = 4) -> set[str]:
    """Table stems the corpus's own ``feature_batch_*/manifest.json`` files declare.

    Sampled: manifests are written by one code path per run, so a few are representative, and
    opening all ~200 to learn a fixed schema is wasted work.
    """
    referenced: set[str] = set()
    for i, manifest in enumerate(sorted(run_root.rglob("feature_batch_*/manifest.json"))):
        if i >= sample:
            break
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("unreadable manifest %s: %s", manifest, exc)
            continue
        referenced |= set(payload.get("tables", {}))
    return referenced


def corpus_manifest_summary(run_root: Path) -> str:
    """One line describing what the corpus says it is, for the pre-publish report.

    Returns a warning string when the corpus describes nothing. That is not fatal -- nothing imports from the manifest
    -- but it is worth seeing at the moment of publishing, since a corpus that reaches a consumer without it forces them
    to infer provenance from the bucket name.
    """
    path = Path(run_root) / CORPUS_MANIFEST_FILE
    if not path.exists():
        return f"corpus manifest: ABSENT ({CORPUS_MANIFEST_FILE}) -- backfill with --write-source-ids"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"corpus manifest: UNREADABLE ({exc})"
    model = (payload.get("model") or {}).get("name", "?")
    source_set = (payload.get("source_set") or {}).get("id", "?")
    corpus = payload.get("prompt_corpus") or {}
    layers = (payload.get("layers") or {}).get("generated") or []
    return (
        f"corpus manifest: {model} / {source_set} / "
        f"{corpus.get('n_prompts', '?')} prompts x {corpus.get('n_tokens_in_prompt', '?')} tokens / "
        f"{len(layers)} layers"
    )


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
    lines.append(corpus_manifest_summary(plan.run_root))
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


class DashboardStore(Protocol):
    """Transport for a dashboard corpus.

    Everything that makes this module worth having -- the plan builder, the copy-rows exclusion, the page-index refusal
    -- operates on LOCAL files and is transport-agnostic. Only these three operations differ between a Hub dataset repo
    and a Storage Bucket, which is why the seam sits here and nowhere else.
    """

    name: str

    #: Whether a publish must state WHERE in the target it lands. True only where the target cannot
    #: express the distinction any other way -- see :class:`BucketPrefixUnspecifiedError`.
    requires_explicit_prefix: bool

    def ensure(self, target: str, *, private: bool, token: str | None) -> None:
        """Create the destination if it does not exist."""
        ...

    def push(self, plan: DashboardPublishPlan, target: str, **kwargs: Any) -> None:
        """Upload the plan's files, honoring the same exclusions the plan reported."""
        ...

    def pull(self, target: str, dest: Path, **kwargs: Any) -> Path:
        """Download the corpus to ``dest``."""
        ...


@dataclass(frozen=True)
class HubDatasetStore:
    """Git-backed dataset repo: versioned, branchable, collection-eligible, pinnable.

    Keeps revisions, which is why it stays supported even once buckets are the default: a bucket
    cannot express "the corpus PR #217 referred to", and `bucket -> dataset` migration does not exist
    yet, so dropping this backend would be one-way.
    """

    name: str = "dataset"
    #: Revisions are versioned and default to a branch, so omitting one destroys nothing.
    requires_explicit_prefix: bool = False

    def ensure(self, target: str, *, private: bool = False, token: str | None = None) -> None:
        from huggingface_hub import HfApi

        HfApi(token=token).create_repo(repo_id=target, repo_type="dataset", private=private, exist_ok=True)

    def push(
        self,
        plan: DashboardPublishPlan,
        target: str,
        *,
        include_copy_rows: bool = False,
        revision: str | None = None,
        path_in_repo: str = "",
        commit_message: str = "Publish columnar dashboard artifacts",
        token: str | None = None,
    ) -> None:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_folder(
            folder_path=str(plan.run_root),
            repo_id=target,
            repo_type="dataset",
            revision=revision,
            path_in_repo=path_in_repo,
            ignore_patterns=publish_ignore_patterns(include_copy_rows),
            commit_message=commit_message,
        )

    def pull(self, target: str, dest: Path, *, revision: str | None = None, token: str | None = None) -> Path:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(repo_id=target, repo_type="dataset", revision=revision, local_dir=str(dest), token=token)
        )


@dataclass(frozen=True)
class HubBucketStore:
    """Xet-backed Storage Bucket: mutable, S3-reachable, no versioning.

    Preferred for consumers whose tooling is S3-shaped -- the ``s3.hf.co`` gateway serves boto3,
    rclone, s5cmd and DuckDB ``read_parquet('s3://...')``, range reads included, which is what the
    Parquet page index exists for. Dataset repos get none of that; the S3 gateway is bucket-only.

    Two differences the caller must absorb rather than ignore:

    - **No revisions.** ``revision`` becomes a path PREFIX. That loses atomic revision semantics, so
      provenance has to be recorded in-tree (the Hub renders a per-directory ``README.md``).
    - **Deletions are immediate and permanent.** ``sync`` is therefore additive here; pass
      ``delete=True`` explicitly, and only when mirroring is actually what you want.
    """

    name: str = "bucket"
    #: No revisions and no history: an unstated prefix means corpora silently share the root.
    requires_explicit_prefix: bool = True

    @staticmethod
    def uri(target: str, prefix: str = "") -> str:
        """``hf://buckets/<namespace>/<bucket>[/<prefix>]``."""
        base = f"hf://buckets/{target.strip('/')}"
        prefix = prefix.strip("/")
        return f"{base}/{prefix}" if prefix else base

    def ensure(self, target: str, *, private: bool = False, token: str | None = None) -> None:
        from huggingface_hub import HfApi

        HfApi(token=token).create_bucket(target, private=private, exist_ok=True)

    def push(
        self,
        plan: DashboardPublishPlan,
        target: str,
        *,
        include_copy_rows: bool = False,
        revision: str | None = None,
        path_in_repo: str = "",
        commit_message: str = "",  # buckets have no commits; accepted for interface parity
        token: str | None = None,
        delete: bool = False,
    ) -> None:
        from huggingface_hub import HfApi

        prefix = "/".join(p for p in (revision or "", path_in_repo or "") if p)
        HfApi(token=token).sync_bucket(
            source=str(plan.run_root),
            dest=self.uri(target, prefix),
            exclude=publish_ignore_patterns(include_copy_rows),
            delete=delete,
        )

    def pull(
        self,
        target: str,
        dest: Path,
        *,
        revision: str | None = None,
        token: str | None = None,
        path_in_repo: str = "",
        verify_sizes: bool = True,
    ) -> Path:
        """Download the corpus, then verify every file is exactly the size the bucket reports.

        The verification is not paranoia. ``sync_bucket`` writes into an existing local file WITHOUT
        TRUNCATING it, so re-syncing over an older corpus leaves the tail of the previous, longer
        file in place. Parquet reads its footer from the END of the file, so the result is a file
        that still starts with ``PAR1``, still ends with ``PAR1``, and fails to open with
        ``Couldn't deserialize thrift: TProtocolException: Invalid data``.

        Observed 2026-08-07 re-fetching the regenerated monology corpus over the previous one:
        ``feature_statistics.parquet`` arrived as 78,404 bytes -- the correct 78,395 bytes followed
        by 9 stale bytes from the file it replaced. Only files that SHRANK are affected, which is
        what makes it insidious: the big activation artifacts grew and were fine, so 1,010 of 1,016
        files imported before the run died on a small metadata table.

        A size check catches exactly this class, costs one listing call, and turns a corrupt corpus
        into a loud failure. It does not catch same-length corruption; nothing short of hashing
        would, and the bucket does not expose per-file hashes cheaply.
        """
        from huggingface_hub import HfApi

        prefix = "/".join(p for p in (revision or "", path_in_repo or "") if p)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        api = HfApi(token=token)
        api.sync_bucket(source=self.uri(target, prefix), dest=str(dest))
        if verify_sizes:
            _verify_bucket_download_sizes(api, target, prefix, dest)
        return dest


def _verify_bucket_download_sizes(api: Any, target: str, prefix: str, dest: Path) -> None:
    """Compare every downloaded file against the size the bucket reports.

    Cheap (one recursive listing) and catches the truncation class described in
    :meth:`HubBucketStore.pull`. Missing files are reported too: a partial sync is as unimportable as
    a corrupt one, and both are worth failing on before the importer spends an hour discovering it.
    """
    try:
        entries = list(api.list_bucket_tree(target, path=prefix or None, recursive=True))
    except Exception as exc:  # listing is best-effort; never fail a good download over a flaky list
        log.warning("could not verify downloaded file sizes for %s: %s", target, exc)
        return

    mismatches: list[str] = []
    missing: list[str] = []
    for entry in entries:
        size = getattr(entry, "size", None)
        path = getattr(entry, "path", None)
        if size is None or path is None:  # directories and anything unexpected
            continue
        relative = path[len(prefix) :].lstrip("/") if prefix and path.startswith(prefix) else path
        local = dest / relative
        if not local.is_file():
            missing.append(relative)
        elif local.stat().st_size != size:
            mismatches.append(f"{relative}: local {local.stat().st_size} B, bucket {size} B")

    if not (mismatches or missing):
        return
    detail = "\n".join([f"  SIZE MISMATCH {m}" for m in mismatches[:10]] + [f"  MISSING {m}" for m in missing[:10]])
    more = len(mismatches) + len(missing) - min(len(mismatches), 10) - min(len(missing), 10)
    raise TruncatedDownloadError(
        f"{len(mismatches)} file(s) differ in size and {len(missing)} are missing after syncing "
        f"{target!r} to {dest}.\n{detail}"
        + (f"\n  ... and {more} more" if more > 0 else "")
        + "\n\nA size mismatch is usually a re-sync over an older corpus: sync_bucket writes into "
        "the existing file without truncating, so a shorter new file keeps the old tail and its "
        "Parquet footer becomes unreadable. Delete the destination and download again."
    )


#: Backends by name.
DASHBOARD_STORES: dict[str, Any] = {"dataset": HubDatasetStore(), "bucket": HubBucketStore()}

#: Default backend. ``bucket`` since 2026-08-03, after an end-to-end round trip established that a
#: corpus published to a bucket, downloaded, and imported yields a database byte-identical to the
#: locally-imported one (1014/1014 files, matching per-(table, layer) content hashes).
#:
#: Buckets are the better default for this artifact because the consumers are S3-shaped: the
#: ``s3.hf.co`` gateway serves boto3, rclone, s5cmd and DuckDB ``read_parquet('s3://...')`` with
#: range reads, which is what the Parquet page index exists for, and dataset repos expose none of it.
#: ``dataset`` remains fully supported and is the right choice when a corpus must be PINNED, since
#: buckets are mutable and unversioned.
DEFAULT_STORE = "bucket"


def get_dashboard_store(name: str | None = None) -> Any:
    """Look up a backend by name, failing loudly on an unknown one."""
    key = (name or DEFAULT_STORE).lower()
    if key not in DASHBOARD_STORES:
        raise ValueError(f"unknown dashboard store {name!r}; expected one of {sorted(DASHBOARD_STORES)}")
    return DASHBOARD_STORES[key]


def publish_dashboard_run(
    run_root: Path,
    repo_id: str,
    *,
    include_copy_rows: bool = True,
    private: bool = False,
    revision: str | None = UNSPECIFIED,
    path_in_repo: str = "",
    commit_message: str = "Publish columnar dashboard artifacts",
    require_page_index: bool = True,
    require_manifest_consistency: bool = True,
    token: str | None = None,
    store: str | None = None,
) -> DashboardPublishPlan:
    """Upload a dashboard run to the Hub. Returns the plan that was executed.

    ``store`` selects the transport (``"dataset"`` or ``"bucket"``); see :data:`DASHBOARD_STORES`.

    ``revision`` is a branch for dataset repos and a path PREFIX for buckets. Buckets require it to
    be stated -- pass :data:`BUCKET_ROOT` for the root -- because they have neither revisions nor
    history; see :class:`BucketPrefixUnspecifiedError`.

    Raises :class:`MissingPageIndexError` when the corpus carries no page index and
    ``require_page_index`` is set -- see the module docstring for why that is fatal rather than a
    warning. Unknown coverage (no pyarrow, no Parquet files) does not trip the check. The check runs
    BEFORE any network call, and is backend-independent: an unstreamable corpus is unstreamable
    wherever it lands.
    """
    plan = build_publish_plan(run_root, include_copy_rows)
    token = token or resolve_dashboards_token()
    backend = get_dashboard_store(store)

    if require_page_index and plan.page_index.absent:
        raise MissingPageIndexError(
            f"{plan.run_root} has no Parquet page index "
            f"({plan.page_index.with_index}/{plan.page_index.checked} sampled files). "
            "Uploading it would ship a corpus that cannot be range-read, and the fix requires "
            "regenerating and re-uploading. Enable write_page_index at generation time, or pass "
            "require_page_index=False to publish anyway."
        )

    if require_manifest_consistency:
        excluded_stems = {p.stem for p in plan.skip if p.suffix == ".parquet"}
        orphaned = excluded_stems & manifest_referenced_tables(plan.run_root)
        if orphaned:
            raise ManifestReferencesExcludedTableError(
                f"{plan.run_root} manifests reference {sorted(orphaned)}, which this publish would "
                "exclude. The importer resolves the activation table from manifest.json and RAISES "
                "when the preferred entry is missing -- it does not fall back -- so the published "
                "corpus would download cleanly and then fail on first import. Publish with "
                "include_copy_rows=True, or pass require_manifest_consistency=False if the consumer "
                "genuinely only reads parquet directly (e.g. DuckDB) and never imports."
            )

    # Destination check LAST among the guards: the corpus checks above report problems whose remedy
    # is regenerating gigabytes, and hearing about a missing flag first would hide that behind a
    # trivial fix and a second attempt.
    if revision is UNSPECIFIED:
        if getattr(backend, "requires_explicit_prefix", False):
            raise BucketPrefixUnspecifiedError(
                f"publishing to {repo_id!r} via the {backend.name!r} store needs an explicit destination. "
                "Buckets have no revisions and no history, so an unstated prefix puts every corpus at "
                "the root, where a later sync overwrites in place with nothing to roll back to. Pass "
                "revision='<prefix>' (e.g. the prompt count, which the bucket name does not carry), or "
                "revision=BUCKET_ROOT to publish at the root deliberately."
            )
        revision = None

    backend.ensure(repo_id, private=private, token=token)
    backend.push(
        plan,
        repo_id,
        include_copy_rows=include_copy_rows,
        revision=revision,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        token=token,
    )
    return plan


def download_dashboard_run(
    repo_id: str,
    dest: Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    store: str | None = None,
    **kwargs: Any,
) -> Path:
    """Download a published corpus. Counterpart to :func:`publish_dashboard_run`.

    ``dest`` may be named anything. This was NOT always true: source ids were once derived from the
    run directory name, so unpacking under a different name imported cleanly into a separate set of
    ids -- a failure indistinguishable from success. Since 2026-08-03 identity travels inside the
    corpus in ``source_ids.json``, and directory-name derivation is gone.

    A corpus published before that sidecar existed still carries no identity. Backfill it with the
    pipeline's ``--write-source-ids`` rather than relying on where it happens to sit.

    ``revision`` defaults to the root here, unlike :func:`publish_dashboard_run`: reading the wrong
    prefix yields a missing or partial corpus that fails loudly and destroys nothing, so demanding an
    explicit answer would be friction without a safety gain.
    """
    backend = get_dashboard_store(store)
    return backend.pull(repo_id, Path(dest), revision=revision, token=token or resolve_dashboards_token(), **kwargs)
