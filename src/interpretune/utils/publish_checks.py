"""Checks a publish runs on what it is about to upload, and on what it uploaded.

Dashboard corpora and AnalysisStore artifacts both reach the Hub as Parquet, and the properties that make them useful
there (range reads, an importable corpus, a destination that cannot interleave with another one) are lost at publish
time or not at all. Each property is therefore a named check with one interface, rather than a keyword and a dedicated
exception per guard.

- **A check that cannot measure its property does not pass.** ``CheckResult.passed`` is ``None`` when the check could
  not tell (no pyarrow, no Parquet to read), and only ``True`` lets a publish proceed.
- **Callers skip a check by name** (``skip_checks={"page-index"}``) and turn on one that is off by default the same way
  (``add_checks={"range-read"}``). An unknown name is refused: silently running the check anyway is safe in the wrong
  way, since the caller believes they skipped it.
- **Every failure is reported at once.** One :class:`PublishCheckError` lists them all, in registration order, so a
  corpus problem (whose remedy is regenerating gigabytes) is never hidden behind a trivial destination fix.
- **Post-upload checks verify the artifact as published.** Their failure means the artifact is already live, so their
  error names the remediation rather than reading like a pre-publish refusal.

Checks register themselves with :func:`register_publish_check`; the dashboard checks live beside the dashboard
publisher, and :class:`RangeReadCheck`, which every Parquet artifact shares, lives here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

__all__ = [
    "ANALYSIS_STORE",
    "DASHBOARD_CORPUS",
    "PRE_UPLOAD",
    "POST_UPLOAD",
    "CheckResult",
    "PublishCheck",
    "PublishCheckError",
    "PublishCheckReport",
    "PublishedArtifactCheckError",
    "PublishTarget",
    "RangeReadCheck",
    "UnknownPublishCheckError",
    "publish_checks_for",
    "register_publish_check",
    "run_publish_checks",
    "validate_check_names",
]

log = logging.getLogger(__name__)

#: Target kinds.
DASHBOARD_CORPUS = "dashboard-corpus"
ANALYSIS_STORE = "analysis-store"

#: Stages. A pre-upload check reads local files; a post-upload check reads the published copy.
PRE_UPLOAD = "pre-upload"
POST_UPLOAD = "post-upload"


@dataclass(frozen=True)
class PublishTarget:
    """What a publish is about to upload, or has uploaded, as the checks see it.

    ``prefix`` is the destination prefix or revision exactly as the caller stated it, including the dashboard
    publisher's ``UNSPECIFIED`` sentinel, so a check can tell "not stated" from "the root". ``remote_files`` is filled
    in for post-upload checks: the ``hf://`` paths of the published Parquet files.
    """

    kind: str
    destination: str
    store: str
    root: Path | None = None
    files: tuple[Path, ...] = ()
    excluded: tuple[Path, ...] = ()
    prefix: Any = None
    remote_files: tuple[str, ...] = ()
    token: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict: ``passed`` is ``None`` when the check could not tell, which never lets a publish
    through."""

    name: str
    passed: bool | None
    finding: str
    remedy: str = ""


@runtime_checkable
class PublishCheck(Protocol):
    """A named property of a publish, checked before or after the upload.

    Read-only, so frozen dataclasses qualify.
    """

    @property
    def name(self) -> str:
        """How a caller names the check to skip or add it."""
        ...

    @property
    def stage(self) -> str:
        """:data:`PRE_UPLOAD` or :data:`POST_UPLOAD`."""
        ...

    @property
    def kinds(self) -> frozenset[str]:
        """The target kinds the check applies to."""
        ...

    @property
    def advisory(self) -> bool:
        """True for a check that reports a finding and never refuses."""
        ...

    def default_on(self, target: PublishTarget) -> bool:
        """Whether the check runs for ``target`` when the caller names it in neither ``skip_checks`` nor
        ``add_checks``."""
        ...

    def check(self, target: PublishTarget) -> CheckResult:
        """Measure the property on ``target``."""
        ...


@dataclass(frozen=True)
class PublishCheckReport:
    """The checks a publish ran and the ones the caller skipped, for the caller's own record."""

    results: tuple[CheckResult, ...] = ()
    skipped: tuple[str, ...] = ()

    def merged(self, other: PublishCheckReport) -> PublishCheckReport:
        """This report followed by ``other`` (a pre-upload report followed by its post-upload one)."""
        return PublishCheckReport(self.results + other.results, self.skipped + other.skipped)

    def lines(self) -> list[str]:
        """One line per result, then the skipped checks, for an operator-facing report."""
        out = [f"{_verdict(r.passed)} {r.name}: {r.finding}" for r in self.results]
        if self.skipped:
            out.append(f"skipped by request: {', '.join(self.skipped)}")
        return out


def _verdict(passed: bool | None) -> str:
    return {True: "ok     ", False: "FAILED ", None: "UNKNOWN"}[passed]


class UnknownPublishCheckError(ValueError):
    """A ``skip_checks`` or ``add_checks`` name matches no check for the target's kind."""


class PublishCheckError(RuntimeError):
    """One or more pre-upload checks refused the publish.

    ``failures`` holds every failed or unknown result.
    """

    def __init__(self, destination: str, failures: Iterable[CheckResult]):
        self.destination = destination
        self.failures = tuple(failures)
        super().__init__(self._message())

    @property
    def names(self) -> frozenset[str]:
        """The names of the checks that refused."""
        return frozenset(r.name for r in self.failures)

    def _header(self) -> str:
        return f"refusing to publish {self.destination!r}: {len(self.failures)} check(s) did not pass."

    def _message(self) -> str:
        lines = [self._header()]
        for r in self.failures:
            state = "could not tell" if r.passed is None else "failed"
            lines.append(f"- {r.name} ({state}): {r.finding}")
            if r.remedy:
                lines.append(f"  remedy: {r.remedy}")
        lines.append("A check can be skipped by name with skip_checks={...}; the skip is recorded with the publish.")
        return "\n".join(lines)


class PublishedArtifactCheckError(PublishCheckError):
    """A post-upload check failed: the artifact IS published, and the published copy does not pass."""

    def _header(self) -> str:
        return (
            f"{self.destination!r} was published, and the published copy failed {len(self.failures)} check(s). "
            "It is live as it stands: delete it or republish the revision, then check it again."
        )


_REGISTRY: dict[str, PublishCheck] = {}


def register_publish_check(check: PublishCheck) -> PublishCheck:
    """Register ``check`` under its name.

    A second check with the same name is refused.
    """
    if not isinstance(check, PublishCheck):
        raise TypeError(f"{check!r} does not implement PublishCheck")
    if check.stage not in (PRE_UPLOAD, POST_UPLOAD):
        raise ValueError(f"check {check.name!r} has stage {check.stage!r}; expected {PRE_UPLOAD!r} or {POST_UPLOAD!r}")
    existing = _REGISTRY.get(check.name)
    if existing is not None and existing is not check:
        raise ValueError(f"a publish check named {check.name!r} is already registered ({existing!r})")
    _REGISTRY[check.name] = check
    return check


def publish_checks_for(kind: str) -> list[PublishCheck]:
    """The registered checks that apply to ``kind``, both stages, in registration order."""
    return [c for c in _REGISTRY.values() if kind in c.kinds]


def validate_check_names(kind: str, skip_checks: Iterable[str] = (), add_checks: Iterable[str] = ()) -> None:
    """Refuse a ``skip_checks`` or ``add_checks`` name that matches no check for ``kind``, or names both."""
    skip, add = set(skip_checks), set(add_checks)
    known = {c.name for c in publish_checks_for(kind)}
    unknown = (skip | add) - known
    if unknown:
        raise UnknownPublishCheckError(
            f"no publish check named {sorted(unknown)} applies to a {kind}; the checks that do are {sorted(known)}"
        )
    if both := skip & add:
        raise UnknownPublishCheckError(f"{sorted(both)} named in both skip_checks and add_checks")


def run_publish_checks(
    target: PublishTarget,
    stage: str,
    *,
    skip_checks: Iterable[str] = (),
    add_checks: Iterable[str] = (),
) -> PublishCheckReport:
    """Run ``target``'s checks for ``stage`` and return the report, raising if any non-advisory check did not pass.

    Raises :class:`PublishCheckError` for a pre-upload stage and :class:`PublishedArtifactCheckError` for a post-upload
    one, listing every failure.
    """
    validate_check_names(target.kind, skip_checks, add_checks)
    skip, add = set(skip_checks), set(add_checks)
    results: list[CheckResult] = []
    skipped: list[str] = []
    failures: list[CheckResult] = []
    for check in publish_checks_for(target.kind):
        if check.stage != stage:
            continue
        if check.name in skip:
            skipped.append(check.name)
            continue
        if not (check.name in add or check.default_on(target)):
            continue
        result = check.check(target)
        results.append(result)
        if not check.advisory and result.passed is not True:
            failures.append(result)
    if failures:
        error = PublishCheckError if stage == PRE_UPLOAD else PublishedArtifactCheckError
        raise error(target.destination, failures)
    return PublishCheckReport(tuple(results), tuple(skipped))


class _CountingFile:
    """A read-only file wrapper that counts the bytes read through it."""

    def __init__(self, handle: Any):
        self._handle = handle
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        self.bytes_read += len(data)
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


@dataclass(frozen=True)
class RangeReadCheck:
    """Read one published Parquet file back through ``HfFileSystem`` with range requests.

    It reads the footer and the first column chunk of the first row group (the requests DuckDB's ``read_parquet`` makes
    against the Hub), checks that the chunk decodes to the row count the footer declares, and reports the bytes
    requested through the file handle against the file's size. Readers fetch the last 64 KiB to find the footer, so for
    a file smaller than that the count covers the whole file, and an already-read range requested twice counts twice.
    It is the only check that reads the artifact as published, so it is the one that catches a transport which
    rewrote it.

    On by default for the bucket store, where a transport rewrite is the live risk, and for AnalysisStore artifacts.
    Elsewhere it runs when named in ``add_checks``.
    """

    name: str = "range-read"
    stage: str = POST_UPLOAD
    kinds: frozenset[str] = frozenset({DASHBOARD_CORPUS, ANALYSIS_STORE})
    advisory: bool = False

    REMEDY: ClassVar[str] = (
        "The artifact is already published: delete it or republish the revision, then check it again. "
        "If the local files read correctly, suspect the transport."
    )

    def default_on(self, target: PublishTarget) -> bool:
        """Bucket-store dashboards and AnalysisStore artifacts."""
        return target.store == "bucket" or target.kind == ANALYSIS_STORE

    def check(self, target: PublishTarget) -> CheckResult:
        """Read the first published Parquet file's footer and first column chunk through ``HfFileSystem``."""
        if not target.remote_files:
            return CheckResult(self.name, None, "no published Parquet file was named to read back", self.REMEDY)
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import HfFileSystem
        except ImportError as exc:  # pragma: no cover - both ship with interpretune's dependencies
            return CheckResult(self.name, None, f"cannot read the published copy: {exc}", self.REMEDY)
        path = target.remote_files[0]
        fs = HfFileSystem(token=target.token)
        try:
            size = fs.size(path)
            with fs.open(path, "rb") as handle:
                counted = _CountingFile(handle)
                parquet = pq.ParquetFile(counted)
                if parquet.metadata.num_row_groups == 0:
                    return CheckResult(self.name, True, f"{path} declares no row groups (read {counted.bytes_read} B)")
                first = parquet.schema_arrow.names[0]
                table = parquet.read_row_group(0, columns=[first])
                declared = parquet.metadata.row_group(0).num_rows
        except Exception as exc:  # any failure to read back is the finding
            return CheckResult(
                self.name, False, f"reading {path} back failed: {type(exc).__name__}: {exc}", self.REMEDY
            )
        if table.num_rows != declared:
            return CheckResult(
                self.name,
                False,
                f"{path}: row group 0 decoded {table.num_rows} rows where the footer declares {declared}",
                self.REMEDY,
            )
        return CheckResult(
            self.name,
            True,
            f"{path}: footer and row group 0 of column {first!r} read back "
            f"({counted.bytes_read} bytes requested, of a {size}-byte file)",
        )


register_publish_check(RangeReadCheck())
