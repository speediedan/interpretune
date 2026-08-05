"""Detect and resolve collisions between an incoming dashboard corpus and what is already imported.

Downloading a published corpus and generating one locally are two ways to populate the SAME
Neuronpedia source set, not two sets. The importer is idempotent by way of ``ON CONFLICT DO NOTHING``,
which means a second import over an occupied set **reports success and writes nothing** -- the
failure that looks exactly like success, and the reason this module exists rather than a note in the
docs.

THE THREE POLICIES
------------------
``error`` (default)
    Refuse before importing anything and describe what occupies the set. Chosen as the default
    because the alternatives either destroy data or silently change identity, and neither should
    happen because a maintainer ran the obvious command.

``overwrite``
    Delete the occupying rows, then import. ``Activation`` **and** ``Explanation`` cascade from
    ``Neuron``: activations are regenerable from a corpus, explanations are not, so this refuses
    unless explanations are absent or ``allow_explanation_loss`` is set explicitly.

``rename``
    Leave the existing set untouched and import under a suffixed source-set id, so both coexist.

WHAT ``rename`` RENAMES, AND WHY IT IS THE INCOMING CORPUS
----------------------------------------------------------
It renames the **incoming** import, not the resident rows. Renaming the resident set would mean
rewriting ``Source.id`` (``{layer}-{setName}``) and every dependent key -- and ``Activation``'s
primary key embeds the source id as ``{prefix}-{feature}-{sequence}``, so a single 26-layer set is
~425k neurons and ~10-16M activation keys to rewrite inside one transaction. Renaming the incoming
corpus reaches the same end state -- both corpora resident under distinct ids -- as a config-level
decision that touches nothing already in the database.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ConflictPolicy",
    "SourceSetOccupancy",
    "ConflictResolution",
    "ExplanationLossRefused",
    "SourceSetConflictError",
    "DEFAULT_RENAME_SUFFIX",
    "describe_source_set",
    "resolve_source_set_conflict",
    "render_conflict_report",
    "suffix_source_set_id",
]

log = logging.getLogger(__name__)

#: Appended to a source-set id under the ``rename`` policy. Double underscore matches the separator
#: the pipeline already uses for run-name variants, so a renamed set sorts beside its sibling.
DEFAULT_RENAME_SUFFIX = "hub"

_SUFFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConflictPolicy(str, Enum):
    """What to do when the target source set is already occupied."""

    ERROR = "error"
    OVERWRITE = "overwrite"
    RENAME = "rename"


class SourceSetConflictError(RuntimeError):
    """The target source set is occupied and the policy is ``error``."""


class ExplanationLossRefused(RuntimeError):
    """``overwrite`` would cascade away explanations, which no corpus can regenerate."""


@dataclass(frozen=True)
class SourceSetOccupancy:
    """What currently occupies a (model, source set) pair."""

    model_id: str
    source_set_id: str
    source_count: int = 0
    neuron_count: int = 0
    explanation_count: int = 0

    @property
    def occupied(self) -> bool:
        # Sources alone count as occupancy: a partially-imported set still collides, and reporting it
        # as empty would send the caller into the silent ON CONFLICT DO NOTHING no-op.
        return bool(self.source_count or self.neuron_count)


@dataclass(frozen=True)
class ConflictResolution:
    """The decision taken, and what it changed."""

    policy: ConflictPolicy
    occupancy: SourceSetOccupancy
    effective_source_set_id: str
    deleted_neurons: int = 0
    deleted_explanations: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def renamed(self) -> bool:
        return self.effective_source_set_id != self.occupancy.source_set_id


def suffix_source_set_id(source_set_id: str, suffix: str) -> str:
    """``foo`` + ``hub`` -> ``foo__hub``, validated so it cannot produce an unusable id."""
    if not _SUFFIX_RE.match(suffix or ""):
        raise ValueError(
            f"rename suffix {suffix!r} is not usable in a Neuronpedia source id; "
            "use letters, digits, dot, dash or underscore, starting alphanumeric."
        )
    return f"{source_set_id}__{suffix}"


def _connect(db_url: str, timeout_seconds: int = 10) -> Any:
    import psycopg

    from interpretune.utils.neuronpedia_db_utils import resolve_local_neuronpedia_db_url

    return psycopg.connect(resolve_local_neuronpedia_db_url(db_url), connect_timeout=timeout_seconds)


def describe_source_set(db_url: str, *, model_id: str, source_set_id: str) -> SourceSetOccupancy:
    """Count what occupies a (model, source set) pair. Read-only.

    Scoped on ``Neuron.sourceSetName`` rather than by joining ``Source`` on ``layer``: the two are
    equivalent (verified equal at 425,984 rows on a full 26-layer set) and this form does not depend
    on ``Source`` rows existing, so a half-imported set still reports its neurons.
    """
    with _connect(db_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            'SELECT count(*) FROM "Source" WHERE "modelId" = %s AND "setName" = %s',
            (model_id, source_set_id),
        )
        source_count = int((cursor.fetchone() or [0])[0])

        cursor.execute(
            'SELECT count(*) FROM "Neuron" WHERE "modelId" = %s AND "sourceSetName" = %s',
            (model_id, source_set_id),
        )
        neuron_count = int((cursor.fetchone() or [0])[0])

        # Explanation has no sourceSetName; it joins to Neuron on (modelId, layer, index).
        cursor.execute(
            'SELECT count(*) FROM "Explanation" e '
            'JOIN "Neuron" n ON n."modelId" = e."modelId" AND n.layer = e.layer AND n.index = e.index '
            'WHERE n."modelId" = %s AND n."sourceSetName" = %s',
            (model_id, source_set_id),
        )
        explanation_count = int((cursor.fetchone() or [0])[0])

    return SourceSetOccupancy(
        model_id=model_id,
        source_set_id=source_set_id,
        source_count=source_count,
        neuron_count=neuron_count,
        explanation_count=explanation_count,
    )


def render_conflict_report(occupancy: SourceSetOccupancy, *, rename_suffix: str = DEFAULT_RENAME_SUFFIX) -> str:
    """The operator-facing explanation of a refusal, including both ways forward."""
    renamed = suffix_source_set_id(occupancy.source_set_id, rename_suffix)
    lines = [
        f"source set {occupancy.source_set_id!r} for model {occupancy.model_id!r} is already populated:",
        f"  {occupancy.source_count} sources, {occupancy.neuron_count} neurons, "
        f"{occupancy.explanation_count} explanations",
        "",
        "Importing on top of it would be a SILENT NO-OP: the importer uses ON CONFLICT DO NOTHING,",
        "so it would report success and write nothing. Choose explicitly:",
        f"  --rename-existing     keep both; import this corpus as {renamed!r}",
        "  --overwrite-existing  delete the rows above, then import",
    ]
    if occupancy.explanation_count:
        lines += [
            "",
            f"NOTE {occupancy.explanation_count} explanations hang off these neurons and CASCADE on delete.",
            "Activations are regenerable from a corpus; explanations are not. --overwrite-existing",
            "therefore also requires --allow-explanation-loss.",
        ]
    return "\n".join(lines)


def resolve_source_set_conflict(
    db_url: str,
    *,
    model_id: str,
    source_set_id: str,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
    rename_suffix: str = DEFAULT_RENAME_SUFFIX,
    allow_explanation_loss: bool = False,
    dry_run: bool = False,
) -> ConflictResolution:
    """Decide (and for ``overwrite``, perform) what happens to an occupied source set.

    Returns the source-set id the caller should actually import under. Safe to call when nothing is
    occupied -- it reports the unchanged id and touches nothing.
    """
    occupancy = describe_source_set(db_url, model_id=model_id, source_set_id=source_set_id)

    if not occupancy.occupied:
        return ConflictResolution(
            policy=policy,
            occupancy=occupancy,
            effective_source_set_id=source_set_id,
            notes=[f"{source_set_id!r} is empty for {model_id!r}; importing as-is"],
        )

    if policy is ConflictPolicy.ERROR:
        raise SourceSetConflictError(render_conflict_report(occupancy, rename_suffix=rename_suffix))

    if policy is ConflictPolicy.RENAME:
        renamed = suffix_source_set_id(source_set_id, rename_suffix)
        # The rename target can itself be occupied -- e.g. re-running the same rename twice. Left as a
        # hard error rather than suffixing again: silently landing in a third set would make the
        # database's contents depend on how many times a command had been run.
        collision = describe_source_set(db_url, model_id=model_id, source_set_id=renamed)
        if collision.occupied:
            raise SourceSetConflictError(
                f"rename target {renamed!r} is ALSO populated "
                f"({collision.neuron_count} neurons); choose another --rename-suffix, or "
                f"--overwrite-existing to replace it."
            )
        return ConflictResolution(
            policy=policy,
            occupancy=occupancy,
            effective_source_set_id=renamed,
            notes=[f"existing {source_set_id!r} left untouched; importing as {renamed!r}"],
        )

    # OVERWRITE
    if occupancy.explanation_count and not allow_explanation_loss:
        raise ExplanationLossRefused(
            f"--overwrite-existing would delete {occupancy.neuron_count} neurons from "
            f"{source_set_id!r}, cascading away {occupancy.explanation_count} explanations. "
            "Activations can be regenerated from a corpus; explanations cannot. Pass "
            "--allow-explanation-loss to proceed, or --rename-existing to keep both."
        )

    if dry_run:
        return ConflictResolution(
            policy=policy,
            occupancy=occupancy,
            effective_source_set_id=source_set_id,
            notes=[f"DRY RUN: would delete {occupancy.neuron_count} neurons from {source_set_id!r}"],
        )

    with _connect(db_url, timeout_seconds=30) as connection:
        with connection.cursor() as cursor:
            # One transaction: a partial purge leaves a set that is neither the old corpus nor
            # importable, and the next run would report a smaller conflict rather than the real state.
            cursor.execute(
                'DELETE FROM "Neuron" WHERE "modelId" = %s AND "sourceSetName" = %s',
                (model_id, source_set_id),
            )
            deleted = cursor.rowcount
            cursor.execute(
                'DELETE FROM "Source" WHERE "modelId" = %s AND "setName" = %s',
                (model_id, source_set_id),
            )
        connection.commit()

    log.info(
        "overwrite: deleted %s neurons (and %s cascaded explanations) from %r/%r",
        deleted,
        occupancy.explanation_count,
        model_id,
        source_set_id,
    )
    return ConflictResolution(
        policy=policy,
        occupancy=occupancy,
        effective_source_set_id=source_set_id,
        deleted_neurons=deleted,
        deleted_explanations=occupancy.explanation_count,
        notes=[f"deleted {deleted} neurons from {source_set_id!r} before import"],
    )
