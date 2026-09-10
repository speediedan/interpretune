"""Case selection: a case names the declaration it needs, and the suite decides from the LIVE session.

The gate is the runtime declaration on the composed module's backends, because it is the only thing that
cannot lie about what the code does. A case whose gate is undeclared is skipped with a reason the report
distinguishes from every other skip, and its negative twin (the refusal case) runs instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from interpretune.analysis.backends import (
    AnalysisBackendCapability,
    ModelBackendCapability,
    InterventionMode,
    ModuleCapabilities,
    PositionScope,
)

#: The reason string every undeclared-gate skip carries. The report keys on it, so it is a constant.
UNDECLARED = "undeclared"


@dataclass(frozen=True)
class Gate:
    """What a case needs the target to have declared.

    Empty means always-on.
    """

    capability: ModelBackendCapability | AnalysisBackendCapability | None = None
    scope: PositionScope | None = None
    scopes: tuple[PositionScope, ...] = ()
    """Every scope a case needs declared at once (a mixed-scope payload); ``scope`` is the one-scope form."""
    mode: InterventionMode | None = None
    batched_hooks: bool | None = None
    family: str | None = None
    single_prompt: bool | None = None
    """Select on whether the TARGET declared it takes one prompt at a time (a property of the target, not of the
    backend's capability set)."""
    negative: bool = False
    """A negative case runs when its capability/scope/mode is ABSENT and is skipped when present."""

    def describe(self) -> str:
        """A one-line rendering for skip reasons and the report."""
        parts = []
        if self.capability is not None:
            parts.append(self.capability.name)
        if self.scope is not None:
            parts.append(f"scope={self.scope.value}")
        if self.scopes:
            parts.append("scopes=" + "+".join(s.value for s in self.scopes))
        if self.mode is not None:
            parts.append(f"mode={self.mode.value}")
        if self.batched_hooks is not None:
            parts.append(f"batched_hooks={self.batched_hooks}")
        if self.family is not None:
            parts.append(f"family={self.family}")
        if self.single_prompt is not None:
            parts.append(f"single_prompt={self.single_prompt}")
        return ("NOT " if self.negative else "") + ", ".join(parts) if parts else "always"

    def declared_by(self, caps: ModuleCapabilities, *, family: str, single_prompt: bool = False) -> bool:
        """Whether the positive form of this gate is satisfied by ``caps`` and the target's declarations."""
        if self.family is not None and family != self.family:
            return False
        if self.single_prompt is not None and single_prompt != self.single_prompt:
            return False
        if self.capability is not None and not caps.supports(self.capability):
            return False
        # Compare by VALUE: the capabilities module can be loaded twice under a test runner, leaving
        # value-equal, identity-distinct members on either side.
        needed = tuple(self.scopes) + ((self.scope,) if self.scope is not None else ())
        if needed:
            if caps.intervention is None:
                return False
            declared_scopes = {s.value for s in caps.intervention.position_scopes}
            if any(s.value not in declared_scopes for s in needed):
                return False
        if self.mode is not None:
            if caps.intervention is None or self.mode.value not in {m.value for m in caps.intervention.modes}:
                return False
        if self.batched_hooks is not None:
            if caps.latent_models is None or caps.latent_models.batched_hooks is not self.batched_hooks:
                return False
        return True

    def selects(self, caps: ModuleCapabilities, *, family: str, single_prompt: bool = False) -> bool:
        """Whether this gate (positive or negative) selects the case for these declarations."""
        declared = self.declared_by(caps, family=family, single_prompt=single_prompt)
        return (not declared) if self.negative else declared


@dataclass
class CaseRecord:
    """Bookkeeping the report reads: which gate each case had and how it ended."""

    name: str
    gate: Gate
    outcome: str = "pending"  # ran | skipped-undeclared | skipped-other | failed
    detail: str = ""


_CASE_GATES: dict[str, Gate] = {}


def conformance_case(gate: Gate | None = None, **kw: Any) -> Callable:
    """Mark a test method as a conformance case with the declaration it needs.

    ``conformance_case()`` is always-on; ``conformance_case(capability=ModelBackendCapability.ACTIVATION_INTERVENTION,
    scope=PositionScope.ALL_POSITIONS)`` needs both; ``negative=True`` inverts the selection.
    """
    resolved = gate if gate is not None else Gate(**kw)

    def mark(fn: Callable) -> Callable:
        fn.__conformance_gate__ = resolved  # type: ignore[attr-defined]
        _CASE_GATES[fn.__qualname__] = resolved
        return fn

    return mark


def gate_of(fn: Callable) -> Gate | None:
    """The gate a case was marked with, or ``None`` for an ordinary test."""
    return getattr(fn, "__conformance_gate__", None)


@dataclass
class SelectionReport:
    """Declared / ran / skipped-because-undeclared / skipped-other, printed at the end of every run.

    The vacuity guards read it: a run in which no gated case ran proved nothing about the adapter beyond
    composition, and a gated case skipped for any reason other than its gate must be visible as such.
    """

    declared: list[str] = field(default_factory=list)
    capture: dict[str, str] = field(default_factory=dict)
    """Each target class's capture declaration, rendered; ``"undeclared"`` when the backend has none."""
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Each target class's measured declaration as data (:func:`describe_capabilities`), for the report
    artifact."""
    ran: list[str] = field(default_factory=list)
    skipped_undeclared: list[str] = field(default_factory=list)
    skipped_other: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def record(self, name: str, outcome: str) -> None:
        """File one case under its outcome."""
        {
            "ran": self.ran,
            "skipped-undeclared": self.skipped_undeclared,
            "skipped-other": self.skipped_other,
            "failed": self.failed,
        }[outcome].append(name)

    def as_artifact(self, **provenance: Any) -> dict[str, Any]:
        """The report as data: per-target declarations plus the four outcome lists, with the caller's provenance.

        A consumer of this artifact (a card, an adapter's CI, a debugging session) gets what the suite MEASURED on a
        composed session, never what a manifest declared; provenance says which tree and which run measured it.
        """
        return {
            "format": "interpretune.conformance.report/1",
            "provenance": dict(provenance),
            "targets": {name: dict(record) for name, record in self.targets.items()},
            "ran": list(self.ran),
            "skipped_undeclared": list(self.skipped_undeclared),
            "skipped_other": list(self.skipped_other),
            "failed": list(self.failed),
        }

    def render(self) -> str:
        """The four counts, as printed at the end of every run."""
        lines = ["conformance selection report"]
        lines.append(f"  declared:            {', '.join(self.declared) or '-'}")
        for target, declaration in self.capture.items():
            lines.append(f"  capture ({target}): {declaration}")
        lines.append(f"  ran:                 {len(self.ran)}")
        lines.append(f"  skipped (undeclared):{len(self.skipped_undeclared)}")
        lines.append(
            f"  skipped (other):     {len(self.skipped_other)}"
            + (f"  <- {self.skipped_other}" if self.skipped_other else "")
        )
        lines.append(f"  failed:              {len(self.failed)}")
        return "\n".join(lines)


def describe_capabilities(caps: Any, *, composition: tuple[str, ...] = (), model_id: str = "") -> dict[str, Any]:
    """A module's :class:`~interpretune.analysis.backends.ModuleCapabilities` as plain data.

    Enums become their values, sets become sorted lists, and each support record is rendered field by field, so the
    result serializes as JSON and reads the same way the records describe themselves. A surface the module does not
    declare is absent rather than present-and-empty, so a consumer cannot mistake "not declared" for "declared with
    nothing".
    """
    import dataclasses
    from enum import Enum

    def plain(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (frozenset, set)):
            return sorted(plain(v) for v in value)
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return value

    out: dict[str, Any] = {
        "composition": list(composition),
        "model_id": model_id,
        "model_capabilities": sorted(c.value for c in caps.model),
        "analysis_capabilities": sorted(c.value for c in caps.analysis),
    }
    for name in ("intervention", "latent_models", "capture", "attribution_graph", "feature_intervention"):
        record = getattr(caps, name, None)
        if record is not None:
            out[name] = plain(record)
    return out
