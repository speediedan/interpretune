"""Which adapter a name reaches when a hub component and a bundled adapter both define it (interpretune#125).

This is the adapter half of the story ``prefer_ops`` / ``op_info`` tells for ops, and it is deliberately the
SAME story: bundled wins bare names by default, opting out is explicit and per-namespace, and there is a
verb that answers "which one am I actually running". A user who has learned one should not have to learn the
other.

**Where it differs, and why the default is stricter.** An op that resolves to the wrong definition computes
a wrong number, which is bad. An adapter composes into the MRO of the module a session runs, so the wrong
one changes what the session *is* -- and it does so at import time, before any result exists to look
suspicious. Ops therefore resolve silently under a documented precedence; adapters REFUSE to load and name
the opt-in. Both are the same rule ("explicit beats implicit"); only the cost of being wrong differs.

**Why refusing beats warning here.** A warning during a load is emitted before the session does anything
interesting and scrolls past whatever comes next. The failure it warns about surfaces much later, as a
composition that behaves unlike its name, at which point the warning is far offscreen and the natural
suspect is the component's own code. Refusal puts the diagnosis at the moment of the decision.

**Shadowing is not forbidden, it is unrequestable by accident.** A hub adapter replacing a bundled one is a
legitimate thing to want -- testing a fix, running a fork, trying a rewrite. The opt-in exists so that it
happens because someone asked for it, and so that ``adapter_info`` can then say plainly that it is in force.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Env parity for :func:`prefer_adapters`, for runs that cannot call it (scripts, CI, `python -m`).
IT_ADAPTER_PRECEDENCE_ENV_VAR = "IT_ADAPTER_PRECEDENCE"

#: In-process opt-in, highest precedence first. Module-level rather than a ContextVar, unlike the supported
#: composition set: that one is per-load state the rails hand down, whereas this is a session-wide user
#: preference, and a preference that failed to apply in a spawned thread would be a bug rather than a
#: documented boundary.
_preferred_adapter_namespaces: list[str] = []

#: Adapter name -> the hub components that have declared it, in load order. Populated by the loader so
#: `adapter_info` can report ALTERNATIVES rather than only the winner. Reporting only the winner makes a
#: shadowed bundled adapter invisible at exactly the moment someone is asking which one they are running.
_hub_adapter_sources: dict[str, list["AdapterCandidate"]] = {}


@dataclass(frozen=True)
class AdapterCandidate:
    """One definition of an adapter name, with the provenance needed to tell it from the others.

    Mirrors ``OpCandidate`` field-for-field where the concepts coincide, so the two ``*_info`` verbs read
    alike. ``component`` is the analogue of ``collection``: the hub repo that published it, and ``None`` for
    a bundled adapter, which HAS no publishing component.
    """

    name: str
    source: str
    component: str | None = None
    version: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, str]:
        """JSON-native form; keys with nothing to say are omitted rather than emitted as ``null``.

        Absence should read as absence: a bundled adapter has no revision, and recording ``"revision": null``
        invites a reader to treat it as a lookup that failed.
        """
        record = {"name": self.name, "source": self.source}
        for key in ("component", "version", "revision"):
            if (value := getattr(self, key)) is not None:
                record[key] = value
        return record

    def __str__(self) -> str:
        if self.component is None:
            return f"{self.name} (bundled)"
        detail = f"{self.component}@{self.revision[:12]}" if self.revision else str(self.component)
        return f"{self.name} (hub: {detail})"


@dataclass(frozen=True)
class AdapterResolution:
    """What an adapter name reaches right now, the alternatives, and the precedence that chose."""

    requested: str
    active: AdapterCandidate
    alternatives: tuple[AdapterCandidate, ...] = ()
    precedence: tuple[str, ...] = ()

    @property
    def is_shadowing_bundled(self) -> bool:
        """Whether a hub adapter is currently winning a name a bundled adapter also defines."""
        return self.active.source != "bundled" and any(c.source == "bundled" for c in self.alternatives)

    def __str__(self) -> str:
        lines = [f"{self.requested!r} resolves to {self.active}"]
        if self.alternatives:
            lines.append("  also available:")
            lines += [f"    {c}" for c in self.alternatives]
        if self.is_shadowing_bundled:
            lines.append("  NOTE: a hub adapter is shadowing the bundled adapter of the same name.")
        lines.append(f"  precedence: {list(self.precedence) or 'none (bundled adapters win bare names)'}")
        return "\n".join(lines)


def _normalize_namespace(repo_id: str) -> str:
    """A repo id as a precedence key.

    Case-folded, because Hub ids are not case sensitive in practice.
    """
    return repo_id.strip().lower()


def adapter_precedence() -> list[str]:
    """Hub components whose adapters may shadow bundled names, highest precedence first.

    In-process :func:`prefer_adapters` declarations come first, then any from ``IT_ADAPTER_PRECEDENCE``
    (comma separated, ordered) they do not already name. **The env var is read on every call rather than
    cached at import**, for the same reason the op dispatcher re-reads its own: this module is imported
    early, and a value exported afterwards would otherwise never be seen.
    """
    from_env = [
        ns
        for entry in os.environ.get(IT_ADAPTER_PRECEDENCE_ENV_VAR, "").split(",")
        if (ns := _normalize_namespace(entry))
    ]
    ordered = list(_preferred_adapter_namespaces)
    ordered += [ns for ns in from_env if ns not in ordered]
    return ordered


def prefer_adapters(*repo_ids: str, replace: bool = False) -> list[str]:
    """Allow these hub components' adapters to shadow bundled names; returns the active precedence list.

    Called with no arguments it clears the in-process opt-in (``IT_ADAPTER_PRECEDENCE`` still applies).
    """
    namespaces = [_normalize_namespace(r) for r in repo_ids]
    global _preferred_adapter_namespaces
    if replace or not repo_ids:
        _preferred_adapter_namespaces = namespaces
    else:
        for ns in namespaces:
            if ns in _preferred_adapter_namespaces:
                _preferred_adapter_namespaces.remove(ns)
            _preferred_adapter_namespaces.append(ns)
    return adapter_precedence()


def bundled_adapter_names() -> set[str]:
    """Adapter names the installed interpretune itself provides, from the entry-point group.

    Asked of the DISCOVERY surface rather than of a hardcoded list, so this cannot drift from what actually
    registers -- the same reason discovery stopped being a rails-owned tuple. An empty group (stale installed
    metadata) already warns there; here it would merely mean nothing is shadowable, which is a safe reading.
    """
    from interpretune.adapters._light_register import discover_adapter_entrypoints

    return set(discover_adapter_entrypoints())


def record_hub_adapter(name: str, component: str, revision: str | None = None, version: str | None = None) -> None:
    """Note that a hub component declared ``name``, so ``adapter_info`` can report it as an alternative."""
    candidate = AdapterCandidate(name=name, source="hub", component=component, revision=revision, version=version)
    existing = _hub_adapter_sources.setdefault(name, [])
    if not any(c.component == component and c.revision == revision for c in existing):
        existing.append(candidate)


class AdapterShadowError(RuntimeError):
    """A hub component declares an adapter name a bundled adapter already owns, with no opt-in in force."""


def enforce_adapter_precedence(repo_id: str, names: list[str], source: str) -> None:
    """Refuse a hub component that would shadow a bundled adapter name without an explicit opt-in.

    Raised BEFORE the enum members are created, which is the only placement that works: `register_dynamic_adapter`
    returns the EXISTING member for a name already present, so by the time registration runs, the component's
    compositions are being registered under the bundled adapter's identity and the shadowing has already
    happened silently. Checking afterwards would report a state it was too late to prevent.

    The message names the opt-in, because a refusal a user cannot act on is only half a diagnosis.
    """
    shadowed = sorted(set(names) & bundled_adapter_names())
    if not shadowed:
        return
    if _normalize_namespace(repo_id) in adapter_precedence():
        return
    plural = "s" if len(shadowed) > 1 else ""
    raise AdapterShadowError(
        f"{source} declares adapter name{plural} {shadowed!r}, which interpretune already provides as a "
        f"bundled adapter{plural}. Loading it would silently change what those name{plural} compose into "
        "for the rest of the session, so it is refused rather than applied.\n"
        f"  To use this component's adapter{plural} instead of the bundled one{plural}, opt in explicitly:\n"
        f"      it.hub.prefer_adapters({repo_id!r})\n"
        f"  or, for a scripted run:  {IT_ADAPTER_PRECEDENCE_ENV_VAR}={repo_id}\n"
        "  Publishing under a name that does not collide avoids the question entirely."
    )


def adapter_info(name: str) -> AdapterResolution:
    """Report which definition an adapter name reaches now, and what the alternatives are.

    The question "am I composing the bundled adapter or the one I pulled" has no answer from the ``Adapter``
    member itself -- both are the same enum member by construction, which is exactly what makes shadowing
    hard to see. This reports the resolution instead.
    """
    bundled = name in bundled_adapter_names()
    hub_candidates = list(_hub_adapter_sources.get(name, ()))
    if not bundled and not hub_candidates:
        raise KeyError(
            f"no adapter named {name!r} is known: it is neither bundled nor declared by a loaded hub "
            "component. `it.hub.pull` and load the component first, or check the name."
        )
    bundled_candidate = AdapterCandidate(name=name, source="bundled") if bundled else None
    precedence = adapter_precedence()
    # The winner is the highest-precedence hub component that named it; absent an opt-in, bundled wins --
    # which is the default this module exists to keep true.
    winner = None
    for ns in precedence:
        for candidate in hub_candidates:
            if candidate.component and _normalize_namespace(candidate.component) == ns:
                winner = candidate
                break
        if winner is not None:
            break
    if winner is None:
        winner = bundled_candidate or hub_candidates[0]
    alternatives = tuple(c for c in ([bundled_candidate] if bundled_candidate else []) + hub_candidates if c != winner)
    return AdapterResolution(requested=name, active=winner, alternatives=alternatives, precedence=tuple(precedence))
