"""Service-agnostic hosted-model fallback: try candidates in order, skip the unavailable, never retry them.

A service (explanations today; summaries or labelling tomorrow) names what it wants and the order it
wants it tried; this module owns the policy every service shares: which failures mean "try the next
one" versus "retry this one", the paid-tier opt-out, per-process and on-disk caching of the choice,
and a warning that names the model actually used.

The policy, in one paragraph: the requested model first, then the candidate ladder cheapest first.
A probe failure whose shape says the model is gone on this route (rotated out, free tier refused,
no funds, region-gated) records a skip reason and moves on -- retrying it would only delay the same
failure. Anything else (timeouts, bare 5xx, connection errors, anything unlisted) propagates
untouched for the caller's own retry loop, exactly as before. Paid candidates are skipped entirely
when ``IT_MODEL_ALLOW_PAID=0``. Nothing here imports any service module, so a second service reuses
it by passing its own candidates and probe.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

#: Master switch, default on. A per-service ``IT_<SERVICE>_MODEL_AUTO_FALLBACK`` takes precedence.
MODEL_AUTO_FALLBACK_ENV_VAR = "IT_MODEL_AUTO_FALLBACK"
#: Opt-out of every paid tier: the ladder stops after the free candidates.
MODEL_ALLOW_PAID_ENV_VAR = "IT_MODEL_ALLOW_PAID"
#: How long a resolved choice is reused before the ladder is re-probed (free tiers recover).
MODEL_FALLBACK_CACHE_TTL_SECONDS = 3600
#: Skip reason recorded when a CLI-route-only free model is offered no CLI route.
FREE_TIER_REQUIRES_OPENCODE = "free-tier-requires-opencode"
#: Skip reason recorded when the requested rung answers first try: nothing was unavailable.
NO_FALLBACK_NEEDED = "no fallback needed"


class ModelUnavailableError(RuntimeError):
    """A model candidate is gone on this route: skip it, do not retry it."""


@dataclass(frozen=True)
class ModelCandidate:
    """One rung of a service's fallback ladder.

    ``kind`` is ``'endpoint'`` (an OpenAI-compatible base URL plus key) or ``'cli'`` (a
    service-defined CLI spec the probe knows how to drive). ``paid`` marks tiers the
    ``IT_MODEL_ALLOW_PAID=0`` opt-out removes; ``free`` marks the free tier; ``cli_only``
    marks a free model reachable only through an in-CLI route, so an endpoint-kind rung
    carrying it is skipped with ``free-tier-requires-opencode`` instead of probed.
    """

    name: str
    """The model id probed, e.g. ``deepseek-v4-flash-free``."""
    kind: str = "endpoint"
    provider: str = ""
    """Who serves it, for messages, e.g. ``opencode`` or ``opencode-go``."""
    base_url: str | None = None
    api_key: str | None = None
    cli_spec: Any | None = None
    paid: bool = False
    free: bool = False
    cli_only: bool = False

    @property
    def display(self) -> str:
        """The name as user-facing messages show it, e.g. ``opencode/space-bunny-free``."""
        return f"{self.provider}/{self.name}" if self.provider else self.name


@dataclass(frozen=True)
class SkippedCandidate:
    """A rung that was not tried, and the reason it was not."""

    candidate: ModelCandidate
    reason: str


@dataclass(frozen=True)
class ResolvedModel:
    """The winning rung, what was skipped to reach it, and whether it came from cache."""

    candidate: ModelCandidate
    skipped: tuple[SkippedCandidate, ...] = ()
    from_cache: bool = False


#: Process-lifetime cache: key -> (ResolvedModel without from_cache set, resolve time). The TTL
#: applies here too, so a long batch process re-probes the ladder periodically instead of pinning
#: the first answer forever; the disk copy below extends the same policy across processes.
_PROCESS_CACHE: dict[str, tuple[ResolvedModel, float]] = {}
#: (service, choice display) pairs already warned about in this process.
_WARNED: set[tuple[str, str]] = set()


def _process_cache_get(key: str, ttl_seconds: int) -> ResolvedModel | None:
    entry = _PROCESS_CACHE.get(key)
    if entry is None:
        return None
    resolved, resolved_at = entry
    if ttl_seconds <= 0 or time.time() - resolved_at >= ttl_seconds:
        _PROCESS_CACHE.pop(key, None)
        return None
    return resolved


def _process_cache_set(key: str, resolved: ResolvedModel) -> None:
    _PROCESS_CACHE[key] = (resolved, time.time())


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def auto_fallback_enabled(service: str, env: dict[str, str] | None = None) -> bool:
    """Whether the service falls back past the requested model (default on).

    The per-service ``IT_<SERVICE>_MODEL_AUTO_FALLBACK`` wins over the master
    ``IT_MODEL_AUTO_FALLBACK`` when set, so one service can pin its model while others fall back.
    """
    env = dict(os.environ) if env is None else env
    override = env.get(f"IT_{service.upper()}_MODEL_AUTO_FALLBACK")
    if override is not None:
        return _truthy(override)
    return _truthy(env.get(MODEL_AUTO_FALLBACK_ENV_VAR, "1"))


def paid_tiers_allowed(env: dict[str, str] | None = None) -> bool:
    """Whether paid rungs stay on the ladder (opt out with ``IT_MODEL_ALLOW_PAID=0``)."""
    env = dict(os.environ) if env is None else env
    return _truthy(env.get(MODEL_ALLOW_PAID_ENV_VAR, "1"))


def classify_error(error: BaseException) -> tuple[str, str]:
    """Classify a probe failure as ``'unavailable'`` (skip, never retry) or ``'transient'`` (propagate).

    The unavailable shapes, checked before anything else because a wrapped message can carry both
    (e.g. ``server_error: ... Model is unavailable``): a model rotated out of the provider, a free
    tier refused on a keyed route, an empty balance, and a region-gated paid model. Timeouts, bare
    5xx, and connection errors stay transient, as does anything unlisted -- an unknown failure keeps
    today's retry behavior rather than silently skipping a model that might answer.
    """
    message = f"{type(error).__name__}: {error}"
    lowered = message.lower()
    if isinstance(error, TimeoutError) or "timed out" in lowered or "timeout" in lowered:
        return "transient", f"timeout: {message}"
    if "requires global regions" in lowered or "global regions" in lowered:
        return "unavailable", f"region-gated paid model: {message}"
    if "model is unavailable" in lowered or "model unavailable" in lowered:
        return "unavailable", f"rotated out: {message}"
    if "freetiererror" in lowered or "free tier can only be used" in lowered:
        return "unavailable", f"free tier refused on this route: {message}"
    if "insufficient account funds" in lowered or "insufficient funds" in lowered:
        return "unavailable", f"no funds: {message}"
    if ("could not find the" in lowered and "on path" in lowered) or "no such file or directory" in lowered:
        return "unavailable", f"CLI route absent on this machine: {message}"
    auth_markers = (
        "unauthorized",
        "unauthenticated",
        "invalid api key",
        "incorrect api key",
        "authentication failed",
        "invalid authentication",
        "authentication required",
        "http 401",
        "error 401",
        "auth required",
        "no api key",
        "missing api key",
        "login required",
        "not logged in",
    )
    if any(marker in lowered for marker in auth_markers):
        return "unavailable", f"auth refused on this route: {message}"
    return "transient", f"not an unavailable shape, retry as before: {message}"


def _cache_key(service: str, requested: ModelCandidate | None, candidates: list[ModelCandidate]) -> str:
    fingerprint = json.dumps(
        [service, requested.display if requested else None, [c.display for c in candidates]],
        sort_keys=True,
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:32]


def _cache_path() -> Path:
    return Path(tempfile.gettempdir()) / "interpretune" / "model_fallback_cache.json"


def _read_disk_cache(key: str, ttl_seconds: int) -> tuple[str, str] | None:
    """The cached (choice display, first-skip reason) for ``key``, or None on miss/expiry/trouble.

    A broken cache degrades to re-probing, never to an error: the cache is a latency optimization, not a correctness
    input. The reason travels with the choice so a process restoring from cache still warns with the real cause rather
    than going silent.
    """
    try:
        payload = json.loads(_cache_path().read_text())
        entry = payload.get(key)
        if not isinstance(entry, dict):
            return None
        if time.time() - float(entry.get("ts", 0)) >= ttl_seconds:
            return None
        choice = entry.get("choice")
        reason = entry.get("reason", "restored from cache")
        if not isinstance(choice, str) or not isinstance(reason, str):
            return None
        return choice, reason
    except (OSError, ValueError, TypeError):
        return None


def _write_disk_cache(key: str, choice: str, reason: str) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[key] = {"choice": choice, "ts": time.time(), "reason": reason}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except OSError:
        pass


def _warn_once(service: str, requested: ModelCandidate | None, choice: ModelCandidate, reason: str) -> None:
    marker = (service, choice.display)
    if marker in _WARNED:
        return
    _WARNED.add(marker)
    asked = requested.display if requested else "no model configured"
    paid = " (PAID: counts against the paid tier)" if choice.paid else ""
    warnings.warn(
        f"{service} model {asked} unavailable ({reason}); using {choice.display}{paid}",
        stacklevel=3,
    )


def resolve_model(
    service: str,
    *,
    requested: ModelCandidate | None = None,
    candidates: list[ModelCandidate] | None = None,
    probe: Callable[[ModelCandidate], Any] | None = None,
    discover: Callable[[], list[ModelCandidate]] | None = None,
    env: dict[str, str] | None = None,
    cache_ttl_seconds: int = MODEL_FALLBACK_CACHE_TTL_SECONDS,
) -> ResolvedModel:
    """Return the first rung that answers the probe, trying cheapest first.

    ``requested`` is the model the user asked for; ``candidates`` is the service's ladder after
    it (free, then paid); ``discover`` optionally supplies more free rungs (e.g. the provider's
    current free list) and is tried between the configured free rungs and the paid ones. ``probe``
    answers a tiny prompt per rung: it returns on success and raises on failure, and only failures
    :func:`classify_error` calls unavailable are skipped -- transient ones propagate untouched for
    the caller's retry loop.

    Raises:
        ValueError: nothing to try (no requested model and no candidates).
        ModelUnavailableError: every rung was skipped, naming each rung and reason; or fallback is
            off and the requested model is unavailable, naming the shape and the switch.
    """
    ladder = list(candidates or [])
    if discover is not None:
        free_rungs = [c for c in ladder if not c.paid]
        ladder = [*free_rungs, *discover(), *[c for c in ladder if c.paid]]
    if requested is None and not ladder:
        raise ValueError(f"{service}: nothing to resolve (no requested model and no candidates)")
    if probe is None:
        raise ValueError(f"{service}: nothing to probe with (no probe callable)")

    allow_paid = paid_tiers_allowed(env)
    key = _cache_key(service, requested, ladder)
    hit = _process_cache_get(key, cache_ttl_seconds)
    if hit is not None:
        return ResolvedModel(candidate=hit.candidate, skipped=hit.skipped, from_cache=True)
    if cache_ttl_seconds > 0:
        hit_disk = _read_disk_cache(key, cache_ttl_seconds)
        if hit_disk is not None:
            choice_display, reason = hit_disk
            for rung in ([requested] if requested else []) + ladder:
                if rung is not None and rung.display == choice_display:
                    resolved = ResolvedModel(candidate=rung, from_cache=True)
                    _process_cache_set(key, ResolvedModel(candidate=rung))
                    if reason != NO_FALLBACK_NEEDED:
                        _warn_once(service, requested, rung, reason)
                    return resolved

    if not auto_fallback_enabled(service, env):
        if requested is None:
            raise ValueError(f"{service}: fallback is off and no model was requested")
        try:
            probe(requested)
        except Exception as exc:
            kind, reason = classify_error(exc)
            if kind == "unavailable":
                var = f"IT_{service.upper()}_MODEL_AUTO_FALLBACK"
                raise ModelUnavailableError(
                    f"{service} model {requested.display} unavailable ({reason}); fallback is off "
                    f"({var} or {MODEL_AUTO_FALLBACK_ENV_VAR}); set a working model or re-enable fallback"
                ) from exc
            raise
        return ResolvedModel(candidate=requested)

    skipped: list[SkippedCandidate] = []
    rungs = ([requested] if requested is not None else []) + ladder
    for rung in rungs:
        if rung.paid and not allow_paid:
            skipped.append(SkippedCandidate(rung, f"paid tier opted out ({MODEL_ALLOW_PAID_ENV_VAR}=0)"))
            continue
        if rung.cli_only and rung.kind != "cli":
            skipped.append(SkippedCandidate(rung, FREE_TIER_REQUIRES_OPENCODE))
            continue
        try:
            probe(rung)
        except Exception as exc:
            kind, reason = classify_error(exc)
            if kind == "unavailable":
                skipped.append(SkippedCandidate(rung, reason))
                continue
            raise
        resolved = ResolvedModel(candidate=rung, skipped=tuple(skipped))
        _process_cache_set(key, resolved)
        first_reason = skipped[0].reason if skipped else NO_FALLBACK_NEEDED
        _write_disk_cache(key, rung.display, first_reason)
        if skipped:
            _warn_once(service, requested, rung, first_reason)
        return resolved

    tried = ", ".join(f"{s.candidate.display} ({s.reason})" for s in skipped) or "no rungs"
    raise ModelUnavailableError(f"{service}: every model rung unavailable: {tried}")


def probe_openai_chat(candidate: ModelCandidate, prompt: str, *, timeout_seconds: int = 60) -> str:
    """Probe an endpoint-kind rung with a tiny chat-completions call (stdlib only).

    Returns the response text. Raises the raw error (HTTP body included) for
    :func:`classify_error` to sort; timeouts propagate as transient.
    """
    from urllib.request import Request, urlopen

    if candidate.kind != "endpoint" or not candidate.base_url:
        raise ValueError(f"cannot probe non-endpoint candidate {candidate.display}")
    body = json.dumps(
        {"model": candidate.name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16}
    ).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "interpretune-model-fallback/1.0"}
    if candidate.api_key:
        headers["Authorization"] = f"Bearer {candidate.api_key}"
    request = Request(candidate.base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode())
    choices = payload.get("choices", []) if isinstance(payload, dict) else []
    if not choices:
        raise RuntimeError(f"HTTP 200 with no choices probing {candidate.display}: {payload!r:.200}")
    return str(choices[0].get("message", {}).get("content", ""))


#: Names the rung a probe failure came from when the probe cannot (kept beside the module so
#: services share one phrasing rather than each inventing it).
def probe_failure(candidate: ModelCandidate, detail: str) -> RuntimeError:
    """Wrap a raw probe failure detail with the rung that produced it."""
    return RuntimeError(f"probing {candidate.display}: {detail}")


# Re-exported for services that build rungs without importing anything service-specific.
__all__ = [
    "FREE_TIER_REQUIRES_OPENCODE",
    "NO_FALLBACK_NEEDED",
    "MODEL_ALLOW_PAID_ENV_VAR",
    "MODEL_AUTO_FALLBACK_ENV_VAR",
    "MODEL_FALLBACK_CACHE_TTL_SECONDS",
    "ModelCandidate",
    "ModelUnavailableError",
    "ResolvedModel",
    "SkippedCandidate",
    "auto_fallback_enabled",
    "classify_error",
    "paid_tiers_allowed",
    "probe_failure",
    "probe_openai_chat",
    "resolve_model",
]
