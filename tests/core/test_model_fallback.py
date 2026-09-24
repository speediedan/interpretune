"""Unit tests for the service-agnostic model fallback (no network, no subprocess).

Every probe here is a canned callable: the tests exercise ordering, classification, caching,
and env gating without reaching any provider.
"""

from __future__ import annotations

import subprocess
import warnings

import pytest

from interpretune.utils import model_fallback
from interpretune.utils.model_fallback import (
    FREE_TIER_REQUIRES_OPENCODE,
    ModelCandidate,
    ModelUnavailableError,
    classify_error,
    resolve_model,
)


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch, tmp_path):
    """Fresh process cache, warn set, and a temp disk cache per test."""
    monkeypatch.setattr(model_fallback, "_PROCESS_CACHE", {})
    monkeypatch.setattr(model_fallback, "_WARNED", set())
    monkeypatch.setattr(model_fallback, "_cache_path", lambda: tmp_path / "fallback_cache.json")
    monkeypatch.delenv("IT_MODEL_AUTO_FALLBACK", raising=False)
    monkeypatch.delenv("IT_MODEL_ALLOW_PAID", raising=False)
    yield


def _endpoint(name, **kwargs):
    return ModelCandidate(name=name, kind="endpoint", provider="zen", base_url="https://x/v1", **kwargs)


def _scripted_probe(script):
    """A probe following a name -> 'ok' | Exception script, recording call order."""
    calls: list[str] = []

    def probe(candidate):
        calls.append(candidate.name)
        outcome = script[candidate.name]
        if isinstance(outcome, BaseException):
            raise outcome
        return "answer"

    probe.calls = calls  # type: ignore[attr-defined]
    return probe


class TestClassifyError:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (RuntimeError("server_error: Upstream request failed: Model is unavailable."), "unavailable"),
            (RuntimeError("Model is unavailable."), "unavailable"),
            (RuntimeError("FreeTierError: OpenCode's free tier can only be used from within OpenCode"), "unavailable"),
            (RuntimeError("server_error: Upstream request failed: Insufficient account funds"), "unavailable"),
            (RuntimeError("HTTP 400 This Go model requires Global regions. Select Global."), "unavailable"),
            (subprocess.TimeoutExpired("opencode", 5), "transient"),
            (TimeoutError("timed out"), "transient"),
            (RuntimeError("server_error: Upstream request failed"), "transient"),
            (RuntimeError("HTTP 503 Service Unavailable"), "transient"),
            (ConnectionError("refused"), "transient"),
            (RuntimeError("Could not find the 'copilot' explanation CLI on PATH."), "unavailable"),
            (RuntimeError("HTTP 401 Unauthorized: invalid api key"), "unavailable"),
            (RuntimeError("auth required, run copilot login"), "unavailable"),
            (ValueError("something nobody listed"), "transient"),
        ],
    )
    def test_shapes(self, error, expected):
        kind, _reason = classify_error(error)
        assert kind == expected


class TestLadder:
    def test_requested_first_then_free_then_paid(self):
        requested = _endpoint("asked")
        free = _endpoint("freebie", free=True)
        paid = _endpoint("paid", paid=True)
        probe = _scripted_probe(
            {
                "asked": RuntimeError("Model is unavailable."),
                "freebie": RuntimeError("Model is unavailable."),
                "paid": "ok",
            }
        )
        with pytest.warns(UserWarning, match="paid"):
            resolved = resolve_model("svc", requested=requested, candidates=[free, paid], probe=probe)
        assert resolved.candidate.name == "paid"
        assert [s.candidate.name for s in resolved.skipped] == ["asked", "freebie"]
        assert probe.calls == ["asked", "freebie", "paid"]

    def test_unavailable_never_retried(self):
        probe = _scripted_probe({"asked": RuntimeError("Model is unavailable."), "freebie": "ok"})
        resolve_model("svc", requested=_endpoint("asked"), candidates=[_endpoint("freebie", free=True)], probe=probe)
        assert probe.calls.count("asked") == 1

    def test_transient_propagates_for_caller_retry(self):
        probe = _scripted_probe({"asked": subprocess.TimeoutExpired("cli", 5)})
        with pytest.raises(subprocess.TimeoutExpired):
            resolve_model("svc", requested=_endpoint("asked"), candidates=[_endpoint("freebie")], probe=probe)
        assert probe.calls == ["asked"]

    def test_paid_opt_out_stops_at_free(self):
        probe = _scripted_probe(
            {
                "asked": RuntimeError("Model is unavailable."),
                "freebie": RuntimeError("Model is unavailable."),
            }
        )
        with pytest.raises(ModelUnavailableError, match="paid tier opted out"):
            resolve_model(
                "svc",
                requested=_endpoint("asked"),
                candidates=[_endpoint("freebie", free=True), _endpoint("go", paid=True)],
                probe=probe,
                env={"IT_MODEL_ALLOW_PAID": "0"},
            )
        assert probe.calls == ["asked", "freebie"]

    def test_exhausted_ladder_names_every_rung(self):
        probe = _scripted_probe({"asked": RuntimeError("Model is unavailable.")})
        with pytest.raises(ModelUnavailableError, match="asked.*Model is unavailable"):
            resolve_model("svc", requested=_endpoint("asked"), candidates=[], probe=probe)

    def test_nothing_to_try_refused_by_name(self):
        with pytest.raises(ValueError, match="nothing to resolve"):
            resolve_model("svc", probe=lambda c: None)

    def test_cli_only_skipped_over_endpoint_route(self):
        cli_only = _endpoint("zen-free", free=True, cli_only=True)
        cli = ModelCandidate(name="zen-free", kind="cli", provider="opencode", free=True, cli_only=True)
        probe = _scripted_probe({"zen-free": "ok"})
        resolved = resolve_model("svc", candidates=[cli_only, cli], probe=probe)
        assert resolved.candidate.kind == "cli"
        assert resolved.skipped[0].reason == FREE_TIER_REQUIRES_OPENCODE
        assert probe.calls == ["zen-free"]

    def test_discovery_sits_between_free_and_paid(self):
        probe = _scripted_probe(
            {
                "asked": RuntimeError("Model is unavailable."),
                "cfg-free": RuntimeError("Model is unavailable."),
                "found-free": "ok",
            }
        )
        resolved = resolve_model(
            "svc",
            requested=_endpoint("asked"),
            candidates=[_endpoint("cfg-free", free=True), _endpoint("go", paid=True)],
            discover=lambda: [_endpoint("found-free", free=True)],
            probe=probe,
        )
        assert resolved.candidate.name == "found-free"
        assert probe.calls == ["asked", "cfg-free", "found-free"]


class TestSwitches:
    def test_fallback_off_tries_requested_only(self):
        probe = _scripted_probe({"asked": RuntimeError("Model is unavailable."), "freebie": "ok"})
        with pytest.raises(ModelUnavailableError, match="IT_MODEL_AUTO_FALLBACK"):
            resolve_model(
                "svc",
                requested=_endpoint("asked"),
                candidates=[_endpoint("freebie")],
                probe=probe,
                env={"IT_MODEL_AUTO_FALLBACK": "0"},
            )
        assert probe.calls == ["asked"]

    def test_per_service_override_beats_master(self):
        probe = _scripted_probe(
            {
                "asked": RuntimeError("Model is unavailable."),
                "freebie": "ok",
            }
        )
        resolved = resolve_model(
            "svc",
            requested=_endpoint("asked"),
            candidates=[_endpoint("freebie")],
            probe=probe,
            env={"IT_MODEL_AUTO_FALLBACK": "0", "IT_SVC_MODEL_AUTO_FALLBACK": "1"},
        )
        assert resolved.candidate.name == "freebie"

    def test_master_off_applies(self):
        probe = _scripted_probe({"asked": RuntimeError("Model is unavailable."), "freebie": "ok"})
        with pytest.raises(ModelUnavailableError):
            resolve_model(
                "svc",
                requested=_endpoint("asked"),
                candidates=[_endpoint("freebie")],
                probe=probe,
                env={"IT_MODEL_AUTO_FALLBACK": "0"},
            )


class TestCacheAndWarning:
    def test_second_resolve_comes_from_process_cache(self):
        probe = _scripted_probe({"asked": RuntimeError("Model is unavailable."), "freebie": "ok"})
        kwargs = dict(requested=_endpoint("asked"), candidates=[_endpoint("freebie", free=True)], probe=probe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = resolve_model("svc", **kwargs)
            second = resolve_model("svc", **kwargs)
        assert not first.from_cache and second.from_cache
        assert second.candidate.name == "freebie"
        assert probe.calls == ["asked", "freebie"]

    def test_zero_ttl_reprobes(self):
        probe = _scripted_probe({"asked": "ok"})
        kwargs = dict(requested=_endpoint("asked"), probe=probe, cache_ttl_seconds=0)
        resolve_model("svc", **kwargs)
        resolve_model("svc", **kwargs)
        assert probe.calls == ["asked", "asked"]

    def test_warning_once_per_process_names_models(self):
        probe = _scripted_probe({"asked": RuntimeError("Model is unavailable."), "freebie": "ok"})
        kwargs = dict(requested=_endpoint("asked"), candidates=[_endpoint("freebie", free=True)], probe=probe)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_model("svc", **kwargs)
            resolve_model("svc", **kwargs)
        fallback_warnings = [w for w in caught if "unavailable" in str(w.message)]
        assert len(fallback_warnings) == 1
        assert "asked" in str(fallback_warnings[0].message) and "freebie" in str(fallback_warnings[0].message)

    def test_no_warning_when_requested_answers(self):
        probe = _scripted_probe({"asked": "ok"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_model("svc", requested=_endpoint("asked"), probe=probe)
        assert [w for w in caught if "unavailable" in str(w.message)] == []


class TestServiceAgnostic:
    def test_dummy_service_without_service_imports(self):
        """The fallback module carries no service imports: a new service reuses it as-is."""
        import inspect

        source = inspect.getsource(model_fallback)
        assert "neuronpedia" not in source
        assert "ExplanationCli" not in source
        probe = _scripted_probe({"dummy-asked": RuntimeError("Model is unavailable."), "dummy-free": "ok"})
        resolved = resolve_model(
            "dummy",
            requested=ModelCandidate(name="dummy-asked", kind="cli", provider="dummy"),
            candidates=[ModelCandidate(name="dummy-free", kind="cli", provider="dummy", free=True)],
            probe=probe,
        )
        assert resolved.candidate.name == "dummy-free"
