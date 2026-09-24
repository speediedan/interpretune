"""Unit tests for the explanation fallback wiring (fake CLI invoke, canned HTTP, no network)."""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import pytest

from interpretune.utils import model_fallback
from interpretune.utils import neuronpedia_explanations as npx
from interpretune.utils.model_fallback import ModelUnavailableError
from interpretune.utils.neuronpedia_explanations import (
    DEFAULT_EXPLANATION_CLI_MODEL,
    OPENCODE_EXPLANATION_CLI_SPEC,
    NeuronpediaExplanationError,
    build_explanation_export_record,
    build_explanation_ladder,
    discover_explanation_free_models,
    resolve_explanation_model,
)


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(model_fallback, "_PROCESS_CACHE", {})
    monkeypatch.setattr(model_fallback, "_WARNED", set())
    monkeypatch.setattr(model_fallback, "_cache_path", lambda: tmp_path / "fallback_cache.json")
    for var in (
        "IT_MODEL_AUTO_FALLBACK",
        "IT_MODEL_ALLOW_PAID",
        "IT_EXPLANATION_CLI_MODEL",
        "IT_EXPLANATION_FREE_FALLBACKS",
        "IT_EXPLANATION_GO_API_KEY",
        "IT_EXPLANATION_PROVIDER_API_KEY",
        "IT_EXPLANATION_CLI",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _scripted_invoke(script):
    """Fake invoke_explanation_cli following a model -> 'ok' | Exception script."""
    calls: list[tuple[str, str]] = []

    def fake(prompt, *, explanation_model=None, timeout_seconds=60, cli_spec=None):
        calls.append((explanation_model, getattr(cli_spec, "executable", None)))
        outcome = script[explanation_model]
        if isinstance(outcome, BaseException):
            raise outcome
        return npx.ExplanationCliInvocationResult(stdout="Method: 1\nExplanation: ok", stderr="")

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


class TestLadder:
    def test_bare_env_is_requested_only(self):
        requested, ladder = build_explanation_ladder(base_env={})
        assert requested.name == DEFAULT_EXPLANATION_CLI_MODEL
        assert ladder == []

    def test_env_model_names_requested(self):
        requested, _ = build_explanation_ladder(base_env={"IT_EXPLANATION_CLI_MODEL": "mine"})
        assert requested.name == "mine"

    def test_tiers_order_free_go_paid(self):
        env = {
            "IT_EXPLANATION_FREE_FALLBACKS": "free-a",
            "IT_EXPLANATION_GO_API_KEY": "go-key",
            "IT_EXPLANATION_PROVIDER_API_KEY": "bal-key",
        }
        _, ladder = build_explanation_ladder(base_env=env)
        kinds = [(c.name, c.free, c.paid) for c in ladder]
        assert kinds[0] == ("free-a", True, False)
        go_names = [c.name for c in ladder if c.provider == "opencode-go"]
        assert go_names == ["glm-5.3-flash", "kimi-k2.6"]
        assert all(c.paid for c in ladder if c.provider == "opencode-go")
        paid = [c for c in ladder if c.provider == "zen"]
        assert [c.name for c in paid] == ["deepseek-v4-flash"] and all(c.paid for c in paid)
        go_pos = min(ladder.index(c) for c in ladder if c.provider == "opencode-go")
        paid_pos = min(ladder.index(c) for c in ladder if c.provider == "zen")
        assert go_pos < paid_pos

    def test_go_and_paid_gated_on_keys(self):
        _, ladder = build_explanation_ladder(base_env={})
        assert all(not c.paid for c in ladder)
        _, ladder = build_explanation_ladder(base_env={"IT_EXPLANATION_GO_API_KEY": "k"})
        assert any(c.provider == "opencode-go" for c in ladder)
        assert not [c for c in ladder if c.provider == "zen"]


class TestResolve:
    def test_requested_answers_no_fallback(self, monkeypatch):
        fake = _scripted_invoke({"asked": "ok"})
        monkeypatch.setattr(npx, "invoke_explanation_cli", fake)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved = resolve_explanation_model(explanation_model="asked", base_env={})
        assert resolved.model_name == "asked"
        assert resolved.requested_name == "asked"
        assert fake.calls == [("asked", "copilot")]
        assert [w for w in caught if "unavailable" in str(w.message)] == []

    def test_falls_back_to_free_cli_route(self, monkeypatch):
        fake = _scripted_invoke(
            {
                "asked": NeuronpediaExplanationError("Model is unavailable."),
                "free-a": "ok",
            }
        )
        monkeypatch.setattr(npx, "invoke_explanation_cli", fake)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = resolve_explanation_model(
                explanation_model="asked",
                base_env={"IT_EXPLANATION_FREE_FALLBACKS": "free-a"},
            )
        assert resolved.model_name == "free-a"
        assert resolved.requested_name == "asked"
        assert resolved.cli_spec is OPENCODE_EXPLANATION_CLI_SPEC
        assert [c for _, c in fake.calls] == ["copilot", "opencode"]

    def test_missing_cli_route_skips_to_free(self, monkeypatch):
        fake = _scripted_invoke(
            {
                "asked": NeuronpediaExplanationError("Could not find the 'copilot' explanation CLI on PATH."),
                "free-a": "ok",
            }
        )
        monkeypatch.setattr(npx, "invoke_explanation_cli", fake)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = resolve_explanation_model(
                explanation_model="asked",
                base_env={"IT_EXPLANATION_FREE_FALLBACKS": "free-a"},
            )
        assert resolved.model_name == "free-a"

    def test_fallback_off_fails_fast_naming_switch(self, monkeypatch):
        fake = _scripted_invoke({"asked": NeuronpediaExplanationError("Model is unavailable.")})
        monkeypatch.setattr(npx, "invoke_explanation_cli", fake)
        with pytest.raises(ModelUnavailableError, match="IT_MODEL_AUTO_FALLBACK"):
            resolve_explanation_model(
                explanation_model="asked",
                base_env={"IT_MODEL_AUTO_FALLBACK": "0", "IT_EXPLANATION_FREE_FALLBACKS": "free-a"},
            )
        assert [m for m, _ in fake.calls] == ["asked"]

    def test_paid_opt_out(self, monkeypatch):
        script = {
            "asked": NeuronpediaExplanationError("Model is unavailable."),
            "free-a": NeuronpediaExplanationError("Model is unavailable."),
            "glm-5.3-flash": "ok",
        }
        monkeypatch.setattr(npx, "invoke_explanation_cli", _scripted_invoke(script))
        env = {"IT_EXPLANATION_FREE_FALLBACKS": "free-a", "IT_EXPLANATION_GO_API_KEY": "k", "IT_MODEL_ALLOW_PAID": "0"}
        with pytest.raises(ModelUnavailableError):
            resolve_explanation_model(explanation_model="asked", base_env=env)


class TestDiscovery:
    def test_discovers_free_ids_after_configured(self, monkeypatch):
        seen = {}

        def fake_urlopen(request, timeout=30):
            seen["url"] = request.full_url
            seen["ua"] = request.get_header("User-agent")
            assert request.get_header("Authorization") == "Bearer bal-key"
            return io.BytesIO(
                json.dumps(
                    {
                        "data": [
                            {"id": "new-free"},
                            {"id": "big-paid"},
                            {"id": "other:free"},
                        ]
                    }
                ).encode()
            )

        monkeypatch.setattr(npx, "urlopen", fake_urlopen)
        found = discover_explanation_free_models(base_env={"IT_EXPLANATION_PROVIDER_API_KEY": "bal-key"})
        assert seen["url"].endswith("/models")
        assert seen["ua"] and "python-urllib" not in seen["ua"].lower()
        assert [c.name for c in found] == ["new-free", "other:free"]
        assert all(c.free and c.cli_spec is OPENCODE_EXPLANATION_CLI_SPEC for c in found)

    def test_discovery_trouble_is_empty_not_error(self, monkeypatch):
        def bad_urlopen(request, timeout=30):
            raise ConnectionError("down")

        monkeypatch.setattr(npx, "urlopen", bad_urlopen)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = discover_explanation_free_models(base_env={"IT_EXPLANATION_PROVIDER_API_KEY": "k"})
        assert found == []
        assert any("discovery failed" in str(w.message) for w in caught)

    def test_no_key_no_discovery_no_network(self, monkeypatch):
        def bad_urlopen(request, timeout=30):  # pragma: no cover - must not be called
            raise AssertionError("network reached without a key")

        monkeypatch.setattr(npx, "urlopen", bad_urlopen)
        assert discover_explanation_free_models(base_env={}) == []


class TestRecord:
    def test_row_names_writer_notes_name_request(self):
        ref = npx.NeuronpediaFeatureRef(model_id="gemma-3-4b-it", layer="23-gemmascope", index="5")
        record = build_explanation_export_record(
            feature_ref=ref,
            cleaned_explanation="caps",
            artifact_path=Path("/tmp/x.md"),
            explanation_model="actual-paid",
            cached_activations_path=None,
            explanation_model_name="requested-free",
        )
        assert record["explanationModelName"] == "actual-paid"
        assert '"explanation_cli_model": "actual-paid"' in record["notes"]
        assert '"requested_explanation_model_name": "requested-free"' in record["notes"]
