"""Tests for memory-aware analysis fixture helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tests.analysis_resource_utils import (
    ANALYSIS_LOW_RAM_GB,
    ExtractedAnalysisStore,
    analysis_fixture_scope,
    conditional_clean_cpu,
    extract_analysis_store_fields,
)
from tests.runif import get_runner_ram_gb


@dataclass
class _DummyFixture:
    result: object | None = "result"
    runner: object | None = "runner"
    run_config: object | None = "run_config"
    it_session: object | None = "it_session"


@dataclass
class _DummyAnalysisResult:
    logit_diffs: list[torch.Tensor]
    preds: list[dict[str, dict[int, torch.Tensor]]]


class TestAnalysisFixtureScope:
    def test_returns_low_ram_scope_at_threshold(self, monkeypatch):
        monkeypatch.setattr("tests.analysis_resource_utils.get_runner_ram_gb", lambda: ANALYSIS_LOW_RAM_GB)
        assert analysis_fixture_scope() == "function"

    def test_returns_high_ram_scope_above_threshold(self, monkeypatch):
        monkeypatch.setattr("tests.analysis_resource_utils.get_runner_ram_gb", lambda: ANALYSIS_LOW_RAM_GB + 0.1)
        assert analysis_fixture_scope() == "class"


class TestConditionalCleanCpu:
    def test_clears_heavyweight_attrs_at_threshold(self, monkeypatch):
        monkeypatch.setattr("tests.analysis_resource_utils.get_runner_ram_gb", lambda: ANALYSIS_LOW_RAM_GB)
        fixture = _DummyFixture()

        with conditional_clean_cpu(fixture):
            assert fixture.result == "result"

        assert fixture.result is None
        assert fixture.runner is None
        assert fixture.run_config is None
        assert fixture.it_session is None

    def test_preserves_fixture_attrs_above_threshold(self, monkeypatch):
        monkeypatch.setattr("tests.analysis_resource_utils.get_runner_ram_gb", lambda: ANALYSIS_LOW_RAM_GB + 0.1)
        fixture = _DummyFixture()

        with conditional_clean_cpu(fixture):
            assert fixture.result == "result"

        assert fixture.result == "result"
        assert fixture.runner == "runner"
        assert fixture.run_config == "run_config"
        assert fixture.it_session == "it_session"


class TestExtractedAnalysisStore:
    def test_by_latent_model_stacks_nested_values(self):
        store = ExtractedAnalysisStore(
            preds=[
                {"sae_a": {0: torch.tensor([1.0]), 1: torch.tensor([2.0])}},
                {"sae_a": {0: torch.tensor([3.0]), 1: torch.tensor([4.0])}},
            ]
        )

        by_model = store.by_latent_model("preds")
        assert list(by_model.keys()) == ["sae_a"]
        assert torch.equal(by_model["sae_a"][0], torch.stack([torch.tensor([1.0]), torch.tensor([2.0])]))
        assert torch.equal(by_model["sae_a"][1], torch.stack([torch.tensor([3.0]), torch.tensor([4.0])]))

    def test_extract_analysis_store_fields_selects_requested_fields(self):
        analysis_result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([1.0])],
            preds=[{"sae_a": {0: torch.tensor([1.0])}}],
        )

        class _DummyRequest:
            def getfixturevalue(self, _fixture_key):
                return _DummyFixture(result=analysis_result)

        extracted = extract_analysis_store_fields(_DummyRequest(), "dummy", ("preds",))

        with pytest.raises(AttributeError):
            _ = extracted.logit_diffs
        assert torch.equal(extracted.by_latent_model("preds")["sae_a"][0], torch.stack([torch.tensor([1.0])]))


class TestRunnerRamOverride:
    def test_env_override_controls_runner_ram(self, monkeypatch):
        monkeypatch.setenv("IT_MOCK_RUNNER_RAM_GB", "12.5")
        get_runner_ram_gb.cache_clear()
        try:
            assert get_runner_ram_gb() == 12.5
        finally:
            get_runner_ram_gb.cache_clear()

    def test_env_override_at_new_threshold_uses_low_ram_scope(self, monkeypatch):
        monkeypatch.setenv("IT_MOCK_RUNNER_RAM_GB", str(ANALYSIS_LOW_RAM_GB))
        get_runner_ram_gb.cache_clear()
        try:
            assert analysis_fixture_scope() == "function"
        finally:
            get_runner_ram_gb.cache_clear()
