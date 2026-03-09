"""Tests for memory-aware analysis fixture helpers."""

from __future__ import annotations

from dataclasses import dataclass

from tests.analysis_resource_utils import ANALYSIS_LOW_RAM_GB, analysis_fixture_scope, conditional_clean_cpu


@dataclass
class _DummyFixture:
    result: object | None = "result"
    runner: object | None = "runner"
    run_config: object | None = "run_config"
    it_session: object | None = "it_session"


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
