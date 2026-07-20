"""Tests for memory-aware analysis fixture helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from tests.analysis_resource_utils import (
    ANALYSIS_LOW_RAM_GB,
    AnalysisFixtureSpec,
    AnalysisExtractionMixin,
    ExtractedAnalysisStore,
    ExtractedFixturePayload,
    analysis_resource_debug_enabled,
    analysis_fixture_scope,
    build_analysis_fixture_payload_extractor,
    clear_nnsight_test_state,
    conditional_clean_cpu,
    extract_analysis_store_fields,
    extract_result_dataset_metadata,
    log_resource_delta,
    serial_test_cleanup,
)
from tests.runif import get_runner_ram_gb


@dataclass
class _DummyFixture:
    result: object | None = "result"
    runner: object | None = "runner"
    run_config: object | None = "run_config"
    it_session: object | None = "it_session"
    model: object | None = None
    replacement_model: object | None = None
    module: object | None = None


@dataclass
class _DummyAnalysisResult:
    logit_diffs: list[torch.Tensor]
    preds: list[dict[str, dict[int, torch.Tensor]]]
    answer_logits: list[torch.Tensor] | None = None
    loss: list[torch.Tensor] | None = None
    dataset: object | None = None


@dataclass
class _DummyDataset:
    column_names: list[str]
    num_rows: int


class _DummyEnvoy:
    def __init__(self, children: list[object] | None = None):
        self._children = children or []
        self._source = object()
        self.clear_edits_calls = 0

    def clear_edits(self):
        self.clear_edits_calls += 1


class TestAnalysisFixtureScope:
    def test_returns_low_ram_scope_at_threshold(self, monkeypatch):
        monkeypatch.setattr("tests.analysis_resource_utils.get_runner_ram_gb", lambda: ANALYSIS_LOW_RAM_GB)
        assert analysis_fixture_scope() == "function"

    def test_returns_high_ram_scope_above_threshold(self, monkeypatch):
        monkeypatch.setattr("tests.analysis_resource_utils.get_runner_ram_gb", lambda: ANALYSIS_LOW_RAM_GB + 0.1)
        assert analysis_fixture_scope() == "class"


class TestResourceDebugHelpers:
    def test_analysis_resource_debug_enabled_checks_single_flag(self, monkeypatch):
        monkeypatch.delenv("IT_RESOURCE_DEBUG", raising=False)

        assert not analysis_resource_debug_enabled()

        monkeypatch.setenv("IT_RESOURCE_DEBUG", "1")

        assert analysis_resource_debug_enabled()

    def test_log_resource_delta_emits_current_values_and_deltas(self, monkeypatch, capsys):
        monkeypatch.setenv("IT_RESOURCE_DEBUG", "1")
        monkeypatch.setattr(
            "tests.analysis_resource_utils.get_resource_snapshot",
            lambda include_cuda=True: {
                "rss_gb": 2.0,
                "vms_gb": 3.0,
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_allocated_gb": 0.5,
                "cuda_reserved_gb": 1.0,
                "cuda_max_allocated_gb": 0.75,
                "cuda_max_reserved_gb": 1.25,
            },
        )

        log_resource_delta(
            "resource-delta",
            before={
                "rss_gb": 1.0,
                "vms_gb": 2.5,
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_allocated_gb": 0.25,
                "cuda_reserved_gb": 0.5,
                "cuda_max_allocated_gb": 0.5,
                "cuda_max_reserved_gb": 1.0,
            },
        )

        output = capsys.readouterr().out

        assert "[analysis_resource_debug] resource-delta:" in output
        assert "rss_gb=2.00" in output
        assert "delta_rss_gb=1.00" in output
        assert "delta_cuda_allocated_gb=0.25" in output


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


class TestSerialTestCleanup:
    def test_clear_nnsight_test_state_clears_envoy_tree_and_globals(self, monkeypatch):
        globals_calls = []

        class _DummyGlobals:
            @staticmethod
            def clear():
                globals_calls.append("cleared")

        monkeypatch.setitem(
            __import__("sys").modules,
            "nnsight.intervention.tracing.globals",
            type("_GlobalsModule", (), {"Globals": _DummyGlobals})(),
        )

        child = _DummyEnvoy()
        root = _DummyEnvoy(children=[child])

        clear_nnsight_test_state(root)

        assert root.clear_edits_calls == 1
        assert child.clear_edits_calls == 1
        assert root._source is None
        assert child._source is None
        assert globals_calls == ["cleared"]

    def test_serial_test_cleanup_releases_fixture_like_attrs(self, monkeypatch):
        empty_cache_calls = []
        monkeypatch.setattr("tests.analysis_resource_utils.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            "tests.analysis_resource_utils.torch.cuda.empty_cache", lambda: empty_cache_calls.append("emptied")
        )

        fixture = _DummyFixture()
        fixture.model = _DummyEnvoy()
        fixture.replacement_model = _DummyEnvoy()
        fixture.module = _DummyFixture()

        with serial_test_cleanup(fixture):
            assert fixture.result == "result"

        assert fixture.result is None
        assert fixture.runner is None
        assert fixture.run_config is None
        assert fixture.it_session is None
        assert fixture.model is None
        assert fixture.replacement_model is None
        assert fixture.module is None
        assert empty_cache_calls == ["emptied"]


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


class TestExtractedFixturePayload:
    def test_extract_result_dataset_metadata_reads_dataset_shape_info(self):
        result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([1.0])],
            preds=[{"sae_a": {0: torch.tensor([1.0])}}],
            dataset=_DummyDataset(column_names=["a", "b"], num_rows=3),
        )

        metadata = extract_result_dataset_metadata(result)

        assert metadata == {"column_names": ["a", "b"], "num_rows": 3}

    def test_build_analysis_fixture_payload_extractor_includes_metadata_and_store(self):
        analysis_result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([1.0])],
            preds=[{"sae_a": {0: torch.tensor([1.0])}}],
            answer_logits=[torch.tensor([1.0, 2.0])],
            dataset=_DummyDataset(column_names=["answer_logits"], num_rows=1),
        )
        fixture = _DummyFixture(result=analysis_result)
        extractor = build_analysis_fixture_payload_extractor(
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        )

        payload = extractor(fixture)

        assert isinstance(payload, ExtractedFixturePayload)
        assert payload.metadata == {"column_names": ["answer_logits"], "num_rows": 1}
        assert torch.equal(payload.store.answer_logits[0], torch.tensor([1.0, 2.0]))

    def test_build_analysis_fixture_payload_extractor_can_include_full_result(self):
        analysis_result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([1.0])],
            preds=[{"sae_a": {0: torch.tensor([1.0])}}],
        )
        fixture = _DummyFixture(result=analysis_result)
        extractor = build_analysis_fixture_payload_extractor(include_result=True)

        payload = extractor(fixture)

        assert payload.result is analysis_result


class TestAnalysisExtractionMixin:
    def test_extract_values_uses_declarative_fixture_specs(self):
        br_result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([1.0])],
            preds=[{"sae_a": {0: torch.tensor([1.0])}}],
        )
        ns_result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([2.0])],
            preds=[{"sae_a": {0: torch.tensor([2.0])}}],
        )

        class _DummyRequest:
            fixtures = {
                "bridge": _DummyFixture(result=br_result),
                "nnsight": _DummyFixture(result=ns_result),
            }

            def getfixturevalue(self, fixture_key):
                return self.fixtures[fixture_key]

        class _ExtractionHarness(AnalysisExtractionMixin):
            _analysis_fixture_specs = {
                "br": AnalysisFixtureSpec(fixture_key="bridge"),
                "ns": AnalysisFixtureSpec(fixture_key="nnsight"),
            }

        extracted = _ExtractionHarness().extract_values(_DummyRequest())

        assert torch.equal(extracted["br"].logit_diffs[0], torch.tensor([1.0]))
        assert torch.equal(extracted["ns"].logit_diffs[0], torch.tensor([2.0]))
        _ExtractionHarness.clear_extracted_values()

    def test_extract_cached_fixture_data_reuses_cached_payload(self):
        analysis_result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([1.0])],
            preds=[{"sae_a": {0: torch.tensor([1.0])}}],
            answer_logits=[torch.tensor([1.0, 2.0])],
            dataset=_DummyDataset(column_names=["answer_logits"], num_rows=1),
        )

        class _DummyRequest:
            def __init__(self):
                self.calls = 0

            def getfixturevalue(self, _fixture_key):
                self.calls += 1
                return _DummyFixture(result=analysis_result)

        class _ExtractionHarness(AnalysisExtractionMixin):
            pass

        harness = _ExtractionHarness()
        request = _DummyRequest()
        extractor = build_analysis_fixture_payload_extractor(
            field_names=("answer_logits",),
            include_dataset_metadata=True,
        )

        first = harness.extract_cached_fixture_data(request, "dummy", extractor)
        second = harness.extract_cached_fixture_data(request, "dummy", extractor)

        assert request.calls == 1
        assert first is second
        assert first.metadata == {"column_names": ["answer_logits"], "num_rows": 1}
        _ExtractionHarness.clear_extracted_values()

    def test_extract_field_store_and_metadata_use_declarative_specs(self):
        analysis_result = _DummyAnalysisResult(
            logit_diffs=[torch.tensor([1.0])],
            preds=[{"sae_a": {0: torch.tensor([1.0])}}],
            answer_logits=[torch.tensor([1.0, 2.0])],
            dataset=_DummyDataset(column_names=["answer_logits"], num_rows=1),
        )

        class _DummyRequest:
            def __init__(self):
                self.calls = 0

            def getfixturevalue(self, _fixture_key):
                self.calls += 1
                return _DummyFixture(result=analysis_result)

        class _ExtractionHarness(AnalysisExtractionMixin):
            _analysis_fixture_specs = {
                "payload": AnalysisFixtureSpec(
                    fixture_key="dummy",
                    field_names=("answer_logits",),
                    include_dataset_metadata=True,
                )
            }

        harness = _ExtractionHarness()
        request = _DummyRequest()

        metadata = harness.extract_dataset_metadata(request, "payload")
        store = harness.extract_field_store(request, "dummy", "answer_logits")

        assert request.calls == 1
        assert metadata == {"column_names": ["answer_logits"], "num_rows": 1}
        assert torch.equal(store.answer_logits[0], torch.tensor([1.0, 2.0]))
        _ExtractionHarness.clear_extracted_values()


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
