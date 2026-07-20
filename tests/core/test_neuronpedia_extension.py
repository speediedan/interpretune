from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("neuronpedia")

import neuronpedia
from neuronpedia.requests.base_request import NPRequest

from interpretune.extensions.neuronpedia import NeuronpediaConfig, NeuronpediaIntegration


class _DummyHandle:
    def __init__(self) -> None:
        self.it_cfg = SimpleNamespace(neuronpedia_cfg=NeuronpediaConfig(enabled=True))
        self.core_log_dir = "/tmp"


def _build_integration() -> NeuronpediaIntegration:
    integration = NeuronpediaIntegration()
    integration.connect(_DummyHandle())
    return integration


def test_prepare_graph_metadata_synchronizes_source_set_fields() -> None:
    integration = _build_integration()
    graph_dict = {
        "metadata": {
            "info": {"creator_name": "tester"},
            "feature_details": {"neuronpedia_source_set": "gemmascope-2-transcoder-16k"},
        }
    }

    prepared = integration.prepare_graph_metadata(graph_dict, slug="demo-graph")

    assert prepared["metadata"]["slug"] == "demo-graph"
    assert prepared["metadata"]["feature_details"]["neuronpedia_source_set"] == "gemmascope-2-transcoder-16k"
    assert prepared["metadata"]["info"]["neuronpedia_source_set"] == "gemmascope-2-transcoder-16k"


def test_prepare_graph_metadata_uses_configured_neuronpedia_model_for_scan() -> None:
    integration = _build_integration()
    graph_dict = {
        "metadata": {
            "scan": {"unexpected": "payload"},
            "info": {"creator_name": "tester", "neuronpedia_model": "gemma-3-4b-it"},
        }
    }

    prepared = integration.prepare_graph_metadata(graph_dict, slug="demo-graph")

    assert prepared["metadata"]["scan"] == "gemma-3-4b-it"


def test_upload_graph_to_neuronpedia_reconfigures_local_request_base_url(monkeypatch, tmp_path: Path) -> None:
    integration = _build_integration()
    graph_path = tmp_path / "graph.json"
    graph_path.write_text('{"metadata": {"slug": "demo"}}', encoding="utf-8")

    seen_base_urls: list[str] = []

    class _FakeGraphMetadataClient:
        @staticmethod
        def upload_file(path: str) -> SimpleNamespace:
            del path
            seen_base_urls.append(NPRequest.BASE_URL)
            return SimpleNamespace(url="http://localhost:3999/gemma-3-1b-it/graph?slug=demo")

    monkeypatch.setattr(integration, "validate_graph", lambda graph_dict: True)
    monkeypatch.setattr(integration, "_np_graph_metadata", _FakeGraphMetadataClient)
    monkeypatch.setattr(neuronpedia, "api_key", lambda api_key: nullcontext())
    monkeypatch.setenv("USE_LOCALHOST", "true")
    monkeypatch.setenv("LOCAL_NEURONPEDIA_WEBAPP_URL", "http://localhost:3999")
    monkeypatch.setenv("DEV_NEURONPEDIA_API_KEY", "dev-key")

    original_use_localhost = NPRequest.USE_LOCALHOST
    original_base_url = NPRequest.BASE_URL
    NPRequest.USE_LOCALHOST = False
    NPRequest.BASE_URL = "https://neuronpedia.org/api"

    try:
        graph_metadata = integration.upload_graph_to_neuronpedia(graph_path)
    finally:
        NPRequest.USE_LOCALHOST = original_use_localhost
        NPRequest.BASE_URL = original_base_url

    assert graph_metadata.url.endswith("slug=demo")
    assert seen_base_urls == ["http://localhost:3999/api"]
