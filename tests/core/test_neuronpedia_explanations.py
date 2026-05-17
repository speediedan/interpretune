from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

utils = importlib.import_module("interpretune.utils")
np_explanations = importlib.import_module("interpretune.utils.neuronpedia_explanations")
RunIf: Any = importlib.import_module("runif").RunIf
DEFAULT_COPILOT_MODEL = utils.DEFAULT_COPILOT_MODEL
DEFAULT_COPILOT_MAX_RETRIES = utils.DEFAULT_COPILOT_MAX_RETRIES
DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS = utils.DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS
NeuronpediaFeatureRef = utils.NeuronpediaFeatureRef
activation_batch_number = utils.activation_batch_number
artifact_output_path = utils.artifact_output_path
build_feature_ref = utils.build_feature_ref
build_explanation_prompt = utils.build_explanation_prompt
feature_tuples_to_feature_refs = utils.feature_tuples_to_feature_refs
build_explanation_export_record = utils.build_explanation_export_record
build_np_max_act_logits_prompt = utils.build_np_max_act_logits_prompt
candidate_cached_activation_batch_paths = utils.candidate_cached_activation_batch_paths
cached_activation_batch_path = utils.cached_activation_batch_path
clean_explanation_text = utils.clean_explanation_text
default_np_cache_dir = utils.default_np_cache_dir
derive_np_max_act_logits_inputs = utils.derive_np_max_act_logits_inputs
generate_explanation_artifact = utils.generate_explanation_artifact
parse_feature_url = utils.parse_feature_url
public_activation_batch_url = utils.public_activation_batch_url
resolve_local_neuronpedia_db_url = utils.resolve_local_neuronpedia_db_url
check_local_explanation_coverage = utils.check_local_explanation_coverage
invoke_copilot_cli_with_retries = utils.invoke_copilot_cli_with_retries


def _sample_feature_payload() -> dict:
    return {
        "pos_str": ["▁Washington", "\nBrussels", "Ottawa"],
        "activations": [
            {
                "tokens": ["the ", "capital", " of", " France"],
                "values": [0.1, 2.0, 0.0, 0.0],
                "maxValue": 2.0,
                "maxValueTokenIndex": 1,
            },
            {
                "tokens": ["visited ", "London", " yesterday"],
                "values": [0.0, 1.5, 0.0],
                "maxValue": 1.5,
                "maxValueTokenIndex": 1,
            },
            {
                "tokens": ["moved to ", "Ottawa"],
                "values": [0.0, 1.0],
                "maxValue": 1.0,
                "maxValueTokenIndex": 1,
            },
            {
                "tokens": ["visited ", "London", " yesterday"],
                "values": [0.0, 0.5, 0.0],
                "maxValue": 0.5,
                "maxValueTokenIndex": 1,
            },
        ],
    }


def test_parse_feature_url_from_public_page() -> None:
    feature_ref = parse_feature_url("https://www.neuronpedia.org/gemma-3-4b-it/23-gemmascope-2-transcoder-262k/13341")

    assert feature_ref == NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
        base_url="https://www.neuronpedia.org",
    )
    assert feature_ref.source_set == "gemmascope-2-transcoder-262k"
    assert feature_ref.layer_number == "23"


def test_build_feature_ref_combines_numeric_layer_and_source_set() -> None:
    feature_ref = build_feature_ref(
        model_id="gemma-3-1b-it",
        layer="23",
        source_set="gemmascope-2-clt-262k",
        index=7,
    )

    assert feature_ref.layer == "23-gemmascope-2-clt-262k"
    assert feature_ref.feature_url.endswith("/gemma-3-1b-it/23-gemmascope-2-clt-262k/7")


def test_feature_tuples_to_feature_refs_builds_expected_routes() -> None:
    feature_refs = feature_tuples_to_feature_refs(
        model_id="gemma-3-1b-it",
        source_set="gemmascope-2-transcoder-16k",
        feature_tuples=[(23, 17), (24, 99)],
    )

    assert [feature_ref.layer for feature_ref in feature_refs] == [
        "23-gemmascope-2-transcoder-16k",
        "24-gemmascope-2-transcoder-16k",
    ]
    assert [feature_ref.index for feature_ref in feature_refs] == ["17", "99"]


def test_derive_np_max_act_logits_inputs_reconstructs_prompt_lists() -> None:
    inputs = derive_np_max_act_logits_inputs(_sample_feature_payload())

    assert inputs.max_activating_tokens == ["capital", "Ottawa", "London"]
    assert inputs.tokens_after_max_activating_token == ["of", "", "yesterday"]
    assert inputs.top_positive_logits == ["Washington", "Brussels", "Ottawa"]
    assert inputs.top_activating_texts == [
        "the capital of France",
        "moved to Ottawa",
        "visited London yesterday",
    ]


def test_load_feature_payload_with_cached_activations_falls_back_to_api_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-1b-it",
        layer="0-gemmascope-2-transcoder-262k-rte",
        index="0",
    )
    payload = _sample_feature_payload()

    monkeypatch.setattr(np_explanations, "fetch_feature_payload", lambda *_args, **_kwargs: dict(payload))

    def _raise_missing_cache(*_args, **_kwargs):
        raise np_explanations.NeuronpediaExplanationError("missing cache")

    monkeypatch.setattr(np_explanations, "load_cached_feature_activations", _raise_missing_cache)

    loaded_payload, batch_path = np_explanations.load_feature_payload_with_cached_activations(feature_ref)

    assert batch_path is None
    assert loaded_payload["activations"] == payload["activations"]


def test_build_np_max_act_logits_prompt_contains_feature_metadata_and_lists() -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )
    inputs = derive_np_max_act_logits_inputs(_sample_feature_payload())

    prompt = build_np_max_act_logits_prompt(feature_ref, inputs)

    assert "Feature URL: https://www.neuronpedia.org/gemma-3-4b-it/23-gemmascope-2-transcoder-262k/13341" in prompt
    assert "Source set: gemmascope-2-transcoder-262k" in prompt
    assert "<TOP_POSITIVE_LOGITS>" in prompt
    assert "Washington" in prompt
    assert "Few-shot examples:" in prompt


def test_build_explanation_prompt_uses_default_type() -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )
    inputs = derive_np_max_act_logits_inputs(_sample_feature_payload())

    prompt, prompt_style = build_explanation_prompt(
        feature_ref,
        inputs,
    )

    assert prompt_style == np_explanations.DEFAULT_EXPLANATION_TYPE_NAME
    assert prompt_style == "np_max-act-logits"
    assert "Method: <method number>" in prompt
    assert "Use the following ordered method exactly" in prompt


def test_build_explanation_prompt_supports_moe_type() -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )
    inputs = derive_np_max_act_logits_inputs(_sample_feature_payload())

    prompt, prompt_style = build_explanation_prompt(
        feature_ref,
        inputs,
        type_name=np_explanations.NP_MOE_MAX_ACT_LOGITS_TYPE_NAME,
    )

    assert prompt_style == np_explanations.NP_MOE_MAX_ACT_LOGITS_TYPE_NAME
    assert "Use all four evidence sources together as a mixture-of-experts judgment" in prompt
    assert "Method: -1" in prompt


def test_invoke_copilot_cli_with_retries_retries_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def _fake_invoke(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise subprocess.TimeoutExpired(cmd=["copilot", "-p"], timeout=120)
        return np_explanations.CopilotInvocationResult(stdout="Explanation: capitals", stderr="")

    monkeypatch.setattr(np_explanations, "invoke_copilot_cli", _fake_invoke)
    sleep_calls: list[float] = []
    monkeypatch.setattr(np_explanations.time, "sleep", sleep_calls.append)

    result = invoke_copilot_cli_with_retries(
        "prompt",
        max_retries=DEFAULT_COPILOT_MAX_RETRIES,
        retry_backoff_seconds=DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS,
    )

    assert result.stdout == "Explanation: capitals"
    assert attempts["count"] == 3
    assert sleep_calls == [DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS, DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS * 2]


def test_check_local_explanation_coverage_filters_by_requested_type(monkeypatch: pytest.MonkeyPatch) -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-1b-it",
        layer="16-gemmascope-2-transcoder-16k",
        index="48",
    )
    executed: list[tuple[str, tuple[Any, ...]]] = []

    class _Cursor:
        def execute(self, query: str, params: tuple[Any, ...]) -> None:
            executed.append((query, params))

        def fetchone(self) -> tuple[int]:
            return (1,)

        def __enter__(self) -> _Cursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class _Connection:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *_args, **_kwargs: _Connection()))

    statuses = check_local_explanation_coverage(
        [feature_ref],
        local_db_url="postgres://postgres:postgres@127.0.0.1:5433/postgres",
        type_name=np_explanations.DEFAULT_EXPLANATION_TYPE_NAME,
    )

    assert statuses[0].explanation_count == 1
    assert executed[0][1][3] == np_explanations.DEFAULT_EXPLANATION_TYPE_NAME
    assert '"requested_type_name": "np_max-act-logits"' in executed[0][1][4]


def test_clean_explanation_text_extracts_final_explanation_line() -> None:
    raw_response = "Method: 3\nReason: The logits are capital-city names.\nExplanation: capital cities."

    assert clean_explanation_text(raw_response) == "capital cities"


def test_clean_explanation_text_preserves_internal_quotes() -> None:
    raw_response = 'Method: 2\nReason: The continuation tokens are city names.\nExplanation: say "cities"'

    assert clean_explanation_text(raw_response) == 'say "cities"'


def test_cleanup_feature_activation_cache_only_removes_feature_cache(tmp_path: Path) -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-1b-it",
        layer="16-gemmascope-2-transcoder-16k",
        index="48",
    )
    feature_cache_path = np_explanations.cached_feature_activation_path(feature_ref, cache_dir=tmp_path)
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_cache_path.write_text("{}\n", encoding="utf-8")

    cleaned = np_explanations.cleanup_feature_activation_cache(
        feature_ref,
        cache_dir=tmp_path,
        cached_activations_path=feature_cache_path,
    )

    assert cleaned is True
    assert not feature_cache_path.exists()


def test_cleanup_feature_activation_cache_keeps_batch_cache(tmp_path: Path) -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-1b-it",
        layer="16-gemmascope-2-transcoder-16k",
        index="48",
    )
    batch_cache_path = np_explanations.cached_activation_batch_path(feature_ref, cache_dir=tmp_path)
    batch_cache_path.parent.mkdir(parents=True, exist_ok=True)
    batch_cache_path.write_text("{}\n", encoding="utf-8")

    cleaned = np_explanations.cleanup_feature_activation_cache(
        feature_ref,
        cache_dir=tmp_path,
        cached_activations_path=batch_cache_path,
    )

    assert cleaned is False
    assert batch_cache_path.exists()


def test_activation_batch_helpers_match_known_public_export_path() -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )

    assert activation_batch_number(feature_ref.index) == 26
    candidate_paths = candidate_cached_activation_batch_paths(feature_ref, cache_dir=Path("/tmp/np-cache"))
    assert public_activation_batch_url(feature_ref).endswith(
        "/v1/gemma-3-4b-it/23-gemmascope-2-transcoder-262k/activations/batch-26.jsonl.gz"
    )
    assert cached_activation_batch_path(feature_ref, cache_dir=Path("/tmp/np-cache")) == Path(
        "/tmp/np-cache/v1/gemma-3-4b-it/23-gemmascope-2-transcoder-262k/activations/batch-26.jsonl.gz"
    )
    assert candidate_paths[1] == Path(
        "/tmp/np-cache/v1/gemma-3-4b-it/23-gemmascope-2-transcoder-262k/activations/batch-52.jsonl.gz"
    )
    assert candidate_paths[-1] == Path(
        "/tmp/np-cache/v1/gemma-3-4b-it/23-gemmascope-2-transcoder-262k/activations/batch-13.jsonl.gz"
    )


def test_artifact_output_path_uses_timestamp_first_filename() -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )

    artifact_path = artifact_output_path(Path("/tmp/generated_np_explanations"), feature_ref)

    assert re.fullmatch(
        r"\d{8}_\d{6}_gemma-3-4b-it_gemmascope-2-transcoder-262k_23_13341\.md",
        artifact_path.name,
    )


def test_build_explanation_export_record_uses_generated_metadata() -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )

    record = build_explanation_export_record(
        feature_ref=feature_ref,
        cleaned_explanation="capital cities",
        artifact_path=Path("/tmp/generated_np_explanations/example.md"),
        copilot_model=DEFAULT_COPILOT_MODEL,
        cached_activations_path=Path("/tmp/cache/batch-26.jsonl.gz"),
    )

    assert record["modelId"] == "gemma-3-4b-it"
    assert record["layer"] == "23-gemmascope-2-transcoder-262k"
    assert record["index"] == "13341"
    assert record["description"] == "capital cities"
    assert record["typeName"] is None
    assert record["explanationModelName"] == DEFAULT_COPILOT_MODEL
    assert "artifact_path" in record["notes"]
    assert '"requested_type_name": "np_max-act-logits"' in record["notes"]


def test_build_explanation_export_record_preserves_requested_moe_type_in_notes() -> None:
    feature_ref = NeuronpediaFeatureRef(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )

    record = build_explanation_export_record(
        feature_ref=feature_ref,
        cleaned_explanation="capital cities",
        artifact_path=Path("/tmp/generated_np_explanations/example.md"),
        copilot_model=DEFAULT_COPILOT_MODEL,
        cached_activations_path=Path("/tmp/cache/batch-26.jsonl.gz"),
        type_name=np_explanations.NP_MOE_MAX_ACT_LOGITS_TYPE_NAME,
    )

    assert record["typeName"] is None
    assert '"requested_type_name": "np_moe-max-act-logits"' in record["notes"]


def test_insert_explanation_record_local_db_keeps_type_name_null_and_records_requested_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explanation_record = {
        "id": "explanation-id",
        "modelId": "gemma-3-4b-it",
        "layer": "23-gemmascope-2-transcoder-262k",
        "index": "13341",
        "description": "capital cities",
        "authorId": "author-id",
        "triggeredByUserId": None,
        "notes": "{}",
        "typeName": np_explanations.NP_MOE_MAX_ACT_LOGITS_TYPE_NAME,
        "explanationModelName": DEFAULT_COPILOT_MODEL,
    }
    executed: list[tuple[str, tuple[Any, ...]]] = []

    class _Cursor:
        def __init__(self) -> None:
            self._last_query = ""

        def execute(self, query: str, params: tuple[Any, ...]) -> None:
            self._last_query = query
            executed.append((query, params))

        def fetchone(self) -> tuple[Any, ...] | None:
            if 'FROM "ExplanationModelType"' in self._last_query:
                return (1,)
            if 'INSERT INTO "Explanation"' in self._last_query:
                return ("inserted-id",)
            return None

        def __enter__(self) -> _Cursor:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class _Connection:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def commit(self) -> None:
            return None

        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *_args, **_kwargs: _Connection()))

    inserted_id = np_explanations.insert_explanation_record_local_db(
        explanation_record,
        local_db_url="postgres://postgres:postgres@127.0.0.1:5433/postgres",
    )

    assert inserted_id == "inserted-id"
    insert_params = executed[-1][1]
    assert insert_params[8] is None
    assert '"requested_type_name": "np_moe-max-act-logits"' in insert_params[7]


def test_resolve_local_neuronpedia_db_url_rewrites_container_host_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST_PORT", "5433")

    resolved = resolve_local_neuronpedia_db_url(
        "postgresql://np_user:secret@postgres:5432/postgres",
        env={"POSTGRES_HOST_PORT": "5433"},
    )

    assert resolved == "postgresql://np_user:secret@127.0.0.1:5433/postgres"


@RunIf(optional=True)
def test_generate_explanation_artifact_with_cached_real_activation_batch(tmp_path: Path, monkeypatch) -> None:
    if shutil.which("copilot") is None:
        pytest.skip("copilot CLI is not available on PATH")

    feature_ref = build_feature_ref(
        model_id="gemma-3-4b-it",
        layer="23-gemmascope-2-transcoder-262k",
        index="13341",
    )
    cached_batch_path = cached_activation_batch_path(feature_ref, cache_dir=default_np_cache_dir())
    if not cached_batch_path.exists():
        pytest.skip(f"cached activation batch not available: {cached_batch_path}")

    def _fail_if_downloaded(*args, **kwargs):
        raise AssertionError("integration test should use the existing cached activation batch")

    monkeypatch.setattr(np_explanations, "download_url_to_path", _fail_if_downloaded)

    artifact = generate_explanation_artifact(
        feature_ref=feature_ref,
        output_dir=tmp_path,
        timeout_seconds=180,
    )

    assert artifact.cached_activations_path == cached_batch_path
    assert re.fullmatch(
        r"\d{8}_\d{6}_gemma-3-4b-it_gemmascope-2-transcoder-262k_23_13341\.md",
        artifact.artifact_path.name,
    )

    content = artifact.artifact_path.read_text(encoding="utf-8")
    assert "artifact_type: neuronpedia_explanation_candidate" in content
    assert f"copilot_model: {DEFAULT_COPILOT_MODEL}" in content
    assert f"cached_activations_path: {cached_batch_path}" in content
    assert "# Neuronpedia Explanation Candidate" in content
    assert re.search(r"^Method:\s*.+$", content, flags=re.MULTILINE)
    reason_match = re.search(r"^Reason:\s*(.+)$", content, flags=re.MULTILINE)
    assert reason_match is not None
    assert re.search(r"\b(capital|capitals|city)\b", reason_match.group(1), flags=re.IGNORECASE)
    explanation_match = re.search(r"^Explanation:\s*(.+)$", content, flags=re.MULTILINE)
    assert explanation_match is not None
    assert re.search(r"\b(capital|capitals|city)\b", explanation_match.group(1), flags=re.IGNORECASE)
    assert artifact.cleaned_explanation
