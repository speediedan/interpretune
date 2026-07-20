from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse, urlunparse
from urllib.request import Request, urlopen
from uuid import uuid4

DEFAULT_NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org"
DEFAULT_NEURONPEDIA_PUBLIC_DATASET_BASE_URL = "https://neuronpedia-datasets.s3.amazonaws.com"
DEFAULT_TOP_ACTIVATIONS_LIMIT = 10
DEFAULT_TOKENS_AROUND_MAX = 24
DEFAULT_COPILOT_TIMEOUT_SECONDS = 120
DEFAULT_COPILOT_MODEL = "gpt-5-mini"
DEFAULT_ACTIVATION_BATCH_SIZE = 512
FALLBACK_ACTIVATION_BATCH_SIZES = (DEFAULT_ACTIVATION_BATCH_SIZE, 1024)
DEFAULT_GENERATED_OUTPUT_DIR = Path(tempfile.gettempdir()) / "generated_np_explanations"
DEFAULT_HF_CACHE_HOME = Path(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
DEFAULT_IT_NP_CACHE = Path(os.getenv("IT_NP_CACHE", os.path.join(DEFAULT_HF_CACHE_HOME, "interpretune", "neuronpedia")))
DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL = os.getenv("LOCAL_NEURONPEDIA_WEBAPP_URL", "http://localhost:3000")
DEFAULT_LOCAL_NEURONPEDIA_DB_ENV_VARS = (
    "LOCAL_NEURONPEDIA_DB_URL",
    "POSTGRES_URL_NON_POOLING",
    "DATABASE_URL",
)
DEFAULT_LOCAL_NEURONPEDIA_DOCKER_HOSTNAME = "postgres"
DEFAULT_LOCAL_NEURONPEDIA_HOST = "127.0.0.1"
DEFAULT_LOCAL_NEURONPEDIA_POSTGRES_PORT_ENV = "POSTGRES_HOST_PORT"
DEFAULT_LOCAL_NEURONPEDIA_DB_TIMEOUT_SECONDS = 5
DEFAULT_EXPLANATION_TYPE_NAME = "np_max-act-logits"
NP_MOE_MAX_ACT_LOGITS_TYPE_NAME = "np_moe-max-act-logits"
DEFAULT_EXPLANATION_AUTHOR_ID = os.getenv("DEFAULT_CREATOR_ID", "clkht01d40000jv08hvalcvly")
DEFAULT_COPILOT_MAX_RETRIES = 2
DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS = 5.0

NP_MAX_ACT_LOGITS_INSTRUCTIONS = """You are generating a Neuronpedia-style explanation for a sparse feature.

Return a concise explanation of what the feature detects or predicts.

Use the following ordered method exactly:
Method 1: inspect MAX_ACTIVATING_TOKENS. If they share a specific token-level concept, use that.
Method 2: inspect TOKENS_AFTER_MAX_ACTIVATING_TOKEN.
If they share a clear continuation pattern, answer as say [pattern].
Method 3: inspect TOP_POSITIVE_LOGITS. If they strongly cluster around a semantic family, use that.
Method 4: inspect TOP_ACTIVATING_TEXTS and make the best concise semantic guess.

Rules:
- Keep the final explanation extremely concise: 1-6 words, usually 1-3.
- Do not say words like token, pattern, concept, or related to.
- If Method 2 is the first good method, the answer must start with say.
- If multiple methods work, prefer the earliest method.
- Do not use markdown code fences.

Respond in this format:
Method: <method number>
Reason: <one short sentence>
Explanation: <final explanation>"""

NP_MOE_MAX_ACT_LOGITS_INSTRUCTIONS = """You are generating a Neuronpedia-style explanation for a sparse feature.

Return a concise explanation of what the feature detects or predicts.

Use all four evidence sources together as a mixture-of-experts judgment:
- MAX_ACTIVATING_TOKENS
- TOKENS_AFTER_MAX_ACTIVATING_TOKEN
- TOP_POSITIVE_LOGITS
- TOP_ACTIVATING_TEXTS

Do not pick a single earliest method. Combine the strongest evidence across all sources and produce the single best
concise explanation.

Rules:
- Keep the final explanation extremely concise: 1-6 words, usually 1-3.
- Do not say words like token, pattern, concept, or related to.
- Do not use markdown code fences.
- Always return Method: -1 to indicate the combined MoE judgment.

Respond in this format:
Method: -1
Reason: <one short sentence combining the strongest evidence>
Explanation: <final explanation>"""

NP_MAX_ACT_LOGITS_FEW_SHOTS = (
    {
        "title": "Neuron A",
        "tokens_after": ["was", "watching"],
        "max_tokens": ["She", "enjoy"],
        "top_logits": ["walking", "WA", "waiting", "was", "we", "WHAM", "wish", "win", "wake", "whisper"],
        "top_texts": [
            "She was taking a nap when her phone started ringing.",
            "I enjoy watching movies with my family.",
        ],
        "assistant": (
            'Method: 2\nReason: The continuation tokens all start with the letter w.\nExplanation: say "w" words'
        ),
    },
    {
        "title": "Neuron B",
        "tokens_after": ["are", ","],
        "max_tokens": ["banana", "blueberries"],
        "top_logits": [
            "apple",
            "orange",
            "pineapple",
            "watermelon",
            "kiwi",
            "peach",
            "pear",
            "grape",
            "cherry",
            "plum",
        ],
        "top_texts": [
            "The apple and banana are delicious foods that provide essential vitamins and nutrients.",
            "I enjoy eating fresh strawberries, blueberries, and mangoes during the summer months.",
        ],
        "assistant": "Method: 1\nReason: The max-activating tokens are both fruits.\nExplanation: fruits",
    },
    {
        "title": "Neuron C",
        "tokens_after": ["warm", "the"],
        "max_tokens": ["and", "And"],
        "top_logits": [
            "elephant",
            "guitar",
            "mountain",
            "bicycle",
            "ocean",
            "telescope",
            "candle",
            "umbrella",
            "tornado",
            "butterfly",
        ],
        "top_texts": [
            "It was a beautiful day outside with clear skies and warm sunshine.",
            "And the garden has roses and tulips and daisies and sunflowers blooming together.",
        ],
        "assistant": "Method: 1\nReason: The max-activating tokens are both the word and.\nExplanation: and",
    },
    {
        "title": "Neuron D",
        "tokens_after": ["was", "places"],
        "max_tokens": ["war", "some"],
        "top_logits": ["4", "four", "fourth", "4th", "IV", "Four", "FOUR", "~4", "4.0", "quartet"],
        "top_texts": [
            "the civil war was a major topic in history class .",
            "seasons of the year are winter , spring , summer , and fall or autumn in some places .",
        ],
        "assistant": "Method: 3\nReason: The positive logits are all variants of the number four.\nExplanation: 4",
    },
)

NP_MOE_MAX_ACT_LOGITS_FEW_SHOTS = (
    {
        "title": "Neuron A",
        "tokens_after": ["was", "watching"],
        "max_tokens": ["She", "enjoy"],
        "top_logits": ["walking", "WA", "waiting", "was", "we", "WHAM", "wish", "win", "wake", "whisper"],
        "top_texts": [
            "She was taking a nap when her phone started ringing.",
            "I enjoy watching movies with my family.",
        ],
        "assistant": (
            "Method: -1\nReason: The continuation tokens and activating texts both point to words or phrases "
            "beginning with w.\n"
            'Explanation: say "w" words'
        ),
    },
    {
        "title": "Neuron B",
        "tokens_after": ["are", ","],
        "max_tokens": ["banana", "blueberries"],
        "top_logits": [
            "apple",
            "orange",
            "pineapple",
            "watermelon",
            "kiwi",
            "peach",
            "pear",
            "grape",
            "cherry",
            "plum",
        ],
        "top_texts": [
            "The apple and banana are delicious foods that provide essential vitamins and nutrients.",
            "I enjoy eating fresh strawberries, blueberries, and mangoes during the summer months.",
        ],
        "assistant": (
            "Method: -1\nReason: The activating tokens, positive logits, and texts all cluster around fruit names.\n"
            "Explanation: fruits"
        ),
    },
    {
        "title": "Neuron C",
        "tokens_after": ["warm", "the"],
        "max_tokens": ["and", "And"],
        "top_logits": [
            "elephant",
            "guitar",
            "mountain",
            "bicycle",
            "ocean",
            "telescope",
            "candle",
            "umbrella",
            "tornado",
            "butterfly",
        ],
        "top_texts": [
            "It was a beautiful day outside with clear skies and warm sunshine.",
            "And the garden has roses and tulips and daisies and sunflowers blooming together.",
        ],
        "assistant": (
            "Method: -1\nReason: The max-activating tokens consistently identify the conjunction and, "
            "which dominates the evidence.\nExplanation: and"
        ),
    },
    {
        "title": "Neuron D",
        "tokens_after": ["was", "places"],
        "max_tokens": ["war", "some"],
        "top_logits": ["4", "four", "fourth", "4th", "IV", "Four", "FOUR", "~4", "4.0", "quartet"],
        "top_texts": [
            "the civil war was a major topic in history class .",
            "seasons of the year are winter , spring , summer , and fall or autumn in some places .",
        ],
        "assistant": (
            "Method: -1\nReason: The strongest combined signal is the dense cluster of positive logits pointing "
            "to the number four.\nExplanation: 4"
        ),
    },
)

EXPLANATION_PROMPT_SPECS: dict[str, tuple[str, tuple[dict[str, Any], ...], str]] = {
    DEFAULT_EXPLANATION_TYPE_NAME: (
        NP_MAX_ACT_LOGITS_INSTRUCTIONS,
        NP_MAX_ACT_LOGITS_FEW_SHOTS,
        DEFAULT_EXPLANATION_TYPE_NAME,
    ),
    NP_MOE_MAX_ACT_LOGITS_TYPE_NAME: (
        NP_MOE_MAX_ACT_LOGITS_INSTRUCTIONS,
        NP_MOE_MAX_ACT_LOGITS_FEW_SHOTS,
        NP_MOE_MAX_ACT_LOGITS_TYPE_NAME,
    ),
}


@dataclass(frozen=True)
class NeuronpediaFeatureRef:
    """Identifies a Neuronpedia feature page and its API endpoint."""

    model_id: str
    layer: str
    index: str
    base_url: str = DEFAULT_NEURONPEDIA_BASE_URL

    @property
    def feature_url(self) -> str:
        return f"{self.base_url}/{self.model_id}/{self.layer}/{self.index}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/api/feature/{self.model_id}/{self.layer}/{self.index}"

    @property
    def layer_number(self) -> str:
        layer_number, _ = split_layer_identifier(self.layer)
        return layer_number

    @property
    def source_set(self) -> str:
        _, source_set = split_layer_identifier(self.layer)
        return source_set

    @property
    def artifact_slug(self) -> str:
        return "_".join(
            [
                slugify(self.model_id),
                slugify(self.source_set),
                slugify(self.layer_number),
                slugify(self.index),
            ]
        )


@dataclass(frozen=True)
class NeuronpediaPromptInputs:
    """Derived prompt inputs for the Neuronpedia-style explanation flow."""

    tokens_after_max_activating_token: list[str]
    max_activating_tokens: list[str]
    top_positive_logits: list[str]
    top_activating_texts: list[str]


@dataclass(frozen=True)
class NeuronpediaExplanationImportArtifact:
    """Paths and record data for a Neuronpedia explanation import bundle."""

    export_root: Path
    explanation_batch_path: Path
    explanation_record: dict[str, Any]


@dataclass(frozen=True)
class NeuronpediaExplanationArtifact:
    """Result of a generated explanation artifact."""

    feature_ref: NeuronpediaFeatureRef
    prompt: str
    raw_response: str
    cleaned_explanation: str
    artifact_path: Path
    cached_activations_path: Path | None = None
    import_artifact: NeuronpediaExplanationImportArtifact | None = None
    database_explanation_id: str | None = None


@dataclass(frozen=True)
class CopilotInvocationResult:
    """Captures the Copilot CLI response payload."""

    stdout: str
    stderr: str


class NeuronpediaExplanationError(RuntimeError):
    """Raised when explanation artifact generation fails."""


@dataclass(frozen=True)
class LocalNeuronpediaServiceStatus:
    """Availability snapshot for the local Neuronpedia webapp and Postgres services."""

    webapp_url: str
    webapp_available: bool
    webapp_status_code: int | None
    webapp_error: str | None
    db_url_redacted: str | None
    db_available: bool
    db_error: str | None


@dataclass(frozen=True)
class NeuronpediaLocalImportSummary:
    """Summary of a local Neuronpedia export-bundle import run."""

    export_root: Path
    imported_row_counts: dict[str, int]
    processed_files: list[Path]


@dataclass(frozen=True)
class NeuronpediaLocalExplanationStatus:
    """Local explanation availability for one Neuronpedia feature route."""

    feature_ref: NeuronpediaFeatureRef
    explanation_count: int

    @property
    def has_local_explanation(self) -> bool:
        return self.explanation_count > 0


@dataclass(frozen=True)
class NeuronpediaExplanationGenerationFailure:
    """Captures a non-fatal local explanation generation failure for one feature."""

    feature_ref: NeuronpediaFeatureRef
    error: str


@dataclass(frozen=True)
class NeuronpediaLocalExplanationCoverage:
    """Coverage summary for local explanation availability and optional backfill."""

    statuses: list[NeuronpediaLocalExplanationStatus]
    generated_artifacts: list[NeuronpediaExplanationArtifact]
    generation_failures: list[NeuronpediaExplanationGenerationFailure]

    @property
    def missing_feature_refs(self) -> list[NeuronpediaFeatureRef]:
        return [status.feature_ref for status in self.statuses if not status.has_local_explanation]


def _redact_connection_url(connection_url: str) -> str:
    parsed = urlparse(connection_url)
    if parsed.password is None:
        return connection_url
    username = parsed.username or ""
    netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@", 1)
    if username and not netloc.startswith(username):
        netloc = f"{username}:***@{parsed.hostname or ''}"
        if parsed.port is not None:
            netloc += f":{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def rewrite_container_postgres_url_for_host(
    connection_url: str,
    *,
    env: Mapping[str, str] | None = None,
    container_hostname: str = DEFAULT_LOCAL_NEURONPEDIA_DOCKER_HOSTNAME,
    host: str = DEFAULT_LOCAL_NEURONPEDIA_HOST,
    port_env_var: str = DEFAULT_LOCAL_NEURONPEDIA_POSTGRES_PORT_ENV,
) -> str:
    """Rewrite a docker-only Postgres URL to the host-mapped local port when configured."""

    env_map = dict(os.environ if env is None else env)
    parsed = urlparse(connection_url)
    if parsed.hostname != container_hostname:
        return connection_url
    host_port = env_map.get(port_env_var)
    if not host_port:
        return connection_url
    username = parsed.username or ""
    password = parsed.password or ""
    auth_prefix = username
    if password:
        auth_prefix = f"{auth_prefix}:{password}"
    if auth_prefix:
        auth_prefix = f"{auth_prefix}@"
    netloc = f"{auth_prefix}{host}:{host_port}"
    return urlunparse(parsed._replace(netloc=netloc))


def resolve_local_neuronpedia_db_url(
    local_db_url: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the best local Neuronpedia Postgres URL from explicit input or environment."""

    env_map = dict(os.environ if env is None else env)
    candidate = local_db_url
    if not candidate:
        for env_var in DEFAULT_LOCAL_NEURONPEDIA_DB_ENV_VARS:
            candidate = env_map.get(env_var)
            if candidate:
                break
    if not candidate:
        raise NeuronpediaExplanationError(
            "Could not resolve a local Neuronpedia DB URL. Set LOCAL_NEURONPEDIA_DB_URL or POSTGRES_URL_NON_POOLING."
        )
    return rewrite_container_postgres_url_for_host(candidate, env=env_map)


def check_local_neuronpedia_services(
    *,
    local_db_url: str | None = None,
    webapp_url: str = DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL,
    timeout_seconds: int = DEFAULT_LOCAL_NEURONPEDIA_DB_TIMEOUT_SECONDS,
) -> LocalNeuronpediaServiceStatus:
    """Probe the local Neuronpedia webapp and Postgres services without raising on failure."""

    import psycopg

    resolved_db_url: str | None = None
    db_url_redacted: str | None = None
    db_available = False
    db_error: str | None = None
    try:
        resolved_db_url = resolve_local_neuronpedia_db_url(local_db_url)
        db_url_redacted = _redact_connection_url(resolved_db_url)
        with psycopg.connect(resolved_db_url, connect_timeout=timeout_seconds) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        db_available = True
    except Exception as exc:  # pragma: no cover - exercised in integration contexts
        db_error = str(exc)

    webapp_available = False
    webapp_status_code: int | None = None
    webapp_error: str | None = None
    try:
        request = Request(webapp_url, method="GET")
        with urlopen(request, timeout=timeout_seconds) as response:
            webapp_status_code = getattr(response, "status", None)
            webapp_available = webapp_status_code is not None and 200 <= webapp_status_code < 500
    except Exception as exc:  # pragma: no cover - exercised in integration contexts
        webapp_error = str(exc)

    return LocalNeuronpediaServiceStatus(
        webapp_url=webapp_url,
        webapp_available=webapp_available,
        webapp_status_code=webapp_status_code,
        webapp_error=webapp_error,
        db_url_redacted=db_url_redacted,
        db_available=db_available,
        db_error=db_error,
    )


def normalize_base_url(base_url: str) -> str:
    """Normalize a Neuronpedia base URL by stripping any trailing slash."""

    return base_url.rstrip("/")


def split_layer_identifier(layer: str) -> tuple[str, str]:
    """Split a Neuronpedia layer identifier into the layer number and source set."""

    if "-" not in layer:
        raise ValueError(f"Layer identifier must include a source-set suffix: {layer}")
    layer_number, source_set = layer.split("-", 1)
    return layer_number, source_set


def combine_layer_identifier(layer: str, source_set: str | None = None) -> str:
    """Combine a numeric layer and source set into the Neuronpedia layer identifier."""

    if source_set is None or layer.endswith(source_set):
        return layer
    if "-" in layer and not layer.isdigit():
        return layer
    return f"{layer}-{source_set}"


def slugify(value: str) -> str:
    """Create a filesystem-safe slug while preserving hyphens and underscores."""

    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def activation_batch_number(feature_index: int | str, batch_size: int = DEFAULT_ACTIVATION_BATCH_SIZE) -> int:
    """Return the Neuronpedia export batch number for a feature index."""

    return int(feature_index) // batch_size


def candidate_activation_batch_sizes(batch_size: int | None = None) -> tuple[int, ...]:
    """Return activation batch sizes to probe, preserving order and uniqueness."""

    configured_size = batch_size or DEFAULT_ACTIVATION_BATCH_SIZE
    ordered_sizes = (configured_size, *FALLBACK_ACTIVATION_BATCH_SIZES)
    return tuple(dict.fromkeys(ordered_sizes))


def parse_feature_url(feature_url: str) -> NeuronpediaFeatureRef:
    """Parse a Neuronpedia feature page URL or feature API URL into a feature reference."""

    parsed = urlparse(feature_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Feature URL must include a scheme and host: {feature_url}")

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) >= 5 and parts[0] == "api" and parts[1] == "feature":
        _, _, model_id, layer, index = parts[:5]
    elif len(parts) >= 3:
        model_id, layer, index = parts[:3]
    else:
        raise ValueError(f"Unrecognized Neuronpedia feature URL: {feature_url}")

    return NeuronpediaFeatureRef(
        model_id=model_id,
        layer=layer,
        index=index,
        base_url=normalize_base_url(f"{parsed.scheme}://{parsed.netloc}"),
    )


def build_feature_ref(
    *,
    feature_url: str | None = None,
    model_id: str | None = None,
    layer: str | None = None,
    source_set: str | None = None,
    index: str | int | None = None,
    base_url: str = DEFAULT_NEURONPEDIA_BASE_URL,
) -> NeuronpediaFeatureRef:
    """Build a feature reference from either a full feature URL or explicit metadata."""

    if feature_url:
        return parse_feature_url(feature_url)

    if model_id is None or layer is None or index is None:
        raise ValueError("Provide either feature_url or model_id, layer, and index.")

    return NeuronpediaFeatureRef(
        model_id=model_id,
        layer=combine_layer_identifier(layer, source_set),
        index=str(index),
        base_url=normalize_base_url(base_url),
    )


def feature_tuples_to_feature_refs(
    *,
    model_id: str,
    source_set: str,
    feature_tuples: Iterable[tuple[int, int] | tuple[str, str]],
    base_url: str = DEFAULT_NEURONPEDIA_BASE_URL,
) -> list[NeuronpediaFeatureRef]:
    """Convert `(layer, feature_index)` tuples into Neuronpedia feature refs."""

    feature_refs: list[NeuronpediaFeatureRef] = []
    for layer_num, feature_index in feature_tuples:
        feature_refs.append(
            build_feature_ref(
                model_id=model_id,
                layer=str(layer_num),
                source_set=source_set,
                index=str(feature_index),
                base_url=base_url,
            )
        )
    return feature_refs


def check_local_explanation_coverage(
    feature_refs: Iterable[NeuronpediaFeatureRef],
    *,
    local_db_url: str | None = None,
    type_name: str | None = None,
) -> list[NeuronpediaLocalExplanationStatus]:
    """Return local explanation counts for the provided feature refs."""

    import psycopg

    resolved_db_url = resolve_local_neuronpedia_db_url(local_db_url)
    refs = list(feature_refs)
    statuses: list[NeuronpediaLocalExplanationStatus] = []
    with psycopg.connect(resolved_db_url) as connection:
        with connection.cursor() as cursor:
            for feature_ref in refs:
                if type_name is None:
                    cursor.execute(
                        'SELECT COUNT(*) FROM "Explanation" WHERE "modelId" = %s AND "layer" = %s AND "index" = %s',
                        (feature_ref.model_id, feature_ref.layer, feature_ref.index),
                    )
                else:
                    cursor.execute(
                        (
                            'SELECT COUNT(*) FROM "Explanation" '
                            'WHERE "modelId" = %s AND "layer" = %s AND "index" = %s '
                            'AND ("typeName" = %s OR COALESCE("notes"::text, \'\') LIKE %s)'
                        ),
                        (
                            feature_ref.model_id,
                            feature_ref.layer,
                            feature_ref.index,
                            type_name,
                            f'%"requested_type_name": "{type_name}"%',
                        ),
                    )
                row = cursor.fetchone()
                count = int(row[0]) if row else 0
                statuses.append(
                    NeuronpediaLocalExplanationStatus(
                        feature_ref=feature_ref,
                        explanation_count=count,
                    )
                )
    return statuses


def ensure_local_feature_explanations(
    feature_refs: Iterable[NeuronpediaFeatureRef],
    *,
    generate_missing: bool = False,
    output_dir: Path = DEFAULT_GENERATED_OUTPUT_DIR,
    copilot_model: str | None = None,
    timeout_seconds: int = DEFAULT_COPILOT_TIMEOUT_SECONDS,
    cache_dir: Path | None = None,
    local_db_url: str | None = None,
    explanation_author_id: str = DEFAULT_EXPLANATION_AUTHOR_ID,
    triggered_by_user_id: str | None = None,
    type_name: str = DEFAULT_EXPLANATION_TYPE_NAME,
    explanation_model_name: str | None = None,
    max_retries: int = DEFAULT_COPILOT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS,
) -> NeuronpediaLocalExplanationCoverage:
    """Check local explanation coverage and optionally backfill missing entries."""

    resolved_db_url = resolve_local_neuronpedia_db_url(local_db_url)
    initial_statuses = check_local_explanation_coverage(
        feature_refs,
        local_db_url=resolved_db_url,
        type_name=type_name,
    )
    generated_artifacts: list[NeuronpediaExplanationArtifact] = []
    generation_failures: list[NeuronpediaExplanationGenerationFailure] = []
    if generate_missing:
        for status in initial_statuses:
            if status.has_local_explanation:
                continue
            try:
                generated_artifacts.append(
                    generate_explanation_artifact(
                        feature_ref=status.feature_ref,
                        output_dir=output_dir,
                        copilot_model=copilot_model,
                        timeout_seconds=timeout_seconds,
                        cache_dir=cache_dir,
                        write_neuronpedia_import_data=True,
                        insert_into_local_db=True,
                        local_db_url=resolved_db_url,
                        explanation_author_id=explanation_author_id,
                        triggered_by_user_id=triggered_by_user_id,
                        type_name=type_name,
                        explanation_model_name=explanation_model_name,
                        max_retries=max_retries,
                        retry_backoff_seconds=retry_backoff_seconds,
                    )
                )
            except Exception as exc:
                generation_failures.append(
                    NeuronpediaExplanationGenerationFailure(
                        feature_ref=status.feature_ref,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    final_statuses = check_local_explanation_coverage(
        feature_refs,
        local_db_url=resolved_db_url,
        type_name=type_name,
    )
    return NeuronpediaLocalExplanationCoverage(
        statuses=final_statuses,
        generated_artifacts=generated_artifacts,
        generation_failures=generation_failures,
    )


def default_np_cache_dir() -> Path:
    """Return the local Interpretune Neuronpedia cache directory."""

    return Path(os.getenv("IT_NP_CACHE", str(DEFAULT_IT_NP_CACHE)))


def cached_activation_batch_path(
    feature_ref: NeuronpediaFeatureRef,
    *,
    cache_dir: Path | None = None,
    batch_size: int = DEFAULT_ACTIVATION_BATCH_SIZE,
) -> Path:
    """Return the expected local cache path for a Neuronpedia activation batch."""

    root_dir = cache_dir or default_np_cache_dir()
    batch_number = activation_batch_number(feature_ref.index, batch_size=batch_size)
    return root_dir / "v1" / feature_ref.model_id / feature_ref.layer / "activations" / f"batch-{batch_number}.jsonl.gz"


def public_activation_batch_url(
    feature_ref: NeuronpediaFeatureRef,
    *,
    dataset_base_url: str = DEFAULT_NEURONPEDIA_PUBLIC_DATASET_BASE_URL,
    batch_size: int = DEFAULT_ACTIVATION_BATCH_SIZE,
) -> str:
    """Return the public S3 URL for a feature's activation batch export."""

    batch_number = activation_batch_number(feature_ref.index, batch_size=batch_size)
    dataset_root = normalize_base_url(dataset_base_url)
    return f"{dataset_root}/v1/{feature_ref.model_id}/{feature_ref.layer}/activations/batch-{batch_number}.jsonl.gz"


def candidate_cached_activation_batch_paths(
    feature_ref: NeuronpediaFeatureRef,
    *,
    cache_dir: Path | None = None,
    batch_size: int | None = None,
) -> list[Path]:
    """Return candidate local cache paths for known activation batch layouts."""

    return [
        cached_activation_batch_path(feature_ref, cache_dir=cache_dir, batch_size=candidate_size)
        for candidate_size in candidate_activation_batch_sizes(batch_size=batch_size)
    ]


def cached_feature_activation_path(
    feature_ref: NeuronpediaFeatureRef,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Return the feature-specific local cache path for activation rows."""

    root_dir = cache_dir or default_np_cache_dir()
    return (
        root_dir
        / "v1"
        / feature_ref.model_id
        / feature_ref.layer
        / "feature-activations"
        / f"feature-{feature_ref.index}.jsonl.gz"
    )


def write_cached_feature_activations(
    feature_ref: NeuronpediaFeatureRef,
    activation_rows: Iterable[dict[str, Any]],
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Write feature-specific activation rows to the local Interpretune Neuronpedia cache."""

    cache_path = cached_feature_activation_path(feature_ref, cache_dir=cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        for activation_row in activation_rows:
            handle.write(json.dumps(activation_row))
            handle.write("\n")
    return cache_path


def candidate_public_activation_batch_urls(
    feature_ref: NeuronpediaFeatureRef,
    *,
    dataset_base_url: str = DEFAULT_NEURONPEDIA_PUBLIC_DATASET_BASE_URL,
    batch_size: int | None = None,
) -> list[str]:
    """Return candidate public activation batch URLs for known batch layouts."""

    return [
        public_activation_batch_url(feature_ref, dataset_base_url=dataset_base_url, batch_size=candidate_size)
        for candidate_size in candidate_activation_batch_sizes(batch_size=batch_size)
    ]


def download_url_to_path(url: str, destination_path: Path, timeout_seconds: int = 60) -> Path:
    """Download a URL to a local path, creating parent directories as needed."""

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "interpretune-neuronpedia-explanations/0.2"})
    with urlopen(request, timeout=timeout_seconds) as response:
        destination_path.write_bytes(response.read())
    return destination_path


def ensure_cached_activation_batch(
    feature_ref: NeuronpediaFeatureRef,
    *,
    cache_dir: Path | None = None,
    timeout_seconds: int = 60,
    dataset_base_url: str = DEFAULT_NEURONPEDIA_PUBLIC_DATASET_BASE_URL,
    batch_size: int = DEFAULT_ACTIVATION_BATCH_SIZE,
) -> Path:
    """Ensure the activation batch for a feature exists in the local cache."""

    batch_paths = candidate_cached_activation_batch_paths(feature_ref, cache_dir=cache_dir, batch_size=batch_size)
    for batch_path in batch_paths:
        if batch_path.exists():
            return batch_path

    batch_urls = candidate_public_activation_batch_urls(
        feature_ref,
        dataset_base_url=dataset_base_url,
        batch_size=batch_size,
    )
    for batch_path, batch_url in zip(batch_paths, batch_urls, strict=True):
        try:
            return download_url_to_path(batch_url, batch_path, timeout_seconds=timeout_seconds)
        except Exception:
            continue

    raise NeuronpediaExplanationError(
        f"Could not download any activation batch candidate for {feature_ref.feature_url}."
    )


def fetch_feature_payload(feature_ref: NeuronpediaFeatureRef, timeout_seconds: int = 30) -> dict[str, Any]:
    """Fetch the full feature payload from Neuronpedia's public feature API."""

    request = Request(
        feature_ref.api_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "interpretune-neuronpedia-explanations/0.2",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def load_activation_batch_records(batch_path: Path) -> list[dict[str, Any]]:
    """Load a cached activation batch file exported by Neuronpedia."""

    open_fn = gzip.open if batch_path.suffix == ".gz" else open
    with open_fn(batch_path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cached_feature_activations(
    feature_ref: NeuronpediaFeatureRef,
    *,
    cache_dir: Path | None = None,
    timeout_seconds: int = 60,
) -> tuple[list[dict[str, Any]], Path]:
    """Load cached activation rows for one feature from the public export batch."""

    feature_cache_path = cached_feature_activation_path(feature_ref, cache_dir=cache_dir)
    if feature_cache_path.exists():
        activation_rows = load_activation_batch_records(feature_cache_path)
        matching_rows = [row for row in activation_rows if str(row.get("index")) == feature_ref.index]
        if matching_rows:
            return matching_rows, feature_cache_path

    candidate_paths = candidate_cached_activation_batch_paths(feature_ref, cache_dir=cache_dir)
    candidate_urls = candidate_public_activation_batch_urls(feature_ref)

    for batch_path in candidate_paths:
        if not batch_path.exists():
            continue
        activation_rows = load_activation_batch_records(batch_path)
        matching_rows = [row for row in activation_rows if str(row.get("index")) == feature_ref.index]
        if matching_rows:
            return matching_rows, batch_path

    for batch_path, batch_url in zip(candidate_paths, candidate_urls, strict=True):
        if batch_path.exists():
            continue
        try:
            download_url_to_path(batch_url, batch_path, timeout_seconds=timeout_seconds)
        except Exception:
            continue
        activation_rows = load_activation_batch_records(batch_path)
        matching_rows = [row for row in activation_rows if str(row.get("index")) == feature_ref.index]
        if matching_rows:
            return matching_rows, batch_path

    raise NeuronpediaExplanationError(
        f"Cached activation batches do not contain feature index {feature_ref.index} for {feature_ref.feature_url}."
    )


def load_feature_payload_with_cached_activations(
    feature_ref: NeuronpediaFeatureRef,
    *,
    cache_dir: Path | None = None,
    timeout_seconds: int = 60,
) -> tuple[dict[str, Any], Path]:
    """Fetch feature metadata from the API and replace activations with cached export rows."""

    feature_payload = fetch_feature_payload(feature_ref, timeout_seconds=timeout_seconds)
    cached_activations, batch_path = load_cached_feature_activations(
        feature_ref,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    feature_payload = dict(feature_payload)
    feature_payload["activations"] = cached_activations
    return feature_payload, batch_path


def _clean_token(token: str) -> str:
    return token.replace("\n", "").strip()


def _max_activation_index(activation: dict[str, Any]) -> int:
    token_index = activation.get("maxValueTokenIndex")
    if isinstance(token_index, int) and token_index >= 0:
        return token_index

    values = activation.get("values") or []
    if not values:
        return 0
    max_value = max(values)
    return values.index(max_value)


def dedupe_and_select_activations(
    activations: list[dict[str, Any]],
    limit: int = DEFAULT_TOP_ACTIVATIONS_LIMIT,
) -> list[dict[str, Any]]:
    """Mirror Neuronpedia's activation dedupe, sort, and top-k selection."""

    deduped: dict[str, dict[str, Any]] = {}
    for activation in activations:
        key = "".join(activation.get("tokens") or [])
        deduped[key] = activation

    unique_activations = list(deduped.values())
    unique_activations.sort(key=lambda activation: float(activation.get("maxValue") or 0.0), reverse=True)
    return unique_activations[:limit]


def derive_np_max_act_logits_inputs(
    feature_payload: dict[str, Any],
    *,
    top_activations_limit: int = DEFAULT_TOP_ACTIVATIONS_LIMIT,
    tokens_around_max: int = DEFAULT_TOKENS_AROUND_MAX,
) -> NeuronpediaPromptInputs:
    """Reconstruct the four Neuronpedia input lists used by np-max-act-logits."""

    activations = dedupe_and_select_activations(feature_payload.get("activations") or [], top_activations_limit)
    if not activations:
        raise NeuronpediaExplanationError("Feature payload does not contain any activation rows.")

    tokens_after_max: list[str] = []
    max_tokens: list[str] = []
    top_texts: list[str] = []

    for activation in activations:
        tokens = activation.get("tokens") or []
        values = activation.get("values") or []
        if not tokens or not values:
            continue

        max_index = _max_activation_index(activation)
        max_tokens.append(_clean_token(tokens[max_index]))

        if max_index + 1 < len(tokens):
            tokens_after_max.append(_clean_token(tokens[max_index + 1]))
        else:
            tokens_after_max.append("")

        start_index = max(0, max_index - tokens_around_max)
        end_index = min(len(tokens), max_index + tokens_around_max + 1)
        trimmed_text = "".join(tokens[start_index:end_index]).replace("\n", "  ")
        top_texts.append(trimmed_text)

    top_positive_logits = [
        _clean_token(str(logit).replace("▁", " ")) for logit in (feature_payload.get("pos_str") or [])
    ]
    top_positive_logits = [logit for logit in top_positive_logits if logit]

    return NeuronpediaPromptInputs(
        tokens_after_max_activating_token=tokens_after_max,
        max_activating_tokens=max_tokens,
        top_positive_logits=top_positive_logits,
        top_activating_texts=top_texts,
    )


def _format_list_block(items: list[str]) -> str:
    non_empty_items = [item for item in items if item != ""]
    return "\n".join(non_empty_items) if non_empty_items else "<empty>"


def _render_few_shot_example(example: dict[str, Any]) -> str:
    return "\n".join(
        [
            example["title"],
            "<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>",
            _format_list_block(example["tokens_after"]),
            "</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>",
            "<MAX_ACTIVATING_TOKENS>",
            _format_list_block(example["max_tokens"]),
            "</MAX_ACTIVATING_TOKENS>",
            "<TOP_POSITIVE_LOGITS>",
            _format_list_block(example["top_logits"]),
            "</TOP_POSITIVE_LOGITS>",
            "<TOP_ACTIVATING_TEXTS>",
            _format_list_block(example["top_texts"]),
            "</TOP_ACTIVATING_TEXTS>",
            example["assistant"],
        ]
    )


def resolve_explanation_prompt_spec(type_name: str) -> tuple[str, tuple[dict[str, Any], ...], str]:
    """Resolve the explanation prompt template for a supported explanation type."""

    try:
        return EXPLANATION_PROMPT_SPECS[type_name]
    except KeyError as exc:
        supported_types = ", ".join(sorted(EXPLANATION_PROMPT_SPECS))
        raise NeuronpediaExplanationError(
            f"Unsupported explanation type '{type_name}'. Supported types: {supported_types}."
        ) from exc


def build_explanation_prompt(
    feature_ref: NeuronpediaFeatureRef,
    prompt_inputs: NeuronpediaPromptInputs,
    *,
    type_name: str = DEFAULT_EXPLANATION_TYPE_NAME,
) -> tuple[str, str]:
    """Build a Copilot explanation prompt and return it with its prompt style name."""

    instructions, few_shot_examples, prompt_style = resolve_explanation_prompt_spec(type_name)
    few_shots = "\n\n".join(_render_few_shot_example(example) for example in few_shot_examples)
    target_block = "\n".join(
        [
            "Target feature",
            f"Feature URL: {feature_ref.feature_url}",
            f"Model ID: {feature_ref.model_id}",
            f"Layer: {feature_ref.layer_number}",
            f"Source set: {feature_ref.source_set}",
            f"Feature index: {feature_ref.index}",
            "<TOKENS_AFTER_MAX_ACTIVATING_TOKEN>",
            _format_list_block(prompt_inputs.tokens_after_max_activating_token),
            "</TOKENS_AFTER_MAX_ACTIVATING_TOKEN>",
            "<MAX_ACTIVATING_TOKENS>",
            _format_list_block(prompt_inputs.max_activating_tokens),
            "</MAX_ACTIVATING_TOKENS>",
            "<TOP_POSITIVE_LOGITS>",
            _format_list_block(prompt_inputs.top_positive_logits),
            "</TOP_POSITIVE_LOGITS>",
            "<TOP_ACTIVATING_TEXTS>",
            _format_list_block(prompt_inputs.top_activating_texts),
            "</TOP_ACTIVATING_TEXTS>",
        ]
    )
    prompt = f"{instructions}\n\nFew-shot examples:\n\n{few_shots}\n\n{target_block}"
    return prompt, prompt_style


def build_np_max_act_logits_prompt(
    feature_ref: NeuronpediaFeatureRef,
    prompt_inputs: NeuronpediaPromptInputs,
) -> str:
    """Build a Copilot prompt that mirrors Neuronpedia's np-max-act-logits workflow."""

    prompt, _ = build_explanation_prompt(feature_ref, prompt_inputs, type_name=DEFAULT_EXPLANATION_TYPE_NAME)
    return prompt


def clean_explanation_text(raw_response: str) -> str:
    """Extract the final explanation text from a Copilot CLI response."""

    cleaned = raw_response.strip().strip("`")
    matches = re.findall(r"Explanation:\s*(.+)", cleaned, flags=re.IGNORECASE)
    explanation = matches[-1] if matches else cleaned.splitlines()[-1]
    explanation = explanation.strip()
    while len(explanation) >= 2 and explanation[0] == explanation[-1] and explanation[0] in {'"', "'"}:
        explanation = explanation[1:-1].strip()
    explanation = explanation.rstrip(".")
    return explanation


def cleanup_feature_activation_cache(
    feature_ref: NeuronpediaFeatureRef,
    *,
    cache_dir: Path | None = None,
    cached_activations_path: Path | None = None,
) -> bool:
    """Delete a feature-specific activation cache file after a successful local DB insert."""

    feature_cache_path = cached_feature_activation_path(feature_ref, cache_dir=cache_dir)
    candidate_path = cached_activations_path or feature_cache_path
    if candidate_path.resolve(strict=False) != feature_cache_path.resolve(strict=False):
        return False
    if not candidate_path.exists():
        return False

    candidate_path.unlink(missing_ok=True)
    try:
        candidate_path.parent.rmdir()
    except OSError:
        pass
    return True


def invoke_copilot_cli(
    prompt: str,
    *,
    copilot_model: str | None = None,
    timeout_seconds: int = DEFAULT_COPILOT_TIMEOUT_SECONDS,
) -> CopilotInvocationResult:
    """Invoke the GitHub Copilot CLI prompt mode and return its raw response."""

    executable = shutil.which("copilot")
    if executable is None:
        raise NeuronpediaExplanationError("Could not find the 'copilot' CLI on PATH.")

    env = os.environ.copy()
    env["COPILOT_MODEL"] = copilot_model or DEFAULT_COPILOT_MODEL

    completed = subprocess.run(
        [executable, "-p", prompt],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise NeuronpediaExplanationError(f"Copilot CLI invocation failed: {detail}")

    return CopilotInvocationResult(stdout=completed.stdout.strip(), stderr=completed.stderr.strip())


def invoke_copilot_cli_with_retries(
    prompt: str,
    *,
    copilot_model: str | None = None,
    timeout_seconds: int = DEFAULT_COPILOT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_COPILOT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS,
) -> CopilotInvocationResult:
    """Invoke the Copilot CLI and retry transient timeout failures with exponential backoff."""

    resolved_max_retries = max(0, int(max_retries))
    resolved_backoff = max(0.0, float(retry_backoff_seconds))
    last_timeout: subprocess.TimeoutExpired | None = None

    for attempt in range(resolved_max_retries + 1):
        try:
            return invoke_copilot_cli(
                prompt,
                copilot_model=copilot_model,
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
            if attempt >= resolved_max_retries:
                break
            sleep_seconds = resolved_backoff * (2**attempt)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    attempts = resolved_max_retries + 1
    raise NeuronpediaExplanationError(
        f"Copilot CLI timed out after {attempts} attempt{'s' if attempts != 1 else ''}"
    ) from last_timeout


def artifact_output_path(output_dir: Path, feature_ref: NeuronpediaFeatureRef) -> Path:
    """Return the markdown artifact path using the requested timestamp-first naming scheme."""

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    file_name = "_".join(
        [
            timestamp,
            slugify(feature_ref.model_id),
            slugify(feature_ref.source_set),
            slugify(feature_ref.layer_number),
            slugify(feature_ref.index),
        ]
    )
    return output_dir / f"{file_name}.md"


def _serialize_notes(notes: dict[str, Any]) -> str:
    return json.dumps(notes, sort_keys=True)


def _with_requested_generation_metadata(
    notes: dict[str, Any],
    *,
    requested_type_name: str | None,
    requested_explanation_model_name: str | None,
) -> dict[str, Any]:
    if requested_type_name is not None:
        notes["requested_type_name"] = requested_type_name
    if requested_explanation_model_name is not None:
        notes["requested_explanation_model_name"] = requested_explanation_model_name
    return notes


def build_explanation_export_record(
    *,
    feature_ref: NeuronpediaFeatureRef,
    cleaned_explanation: str,
    artifact_path: Path,
    copilot_model: str,
    cached_activations_path: Path | None,
    author_id: str = DEFAULT_EXPLANATION_AUTHOR_ID,
    triggered_by_user_id: str | None = None,
    type_name: str = DEFAULT_EXPLANATION_TYPE_NAME,
    explanation_model_name: str | None = None,
) -> dict[str, Any]:
    """Build a Neuronpedia-compatible explanation row for import or direct DB insertion."""

    notes = _with_requested_generation_metadata(
        {
            "generated_by": "interpretune-github-copilot-cli",
            "copilot_model": copilot_model,
            "artifact_path": str(artifact_path),
            "cached_activations_path": str(cached_activations_path) if cached_activations_path else None,
        },
        requested_type_name=type_name,
        requested_explanation_model_name=explanation_model_name or copilot_model,
    )
    return {
        "id": str(uuid4()),
        "modelId": feature_ref.model_id,
        "layer": feature_ref.layer,
        "index": feature_ref.index,
        "authorId": author_id,
        "description": cleaned_explanation,
        "typeName": None,
        "explanationModelName": explanation_model_name or copilot_model,
        "triggeredByUserId": triggered_by_user_id,
        "notes": _serialize_notes(notes),
        "umap_x": 0,
        "umap_y": 0,
        "umap_cluster": 0,
        "umap_log_feature_sparsity": 0,
    }


def write_explanation_import_bundle(
    *,
    output_dir: Path,
    feature_ref: NeuronpediaFeatureRef,
    explanation_record: dict[str, Any],
) -> NeuronpediaExplanationImportArtifact:
    """Write a Neuronpedia explanation import bundle under a source-root layout."""

    export_root = output_dir / "neuronpedia_import" / feature_ref.model_id / feature_ref.layer
    explanations_dir = export_root / "explanations"
    explanations_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    batch_path = explanations_dir / f"copilot-{timestamp}.jsonl.gz"
    with gzip.open(batch_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(explanation_record))
        handle.write("\n")
    return NeuronpediaExplanationImportArtifact(
        export_root=export_root,
        explanation_batch_path=batch_path,
        explanation_record=explanation_record,
    )


def _resolve_supported_fk_name(cursor: Any, table_name: str, requested_name: str | None) -> str | None:
    if requested_name is None:
        return None
    cursor.execute(f'SELECT 1 FROM "{table_name}" WHERE "name" = %s LIMIT 1', (requested_name,))
    return requested_name if cursor.fetchone() else None


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _export_bundle_table_paths(export_root: Path) -> list[tuple[str, Path]]:
    table_paths: list[tuple[str, Path]] = []
    metadata_file_map = (
        ("SourceRelease", export_root / "release.jsonl"),
        ("Model", export_root / "model.jsonl"),
        ("SourceSet", export_root / "sourceset.jsonl"),
        ("Source", export_root / "source.jsonl"),
    )
    for table_name, path in metadata_file_map:
        if path.exists():
            table_paths.append((table_name, path))
    for dir_name, table_name in (
        ("features", "Neuron"),
        ("activations", "Activation"),
        ("explanations", "Explanation"),
    ):
        directory = export_root / dir_name
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl.gz")):
            table_paths.append((table_name, path))
    return table_paths


def _resolve_table_column_defs(cursor: Any, table_name: str, available_columns: Iterable[str]) -> list[tuple[str, str]]:
    column_names = list(dict.fromkeys(available_columns))
    cursor.execute(
        """
        SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s
          AND n.nspname = current_schema()
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND a.attname = ANY(%s)
        ORDER BY a.attnum
        """,
        (table_name, column_names),
    )
    return [(row[0], row[1]) for row in cursor.fetchall()]


def _import_records_local_db(
    connection: Any,
    table_name: str,
    records: list[dict[str, Any]],
    *,
    chunk_size: int = 65000,
) -> int:
    if not records:
        return 0

    import psycopg.sql as sql

    available_columns: list[str] = []
    for record in records:
        for key in record:
            if key not in available_columns:
                available_columns.append(key)

    with connection.cursor() as cursor:
        column_defs = _resolve_table_column_defs(cursor, table_name, available_columns)
        if not column_defs:
            raise NeuronpediaExplanationError(f"Could not resolve columns for table {table_name}.")
        column_names = [column_name for column_name, _ in column_defs]
        query = sql.SQL(
            "INSERT INTO {table_name} ({column_list}) "
            "SELECT {column_list} FROM jsonb_to_recordset(%s::jsonb) AS t({column_defs}) "
            "ON CONFLICT DO NOTHING"
        ).format(
            table_name=sql.Identifier(table_name),
            column_list=sql.SQL(", ").join(sql.Identifier(column_name) for column_name in column_names),
            column_defs=sql.SQL(", ").join(
                sql.SQL("{} {}").format(sql.Identifier(column_name), sql.SQL(cast(Any, column_type)))
                for column_name, column_type in column_defs
            ),
        )

        imported_rows = 0
        for start_idx in range(0, len(records), chunk_size):
            chunk = records[start_idx : start_idx + chunk_size]
            payload = json.dumps(chunk).replace("\\u0000", " ")
            cursor.execute(query, (payload,))
            if cursor.rowcount > 0:
                imported_rows += cursor.rowcount
    return imported_rows


def import_neuronpedia_export_bundle_local_db(
    export_root: Path | str,
    *,
    local_db_url: str,
) -> NeuronpediaLocalImportSummary:
    """Import a Neuronpedia export bundle into a local Neuronpedia Postgres database."""

    import psycopg

    export_root_path = Path(export_root)
    if not export_root_path.exists():
        raise NeuronpediaExplanationError(f"Export root does not exist: {export_root_path}")

    table_paths = _export_bundle_table_paths(export_root_path)
    if not table_paths:
        raise NeuronpediaExplanationError(f"No importable Neuronpedia bundle files found under {export_root_path}")

    imported_row_counts: dict[str, int] = {}
    processed_files: list[Path] = []
    resolved_db_url = resolve_local_neuronpedia_db_url(local_db_url)
    with psycopg.connect(resolved_db_url) as connection:
        for table_name, path in table_paths:
            records = _load_jsonl_records(path)
            imported_count = _import_records_local_db(connection, table_name, records)
            imported_row_counts[table_name] = imported_row_counts.get(table_name, 0) + imported_count
            processed_files.append(path)
        connection.commit()

    return NeuronpediaLocalImportSummary(
        export_root=export_root_path,
        imported_row_counts=imported_row_counts,
        processed_files=processed_files,
    )


def insert_explanation_record_local_db(
    explanation_record: dict[str, Any],
    *,
    local_db_url: str,
) -> str:
    """Insert a generated explanation row into a local Neuronpedia Postgres database."""

    import psycopg

    record = dict(explanation_record)
    with psycopg.connect(local_db_url) as connection:
        with connection.cursor() as cursor:
            resolved_model_name = _resolve_supported_fk_name(
                cursor,
                "ExplanationModelType",
                record.get("explanationModelName"),
            )
            notes = _with_requested_generation_metadata(
                json.loads(record.get("notes") or "{}"),
                requested_type_name=record.get("typeName"),
                requested_explanation_model_name=(
                    None
                    if resolved_model_name == record.get("explanationModelName")
                    else record.get("explanationModelName")
                ),
            )
            record["notes"] = _serialize_notes(notes)
            record["typeName"] = None
            record["explanationModelName"] = resolved_model_name

            columns = [
                "id",
                "modelId",
                "layer",
                "index",
                "description",
                "authorId",
                "triggeredByUserId",
                "notes",
                "typeName",
                "explanationModelName",
            ]
            cursor.execute(
                'INSERT INTO "Explanation" ('
                '"id", "modelId", "layer", "index", "description", "authorId", '
                '"triggeredByUserId", "notes", "typeName", "explanationModelName"'
                ') VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING "id"',
                tuple(record.get(column) for column in columns),
            )
            inserted_row = cursor.fetchone()
            if inserted_row is None:
                raise NeuronpediaExplanationError("Local DB insertion did not return an explanation id.")
            inserted_id = inserted_row[0]
        connection.commit()
    return str(inserted_id)


def render_markdown_artifact(
    *,
    feature_ref: NeuronpediaFeatureRef,
    prompt_inputs: NeuronpediaPromptInputs,
    prompt: str,
    raw_response: str,
    cleaned_explanation: str,
    copilot_model: str,
    prompt_style: str,
    cached_activations_path: Path | None,
    import_artifact: NeuronpediaExplanationImportArtifact | None,
    database_explanation_id: str | None,
) -> str:
    """Render a markdown artifact suitable for explanation review or ingestion."""

    generated_at = datetime.now(tz=UTC).isoformat()
    return "\n".join(
        [
            "---",
            "artifact_type: neuronpedia_explanation_candidate",
            f"generated_at: {generated_at}",
            f"feature_url: {feature_ref.feature_url}",
            f"api_url: {feature_ref.api_url}",
            f"model_id: {feature_ref.model_id}",
            f"layer: {feature_ref.layer}",
            f"index: {feature_ref.index}",
            f"source_set: {feature_ref.source_set}",
            f"prompt_style: {prompt_style}",
            f"copilot_model: {copilot_model}",
            f"cached_activations_path: {cached_activations_path or ''}",
            f"neuronpedia_import_batch: {import_artifact.explanation_batch_path if import_artifact else ''}",
            f"local_db_explanation_id: {database_explanation_id or ''}",
            "---",
            "",
            "# Neuronpedia Explanation Candidate",
            "",
            "## Candidate",
            "",
            cleaned_explanation,
            "",
            "## Prompt Inputs",
            "",
            "### MAX_ACTIVATING_TOKENS",
            "",
            _format_list_block(prompt_inputs.max_activating_tokens),
            "",
            "### TOKENS_AFTER_MAX_ACTIVATING_TOKEN",
            "",
            _format_list_block(prompt_inputs.tokens_after_max_activating_token),
            "",
            "### TOP_POSITIVE_LOGITS",
            "",
            _format_list_block(prompt_inputs.top_positive_logits),
            "",
            "### TOP_ACTIVATING_TEXTS",
            "",
            _format_list_block(prompt_inputs.top_activating_texts),
            "",
            "## Raw Copilot Output",
            "",
            raw_response,
            "",
            "## Copilot Prompt",
            "",
            prompt,
            "",
        ]
    )


def generate_explanation_artifact(
    *,
    feature_ref: NeuronpediaFeatureRef,
    output_dir: Path = DEFAULT_GENERATED_OUTPUT_DIR,
    copilot_model: str | None = None,
    timeout_seconds: int = DEFAULT_COPILOT_TIMEOUT_SECONDS,
    cache_dir: Path | None = None,
    write_neuronpedia_import_data: bool = False,
    insert_into_local_db: bool = False,
    local_db_url: str | None = None,
    explanation_author_id: str = DEFAULT_EXPLANATION_AUTHOR_ID,
    triggered_by_user_id: str | None = None,
    type_name: str = DEFAULT_EXPLANATION_TYPE_NAME,
    explanation_model_name: str | None = None,
    max_retries: int = DEFAULT_COPILOT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS,
) -> NeuronpediaExplanationArtifact:
    """Generate a Neuronpedia-style explanation artifact using cached activations and GitHub Copilot CLI."""

    selected_model = copilot_model or DEFAULT_COPILOT_MODEL
    feature_payload, cached_activations_path = load_feature_payload_with_cached_activations(
        feature_ref,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    prompt_inputs = derive_np_max_act_logits_inputs(feature_payload)
    prompt, prompt_style = build_explanation_prompt(feature_ref, prompt_inputs, type_name=type_name)
    copilot_result = invoke_copilot_cli_with_retries(
        prompt,
        copilot_model=selected_model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    cleaned_explanation = clean_explanation_text(copilot_result.stdout)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_output_path(output_dir, feature_ref)
    explanation_record: dict[str, Any] | None = None
    import_artifact: NeuronpediaExplanationImportArtifact | None = None
    database_explanation_id: str | None = None

    if write_neuronpedia_import_data or insert_into_local_db:
        explanation_record = build_explanation_export_record(
            feature_ref=feature_ref,
            cleaned_explanation=cleaned_explanation,
            artifact_path=artifact_path,
            copilot_model=selected_model,
            cached_activations_path=cached_activations_path,
            author_id=explanation_author_id,
            triggered_by_user_id=triggered_by_user_id,
            type_name=type_name,
            explanation_model_name=explanation_model_name,
        )
    if write_neuronpedia_import_data and explanation_record is not None:
        import_artifact = write_explanation_import_bundle(
            output_dir=output_dir,
            feature_ref=feature_ref,
            explanation_record=explanation_record,
        )
    if insert_into_local_db:
        if local_db_url is None:
            raise NeuronpediaExplanationError("A local_db_url is required when insert_into_local_db is enabled.")
        if explanation_record is None:
            raise NeuronpediaExplanationError("Expected an explanation record before local DB insertion.")
        database_explanation_id = insert_explanation_record_local_db(explanation_record, local_db_url=local_db_url)
        cleanup_feature_activation_cache(
            feature_ref,
            cache_dir=cache_dir,
            cached_activations_path=cached_activations_path,
        )

    artifact_path.write_text(
        render_markdown_artifact(
            feature_ref=feature_ref,
            prompt_inputs=prompt_inputs,
            prompt=prompt,
            raw_response=copilot_result.stdout,
            cleaned_explanation=cleaned_explanation,
            copilot_model=selected_model,
            prompt_style=prompt_style,
            cached_activations_path=cached_activations_path,
            import_artifact=import_artifact,
            database_explanation_id=database_explanation_id,
        ),
        encoding="utf-8",
    )
    return NeuronpediaExplanationArtifact(
        feature_ref=feature_ref,
        prompt=prompt,
        raw_response=copilot_result.stdout,
        cleaned_explanation=cleaned_explanation,
        artifact_path=artifact_path,
        cached_activations_path=cached_activations_path,
        import_artifact=import_artifact,
        database_explanation_id=database_explanation_id,
    )
