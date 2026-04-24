from __future__ import annotations

import gzip
import hashlib
import html
import json
import os
import re
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.request import Request, urlopen

import psycopg
import requests
import torch
from dotenv import load_dotenv

import interpretune as it
from interpretune.analysis.ops.helpers import (
    FeatureSelectionSpec,
    apply_feature_selection_filter,
)
from interpretune.config import AnalysisCfg, init_analysis_cfgs
from interpretune.utils import (
    DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL,
    DEFAULT_NEURONPEDIA_BASE_URL,
    check_local_explanation_coverage,
    clean_explanation_text,
    default_np_cache_dir,
    fetch_feature_payload,
    feature_tuples_to_feature_refs,
    parse_feature_url,
    resolve_local_neuronpedia_db_url,
)
from interpretune.utils.neuronpedia_explanations import (
    NeuronpediaFeatureRef,
    NeuronpediaLocalExplanationStatus,
    cached_feature_activation_path,
    candidate_cached_activation_batch_paths,
    load_activation_batch_records,
    load_cached_feature_activations,
    write_cached_feature_activations,
)

from it_examples.example_prompt_configs import GemmaPromptConfig
from it_examples.utils.nb_ui_utils import display_layer_divergence_summary, display_logit_drift_summary
from tests.nb_experiment_harness.config import get_config_value
from tests.parity_analysis.concept_direction_parity_analysis import build_classification_prompt_text
from tests.parity_analysis.intervention_drift_analysis import (
    resolve_artifact_output_dir,
    save_preserved_intervention_artifacts,
    tensor_fingerprint,
)

_ipython_display: Any
_ipython_html: Any

try:
    from IPython.display import HTML as _ipython_html
    from IPython.display import display as _ipython_display
except ImportError:  # pragma: no cover - notebook display fallback
    _ipython_display = None
    _ipython_html = None


def _display(value: Any) -> None:
    if _ipython_display is None:
        print(value)
        return
    _ipython_display(value)


def _visible_token_text(token_text: object) -> str:
    if token_text is None:
        return ""
    return str(token_text).replace("\r", "␍").replace("\n", "↵").replace("\t", "⇥")


def _token_debug_entry_html(
    row: Mapping[str, Any],
    *,
    token_text_key: str,
    token_id_key: str,
    marks_key: str | None,
    index_key: str | None,
) -> str:
    index = row.get(index_key) if index_key else None
    token_id = row.get(token_id_key)
    token_text = _visible_token_text(row.get(token_text_key))
    marks = tuple(row.get(marks_key) or ()) if marks_key else ()
    title_parts = []
    if index is not None:
        title_parts.append(f"index={index}")
    title_parts.append(f"token_id={token_id}")
    if marks_key:
        title_parts.append(f"marks={','.join(marks) if marks else 'none'}")
    title_parts.append(f"raw={row.get(token_text_key)!r}")
    title_text = " | ".join(title_parts)
    style = [
        "display:inline-block",
        "margin:0.05rem 0.15rem 0.05rem 0",
        "padding:0.08rem 0.24rem",
        "border:1px solid #cbd5e1",
        "border-radius:0.35rem",
        "background:#f8fafc",
        "font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        "font-size:0.75rem",
        "white-space:pre",
    ]
    if "cache_answer" in marks or "answer" in marks:
        style.extend(["background:#fff3cd", "border-color:#b7791f"])
    elif "context" in marks:
        style.extend(["background:#dcfce7", "border-color:#15803d"])
    elif "probe" in marks:
        style.extend(["background:#dbeafe", "border-color:#2563eb"])
    label = token_text if token_text else "∅"
    style_text = "; ".join(style)
    return f'<span title="{html.escape(title_text, quote=True)}" style="{style_text}">{html.escape(label)}</span>'


def format_token_debug_html(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    token_text_key: str = "token_text",
    token_id_key: str = "token_id",
    marks_key: str | None = "marks",
    index_key: str | None = "index",
) -> str:
    normalized_rows = [row for row in (rows or []) if isinstance(row, Mapping)]
    if not normalized_rows:
        return ""
    rendered_rows = "".join(
        _token_debug_entry_html(
            row,
            token_text_key=token_text_key,
            token_id_key=token_id_key,
            marks_key=marks_key,
            index_key=index_key,
        )
        for row in normalized_rows
    )
    return f'<div style="line-height:1.5; max-width:96rem; white-space:normal">{rendered_rows}</div>'


def display_html_frame(frame: Any) -> None:
    if _ipython_html is None or _ipython_display is None:
        print(frame)
        return
    html_columns = tuple(getattr(frame, "attrs", {}).get("html_columns", ()))
    safe_frame = frame.copy()
    for column in safe_frame.columns:
        if column in html_columns:
            continue
        safe_frame[column] = safe_frame[column].map(
            lambda value: html.escape(value) if isinstance(value, str) else value
        )
    _display(_ipython_html(safe_frame.to_html(index=False, escape=False)))


def display_full_frame(frame: Any) -> None:
    if getattr(frame, "attrs", {}).get("html_columns"):
        display_html_frame(frame)
        return
    import pandas as pd  # type: ignore[import-untyped]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        240,
    ):
        _display(frame)


if TYPE_CHECKING:
    from interpretune.extensions.debug_generation import DebugGeneration
    from tests.concept_direction_approach_parity.concept_direction import (
        NotebookHarnessConfig,
        PromptRenderMode,
    )


DEFAULT_LOCAL_NEURONPEDIA_EXPORT_ROOT = Path(
    os.getenv(
        "LOCAL_NEURONPEDIA_EXPORT_ROOT",
        "/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports",
    )
)


@dataclass
class UploadedGraphMetadata:
    model_id: str
    slug: str
    prompt_tokens: list[str]
    prompt: str
    title_prefix: str
    json_url: str
    url: str | None = None
    url_embed: str | None = None
    id: str | None = None


@dataclass
class PublicGraphUploadResult:
    graph_metadata: UploadedGraphMetadata
    public_saved_to_db: bool
    public_save_to_db_error: str | None = None


ConstrainedFeatureSelectionRefValue = str | tuple[str, str, int, int]


@dataclass(frozen=True)
class ConstrainedFeatureSelectionRef:
    ref: ConstrainedFeatureSelectionRefValue
    activation_value: float | None = None


@dataclass
class ConstrainedFeatureSelection:
    specific_features: tuple[ConstrainedFeatureSelectionRef | tuple[int, int, int], ...] = ()
    layers: tuple[int, ...] = ()
    positions: tuple[int, ...] = ()
    feature_ids: tuple[int, ...] = ()
    layer_slices: tuple[slice, ...] = ()
    position_slices: tuple[slice, ...] = ()
    triples: tuple[tuple[int, int, int], ...] = ()
    layer_feature_pairs: tuple[tuple[int, int], ...] = ()
    activation_overrides: dict[tuple[int, int], float] = field(default_factory=dict)


def _split_constrained_feature_selection_ref(raw_ref: Any) -> tuple[Any, float | None]:
    if isinstance(raw_ref, Mapping):
        ref_value = raw_ref.get("ref", raw_ref.get("feature_ref"))
        activation_value = raw_ref.get("activation_value")
        if ref_value is None:
            raise ValueError("Constrained feature selection mappings must include 'ref' (or legacy 'feature_ref').")
        return ref_value, None if activation_value is None else float(activation_value)

    ref_value = getattr(raw_ref, "ref", raw_ref)
    activation_value = getattr(raw_ref, "activation_value", None)
    return ref_value, None if activation_value is None else float(activation_value)


def _serialize_constrained_feature_selection_ref(
    raw_ref: ConstrainedFeatureSelectionRef,
) -> str | list[Any] | dict[str, Any]:
    ref_value, activation_value = _split_constrained_feature_selection_ref(raw_ref)
    if isinstance(ref_value, str):
        serialized_ref: str | list[Any] = ref_value
    else:
        model_id, source_set, layer_number, feature_index = ref_value
        serialized_ref = [model_id, source_set, layer_number, feature_index]

    if activation_value is None:
        return serialized_ref
    return {"ref": serialized_ref, "activation_value": float(activation_value)}


def _normalize_constrained_feature_selection_ref(raw_ref: Any) -> ConstrainedFeatureSelectionRef:
    if isinstance(raw_ref, ConstrainedFeatureSelectionRef):
        return raw_ref

    activation_value = None
    if isinstance(raw_ref, Mapping):
        ref_value = raw_ref.get("ref", raw_ref.get("feature_ref"))
        activation_value_raw = raw_ref.get("activation_value")
        if ref_value is None:
            raise ValueError("Constrained feature selection mappings must include 'ref' (or legacy 'feature_ref').")
        raw_ref = ref_value
        activation_value = None if activation_value_raw is None else float(activation_value_raw)

    if isinstance(raw_ref, str):
        return ConstrainedFeatureSelectionRef(ref=raw_ref, activation_value=activation_value)
    if isinstance(raw_ref, Sequence) and not isinstance(raw_ref, (str, bytes)) and len(raw_ref) == 4:
        model_id, source_set, layer_number, feature_index = raw_ref
        return ConstrainedFeatureSelectionRef(
            ref=(str(model_id), str(source_set), int(layer_number), int(feature_index)),
            activation_value=activation_value,
        )
    raise ValueError(
        "Constrained feature selection entries must be full Neuronpedia refs or "
        "(model_id, source_set, layer, feature_index) tuples, optionally wrapped in a mapping with "
        "an activation_value override."
    )


def _normalize_feature_row_triple(raw_triple: Any, *, field_name: str) -> tuple[int, int, int]:
    if not isinstance(raw_triple, Sequence) or isinstance(raw_triple, (str, bytes)) or len(raw_triple) != 3:
        raise ValueError(f"{field_name} entries must be [layer, position, feature_id] rows. Got: {raw_triple!r}")
    layer_number, position_index, feature_id = raw_triple
    return int(layer_number), int(position_index), int(feature_id)


def _normalize_layer_feature_pair(raw_pair: Any, *, field_name: str) -> tuple[int, int]:
    if not isinstance(raw_pair, Sequence) or isinstance(raw_pair, (str, bytes)) or len(raw_pair) != 2:
        raise ValueError(f"{field_name} entries must be [layer, feature_id] rows. Got: {raw_pair!r}")
    layer_number, feature_id = raw_pair
    return int(layer_number), int(feature_id)


def _normalize_feature_selection_slice(raw_slice: Any, *, field_name: str) -> slice:
    if isinstance(raw_slice, slice):
        return raw_slice
    if isinstance(raw_slice, Mapping):
        return slice(raw_slice.get("start"), raw_slice.get("stop"), raw_slice.get("step"))
    if not isinstance(raw_slice, Sequence) or isinstance(raw_slice, (str, bytes)):
        raise ValueError(f"{field_name} entries must be [start, stop] or [start, stop, step] rows. Got: {raw_slice!r}")
    if not 2 <= len(raw_slice) <= 3:
        raise ValueError(
            f"{field_name} entries must contain 2 or 3 values. Got {len(raw_slice)} values in {raw_slice!r}"
        )
    start = raw_slice[0]
    stop = raw_slice[1]
    step = raw_slice[2] if len(raw_slice) == 3 else None
    return slice(start, stop, step)


def _serialize_feature_selection_slice(raw_slice: slice) -> list[int | None]:
    payload: list[int | None] = [
        None if raw_slice.start is None else int(raw_slice.start),
        None if raw_slice.stop is None else int(raw_slice.stop),
    ]
    if raw_slice.step is not None:
        payload.append(int(raw_slice.step))
    return payload


def _normalize_constrained_feature_selection_specific_feature(
    raw_feature: Any,
) -> ConstrainedFeatureSelectionRef | tuple[int, int, int]:
    if isinstance(raw_feature, Mapping) and any(key in raw_feature for key in ("triple", "feature_row")):
        return _normalize_feature_row_triple(
            raw_feature.get("triple", raw_feature.get("feature_row")),
            field_name="specific_features",
        )
    if isinstance(raw_feature, Sequence) and not isinstance(raw_feature, (str, bytes)) and len(raw_feature) == 3:
        return _normalize_feature_row_triple(raw_feature, field_name="specific_features")
    return _normalize_constrained_feature_selection_ref(raw_feature)


def _normalize_feature_selection_activation_overrides(raw_overrides: Any) -> dict[tuple[int, int], float]:
    if raw_overrides is None:
        return {}

    normalized: dict[tuple[int, int], float] = {}
    if isinstance(raw_overrides, Mapping):
        iterator = raw_overrides.items()
        for raw_key, raw_value in iterator:
            if isinstance(raw_key, str):
                key_parts = [part for part in raw_key.split("/") if part]
                if len(key_parts) != 2:
                    raise ValueError(
                        "activation_overrides mapping keys must be 'layer/feature_id' strings when using mapping "
                        f"syntax. Got: {raw_key!r}"
                    )
                layer_number, feature_id = (int(key_parts[0]), int(key_parts[1]))
            else:
                layer_number, feature_id = _normalize_layer_feature_pair(raw_key, field_name="activation_overrides")
            normalized[(layer_number, feature_id)] = float(raw_value)
        return normalized

    if not isinstance(raw_overrides, Sequence) or isinstance(raw_overrides, (str, bytes)):
        raise ValueError("activation_overrides must be a mapping or a list of {layer, feature_id, value} mappings.")

    for raw_entry in raw_overrides:
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                "activation_overrides list entries must be mappings with layer, feature_id, and value fields."
            )
        normalized[(int(raw_entry["layer"]), int(raw_entry["feature_id"]))] = float(raw_entry["value"])
    return normalized


def _normalize_constrained_feature_selection(raw_selection: Any) -> ConstrainedFeatureSelection | None:
    if raw_selection is None:
        return None
    if isinstance(raw_selection, ConstrainedFeatureSelection):
        return raw_selection

    if isinstance(raw_selection, Mapping):
        if ("ref" in raw_selection or "feature_ref" in raw_selection) and set(raw_selection.keys()).issubset(
            {"ref", "feature_ref", "activation_value"}
        ):
            return ConstrainedFeatureSelection(
                specific_features=(_normalize_constrained_feature_selection_specific_feature(raw_selection),),
            )

        specific_features = tuple(
            _normalize_constrained_feature_selection_specific_feature(raw_feature)
            for raw_feature in raw_selection.get("specific_features", raw_selection.get("refs", ()))
        )
        layers = tuple(int(value) for value in raw_selection.get("layers", ()))
        positions = tuple(int(value) for value in raw_selection.get("positions", ()))
        feature_ids = tuple(int(value) for value in raw_selection.get("feature_ids", ()))
        layer_slices = tuple(
            _normalize_feature_selection_slice(raw_slice, field_name="layer_slices")
            for raw_slice in raw_selection.get("layer_slices", ())
        )
        position_slices = tuple(
            _normalize_feature_selection_slice(raw_slice, field_name="position_slices")
            for raw_slice in raw_selection.get("position_slices", ())
        )
        triples = tuple(
            _normalize_feature_row_triple(raw_triple, field_name="triples")
            for raw_triple in raw_selection.get("triples", ())
        )
        layer_feature_pairs = tuple(
            _normalize_layer_feature_pair(raw_pair, field_name="layer_feature_pairs")
            for raw_pair in raw_selection.get("layer_feature_pairs", ())
        )
        activation_overrides = _normalize_feature_selection_activation_overrides(
            raw_selection.get("activation_overrides")
        )
        return ConstrainedFeatureSelection(
            specific_features=specific_features,
            layers=layers,
            positions=positions,
            feature_ids=feature_ids,
            layer_slices=layer_slices,
            position_slices=position_slices,
            triples=triples,
            layer_feature_pairs=layer_feature_pairs,
            activation_overrides=activation_overrides,
        )

    if isinstance(raw_selection, Iterable) and not isinstance(raw_selection, (str, bytes)):
        return ConstrainedFeatureSelection(
            specific_features=tuple(
                _normalize_constrained_feature_selection_specific_feature(raw_feature) for raw_feature in raw_selection
            )
        )
    raise ValueError(
        "constrained_feature_selection must be either a legacy list of feature refs or a mapping containing "
        "specific_features / layer_slices / position_slices / FeatureSelectionSpec-style filters."
    )


def _serialize_constrained_feature_selection_specific_feature(
    raw_feature: ConstrainedFeatureSelectionRef | tuple[int, int, int],
) -> str | list[Any] | dict[str, Any]:
    if isinstance(raw_feature, ConstrainedFeatureSelectionRef):
        return _serialize_constrained_feature_selection_ref(raw_feature)
    return [int(raw_feature[0]), int(raw_feature[1]), int(raw_feature[2])]


def _serialize_constrained_feature_selection(raw_selection: Any) -> list[Any] | dict[str, Any] | None:
    normalized = _normalize_constrained_feature_selection(raw_selection)
    if normalized is None:
        return None

    has_extended_filters = any(
        (
            normalized.layers,
            normalized.positions,
            normalized.feature_ids,
            normalized.layer_slices,
            normalized.position_slices,
            normalized.triples,
            normalized.layer_feature_pairs,
            normalized.activation_overrides,
        )
    )
    serialized_specific_features = [
        _serialize_constrained_feature_selection_specific_feature(raw_feature)
        for raw_feature in normalized.specific_features
    ]
    if not has_extended_filters:
        return serialized_specific_features

    payload: dict[str, Any] = {}
    if serialized_specific_features:
        payload["specific_features"] = serialized_specific_features
    if normalized.layers:
        payload["layers"] = [int(value) for value in normalized.layers]
    if normalized.positions:
        payload["positions"] = [int(value) for value in normalized.positions]
    if normalized.feature_ids:
        payload["feature_ids"] = [int(value) for value in normalized.feature_ids]
    if normalized.layer_slices:
        payload["layer_slices"] = [
            _serialize_feature_selection_slice(raw_slice) for raw_slice in normalized.layer_slices
        ]
    if normalized.position_slices:
        payload["position_slices"] = [
            _serialize_feature_selection_slice(raw_slice) for raw_slice in normalized.position_slices
        ]
    if normalized.triples:
        payload["triples"] = [
            [int(layer), int(position), int(feature_id)] for layer, position, feature_id in normalized.triples
        ]
    if normalized.layer_feature_pairs:
        payload["layer_feature_pairs"] = [
            [int(layer), int(feature_id)] for layer, feature_id in normalized.layer_feature_pairs
        ]
    if normalized.activation_overrides:
        payload["activation_overrides"] = [
            {"layer": int(layer), "feature_id": int(feature_id), "value": float(value)}
            for (layer, feature_id), value in sorted(normalized.activation_overrides.items())
        ]
    return payload


def _count_specific_feature_requests(raw_selection: Any) -> int:
    normalized = _normalize_constrained_feature_selection(raw_selection)
    return 0 if normalized is None else len(normalized.specific_features)


def _build_classification_prompt(entity_name: str, question: str) -> str:
    return build_classification_prompt_text(entity_name, question)


def _chattify_apply_chat_template(prompt: str, tokenizer: Any) -> str:
    cfg = GemmaPromptConfig()
    return cfg.apply_chat_template_fn(tokenizer, prompt, tokenize=False, add_generation_prompt=True)


def _chattify_gemma_dataclass(prompt: str) -> str:
    cfg = GemmaPromptConfig()
    return cfg.model_chat_template_fn(prompt, tokenization_pattern="gemma-chat")


def _chattify(prompt: str, tokenizer: Any, method: str = "apply_chat_template") -> str:
    if method == "gemma_dataclass":
        return _chattify_gemma_dataclass(prompt)
    return _chattify_apply_chat_template(prompt, tokenizer)


def create_work_root(base_dir: str | None, experiment_name: str, *, prefix: str = "nb_experiment") -> Path:
    if base_dir:
        work_root = Path(base_dir).expanduser().resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        return work_root
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_{experiment_name}_"))


def tensor_to_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.float32)


def feature_ids_to_tuples(feature_ids: Any) -> list[tuple[int, ...]]:
    return [tuple(feature.tolist()) for feature in feature_ids]


def scalar_tensor_list(values: list[float] | tuple[float, ...], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(list(values), dtype=dtype)


def phase_run_name(experiment_name: str, label: str) -> str:
    cleaned = label.lower().replace(" ", "_").replace("/", "_")
    return f"{experiment_name}_{cleaned}"


def _slugify_graph_component(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "graph"


def _truncate_http_error_body(response: requests.Response, *, limit: int = 500) -> str:
    body_text = response.text.strip()
    if len(body_text) > limit:
        body_text = f"{body_text[:limit]}..."
    return f"HTTP {response.status_code}: {body_text}" if body_text else f"HTTP {response.status_code}"


def _load_uploaded_graph_metadata(graph_path: str | Path) -> tuple[str, UploadedGraphMetadata]:
    graph_text = Path(graph_path).read_text(encoding="utf-8")
    graph_payload = json.loads(graph_text)
    metadata = graph_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Graph JSON must include a metadata mapping before upload.")

    model_id = str(metadata.get("scan", "") or "").strip()
    slug = str(metadata.get("slug", "") or "").strip()
    prompt = str(metadata.get("prompt", "") or "")
    prompt_tokens = [str(token) for token in metadata.get("prompt_tokens", [])]
    title_prefix = str(metadata.get("title_prefix", "") or "")
    if not model_id or not slug or not prompt or not prompt_tokens:
        raise ValueError("Graph JSON metadata must include scan, slug, prompt, and prompt_tokens before upload.")

    return (
        graph_text,
        UploadedGraphMetadata(
            model_id=model_id,
            slug=slug,
            prompt_tokens=prompt_tokens,
            prompt=prompt,
            title_prefix=title_prefix,
            json_url="",
        ),
    )


def _upload_graph_for_public_then_sync_local(
    graph_path: str | Path,
    *,
    api_key: str,
    public_base_url: str = DEFAULT_NEURONPEDIA_BASE_URL,
) -> PublicGraphUploadResult:
    graph_text, graph_metadata = _load_uploaded_graph_metadata(graph_path)
    graph_gzip = gzip.compress(graph_text.encode("utf-8"))
    api_root = f"{public_base_url.rstrip('/')}/api/graph"
    headers = {"X-Api-Key": api_key}

    signed_put_response = requests.post(
        f"{api_root}/signed-put",
        headers=headers,
        json={
            "filename": f"{graph_metadata.slug}.json",
            "contentLength": len(graph_gzip),
            "contentType": "application/json",
        },
        timeout=60,
    )
    if not signed_put_response.ok:
        raise RuntimeError(
            f"Failed to create public graph upload URL for '{graph_metadata.slug}': "
            f"{_truncate_http_error_body(signed_put_response)}"
        )

    signed_put_payload = signed_put_response.json()
    signed_put_url = str(signed_put_payload.get("url", "") or "").strip()
    put_request_id = str(signed_put_payload.get("putRequestId", "") or "").strip()
    if not signed_put_url or not put_request_id:
        raise RuntimeError(
            f"Public graph upload response for '{graph_metadata.slug}' did not include both url and putRequestId."
        )

    put_response = requests.put(
        signed_put_url,
        data=graph_gzip,
        headers={"Content-Encoding": "gzip"},
        timeout=120,
    )
    if not put_response.ok:
        raise RuntimeError(
            f"Failed to upload public graph payload for '{graph_metadata.slug}': "
            f"{_truncate_http_error_body(put_response)}"
        )

    graph_metadata.json_url = signed_put_url.split("?", 1)[0]

    save_to_db_response = requests.post(
        f"{api_root}/save-to-db",
        headers=headers,
        json={"putRequestId": put_request_id},
        timeout=60,
    )
    public_save_to_db_error = None
    if save_to_db_response.ok:
        response_payload = save_to_db_response.json()
        public_graph_url = response_payload.get("url")
        if isinstance(public_graph_url, str) and public_graph_url.strip():
            graph_metadata.url = public_graph_url.strip()
            graph_metadata.url_embed = f"{graph_metadata.url}&embed=true"
    else:
        public_save_to_db_error = _truncate_http_error_body(save_to_db_response)

    return PublicGraphUploadResult(
        graph_metadata=graph_metadata,
        public_saved_to_db=save_to_db_response.ok,
        public_save_to_db_error=public_save_to_db_error,
    )


@contextmanager
def _temporary_neuronpedia_upload_env(cfg: NotebookHarnessConfig, *, upload_target: str) -> Any:
    env_updates = {"USE_LOCALHOST": "true" if upload_target == "localhost" else "false"}
    if upload_target == "localhost":
        api_key = os.environ.get("DEV_NEURONPEDIA_API_KEY")
        if not api_key:
            raise ValueError("DEV_NEURONPEDIA_API_KEY is required when local graph upload is enabled.")
        env_updates["LOCAL_NEURONPEDIA_WEBAPP_URL"] = cfg.local_neuronpedia_webapp_url.rstrip("/")
    elif upload_target == "public_then_sync_local":
        api_key = os.environ.get("NEURONPEDIA_API_KEY")
        if not api_key:
            raise ValueError("NEURONPEDIA_API_KEY is required when public_then_sync_local graph upload is enabled.")
    else:
        raise ValueError(f"Unsupported local graph upload target: {upload_target!r}")

    previous_env = {key: os.environ.get(key) for key in ("USE_LOCALHOST", "LOCAL_NEURONPEDIA_WEBAPP_URL")}
    os.environ.update(env_updates)
    if upload_target != "localhost":
        os.environ.pop("LOCAL_NEURONPEDIA_WEBAPP_URL", None)
    try:
        yield api_key
    finally:
        for key, previous_value in previous_env.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _load_graph_payload_from_json_url(graph_json_url: str) -> dict[str, Any]:
    request = Request(graph_json_url, method="GET")
    with urlopen(request, timeout=60) as response:
        raw_payload = response.read()
    if raw_payload.startswith(b"\x1f\x8b"):
        raw_payload = gzip.decompress(raw_payload)
    payload = json.loads(raw_payload.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Neuronpedia graph payload must decode to a mapping.")
    return payload


def _resolve_graph_source_set_name(cursor: Any, *, model_id: str, graph_payload: Mapping[str, Any]) -> str | None:
    metadata = graph_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None

    source_set_name: str | None = None
    for metadata_key in ("info", "feature_details"):
        candidate_mapping = metadata.get(metadata_key)
        if not isinstance(candidate_mapping, Mapping):
            continue
        candidate_value = candidate_mapping.get("neuronpedia_source_set")
        if isinstance(candidate_value, str) and candidate_value.strip():
            source_set_name = candidate_value.strip()
            break

    if source_set_name is None:
        return None
    if "/" in source_set_name:
        source_set_name = source_set_name.split("/")[-1] or source_set_name

    cursor.execute(
        'SELECT name FROM public."SourceSet" WHERE "modelId"=%s AND name=%s;',
        (model_id, source_set_name),
    )
    source_set_row = cursor.fetchone()
    return None if source_set_row is None else cast(str, source_set_row[0])


def sync_graph_metadata_to_local_dev(
    *,
    username: str,
    graph_metadata: Any | None = None,
    model_id: str | None = None,
    slug: str | None = None,
    public_api_key: str | None = None,
    local_db_url: str | None = None,
) -> dict[str, Any]:
    """Sync a public Neuronpedia graph into the local dev GraphMetadata table."""

    normalized_username = str(username).strip()
    if not normalized_username:
        raise ValueError("username is required for local graph metadata sync")

    load_dotenv(".env.localhost_np")

    if graph_metadata is None:
        if not model_id or not slug:
            raise ValueError("model_id and slug are required when graph_metadata is not provided")

        import neuronpedia
        from neuronpedia.np_graph_metadata import NPGraphMetadata

        resolved_public_api_key = public_api_key or os.environ.get("NEURONPEDIA_API_KEY")
        if not resolved_public_api_key:
            raise ValueError("NEURONPEDIA_API_KEY is not set in the environment.")

        with neuronpedia.api_key(resolved_public_api_key):
            graph_metadata = NPGraphMetadata.get(model_id, slug)

    resolved_model_id = str(getattr(graph_metadata, "model_id", model_id or "")).strip()
    resolved_slug = str(getattr(graph_metadata, "slug", slug or "")).strip()
    graph_json_url = getattr(graph_metadata, "json_url", None)
    if not resolved_model_id or not resolved_slug or not isinstance(graph_json_url, str) or not graph_json_url.strip():
        raise ValueError("graph_metadata must include model_id, slug, and json_url")

    resolved_db_url = resolve_local_neuronpedia_db_url(local_db_url)
    graph_payload = _load_graph_payload_from_json_url(graph_json_url)
    resolved_graph_metadata_id = str(getattr(graph_metadata, "id", "") or "").strip()
    if not resolved_graph_metadata_id:
        slug_digest = hashlib.sha1(f"{resolved_model_id}:{resolved_slug}".encode("utf-8")).hexdigest()[:24]
        resolved_graph_metadata_id = f"local-sync-{slug_digest}"

    with psycopg.connect(resolved_db_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT id FROM "User" WHERE name=%s;', (normalized_username,))
            user_row = cursor.fetchone()
            if user_row is None:
                raise ValueError(f"User '{normalized_username}' not found in the local Neuronpedia database.")

            resolved_user_id = cast(str, user_row[0])
            source_set_name = _resolve_graph_source_set_name(
                cursor,
                model_id=resolved_model_id,
                graph_payload=graph_payload,
            )
            prompt_tokens = [str(token) for token in getattr(graph_metadata, "prompt_tokens", [])]
            title_prefix = str(getattr(graph_metadata, "title_prefix", "") or "")
            prompt = str(getattr(graph_metadata, "prompt", "") or "")

            cursor.execute(
                """
                INSERT INTO public."GraphMetadata" (
                    id, "modelId", "sourceSetName", slug, "promptTokens", prompt, "titlePrefix", url, "userId",
                    "createdAt", "updatedAt", "isFeatured"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(3), current_timestamp(3), FALSE)
                ON CONFLICT ("modelId", slug) DO UPDATE SET
                    "sourceSetName" = EXCLUDED."sourceSetName",
                    "promptTokens" = EXCLUDED."promptTokens",
                    prompt = EXCLUDED.prompt,
                    "titlePrefix" = EXCLUDED."titlePrefix",
                    url = EXCLUDED.url,
                    "userId" = EXCLUDED."userId",
                    "updatedAt" = current_timestamp(3),
                    "isFeatured" = FALSE
                RETURNING id, "modelId", slug, url, "sourceSetName";
                """,
                (
                    resolved_graph_metadata_id,
                    resolved_model_id,
                    source_set_name,
                    resolved_slug,
                    prompt_tokens,
                    prompt,
                    title_prefix,
                    graph_json_url,
                    resolved_user_id,
                ),
            )
            synced_row = cursor.fetchone()
        conn.commit()

    local_webapp_url = os.environ.get("LOCAL_NEURONPEDIA_WEBAPP_URL", DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL).rstrip("/")
    local_graph_url = f"{local_webapp_url}/{resolved_model_id}/graph?slug={resolved_slug}"

    return {
        "model_id": resolved_model_id,
        "slug": resolved_slug,
        "username": normalized_username,
        "user_id": resolved_user_id,
        "graph_json_url": graph_json_url,
        "public_graph_url": getattr(graph_metadata, "url", None),
        "public_graph_embed_url": getattr(graph_metadata, "url_embed", None),
        "local_graph_url": local_graph_url,
        "local_graph_embed_url": f"{local_graph_url}&embed=true",
        "source_set_name": None if synced_row is None else synced_row[4],
        "graph_metadata_id": None if synced_row is None else synced_row[0],
    }


def maybe_save_local_neuronpedia_graph(
    cfg: NotebookHarnessConfig,
    module: Any,
    graph: Any,
    *,
    phase_label: str,
    rendered_prompt: str,
    graph_target_tokens: Sequence[str] | None = None,
    graph_target_ids: Sequence[int] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not cfg.upload_local_graphs:
        return None

    if not cfg.use_localhost:
        warning_message = "Skipping local Neuronpedia graph upload because localhost mode is disabled."
        warnings.warn(warning_message, stacklevel=2)
        return {
            "phase_label": phase_label,
            "uploaded": False,
            "skipped": True,
            "reason": warning_message,
        }

    neuronpedia_handle = getattr(module, "neuronpedia", None)
    if neuronpedia_handle is None:
        warning_message = "Skipping local Neuronpedia graph upload because the module has no neuronpedia extension."
        warnings.warn(warning_message, stacklevel=2)
        return {
            "phase_label": phase_label,
            "uploaded": False,
            "skipped": True,
            "reason": warning_message,
        }

    slug_prefix_base = cfg.local_graph_slug_prefix or cfg.experiment_name
    upload_target = str(getattr(cfg, "local_graph_upload_target", "localhost")).strip() or "localhost"
    slug = (
        f"{_slugify_graph_component(slug_prefix_base)}-"
        f"{_slugify_graph_component(phase_label)}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir = Path(cfg.work_root) / "graph_artifacts" / _slugify_graph_component(phase_label)

    custom_metadata: dict[str, Any] = {
        "feature_details": {"neuronpedia_source_set": cfg.neuronpedia_set},
        "info": {
            "neuronpedia_source_set": cfg.neuronpedia_set,
            "experiment_name": cfg.experiment_name,
            "experiment_config_name": cfg.experiment_config_name,
            "analysis_mode": cfg.analysis_mode,
            "prompt_render_mode": cfg.prompt_render_mode,
            "phase_label": phase_label,
            "neuronpedia_model": cfg.neuronpedia_model,
            "rendered_prompt_preview": rendered_prompt[:200],
        },
    }
    if graph_target_tokens is not None:
        custom_metadata["info"]["graph_target_tokens"] = [str(token) for token in graph_target_tokens]
    if graph_target_ids is not None:
        custom_metadata["info"]["graph_target_ids"] = [int(token_id) for token_id in graph_target_ids]
    if extra_metadata:
        for key, value in extra_metadata.items():
            if isinstance(value, Mapping) and isinstance(custom_metadata.get(key), dict):
                cast(dict[str, Any], custom_metadata[key]).update(dict(value))
            else:
                custom_metadata[key] = value

    try:
        graph_path = module.save_graph(
            graph=graph,
            output_path=output_dir,
            slug=slug,
            custom_metadata=custom_metadata,
            use_neuronpedia=True,
        )
    except Exception as exc:
        warning_message = f"Failed to save local Neuronpedia graph artifact for phase '{phase_label}': {exc}"
        warnings.warn(warning_message, stacklevel=2)
        return {
            "phase_label": phase_label,
            "slug": slug,
            "uploaded": False,
            "skipped": True,
            "reason": warning_message,
        }

    public_upload_result: PublicGraphUploadResult | None = None
    try:
        with _temporary_neuronpedia_upload_env(cfg, upload_target=upload_target) as api_key:
            if upload_target == "public_then_sync_local":
                public_upload_result = _upload_graph_for_public_then_sync_local(graph_path, api_key=api_key)
                graph_metadata = public_upload_result.graph_metadata
            else:
                graph_metadata = neuronpedia_handle.upload_graph_to_neuronpedia(graph_path, api_key=api_key)
    except Exception as exc:
        warning_message = f"Failed to upload local Neuronpedia graph '{slug}': {exc}"
        warnings.warn(warning_message, stacklevel=2)
        return {
            "phase_label": phase_label,
            "slug": slug,
            "local_graph_path": str(graph_path),
            "uploaded": False,
            "skipped": False,
            "reason": warning_message,
        }

    graph_url = getattr(graph_metadata, "url", None)
    if graph_url is None:
        fallback_base_url = (
            cfg.local_neuronpedia_webapp_url if upload_target == "localhost" else DEFAULT_NEURONPEDIA_BASE_URL
        )
        graph_url = f"{fallback_base_url.rstrip('/')}/{cfg.neuronpedia_model}/graph?slug={slug}"

    artifact = {
        "phase_label": phase_label,
        "slug": slug,
        "local_graph_path": str(graph_path),
        "uploaded": True,
        "skipped": False,
        "upload_target": upload_target,
        "graph_url": graph_url,
        "graph_json_url": getattr(graph_metadata, "json_url", None),
        "graph_embed_url": getattr(graph_metadata, "url_embed", None),
        "neuronpedia_model": cfg.neuronpedia_model,
        "neuronpedia_source_set": cfg.neuronpedia_set,
    }

    if upload_target != "public_then_sync_local":
        return artifact

    artifact.update(
        public_graph_url=getattr(graph_metadata, "url", None),
        public_graph_json_url=getattr(graph_metadata, "json_url", None),
        public_graph_embed_url=getattr(graph_metadata, "url_embed", None),
        public_saved_to_db=False if public_upload_result is None else public_upload_result.public_saved_to_db,
        public_save_to_db_error=None if public_upload_result is None else public_upload_result.public_save_to_db_error,
        synced_to_local=False,
    )
    try:
        sync_summary = sync_graph_metadata_to_local_dev(
            username=cast(str, getattr(cfg, "local_graph_owner_username", None)),
            graph_metadata=graph_metadata,
            local_db_url=cfg.local_neuronpedia_db_url,
        )
    except Exception as exc:
        warning_message = f"Uploaded public Neuronpedia graph '{slug}' but failed to sync local metadata: {exc}"
        warnings.warn(warning_message, stacklevel=2)
        artifact["sync_error"] = warning_message
        return artifact

    artifact.update(
        synced_to_local=True,
        graph_url=sync_summary["local_graph_url"],
        graph_embed_url=sync_summary["local_graph_embed_url"],
        local_graph_url=sync_summary["local_graph_url"],
        local_graph_embed_url=sync_summary["local_graph_embed_url"],
        local_graph_sync=sync_summary,
    )
    return artifact


def sync_dev_graph_metadata() -> None:
    """Sync graph metadata from Neuronpedia production service to a local dev database."""
    model_id = os.environ.get("MODEL_ID")
    slug = os.environ.get("SLUG")
    username = os.environ.get("USERNAME")

    if not all([model_id, slug, username]):
        raise ValueError("MODEL_ID, SLUG, and USERNAME environment variables are required.")

    summary = sync_graph_metadata_to_local_dev(
        model_id=model_id,
        slug=slug,
        username=cast(str, username),
        local_db_url=os.environ.get("LOCAL_NEURONPEDIA_DB_URL"),
    )
    print(json.dumps(summary, indent=2, default=str))


def _get_config_value_with_preset_default(
    payload: Mapping[str, Any],
    preset_defaults: Mapping[str, Any],
    *,
    section: str,
    key: str,
    flat_key: str,
    default: Any,
) -> Any:
    value = get_config_value(payload, section=section, key=key, flat_key=flat_key)
    if value is not None:
        return value

    section_defaults = preset_defaults.get(section)
    if isinstance(section_defaults, Mapping) and key in section_defaults:
        return section_defaults[key]

    return default


def _print_json(payload: Mapping[str, Any] | Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def serialize_notebook_config(
    cfg: NotebookHarnessConfig,
    *,
    config_path: str | Path,
    work_root_is_temporary: bool,
) -> dict[str, Any]:
    return {
        "experiment_name": cfg.experiment_name,
        "experiment_config_name": cfg.experiment_config_name,
        "experiment_config_path": str(Path(config_path).expanduser().resolve()),
        "model_name": cfg.model_name,
        "transcoder_set": cfg.transcoder_set,
        "neuronpedia_model": cfg.neuronpedia_model,
        "neuronpedia_set": cfg.neuronpedia_set,
        "analysis_mode": cfg.analysis_mode,
        "concept_direction_mode": cfg.analysis_direction_mode_name,
        "analysis_concept_label": cfg.analysis_concept_label,
        "explicit_direction_tokens": cfg.explicit_direction_tokens,
        "enable_zero_softcap": cfg.enable_zero_softcap,
        "debug_session_surface_preset": cfg.debug_session_surface_preset,
        "prompt": cfg.prompt,
        "target_tokens": cfg.target_tokens,
        "target_token_ids": cfg.target_token_ids,
        "top_n": cfg.top_n,
        "default_scale_factor": cfg.default_scale_factor,
        "scale_factor_sweep": cfg.scale_factor_sweep,
        "ablation_n_list": cfg.ablation_n_list,
        "enable_sign_aware": cfg.enable_sign_aware,
        "batch_size": cfg.batch_size,
        "max_feature_nodes": cfg.max_feature_nodes,
        "force_device": cfg.force_device,
        "work_root": str(cfg.work_root),
        "work_root_is_temporary": work_root_is_temporary,
        "use_localhost": cfg.use_localhost,
        "neuronpedia_base_url": cfg.neuronpedia_base_url,
        "local_neuronpedia_db_url": cfg.local_neuronpedia_db_url,
        "local_neuronpedia_webapp_url": cfg.local_neuronpedia_webapp_url,
        "upload_local_graphs": cfg.upload_local_graphs,
        "local_graph_slug_prefix": cfg.local_graph_slug_prefix,
        "check_local_explanation_coverage": cfg.check_local_explanation_coverage,
        "generate_missing_local_explanations": cfg.generate_missing_local_explanations,
        "local_explanation_feature_limit": cfg.local_explanation_feature_limit,
        "local_explanation_type_name": cfg.local_explanation_type_name,
        "prompt_render_mode": cfg.prompt_render_mode,
        "constrained_feature_selection_refs": _serialize_constrained_feature_selection(
            cfg.constrained_feature_selection_refs
        ),
        "store_latent_extraction_mode": cfg.store_latent_extraction_mode,
        "context_enhanced_scale": cfg.context_enhanced_scale,
        "debug_validation_top_k": cfg.debug_validation_top_k,
        "debug_validation_raise_on_failure": cfg.debug_validation_raise_on_failure,
        "debug_validation_tolerances": {
            "act_atol": cfg.debug_validation_act_atol,
            "act_rtol": cfg.debug_validation_act_rtol,
            "logit_atol": cfg.debug_validation_logit_atol,
            "logit_rtol": cfg.debug_validation_logit_rtol,
        },
    }


def log_notebook_config(
    cfg: NotebookHarnessConfig,
    *,
    config_path: str | Path,
    work_root_is_temporary: bool,
    verbose: bool,
) -> dict[str, Any]:
    config_summary = serialize_notebook_config(
        cfg,
        config_path=config_path,
        work_root_is_temporary=work_root_is_temporary,
    )
    if verbose:
        _print_json(config_summary)
    for warning_message in cfg.mode_warning_messages:
        print(f"WARNING: {warning_message}")
    return config_summary


def display_tokenizer_verification_report(report: Mapping[str, Any]) -> None:
    print(f"Module type: {report['module_type']}")
    print(f"Prompt render mode: {report['prompt_render_mode']}")
    print(f"Prompt token count: {report['prompt_token_count']}")

    print("\nKey tokens:")
    for token, payload in report.get("key_tokens", {}).items():
        print(f"  {token}: ids={payload['ids']}, decoded={payload['decoded']!r}")

    print("\nResolved key-token candidates:")
    for entry in report.get("resolved_key_token_candidates", []):
        print(
            "  "
            f"{entry['label']} [{entry['variant']}] from {entry['source_token']}: "
            f"token_id={entry['token_id']}, ids={entry['ids']}, decoded={entry['decoded']!r}"
        )

    print("\nRendered prompt variants:")
    for mode_name, preview in report.get("render_variants", {}).items():
        if preview is None:
            print(f"\n[{mode_name}]\n(not available - model has no chat template)")
            continue
        print(f"\n[{mode_name}]\n{preview[:300]}")
        if mode_name in report.get("render_variant_token_ids", {}):
            print(f"token_ids={report['render_variant_token_ids'][mode_name]}")
        if mode_name in report.get("render_variant_tokens", {}):
            print(f"tokens={report['render_variant_tokens'][mode_name][:40]}")

    equalities = report.get("render_variant_equalities", {})
    print("\nPrompt parity checks:")
    print(
        f"  apply_chat_template vs gemma_dataclass string parity: {equalities.get('apply_chat_template_vs_dataclass')}"
    )
    print(
        "  apply_chat_template vs gemma_dataclass token-id parity: "
        f"{equalities.get('apply_chat_template_vs_dataclass_token_ids')}"
    )

    print("\nSelected prompt preview:")
    print(report.get("selected_prompt_preview", ""))
    print(f"Selected prompt token ids: {report.get('selected_prompt_token_ids', [])}")
    print(f"Selected prompt tokens: {report.get('selected_prompt_tokens', [])[:60]}")


def display_initial_sanity_check_report(report: Mapping[str, Any]) -> None:
    print("Initial Sanity Check")
    print(f"Prompt style: {report['prompt_style']}")
    print(f"Prompt: {str(report['rendered_prompt'])[:200]}")
    print(f"Generated: {report['generated_text']}")

    print("\nFirst-token logit analysis:")
    print(f"  {'Token':<12} {'ID':>8} {'Logit':>10} {'Prob':>12}")
    print(f"  {'-' * 46}")
    for entry in report.get("key_tokens", []):
        token_id = entry.get("token_id")
        if token_id is not None:
            print(
                f"  {entry['label']:<12} {int(token_id):>8} {float(entry['logit']):>10.4f} "
                f"{float(entry['prob']):>12.6f}"
            )
            continue
        print(f"  {entry['label']:<12} {'N/A':>8} {'N/A':>10} {'N/A':>12}")
    print(f"  {'-' * 46}")
    print(
        f"  {'Top-1':<12} {int(report['top1_id']):>8} {float(report['top1_logit']):>10.4f} "
        f"{float(report['top1_prob']):>12.6f}  ({report['top1_token']!r})"
    )


def display_baseline_path_debug_report(report: Mapping[str, Any]) -> None:
    print("Baseline Path Debug")
    print(f"Prompt render mode: {report['prompt_render_mode']}")
    _print_json({"generation_kwargs": report.get("generation_kwargs", {})})
    prompt_debug = cast(Mapping[str, Any], report.get("prompt_debug", {}))
    print(f"Raw prompt: {prompt_debug.get('raw_prompt')}")
    print(f"Rendered prompt: {prompt_debug.get('rendered_prompt')}")
    print(f"Input IDs: {prompt_debug.get('input_ids')}")
    print(f"Tokens: {prompt_debug.get('tokens', [])[:80]}")

    print("\nBaseline source top-1 summaries:")
    for source_name, entries in cast(Mapping[str, Any], report.get("baseline_sources", {})).items():
        if not entries:
            print(f"  {source_name:<28} <none>")
            continue
        top_entry = entries[0]
        print(
            f"  {source_name:<28} {top_entry['token']!r} "
            f"(id={top_entry['token_id']}, logit={top_entry['logit']:.4f}, prob={top_entry['prob']:.6f})"
        )

    print("\nMax abs diffs:")
    for diff_name, diff_value in cast(Mapping[str, Any], report.get("max_abs_diffs", {})).items():
        print(f"  {diff_name}: {float(diff_value):.6f}")


def _display_diagnostic_rows(title: str, rows: Sequence[Any]) -> None:
    print(title)
    if not rows:
        print("  <none>")
        return
    _display(list(rows))


def display_debug_intervention_validation_report(report: Mapping[str, Any]) -> None:
    print("Debug Intervention Validation")
    print(f"Selected feature: {report.get('selected_feature')}")
    if report.get("selected_feature_score") is not None:
        print(f"Selected feature score: {float(report['selected_feature_score']):+.6f}")
    if report.get("baseline_activation") is not None:
        print(f"Baseline activation: {float(report['baseline_activation']):+.6f}")
    if report.get("intervention_value") is not None:
        print(f"Intervention value: {float(report['intervention_value']):+.6f}")
    if all(report.get(key) is not None for key in ("pre_gap", "post_gap", "gap_delta")):
        print(f"Pre-gap: {float(report['pre_gap']):+.6f}")
        print(f"Post-gap: {float(report['post_gap']):+.6f}")
        print(f"Gap Delta: {float(report['gap_delta']):+.6f}")
    print(f"Activation check passed: {bool(report.get('activation_passed'))}")
    print(f"Logit check passed: {bool(report.get('logit_passed'))}")
    if report.get("artifact_dir") is not None:
        print(f"Preserved artifact dir: {report['artifact_dir']}")

    input_diagnostics = cast(Mapping[str, Any], report.get("input_diagnostics", {}))
    drift_report = cast(Mapping[str, Any], report.get("drift_report", {}))
    drift_summary = cast(Mapping[str, Any], drift_report.get("logit_summary", {}))
    _print_json(
        {
            "activation_max_abs_error": report.get("activation_max_abs_error"),
            "activation_mean_abs_error": report.get("activation_mean_abs_error"),
            "activation_sign_mismatch_count": report.get("activation_sign_mismatch_count"),
            "logit_max_abs_error": report.get("logit_max_abs_error"),
            "logit_mean_abs_error": report.get("logit_mean_abs_error"),
            "logit_sign_mismatch_count": report.get("logit_sign_mismatch_count"),
            "validation_tolerances": report.get("validation_tolerances"),
            "graph_summary": report.get("graph_summary"),
            "graph_target_tokens": report.get("graph_target_tokens"),
            "input_summary": {
                key: input_diagnostics.get(key)
                for key in (
                    "graph_input_source",
                    "rendered_prompt_token_count",
                    "graph_input_token_count",
                    "graph_inputs_match_rendered_prompt",
                    "first_difference_index",
                )
            },
            "drift_summary": {
                "divergent_feature_count": drift_report.get("divergent_feature_count"),
                "total_feature_count": drift_report.get("total_feature_count"),
                "divergent_logit_count": drift_summary.get("divergent_logit_count"),
                "total_logit_count": drift_summary.get("total_logit_count"),
                "layer_with_max_divergence": drift_report.get("layer_with_max_divergence"),
            },
            "artifact_dir": report.get("artifact_dir"),
        }
    )

    layer_summaries = drift_report.get("layer_summaries")
    if layer_summaries:
        display_layer_divergence_summary(layer_summaries)
    if drift_summary:
        display_logit_drift_summary(drift_summary)

    key_logit_rows = [
        {
            "token_id": int(token_id),
            "token": label,
            "pre_logit": float(pre_logit),
            "post_logit": float(post_logit),
            "delta": float(post_logit - pre_logit),
        }
        for token_id, label, pre_logit, post_logit in zip(
            report.get("key_ids", []),
            report.get("key_labels", []),
            report.get("key_logits_pre", []),
            report.get("key_logits_post", []),
            strict=False,
        )
    ]
    _display_diagnostic_rows("Key-token logit deltas", key_logit_rows)

    if report.get("intervention_paths") is not None:
        print("Intervention path comparison")
        _print_json(report["intervention_paths"])
    if report.get("selected_feature_self_effect") is not None:
        _display_diagnostic_rows("Selected feature self-effect", [report["selected_feature_self_effect"]])
    _display_diagnostic_rows(
        "Same feature rows in graph",
        cast(Sequence[Any], report.get("same_feature_rows_in_graph", [])),
    )
    debug_validation_top_k = int(report.get("debug_validation_top_k", 0) or 0)
    _display_diagnostic_rows(
        f"Top {debug_validation_top_k} activation error rows",
        cast(Sequence[Any], report.get("activation_error_rows", [])),
    )
    _display_diagnostic_rows(
        "Activation layer summary",
        cast(Sequence[Any], report.get("activation_layer_summary", [])),
    )
    _display_diagnostic_rows(
        f"Top {debug_validation_top_k} expected-effect rows",
        cast(Sequence[Any], report.get("expected_effect_rows", [])),
    )
    _display_diagnostic_rows(
        f"Top {debug_validation_top_k} actual-effect rows",
        cast(Sequence[Any], report.get("actual_effect_rows", [])),
    )
    _display_diagnostic_rows(
        f"Top {debug_validation_top_k} logit error rows",
        cast(Sequence[Any], report.get("logit_error_rows", [])),
    )


def display_local_explanation_report(
    *,
    prefetch_summary: Mapping[str, Any] | None = None,
    coverage_summary: Mapping[str, Any] | None = None,
) -> None:
    if prefetch_summary is not None:
        print("Local explanation prefetch")
        _print_json(prefetch_summary)
    if coverage_summary is not None:
        print("Local explanation coverage")
        _print_json(coverage_summary)


def build_shared_summary_record(
    cfg: NotebookHarnessConfig,
    *,
    config_path: str | Path,
    work_root_removed: bool,
) -> dict[str, Any]:
    return {
        "experiment_name": cfg.experiment_name,
        "config_name": cfg.experiment_config_name,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "model_family": cfg.model_family,
        "model_variant": cfg.model_variant,
        "model_name": cfg.model_name,
        "analysis_mode": cfg.analysis_mode,
        "concept_direction_mode": cfg.analysis_direction_mode_name,
        "prompt_render_mode": cfg.prompt_render_mode,
        "target_tokens": list(cfg.target_tokens) if cfg.target_tokens is not None else None,
        "target_token_ids": list(cfg.target_token_ids) if cfg.target_token_ids is not None else None,
        "work_root_removed": work_root_removed,
    }


def resolve_key_tokens(cfg: NotebookHarnessConfig) -> tuple[str, ...]:
    """Return the experiment-owned key tokens used for analysis and reporting."""

    if cfg.key_tokens_override is None or not cfg.key_tokens_override:
        raise ValueError(
            "NotebookHarnessConfig requires KEY_TOKENS in the experiment config; concept-pair YAMLs no longer "
            "provide key token defaults."
        )
    return tuple(cfg.key_tokens_override)


def _build_key_token_candidates(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    *,
    include_space_prefixed_variants: bool = True,
    include_bare_variants: bool = True,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _append_candidate(
        token_id: int,
        *,
        label: str,
        source_token: str,
        variant: str,
        encoded_ids: Sequence[int],
    ) -> None:
        if token_id in seen_ids:
            return
        seen_ids.add(token_id)
        candidates.append(
            {
                "label": label,
                "token_id": int(token_id),
                "source_token": source_token,
                "variant": variant,
                "ids": [int(value) for value in encoded_ids],
                "decoded": tokenizer.decode([int(token_id)]),
            }
        )

    for source_token in resolve_key_tokens(cfg):
        if cfg.use_chat_template:
            normalized = _normalize_target_token_for_prompt_mode(source_token, cfg)

            if include_space_prefixed_variants:
                prefixed_ids = tokenizer.encode(f" {normalized}", add_special_tokens=False)
                if len(prefixed_ids) == 1:
                    _append_candidate(
                        int(prefixed_ids[0]),
                        label=f"▁{normalized}",
                        source_token=source_token,
                        variant="space_prefixed",
                        encoded_ids=prefixed_ids,
                    )

            if include_bare_variants:
                bare_ids = tokenizer.encode(normalized, add_special_tokens=False)
                if bare_ids:
                    bare_token_id = int(bare_ids[-1])
                    bare_label = normalized if len(bare_ids) == 1 else tokenizer.decode([bare_token_id]).lstrip()
                    _append_candidate(
                        bare_token_id,
                        label=bare_label,
                        source_token=source_token,
                        variant="bare",
                        encoded_ids=bare_ids,
                    )
            continue

        literal_ids = tokenizer.encode(source_token, add_special_tokens=False)
        if literal_ids:
            literal_token_id = int(literal_ids[-1])
            literal_label = source_token if len(literal_ids) == 1 else tokenizer.decode([literal_token_id]).lstrip()
            _append_candidate(
                literal_token_id,
                label=literal_label,
                source_token=source_token,
                variant="literal",
                encoded_ids=literal_ids,
            )

    return candidates


@dataclass(frozen=True)
class LocalExplanationPrefetchStatus:
    """Activation-cache readiness for one local Neuronpedia feature explanation."""

    feature_ref: NeuronpediaFeatureRef
    explanation_count: int
    cache_ready: bool
    cache_source: str
    cache_path: str | None = None
    activation_rows: int = 0
    error: str | None = None


@dataclass(frozen=True)
class LocalExplanationPreparationResult:
    """Summary of explanation availability and cache-prefetch readiness."""

    feature_refs: list[NeuronpediaFeatureRef]
    initial_statuses: list[NeuronpediaLocalExplanationStatus]
    prefetch_statuses: list[LocalExplanationPrefetchStatus]
    export_roots: tuple[str, ...]
    cache_dir: str

    @property
    def missing_feature_refs(self) -> list[NeuronpediaFeatureRef]:
        return [status.feature_ref for status in self.initial_statuses if not status.has_local_explanation]


def _resolve_local_export_roots(local_export_roots: Iterable[Path | str] | None = None) -> tuple[Path, ...]:
    candidate_roots = tuple(Path(root) for root in (local_export_roots or (DEFAULT_LOCAL_NEURONPEDIA_EXPORT_ROOT,)))
    return tuple(root for root in candidate_roots if root.exists())


def _feature_rows_to_layer_feature_tuples(feature_groups: Iterable[Any]) -> list[tuple[int, int]]:
    candidate_feature_tuples: list[tuple[int, int]] = []
    for feature_group in feature_groups:
        for feature_row in feature_group:
            normalized_row = tuple(int(value) for value in feature_row)
            if len(normalized_row) < 2:
                raise ValueError(f"Expected at least 2 values in feature row, got {normalized_row!r}")
            candidate_feature_tuples.append((normalized_row[0], normalized_row[-1]))
    return list(dict.fromkeys(candidate_feature_tuples))


def _select_feature_explanation(
    payload: Mapping[str, Any],
    *,
    preferred_type_name: str | None = None,
) -> dict[str, Any] | None:
    explanations = payload.get("explanations")
    if not isinstance(explanations, list):
        return None

    explanation_rows = [row for row in explanations if isinstance(row, Mapping)]
    if not explanation_rows:
        return None

    if preferred_type_name:
        preferred_rows = [
            row for row in explanation_rows if str(row.get("typeName") or "").strip() == preferred_type_name
        ]
        if preferred_rows:
            explanation_rows = preferred_rows

    for row in explanation_rows:
        description = row.get("description")
        if not isinstance(description, str) or not description.strip():
            continue
        return {
            "feature_explanation": clean_explanation_text(description),
            "feature_explanation_type_name": row.get("typeName"),
            "feature_explanation_model_name": row.get("explanationModelName"),
        }
    return None


def resolve_feature_dashboard_metadata(
    *,
    model_id: str,
    source_set: str,
    feature_rows: Iterable[Any],
    base_url: str = DEFAULT_NEURONPEDIA_BASE_URL,
    preferred_type_name: str | None = None,
    timeout_seconds: int = 15,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Resolve Neuronpedia dashboard URLs and best-effort explanation text for feature rows."""

    feature_tuples = _feature_rows_to_layer_feature_tuples(feature_rows)
    if not feature_tuples:
        return {}

    feature_refs = feature_tuples_to_feature_refs(
        model_id=model_id,
        source_set=source_set,
        feature_tuples=feature_tuples,
        base_url=base_url,
    )

    metadata_by_feature: dict[tuple[int, int], dict[str, Any]] = {}
    payload_cache: dict[str, Mapping[str, Any]] = {}

    for feature_tuple, feature_ref in zip(feature_tuples, feature_refs, strict=True):
        metadata = {
            "feature_url": feature_ref.feature_url,
            "feature_api_url": feature_ref.api_url,
            "feature_model_id": feature_ref.model_id,
            "feature_layer": feature_ref.layer,
            "feature_index": feature_ref.index,
        }
        try:
            payload = payload_cache.get(feature_ref.feature_url)
            if payload is None:
                payload = fetch_feature_payload(feature_ref, timeout_seconds=timeout_seconds)
                payload_cache[feature_ref.feature_url] = payload
        except Exception as exc:
            metadata["feature_fetch_error"] = f"{type(exc).__name__}: {exc}"
        else:
            selected_explanation = _select_feature_explanation(
                payload,
                preferred_type_name=preferred_type_name,
            )
            if selected_explanation is not None:
                metadata.update(selected_explanation)

        metadata_by_feature[(int(feature_tuple[0]), int(feature_tuple[1]))] = metadata

    return metadata_by_feature


def _populate_feature_cache_from_local_exports(
    feature_ref: NeuronpediaFeatureRef,
    *,
    export_roots: tuple[Path, ...],
    cache_dir: Path | None = None,
) -> tuple[int, Path] | None:
    for export_root in export_roots:
        activations_dir = export_root / feature_ref.model_id / feature_ref.layer / "activations"
        if not activations_dir.exists():
            continue
        for batch_path in sorted(activations_dir.glob("batch-*.jsonl.gz")):
            activation_rows = load_activation_batch_records(batch_path)
            matching_rows = [row for row in activation_rows if str(row.get("index")) == feature_ref.index]
            if not matching_rows:
                continue
            cache_path = write_cached_feature_activations(feature_ref, matching_rows, cache_dir=cache_dir)
            return len(matching_rows), cache_path
    return None


def prepare_local_explanation_backfill(
    cfg: NotebookHarnessConfig,
    *feature_groups: Any,
    cache_dir: Path | None = None,
    local_export_roots: Iterable[Path | str] | None = None,
    timeout_seconds: int = 60,
) -> LocalExplanationPreparationResult:
    """Collect top feature refs, inspect local explanation coverage, and prefetch activation cache rows."""

    feature_tuples = _feature_rows_to_layer_feature_tuples(feature_groups)
    feature_refs = feature_tuples_to_feature_refs(
        model_id=cfg.neuronpedia_model,
        source_set=cfg.neuronpedia_set,
        feature_tuples=feature_tuples,
        base_url=cfg.neuronpedia_base_url,
    )
    initial_statuses = check_local_explanation_coverage(
        feature_refs,
        local_db_url=cfg.local_neuronpedia_db_url,
        type_name=cfg.local_explanation_type_name,
    )
    resolved_export_roots = _resolve_local_export_roots(local_export_roots)
    prefetch_statuses: list[LocalExplanationPrefetchStatus] = []

    for explanation_status in initial_statuses:
        feature_ref = explanation_status.feature_ref
        if explanation_status.has_local_explanation:
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=True,
                    cache_source="existing_explanation",
                )
            )
            continue

        feature_cache_path = cached_feature_activation_path(feature_ref, cache_dir=cache_dir)
        existing_batch_paths = candidate_cached_activation_batch_paths(feature_ref, cache_dir=cache_dir)
        had_cache_before = feature_cache_path.exists() or any(
            batch_path.exists() for batch_path in existing_batch_paths
        )

        if had_cache_before:
            try:
                cached_rows, resolved_cache_path = load_cached_feature_activations(
                    feature_ref,
                    cache_dir=cache_dir,
                    timeout_seconds=timeout_seconds,
                )
                prefetch_statuses.append(
                    LocalExplanationPrefetchStatus(
                        feature_ref=feature_ref,
                        explanation_count=explanation_status.explanation_count,
                        cache_ready=True,
                        cache_source="existing_cache",
                        cache_path=str(resolved_cache_path),
                        activation_rows=len(cached_rows),
                    )
                )
                continue
            except Exception:
                pass

        local_export_result = _populate_feature_cache_from_local_exports(
            feature_ref,
            export_roots=resolved_export_roots,
            cache_dir=cache_dir,
        )
        if local_export_result is not None:
            activation_rows, resolved_cache_path = local_export_result
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=True,
                    cache_source="local_export_cache",
                    cache_path=str(resolved_cache_path),
                    activation_rows=activation_rows,
                )
            )
            continue

        try:
            cached_rows, resolved_cache_path = load_cached_feature_activations(
                feature_ref,
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
            )
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=True,
                    cache_source="existing_cache" if had_cache_before else "downloaded_public_cache",
                    cache_path=str(resolved_cache_path),
                    activation_rows=len(cached_rows),
                )
            )
            continue
        except Exception as exc:
            prefetch_statuses.append(
                LocalExplanationPrefetchStatus(
                    feature_ref=feature_ref,
                    explanation_count=explanation_status.explanation_count,
                    cache_ready=False,
                    cache_source="unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return LocalExplanationPreparationResult(
        feature_refs=feature_refs,
        initial_statuses=initial_statuses,
        prefetch_statuses=prefetch_statuses,
        export_roots=tuple(str(root) for root in resolved_export_roots),
        cache_dir=str(cache_dir or default_np_cache_dir()),
    )


def render_prompt(prompt: str, tokenizer: Any, mode: PromptRenderMode) -> str:
    if mode == "plain":
        return prompt
    chat_method = "gemma_dataclass" if mode == "gemma_dataclass" else "apply_chat_template"
    return _chattify(prompt, tokenizer, chat_method)


def render_prompt_variants(prompt: str, tokenizer: Any) -> dict[str, str | None]:
    gemma_cfg = GemmaPromptConfig()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    return {
        "plain": prompt,
        "apply_chat_template": gemma_cfg.apply_chat_template_fn(
            tokenizer,
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_chat_template
        else None,
        "gemma_dataclass": gemma_cfg.model_chat_template_fn(prompt, tokenization_pattern="gemma-chat"),
    }


def _tokenize_rendered_prompt(tokenizer: Any, rendered_prompt: str, mode: PromptRenderMode | str) -> list[int]:
    add_special_tokens = mode == "plain"
    return cast(list[int], tokenizer(rendered_prompt, add_special_tokens=add_special_tokens)["input_ids"])


def _build_prompt_batch(tokenizer: Any, rendered_prompt: str, mode: PromptRenderMode, device: Any) -> dict[str, Any]:
    add_special_tokens = mode == "plain"
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}


def _topk_token_summaries(logits: torch.Tensor, tokenizer: Any, *, k: int = 5) -> list[dict[str, Any]]:
    logits = logits.float().cpu()
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(logits, k)
    summaries: list[dict[str, Any]] = []
    for token_id, logit in zip(topk.indices.tolist(), topk.values.tolist(), strict=False):
        summaries.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "logit": float(logit),
                "prob": float(probs[int(token_id)].item()),
            }
        )
    return summaries


def _get_prompt_debugger(module: Any) -> DebugGeneration:
    from interpretune.extensions.debug_generation import DebugGeneration

    debug_lm = getattr(module, "debug_lm", None)
    if debug_lm is None:
        debug_lm = DebugGeneration()
        debug_lm.connect(module)
    return cast(DebugGeneration, debug_lm)


def _normalize_target_token_for_prompt_mode(token: str, cfg: NotebookHarnessConfig) -> str:
    if not cfg.use_chat_template:
        return token
    normalized = token.lstrip(" ▁Ġ")
    return normalized or token


def resolve_target_tokens(cfg: NotebookHarnessConfig, tokenizer: Any) -> tuple[tuple[int, int], tuple[str, str]]:
    if cfg.target_tokens is not None:
        resolved_tokens = tuple(_normalize_target_token_for_prompt_mode(token, cfg) for token in cfg.target_tokens)
        resolved_ids = tuple(tokenizer.encode(token, add_special_tokens=False)[-1] for token in resolved_tokens)
        return cast(tuple[int, int], resolved_ids), cast(tuple[str, str], resolved_tokens)
    assert cfg.target_token_ids is not None
    decoded_tokens = tuple(tokenizer.decode([token_id]) for token_id in cfg.target_token_ids)
    return cfg.target_token_ids, cast(tuple[str, str], decoded_tokens)


def resolve_explicit_direction_tokens(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
) -> tuple[tuple[int, int], tuple[str, str]]:
    if cfg.explicit_direction_tokens is None:
        raise ValueError("explicit_direction_tokens must be provided for explicit embedding-difference mode")
    resolved_tokens = tuple(
        _normalize_target_token_for_prompt_mode(token, cfg) for token in cfg.explicit_direction_tokens
    )
    resolved_ids = tuple(tokenizer.encode(token, add_special_tokens=False)[-1] for token in resolved_tokens)
    return cast(tuple[int, int], resolved_ids), cast(tuple[str, str], resolved_tokens)


def resolve_graph_target_tokens(cfg: NotebookHarnessConfig, tokenizer: Any) -> tuple[list[int], list[str]]:
    if not cfg.is_debug_intervention_mode:
        raise ValueError("resolve_graph_target_tokens is only available in debug_intervention_pipelines mode")

    ids: list[int] = []
    labels: list[str] = []
    seen_ids: set[int] = set()
    for token in resolve_key_tokens(cfg):
        resolved = _normalize_target_token_for_prompt_mode(token, cfg)
        encoded = tokenizer.encode(resolved, add_special_tokens=False)
        if not encoded:
            continue
        token_id = int(encoded[-1])
        if token_id in seen_ids:
            continue
        seen_ids.add(token_id)
        ids.append(token_id)
        labels.append(resolved)
    if not ids:
        raise ValueError("Unable to resolve any debug graph target tokens from KEY_TOKENS")
    return ids, labels


def get_key_token_ids_and_labels(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    *,
    include_bare_variants: bool = True,
) -> tuple[list[int], list[str]]:
    """Resolve key-token IDs and labels for reporting and logit displays.

    In chat mode this includes the single-token space-prefixed variant (for example
    ``▁Austin``) plus the bare completion token when available, even if the experiment config
    uses bare labels like ``Austin``.
    """
    candidates = _build_key_token_candidates(
        cfg,
        tokenizer,
        include_space_prefixed_variants=True,
        include_bare_variants=include_bare_variants,
    )
    return [entry["token_id"] for entry in candidates], [entry["label"] for entry in candidates]


def summarize_gap(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_a_id: int,
    target_b_id: int,
) -> tuple[float, float, float]:
    pre_gap = float((pre_logits[target_a_id] - pre_logits[target_b_id]).item())
    post_gap = float((post_logits[target_a_id] - post_logits[target_b_id]).item())
    return pre_gap, post_gap, post_gap - pre_gap


def configure_analysis(module: Any, graph_op: Any, scale_factor: float) -> None:
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = scale_factor
    module.circuit_tracer_cfg.intervention_constrained_layers = list(range(_resolve_model_layer_count(module)))
    module.circuit_tracer_cfg.intervention_apply_activation_function = False
    module.circuit_tracer_cfg.intervention_freeze_attention = None
    module.circuit_tracer_cfg.intervention_sparse = False
    module.circuit_tracer_cfg.intervention_return_activations = False
    module.analysis_cfg = AnalysisCfg(target_op=graph_op, ignore_manual=True, save_tokens=False)
    if not module.analysis_cfg.applied_to(module):
        init_analysis_cfgs(module, [module.analysis_cfg])


def _debug_intervention_artifact_name(
    cfg: NotebookHarnessConfig,
    feature_row: Sequence[int],
) -> str:
    feature_suffix = "_".join(str(int(value)) for value in feature_row)
    config_name = str(cfg.experiment_config_name or "manual").strip().replace(" ", "_")
    return f"{cfg.experiment_name}_{config_name}_{cfg.model_family}_{cfg.model_variant}_{feature_suffix}"


def _maybe_preserve_debug_intervention_artifacts(
    cfg: NotebookHarnessConfig,
    *,
    graph: Any,
    feature_row: Sequence[int],
    interventions: Sequence[Sequence[int | float]],
    baseline_activation_cache: torch.Tensor,
    intervention_activation_cache: torch.Tensor,
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    graph_target_ids: Sequence[int],
    graph_target_tokens: Sequence[str],
    selected_feature_score: float,
    selected_feature_activation: float,
    report: Any,
    runtime_state: dict[str, Any] | None = None,
) -> Path | None:
    artifact_dir = resolve_artifact_output_dir(
        artifact_name=_debug_intervention_artifact_name(cfg, feature_row),
    )
    if artifact_dir is None:
        return None

    metadata = {
        "artifact_kind": "concept_direction_debug_validation",
        "experiment_name": cfg.experiment_name,
        "experiment_config_name": cfg.experiment_config_name,
        "analysis_mode": cfg.analysis_mode,
        "model_family": cfg.model_family,
        "model_variant": cfg.model_variant,
        "model_name": cfg.model_name,
        "transcoder_set": cfg.transcoder_set,
        "prompt": cfg.prompt,
        "prompt_render_mode": cfg.prompt_render_mode,
        "graph_target_ids": [int(token_id) for token_id in graph_target_ids],
        "graph_target_tokens": [str(token) for token in graph_target_tokens],
        "selected_feature_score": float(selected_feature_score),
        "selected_feature_activation": float(selected_feature_activation),
        "requested_constrained_feature_selection": _serialize_constrained_feature_selection(
            cfg.constrained_feature_selection_refs
        ),
        "validation_tolerances": {
            "act_atol": cfg.debug_validation_act_atol,
            "act_rtol": cfg.debug_validation_act_rtol,
            "logit_atol": cfg.debug_validation_logit_atol,
            "logit_rtol": cfg.debug_validation_logit_rtol,
        },
        "runtime_state": runtime_state or {},
    }
    save_preserved_intervention_artifacts(
        artifact_dir,
        graph=graph,
        feature_row=feature_row,
        interventions=interventions,
        baseline_activation_cache=baseline_activation_cache,
        intervention_activation_cache=intervention_activation_cache,
        baseline_logits=baseline_logits,
        intervention_logits=intervention_logits,
        activation_atol=cfg.debug_validation_act_atol,
        activation_rtol=cfg.debug_validation_act_rtol,
        logit_atol=cfg.debug_validation_logit_atol,
        logit_rtol=cfg.debug_validation_logit_rtol,
        report=report,
        metadata=metadata,
    )
    return artifact_dir


@contextmanager
def maybe_zero_softcap(module: Any, cfg: NotebookHarnessConfig):
    if not cfg.enable_zero_softcap:
        yield
        return

    replacement_model = getattr(module, "replacement_model", None)
    zero_softcap = getattr(replacement_model, "zero_softcap", None)
    if callable(zero_softcap):
        zero_softcap_cm = zero_softcap()
        with cast(Any, zero_softcap_cm):
            yield
        return

    warning_flag = "_interpretune_zero_softcap_warning_emitted"
    if replacement_model is not None and not getattr(replacement_model, warning_flag, False):
        warnings.warn(
            "enable_zero_softcap was requested but the current replacement model does not expose zero_softcap(); "
            "continuing without it.",
            stacklevel=2,
        )
        setattr(replacement_model, warning_flag, True)
    yield


def _build_graph_analysis_inputs(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    rendered_prompt: str,
    *,
    direction: torch.Tensor | None,
    group_a_ids: list[int] | None,
    group_b_ids: list[int] | None,
    attribution_target_device: torch.device | str | None = None,
    attribution_targets: Any | None = None,
    graph_call_kwargs: Mapping[str, Any] | None = None,
    analysis_batch_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    batch_kwargs: dict[str, Any] = {"prompts": [rendered_prompt]}
    call_kwargs: dict[str, Any] = dict(graph_call_kwargs or {})

    if cfg.is_debug_intervention_mode:
        graph_target_ids, graph_target_tokens = resolve_graph_target_tokens(cfg, tokenizer)
        del graph_target_tokens
        attribution_target_ids = torch.tensor(graph_target_ids, dtype=torch.long)
        batch_kwargs["logit_target_ids"] = attribution_target_ids
        call_kwargs.setdefault(
            "attribution_targets",
            attribution_target_ids
            if attribution_target_device is None
            else attribution_target_ids.to(attribution_target_device),
        )
    else:
        any_direction_inputs = any(value is not None for value in (direction, group_a_ids, group_b_ids))
        if any_direction_inputs:
            if direction is None or group_a_ids is None or group_b_ids is None:
                raise ValueError("direction and concept-group token ids must either all be provided or all be omitted")
            batch_kwargs.update(
                concept_direction=direction,
                concept_label=cfg.analysis_concept_label,
                concept_direction_mode=cfg.analysis_direction_mode_name,
                concept_group_a_token_ids=group_a_ids,
                concept_group_b_token_ids=group_b_ids,
            )

    if attribution_targets is not None:
        resolved_attribution_targets = attribution_targets
        if attribution_target_device is not None and isinstance(resolved_attribution_targets, torch.Tensor):
            resolved_attribution_targets = resolved_attribution_targets.to(attribution_target_device)
        call_kwargs["attribution_targets"] = resolved_attribution_targets

    if analysis_batch_kwargs:
        batch_kwargs.update(dict(analysis_batch_kwargs))

    return it.AnalysisBatch(**batch_kwargs), call_kwargs


def _resolve_model_layer_count(module: Any) -> int:
    replacement_model = getattr(module, "replacement_model", None)
    model_cfg = getattr(replacement_model, "cfg", None)
    if model_cfg is not None:
        for attr_name in ("n_layers", "num_hidden_layers"):
            value = getattr(model_cfg, attr_name, None)
            if value is not None:
                return int(value)

    config = getattr(replacement_model, "config", None)
    for candidate in (config, getattr(config, "text_config", None) if config is not None else None):
        if candidate is None:
            continue
        value = getattr(candidate, "num_hidden_layers", None)
        if value is not None:
            return int(value)

    raise ValueError("Unable to resolve the replacement model layer count for debug intervention validation")


def _match_feature_row_index(active_features: torch.Tensor, feature_row: torch.Tensor) -> int:
    matches = (active_features == feature_row.reshape(1, 3)).all(dim=1).nonzero(as_tuple=False).reshape(-1)
    if matches.numel() != 1:
        raise ValueError(
            "Expected exactly one active-feature row to match the selected debug intervention feature; "
            f"found {int(matches.numel())} matches for {feature_row.tolist()}"
        )
    return int(matches.item())


def _rank_top_indices(values: torch.Tensor, top_k: int) -> torch.Tensor:
    flat_values = tensor_to_cpu(torch.as_tensor(values, dtype=torch.float32)).reshape(-1)
    if flat_values.numel() == 0 or top_k <= 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.argsort(flat_values, descending=True)[: min(int(top_k), flat_values.numel())]


def _summarize_feature_row_deltas(
    feature_rows: torch.Tensor,
    baseline_values: torch.Tensor,
    post_values: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    top_k: int,
    ranking: Literal["abs_error", "expected_delta", "actual_delta"] = "abs_error",
) -> list[dict[str, Any]]:
    feature_rows_t = tensor_to_cpu(torch.as_tensor(feature_rows, dtype=torch.long)).reshape(-1, 3)
    baseline_t = tensor_to_cpu(torch.as_tensor(baseline_values, dtype=torch.float32)).reshape(-1)
    post_t = tensor_to_cpu(torch.as_tensor(post_values, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = feature_rows_t.shape[0]
    if not (baseline_t.shape[0] == count == post_t.shape[0] == expected_t.shape[0]):
        raise ValueError("Feature-row diagnostic inputs must all have matching lengths")
    if count == 0:
        return []

    actual_delta_t = post_t - baseline_t
    predicted_t = baseline_t + expected_t
    signed_error_t = post_t - predicted_t
    abs_error_t = signed_error_t.abs()
    if ranking == "expected_delta":
        rank_values = expected_t.abs()
    elif ranking == "actual_delta":
        rank_values = actual_delta_t.abs()
    else:
        rank_values = abs_error_t

    rows: list[dict[str, Any]] = []
    for display_rank, graph_index in enumerate(_rank_top_indices(rank_values, top_k).tolist(), start=1):
        layer, position, feature_id = (int(value) for value in feature_rows_t[graph_index].tolist())
        expected_delta_value = float(expected_t[graph_index].item())
        actual_delta_value = float(actual_delta_t[graph_index].item())
        abs_error_value = float(abs_error_t[graph_index].item())
        rows.append(
            {
                "rank": display_rank,
                "graph_index": int(graph_index),
                "layer": layer,
                "position": position,
                "feature_id": feature_id,
                "row": [layer, position, feature_id],
                "baseline_activation": float(baseline_t[graph_index].item()),
                "predicted_activation": float(predicted_t[graph_index].item()),
                "post_activation": float(post_t[graph_index].item()),
                "expected_delta": expected_delta_value,
                "actual_delta": actual_delta_value,
                "abs_error": abs_error_value,
                "signed_error": float(signed_error_t[graph_index].item()),
                "relative_abs_error": abs_error_value / max(abs(expected_delta_value), 1e-12),
                "sign_mismatch": bool(expected_delta_value * actual_delta_value < 0.0),
                "ranking": ranking,
                "rank_metric": float(rank_values[graph_index].item()),
            }
        )
    return rows


def _summarize_layer_error_rows(
    feature_rows: torch.Tensor,
    baseline_values: torch.Tensor,
    post_values: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    feature_rows_t = tensor_to_cpu(torch.as_tensor(feature_rows, dtype=torch.long)).reshape(-1, 3)
    baseline_t = tensor_to_cpu(torch.as_tensor(baseline_values, dtype=torch.float32)).reshape(-1)
    post_t = tensor_to_cpu(torch.as_tensor(post_values, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = feature_rows_t.shape[0]
    if not (baseline_t.shape[0] == count == post_t.shape[0] == expected_t.shape[0]):
        raise ValueError("Layer diagnostic inputs must all have matching lengths")
    if count == 0:
        return []

    actual_delta_t = post_t - baseline_t
    abs_error_t = (post_t - (baseline_t + expected_t)).abs()
    summaries: list[dict[str, Any]] = []
    for layer_value in sorted({int(layer) for layer in feature_rows_t[:, 0].tolist()}):
        layer_mask = feature_rows_t[:, 0] == layer_value
        layer_errors = abs_error_t[layer_mask]
        layer_expected = expected_t[layer_mask]
        layer_actual = actual_delta_t[layer_mask]
        sign_mismatches = ((layer_expected * layer_actual) < 0.0).sum().item()
        summaries.append(
            {
                "layer": int(layer_value),
                "feature_count": int(layer_mask.sum().item()),
                "max_abs_error": float(layer_errors.max().item()),
                "mean_abs_error": float(layer_errors.mean().item()),
                "max_abs_expected_delta": float(layer_expected.abs().max().item()),
                "mean_abs_expected_delta": float(layer_expected.abs().mean().item()),
                "max_abs_actual_delta": float(layer_actual.abs().max().item()),
                "mean_abs_actual_delta": float(layer_actual.abs().mean().item()),
                "sign_mismatch_count": int(sign_mismatches),
            }
        )
    summaries.sort(key=lambda entry: entry["max_abs_error"], reverse=True)
    return summaries[: min(int(top_k), len(summaries))]


def _summarize_logit_delta_rows(
    token_ids: Sequence[int],
    token_labels: Sequence[str],
    baseline_logits: torch.Tensor,
    post_logits: torch.Tensor,
    baseline_demeaned_logits: torch.Tensor,
    post_demeaned_logits: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    baseline_logits_t = tensor_to_cpu(torch.as_tensor(baseline_logits, dtype=torch.float32)).reshape(-1)
    post_logits_t = tensor_to_cpu(torch.as_tensor(post_logits, dtype=torch.float32)).reshape(-1)
    baseline_demeaned_t = tensor_to_cpu(torch.as_tensor(baseline_demeaned_logits, dtype=torch.float32)).reshape(-1)
    post_demeaned_t = tensor_to_cpu(torch.as_tensor(post_demeaned_logits, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = len(token_ids)
    if not (
        baseline_logits_t.shape[0]
        == count
        == post_logits_t.shape[0]
        == baseline_demeaned_t.shape[0]
        == post_demeaned_t.shape[0]
        == expected_t.shape[0]
    ):
        raise ValueError("Logit diagnostic inputs must all have matching lengths")
    if count == 0:
        return []

    actual_delta_t = post_demeaned_t - baseline_demeaned_t
    predicted_t = baseline_demeaned_t + expected_t
    signed_error_t = post_demeaned_t - predicted_t
    abs_error_t = signed_error_t.abs()

    rows: list[dict[str, Any]] = []
    for display_rank, token_index in enumerate(_rank_top_indices(abs_error_t, top_k).tolist(), start=1):
        expected_delta_value = float(expected_t[token_index].item())
        actual_delta_value = float(actual_delta_t[token_index].item())
        abs_error_value = float(abs_error_t[token_index].item())
        rows.append(
            {
                "rank": display_rank,
                "token_id": int(token_ids[token_index]),
                "token": str(token_labels[token_index]),
                "baseline_logit": float(baseline_logits_t[token_index].item()),
                "post_logit": float(post_logits_t[token_index].item()),
                "baseline_demeaned_logit": float(baseline_demeaned_t[token_index].item()),
                "post_demeaned_logit": float(post_demeaned_t[token_index].item()),
                "expected_delta": expected_delta_value,
                "actual_delta": actual_delta_value,
                "abs_error": abs_error_value,
                "signed_error": float(signed_error_t[token_index].item()),
                "relative_abs_error": abs_error_value / max(abs(expected_delta_value), 1e-12),
                "sign_mismatch": bool(expected_delta_value * actual_delta_value < 0.0),
            }
        )
    return rows


def _summarize_same_feature_rows(
    feature_rows: torch.Tensor,
    baseline_values: torch.Tensor,
    post_values: torch.Tensor,
    expected_delta: torch.Tensor,
    *,
    layer: int,
    feature_id: int,
) -> list[dict[str, Any]]:
    feature_rows_t = tensor_to_cpu(torch.as_tensor(feature_rows, dtype=torch.long)).reshape(-1, 3)
    baseline_t = tensor_to_cpu(torch.as_tensor(baseline_values, dtype=torch.float32)).reshape(-1)
    post_t = tensor_to_cpu(torch.as_tensor(post_values, dtype=torch.float32)).reshape(-1)
    expected_t = tensor_to_cpu(torch.as_tensor(expected_delta, dtype=torch.float32)).reshape(-1)

    count = feature_rows_t.shape[0]
    if not (baseline_t.shape[0] == count == post_t.shape[0] == expected_t.shape[0]):
        raise ValueError("Same-feature diagnostic inputs must all have matching lengths")

    actual_delta_t = post_t - baseline_t
    rows: list[dict[str, Any]] = []
    for graph_index in (
        ((feature_rows_t[:, 0] == layer) & (feature_rows_t[:, 2] == feature_id))
        .nonzero(as_tuple=False)
        .reshape(-1)
        .tolist()
    ):
        rows.append(
            {
                "graph_index": int(graph_index),
                "row": [int(value) for value in feature_rows_t[graph_index].tolist()],
                "position": int(feature_rows_t[graph_index, 1].item()),
                "baseline_activation": float(baseline_t[graph_index].item()),
                "post_activation": float(post_t[graph_index].item()),
                "expected_delta": float(expected_t[graph_index].item()),
                "actual_delta": float(actual_delta_t[graph_index].item()),
                "abs_error": float(
                    (post_t[graph_index] - (baseline_t[graph_index] + expected_t[graph_index])).abs().item()
                ),
            }
        )
    rows.sort(key=lambda entry: (entry["position"], entry["graph_index"]))
    return rows


def _serialize_intervention_call_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            serialized[key] = tensor_fingerprint(value)
        elif isinstance(value, range):
            serialized[key] = {"kind": "range", "start": value.start, "stop": value.stop, "step": value.step}
        else:
            serialized[key] = value
    return serialized


def _summarize_graph_input_tokens(
    tokenizer: Any,
    rendered_prompt: str,
    prompt_render_mode: PromptRenderMode,
    graph_inputs: Any,
) -> dict[str, Any]:
    rendered_prompt_token_ids = _tokenize_rendered_prompt(tokenizer, rendered_prompt, prompt_render_mode)
    if isinstance(graph_inputs, torch.Tensor):
        graph_input_token_ids = tensor_to_cpu(torch.as_tensor(graph_inputs, dtype=torch.long)).reshape(-1).tolist()
        graph_input_source = "graph_result.input_tokens"
    else:
        graph_input_token_ids = _tokenize_rendered_prompt(tokenizer, str(graph_inputs), prompt_render_mode)
        graph_input_source = type(graph_inputs).__name__

    first_difference_index = next(
        (
            index
            for index, (rendered_token_id, graph_token_id) in enumerate(
                zip(rendered_prompt_token_ids, graph_input_token_ids, strict=False)
            )
            if rendered_token_id != graph_token_id
        ),
        None,
    )
    if first_difference_index is None and len(rendered_prompt_token_ids) != len(graph_input_token_ids):
        first_difference_index = min(len(rendered_prompt_token_ids), len(graph_input_token_ids))

    return {
        "graph_input_source": graph_input_source,
        "rendered_prompt_token_count": len(rendered_prompt_token_ids),
        "graph_input_token_count": len(graph_input_token_ids),
        "graph_inputs_match_rendered_prompt": rendered_prompt_token_ids == graph_input_token_ids,
        "first_difference_index": first_difference_index,
        "rendered_prompt_token_ids": rendered_prompt_token_ids,
        "graph_input_token_ids": graph_input_token_ids,
    }


def _parse_constrained_feature_selection_ref(
    raw_ref: ConstrainedFeatureSelectionRef,
    cfg: NotebookHarnessConfig,
) -> tuple[int, int]:
    ref_value, _ = _split_constrained_feature_selection_ref(raw_ref)
    layer_identifier: str
    feature_index: str
    if not isinstance(ref_value, str):
        model_id = ref_value[0]
        source_set = ref_value[1]
        layer_number = int(ref_value[2])
        feature_index_value = int(ref_value[3])
        if model_id != cfg.neuronpedia_model:
            raise ValueError(
                f"Constrained feature selection tuple {raw_ref!r} targets model {model_id}, "
                f"expected {cfg.neuronpedia_model}."
            )
        if source_set != cfg.neuronpedia_set:
            raise ValueError(
                f"Constrained feature selection tuple {raw_ref!r} targets source set {source_set}, "
                f"expected {cfg.neuronpedia_set}."
            )
        return int(layer_number), int(feature_index_value)

    if "://" in ref_value:
        feature_ref = parse_feature_url(ref_value)
        if feature_ref.model_id != cfg.neuronpedia_model:
            raise ValueError(
                f"Constrained feature selection ref {ref_value!r} targets model {feature_ref.model_id}, "
                f"expected {cfg.neuronpedia_model}."
            )
        layer_identifier = feature_ref.layer
        feature_index = feature_ref.index
    else:
        parts = [part for part in ref_value.split("/") if part]
        if len(parts) == 3:
            model_id, layer_identifier, feature_index = parts
            if model_id != cfg.neuronpedia_model:
                raise ValueError(
                    f"Constrained feature selection ref {ref_value!r} targets model {model_id}, "
                    f"expected {cfg.neuronpedia_model}."
                )
        elif len(parts) == 2:
            layer_identifier, feature_index = parts
        else:
            raise ValueError(
                "Constrained feature selection refs must be full Neuronpedia URLs or 'model/layer/index' or "
                "'layer/index' strings. "
                f"Got: {ref_value!r}"
            )

    layer_parts = str(layer_identifier).split("-", 1)
    if len(layer_parts) == 2 and layer_parts[1] != cfg.neuronpedia_set:
        raise ValueError(
            f"Constrained feature selection ref {raw_ref!r} targets source set {layer_parts[1]}, "
            f"expected {cfg.neuronpedia_set}."
        )
    layer_number = int(str(layer_identifier).split("-", 1)[0])
    return layer_number, int(feature_index)


def _build_feature_selection_spec(
    cfg: NotebookHarnessConfig,
    active_features: Any,
) -> FeatureSelectionSpec | None:
    requested_selection = _normalize_constrained_feature_selection(cfg.constrained_feature_selection_refs)
    if requested_selection is None:
        return None

    feature_rows = torch.as_tensor(active_features, dtype=torch.long)
    if feature_rows.numel() == 0:
        raise ValueError(
            "Feature-selection filtering was requested but the attribution graph produced no active features."
        )
    feature_rows = feature_rows.reshape(-1, 3)

    observed_layers = sorted(int(value) for value in feature_rows[:, 0].unique().tolist())
    observed_positions = sorted(int(value) for value in feature_rows[:, 1].unique().tolist())

    triples: list[tuple[int, int, int]] = list(requested_selection.triples)
    layer_feature_pairs: list[tuple[int, int]] = list(requested_selection.layer_feature_pairs)
    activation_overrides: dict[tuple[int, int], float] = dict(requested_selection.activation_overrides)
    layers: list[int] = list(requested_selection.layers)
    positions: list[int] = list(requested_selection.positions)
    feature_ids: list[int] = list(requested_selection.feature_ids)
    unmatched_refs: list[str] = []

    def _extend_values_for_slice(target: list[int], observed_values: list[int], raw_slice: slice) -> None:
        for value in observed_values:
            if raw_slice.start is not None and value < int(raw_slice.start):
                continue
            if raw_slice.stop is not None and value >= int(raw_slice.stop):
                continue
            if raw_slice.step not in (None, 1):
                base = int(raw_slice.start) if raw_slice.start is not None else observed_values[0]
                if (value - base) % int(raw_slice.step) != 0:
                    continue
            target.append(int(value))

    for raw_slice in requested_selection.layer_slices:
        _extend_values_for_slice(layers, observed_layers, raw_slice)
    for raw_slice in requested_selection.position_slices:
        _extend_values_for_slice(positions, observed_positions, raw_slice)

    for raw_feature in requested_selection.specific_features:
        if isinstance(raw_feature, ConstrainedFeatureSelectionRef):
            layer_number, feature_id = _parse_constrained_feature_selection_ref(raw_feature, cfg)
            layer_feature_pairs.append((layer_number, feature_id))
            _, activation_value = _split_constrained_feature_selection_ref(raw_feature)
            if activation_value is not None:
                activation_overrides[(layer_number, feature_id)] = float(activation_value)
            matches = feature_rows[(feature_rows[:, 0] == layer_number) & (feature_rows[:, 2] == feature_id)]
            if matches.numel() == 0:
                unmatched_refs.append(str(_serialize_constrained_feature_selection_ref(raw_feature)))
                continue
            for row in matches:
                row_tuple = tuple(int(value) for value in row.tolist())
                triples.append(cast(tuple[int, int, int], row_tuple))
            continue

        triple = cast(tuple[int, int, int], raw_feature)
        triples.append(triple)
        matches = feature_rows[
            (feature_rows[:, 0] == triple[0]) & (feature_rows[:, 1] == triple[1]) & (feature_rows[:, 2] == triple[2])
        ]
        if matches.numel() == 0:
            unmatched_refs.append(str(_serialize_constrained_feature_selection_specific_feature(triple)))

    if unmatched_refs:
        warnings.warn(
            "Some requested feature-selection refs were not present in the extracted attribution graph; "
            "extract_top_features will synthesize candidate rows for same-layer ref requests when possible: "
            + ", ".join(unmatched_refs),
            stacklevel=2,
        )

    unique_triples = list(dict.fromkeys(triples))
    unique_pairs = list(dict.fromkeys(layer_feature_pairs))
    unique_layers = list(dict.fromkeys(layers))
    unique_positions = list(dict.fromkeys(positions))
    unique_feature_ids = list(dict.fromkeys(feature_ids))
    return (
        FeatureSelectionSpec(
            layers=unique_layers,
            positions=unique_positions,
            feature_ids=unique_feature_ids,
            triples=unique_triples,
            layer_feature_pairs=unique_pairs,
            activation_overrides=activation_overrides,
        )
        if unique_layers or unique_positions or unique_feature_ids or unique_triples or unique_pairs
        else None
    )


def _extract_top_features_with_optional_filter(
    module: Any,
    cfg: NotebookHarnessConfig,
    top_payload: dict[str, Any],
    *,
    top_n: int,
) -> tuple[Any, list[tuple[int, int, int]]]:
    feature_selection = _build_feature_selection_spec(cfg, top_payload.get("active_features", []))
    call_kwargs: dict[str, Any] = {"top_n": top_n}
    applied_triples: list[tuple[int, int, int]] = []
    if feature_selection is not None:
        call_kwargs["feature_selection"] = feature_selection
        feature_rows = torch.as_tensor(top_payload.get("active_features", []), dtype=torch.long)
        if feature_rows.numel() > 0:
            feature_rows = feature_rows.reshape(-1, 3)
            applied_triples = [
                cast(tuple[int, int, int], tuple(int(value) for value in row.tolist()))
                for row in feature_rows[apply_feature_selection_filter(feature_rows, feature_selection)]
            ]

    top_features_result = cast(
        Any,
        it.extract_top_features(module, it.AnalysisBatch(**top_payload), cast(Any, None), 0, **call_kwargs),
    )
    return top_features_result, applied_triples


def _reduce_top_features_result_to_single_feature(result: Any) -> tuple[Any, int]:
    feature_ids = torch.as_tensor(getattr(result, "top_feature_ids", []), dtype=torch.long).reshape(-1, 3)
    feature_scores = torch.as_tensor(getattr(result, "top_feature_scores", []), dtype=torch.float32).reshape(-1)
    activation_values = getattr(result, "top_feature_activation_values", None)
    activation_tensor = (
        None if activation_values is None else torch.as_tensor(activation_values, dtype=torch.float32).reshape(-1)
    )

    candidate_count = int(feature_ids.shape[0])
    if candidate_count == 0:
        raise ValueError(
            "debug_intervention_pipelines mode expected at least one selected feature row after filtering."
        )
    if candidate_count == 1:
        return result, candidate_count

    if feature_scores.shape[0] == feature_ids.shape[0]:
        selected_index = int(torch.argmax(feature_scores.abs()).item())
    else:
        selected_index = 0

    selected_indices = torch.tensor([selected_index], dtype=torch.long)
    setattr(result, "top_feature_ids", feature_ids.index_select(0, selected_indices).detach().cpu())
    if feature_scores.shape[0] == feature_ids.shape[0]:
        setattr(result, "top_feature_scores", feature_scores.index_select(0, selected_indices).detach().cpu())
    if activation_tensor is not None and activation_tensor.shape[0] == feature_ids.shape[0]:
        setattr(
            result,
            "top_feature_activation_values",
            activation_tensor.index_select(0, selected_indices).detach().cpu(),
        )
    return result, candidate_count
