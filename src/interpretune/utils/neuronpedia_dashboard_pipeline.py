from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import (
    MISSING as DATACLASS_MISSING,
    dataclass,
    field,
    fields as dataclass_fields,
    replace as dataclass_replace,
)
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from urllib.parse import urlparse

import certifi
import yaml  # type: ignore[import-untyped]
from sae_dashboard.neuronpedia.prompt_bucketing import derive_prompt_bucket_ceilings

from interpretune.utils.neuronpedia_db_utils import (
    DEFAULT_COLUMNAR_COPY_IMPORT_TABLES,
    DEFAULT_COLUMNAR_IMPORT_TABLES,
    DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL,
    LocalNeuronpediaServiceStatus,
    NeuronpediaLocalImportSummary,
    check_local_neuronpedia_services,
    import_neuronpedia_export_bundle_local_db,
    import_saedashboard_columnar_bundle_local_db,
)
from interpretune.utils.neuronpedia_explanations import (
    DEFAULT_EXPLANATION_AUTHOR_ID,
    DEFAULT_IT_NP_CACHE,
)

DONE_LAYER_RE = re.compile(r"\bDONE layer=(\d+)\b")
OOM_LOG_RE = re.compile(r"(CUDA out of memory|out of memory|torch\.OutOfMemoryError)", re.IGNORECASE)
BATCH_JSON_RE = re.compile(r"^batch-\d+\.json$")
LEGACY_LOCAL_DATASET_ALIAS_SOURCE_FILE = ".interpretune_legacy_alias_source"
CONFIG_EXTENDS_KEY = "EXTENDS"
PIPELINE_CONFIG_SECTION = "pipeline"
LAUNCHER_CONFIG_SECTION = "launcher"
RUN_SETTINGS_FILE = "run_settings.json"
DEFAULT_LOCAL_DB_COLUMNAR_IMPORT_TABLES = tuple(sorted(DEFAULT_COLUMNAR_IMPORT_TABLES))
DEFAULT_LOCAL_DB_COLUMNAR_COPY_TABLES = tuple(sorted(DEFAULT_COLUMNAR_COPY_IMPORT_TABLES))
DEFAULT_DASHBOARD_EXPORT_ROOT_ENV_VARS = ("NEURONPEDIA_EXPORT_ROOT",)
PROMPT_DATASET_MODES = ("load_dataset", "load_from_disk", "legacy_jsonl")
LEGACY_EXPORT_BUNDLE_CONTRACTS = ("auto", "preserved_baseline")


def _valid_prompts_dataset_mode_values() -> str:
    return ", ".join(PROMPT_DATASET_MODES)


def _is_legacy_jsonl_dataset_path(dataset_path: str) -> bool:
    candidate_path = Path(dataset_path).expanduser()
    try:
        resolved_path = candidate_path.resolve(strict=True)
    except OSError:
        return False
    return resolved_path.is_dir() and (resolved_path / "sae_lens.json").exists() and any(resolved_path.glob("*.jsonl"))


def _warn_deprecated_legacy_jsonl() -> None:
    warnings.warn(
        "prompts_dataset_mode='legacy_jsonl' is deprecated and exists only for legacy JSONL dashboard "
        "compatibility. Prefer 'load_dataset' for Hugging Face/local datasets or 'load_from_disk' for "
        "save_to_disk() prompt caches.",
        DeprecationWarning,
        stacklevel=3,
    )


def _default_dashboard_export_root() -> Path:
    for env_var in DEFAULT_DASHBOARD_EXPORT_ROOT_ENV_VARS:
        env_value = os.getenv(env_var)
        if env_value:
            return Path(env_value).expanduser()
    return Path(os.getenv("IT_NP_CACHE", str(DEFAULT_IT_NP_CACHE))).expanduser() / "exports"


def _default_repo_root(env_var: str, *relative_to_home: str) -> Path:
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value).expanduser()
    return Path.home().joinpath(*relative_to_home)


def _default_interpretune_env_file() -> Path | None:
    env_value = os.getenv("IT_ENV_FILE")
    if env_value:
        return Path(env_value).expanduser()
    candidate = Path(__file__).resolve().parents[3] / ".env"
    return candidate if candidate.exists() else None


@dataclass(frozen=True)
class NeuronpediaDashboardLayerResult:
    """Result metadata for one generated layer."""

    layer_num: int
    output_dir: Path
    export_root: Path | None
    import_summary: NeuronpediaLocalImportSummary | None
    elapsed_seconds: float
    skipped: bool = False


@dataclass(frozen=True)
class SharedPromptRunSettings:
    """Resolved prompt-token inputs and batch sizes for one runner invocation."""

    shared_tokens_file: Path | None
    n_prompts_total: int
    n_tokens_in_prompt: int
    n_prompts_in_forward_pass: int
    primary_acts_batch_size: int | None
    prompt_bucket_ceiling: int | None = None
    bucket_prompt_count: int | None = None
    effective_length_min: int | None = None
    effective_length_max: int | None = None


@dataclass(kw_only=True)
class NeuronpediaDashboardPipelineConfig:
    """Configuration for generating, converting, and importing Neuronpedia dashboard layers."""

    model_name: str
    model_layers: int
    sae_set: str
    neuronpedia_source_set_id: str
    neuronpedia_source_set_description: str
    creator_name: str
    release_id: str
    release_title: str
    release_url: str
    hf_weights_repo_id: str
    hf_weights_path_template: str
    hook_point: str
    prompts_huggingface_dataset_path: str
    start_layer: int
    end_layer: int
    # Optional explicit (possibly non-contiguous) layer list; overrides start_layer/end_layer iteration.
    layer_list: list[int] | None = None
    sae_path_template: str
    prompts_dataset_mode: str = "load_dataset"
    hf_model_path: str | None = None
    prompts_huggingface_dataset_config_name: str | None = None
    prompts_huggingface_dataset_split: str | None = None
    prompts_dataset_text_field: str | None = None
    prompts_pretokenized_dataset_path: Path | None = None
    prompts_shared_tokens_file: Path | None = None
    run_root: Path = DEFAULT_IT_NP_CACHE / "dashboard_runs"
    run_name_suffix: str | None = None
    export_root: Path = field(default_factory=_default_dashboard_export_root)
    existing_log_path: Path | None = None
    pipeline_log_path: Path | None = None
    worker_id: str | None = None
    enable_layer_locks: bool = False
    layer_lock_stale_seconds: int = 0
    saedashboard_repo_root: Path = field(
        default_factory=lambda: _default_repo_root("SAEDASHBOARD_REPO_ROOT", "repos", "SAEDashboard")
    )
    saelens_repo_root: Path = field(default_factory=lambda: _default_repo_root("SAELENS_REPO_ROOT", "repos", "SAELens"))
    neuronpedia_utils_root: Path = field(
        default_factory=lambda: _default_repo_root(
            "NEURONPEDIA_UTILS_ROOT", "repos", "neuronpedia", "utils", "neuronpedia-utils"
        )
    )
    interpretune_env_file: Path | None = field(default_factory=_default_interpretune_env_file)
    python_executable: str = sys.executable
    use_skip_transcoder: bool = False
    sae_dtype: str = "float32"
    model_dtype: str = "bfloat16"
    sparsity_threshold: int = 1
    n_prompts_total: int = 24576
    n_tokens_in_prompt: int = 128
    n_features_per_batch: int = 128
    n_prompts_in_forward_pass: int = 32
    primary_acts_batch_size: int | None = None
    start_batch: int = 0
    end_batch: int | None = None
    zero_out_bos_token: bool = False
    use_clt: bool = False
    clt_dtype: str = ""
    clt_weights_filename: str = ""
    dataset_streaming: bool = True
    model_wrapper: str = "hooked"
    bridge_enable_compatibility_mode: bool = True
    bridge_compatibility_mode_kwargs: dict[str, Any] = field(default_factory=lambda: {"no_processing": True})
    runner_log_resource_snapshots: bool = False
    runner_log_hook_aliases: bool = False
    runner_log_performance: bool = False
    runner_profile_rolling_substages: bool = False
    runner_shuffle_tokens: bool = True
    runner_implementation: str = "current"
    runner_cleanup_each_minibatch: bool = False
    runner_correlation_accumulation_device: str = "auto"
    runner_rolling_coefficient_num_threads: int | None = None
    runner_activation_significance_floor: float = 0.0
    runner_converter_input_artifact_dir: Path | None = None
    runner_feature_statistics_backend: str = "arrow"
    runner_logits_histogram_backend: str = "arrow"
    runner_defer_component_construction: bool = False
    runner_sequence_selection_backend: str = "columnar_gpu"
    runner_dashboard_output_format: str = "auto"
    legacy_export_bundle_contract: str = "auto"
    runner_columnar_artifact_format: str = "parquet"
    runner_emit_activation_copy_rows: bool | None = None
    runner_overlap_batch_packaging: bool = False
    # Opt-in SAEDashboard selection/logits hygiene (columnar backend only; defaults off to
    # preserve the baseline selection/parity contract — see the Phase 7 quality investigation).
    runner_sequence_top_acts_positive_only: bool = False
    runner_sequence_dedup_across_groups: bool = False
    runner_sequence_skip_dead_features: bool = False
    runner_sequence_half_open_interval_bins: bool = False
    # Opt-in SAEDashboard columnar peak-GPU-memory controls (bit-identical outputs; None keeps
    # the SD-side fixed 4 GiB device budgets / packaging chunk shapes). The byte budget caps
    # device retention/staging of the activation matrix (0 forces host staging) so dense layers
    # fit at large n_prompts; the row chunk bounds density-scaled packaging transients.
    runner_columnar_max_staged_acts_bytes: int | None = None
    runner_columnar_row_chunk_size: int | None = None
    runner_logits_table_mask_token_pattern: str | None = None
    # Opt-in neuronpedia-utils import hygiene (defaults off to preserve the import contract).
    local_db_import_dedup_activation_rows: bool = False
    local_db_import_drop_zero_activation_rows: bool = False
    runner_prompt_bucket_schedule_file: Path | None = None
    runner_auto_prompt_bucket_schedule: bool = False
    runner_prompt_bucket_ceilings: tuple[int, ...] = field(default_factory=tuple)
    runner_prompt_bucket_scale_limit: float = 4.0
    runner_prompt_primary_acts_scale_limit: float = 4.0
    runner_prompt_batch_size_round_to: int = 8
    runner_torch_profile: bool = False
    runner_torch_profile_dir: Path | None = None
    runner_use_cached_activations: bool = True
    cuda_visible_devices: str | None = "0"
    heartbeat_seconds: int = 60
    stall_timeout_seconds: int = 0
    import_to_local_db: bool = True
    # Opt-in silent-degradation acceptance: when the local DB is unavailable and
    # import_to_local_db is true, warn and continue generation-only instead of hard-erroring.
    # Default off because a run that silently skips DB import is easy to mistake for a full run.
    allow_missing_local_db: bool = False
    local_db_url: str | None = None
    local_db_import_chunk_size: int = 65000
    overlap_local_db_import: bool = False
    webapp_url: str = DEFAULT_LOCAL_NEURONPEDIA_WEBAPP_URL
    archive_partial_dirs: bool = True
    resume_from_existing_logs: bool = True
    deduplicate_shared_prompt_tokens: bool = True
    strict_shared_prompt_count: bool = False
    prompt_bucket_ceilings: tuple[int, ...] = field(default_factory=tuple)
    prompt_bucket_ceiling: int | None = None
    cert_bundle_path: Path = field(default_factory=lambda: Path(certifi.where()))
    torch_cuda_alloc_conf: str | None = "expandable_segments:True"
    import_only_local_db: bool = False

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root)
        self.export_root = Path(self.export_root)
        self.saedashboard_repo_root = Path(self.saedashboard_repo_root)
        self.saelens_repo_root = Path(self.saelens_repo_root)
        self.neuronpedia_utils_root = Path(self.neuronpedia_utils_root)
        if self.run_name_suffix is not None:
            self.run_name_suffix = str(self.run_name_suffix).strip()
            if not self.run_name_suffix:
                self.run_name_suffix = None
            elif any(separator in self.run_name_suffix for separator in (os.sep, os.altsep) if separator):
                raise ValueError("run_name_suffix must not contain path separators.")
        if self.worker_id is not None:
            self.worker_id = str(self.worker_id).strip()
            if not self.worker_id:
                self.worker_id = None
            elif any(separator in self.worker_id for separator in (os.sep, os.altsep) if separator):
                raise ValueError("worker_id must not contain path separators.")
        if self.prompts_pretokenized_dataset_path is not None:
            self.prompts_pretokenized_dataset_path = Path(self.prompts_pretokenized_dataset_path)
        self.prompts_dataset_mode = str(self.prompts_dataset_mode).strip() or "load_dataset"
        if self.prompts_dataset_mode not in PROMPT_DATASET_MODES:
            raise ValueError(
                f"prompts_dataset_mode must be one of {_valid_prompts_dataset_mode_values()}; "
                f"got {self.prompts_dataset_mode!r}."
            )
        self.legacy_export_bundle_contract = str(self.legacy_export_bundle_contract).strip() or "auto"
        if self.legacy_export_bundle_contract not in LEGACY_EXPORT_BUNDLE_CONTRACTS:
            raise ValueError(
                "legacy_export_bundle_contract must be one of "
                f"{', '.join(LEGACY_EXPORT_BUNDLE_CONTRACTS)}; got {self.legacy_export_bundle_contract!r}."
            )
        if self.prompts_shared_tokens_file is not None:
            self.prompts_shared_tokens_file = Path(self.prompts_shared_tokens_file)
        if self.existing_log_path is not None:
            self.existing_log_path = Path(self.existing_log_path)
        if self.pipeline_log_path is not None:
            self.pipeline_log_path = Path(self.pipeline_log_path)
        if self.interpretune_env_file is not None:
            self.interpretune_env_file = Path(self.interpretune_env_file)
        if self.runner_prompt_bucket_schedule_file is not None:
            self.runner_prompt_bucket_schedule_file = Path(self.runner_prompt_bucket_schedule_file)
        if isinstance(self.prompt_bucket_ceilings, str):
            raise TypeError("prompt_bucket_ceilings must be normalized before config construction.")
        if isinstance(self.runner_prompt_bucket_ceilings, str):
            raise TypeError("runner_prompt_bucket_ceilings must be normalized before config construction.")
        self.prompt_bucket_ceilings = tuple(int(value) for value in self.prompt_bucket_ceilings)
        self.runner_prompt_bucket_ceilings = tuple(int(value) for value in self.runner_prompt_bucket_ceilings)
        if self.prompt_bucket_ceiling is not None:
            self.prompt_bucket_ceiling = int(self.prompt_bucket_ceiling)
            if self.prompt_bucket_ceiling <= 0:
                raise ValueError("prompt_bucket_ceiling must be positive when provided.")
        if self.runner_prompt_bucket_scale_limit <= 0:
            raise ValueError("runner_prompt_bucket_scale_limit must be positive.")
        if self.runner_prompt_primary_acts_scale_limit <= 0:
            raise ValueError("runner_prompt_primary_acts_scale_limit must be positive.")
        if self.runner_prompt_batch_size_round_to <= 0:
            raise ValueError("runner_prompt_batch_size_round_to must be positive.")
        if self.runner_rolling_coefficient_num_threads is not None:
            self.runner_rolling_coefficient_num_threads = int(self.runner_rolling_coefficient_num_threads)
            if self.runner_rolling_coefficient_num_threads <= 0:
                raise ValueError("runner_rolling_coefficient_num_threads must be positive when provided.")
        self.cert_bundle_path = Path(self.cert_bundle_path)
        if self.existing_log_path is None:
            self.existing_log_path = self.run_directory / "run.log"
        if self.pipeline_log_path is None:
            worker_segment = f".{self.worker_id}" if self.worker_id else ""
            log_name = f"run{worker_segment}.resume-{self.start_layer}-{self.end_layer}.log"
            self.pipeline_log_path = self.run_directory / log_name

    @property
    def run_name(self) -> str:
        base_name = f"{self.model_name}_{self.neuronpedia_source_set_id}"
        if self.run_name_suffix:
            return f"{base_name}_{self.run_name_suffix}"
        return base_name

    @property
    def run_directory(self) -> Path:
        return self.run_root / self.run_name

    def sae_path_for_layer(self, layer_num: int) -> str:
        return self.sae_path_template.format(layer=layer_num)

    def hf_weights_path_for_layer(self, layer_num: int) -> str:
        return self.hf_weights_path_template.format(layer=layer_num)

    def output_dir_for_layer(self, layer_num: int) -> Path:
        return self.run_directory / f"layer_{layer_num}"

    def layer_lock_path(self, layer_num: int) -> Path:
        return self.run_directory / "layer_locks" / f"layer_{layer_num}.lock"

    def requested_layer_numbers(self) -> list[int]:
        if self.layer_list:
            return list(dict.fromkeys(self.layer_list))
        return list(range(self.start_layer, self.end_layer + 1))

    @property
    def shared_prompt_tokens_file(self) -> Path | None:
        if self.prompts_shared_tokens_file is not None:
            return self.prompts_shared_tokens_file
        if self.prompts_pretokenized_dataset_path is None:
            return None
        return self.prompts_pretokenized_dataset_path / f"tokens_{self.n_prompts_total}.pt"

    @property
    def prompts_dataset_identifier(self) -> str:
        dataset_id = self.prompts_huggingface_dataset_path
        if self.prompts_huggingface_dataset_config_name:
            dataset_id = f"{dataset_id}:{self.prompts_huggingface_dataset_config_name}"
        if self.prompts_huggingface_dataset_split:
            dataset_id = f"{dataset_id}[{self.prompts_huggingface_dataset_split}]"
        if self.prompts_dataset_text_field:
            dataset_id = f"{dataset_id}#text_field={self.prompts_dataset_text_field}"
        if self.prompts_pretokenized_dataset_path:
            dataset_id = f"{dataset_id}#pretokenized={self.prompts_pretokenized_dataset_path}"
        if self.prompts_shared_tokens_file:
            dataset_id = f"{dataset_id}#shared_tokens={self.prompts_shared_tokens_file}"
        dataset_id = f"{dataset_id}#dataset_mode={_resolve_prompts_dataset_mode(self)}"
        return dataset_id


PIPELINE_CONFIG_FIELD_NAMES = {field_info.name for field_info in dataclass_fields(NeuronpediaDashboardPipelineConfig)}
REQUIRED_PIPELINE_CONFIG_FIELD_NAMES = {
    field_info.name
    for field_info in dataclass_fields(NeuronpediaDashboardPipelineConfig)
    if field_info.default is DATACLASS_MISSING and field_info.default_factory is DATACLASS_MISSING
}
PIPELINE_CONFIG_ALIAS_NAMES = {
    "bridge_compatibility_mode_kwargs_json",
    "skip_local_db_import",
    "no_archive_partials",
    "no_resume",
}
ALLOWED_PIPELINE_CONFIG_KEYS = PIPELINE_CONFIG_FIELD_NAMES | PIPELINE_CONFIG_ALIAS_NAMES
ALLOWED_TOP_LEVEL_CONFIG_KEYS = ALLOWED_PIPELINE_CONFIG_KEYS | {
    PIPELINE_CONFIG_SECTION,
    LAUNCHER_CONFIG_SECTION,
}
ALLOWED_LAUNCHER_CONFIG_KEYS = {"background", "env", "log_path", "monitor", "monitor_heartbeat_seconds", "workers"}


def _resolve_prompts_dataset_mode(config: NeuronpediaDashboardPipelineConfig) -> str:
    resolved_mode = config.prompts_dataset_mode
    if resolved_mode == "load_dataset":
        if _is_legacy_jsonl_dataset_path(config.prompts_huggingface_dataset_path):
            resolved_mode = "legacy_jsonl"
        elif config.prompts_pretokenized_dataset_path is not None and config.runner_implementation != "legacy":
            resolved_mode = "load_from_disk"

    if resolved_mode == "legacy_jsonl":
        _warn_deprecated_legacy_jsonl()

    if resolved_mode == "load_from_disk" and config.runner_implementation == "legacy":
        raise ValueError(
            "legacy does not accept prompts_dataset_mode='load_from_disk'; use load_dataset or legacy_jsonl instead."
        )
    if resolved_mode == "load_from_disk" and config.prompts_pretokenized_dataset_path is None:
        raise ValueError("prompts_dataset_mode='load_from_disk' requires prompts_pretokenized_dataset_path to be set.")
    return resolved_mode


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must parse to a mapping.")
    return dict(payload)


def _deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge_mappings(base_value, value)
        else:
            merged[key] = value
    return merged


def _resolve_config_extends_paths(config_path: Path, extends_value: Any) -> list[Path]:
    if extends_value is None:
        return []
    if isinstance(extends_value, str):
        raw_values = [extends_value]
    elif isinstance(extends_value, list):
        raw_values = [str(item) for item in extends_value]
    else:
        raise ValueError(f"{CONFIG_EXTENDS_KEY} in {config_path} must be a string or list of strings.")

    resolved_paths: list[Path] = []
    for raw_value in raw_values:
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (config_path.parent / candidate).resolve()
        resolved_paths.append(candidate)
    return resolved_paths


def _expand_config_env_vars(value: Any) -> Any:
    """Expand ``${VAR}``/``$VAR`` references in string config values (unset vars are left literal)."""

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, Mapping):
        return {key: _expand_config_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_env_vars(item) for item in value]
    return value


def load_dashboard_pipeline_config_payload(config_path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a dashboard pipeline YAML config with EXTENDS inheritance."""

    resolved_path = Path(config_path).expanduser().resolve()
    if resolved_path in _seen:
        chain = " -> ".join(str(path) for path in (*_seen, resolved_path))
        raise ValueError(f"Detected cyclic config inheritance: {chain}")

    payload = _expand_config_env_vars(_load_yaml_mapping(resolved_path))
    extends_value = payload.pop(CONFIG_EXTENDS_KEY, None)

    merged_payload: dict[str, Any] = {}
    for parent_path in _resolve_config_extends_paths(resolved_path, extends_value):
        merged_payload = _deep_merge_mappings(
            merged_payload,
            load_dashboard_pipeline_config_payload(parent_path, _seen=(*_seen, resolved_path)),
        )

    return _deep_merge_mappings(merged_payload, payload)


def _extract_dashboard_pipeline_values(payload: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    raw_pipeline_section = payload.get(PIPELINE_CONFIG_SECTION, {})
    if raw_pipeline_section is None:
        raw_pipeline_section = {}
    if not isinstance(raw_pipeline_section, Mapping):
        raise ValueError(f"Config {config_path} must define '{PIPELINE_CONFIG_SECTION}' as a mapping when present.")

    raw_launcher_section = payload.get(LAUNCHER_CONFIG_SECTION, {})
    if raw_launcher_section is None:
        raw_launcher_section = {}
    if not isinstance(raw_launcher_section, Mapping):
        raise ValueError(f"Config {config_path} must define '{LAUNCHER_CONFIG_SECTION}' as a mapping when present.")
    unknown_launcher_keys = sorted(set(raw_launcher_section) - ALLOWED_LAUNCHER_CONFIG_KEYS)
    if unknown_launcher_keys:
        raise ValueError(f"Config {config_path} has unknown launcher keys: {', '.join(unknown_launcher_keys)}")

    unknown_top_level_keys = sorted(set(payload) - ALLOWED_TOP_LEVEL_CONFIG_KEYS)
    if unknown_top_level_keys:
        raise ValueError(f"Config {config_path} has unknown top-level keys: {', '.join(unknown_top_level_keys)}")

    unknown_pipeline_keys = sorted(set(raw_pipeline_section) - ALLOWED_PIPELINE_CONFIG_KEYS)
    if unknown_pipeline_keys:
        raise ValueError(f"Config {config_path} has unknown pipeline keys: {', '.join(unknown_pipeline_keys)}")

    flat_pipeline_values = {key: value for key, value in payload.items() if key in ALLOWED_PIPELINE_CONFIG_KEYS}
    return _deep_merge_mappings(flat_pipeline_values, raw_pipeline_section)


def _normalize_mapping_override(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        parsed_value = json.loads(value)
    elif isinstance(value, Mapping):
        parsed_value = dict(value)
    else:
        raise TypeError(f"{field_name} must be a mapping or JSON object string.")
    if not isinstance(parsed_value, dict):
        raise ValueError(f"{field_name} must resolve to a mapping.")
    return dict(parsed_value)


def _normalize_int_sequence_override(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return ()
        if stripped_value.startswith("["):
            parsed_value = json.loads(stripped_value)
        else:
            parsed_value = [segment.strip() for segment in stripped_value.split(",") if segment.strip()]
    else:
        parsed_value = value
    if not isinstance(parsed_value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list, tuple, or comma-separated string.")
    return tuple(int(item) for item in parsed_value)


def _normalize_pipeline_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(values)

    if "bridge_compatibility_mode_kwargs_json" in normalized:
        bridge_kwargs = normalized.pop("bridge_compatibility_mode_kwargs_json")
        normalized.setdefault("bridge_compatibility_mode_kwargs", bridge_kwargs)

    if "skip_local_db_import" in normalized:
        skip_local_db_import = bool(normalized.pop("skip_local_db_import"))
        normalized.setdefault("import_to_local_db", not skip_local_db_import)
    if "no_archive_partials" in normalized:
        no_archive_partials = bool(normalized.pop("no_archive_partials"))
        normalized.setdefault("archive_partial_dirs", not no_archive_partials)
    if "no_resume" in normalized:
        no_resume = bool(normalized.pop("no_resume"))
        normalized.setdefault("resume_from_existing_logs", not no_resume)

    if "bridge_compatibility_mode_kwargs" in normalized:
        normalized["bridge_compatibility_mode_kwargs"] = _normalize_mapping_override(
            normalized["bridge_compatibility_mode_kwargs"],
            field_name="bridge_compatibility_mode_kwargs",
        )
    if "prompt_bucket_ceilings" in normalized:
        normalized["prompt_bucket_ceilings"] = _normalize_int_sequence_override(
            normalized["prompt_bucket_ceilings"],
            field_name="prompt_bucket_ceilings",
        )
    if "runner_prompt_bucket_ceilings" in normalized:
        normalized["runner_prompt_bucket_ceilings"] = _normalize_int_sequence_override(
            normalized["runner_prompt_bucket_ceilings"],
            field_name="runner_prompt_bucket_ceilings",
        )

    return normalized


def _build_dashboard_pipeline_config(args: argparse.Namespace) -> NeuronpediaDashboardPipelineConfig:
    cli_values = dict(vars(args))
    config_path_value = cli_values.pop("config", None)

    merged_values: dict[str, Any] = {}
    if config_path_value is not None:
        resolved_config_path = Path(config_path_value).expanduser().resolve()
        config_payload = load_dashboard_pipeline_config_payload(resolved_config_path)
        merged_values.update(
            _normalize_pipeline_overrides(
                _extract_dashboard_pipeline_values(config_payload, config_path=resolved_config_path)
            )
        )

    merged_values.update(_normalize_pipeline_overrides(cli_values))

    missing_required_fields = sorted(
        field_name for field_name in REQUIRED_PIPELINE_CONFIG_FIELD_NAMES if field_name not in merged_values
    )
    if missing_required_fields:
        raise ValueError("Missing required dashboard pipeline config values: " + ", ".join(missing_required_fields))

    return NeuronpediaDashboardPipelineConfig(**merged_values)


def load_dashboard_launcher_settings(
    config_path: str | Path,
    *,
    pipeline_config: NeuronpediaDashboardPipelineConfig | None = None,
) -> dict[str, Any]:
    """Load launcher-only settings from a dashboard pipeline config file."""

    resolved_config_path = Path(config_path).expanduser().resolve()
    payload = load_dashboard_pipeline_config_payload(resolved_config_path)
    raw_launcher_section = payload.get(LAUNCHER_CONFIG_SECTION, {})
    if raw_launcher_section is None:
        raw_launcher_section = {}
    if not isinstance(raw_launcher_section, Mapping):
        raise ValueError(
            f"Config {resolved_config_path} must define '{LAUNCHER_CONFIG_SECTION}' as a mapping when present."
        )

    launcher_env = raw_launcher_section.get("env", {})
    if launcher_env is None:
        launcher_env = {}
    if not isinstance(launcher_env, Mapping):
        raise ValueError(f"Config {resolved_config_path} must define launcher.env as a mapping when present.")

    raw_workers = raw_launcher_section.get("workers", [])
    if raw_workers is None:
        raw_workers = []
    if not isinstance(raw_workers, list):
        raise ValueError(f"Config {resolved_config_path} must define launcher.workers as a list when present.")
    workers: list[dict[str, Any]] = []
    for index, worker in enumerate(raw_workers):
        if not isinstance(worker, Mapping):
            raise ValueError(f"Config {resolved_config_path} launcher.workers[{index}] must be a mapping.")
        workers.append(dict(worker))

    log_path = raw_launcher_section.get("log_path")
    if log_path is None and pipeline_config is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = pipeline_config.run_directory / f"launcher.{timestamp}.log"
    monitor_heartbeat_seconds = raw_launcher_section.get("monitor_heartbeat_seconds", 60)

    return {
        "background": bool(raw_launcher_section.get("background", True)),
        "env": {str(key): str(value) for key, value in launcher_env.items()},
        "log_path": Path(log_path).expanduser() if log_path is not None else None,
        "monitor": bool(raw_launcher_section.get("monitor", False)),
        "monitor_heartbeat_seconds": int(monitor_heartbeat_seconds),
        "workers": workers,
    }


def _load_env_file_values(env_file: Path | None) -> dict[str, str]:
    if env_file is None or not env_file.exists():
        return {}

    from dotenv import dotenv_values

    return {key: value for key, value in dotenv_values(env_file).items() if value is not None}


def completed_layers_from_logs(*log_paths: Path) -> set[int]:
    """Parse all completed layer markers from one or more pipeline log files."""

    completed: set[int] = set()
    for log_path in log_paths:
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = DONE_LAYER_RE.search(line)
            if match:
                completed.add(int(match.group(1)))
    return completed


def dashboard_log_contains_oom(log_path: Path, *, start_offset: int = 0) -> bool:
    """Return whether a dashboard log segment contains an OOM marker."""

    if not log_path.exists():
        return False
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        if start_offset > 0:
            log_file.seek(start_offset)
        return any(OOM_LOG_RE.search(line) for line in log_file)


def _completed_log_paths(config: NeuronpediaDashboardPipelineConfig) -> list[Path]:
    log_paths = [cast(Path, config.existing_log_path), cast(Path, config.pipeline_log_path)]
    if config.enable_layer_locks or config.worker_id:
        log_paths.extend(sorted(config.run_directory.glob("run*.log")))
    return list(dict.fromkeys(log_paths))


def _existing_layer_run_settings_path(output_dir: Path) -> Path | None:
    direct_path = output_dir / RUN_SETTINGS_FILE
    if direct_path.exists():
        return direct_path

    leaf_dirs = _dashboard_leaf_dirs(output_dir)
    if len(leaf_dirs) != 1:
        return None

    leaf_path = leaf_dirs[0] / RUN_SETTINGS_FILE
    if leaf_path.exists():
        return leaf_path
    return None


def _validate_partial_layer_resume_compatibility(
    config: NeuronpediaDashboardPipelineConfig,
    *,
    layer_num: int,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    leaf_dirs = _dashboard_leaf_dirs(output_dir)
    if not leaf_dirs:
        return

    run_settings_path = _existing_layer_run_settings_path(output_dir)
    if run_settings_path is None:
        raise RuntimeError(
            "Cannot safely resume partial dashboard output for "
            f"layer {layer_num}: batch JSON files exist under {output_dir} but {RUN_SETTINGS_FILE} is missing. "
            "Archive or remove the partial layer output before retrying."
        )

    try:
        run_settings = json.loads(run_settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Cannot safely resume partial dashboard output for "
            f"layer {layer_num}: failed to parse {run_settings_path}: {exc}"
        ) from exc

    existing_n_features = run_settings.get("n_features_at_a_time", run_settings.get("n_features_per_batch"))
    try:
        existing_n_features = int(existing_n_features)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot safely resume partial dashboard output for "
            f"layer {layer_num}: {run_settings_path} does not record a valid feature batch size."
        ) from exc

    if existing_n_features != config.n_features_per_batch:
        raise RuntimeError(
            "Cannot safely resume partial dashboard output for "
            f"layer {layer_num}: existing {RUN_SETTINGS_FILE} records n_features_per_batch={existing_n_features} "
            f"but the current config requests {config.n_features_per_batch}. Mixed n_features_per_batch values are "
            "safe for fresh layers, but partial-layer resume must reuse the same feature batch size. "
            "Restart the matching worker or archive the partial layer output before retrying."
        )

    logger.info(
        "Resuming partial layer=%s with matching n_features_per_batch=%s from %s",
        layer_num,
        existing_n_features,
        run_settings_path,
    )


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_payload(config: NeuronpediaDashboardPipelineConfig, layer_num: int) -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "worker_id": config.worker_id,
        "cuda_visible_devices": config.cuda_visible_devices,
        "layer": layer_num,
        "created_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pipeline_log_path": str(config.pipeline_log_path),
    }


def _remove_stale_layer_lock(lock_path: Path, *, stale_seconds: int, logger: logging.Logger) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    pid = payload.get("pid")
    lock_age_seconds = time.time() - lock_path.stat().st_mtime
    pid_is_running = isinstance(pid, int) and _pid_is_running(pid)
    should_remove = not pid_is_running
    if stale_seconds > 0 and lock_age_seconds > stale_seconds:
        should_remove = True
    if not should_remove:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    logger.warning("Removed stale layer lock path=%s payload=%s age_seconds=%.1f", lock_path, payload, lock_age_seconds)
    return True


@contextmanager
def _try_layer_lock(
    config: NeuronpediaDashboardPipelineConfig,
    layer_num: int,
    *,
    logger: logging.Logger,
) -> Iterator[bool]:
    if not config.enable_layer_locks:
        yield True
        return

    lock_path = config.layer_lock_path(layer_num)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    for attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            if attempt == 0 and _remove_stale_layer_lock(
                lock_path,
                stale_seconds=config.layer_lock_stale_seconds,
                logger=logger,
            ):
                continue
            logger.info("Skipping layer=%s because lock is held at %s", layer_num, lock_path)
            yield False
            return
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as lock_handle:
                json.dump(_lock_payload(config, layer_num), lock_handle, sort_keys=True)
                lock_handle.write("\n")
            acquired = True
            logger.info("Acquired layer lock layer=%s path=%s", layer_num, lock_path)
            break

    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                logger.warning("Layer lock disappeared before release layer=%s path=%s", layer_num, lock_path)
            else:
                logger.info("Released layer lock layer=%s path=%s", layer_num, lock_path)


def _parse_meminfo_kib() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw_value = line.split(":", 1)
            value_parts = raw_value.strip().split()
            if value_parts:
                meminfo[key] = int(value_parts[0])
    except OSError:
        return {}
    return meminfo


def _process_status_kib() -> dict[str, int]:
    status: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if not line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                continue
            key, raw_value = line.split(":", 1)
            value_parts = raw_value.strip().split()
            if value_parts:
                status[key] = int(value_parts[0])
    except OSError:
        return {}
    return status


def _format_gib_from_kib(value: int | None) -> str:
    if value is None:
        return "na"
    return f"{value / 1024 / 1024:.2f}"


def _host_memory_snapshot() -> str:
    meminfo = _parse_meminfo_kib()
    status = _process_status_kib()
    return (
        f"rss_gib={_format_gib_from_kib(status.get('VmRSS'))} "
        f"hwm_gib={_format_gib_from_kib(status.get('VmHWM'))} "
        f"mem_available_gib={_format_gib_from_kib(meminfo.get('MemAvailable'))} "
        f"swap_free_gib={_format_gib_from_kib(meminfo.get('SwapFree'))} "
        f"swap_total_gib={_format_gib_from_kib(meminfo.get('SwapTotal'))}"
    )


def _log_host_memory(logger: logging.Logger, *, stage: str, layer_num: int | None = None) -> None:
    layer_segment = "" if layer_num is None else f" layer={layer_num}"
    logger.info("HostMemory stage=%s%s %s", stage, layer_segment, _host_memory_snapshot())


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"interpretune.neuronpedia_dashboard_pipeline.{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _build_generation_env(config: NeuronpediaDashboardPipelineConfig) -> dict[str, str]:
    env = os.environ.copy()
    env_file_values = _load_env_file_values(config.interpretune_env_file)
    for key, value in env_file_values.items():
        env.setdefault(key, value)

    default_hf_home = str(DEFAULT_IT_NP_CACHE.parents[2])
    env["IT_NP_CACHE"] = env.get("IT_NP_CACHE", str(DEFAULT_IT_NP_CACHE))
    env["NEURONPEDIA_EXPORT_ROOT"] = str(config.export_root)
    env["HF_HOME"] = env.get("HF_HOME", default_hf_home)
    env["HF_DATASETS_CACHE"] = env.get("HF_DATASETS_CACHE", os.path.join(env["HF_HOME"], "datasets"))
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", os.path.join(env["HF_HOME"], "hub"))
    env["HF_TOKEN"] = env.get("HF_TOKEN") or env.get("HF_GATED_PUBLIC_REPO_AUTH_KEY") or env.get("HF_MCP_TOKEN_RW", "")
    env["SSL_CERT_FILE"] = env.get("SSL_CERT_FILE", str(config.cert_bundle_path))
    env["REQUESTS_CA_BUNDLE"] = env.get("REQUESTS_CA_BUNDLE", str(config.cert_bundle_path))
    env["CURL_CA_BUNDLE"] = env.get("CURL_CA_BUNDLE", str(config.cert_bundle_path))
    env["DEFAULT_CREATOR_ID"] = env.get("DEFAULT_CREATOR_ID", DEFAULT_EXPLANATION_AUTHOR_ID)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
    env["TQDM_DISABLE"] = env.get("TQDM_DISABLE", "1")
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = env.get("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if config.torch_cuda_alloc_conf is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", config.torch_cuda_alloc_conf)
    if config.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

    pythonpath_entries = [str(config.saelens_repo_root), str(config.saedashboard_repo_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _archive_partial_output(output_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = output_dir.with_name(f"{output_dir.name}.partial.{timestamp}")
    output_dir.rename(archive_path)
    return archive_path


def _path_base_name(path: Path) -> str:
    if not path.suffix:
        return path.name
    return path.name[: -len(path.suffix)]


def _shared_prompt_metadata_file(shared_tokens_file: Path) -> Path:
    return shared_tokens_file.with_suffix(".metadata.json")


def _shared_prompt_effective_lengths_file(shared_tokens_file: Path) -> Path:
    return shared_tokens_file.with_suffix(".effective_lengths.pt")


def _shared_prompt_bucket_manifest_file(shared_tokens_file: Path) -> Path:
    return shared_tokens_file.with_suffix(".buckets.json")


def _shared_prompt_bucket_tokens_file(shared_tokens_file: Path, *, bucket_ceiling: int) -> Path:
    base_name = _path_base_name(shared_tokens_file)
    return shared_tokens_file.with_name(f"{base_name}.bucket_leq_{bucket_ceiling}.pt")


def _shared_prompt_bucket_metadata_file(shared_tokens_file: Path, *, bucket_ceiling: int) -> Path:
    return _shared_prompt_bucket_tokens_file(shared_tokens_file, bucket_ceiling=bucket_ceiling).with_suffix(
        ".metadata.json"
    )


def _shared_prompt_bucket_effective_lengths_file(shared_tokens_file: Path, *, bucket_ceiling: int) -> Path:
    return _shared_prompt_bucket_tokens_file(shared_tokens_file, bucket_ceiling=bucket_ceiling).with_suffix(
        ".effective_lengths.pt"
    )


def _load_json_mapping_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return payload


def _normalized_prompt_bucket_ceilings(
    config: NeuronpediaDashboardPipelineConfig,
    effective_lengths: list[int],
) -> tuple[int, ...]:
    return derive_prompt_bucket_ceilings(
        effective_lengths,
        max_context_size=config.n_tokens_in_prompt,
        explicit_bucket_ceilings=config.prompt_bucket_ceilings,
    )


def _resolve_shared_prompt_run_settings(config: NeuronpediaDashboardPipelineConfig) -> SharedPromptRunSettings:
    resolved_dataset_mode = _resolve_prompts_dataset_mode(config)
    default_settings = SharedPromptRunSettings(
        shared_tokens_file=config.shared_prompt_tokens_file,
        n_prompts_total=config.n_prompts_total,
        n_tokens_in_prompt=config.n_tokens_in_prompt,
        n_prompts_in_forward_pass=config.n_prompts_in_forward_pass,
        primary_acts_batch_size=config.primary_acts_batch_size,
    )
    if config.prompt_bucket_ceiling is None:
        return default_settings

    if resolved_dataset_mode != "load_from_disk":
        raise ValueError("prompt_bucket_ceiling requires prompts_dataset_mode='load_from_disk'.")

    shared_tokens_file = config.shared_prompt_tokens_file
    if shared_tokens_file is None:
        raise ValueError("prompt_bucket_ceiling requires prompts_pretokenized_dataset_path to stage shared tokens.")

    manifest_payload = _load_json_mapping_if_exists(_shared_prompt_bucket_manifest_file(shared_tokens_file))
    buckets = manifest_payload.get("buckets")
    if not isinstance(buckets, list):
        raise ValueError(f"Shared prompt bucket manifest is missing or invalid for {shared_tokens_file}.")

    selected_bucket = None
    for bucket in buckets:
        if isinstance(bucket, dict) and int(bucket.get("upper_inclusive", -1)) == config.prompt_bucket_ceiling:
            selected_bucket = bucket
            break
    if selected_bucket is None:
        available_ceilings = [bucket.get("upper_inclusive") for bucket in buckets if isinstance(bucket, dict)]
        raise ValueError(
            "prompt_bucket_ceiling does not match an available staged bucket: "
            f"selected={config.prompt_bucket_ceiling} available={available_ceilings}"
        )

    bucket_prompt_count = int(selected_bucket.get("prompt_count", 0))
    if bucket_prompt_count <= 0:
        raise ValueError(
            f"Selected prompt bucket {config.prompt_bucket_ceiling} contains no prompts in {shared_tokens_file}."
        )

    bucket_tokens_in_prompt = int(selected_bucket["upper_inclusive"])
    n_prompts_in_forward_pass = config.n_prompts_in_forward_pass
    primary_acts_batch_size = config.primary_acts_batch_size

    return SharedPromptRunSettings(
        shared_tokens_file=Path(str(selected_bucket["tokens_file"])),
        n_prompts_total=bucket_prompt_count,
        n_tokens_in_prompt=bucket_tokens_in_prompt,
        n_prompts_in_forward_pass=n_prompts_in_forward_pass,
        primary_acts_batch_size=primary_acts_batch_size,
        prompt_bucket_ceiling=bucket_tokens_in_prompt,
        bucket_prompt_count=bucket_prompt_count,
        effective_length_min=(
            int(selected_bucket["effective_length_min"])
            if selected_bucket.get("effective_length_min") is not None
            else None
        ),
        effective_length_max=(
            int(selected_bucket["effective_length_max"])
            if selected_bucket.get("effective_length_max") is not None
            else None
        ),
    )


def _requires_pipeline_shared_prompt_artifacts(config: NeuronpediaDashboardPipelineConfig) -> bool:
    return config.prompt_bucket_ceiling is not None


def _ensure_shared_prompt_tokens_file(
    config: NeuronpediaDashboardPipelineConfig,
    *,
    logger: logging.Logger,
) -> Path | None:
    if _resolve_prompts_dataset_mode(config) != "load_from_disk":
        return None

    shared_tokens_file = config.shared_prompt_tokens_file
    if shared_tokens_file is None:
        return None
    metadata_file = _shared_prompt_metadata_file(shared_tokens_file)
    effective_lengths_file = _shared_prompt_effective_lengths_file(shared_tokens_file)
    bucket_manifest_file = _shared_prompt_bucket_manifest_file(shared_tokens_file)
    if (
        shared_tokens_file.exists()
        and metadata_file.exists()
        and effective_lengths_file.exists()
        and bucket_manifest_file.exists()
    ):
        return shared_tokens_file
    if config.prompts_pretokenized_dataset_path is None:
        return None

    from datasets import Dataset, load_from_disk  # type: ignore[import-untyped]
    import torch

    dataset = load_from_disk(str(config.prompts_pretokenized_dataset_path))
    if not isinstance(dataset, Dataset):
        raise ValueError("Pretokenized Neuronpedia prompt datasets must be saved as a single HuggingFace Dataset.")

    if "input_ids" in dataset.column_names:
        tokens_column = "input_ids"
    elif "tokens" in dataset.column_names:
        tokens_column = "tokens"
    else:
        tokens_column = None
    if tokens_column is None:
        raise ValueError(
            "Pretokenized dataset "
            f"{config.prompts_pretokenized_dataset_path} must contain an input_ids or tokens column."
        )

    dataset_metadata = _load_json_mapping_if_exists(config.prompts_pretokenized_dataset_path / "sae_lens.json")
    pad_token_id_raw = dataset_metadata.get("pad_token_id")
    pad_token_id = int(pad_token_id_raw) if pad_token_id_raw is not None else None

    dataset_rows = len(dataset)
    unique_sequences: set[tuple[int, ...]] = set()
    token_rows: list[torch.Tensor] = []
    effective_lengths: list[int] = []
    for row in dataset:
        row_dict = cast(dict[str, Any], row)
        row_tokens = torch.as_tensor(row_dict[tokens_column], dtype=torch.long)
        if row_tokens.numel() < config.n_tokens_in_prompt:
            raise ValueError(
                f"Pretokenized row is shorter than n_tokens_in_prompt={config.n_tokens_in_prompt}: {row_tokens.numel()}"
            )
        row_tokens = row_tokens[: config.n_tokens_in_prompt].cpu()
        attention_mask_value = row_dict.get("attention_mask")
        if attention_mask_value is not None:
            attention_mask = torch.as_tensor(attention_mask_value, dtype=torch.long)
            effective_length = int(attention_mask[: config.n_tokens_in_prompt].sum().item())
        elif pad_token_id is not None:
            nonpad_indices = torch.nonzero(row_tokens != pad_token_id, as_tuple=False)
            effective_length = int(nonpad_indices[-1].item()) + 1 if nonpad_indices.numel() > 0 else 0
        else:
            effective_length = int(row_tokens.numel())
        row_key = tuple(row_tokens.tolist())
        if row_key in unique_sequences and config.deduplicate_shared_prompt_tokens:
            continue
        unique_sequences.add(row_key)
        token_rows.append(row_tokens)
        effective_lengths.append(effective_length)
        if len(token_rows) >= config.n_prompts_total:
            break

    if not token_rows:
        raise ValueError(
            f"Pretokenized dataset {config.prompts_pretokenized_dataset_path} did not yield any prompt tokens."
        )
    if config.strict_shared_prompt_count and len(token_rows) < config.n_prompts_total:
        raise ValueError(
            "Pretokenized dataset did not satisfy the requested prompt count: "
            f"rows={len(token_rows)} requested_prompts={config.n_prompts_total} "
            f"dataset_rows={dataset_rows} unique_rows={len(unique_sequences)} "
            f"deduplicate={config.deduplicate_shared_prompt_tokens} "
            f"path={config.prompts_pretokenized_dataset_path}"
        )

    shared_tokens_file.parent.mkdir(parents=True, exist_ok=True)
    token_tensor = torch.stack(token_rows, dim=0)
    effective_length_tensor = torch.tensor(effective_lengths, dtype=torch.int32)
    torch.save(token_tensor, shared_tokens_file)
    torch.save(effective_length_tensor, effective_lengths_file)
    bucket_ceilings = _normalized_prompt_bucket_ceilings(config, effective_lengths)
    bucket_entries: list[dict[str, Any]] = []
    lower_exclusive = 0
    for bucket_ceiling in bucket_ceilings:
        bucket_indices = [
            index for index, length in enumerate(effective_lengths) if lower_exclusive < length <= bucket_ceiling
        ]
        bucket_prompt_count = len(bucket_indices)
        bucket_tokens_file = _shared_prompt_bucket_tokens_file(shared_tokens_file, bucket_ceiling=bucket_ceiling)
        bucket_lengths_file = _shared_prompt_bucket_effective_lengths_file(
            shared_tokens_file,
            bucket_ceiling=bucket_ceiling,
        )
        bucket_metadata_file = _shared_prompt_bucket_metadata_file(shared_tokens_file, bucket_ceiling=bucket_ceiling)
        if bucket_prompt_count > 0:
            torch.save(token_tensor[bucket_indices, :bucket_ceiling].contiguous(), bucket_tokens_file)
            torch.save(effective_length_tensor[bucket_indices].clone(), bucket_lengths_file)
        bucket_entry = {
            "bucket_label": f"({lower_exclusive}, {bucket_ceiling}]",
            "lower_exclusive": lower_exclusive,
            "upper_inclusive": bucket_ceiling,
            "prompt_count": bucket_prompt_count,
            "effective_length_min": min(effective_lengths[index] for index in bucket_indices)
            if bucket_indices
            else None,
            "effective_length_max": max(effective_lengths[index] for index in bucket_indices)
            if bucket_indices
            else None,
            "effective_length_mean": (
                sum(effective_lengths[index] for index in bucket_indices) / bucket_prompt_count
                if bucket_prompt_count
                else None
            ),
            "tokens_file": str(bucket_tokens_file),
            "effective_lengths_file": str(bucket_lengths_file),
            "metadata_file": str(bucket_metadata_file),
        }
        bucket_metadata_file.write_text(
            json.dumps(bucket_entry, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        bucket_entries.append(bucket_entry)
        lower_exclusive = bucket_ceiling

    bucket_manifest_file.write_text(
        json.dumps(
            {
                "requested_prompts": config.n_prompts_total,
                "source_dataset_path": str(config.prompts_pretokenized_dataset_path),
                "bucket_ceilings": list(bucket_ceilings),
                "buckets": bucket_entries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metadata_file.write_text(
        json.dumps(
            {
                "requested_prompts": config.n_prompts_total,
                "tensor_shape": list(token_tensor.shape),
                "dataset_rows": dataset_rows,
                "unique_rows": len(unique_sequences),
                "tokens_per_prompt": config.n_tokens_in_prompt,
                "deduplicate": config.deduplicate_shared_prompt_tokens,
                "source_dataset_path": str(config.prompts_pretokenized_dataset_path),
                "effective_lengths_file": str(effective_lengths_file),
                "effective_length_min": min(effective_lengths),
                "effective_length_max": max(effective_lengths),
                "effective_length_mean": sum(effective_lengths) / len(effective_lengths),
                "pad_token_id": pad_token_id,
                "bucket_ceilings": list(bucket_ceilings),
                "bucket_manifest": str(bucket_manifest_file),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info(
        (
            "Prepared shared prompt tokens file path=%s rows=%s dataset_rows=%s unique_rows=%s "
            "requested_prompts=%s tokens_per_prompt=%s deduplicate=%s effective_length_range=%s-%s "
            "bucket_manifest=%s metadata=%s"
        ),
        shared_tokens_file,
        len(token_rows),
        dataset_rows,
        len(unique_sequences),
        config.n_prompts_total,
        config.n_tokens_in_prompt,
        config.deduplicate_shared_prompt_tokens,
        min(effective_lengths),
        max(effective_lengths),
        bucket_manifest_file,
        metadata_file,
    )
    return shared_tokens_file


def _directory_stats(root_dir: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    if not root_dir.exists():
        return file_count, total_bytes
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return file_count, total_bytes


def _dashboard_leaf_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    leaf_dirs: list[Path] = []
    for root, _, files in os.walk(output_dir):
        if any(BATCH_JSON_RE.match(file_name) for file_name in files):
            leaf_dirs.append(Path(root))
    return sorted(leaf_dirs)


def _resolve_dashboard_leaf_dir(output_dir: Path) -> Path:
    leaf_dirs = _dashboard_leaf_dirs(output_dir)
    if not leaf_dirs:
        raise RuntimeError(f"No dashboard leaf directory with batch JSON files found under {output_dir}")
    if len(leaf_dirs) == 1:
        return leaf_dirs[0]
    return max(
        leaf_dirs,
        key=lambda path: (len(list(path.glob("batch-*.json"))), len(path.parts), str(path)),
    )


def _run_command_lines(command: list[str]) -> list[str]:
    executable = shutil.which(command[0])
    if executable is None:
        return [f"missing executable: {command[0]}"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except Exception as exc:
        return [f"command failed: {' '.join(command)}: {exc}"]
    stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    stderr_lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    lines = stdout_lines + stderr_lines
    if not lines:
        lines = [f"exit={completed.returncode} no output"]
    return lines[:20]


def _process_snapshot(pid: int) -> str:
    lines = _run_command_lines(
        [
            "ps",
            "-o",
            "pid=,ppid=,pgid=,stat=,%cpu=,%mem=,rss=,vsz=,etimes=,cmd=",
            "-p",
            str(pid),
        ]
    )
    return " | ".join(lines)


def _gpu_snapshot(pid: int) -> str:
    lines = _run_command_lines(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_lines = _run_command_lines(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    pid_prefix = f"{pid},"
    matching_process_lines = [line for line in process_lines if line.startswith(pid_prefix)]
    return "gpus=" + " || ".join(lines[:8]) + " ; pid=" + " || ".join(matching_process_lines or ["not listed"])


def _kernel_snapshot() -> str:
    return " || ".join(_run_command_lines(["dmesg", "-T", "--level=err,crit,alert,emerg"]))


def _log_runtime_diagnostics(logger: logging.Logger, *, pid: int, output_dir: Path, reason: str) -> None:
    file_count, total_bytes = _directory_stats(output_dir)
    leaf_dirs = _dashboard_leaf_dirs(output_dir)
    logger.info(
        "Diagnostics reason=%s pid=%s files=%s bytes=%s leaf_dirs=%s ps=%s gpu=%s kernel=%s",
        reason,
        pid,
        file_count,
        total_bytes,
        [str(path) for path in leaf_dirs[:4]],
        _process_snapshot(pid),
        _gpu_snapshot(pid),
        _kernel_snapshot(),
    )


def _resolve_runner_dashboard_output_format(config: NeuronpediaDashboardPipelineConfig) -> str:
    if config.runner_implementation == "legacy":
        return "legacy_json"
    dashboard_output_format = config.runner_dashboard_output_format
    if dashboard_output_format == "auto":
        return (
            "columnar"
            if config.runner_defer_component_construction and config.runner_sequence_selection_backend == "columnar_gpu"
            else "legacy_json"
        )
    if dashboard_output_format not in {"legacy_json", "columnar"}:
        raise ValueError(
            "runner_dashboard_output_format must be one of 'auto', 'legacy_json', or 'columnar'; "
            f"got {dashboard_output_format!r}."
        )
    return dashboard_output_format


def _resolve_runner_emit_activation_copy_rows(config: NeuronpediaDashboardPipelineConfig) -> bool:
    if config.runner_emit_activation_copy_rows is not None:
        return config.runner_emit_activation_copy_rows
    return config.import_to_local_db


def _legacy_export_bundle_emit_arrow_override(
    config: NeuronpediaDashboardPipelineConfig,
) -> bool | None:
    if config.legacy_export_bundle_contract == "preserved_baseline":
        return False
    return None


def _legacy_export_bundle_import_preferences(
    config: NeuronpediaDashboardPipelineConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if config.legacy_export_bundle_contract == "preserved_baseline":
        return (), ()
    return DEFAULT_LOCAL_DB_COLUMNAR_IMPORT_TABLES, DEFAULT_LOCAL_DB_COLUMNAR_COPY_TABLES


def _legacy_local_dataset_alias_name(dataset_path: Path) -> str:
    dataset_slug = re.sub(r"[^a-z0-9]+", "-", dataset_path.name.lower()).strip("-")
    dataset_slug = re.sub(r"-{2,}", "-", dataset_slug)
    if not dataset_slug:
        dataset_slug = "dataset"
    digest = hashlib.sha1(str(dataset_path).encode("utf-8")).hexdigest()[:10]
    return f"it-legacy-local-{dataset_slug[:40]}-{digest}"


def _materialize_legacy_local_dataset_alias(
    dataset_path: str,
    *,
    saedashboard_repo_root: Path,
) -> str:
    candidate_path = Path(dataset_path).expanduser()
    if not candidate_path.is_absolute():
        return dataset_path

    try:
        resolved_path = candidate_path.resolve(strict=True)
    except OSError:
        return dataset_path

    if not resolved_path.is_dir():
        return dataset_path
    if not (resolved_path / "sae_lens.json").exists():
        return dataset_path
    if not any(resolved_path.glob("*.jsonl")):
        return dataset_path

    alias_name = _legacy_local_dataset_alias_name(resolved_path)
    alias_path = saedashboard_repo_root / alias_name
    alias_marker_path = alias_path / LEGACY_LOCAL_DATASET_ALIAS_SOURCE_FILE

    if alias_path.is_symlink():
        try:
            if alias_path.resolve(strict=True) == resolved_path:
                return alias_name
        except OSError:
            pass
        alias_path.unlink()
    elif alias_path.is_dir():
        if alias_marker_path.exists() and alias_marker_path.read_text(encoding="utf-8").strip() == str(resolved_path):
            return alias_name
        raise RuntimeError(
            "Cannot materialize a legacy local dataset alias because the target path already exists "
            f"and is not a managed alias: {alias_path}"
        )
    elif alias_path.exists():
        raise RuntimeError(
            f"Cannot materialize a legacy local dataset alias because the target path already exists: {alias_path}"
        )

    try:
        alias_path.symlink_to(resolved_path, target_is_directory=True)
    except OSError:
        shutil.copytree(resolved_path, alias_path)
        alias_marker_path.write_text(f"{resolved_path}\n", encoding="utf-8")

    return alias_name


def _layer_runner_command(
    config: NeuronpediaDashboardPipelineConfig,
    layer_num: int,
    output_dir: Path,
    *,
    prompt_settings: SharedPromptRunSettings | None = None,
) -> list[str]:
    resolved_dataset_mode = _resolve_prompts_dataset_mode(config)
    prompt_dataset_path = (
        str(config.prompts_pretokenized_dataset_path)
        if resolved_dataset_mode == "load_from_disk" and config.prompts_pretokenized_dataset_path is not None
        else config.prompts_huggingface_dataset_path
    )
    if prompt_settings is None:
        prompt_settings = _resolve_shared_prompt_run_settings(config)
    if config.runner_implementation == "legacy":
        return _legacy_layer_runner_command(
            config,
            layer_num=layer_num,
            output_dir=output_dir,
            prompt_settings=prompt_settings,
        )
    command = [
        config.python_executable,
        "-m",
        "sae_dashboard.neuronpedia.neuronpedia_runner",
        f"--sae-set={config.sae_set}",
        f"--sae-path={config.sae_path_for_layer(layer_num)}",
        f"--np-set-name={config.neuronpedia_source_set_id}",
        f"--prompt-dataset-path={prompt_dataset_path}",
        f"--prompt-dataset-mode={resolved_dataset_mode}",
        f"--model-wrapper={config.model_wrapper}",
        f"--output-dir={output_dir}",
        f"--sae_dtype={config.sae_dtype}",
        f"--model_dtype={config.model_dtype}",
        f"--sparsity-threshold={config.sparsity_threshold}",
        f"--n-prompts={prompt_settings.n_prompts_total}",
        f"--n-tokens-in-prompt={prompt_settings.n_tokens_in_prompt}",
        f"--n-features-per-batch={config.n_features_per_batch}",
        f"--n-prompts-in-forward-pass={prompt_settings.n_prompts_in_forward_pass}",
        f"--start-batch={config.start_batch}",
    ]
    if config.run_name_suffix:
        command.append(f"--np-sae-id-suffix={config.run_name_suffix}")
    if prompt_settings.primary_acts_batch_size is not None:
        command.append(f"--primary-acts-batch-size={prompt_settings.primary_acts_batch_size}")
    if config.hf_model_path:
        command.append(f"--hf-model-path={config.hf_model_path}")
    if config.prompts_huggingface_dataset_config_name:
        command.append(f"--prompt-dataset-name={config.prompts_huggingface_dataset_config_name}")
    if config.prompts_huggingface_dataset_split:
        command.append(f"--prompt-dataset-split={config.prompts_huggingface_dataset_split}")
    if config.prompts_dataset_text_field:
        command.append(f"--prompt-dataset-text-field={config.prompts_dataset_text_field}")
    if prompt_settings.shared_tokens_file:
        command.append(f"--shared-tokens-file={prompt_settings.shared_tokens_file}")
    if not config.deduplicate_shared_prompt_tokens:
        command.append("--no-deduplicate-shared-prompt-tokens")
    if config.strict_shared_prompt_count:
        command.append("--strict-shared-prompt-count")
    if config.end_batch is not None:
        command.append(f"--end-batch={config.end_batch}")
    if not config.dataset_streaming:
        command.append("--no-dataset-streaming")
    if config.use_clt:
        command.append("--from-local-sae")
        command.append("--use-clt")
        command.append(f"--clt-layer-idx={layer_num}")
        if config.clt_dtype:
            command.append(f"--clt-dtype={config.clt_dtype}")
        if config.clt_weights_filename:
            command.append(f"--clt-weights-filename={config.clt_weights_filename}")
    if config.model_wrapper == "bridge":
        command.append(
            "--bridge-enable-compatibility-mode"
            if config.bridge_enable_compatibility_mode
            else "--no-bridge-enable-compatibility-mode"
        )
        if config.bridge_compatibility_mode_kwargs:
            bridge_compatibility_kwargs_json = json.dumps(
                config.bridge_compatibility_mode_kwargs,
                sort_keys=True,
                separators=(",", ":"),
            )
            command.append(f"--bridge-compatibility-mode-kwargs-json={bridge_compatibility_kwargs_json}")
    if config.runner_log_resource_snapshots:
        command.append("--log-resource-snapshots")
    if config.runner_log_hook_aliases:
        command.append("--log-hook-aliases")
    if config.runner_log_performance:
        command.append("--log-performance")
    if config.runner_profile_rolling_substages:
        command.append("--profile-rolling-substages")
    if not config.runner_shuffle_tokens:
        command.append("--no-shuffle-tokens")
    command.append(
        "--cleanup-each-minibatch" if config.runner_cleanup_each_minibatch else "--no-cleanup-each-minibatch"
    )
    command.append(f"--correlation-accumulation-device={config.runner_correlation_accumulation_device}")
    if config.runner_rolling_coefficient_num_threads is not None:
        command.append(f"--rolling-coefficient-num-threads={config.runner_rolling_coefficient_num_threads}")
    if config.runner_activation_significance_floor > 0:
        command.append(f"--activation-significance-floor={config.runner_activation_significance_floor}")
    if config.runner_converter_input_artifact_dir:
        command.append(f"--converter-input-artifact-dir={config.runner_converter_input_artifact_dir}")
    command.append(f"--feature-statistics-backend={config.runner_feature_statistics_backend}")
    command.append(f"--logits-histogram-backend={config.runner_logits_histogram_backend}")
    command.append(
        "--defer-component-construction"
        if config.runner_defer_component_construction
        else "--no-defer-component-construction"
    )
    command.append(f"--sequence-selection-backend={config.runner_sequence_selection_backend}")
    if config.runner_sequence_top_acts_positive_only:
        command.append("--sequence-top-acts-positive-only")
    if config.runner_sequence_dedup_across_groups:
        command.append("--sequence-dedup-across-groups")
    if config.runner_sequence_skip_dead_features:
        command.append("--sequence-skip-dead-features")
    if config.runner_sequence_half_open_interval_bins:
        command.append("--sequence-half-open-interval-bins")
    if config.runner_columnar_max_staged_acts_bytes is not None:
        command.append(f"--columnar-max-device-staged-acts-bytes={config.runner_columnar_max_staged_acts_bytes}")
    if config.runner_columnar_row_chunk_size is not None:
        command.append(f"--columnar-row-chunk-size={config.runner_columnar_row_chunk_size}")
    if config.runner_logits_table_mask_token_pattern:
        command.append(f"--logits-table-mask-token-pattern={config.runner_logits_table_mask_token_pattern}")
    dashboard_output_format = _resolve_runner_dashboard_output_format(config)
    command.append(f"--dashboard-output-format={dashboard_output_format}")
    if dashboard_output_format == "columnar":
        command.append(f"--columnar-artifact-format={config.runner_columnar_artifact_format}")
        command.append("--columnar-emit-activation-rows")
        if config.runner_overlap_batch_packaging:
            command.append("--overlap-batch-packaging")
        command.append(
            "--columnar-emit-activation-copy-rows"
            if _resolve_runner_emit_activation_copy_rows(config)
            else "--no-columnar-emit-activation-copy-rows"
        )
        if _resolve_runner_emit_activation_copy_rows(config):
            command.append(f"--columnar-activation-copy-model-id={config.model_name}")
    if config.runner_prompt_bucket_schedule_file:
        command.append(f"--prompt-bucket-schedule-file={config.runner_prompt_bucket_schedule_file}")
    if config.runner_auto_prompt_bucket_schedule:
        command.append("--auto-prompt-bucket-schedule")
        if config.runner_prompt_bucket_ceilings:
            runner_bucket_ceilings = ",".join(str(value) for value in config.runner_prompt_bucket_ceilings)
            command.append(f"--prompt-bucket-ceilings={runner_bucket_ceilings}")
        command.append(f"--prompt-bucket-scale-limit={config.runner_prompt_bucket_scale_limit}")
        command.append(f"--prompt-primary-acts-scale-limit={config.runner_prompt_primary_acts_scale_limit}")
        command.append(f"--prompt-batch-size-round-to={config.runner_prompt_batch_size_round_to}")
    if config.runner_torch_profile:
        command.append("--torch-profile")
    if config.runner_torch_profile_dir:
        command.append(f"--torch-profile-dir={config.runner_torch_profile_dir}")
    command.append(
        "--use-cached-activations" if config.runner_use_cached_activations else "--no-use-cached-activations"
    )
    if config.use_skip_transcoder:
        command.append("--use-skip-transcoder")
    if config.zero_out_bos_token:
        command.append("--zero-out-bos-token")
    return command


def _legacy_layer_runner_command(
    config: NeuronpediaDashboardPipelineConfig,
    layer_num: int,
    output_dir: Path,
    *,
    prompt_settings: SharedPromptRunSettings,
) -> list[str]:
    resolved_dataset_mode = _resolve_prompts_dataset_mode(config)
    detached_baseline_supports_prompt_dataset_contract = _legacy_runner_supports_prompt_dataset_contract(
        config.saedashboard_repo_root
    )
    dataset_path = config.prompts_huggingface_dataset_path
    if not detached_baseline_supports_prompt_dataset_contract:
        dataset_path = _materialize_legacy_local_dataset_alias(
            config.prompts_huggingface_dataset_path,
            saedashboard_repo_root=config.saedashboard_repo_root,
        )
    command = [
        config.python_executable,
        "-m",
        "sae_dashboard.neuronpedia.neuronpedia_runner",
        f"--sae-set={config.sae_set}",
        f"--sae-path={config.sae_path_for_layer(layer_num)}",
        f"--np-set-name={config.neuronpedia_source_set_id}",
        f"--output-dir={output_dir}",
        f"--sae_dtype={config.sae_dtype}",
        f"--model_dtype={config.model_dtype}",
        f"--sparsity-threshold={config.sparsity_threshold}",
        f"--n-prompts={prompt_settings.n_prompts_total}",
        f"--n-tokens-in-prompt={prompt_settings.n_tokens_in_prompt}",
        f"--n-features-per-batch={config.n_features_per_batch}",
        f"--n-prompts-in-forward-pass={prompt_settings.n_prompts_in_forward_pass}",
        f"--start-batch={config.start_batch}",
    ]
    if detached_baseline_supports_prompt_dataset_contract:
        command.extend(
            [
                "--sequence-selection-backend=legacy",
                "--dashboard-output-format=legacy_json",
            ]
        )
    if detached_baseline_supports_prompt_dataset_contract:
        command.extend(
            [
                f"--prompt-dataset-path={dataset_path}",
                f"--prompt-dataset-mode={resolved_dataset_mode}",
            ]
        )
    else:
        command.append(f"--dataset-path={dataset_path}")
    if config.run_name_suffix:
        command.append(f"--np-sae-id-suffix={config.run_name_suffix}")
    if config.end_batch is not None:
        command.append(f"--end-batch={config.end_batch}")
    if not config.runner_shuffle_tokens:
        command.append("--no-shuffle-tokens")
    if config.runner_log_resource_snapshots and _legacy_runner_supports_option(
        config.saedashboard_repo_root, "--log-resource-snapshots"
    ):
        command.append("--log-resource-snapshots")
    if config.runner_log_hook_aliases and _legacy_runner_supports_option(
        config.saedashboard_repo_root, "--log-hook-aliases"
    ):
        command.append("--log-hook-aliases")
    if config.runner_log_performance and _legacy_runner_supports_option(
        config.saedashboard_repo_root, "--log-performance"
    ):
        command.append("--log-performance")
    if config.runner_profile_rolling_substages and _legacy_runner_supports_option(
        config.saedashboard_repo_root, "--profile-rolling-substages"
    ):
        command.append("--profile-rolling-substages")
    if _legacy_runner_supports_option(config.saedashboard_repo_root, "--cleanup-each-minibatch"):
        command.append(
            "--cleanup-each-minibatch" if config.runner_cleanup_each_minibatch else "--no-cleanup-each-minibatch"
        )
    if _legacy_runner_supports_option(config.saedashboard_repo_root, "--correlation-accumulation-device"):
        command.append(f"--correlation-accumulation-device={config.runner_correlation_accumulation_device}")
    if config.runner_rolling_coefficient_num_threads is not None and _legacy_runner_supports_option(
        config.saedashboard_repo_root, "--rolling-coefficient-num-threads"
    ):
        command.append(f"--rolling-coefficient-num-threads={config.runner_rolling_coefficient_num_threads}")
    if config.runner_activation_significance_floor > 0:
        command.append(f"--activation-significance-floor={config.runner_activation_significance_floor}")
    if _legacy_runner_supports_option(config.saedashboard_repo_root, "--logits-histogram-backend"):
        command.append(f"--logits-histogram-backend={config.runner_logits_histogram_backend}")
    if _legacy_runner_supports_option(config.saedashboard_repo_root, "--use-cached-activations"):
        command.append(
            "--use-cached-activations" if config.runner_use_cached_activations else "--no-use-cached-activations"
        )
    if config.hf_model_path:
        command.append(f"--hf-model-path={config.hf_model_path}")
    if detached_baseline_supports_prompt_dataset_contract and config.prompts_huggingface_dataset_config_name:
        command.append(f"--prompt-dataset-name={config.prompts_huggingface_dataset_config_name}")
    if detached_baseline_supports_prompt_dataset_contract and config.prompts_huggingface_dataset_split:
        command.append(f"--prompt-dataset-split={config.prompts_huggingface_dataset_split}")
    if detached_baseline_supports_prompt_dataset_contract and config.prompts_dataset_text_field:
        command.append(f"--prompt-dataset-text-field={config.prompts_dataset_text_field}")
    if prompt_settings.shared_tokens_file and _legacy_runner_supports_option(
        config.saedashboard_repo_root, "--shared-tokens-file"
    ):
        command.append(f"--shared-tokens-file={prompt_settings.shared_tokens_file}")
    if config.use_clt:
        command.append("--from-local-sae")
        command.append("--use-clt")
        command.append(f"--clt-layer-idx={layer_num}")
        if config.clt_dtype:
            command.append(f"--clt-dtype={config.clt_dtype}")
        if config.clt_weights_filename:
            command.append(f"--clt-weights-filename={config.clt_weights_filename}")
    if config.use_skip_transcoder:
        command.append("--use-skip-transcoder")
    if resolved_dataset_mode == "load_from_disk":
        raise ValueError("legacy must not be invoked with prompts_dataset_mode='load_from_disk'.")
    return command


def _legacy_runner_supports_prompt_dataset_contract(saedashboard_repo_root: Path) -> bool:
    return _legacy_runner_supports_option(saedashboard_repo_root, "--prompt-dataset-path")


def _legacy_runner_supports_option(saedashboard_repo_root: Path, option_name: str) -> bool:
    runner_path = saedashboard_repo_root / "sae_dashboard" / "neuronpedia" / "neuronpedia_runner.py"
    if not runner_path.exists():
        return True
    return option_name in runner_path.read_text(encoding="utf-8")


def _monitor_process(
    process: subprocess.Popen[str],
    *,
    output_dir: Path,
    logger: logging.Logger,
    heartbeat_seconds: int,
    stall_timeout_seconds: int,
) -> int:
    last_seen_size = 0
    last_seen_file_count = 0
    last_growth_time = time.monotonic()
    while True:
        return_code = process.poll()
        if return_code is not None:
            return return_code
        if heartbeat_seconds > 0:
            time.sleep(heartbeat_seconds)

        current_file_count, current_size = _directory_stats(output_dir)
        if current_size > last_seen_size or current_file_count > last_seen_file_count:
            last_seen_size = current_size
            last_seen_file_count = current_file_count
            last_growth_time = time.monotonic()

        elapsed_without_growth = time.monotonic() - last_growth_time
        logger.info(
            "Heartbeat output_dir=%s files=%s bytes=%s elapsed_without_growth=%.1fs pid=%s ps=%s gpu=%s host=%s",
            output_dir,
            current_file_count,
            current_size,
            elapsed_without_growth,
            process.pid,
            _process_snapshot(process.pid),
            _gpu_snapshot(process.pid),
            _host_memory_snapshot(),
        )
        if stall_timeout_seconds > 0 and elapsed_without_growth >= stall_timeout_seconds:
            _log_runtime_diagnostics(logger, pid=process.pid, output_dir=output_dir, reason="stall-timeout")
            process.terminate()
            raise RuntimeError(
                f"Dashboard generation stalled for {elapsed_without_growth:.0f}s without output growth: {output_dir}"
            )


def _load_converter_module(neuronpedia_utils_root: Path) -> ModuleType:
    module_name = "interpretune_np_convert_saedashboard"
    script_path = neuronpedia_utils_root / "neuronpedia_utils" / "convert-saedashboard-to-neuronpedia-export.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find converter script at {script_path}")
    if str(neuronpedia_utils_root) not in sys.path:
        sys.path.insert(0, str(neuronpedia_utils_root))
    os.environ.setdefault("DEFAULT_CREATOR_ID", DEFAULT_EXPLANATION_AUTHOR_ID)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_source_id(output_dir: Path, layer_num: int, neuronpedia_source_set_id: str) -> str:
    batch_files = sorted(output_dir.glob("batch-*.json"))
    if not batch_files and output_dir.exists():
        leaf_dirs = _dashboard_leaf_dirs(output_dir)
        if leaf_dirs:
            batch_files = sorted(_resolve_dashboard_leaf_dir(output_dir).glob("batch-*.json"))
    if not batch_files:
        layer_dir_name = f"layer_{layer_num}"
        layer_dir = output_dir if output_dir.name == layer_dir_name else None
        if layer_dir is None:
            layer_dir = next((parent for parent in output_dir.parents if parent.name == layer_dir_name), None)
        if layer_dir is not None:
            run_dir_name = layer_dir.parent.name
            run_name_marker = f"_{neuronpedia_source_set_id}"
            marker_index = run_dir_name.find(run_name_marker)
            if marker_index != -1:
                suffix = run_dir_name[marker_index + len(run_name_marker) :].lstrip("_")
                if suffix:
                    return f"{layer_num}-{neuronpedia_source_set_id}__{suffix}"
        return f"{layer_num}-{neuronpedia_source_set_id}"
    batch_data = json.loads(batch_files[0].read_text(encoding="utf-8"))
    source_suffix = batch_data.get("sae_id_suffix") or ""
    if source_suffix:
        return f"{layer_num}-{neuronpedia_source_set_id}__{source_suffix}"
    return f"{layer_num}-{neuronpedia_source_set_id}"


def _resolve_export_root(export_parent: Path, source_id: str) -> Path:
    direct_path = export_parent / source_id
    if direct_path.exists():
        return direct_path
    candidates = sorted(export_parent.glob(f"{source_id}*"))
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Expected Neuronpedia export root was not created: {direct_path}. candidates={candidates}")


def _resolve_columnar_pad_token_id(
    config: NeuronpediaDashboardPipelineConfig, tokenizer_pad_token_id: int | None
) -> int | None:
    if tokenizer_pad_token_id is not None:
        return int(tokenizer_pad_token_id)
    if config.prompts_pretokenized_dataset_path is None:
        return None
    metadata = _load_json_mapping_if_exists(config.prompts_pretokenized_dataset_path / "sae_lens.json")
    pad_token_id = metadata.get("pad_token_id")
    return int(pad_token_id) if pad_token_id is not None else None


def _resolve_columnar_tokenizer_name(config: NeuronpediaDashboardPipelineConfig) -> str:
    if config.prompts_pretokenized_dataset_path is not None:
        metadata = _load_json_mapping_if_exists(config.prompts_pretokenized_dataset_path / "sae_lens.json")
        tokenizer_name = metadata.get("tokenizer_name")
        if isinstance(tokenizer_name, str) and tokenizer_name.strip():
            return tokenizer_name
    if config.hf_model_path:
        return config.hf_model_path

    model_name = config.model_name.strip()
    if "/" in model_name:
        return model_name

    parsed_release_url = urlparse(config.release_url)
    release_path_parts = [part for part in parsed_release_url.path.split("/") if part]
    if parsed_release_url.netloc.endswith("huggingface.co") and release_path_parts:
        return f"{release_path_parts[0]}/{model_name}"

    return model_name


def _build_columnar_token_decoder(
    config: NeuronpediaDashboardPipelineConfig,
) -> tuple[Callable[[list[int]], list[str]], int | None]:
    from transformers import AutoTokenizer

    pretrained_name = _resolve_columnar_tokenizer_name(config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    except ValueError as exc:
        if "Couldn't instantiate the backend tokenizer" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=False)

    def decode_token_ids(token_ids: list[int]) -> list[str]:
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        if isinstance(tokens, str):
            return [tokens]
        return [str(token) for token in tokens]

    return decode_token_ids, _resolve_columnar_pad_token_id(config, tokenizer.pad_token_id)


def import_columnar_dashboard_output(
    config: NeuronpediaDashboardPipelineConfig,
    *,
    layer_num: int,
    output_dir: Path,
    activation_use_stage_table: bool = True,
    source_id_override: str | None = None,
    activation_id_prefix: str | None = None,
) -> NeuronpediaLocalImportSummary:
    """Import SAEDashboard columnar output directly into local Neuronpedia tables."""

    source_id = source_id_override or _resolve_source_id(output_dir, layer_num, config.neuronpedia_source_set_id)
    decode_token_ids, pad_token_id = _build_columnar_token_decoder(config)
    return import_saedashboard_columnar_bundle_local_db(
        output_dir,
        local_db_url=config.local_db_url or "",
        model_id=config.model_name,
        source_set_name=config.neuronpedia_source_set_id,
        source_id=source_id,
        creator_id=DEFAULT_EXPLANATION_AUTHOR_ID,
        creator_name=config.creator_name,
        decode_token_ids=decode_token_ids,
        activation_id_prefix=activation_id_prefix or f"{source_id}-activation",
        pad_token_id=pad_token_id,
        hook_name=config.hook_point,
        release_name=config.release_id,
        release_description=config.release_title,
        release_description_short=config.release_title,
        release_urls=[config.release_url] if config.release_url else [],
        model_display_name=config.model_name,
        model_layers=config.model_layers,
        model_neurons_per_layer=0,
        model_owner="",
        source_set_description=config.neuronpedia_source_set_description,
        source_dataset=config.prompts_dataset_identifier,
        source_hf_repo_id=config.hf_weights_repo_id,
        source_hf_folder_id=config.hf_weights_path_for_layer(layer_num),
        source_saelens_release=config.sae_set,
        source_saelens_sae_id=config.sae_path_for_layer(layer_num),
        num_prompts=config.n_prompts_total,
        num_tokens_in_prompt=config.n_tokens_in_prompt,
        chunk_size=config.local_db_import_chunk_size,
        activation_use_stage_table=activation_use_stage_table,
        dedup_activation_rows=config.local_db_import_dedup_activation_rows,
        drop_zero_activation_rows=config.local_db_import_drop_zero_activation_rows,
    )


def _has_existing_columnar_output(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(output_dir.rglob("*.columnar"))


def _find_existing_export_root(config: NeuronpediaDashboardPipelineConfig, layer_num: int) -> Path:
    export_parent = config.export_root / config.model_name
    source_id_prefix = f"{layer_num}-{config.neuronpedia_source_set_id}"
    if config.run_name_suffix:
        suffixed_candidates = sorted(export_parent.glob(f"{source_id_prefix}__{config.run_name_suffix}*"))
        if len(suffixed_candidates) == 1:
            return suffixed_candidates[0]
        if not suffixed_candidates:
            raise RuntimeError(
                f"No existing export bundle found for layer {layer_num} under {export_parent} with source-set "
                f"{config.neuronpedia_source_set_id} and run-name suffix {config.run_name_suffix!r}."
            )
        raise RuntimeError(
            f"Multiple export bundles found for layer {layer_num} under {export_parent} with source-set "
            f"{config.neuronpedia_source_set_id} and run-name suffix {config.run_name_suffix!r}: "
            f"{suffixed_candidates}. Resolve the ambiguity before using --import-only-local-db."
        )

    candidates = sorted(export_parent.glob(f"{source_id_prefix}*"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            f"No existing export bundle found for layer {layer_num} under {export_parent} "
            f"with source-set {config.neuronpedia_source_set_id}."
        )
    raise RuntimeError(
        f"Multiple export bundles found for layer {layer_num} under {export_parent}: {candidates}. "
        "Resolve the ambiguity before using --import-only-local-db."
    )


def _find_existing_import_root(
    config: NeuronpediaDashboardPipelineConfig,
    layer_num: int,
) -> tuple[str, Path]:
    dashboard_output_format = _resolve_runner_dashboard_output_format(config)
    output_dir = config.output_dir_for_layer(layer_num)
    if dashboard_output_format == "columnar":
        if _has_existing_columnar_output(output_dir):
            return "columnar", output_dir
        raise RuntimeError(
            f"No existing columnar output found for layer {layer_num} under {output_dir}. "
            "Generate the layer with columnar dashboard output before using --import-only-local-db."
        )
    return "legacy_export_bundle", _find_existing_export_root(config, layer_num)


def convert_dashboard_output(
    config: NeuronpediaDashboardPipelineConfig,
    *,
    layer_num: int,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Convert a SAEDashboard layer output into a Neuronpedia export bundle."""

    logger = logger or logging.getLogger(__name__)
    dashboard_leaf_dir = _resolve_dashboard_leaf_dir(output_dir)
    module = _load_converter_module(config.neuronpedia_utils_root)
    module_any = cast(Any, module)
    params = {
        "saedashboard_output_dir": str(dashboard_leaf_dir),
        "export_root": str(config.export_root),
        "creator_name": config.creator_name,
        "release_id": config.release_id,
        "release_title": config.release_title,
        "url": config.release_url,
        "model_name": config.model_name,
        "model_layers": config.model_layers,
        "neuronpedia_source_set_id": config.neuronpedia_source_set_id,
        "neuronpedia_source_set_description": config.neuronpedia_source_set_description,
        "hf_weights_repo_id": config.hf_weights_repo_id,
        "hf_weights_path": config.hf_weights_path_for_layer(layer_num),
        "hook_point": module_any.HOOK_POINT_TYPE_CHOICES(config.hook_point),
        "layer_num": layer_num,
        "prompts_huggingface_dataset_path": config.prompts_dataset_identifier,
        "n_prompts_total": config.n_prompts_total,
        "n_tokens_in_prompt": config.n_tokens_in_prompt,
        "zero_out_bos_token": config.zero_out_bos_token,
    }
    converter_params = inspect.signature(module_any.main).parameters
    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in converter_params.values()
    )
    emit_arrow_override = _legacy_export_bundle_emit_arrow_override(config)
    if emit_arrow_override is not None and (accepts_var_keyword or "emit_arrow" in converter_params):
        params["emit_arrow"] = emit_arrow_override
    if "export_root" not in converter_params:
        module_any.OUTPUT_DIR = str(config.export_root)
    call_params = (
        params if accepts_var_keyword else {name: value for name, value in params.items() if name in converter_params}
    )
    module_any.main(SimpleNamespace(params=call_params), **call_params)
    source_id = _resolve_source_id(dashboard_leaf_dir, layer_num, config.neuronpedia_source_set_id)
    export_root = _resolve_export_root(config.export_root / config.model_name, source_id)
    return export_root


def run_dashboard_pipeline(config: NeuronpediaDashboardPipelineConfig) -> list[NeuronpediaDashboardLayerResult]:
    """Run dashboard generation, conversion, and optional local import for a layer range."""

    if config.import_only_local_db and not config.import_to_local_db:
        raise ValueError("--import-only-local-db cannot be combined with --skip-local-db-import")

    config.run_directory.mkdir(parents=True, exist_ok=True)
    pipeline_log_path = cast(Path, config.pipeline_log_path)
    existing_log_path = cast(Path, config.existing_log_path)
    logger = _configure_logger(pipeline_log_path)
    env = _build_generation_env(config)

    service_status: LocalNeuronpediaServiceStatus | None = None
    should_import = config.import_to_local_db
    if config.import_to_local_db:
        service_status = check_local_neuronpedia_services(
            local_db_url=config.local_db_url,
            webapp_url=config.webapp_url,
        )
        if not service_status.db_available:
            # Launch-5 lesson: a requested import silently degrading to generation-only cost a full
            # backfill pass. An unavailable/unresolvable DB with import_to_local_db=true is a hard
            # error unless the caller explicitly accepts degradation.
            if not config.allow_missing_local_db:
                raise RuntimeError(
                    "import_to_local_db=true but the local DB is unavailable/unresolvable: "
                    f"{service_status.db_error}. Fix the DB URL (--local-db-url / LOCAL_NEURONPEDIA_DB_URL / "
                    "POSTGRES_URL_NON_POOLING), pass --skip-local-db-import for an intentional "
                    "generation-only run, or pass --allow-missing-local-db to degrade with a warning."
                )
            should_import = False
            logger.warning("Local DB unavailable; continuing without DB import: %s", service_status.db_error)
        if not service_status.webapp_available:
            logger.warning("Local Neuronpedia webapp unavailable: %s", service_status.webapp_error)
        if service_status.db_url_redacted:
            logger.info("Resolved local DB URL: %s", service_status.db_url_redacted)
        if config.import_only_local_db and not should_import:
            raise RuntimeError(f"Import-only local DB mode requires a reachable local DB: {service_status.db_error}")

    completed_layers: set[int] = set()
    if config.resume_from_existing_logs:
        completed_layers = completed_layers_from_logs(*_completed_log_paths(config))

    requested_layers = set(config.requested_layer_numbers())
    completed_requested_layers = requested_layers & completed_layers

    logger.info(
        (
            "Starting dashboard pipeline model=%s set=%s worker=%s locks=%s layers=%s-%s run_directory=%s "
            "IT_NP_CACHE=%s HF_HOME=%s CUDA_VISIBLE_DEVICES=%s PYTORCH_CUDA_ALLOC_CONF=%s"
        ),
        config.model_name,
        config.neuronpedia_source_set_id,
        config.worker_id or "default",
        config.enable_layer_locks,
        config.start_layer,
        config.end_layer,
        config.run_directory,
        env.get("IT_NP_CACHE"),
        env.get("HF_HOME"),
        env.get("CUDA_VISIBLE_DEVICES"),
        env.get("PYTORCH_CUDA_ALLOC_CONF"),
    )

    if completed_requested_layers == requested_layers and requested_layers:
        logger.warning(
            "Requested layer range %s-%s is already complete in existing logs (%s, %s). "
            "Use --run-name-suffix or --run-root for a fresh run lineage, or --no-resume to ignore prior logs.",
            config.start_layer,
            config.end_layer,
            existing_log_path,
            pipeline_log_path,
        )

    _log_host_memory(logger, stage="pipeline_start")
    if _requires_pipeline_shared_prompt_artifacts(config):
        _ensure_shared_prompt_tokens_file(config, logger=logger)
        _log_host_memory(logger, stage="after_shared_prompt_tokens")
    else:
        logger.info("Deferring shared prompt token/effective-length preparation to the runner for this prompt path.")
        _log_host_memory(logger, stage="shared_prompt_tokens_deferred_to_runner")
    prompt_settings = _resolve_shared_prompt_run_settings(config)
    logger.info(
        (
            "Resolved prompt scheduling shared_tokens=%s prompts_total=%s tokens_per_prompt=%s "
            "prompts_in_forward_pass=%s primary_acts_batch_size=%s bucket_ceiling=%s"
        ),
        prompt_settings.shared_tokens_file,
        prompt_settings.n_prompts_total,
        prompt_settings.n_tokens_in_prompt,
        prompt_settings.n_prompts_in_forward_pass,
        prompt_settings.primary_acts_batch_size,
        prompt_settings.prompt_bucket_ceiling,
    )

    results: list[NeuronpediaDashboardLayerResult] = []
    import_executor: ThreadPoolExecutor | None = None
    # (layer_num, results index, layer_start, transferred layer-lock context manager, import future)
    pending_imports: deque[tuple[int, int, float, Any, Future]] = deque()

    def _drain_oldest_deferred_import() -> None:
        deferred_layer, result_index, deferred_layer_start, lock_cm, import_future = pending_imports.popleft()
        try:
            deferred_summary = import_future.result()
        except BaseException:
            logger.error(
                "Deferred local DB import failed for layer=%s; generated artifacts remain at %s and can be "
                "imported with --import-only-local-db (or by re-running: completed batch markers make "
                "regeneration a fast skip).",
                deferred_layer,
                results[result_index].output_dir,
            )
            raise
        finally:
            lock_cm.__exit__(None, None, None)
        _log_host_memory(logger, stage="after_columnar_import", layer_num=deferred_layer)
        logger.info(
            "Imported columnar layer=%s into local DB counts=%s",
            deferred_layer,
            deferred_summary.imported_row_counts,
        )
        deferred_elapsed = time.monotonic() - deferred_layer_start
        logger.info("DONE layer=%s elapsed_seconds=%.1f", deferred_layer, deferred_elapsed)
        results[result_index] = dataclass_replace(
            results[result_index], import_summary=deferred_summary, elapsed_seconds=deferred_elapsed
        )

    def _enqueue_deferred_layer_import(layer_num: int, output_dir: Path, layer_start: float, lock_cm: Any) -> None:
        nonlocal import_executor
        while pending_imports:
            _drain_oldest_deferred_import()
        if import_executor is None:
            import_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="np-local-db-import")
        import_future = import_executor.submit(
            import_columnar_dashboard_output, config, layer_num=layer_num, output_dir=output_dir
        )
        pending_imports.append((layer_num, len(results) - 1, layer_start, lock_cm, import_future))

    for layer_num in config.requested_layer_numbers():
        output_dir = config.output_dir_for_layer(layer_num)
        if config.import_only_local_db:
            layer_start = time.monotonic()
            import_kind, import_root = _find_existing_import_root(config, layer_num)
            logger.info("IMPORT_ONLY layer=%s import_kind=%s import_root=%s", layer_num, import_kind, import_root)
            if import_kind == "columnar":
                import_only_summary = import_columnar_dashboard_output(
                    config,
                    layer_num=layer_num,
                    output_dir=import_root,
                )
            else:
                prefer_arrow_for_tables, prefer_copy_for_tables = _legacy_export_bundle_import_preferences(config)
                import_only_summary = import_neuronpedia_export_bundle_local_db(
                    import_root,
                    local_db_url=config.local_db_url or "",
                    prefer_arrow_for_tables=prefer_arrow_for_tables,
                    prefer_copy_for_tables=prefer_copy_for_tables,
                )
            elapsed_seconds = time.monotonic() - layer_start
            logger.info(
                "Imported layer=%s into local DB counts=%s table_load_seconds=%s table_import_seconds=%s "
                "table_import_substage_seconds=%s",
                layer_num,
                import_only_summary.imported_row_counts,
                getattr(import_only_summary, "table_load_seconds", {}),
                getattr(import_only_summary, "table_import_seconds", {}),
                getattr(import_only_summary, "table_import_substage_seconds", {}),
            )
            logger.info("DONE layer=%s elapsed_seconds=%.1f", layer_num, elapsed_seconds)
            results.append(
                NeuronpediaDashboardLayerResult(
                    layer_num=layer_num,
                    output_dir=output_dir,
                    export_root=import_root,
                    import_summary=import_only_summary,
                    elapsed_seconds=elapsed_seconds,
                )
            )
            continue

        if config.resume_from_existing_logs:
            completed_layers = completed_layers_from_logs(*_completed_log_paths(config))
        if layer_num in completed_layers:
            logger.info("Skipping already completed layer=%s based on existing logs.", layer_num)
            results.append(
                NeuronpediaDashboardLayerResult(
                    layer_num=layer_num,
                    output_dir=output_dir,
                    export_root=None,
                    import_summary=None,
                    elapsed_seconds=0.0,
                    skipped=True,
                )
            )
            continue

        layer_lock_cm = _try_layer_lock(config, layer_num, logger=logger)
        lock_acquired = layer_lock_cm.__enter__()
        lock_transferred = False
        try:
            if not lock_acquired:
                results.append(
                    NeuronpediaDashboardLayerResult(
                        layer_num=layer_num,
                        output_dir=output_dir,
                        export_root=None,
                        import_summary=None,
                        elapsed_seconds=0.0,
                        skipped=True,
                    )
                )
                continue

            if config.resume_from_existing_logs:
                completed_layers = completed_layers_from_logs(*_completed_log_paths(config))
            if layer_num in completed_layers:
                logger.info("Skipping already completed layer=%s after acquiring lock.", layer_num)
                results.append(
                    NeuronpediaDashboardLayerResult(
                        layer_num=layer_num,
                        output_dir=output_dir,
                        export_root=None,
                        import_summary=None,
                        elapsed_seconds=0.0,
                        skipped=True,
                    )
                )
                continue

            if output_dir.exists() and not config.archive_partial_dirs:
                _validate_partial_layer_resume_compatibility(
                    config,
                    layer_num=layer_num,
                    output_dir=output_dir,
                    logger=logger,
                )

            if output_dir.exists() and config.archive_partial_dirs:
                archive_path = _archive_partial_output(output_dir)
                logger.info("Archived partial output for layer=%s to %s", layer_num, archive_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            layer_start = time.monotonic()
            logger.info(
                "START layer=%s worker=%s sae_path=%s output_dir=%s",
                layer_num,
                config.worker_id or "default",
                config.sae_path_for_layer(layer_num),
                output_dir,
            )
            _log_host_memory(logger, stage="before_generation", layer_num=layer_num)
            with pipeline_log_path.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    _layer_runner_command(config, layer_num, output_dir, prompt_settings=prompt_settings),
                    cwd=str(config.saedashboard_repo_root),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                return_code = _monitor_process(
                    process,
                    output_dir=output_dir,
                    logger=logger,
                    heartbeat_seconds=config.heartbeat_seconds,
                    stall_timeout_seconds=config.stall_timeout_seconds,
                )

            _log_host_memory(logger, stage="after_generation", layer_num=layer_num)
            if return_code != 0:
                _log_runtime_diagnostics(
                    logger,
                    pid=process.pid,
                    output_dir=output_dir,
                    reason=f"nonzero-exit-{return_code}",
                )
                try:
                    while pending_imports:
                        _drain_oldest_deferred_import()
                except Exception:
                    logger.exception("Deferred local DB import failed while handling a generation failure.")
                raise RuntimeError(f"Dashboard generation failed for layer {layer_num} with exit code {return_code}")

            dashboard_output_format = _resolve_runner_dashboard_output_format(config)
            if dashboard_output_format == "columnar" and not should_import:
                elapsed_seconds = time.monotonic() - layer_start
                _log_host_memory(logger, stage="layer_done", layer_num=layer_num)
                logger.info("Layer=%s columnar generation completed; output_dir=%s", layer_num, output_dir)
                logger.info(
                    "Skipping legacy Neuronpedia conversion for columnar dashboard output layer=%s; "
                    "local DB import is disabled.",
                    layer_num,
                )
                logger.info("DONE layer=%s elapsed_seconds=%.1f", layer_num, elapsed_seconds)
                results.append(
                    NeuronpediaDashboardLayerResult(
                        layer_num=layer_num,
                        output_dir=output_dir,
                        export_root=None,
                        import_summary=None,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
                continue
            if dashboard_output_format == "columnar":
                logger.info(
                    "Layer=%s columnar generation completed; columnar_root=%s",
                    layer_num,
                    output_dir,
                )
                if should_import and config.overlap_local_db_import:
                    generation_seconds = time.monotonic() - layer_start
                    _log_host_memory(logger, stage="layer_generation_done", layer_num=layer_num)
                    logger.info(
                        "Layer=%s generation completed in %.1f s; deferring local DB import to overlap with "
                        "the next layer's generation (layer lock held until the import completes).",
                        layer_num,
                        generation_seconds,
                    )
                    results.append(
                        NeuronpediaDashboardLayerResult(
                            layer_num=layer_num,
                            output_dir=output_dir,
                            export_root=output_dir,
                            import_summary=None,
                            elapsed_seconds=generation_seconds,
                        )
                    )
                    _enqueue_deferred_layer_import(layer_num, output_dir, layer_start, layer_lock_cm)
                    lock_transferred = True
                    continue
                import_summary = None
                if should_import:
                    _log_host_memory(logger, stage="before_columnar_import", layer_num=layer_num)
                    import_summary = import_columnar_dashboard_output(
                        config, layer_num=layer_num, output_dir=output_dir
                    )
                    _log_host_memory(logger, stage="after_columnar_import", layer_num=layer_num)
                    logger.info(
                        "Imported columnar layer=%s into local DB counts=%s",
                        layer_num,
                        import_summary.imported_row_counts,
                    )
                elapsed_seconds = time.monotonic() - layer_start
                _log_host_memory(logger, stage="layer_done", layer_num=layer_num)
                logger.info("DONE layer=%s elapsed_seconds=%.1f", layer_num, elapsed_seconds)
                results.append(
                    NeuronpediaDashboardLayerResult(
                        layer_num=layer_num,
                        output_dir=output_dir,
                        export_root=output_dir,
                        import_summary=import_summary,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
                continue
            logger.info(
                "Layer=%s generation completed; dashboard_leaf_dir=%s",
                layer_num,
                _resolve_dashboard_leaf_dir(output_dir),
            )
            if config.runner_implementation == "legacy" and not config.import_to_local_db:
                elapsed_seconds = time.monotonic() - layer_start
                _log_host_memory(logger, stage="layer_done", layer_num=layer_num)
                logger.info(
                    "Skipping Neuronpedia conversion for legacy generation-only layer=%s; local DB import is disabled.",
                    layer_num,
                )
                logger.info("DONE layer=%s elapsed_seconds=%.1f", layer_num, elapsed_seconds)
                results.append(
                    NeuronpediaDashboardLayerResult(
                        layer_num=layer_num,
                        output_dir=output_dir,
                        export_root=None,
                        import_summary=None,
                        elapsed_seconds=elapsed_seconds,
                    )
                )
                continue
            _log_host_memory(logger, stage="before_conversion", layer_num=layer_num)
            export_root = convert_dashboard_output(config, layer_num=layer_num, output_dir=output_dir, logger=logger)
            _log_host_memory(logger, stage="after_conversion", layer_num=layer_num)
            logger.info("Converted layer=%s to export_root=%s", layer_num, export_root)

            import_summary = None
            if should_import:
                prefer_arrow_for_tables, prefer_copy_for_tables = _legacy_export_bundle_import_preferences(config)
                _log_host_memory(logger, stage="before_import", layer_num=layer_num)
                import_summary = import_neuronpedia_export_bundle_local_db(
                    export_root,
                    local_db_url=config.local_db_url or "",
                    prefer_arrow_for_tables=prefer_arrow_for_tables,
                    prefer_copy_for_tables=prefer_copy_for_tables,
                )
                _log_host_memory(logger, stage="after_import", layer_num=layer_num)
                logger.info(
                    "Imported layer=%s into local DB counts=%s table_load_seconds=%s table_import_seconds=%s "
                    "table_import_substage_seconds=%s",
                    layer_num,
                    import_summary.imported_row_counts,
                    getattr(import_summary, "table_load_seconds", {}),
                    getattr(import_summary, "table_import_seconds", {}),
                    getattr(import_summary, "table_import_substage_seconds", {}),
                )

            elapsed_seconds = time.monotonic() - layer_start
            _log_host_memory(logger, stage="layer_done", layer_num=layer_num)
            logger.info("DONE layer=%s elapsed_seconds=%.1f", layer_num, elapsed_seconds)
            results.append(
                NeuronpediaDashboardLayerResult(
                    layer_num=layer_num,
                    output_dir=output_dir,
                    export_root=export_root,
                    import_summary=import_summary,
                    elapsed_seconds=elapsed_seconds,
                )
            )
        finally:
            if not lock_transferred:
                layer_lock_cm.__exit__(None, None, None)

    while pending_imports:
        _drain_oldest_deferred_import()
    if import_executor is not None:
        import_executor.shutdown(wait=True)
    return results


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description="Generate SAEDashboard layers and import them into a local Neuronpedia DB.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML config file for dashboard generation. Supports an optional 'pipeline' section and EXTENDS "
            "inheritance. Explicit CLI flags override config-file values."
        ),
    )
    parser.add_argument("--model-name")
    parser.add_argument("--model-layers", type=int)
    parser.add_argument("--sae-set")
    parser.add_argument("--neuronpedia-source-set-id")
    parser.add_argument("--neuronpedia-source-set-description")
    parser.add_argument("--creator-name")
    parser.add_argument("--release-id")
    parser.add_argument("--release-title")
    parser.add_argument("--release-url")
    parser.add_argument("--hf-weights-repo-id")
    parser.add_argument("--hf-weights-path-template")
    parser.add_argument("--hook-point")
    parser.add_argument("--prompts-huggingface-dataset-path")
    parser.add_argument("--prompts-dataset-mode", choices=PROMPT_DATASET_MODES)
    parser.add_argument("--prompts-huggingface-dataset-config-name")
    parser.add_argument("--prompts-huggingface-dataset-split")
    parser.add_argument("--prompts-dataset-text-field")
    parser.add_argument("--prompts-pretokenized-dataset-path", type=Path)
    parser.add_argument("--prompts-shared-tokens-file", type=Path)
    parser.add_argument("--start-layer", type=int)
    parser.add_argument("--end-layer", type=int)
    parser.add_argument(
        "--layer-list",
        type=lambda raw: [int(part) for part in raw.split(",") if part.strip()],
        help=(
            "Comma-separated explicit layer list (may be non-contiguous, e.g. 9,23). Overrides the "
            "--start-layer/--end-layer range for layer iteration; both bounds are still required for "
            "metadata."
        ),
    )
    parser.add_argument("--sae-path-template")
    parser.add_argument("--hf-model-path")
    parser.add_argument("--run-root")
    parser.add_argument("--run-name-suffix")
    parser.add_argument("--existing-log-path", type=Path)
    parser.add_argument("--pipeline-log-path", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--enable-layer-locks", action="store_true")
    parser.add_argument("--layer-lock-stale-seconds", type=int)
    parser.add_argument("--export-root")
    parser.add_argument("--saedashboard-repo-root")
    parser.add_argument("--saelens-repo-root")
    parser.add_argument("--neuronpedia-utils-root")
    parser.add_argument("--interpretune-env-file")
    parser.add_argument("--python-executable")
    parser.add_argument("--sae-dtype")
    parser.add_argument("--model-dtype")
    parser.add_argument("--sparsity-threshold", type=int)
    parser.add_argument("--n-prompts-total", type=int)
    parser.add_argument("--n-tokens-in-prompt", type=int)
    parser.add_argument("--n-features-per-batch", type=int)
    parser.add_argument("--n-prompts-in-forward-pass", type=int)
    parser.add_argument("--deduplicate-shared-prompt-tokens", action=argparse.BooleanOptionalAction)
    parser.add_argument("--strict-shared-prompt-count", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--prompt-bucket-ceilings",
        help="Optional comma-separated list or JSON array of inclusive prompt-length bucket ceilings.",
    )
    parser.add_argument(
        "--prompt-bucket-ceiling",
        type=int,
        help="Select one staged prompt-length bucket for a bucket-specific pilot run.",
    )
    parser.add_argument(
        "--primary-acts-batch-size",
        type=int,
        help=(
            "Optional internal activation-capture chunk size passed to SAEDashboard. This keeps "
            "--n-prompts-in-forward-pass as the logical dashboard batch while lowering model-forward peak memory."
        ),
    )
    parser.add_argument("--start-batch", type=int)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument("--use-clt", action="store_true")
    parser.add_argument("--clt-dtype")
    parser.add_argument("--clt-weights-filename")
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model-wrapper", choices=("hooked", "bridge"))
    parser.add_argument(
        "--bridge-enable-compatibility-mode",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--bridge-compatibility-mode-kwargs-json")
    parser.add_argument("--runner-log-resource-snapshots", action="store_true")
    parser.add_argument("--runner-log-hook-aliases", action="store_true")
    parser.add_argument("--runner-log-performance", action="store_true")
    parser.add_argument("--runner-profile-rolling-substages", action="store_true")
    parser.add_argument("--runner-shuffle-tokens", action=argparse.BooleanOptionalAction)
    parser.add_argument("--runner-implementation", choices=("current", "legacy"))
    parser.add_argument("--runner-cleanup-each-minibatch", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--runner-correlation-accumulation-device",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--runner-rolling-coefficient-num-threads", type=int)
    parser.add_argument("--runner-activation-significance-floor", type=float)
    parser.add_argument("--runner-converter-input-artifact-dir", type=Path)
    parser.add_argument("--runner-feature-statistics-backend", choices=("object", "arrow"))
    parser.add_argument("--runner-logits-histogram-backend", choices=("object", "arrow"))
    parser.add_argument("--runner-defer-component-construction", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--runner-sequence-selection-backend",
        choices=("legacy", "columnar_gpu"),
    )
    parser.add_argument(
        "--runner-dashboard-output-format",
        choices=("auto", "legacy_json", "columnar"),
    )
    parser.add_argument(
        "--legacy-export-bundle-contract",
        choices=LEGACY_EXPORT_BUNDLE_CONTRACTS,
        help=(
            "Override the legacy export/import contract used after legacy_json dashboard generation. "
            "Use 'preserved_baseline' to force detached-baseline JSONL-only conversion and local import behavior."
        ),
    )
    parser.add_argument(
        "--runner-columnar-artifact-format",
        choices=("arrow", "parquet"),
    )
    parser.add_argument(
        "--runner-emit-activation-copy-rows",
        action=argparse.BooleanOptionalAction,
        help=(
            "Override whether columnar dashboard runs also emit activation_copy_rows. "
            "Defaults to enabled when local DB import is requested and disabled for "
            "generation-only runs."
        ),
    )
    parser.add_argument(
        "--runner-overlap-batch-packaging",
        action=argparse.BooleanOptionalAction,
        help=(
            "Overlap each columnar batch's CPU packaging tail and artifact writes with "
            "the next batch's forward/encode inside the SAEDashboard runner. Batch "
            "completion markers stay ordered, so batch-level resume (including across "
            "GPUs) is unchanged; a crash re-generates at most the in-flight batches."
        ),
    )
    parser.add_argument(
        "--runner-sequence-top-acts-positive-only",
        action=argparse.BooleanOptionalAction,
        help="Opt-in: restrict TOP-ACTIVATIONS candidates to strictly positive activations (columnar backend only).",
    )
    parser.add_argument(
        "--runner-sequence-dedup-across-groups",
        action=argparse.BooleanOptionalAction,
        help="Opt-in: a coordinate appears in at most one sequence group (columnar backend only).",
    )
    parser.add_argument(
        "--runner-sequence-skip-dead-features",
        action=argparse.BooleanOptionalAction,
        help="Opt-in: emit no sequence rows for features with no positive activation (columnar backend only).",
    )
    parser.add_argument(
        "--runner-sequence-half-open-interval-bins",
        action=argparse.BooleanOptionalAction,
        help=(
            "Opt-in: numpy-histogram-style interval membership ([lower, upper), highest interval "
            "closed) in the in-tree legacy and columnar selectors."
        ),
    )
    parser.add_argument(
        "--runner-columnar-max-staged-acts-bytes",
        type=int,
        default=None,
        help=(
            "Opt-in: byte budget capping SAEDashboard's device retention/staging of the columnar "
            "activation matrix (default: SD's fixed 4 GiB; 0 forces host staging so dense layers fit "
            "at large n_prompts). Outputs are bit-identical; only peak GPU memory and speed move."
        ),
    )
    parser.add_argument(
        "--runner-columnar-row-chunk-size",
        type=int,
        default=None,
        help=(
            "Opt-in: feature-row chunk size for SAEDashboard's arrow statistics/histogram packaging "
            "loops (SD defaults 256/128), bounding density-scaled per-chunk GPU transients on dense "
            "layers. Outputs are bit-identical."
        ),
    )
    parser.add_argument(
        "--runner-logits-table-mask-token-pattern",
        type=str,
        help="Optional regex over token strings; matching vocab rows are excluded from logits tables.",
    )
    parser.add_argument(
        "--local-db-import-dedup-activation-rows",
        action=argparse.BooleanOptionalAction,
        help="Opt-in: deduplicate per-feature (tokens, values) activation records at local DB import.",
    )
    parser.add_argument(
        "--local-db-import-drop-zero-activation-rows",
        action=argparse.BooleanOptionalAction,
        help="Opt-in: drop maxValue == 0 activation records at local DB import.",
    )
    parser.add_argument("--runner-prompt-bucket-schedule-file", type=Path)
    parser.add_argument("--runner-auto-prompt-bucket-schedule", action="store_true")
    parser.add_argument("--runner-prompt-bucket-ceilings")
    parser.add_argument("--runner-prompt-bucket-scale-limit", type=float)
    parser.add_argument("--runner-prompt-primary-acts-scale-limit", type=float)
    parser.add_argument("--runner-prompt-batch-size-round-to", type=int)
    parser.add_argument("--runner-torch-profile", action="store_true")
    parser.add_argument("--runner-torch-profile-dir", type=Path)
    parser.add_argument(
        "--runner-use-cached-activations",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--heartbeat-seconds", type=int)
    parser.add_argument("--stall-timeout-seconds", type=int)
    parser.add_argument("--local-db-url")
    parser.add_argument("--local-db-import-chunk-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--overlap-local-db-import",
        action=argparse.BooleanOptionalAction,
        help=(
            "Overlap each columnar layer's local DB import with the next layer's generation. The layer lock "
            "is held and the DONE resume marker is only written once the deferred import completes, so "
            "resume and cross-GPU semantics are unchanged; at most one layer's import is in flight."
        ),
    )
    parser.add_argument("--webapp-url")
    parser.add_argument("--use-skip-transcoder", action="store_true")
    parser.add_argument("--zero-out-bos-token", action="store_true")
    parser.add_argument("--skip-local-db-import", action="store_true")
    parser.add_argument(
        "--allow-missing-local-db",
        action=argparse.BooleanOptionalAction,
        help=(
            "Opt-in: when import_to_local_db is requested but the local DB is unavailable/unresolvable, "
            "warn and continue generation-only instead of hard-erroring."
        ),
    )
    parser.add_argument(
        "--import-only-local-db",
        action="store_true",
        help=(
            "Skip generation and conversion, then import previously generated Neuronpedia export bundles or columnar "
            "dashboard layer outputs for the requested layer range into the local DB."
        ),
    )
    parser.add_argument("--no-archive-partials", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--torch-cuda-alloc-conf")
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _create_argument_parser().parse_args(argv)


def main() -> int:
    parser = _create_argument_parser()
    args = parser.parse_args()
    try:
        config = _build_dashboard_pipeline_config(args)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    run_dashboard_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
