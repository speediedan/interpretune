from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


CONFIG_EXTENDS_KEY = "EXTENDS"
_MISSING = object()


def _parse_env_override(raw_value: str) -> Any:
    stripped = raw_value.strip()
    if not stripped:
        return raw_value

    lower_value = stripped.lower()
    if lower_value in {"true", "false", "null", "none", "~"}:
        return yaml.safe_load(stripped)

    if stripped[0] in "[{(" or stripped[0] in "\"'":
        return yaml.safe_load(stripped)

    number_chars = set("0123456789+-.eE")
    if set(stripped) <= number_chars and any(char.isdigit() for char in stripped):
        return yaml.safe_load(stripped)

    return raw_value


def _get_env_override(flat_key: str) -> Any:
    if flat_key not in os.environ:
        return _MISSING
    return _parse_env_override(os.environ[flat_key])


@dataclass(frozen=True)
class HarnessModelConfig:
    family: str
    variant: str
    model_name: str
    transcoder_set: str
    neuronpedia_model: str
    neuronpedia_set: str
    hf_model_head: str | None = None


@dataclass(frozen=True)
class HarnessPromptConfig:
    prompt: str
    render_mode: str
    target_tokens: tuple[str, str] | None
    target_token_ids: tuple[int, int] | None
    key_tokens: tuple[str, ...] | None
    explicit_direction_tokens: tuple[str, str] | None


@dataclass(frozen=True)
class HarnessSessionConfig:
    force_device: str | None
    batch_size: int | None
    max_feature_nodes: int | None
    debug_session_surface_preset: str


@dataclass(frozen=True)
class HarnessNeuronpediaConfig:
    base_url: str
    use_localhost: bool
    local_db_url: str | None
    local_webapp_url: str
    upload_local_graphs: bool
    local_graph_slug_prefix: str | None
    local_graph_upload_target: str
    local_graph_owner_username: str | None
    check_local_explanation_coverage: bool
    generate_missing_local_explanations: bool
    local_explanation_feature_limit: int
    local_explanation_type_name: str
    local_explanation_timeout_seconds: int
    local_explanation_max_retries: int
    local_explanation_retry_backoff_seconds: float
    local_neuronpedia_service_status: Any = None
    mode_warning_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class HarnessDebugValidationConfig:
    enable_zero_softcap: bool
    enable_baseline_path_debug: bool
    logit_atol: float
    logit_rtol: float
    act_atol: float
    act_rtol: float
    top_k: int
    raise_on_failure: bool


@dataclass(frozen=True)
class SharedHarnessSections:
    model: HarnessModelConfig
    prompt: HarnessPromptConfig
    session: HarnessSessionConfig
    neuronpedia: HarnessNeuronpediaConfig
    debug_validation: HarnessDebugValidationConfig


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must parse to a mapping.")
    return dict(payload)


def deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge_mappings(base_value, value)
        else:
            merged[key] = value
    return merged


def _resolve_extends_paths(config_path: Path, extends_value: Any) -> list[Path]:
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


def load_experiment_config(config_path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved_path = Path(config_path).expanduser().resolve()
    if resolved_path in _seen:
        chain = " -> ".join(str(path) for path in (*_seen, resolved_path))
        raise ValueError(f"Detected cyclic config inheritance: {chain}")

    payload = _load_yaml_mapping(resolved_path)
    extends_value = payload.pop(CONFIG_EXTENDS_KEY, None)

    merged_payload: dict[str, Any] = {}
    for parent_path in _resolve_extends_paths(resolved_path, extends_value):
        merged_payload = deep_merge_mappings(
            merged_payload,
            load_experiment_config(parent_path, _seen=(*_seen, resolved_path)),
        )

    merged_payload = deep_merge_mappings(merged_payload, payload)
    merged_payload.setdefault("EXPERIMENT_NAME", resolved_path.stem)
    merged_payload.setdefault("EXPERIMENT_CONFIG_NAME", resolved_path.stem)
    merged_payload["EXPERIMENT_CONFIG_PATH"] = str(resolved_path)
    return merged_payload


def get_config_value(
    payload: Mapping[str, Any],
    *,
    section: str,
    key: str,
    flat_key: str,
    default: Any = None,
) -> Any:
    env_override = _get_env_override(flat_key)
    if env_override is not _MISSING:
        return env_override

    section_value = payload.get(section)
    if isinstance(section_value, Mapping) and key in section_value:
        return section_value[key]
    return payload.get(flat_key, default)


def get_required_config_value(
    payload: Mapping[str, Any],
    *,
    section: str,
    key: str,
    flat_key: str,
) -> Any:
    value = get_config_value(payload, section=section, key=key, flat_key=flat_key)
    if value is None:
        raise ValueError(f"Config is missing required value '{flat_key}' (or nested '{section}.{key}').")
    return value


def build_shared_harness_sections(
    *,
    model_family: str,
    model_variant: str,
    model_name: str,
    transcoder_set: str,
    neuronpedia_model: str,
    neuronpedia_set: str,
    hf_model_head: str | None,
    prompt: str,
    prompt_render_mode: str,
    target_tokens: tuple[str, str] | None,
    target_token_ids: tuple[int, int] | None,
    key_tokens: tuple[str, ...] | None,
    explicit_direction_tokens: tuple[str, str] | None,
    force_device: str | None,
    batch_size: int | None,
    max_feature_nodes: int | None,
    debug_session_surface_preset: str,
    neuronpedia_base_url: str,
    use_localhost: bool,
    local_neuronpedia_db_url: str | None,
    local_neuronpedia_webapp_url: str,
    upload_local_graphs: bool,
    local_graph_slug_prefix: str | None,
    local_graph_upload_target: str,
    local_graph_owner_username: str | None,
    check_local_explanation_coverage: bool,
    generate_missing_local_explanations: bool,
    local_explanation_feature_limit: int,
    local_explanation_type_name: str,
    local_explanation_timeout_seconds: int,
    local_explanation_max_retries: int,
    local_explanation_retry_backoff_seconds: float,
    local_neuronpedia_service_status: Any,
    mode_warning_messages: tuple[str, ...],
    enable_zero_softcap: bool,
    enable_baseline_path_debug: bool,
    debug_validation_logit_atol: float,
    debug_validation_logit_rtol: float,
    debug_validation_act_atol: float,
    debug_validation_act_rtol: float,
    debug_validation_top_k: int,
    debug_validation_raise_on_failure: bool,
) -> SharedHarnessSections:
    return SharedHarnessSections(
        model=HarnessModelConfig(
            family=model_family,
            variant=model_variant,
            model_name=model_name,
            transcoder_set=transcoder_set,
            neuronpedia_model=neuronpedia_model,
            neuronpedia_set=neuronpedia_set,
            hf_model_head=hf_model_head,
        ),
        prompt=HarnessPromptConfig(
            prompt=prompt,
            render_mode=prompt_render_mode,
            target_tokens=target_tokens,
            target_token_ids=target_token_ids,
            key_tokens=key_tokens,
            explicit_direction_tokens=explicit_direction_tokens,
        ),
        session=HarnessSessionConfig(
            force_device=force_device,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            debug_session_surface_preset=debug_session_surface_preset,
        ),
        neuronpedia=HarnessNeuronpediaConfig(
            base_url=neuronpedia_base_url,
            use_localhost=use_localhost,
            local_db_url=local_neuronpedia_db_url,
            local_webapp_url=local_neuronpedia_webapp_url,
            upload_local_graphs=upload_local_graphs,
            local_graph_slug_prefix=local_graph_slug_prefix,
            local_graph_upload_target=local_graph_upload_target,
            local_graph_owner_username=local_graph_owner_username,
            check_local_explanation_coverage=check_local_explanation_coverage,
            generate_missing_local_explanations=generate_missing_local_explanations,
            local_explanation_feature_limit=local_explanation_feature_limit,
            local_explanation_type_name=local_explanation_type_name,
            local_explanation_timeout_seconds=local_explanation_timeout_seconds,
            local_explanation_max_retries=local_explanation_max_retries,
            local_explanation_retry_backoff_seconds=local_explanation_retry_backoff_seconds,
            local_neuronpedia_service_status=local_neuronpedia_service_status,
            mode_warning_messages=mode_warning_messages,
        ),
        debug_validation=HarnessDebugValidationConfig(
            enable_zero_softcap=enable_zero_softcap,
            enable_baseline_path_debug=enable_baseline_path_debug,
            logit_atol=debug_validation_logit_atol,
            logit_rtol=debug_validation_logit_rtol,
            act_atol=debug_validation_act_atol,
            act_rtol=debug_validation_act_rtol,
            top_k=debug_validation_top_k,
            raise_on_failure=debug_validation_raise_on_failure,
        ),
    )
