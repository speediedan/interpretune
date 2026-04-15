from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, cast

import torch
import yaml  # type: ignore[import-untyped]

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
for _path in (REPO_ROOT, TESTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from interpretune.config.nnsight import NNsightConfig  # noqa: E402
from interpretune.utils.resource_mgmt import cleanup_python_cuda, safe_clean_cuda  # noqa: E402
from tests import load_dotenv  # noqa: E402
from tests.analysis_resource_utils import clear_nnsight_test_state, serial_test_cleanup  # noqa: E402
from tests.configuration import config_modules  # noqa: E402
from tests.conftest import FixtPhase, session_fixture_hook_exec  # noqa: E402


DebugSessionSurfacePreset = Literal["notebook_default", "parity_surface"]


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MODEL_SPECS_PATH = CONFIG_DIR / "model_specs.yaml"
SESSION_SURFACE_PRESETS_PATH = CONFIG_DIR / "session_surface_presets.yaml"


@dataclass(frozen=True)
class ModelSpec:
    family: str
    variant: str
    model_name: str
    transcoder_set: str
    neuronpedia_model: str
    neuronpedia_set: str
    use_chat_template: bool
    cfg_class: str
    hf_model_head: str | None = None
    nnsight_overrides: dict[str, Any] | None = None
    circuit_tracer_overrides: dict[str, Any] | None = None


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return payload


def _import_object(import_path: str) -> Any:
    module_name, _, attr_path = import_path.partition(":")
    if not attr_path:
        module_name, _, attr_path = import_path.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(f"Invalid import path: {import_path}")
    module = importlib.import_module(module_name)
    value = module
    for attr_name in attr_path.split("."):
        value = getattr(value, attr_name)
    return value


def _resolve_special_value(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("torch."):
            return getattr(torch, value.removeprefix("torch."))
        if value.startswith("import:"):
            return _import_object(value.removeprefix("import:"))
    if isinstance(value, list):
        return [_resolve_special_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_special_value(item) for key, item in value.items()}
    return value


def _format_override_value(value: Any, *, spec: ModelSpec, device_type: str) -> Any:
    if isinstance(value, str):
        format_values = {
            "device_type": device_type,
            "family": spec.family,
            "variant": spec.variant,
            "model_name": spec.model_name,
            "transcoder_set": spec.transcoder_set,
            "neuronpedia_model": spec.neuronpedia_model,
            "neuronpedia_set": spec.neuronpedia_set,
        }
        try:
            return value.format(**format_values)
        except KeyError:
            return value
    if isinstance(value, list):
        return [_format_override_value(item, spec=spec, device_type=device_type) for item in value]
    if isinstance(value, dict):
        return {key: _format_override_value(item, spec=spec, device_type=device_type) for key, item in value.items()}
    return value


def _set_nested_attr(target: Any, attr_path: str, value: Any) -> None:
    path_parts = attr_path.split(".")
    current = target
    for attr_name in path_parts[:-1]:
        next_value = getattr(current, attr_name, None)
        if next_value is None:
            raise AttributeError(f"Cannot resolve '{attr_path}' on {type(target).__name__}")
        current = next_value
    setattr(current, path_parts[-1], value)


@lru_cache(maxsize=1)
def _model_spec_registry() -> dict[tuple[str, str], ModelSpec]:
    payload = _load_yaml_mapping(MODEL_SPECS_PATH)
    raw_specs = payload.get("model_specs")
    if not isinstance(raw_specs, dict):
        raise ValueError(f"Expected 'model_specs' mapping in {MODEL_SPECS_PATH}")

    registry: dict[tuple[str, str], ModelSpec] = {}
    for raw_key, raw_value in raw_specs.items():
        if not isinstance(raw_value, dict):
            raise ValueError(f"Model spec '{raw_key}' must map to a dictionary")
        family, _, variant = str(raw_key).partition(".")
        if not family or not variant:
            raise ValueError(f"Model spec key '{raw_key}' must use 'family.variant' format")
        registry[(family, variant)] = ModelSpec(
            family=family,
            variant=variant,
            model_name=str(raw_value["model_name"]),
            transcoder_set=str(raw_value["transcoder_set"]),
            neuronpedia_model=str(raw_value["neuronpedia_model"]),
            neuronpedia_set=str(raw_value["neuronpedia_set"]),
            use_chat_template=bool(raw_value["use_chat_template"]),
            cfg_class=str(raw_value["cfg_class"]),
            hf_model_head=cast(str | None, raw_value.get("hf_model_head")),
            nnsight_overrides=cast(dict[str, Any] | None, raw_value.get("nnsight_overrides")),
            circuit_tracer_overrides=cast(dict[str, Any] | None, raw_value.get("circuit_tracer_overrides")),
        )
    return registry


@lru_cache(maxsize=1)
def _session_surface_presets() -> dict[str, dict[str, Any]]:
    payload = _load_yaml_mapping(SESSION_SURFACE_PRESETS_PATH)
    raw_presets = payload.get("session_surface_presets")
    if not isinstance(raw_presets, dict):
        raise ValueError(f"Expected 'session_surface_presets' mapping in {SESSION_SURFACE_PRESETS_PATH}")
    return {str(key): cast(dict[str, Any], value) for key, value in raw_presets.items()}


def resolve_model_spec(
    model_family: str,
    model_variant: str,
    *,
    model_name_override: str | None = None,
    transcoder_set_override: str | None = None,
    neuronpedia_model_override: str | None = None,
    neuronpedia_set_override: str | None = None,
    use_chat_template_override: bool | None = None,
) -> ModelSpec:
    try:
        spec = _model_spec_registry()[(model_family, model_variant)]
    except KeyError as exc:
        supported = ", ".join(f"{family}.{variant}" for family, variant in sorted(_model_spec_registry()))
        raise ValueError(f"Unsupported model selection {model_family}.{model_variant}. Supported: {supported}") from exc

    return ModelSpec(
        family=spec.family,
        variant=spec.variant,
        model_name=model_name_override or spec.model_name,
        transcoder_set=transcoder_set_override or spec.transcoder_set,
        neuronpedia_model=neuronpedia_model_override or spec.neuronpedia_model,
        neuronpedia_set=neuronpedia_set_override or spec.neuronpedia_set,
        use_chat_template=spec.use_chat_template if use_chat_template_override is None else use_chat_template_override,
        cfg_class=spec.cfg_class,
        hf_model_head=spec.hf_model_head,
        nnsight_overrides=spec.nnsight_overrides,
        circuit_tracer_overrides=spec.circuit_tracer_overrides,
    )


def _find_model_spec_for_cfg(model_family: str, model_name: str | None, model_variant: str | None) -> ModelSpec:
    registry = _model_spec_registry()
    if model_variant is not None:
        return registry[(model_family, model_variant)]

    matches = [
        spec
        for (family, _variant), spec in registry.items()
        if family == model_family and (model_name is None or spec.model_name == model_name)
    ]
    if not matches:
        raise ValueError(f"No model spec matches family={model_family!r} model_name={model_name!r}")
    if len(matches) > 1:
        variants = ", ".join(spec.variant for spec in matches)
        raise ValueError(
            f"Model family {model_family!r} with model_name={model_name!r} is ambiguous; choose a variant: {variants}"
        )
    return matches[0]


def _apply_override_mapping(
    target: Any,
    override_mapping: Mapping[str, Any],
    *,
    spec: ModelSpec,
    device_type: str,
) -> None:
    for attr_path, raw_value in override_mapping.items():
        resolved_value = _resolve_special_value(_format_override_value(raw_value, spec=spec, device_type=device_type))
        _set_nested_attr(target, attr_path, resolved_value)


def _apply_generic_override_mapping(target: Any, override_mapping: Mapping[str, Any]) -> None:
    for attr_path, raw_value in override_mapping.items():
        _set_nested_attr(target, attr_path, _resolve_special_value(raw_value))


def apply_debug_session_surface_preset(cfg: Any, *, preset: DebugSessionSurfacePreset) -> None:
    if preset == "notebook_default":
        return

    preset_payload = _session_surface_presets().get(preset)
    if preset_payload is None:
        supported = ", ".join(sorted(_session_surface_presets()))
        raise ValueError(f"Unsupported debug session surface preset: {preset}. Supported: {supported}")

    raw_session_overrides = preset_payload.get("session_overrides", preset_payload)
    if not isinstance(raw_session_overrides, Mapping):
        raise ValueError(f"Session surface preset '{preset}' must define a mapping of session overrides")

    for target_name, override_mapping in raw_session_overrides.items():
        target = getattr(cfg, target_name, None)
        if target is None:
            continue
        if not isinstance(override_mapping, Mapping):
            raise ValueError(
                f"Session surface preset '{preset}' target '{target_name}' must map to a dictionary of overrides"
            )
        _apply_generic_override_mapping(target, override_mapping)


def resolve_session_surface_preset_config_defaults(preset: DebugSessionSurfacePreset) -> dict[str, Any]:
    if preset == "notebook_default":
        return {}

    preset_payload = _session_surface_presets().get(preset)
    if preset_payload is None:
        supported = ", ".join(sorted(_session_surface_presets()))
        raise ValueError(f"Unsupported debug session surface preset: {preset}. Supported: {supported}")

    config_defaults = preset_payload.get("config_defaults", {})
    if not isinstance(config_defaults, Mapping):
        raise ValueError(
            f"Session surface preset '{preset}' config_defaults must map to a dictionary of section defaults"
        )

    return {str(key): value for key, value in config_defaults.items()}


def build_test_cfg(
    model_family: str,
    *,
    model_variant: str | None = None,
    model_name: str,
    transcoder_set: str,
    force_device: str | None = None,
    hf_model_head: str | None = None,
    batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    debug_session_surface_preset: DebugSessionSurfacePreset = "notebook_default",
) -> Any:
    device_type = force_device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = _find_model_spec_for_cfg(model_family, model_name, model_variant)
    cfg_class = _import_object(spec.cfg_class)
    cfg = cfg_class(phase="test", device_type=device_type)

    circuit_tracer_cfg = getattr(cfg, "circuit_tracer_cfg", None)
    if circuit_tracer_cfg is not None:
        circuit_tracer_cfg.model_name = model_name
        circuit_tracer_cfg.transcoder_set = transcoder_set
        if hf_model_head is not None and hasattr(circuit_tracer_cfg, "hf_model_head"):
            circuit_tracer_cfg.hf_model_head = hf_model_head
        if batch_size is not None:
            circuit_tracer_cfg.batch_size = batch_size
        if max_feature_nodes is not None:
            circuit_tracer_cfg.max_feature_nodes = max_feature_nodes
        if spec.circuit_tracer_overrides:
            _apply_override_mapping(
                circuit_tracer_cfg,
                spec.circuit_tracer_overrides,
                spec=spec,
                device_type=device_type,
            )

    nnsight_cfg = getattr(cfg, "nnsight_cfg", None)
    if nnsight_cfg is None and spec.nnsight_overrides is not None:
        cfg.nnsight_cfg = NNsightConfig()
        nnsight_cfg = cfg.nnsight_cfg
    if nnsight_cfg is not None and spec.nnsight_overrides:
        _apply_override_mapping(nnsight_cfg, spec.nnsight_overrides, spec=spec, device_type=device_type)

    apply_debug_session_surface_preset(cfg, preset=debug_session_surface_preset)
    return cfg


@contextmanager
def experiment_session(
    work_root: Path,
    run_name: str,
    *,
    model_family: str,
    model_name: str,
    transcoder_set: str,
    force_device: str | None = None,
    use_cuda_cleanup: bool = True,
    hf_model_head: str | None = None,
    batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    debug_session_surface_preset: DebugSessionSurfacePreset = "notebook_default",
    model_variant: str | None = None,
) -> Iterator[tuple[Any, Any, Any]]:
    session_dir = work_root / run_name
    session_dir.mkdir(parents=True, exist_ok=True)

    clear_nnsight_test_state(None)
    cleanup_python_cuda()
    load_dotenv()

    cfg = build_test_cfg(
        model_family,
        model_variant=model_variant,
        model_name=model_name,
        transcoder_set=transcoder_set,
        force_device=force_device,
        hf_model_head=hf_model_head,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        debug_session_surface_preset=debug_session_surface_preset,
    )
    it_session = config_modules(cfg, run_name, {}, session_dir, {}, False)
    session_fixture_hook_exec(it_session, cast(FixtPhase, FixtPhase.setup))

    module = it_session.module
    assert module is not None
    replacement_model = cast(Any, module).replacement_model
    tokenizer = replacement_model.tokenizer

    cuda_target = module if hasattr(module, "to") else getattr(module, "model", None)
    cuda_cleanup = (
        safe_clean_cuda(cuda_target)
        if use_cuda_cleanup and torch.cuda.is_available() and cuda_target is not None
        else nullcontext()
    )

    try:
        with serial_test_cleanup(it_session, module, replacement_model, clear_cuda=not use_cuda_cleanup):
            with cuda_cleanup:
                yield it_session, module, tokenizer
    finally:
        clear_nnsight_test_state(it_session)
        cleanup_python_cuda()
