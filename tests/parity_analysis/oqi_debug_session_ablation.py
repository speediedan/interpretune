#!/usr/bin/env python3
"""Manual ablation driver for notebook OQI debug-session parity.

This script runs the notebook-side debug intervention validation under targeted session-construction overrides and
compares the resulting pre-intervention runtime-state fingerprints against a known-good standalone artifact.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence, cast
from unittest.mock import patch

import torch
from circuit_tracer import Graph

from interpretune.utils.resource_mgmt import cleanup_python_cuda, safe_clean_cuda
from tests import load_dotenv
from tests.analysis_resource_utils import clear_nnsight_test_state, serial_test_cleanup
from tests.configuration import config_modules
from tests.conftest import FixtPhase, session_fixture_hook_exec
from tests.concept_direction_approach_parity import concept_direction_experiment_utils as cdeu
from tests.concept_direction_approach_parity import experiment_resource_utils as eru
from tests.concept_direction_approach_parity.nb_experiment_launcher import load_flat_yaml
from tests.parity_analysis.intervention_drift_analysis import (
    PRESERVE_ARTIFACTS_ENV,
    PRESERVE_ARTIFACT_DIR_ENV,
    tensor_fingerprint,
)


UNSET = object()
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "concept_direction_approach_parity"
    / "configs"
    / "gemma3_4b_it_local_oqi_reasoning_single_fs_di_60.yaml"
)
DEFAULT_REFERENCE_GLOB = "gemma3_4b_it_oqi_gemma_3_4b_it__25_gemmascope_2_transcoder_16k__60_*"


ConfigSurfaceValue = Any
CudaWindowMode = Literal["module", "replacement_model", "none"]


@dataclass(frozen=True)
class SessionSurfaceOverride:
    name: str
    description: str
    force_device: ConfigSurfaceValue = UNSET
    nnsight_device_map: ConfigSurfaceValue = UNSET
    nnsight_attn_implementation: ConfigSurfaceValue = UNSET
    nnsight_torch_dtype: ConfigSurfaceValue = UNSET
    circuit_tracer_dtype: ConfigSurfaceValue = UNSET
    analysis_target_tokens: ConfigSurfaceValue = UNSET
    target_token_ids: ConfigSurfaceValue = UNSET
    offload: ConfigSurfaceValue = UNSET
    verbose: ConfigSurfaceValue = UNSET
    cuda_window: CudaWindowMode = "module"


VARIANT_PRESETS: dict[str, SessionSurfaceOverride] = {
    "dtype_only": SessionSurfaceOverride(
        name="dtype_only",
        description="Force circuit-tracer dtype to float32 while leaving the notebook session surface unchanged.",
        circuit_tracer_dtype=torch.float32,
    ),
    "dtype_eager_no_targets": SessionSurfaceOverride(
        name="dtype_eager_no_targets",
        description="Add eager attention and clear analysis target-token defaults on top of float32.",
        circuit_tracer_dtype=torch.float32,
        nnsight_attn_implementation="eager",
        analysis_target_tokens=None,
        target_token_ids=None,
    ),
    "parity_surface": SessionSurfaceOverride(
        name="parity_surface",
        description="Match the parity fixture config surface while preserving the auto-selected notebook device.",
        nnsight_attn_implementation="eager",
        nnsight_torch_dtype="float32",
        circuit_tracer_dtype=torch.float32,
        analysis_target_tokens=None,
        target_token_ids=None,
        offload="cpu",
        verbose=False,
        cuda_window="module",
    ),
    "full_parity_surface": SessionSurfaceOverride(
        name="full_parity_surface",
        description="Match the parity fixture config surface and replacement-model CUDA window semantics.",
        nnsight_attn_implementation="eager",
        nnsight_torch_dtype="float32",
        circuit_tracer_dtype=torch.float32,
        analysis_target_tokens=None,
        target_token_ids=None,
        offload="cpu",
        verbose=False,
        cuda_window="replacement_model",
    ),
}

DEFAULT_VARIANTS = ["dtype_only", "dtype_eager_no_targets", "parity_surface", "full_parity_surface"]

RUNTIME_COMPARE_PATHS: dict[str, tuple[str, ...]] = {
    "module.device_map": ("module", "nnsight_cfg", "device_map"),
    "module.attn_implementation": ("module", "nnsight_cfg", "attn_implementation"),
    "module.circuit_tracer_dtype": ("module", "circuit_tracer_cfg", "dtype"),
    "module.analysis_target_tokens": ("module", "circuit_tracer_cfg", "analysis_target_tokens"),
    "module.offload": ("module", "circuit_tracer_cfg", "offload"),
    "module.verbose": ("module", "circuit_tracer_cfg", "verbose"),
    "replacement_model.dtype": ("module", "replacement_model", "dtype"),
    "graph.input_tokens": ("graph_op", "result", "input_tokens", "sha256"),
    "graph.active_features": ("graph_op", "result", "active_features", "sha256"),
    "graph.selected_feature_rows": ("graph_op", "result", "selected_feature_rows", "sha256"),
    "graph.adjacency_matrix": ("graph_op", "result", "adjacency_matrix", "sha256"),
    "graph.logit_target_ids": ("graph_op", "result", "logit_target_ids", "sha256"),
    "baseline.baseline_logits": ("baseline_forward", "baseline_logits", "sha256"),
    "baseline.activation_cache": ("baseline_forward", "baseline_activation_cache", "sha256"),
    "baseline.selected_feature_activation": ("baseline_forward", "selected_feature_baseline_activation"),
}

GRAPH_FINGERPRINT_KEYS = {
    "graph.input_tokens",
    "graph.active_features",
    "graph.selected_feature_rows",
    "graph.adjacency_matrix",
    "graph.logit_target_ids",
}
BASELINE_FINGERPRINT_KEYS = {
    "baseline.baseline_logits",
    "baseline.activation_cache",
    "baseline.selected_feature_activation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Flat YAML config for the notebook debug run.")
    parser.add_argument(
        "--reference-artifact",
        help="Reference artifact directory or summary.json. Defaults to the latest passing standalone 25/60 artifact.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parent / "artifacts"),
        help="Root directory used for preserved ablation artifacts.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=sorted(VARIANT_PRESETS),
        help="Variant presets to run in order.",
    )
    parser.add_argument(
        "--stop-on-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after the first variant that matches all pre-intervention fingerprints.",
    )
    parser.add_argument("--output-json", help="Optional path to write the full ablation summary JSON.")
    return parser.parse_args()


def _nested_get(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _short_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) == 64:
        return value[:12]
    return value


def _resolve_reference_artifact(path: str | None, artifact_root: Path) -> Path:
    if path is not None:
        candidate = Path(path)
        if candidate.is_dir():
            return candidate / "summary.json"
        return candidate

    candidates = sorted(artifact_root.glob(DEFAULT_REFERENCE_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"Unable to find a reference artifact matching {DEFAULT_REFERENCE_GLOB!r} under {artifact_root}"
        )
    return candidates[-1] / "summary.json"


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = path if path.name == "summary.json" else path / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _artifact_dir_from_path(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _summarize_comparisons(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    tracked_keys = GRAPH_FINGERPRINT_KEYS | BASELINE_FINGERPRINT_KEYS
    graph_matches = {
        label: payload["matches"] for label, payload in comparisons.items() if label in GRAPH_FINGERPRINT_KEYS
    }
    baseline_matches = {
        label: payload["matches"] for label, payload in comparisons.items() if label in BASELINE_FINGERPRINT_KEYS
    }
    return {
        "graph_fingerprints_match": all(graph_matches.values()),
        "baseline_fingerprints_match": all(baseline_matches.values()),
        "all_pre_intervention_matches": all(
            payload["matches"] for label, payload in comparisons.items() if label in tracked_keys
        ),
        "pre_intervention_match_count": sum(
            1 for label, payload in comparisons.items() if label in tracked_keys and payload["matches"]
        ),
        "pre_intervention_total": len(tracked_keys),
    }


def _fingerprint_sha(value: Any) -> str:
    fingerprint = tensor_fingerprint(value)
    if fingerprint is None:
        raise ValueError("Expected a tensor-like value for fingerprinting.")
    return str(fingerprint["sha256"])


def _compare_preserved_artifacts(
    reference_artifact_dir: Path, candidate_artifact_dir: Path
) -> dict[str, dict[str, Any]]:
    reference_bundle = torch.load(reference_artifact_dir / "bundle.pt", map_location="cpu", weights_only=False)
    candidate_bundle = torch.load(candidate_artifact_dir / "bundle.pt", map_location="cpu", weights_only=False)
    reference_logits = (
        torch.as_tensor(reference_bundle["baseline_logits"], dtype=torch.float32).detach().cpu().reshape(-1)
    )
    candidate_logits = (
        torch.as_tensor(candidate_bundle["baseline_logits"], dtype=torch.float32).detach().cpu().reshape(-1)
    )
    logits_diff = (candidate_logits - reference_logits).abs()

    reference_graph = Graph.from_pt(str(reference_artifact_dir / "graph.pt"), map_location="cpu")
    candidate_graph = Graph.from_pt(str(candidate_artifact_dir / "graph.pt"), map_location="cpu")
    reference_adjacency = (
        torch.as_tensor(reference_graph.adjacency_matrix, dtype=torch.float32).detach().cpu().contiguous()
    )
    candidate_adjacency = (
        torch.as_tensor(candidate_graph.adjacency_matrix, dtype=torch.float32).detach().cpu().contiguous()
    )
    adjacency_diff = (candidate_adjacency - reference_adjacency).abs()
    logit_column_start = reference_adjacency.shape[1] - len(reference_graph.logit_targets)
    feature_column_diff = adjacency_diff[:, :logit_column_start]
    logit_column_diff = adjacency_diff[:, logit_column_start:]

    return {
        "baseline.baseline_logits": {
            "matches": bool(torch.equal(reference_logits, candidate_logits)),
            "reference": _short_value(_fingerprint_sha(reference_logits)),
            "candidate": _short_value(_fingerprint_sha(candidate_logits)),
            "max_abs_diff": float(logits_diff.max().item()) if logits_diff.numel() else 0.0,
            "source": "bundle.pt",
        },
        "graph.adjacency_matrix": {
            "matches": bool(torch.equal(reference_adjacency, candidate_adjacency)),
            "reference": _short_value(_fingerprint_sha(reference_adjacency)),
            "candidate": _short_value(_fingerprint_sha(candidate_adjacency)),
            "max_abs_diff": float(adjacency_diff.max().item()) if adjacency_diff.numel() else 0.0,
            "feature_cols_max_abs_diff": float(feature_column_diff.max().item())
            if feature_column_diff.numel()
            else 0.0,
            "logit_cols_max_abs_diff": float(logit_column_diff.max().item()) if logit_column_diff.numel() else 0.0,
            "count_gt_1e_5": int((adjacency_diff > 1e-5).sum().item()),
            "count_gt_1e_4": int((adjacency_diff > 1e-4).sum().item()),
            "source": "graph.pt",
        },
    }


def _build_notebook_cfg(config_path: Path) -> tuple[cdeu.NotebookHarnessConfig, bool]:
    payload = load_flat_yaml(config_path)
    config_name = config_path.stem
    experiment_name = str(payload.get("EXPERIMENT_NAME", config_name))
    work_root = eru.create_work_root(payload.get("EXPERIMENT_WORK_DIR"), experiment_name)
    should_cleanup_work_root = payload.get("EXPERIMENT_WORK_DIR") is None

    model_family = str(payload["MODEL_FAMILY"])
    model_variant = str(payload["MODEL_VARIANT"])
    prompt_render_mode = cast(
        cdeu.PromptRenderMode,
        str(payload.get("PROMPT_RENDER_MODE", "plain")).strip(),
    )
    model_spec = eru.resolve_model_spec(
        model_family,
        model_variant,
        model_name_override=payload.get("MODEL_NAME_OVERRIDE"),
        transcoder_set_override=payload.get("TRANSCODER_SET_OVERRIDE"),
        neuronpedia_model_override=payload.get("NEURONPEDIA_MODEL_OVERRIDE"),
        neuronpedia_set_override=payload.get("NEURONPEDIA_SET_OVERRIDE"),
    )
    runtime = cdeu.resolve_neuronpedia_runtime_config(
        use_localhost=bool(payload.get("USE_LOCALHOST", False)),
        neuronpedia_base_url_override=payload.get("NEURONPEDIA_BASE_URL_OVERRIDE"),
        local_db_url=payload.get("LOCAL_NEURONPEDIA_DB_URL"),
        local_webapp_url=payload.get("LOCAL_NEURONPEDIA_WEBAPP_URL"),
        check_local_explanation_coverage=bool(payload.get("CHECK_LOCAL_EXPLANATION_COVERAGE", False)),
        generate_missing_local_explanations=bool(payload.get("GENERATE_MISSING_LOCAL_EXPLANATIONS", False)),
        local_explanation_feature_limit=int(payload.get("LOCAL_EXPLANATION_FEATURE_LIMIT", 20)),
    )
    concept_pair_path = cdeu.resolve_concept_pair_config_path(
        payload.get("CONCEPT_PAIR_NAME"),
        model_family=model_family,
        prompt_render_mode=prompt_render_mode,
        concept_pair_config_path=payload.get("CONCEPT_PAIR_CONFIG_PATH"),
    )
    concept_pair = cdeu.load_concept_pair(concept_pair_path)
    prompt_override = payload.get("PROMPT_OVERRIDE")
    prompt = (
        str(prompt_override)
        if prompt_override
        else (
            concept_pair.chat_intervention_prompt
            if prompt_render_mode != "plain" and concept_pair.chat_intervention_prompt is not None
            else concept_pair.intervention_prompt
        )
    )

    target_tokens = None
    if payload.get("TARGET_TOKENS") is not None:
        target_tokens_raw = tuple(str(token) for token in payload["TARGET_TOKENS"])
        if len(target_tokens_raw) != 2:
            raise ValueError("TARGET_TOKENS must contain exactly two entries when provided")
        target_tokens = cast(tuple[str, str], target_tokens_raw)

    target_token_ids = None
    if payload.get("TARGET_TOKEN_IDS") is not None:
        target_token_ids_raw = tuple(int(token_id) for token_id in payload["TARGET_TOKEN_IDS"])
        if len(target_token_ids_raw) != 2:
            raise ValueError("TARGET_TOKEN_IDS must contain exactly two entries when provided")
        target_token_ids = cast(tuple[int, int], target_token_ids_raw)

    explicit_direction_tokens = None
    if payload.get("EXPLICIT_DIRECTION_TOKENS") is not None:
        explicit_direction_tokens_raw = tuple(str(token) for token in payload["EXPLICIT_DIRECTION_TOKENS"])
        if len(explicit_direction_tokens_raw) != 2:
            raise ValueError("EXPLICIT_DIRECTION_TOKENS must contain exactly two entries when provided")
        explicit_direction_tokens = cast(tuple[str, str], explicit_direction_tokens_raw)

    analysis_mode = cast(cdeu.AnalysisMode, str(payload.get("ANALYSIS_MODE", "concept_pair")).strip())
    store_latent_extraction_mode = cast(
        cdeu.StoreLatentExtractionMode,
        str(payload.get("STORE_LATENT_EXTRACTION_MODE", "answer_position_state")).strip(),
    )
    debug_session_surface_preset = cast(
        cdeu.DebugSessionSurfacePreset,
        str(payload.get("DEBUG_SESSION_SURFACE_PRESET", "notebook_default")).strip(),
    )

    cfg = cdeu.NotebookHarnessConfig(
        experiment_name=experiment_name,
        experiment_config_name=str(payload.get("EXPERIMENT_CONFIG_NAME", config_name)),
        model_family=model_family,
        model_variant=model_variant,
        model_name=model_spec.model_name,
        transcoder_set=model_spec.transcoder_set,
        hf_model_head=model_spec.hf_model_head,
        neuronpedia_model=model_spec.neuronpedia_model,
        neuronpedia_set=model_spec.neuronpedia_set,
        neuronpedia_base_url=runtime.base_url,
        concept_pair_name=payload.get("CONCEPT_PAIR_NAME"),
        concept_pair_config_path=str(concept_pair_path),
        prompt=prompt,
        prompt_render_mode=prompt_render_mode,
        target_tokens=target_tokens,
        target_token_ids=target_token_ids,
        top_n=int(payload.get("TOP_N", 10)),
        default_scale_factor=float(payload.get("DEFAULT_SCALE_FACTOR", 10.0)),
        scale_factor_sweep=list(payload.get("SCALE_FACTOR_SWEEP", [2.0, 5.0, 10.0, 20.0, 50.0])),
        ablation_n_list=list(payload.get("ABLATION_N_LIST", [5, 10, 25, 50, 100])),
        enable_sign_aware=bool(payload.get("ENABLE_SIGN_AWARE", True)),
        force_device=payload.get("FORCE_DEVICE"),
        work_root=work_root,
        analysis_mode=analysis_mode,
        explicit_direction_tokens=explicit_direction_tokens,
        enable_zero_softcap=bool(payload.get("ENABLE_ZERO_SOFTCAP", False)),
        batch_size=int(payload["BATCH_SIZE"]) if payload.get("BATCH_SIZE") is not None else None,
        max_feature_nodes=(int(payload["MAX_FEATURE_NODES"]) if payload.get("MAX_FEATURE_NODES") is not None else None),
        use_localhost=runtime.use_localhost,
        local_neuronpedia_db_url=runtime.local_db_url,
        local_neuronpedia_webapp_url=runtime.local_webapp_url,
        check_local_explanation_coverage=runtime.check_local_explanation_coverage,
        generate_missing_local_explanations=runtime.generate_missing_local_explanations,
        local_explanation_feature_limit=runtime.local_explanation_feature_limit,
        local_neuronpedia_service_status=runtime.service_status,
        mode_warning_messages=runtime.warning_messages,
        local_explanation_type_name=str(payload.get("LOCAL_EXPLANATION_TYPE_NAME", cdeu.DEFAULT_EXPLANATION_TYPE_NAME)),
        local_explanation_timeout_seconds=int(
            payload.get("LOCAL_EXPLANATION_TIMEOUT_SECONDS", cdeu.DEFAULT_COPILOT_TIMEOUT_SECONDS)
        ),
        local_explanation_max_retries=int(
            payload.get("LOCAL_EXPLANATION_MAX_RETRIES", cdeu.DEFAULT_COPILOT_MAX_RETRIES)
        ),
        local_explanation_retry_backoff_seconds=float(
            payload.get(
                "LOCAL_EXPLANATION_RETRY_BACKOFF_SECONDS",
                cdeu.DEFAULT_COPILOT_RETRY_BACKOFF_SECONDS,
            )
        ),
        key_tokens_override=tuple(payload["KEY_TOKENS"]) if payload.get("KEY_TOKENS") is not None else None,
        constrained_feature_selection_refs=payload.get("CONSTRAINED_FEATURE_SELECTION_LIST"),
        store_latent_extraction_mode=store_latent_extraction_mode,
        context_enhanced_scale=float(payload.get("CONTEXT_ENHANCED_SCALE", 1.0)),
        enable_baseline_path_debug=bool(payload.get("ENABLE_BASELINE_PATH_DEBUG", False)),
        debug_validation_logit_atol=float(payload.get("DEBUG_VALIDATION_LOGIT_ATOL", 1e-4)),
        debug_validation_logit_rtol=float(payload.get("DEBUG_VALIDATION_LOGIT_RTOL", 1e-3)),
        debug_validation_act_atol=float(payload.get("DEBUG_VALIDATION_ACT_ATOL", 1e-3)),
        debug_validation_act_rtol=float(payload.get("DEBUG_VALIDATION_ACT_RTOL", 1e-5)),
        debug_validation_top_k=int(payload.get("DEBUG_VALIDATION_TOP_K", 10)),
        debug_validation_raise_on_failure=bool(payload.get("DEBUG_VALIDATION_RAISE_ON_FAILURE", True)),
        debug_session_surface_preset=debug_session_surface_preset,
    )
    return cfg, should_cleanup_work_root


def _apply_override_attr(target: Any, attr_name: str, value: Any) -> None:
    if value is not UNSET and target is not None:
        setattr(target, attr_name, value)


def _select_cuda_context(module: Any, replacement_model: Any, mode: CudaWindowMode):
    if not torch.cuda.is_available():
        return nullcontext()
    if mode == "module":
        target = module if hasattr(module, "to") else getattr(module, "model", None)
    elif mode == "replacement_model":
        target = replacement_model
    else:
        return nullcontext()
    return safe_clean_cuda(target) if target is not None else nullcontext()


def _make_experiment_session(override: SessionSurfaceOverride):
    @contextmanager
    def _experiment_session(
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
    ) -> Iterator[tuple[Any, Any, Any]]:
        del use_cuda_cleanup
        full_run_name = f"{run_name}_{override.name}"
        session_dir = Path(work_root) / full_run_name
        session_dir.mkdir(parents=True, exist_ok=True)

        clear_nnsight_test_state(None)
        cleanup_python_cuda()
        load_dotenv()

        effective_force_device = force_device if override.force_device is UNSET else override.force_device
        cfg = eru.build_test_cfg(
            model_family,
            model_name=model_name,
            transcoder_set=transcoder_set,
            force_device=effective_force_device,
            hf_model_head=hf_model_head,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
        )
        _apply_override_attr(getattr(cfg, "nnsight_cfg", None), "device_map", override.nnsight_device_map)
        _apply_override_attr(
            getattr(cfg, "nnsight_cfg", None), "attn_implementation", override.nnsight_attn_implementation
        )
        _apply_override_attr(getattr(cfg, "nnsight_cfg", None), "torch_dtype", override.nnsight_torch_dtype)
        _apply_override_attr(getattr(cfg, "circuit_tracer_cfg", None), "dtype", override.circuit_tracer_dtype)
        _apply_override_attr(
            getattr(cfg, "circuit_tracer_cfg", None), "analysis_target_tokens", override.analysis_target_tokens
        )
        _apply_override_attr(getattr(cfg, "circuit_tracer_cfg", None), "target_token_ids", override.target_token_ids)
        _apply_override_attr(getattr(cfg, "circuit_tracer_cfg", None), "offload", override.offload)
        _apply_override_attr(getattr(cfg, "circuit_tracer_cfg", None), "verbose", override.verbose)

        it_session = config_modules(cfg, full_run_name, {}, session_dir, {}, False)
        session_fixture_hook_exec(it_session, FixtPhase.setup)
        module = it_session.module
        assert module is not None
        replacement_model = module.replacement_model
        tokenizer = replacement_model.tokenizer

        try:
            with serial_test_cleanup(
                it_session,
                module,
                replacement_model,
                clear_cuda=override.cuda_window == "none",
            ):
                with _select_cuda_context(module, replacement_model, override.cuda_window):
                    yield it_session, module, tokenizer
        finally:
            clear_nnsight_test_state(it_session)
            cleanup_python_cuda()

    return _experiment_session


def _compare_runtime_state(reference_state: Mapping[str, Any], candidate_state: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for label, path in RUNTIME_COMPARE_PATHS.items():
        reference_value = _nested_get(reference_state, path)
        candidate_value = _nested_get(candidate_state, path)
        comparisons[label] = {
            "matches": reference_value == candidate_value,
            "reference": _short_value(reference_value),
            "candidate": _short_value(candidate_value),
        }

    return {"comparisons": comparisons, **_summarize_comparisons(comparisons)}


def _run_variant(
    cfg: cdeu.NotebookHarnessConfig,
    override: SessionSurfaceOverride,
    reference_summary: Mapping[str, Any],
    reference_artifact_dir: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    with patch.dict(
        os.environ,
        {
            PRESERVE_ARTIFACTS_ENV: "1",
            PRESERVE_ARTIFACT_DIR_ENV: str(artifact_root),
        },
        clear=False,
    ):
        with patch.object(cdeu, "experiment_session", _make_experiment_session(override)):
            result = cdeu.run_debug_intervention_validation(cfg)

    runtime_comparison = _compare_runtime_state(
        reference_summary["metadata"]["runtime_state"],
        result["runtime_state"],
    )
    artifact_dir = Path(result["artifact_dir"]).resolve() if result.get("artifact_dir") else None
    artifact_comparison: dict[str, dict[str, Any]] | None = None
    if artifact_dir is not None:
        artifact_comparison = _compare_preserved_artifacts(reference_artifact_dir, artifact_dir)
        runtime_comparison["comparisons"].update(artifact_comparison)
        runtime_comparison.update(_summarize_comparisons(runtime_comparison["comparisons"]))

    variant_summary = {
        "name": override.name,
        "description": override.description,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "selected_feature": list(result["selected_feature"]),
        "selected_feature_graph_index": int(result["selected_feature_graph_index"]),
        "reference_selected_feature_index": int(reference_summary["report"]["selected_feature_index"]),
        "candidate_selected_feature_index": int(result["drift_report"]["selected_feature_index"]),
        "all_passed": bool(result["all_passed"]),
        "activation_max_abs_error": float(result["activation_max_abs_error"]),
        "logit_max_abs_error": float(result["logit_max_abs_error"]),
        "runtime_comparison": runtime_comparison,
        "artifact_comparison": artifact_comparison,
        "surface_snapshot": {
            "device_map": _nested_get(result["runtime_state"], ("module", "nnsight_cfg", "device_map")),
            "attn_implementation": _nested_get(
                result["runtime_state"], ("module", "nnsight_cfg", "attn_implementation")
            ),
            "circuit_tracer_dtype": _nested_get(result["runtime_state"], ("module", "circuit_tracer_cfg", "dtype")),
            "analysis_target_tokens": _nested_get(
                result["runtime_state"], ("module", "circuit_tracer_cfg", "analysis_target_tokens")
            ),
            "offload": _nested_get(result["runtime_state"], ("module", "circuit_tracer_cfg", "offload")),
            "verbose": _nested_get(result["runtime_state"], ("module", "circuit_tracer_cfg", "verbose")),
            "cuda_window": override.cuda_window,
        },
    }
    return variant_summary


def _print_variant_summary(summary: Mapping[str, Any]) -> None:
    runtime_comparison = summary["runtime_comparison"]
    baseline_payload = runtime_comparison["comparisons"]["baseline.baseline_logits"]
    adjacency_payload = runtime_comparison["comparisons"]["graph.adjacency_matrix"]
    print(
        "[{}] pre-match={}/{} graph_match={} baseline_match={} all_passed={} act_err={:.4f} logit_err={:.4f}".format(
            summary["name"],
            runtime_comparison["pre_intervention_match_count"],
            runtime_comparison["pre_intervention_total"],
            runtime_comparison["graph_fingerprints_match"],
            runtime_comparison["baseline_fingerprints_match"],
            summary["all_passed"],
            summary["activation_max_abs_error"],
            summary["logit_max_abs_error"],
        )
    )
    print(
        "  surface device_map={} attn={} dtype={} targets={} offload={} verbose={} cuda_window={}".format(
            summary["surface_snapshot"]["device_map"],
            summary["surface_snapshot"]["attn_implementation"],
            summary["surface_snapshot"]["circuit_tracer_dtype"],
            summary["surface_snapshot"]["analysis_target_tokens"],
            summary["surface_snapshot"]["offload"],
            summary["surface_snapshot"]["verbose"],
            summary["surface_snapshot"]["cuda_window"],
        )
    )
    print(f"  artifact {summary['artifact_dir']}")
    if baseline_payload.get("source") is not None:
        print(
            "  baseline logits via {} match={} max_abs_diff={:.4g}".format(
                baseline_payload["source"],
                baseline_payload["matches"],
                baseline_payload.get("max_abs_diff", 0.0),
            )
        )
    if adjacency_payload.get("source") is not None:
        print(
            "  adjacency via {} match={} max_abs_diff={:.4g} feature_cols_max_abs_diff={:.4g} "
            "logit_cols_max_abs_diff={:.4g} count_gt_1e-5={} count_gt_1e-4={}".format(
                adjacency_payload["source"],
                adjacency_payload["matches"],
                adjacency_payload.get("max_abs_diff", 0.0),
                adjacency_payload.get("feature_cols_max_abs_diff", 0.0),
                adjacency_payload.get("logit_cols_max_abs_diff", 0.0),
                adjacency_payload.get("count_gt_1e_5", 0),
                adjacency_payload.get("count_gt_1e_4", 0),
            )
        )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    reference_path = _resolve_reference_artifact(args.reference_artifact, artifact_root)
    reference_artifact_dir = _artifact_dir_from_path(reference_path)
    reference_summary = _load_summary(reference_path)
    cfg, should_cleanup_work_root = _build_notebook_cfg(config_path)

    run_summaries: list[dict[str, Any]] = []
    try:
        for variant_name in args.variants:
            override = VARIANT_PRESETS[variant_name]
            summary = _run_variant(cfg, override, reference_summary, reference_artifact_dir, artifact_root)
            run_summaries.append(summary)
            _print_variant_summary(summary)
            if args.stop_on_match and summary["runtime_comparison"]["all_pre_intervention_matches"]:
                break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if should_cleanup_work_root:
            shutil.rmtree(cfg.work_root, ignore_errors=True)

    output = {
        "config": str(config_path),
        "reference_artifact": str(reference_path),
        "variant_order": list(args.variants),
        "variants": run_summaries,
    }
    output_json = json.dumps(output, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(output_json + "\n", encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
