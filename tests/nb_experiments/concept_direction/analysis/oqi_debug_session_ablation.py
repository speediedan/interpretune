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
from tests.nb_experiments import pipeline_patterns
from tests import load_dotenv
from tests.analysis_resource_utils import clear_nnsight_test_state, serial_test_cleanup
from tests.configuration import config_modules
from tests.conftest import FixtPhase, session_fixture_hook_exec
from tests.nb_experiments.concept_direction.concept_direction import (
    NotebookHarnessConfig,
    build_notebook_harness_config,
)
from tests.nb_experiments.pipeline_patterns import (
    run_debug_intervention_validation,
)
from tests.nb_experiments.session import build_test_cfg
from tests.nb_experiments.concept_direction.analysis.intervention_drift_analysis import (
    PRESERVE_ARTIFACTS_ENV,
    PRESERVE_ARTIFACT_DIR_ENV,
    tensor_fingerprint,
)


UNSET = object()
# The pinned gemma3_4b_it_local_oqi_reasoning_oh_2975_15708.yaml debug config was removed 2026-07-07
# (Ohio 4B parity closed out; see tests/nb_experiments/EXPERIMENT_STATUS.md). Pass --config explicitly
# to target a specific experiment surface; the Ohio base config remains the default.
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base_oqi_reasoning_oh.yaml"
DEFAULT_REFERENCE_GLOB = "gemma3_4b_it_ohio_fs_2975_15708_*"


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
        help="Reference artifact directory or summary.json. Defaults to the latest passing Ohio 2975/15708 artifact.",
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


def _build_notebook_cfg(config_path: Path) -> tuple[NotebookHarnessConfig, bool]:
    cfg, should_cleanup_work_root, _resolved_payload = build_notebook_harness_config(config_path)
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
        model_variant: str | None = None,
        model_name: str,
        transcoder_set: str,
        force_device: str | None = None,
        use_cuda_cleanup: bool = True,
        hf_model_head: str | None = None,
        batch_size: int | None = None,
        max_feature_nodes: int | None = None,
        debug_session_surface_preset: str = "notebook_default",
    ) -> Iterator[tuple[Any, Any, Any]]:
        del use_cuda_cleanup
        full_run_name = f"{run_name}_{override.name}"
        session_dir = Path(work_root) / full_run_name
        session_dir.mkdir(parents=True, exist_ok=True)

        clear_nnsight_test_state(None)
        cleanup_python_cuda()
        load_dotenv()

        effective_force_device = force_device if override.force_device is UNSET else override.force_device
        cfg = build_test_cfg(
            model_family,
            model_variant=model_variant,
            model_name=model_name,
            transcoder_set=transcoder_set,
            force_device=effective_force_device,
            hf_model_head=hf_model_head,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            debug_session_surface_preset=cast(Any, debug_session_surface_preset),
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
    cfg: NotebookHarnessConfig,
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
        with patch.object(pipeline_patterns, "experiment_session", _make_experiment_session(override)):
            result = run_debug_intervention_validation(cfg)

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
