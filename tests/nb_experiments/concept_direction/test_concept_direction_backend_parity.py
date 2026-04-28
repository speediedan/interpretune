from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import interpretune as it
import pytest
import torch
from circuit_tracer import Graph, attribute
from circuit_tracer.attribution.targets import CustomTarget
from torch.testing import assert_close
from transformers import BatchEncoding

from interpretune.analysis.backends import require_analysis_backend
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.helpers import _flatten_concept_store_rows, last_token_logits
from tests.conftest import clean_cuda
from tests.nb_experiments.concept_direction.concept_direction import PromptRenderMode
from tests.core.test_analysis_backend_parity import (
    EDGE_LOGIT_ATOL,
    EDGE_LOGIT_RTOL,
    FEATURE_EDGE_ACT_ATOL,
    FEATURE_EDGE_ACT_RTOL,
    Gemma3InstructionInterventionCase,
    VALUE_ATOL,
    VALUE_RTOL,
    ConceptDirectionParityArtifacts,
    ConceptDirectionStageResult,
    Gemma3ConceptDirectionParityCase,
    _build_graph_edge_validation_context,
    _ensure_analysis_cfg,
    _feature_row_from_tensor,
    _find_active_feature_index,
    _verify_wrapper_feature_interventions,
)
from tests.nb_experiments.nb_harness_utils import _build_feature_selection_spec
from tests.nb_experiments.nb_harness_utils import (
    _extract_top_features_with_optional_filter,
    _serialize_constrained_feature_selection,
    _serialize_intervention_call_kwargs,
    _summarize_graph_input_tokens,
)
from tests.nb_experiments.config import load_experiment_config
from tests.nb_experiments.session import experiment_session, resolve_model_spec
from tests.nb_experiments.concept_direction.analysis.concept_direction_analysis import (
    DEFAULT_RANDOM_PERTURBATION_SCALE,
    DEFAULT_RANDOM_PERTURBATION_SEED,
    PRESERVE_ARTIFACTS_ENV,
    build_classification_prompt_text,
    build_concept_direction_stage_artifact,
    build_context_extraction_artifact,
    build_prompt_alignment_artifact,
    build_prompt_alignment_snapshot,
    build_random_vector_perturbation,
    capture_context_enhanced_extraction_snapshot,
    compare_top_feature_sets,
    compute_concept_direction_geometry,
    cosine_similarity_value,
    normalize_prompt_entity_text,
    resolve_prompt_alignment_context_index,
    save_concept_direction_parity_report,
    save_concept_direction_pipeline_state_artifacts,
    save_concept_direction_reference_graph_report,
)
from tests.nb_experiments.concept_direction.analysis.intervention_drift_analysis import (
    build_intervention_drift_report,
    resolve_artifact_output_dir,
    save_preserved_intervention_artifacts,
    snapshot_analysis_batch,
    snapshot_module_runtime_state,
    tensor_fingerprint,
)
from tests.runif import RunIf


RUNIF: Any = RunIf
pytest_plugins = ("tests.core.test_analysis_backend_parity",)
FEATURE_DELTA_SIGN_ATOL = 1.0


@pytest.fixture
def gemma3_instruction_intervention_case() -> Gemma3InstructionInterventionCase:
    return Gemma3InstructionInterventionCase(
        prompt="<bos><start_of_turn>user\nThe National Digital Analytics Group (ND",
        model_name="google/gemma-3-1b-it",
        transcoder_set="mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine",
        pos_start=4,
        token_position_limit=2,
        error_layer_limit=3,
        feature_sample_count=4,
        reference_feature_sample_count=100,
        intervention_scale_factor=2.0,
    )


def _render_debug_prompt(tokenizer: Any, prompt: str, prompt_render_mode: str) -> str:
    if prompt_render_mode == "plain":
        return prompt
    if prompt_render_mode == "apply_chat_template":
        return cast(
            str,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
    raise ValueError(f"Unsupported debug prompt render mode: {prompt_render_mode!r}")


def _normalize_debug_graph_target_token(token: str, *, use_chat_template: bool) -> str:
    if not use_chat_template:
        return token
    normalized = token.lstrip(" ▁Ġ")
    return normalized or token


def _resolve_debug_graph_target_ids(
    tokenizer: Any,
    key_tokens: tuple[str, ...],
    *,
    use_chat_template: bool,
) -> tuple[list[int], list[str]]:
    token_ids: list[int] = []
    labels: list[str] = []
    seen_ids: set[int] = set()
    for token in key_tokens:
        normalized = _normalize_debug_graph_target_token(token, use_chat_template=use_chat_template)
        encoded = tokenizer.encode(normalized, add_special_tokens=False)
        if not encoded:
            continue
        token_id = int(encoded[-1])
        if token_id in seen_ids:
            continue
        seen_ids.add(token_id)
        token_ids.append(token_id)
        labels.append(normalized)
    assert token_ids, "Expected at least one debug graph target token id"
    return token_ids, labels


def _resolve_debug_graph_target_token_variants(
    tokenizer: Any,
    token: str,
    *,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    candidates = {token, _normalize_debug_graph_target_token(token, use_chat_template=use_chat_template)}
    if use_chat_template:
        normalized = _normalize_debug_graph_target_token(token, use_chat_template=True)
        candidates.add(f" {normalized}")
        if token.startswith("▁"):
            candidates.add(" " + token.lstrip("▁"))

    variants: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1:
            continue
        token_id = int(encoded[0])
        if token_id in seen_ids:
            continue
        seen_ids.add(token_id)
        variants.append({"token": candidate, "token_id": token_id, "decoded": tokenizer.decode([token_id])})
    return sorted(variants, key=lambda item: int(item["token_id"]))


def _summarize_top_logits(tokenizer: Any, logits: torch.Tensor, *, top_k: int = 10) -> list[dict[str, Any]]:
    logits = logits.detach().float().cpu().reshape(-1)
    values, indices = torch.topk(logits, min(top_k, int(logits.numel())))
    return [
        {
            "rank": rank,
            "token_id": int(token_id.item()),
            "token": tokenizer.decode([int(token_id.item())]),
            "logit": float(value.item()),
        }
        for rank, (token_id, value) in enumerate(zip(indices, values))
    ]


def _resolve_parity_config_path(config_name: str) -> Path:
    root = Path(__file__).resolve().parent
    candidates = (
        root / "configs" / config_name,
        root / "archived_cfgs" / config_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve concept-direction parity config: {config_name}")


def _build_gemma3_it_concept_direction_parity_case(
    *,
    config_path: Path,
    concept_pair_path: Path,
    calibration_surface: str,
    parity_artifact_name: str,
    reference_artifact_name: str,
    session_name: str,
    require_cross_path_feature_overlap: bool = True,
    require_gap_improvement: bool = True,
) -> Gemma3ConceptDirectionParityCase:
    payload = load_experiment_config(config_path)
    concept_pair_payload = load_experiment_config(concept_pair_path)

    model_payload = cast(dict[str, Any], payload["MODEL"])
    prompt_payload = cast(dict[str, Any], payload["PROMPT"])
    analysis_payload = cast(dict[str, Any], payload["ANALYSIS"])
    session_payload = cast(dict[str, Any], payload["SESSION"])
    model_spec = resolve_model_spec(str(model_payload["family"]), str(model_payload["variant"]))
    target_tokens = tuple(str(token) for token in prompt_payload["target_tokens"])
    assert len(target_tokens) == 2
    resolved_config_path = Path(payload["EXPERIMENT_CONFIG_PATH"]).resolve()
    config_name = str(payload.get("EXPERIMENT_CONFIG_NAME", resolved_config_path.stem))
    experiment_name = str(payload.get("EXPERIMENT_NAME", config_name))
    batch_size = int(session_payload["batch_size"])
    max_feature_nodes = 8192
    if model_spec.variant.startswith("4b_") or model_spec.variant == "4b_it":
        batch_size = min(batch_size, 4)
        max_feature_nodes = min(max_feature_nodes, 4096)

    group_a_tokens = tuple(str(token) for token in concept_pair_payload.get("group_a_tokens", ()))
    group_b_tokens = tuple(str(token) for token in concept_pair_payload.get("group_b_tokens", ()))
    direction_mode = "single_group" if not group_b_tokens else str(analysis_payload["concept_direction_mode"])
    direct_projection_payload = cast(dict[str, Any], analysis_payload.get("direct_projection", {}))
    intervention_payloads = cast(dict[str, Any], direct_projection_payload.get("interventions", {}))
    intervention_scale_factor = float(analysis_payload["default_scale_factor"])
    store_concept_cache_key = next(iter(intervention_payloads), "unembed.hook_in")
    for intervention_payload in intervention_payloads.values():
        if isinstance(intervention_payload, dict) and "scale_factor" in intervention_payload:
            intervention_scale_factor = float(intervention_payload["scale_factor"])
            break
    return Gemma3ConceptDirectionParityCase(
        experiment_name=experiment_name,
        config_name=config_name,
        calibration_surface=calibration_surface,
        parity_artifact_name=parity_artifact_name,
        reference_artifact_name=reference_artifact_name,
        notebook_pipeline_artifact_name=cast(str | None, analysis_payload.get("debug_pipeline_state_artifact_name")),
        session_name=session_name,
        prompt=str(prompt_payload["text"]),
        prompt_render_mode=str(prompt_payload["render_mode"]),
        target_tokens=(target_tokens[0], target_tokens[1]),
        key_tokens=tuple(str(token) for token in prompt_payload["key_tokens"]),
        model_name=model_spec.model_name,
        transcoder_set=model_spec.transcoder_set,
        neuronpedia_model=model_spec.neuronpedia_model,
        neuronpedia_set=model_spec.neuronpedia_set,
        max_feature_nodes=max_feature_nodes,
        batch_size=batch_size,
        intervention_scale_factor=intervention_scale_factor,
        intervention_max_influence_norm_scale=bool(
            cast(Mapping[str, Any], analysis_payload.get("feature_intervention", {})).get(
                "max_influence_norm_scale",
                analysis_payload.get("intervention_max_influence_norm_scale", False),
            )
        ),
        intervention_sign_aware_scale=bool(
            cast(Mapping[str, Any], analysis_payload.get("feature_intervention", {})).get(
                "sign_aware_scale",
                analysis_payload.get("intervention_sign_aware_scale", True),
            )
        ),
        top_n=int(analysis_payload.get("top_n", 10)),
        concept_direction_mode=direction_mode,
        group_a_tokens=group_a_tokens,
        group_b_tokens=group_b_tokens,
        group_a_entities=tuple(
            (str(name), str(label)) for name, label in concept_pair_payload.get("group_a_entities", ())
        ),
        group_b_entities=tuple(
            (str(name), str(label)) for name, label in concept_pair_payload.get("group_b_entities", ())
        ),
        group_a_name=str(concept_pair_payload["group_a_name"]),
        group_b_name=str(concept_pair_payload["group_b_name"]),
        concept_label=str(concept_pair_payload["concept_label"]),
        classification_question=str(concept_pair_payload["classification_question"]),
        store_concept_cache_key=store_concept_cache_key,
        context_enhanced_scale=float(analysis_payload["context_enhanced_scale"]),
        use_answer_state_as_basis=bool(analysis_payload.get("use_answer_state_as_basis", False)),
        constrained_feature_selection_refs=analysis_payload.get("constrained_feature_selection"),
        require_cross_path_feature_overlap=require_cross_path_feature_overlap,
        require_gap_improvement=require_gap_improvement,
    )


def _load_gemma3_it_concept_direction_parity_case(
    *,
    concept_pair_config_name: str,
) -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_experiment_concept_direction_parity_case(
        config_name="gemma3_1b_it_local_color_fruit_orange_155_4973.yaml",
        calibration_surface="orange",
        parity_artifact_name="gemma3_1b_it_orange_155_4973",
        reference_artifact_name="gemma3_1b_it_orange_155_4973_reference_graph_sanity",
        session_name="ct_gemma3_1b_it_orange_155_4973",
        require_cross_path_feature_overlap=False,
        require_gap_improvement=False,
    )


def _load_gemma3_1b_it_bat_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_experiment_concept_direction_parity_case(
        config_name="gemma3_1b_it_local_bird_mammal_bat.yaml",
        calibration_surface="bat",
        parity_artifact_name="gemma3_1b_it_bat",
        reference_artifact_name="gemma3_1b_it_bat_reference_graph_sanity",
        session_name="ct_gemma3_1b_it_bat",
    )


def _load_gemma3_it_experiment_concept_direction_parity_case(
    *,
    config_name: str,
    calibration_surface: str,
    parity_artifact_name: str,
    reference_artifact_name: str,
    session_name: str,
    require_cross_path_feature_overlap: bool = True,
    require_gap_improvement: bool = True,
) -> Gemma3ConceptDirectionParityCase:
    config_path = _resolve_parity_config_path(config_name)
    payload = load_experiment_config(config_path)
    experiment_payload = cast(dict[str, Any], payload["EXPERIMENT"])
    concept_pair_path = _resolve_parity_config_path(str(experiment_payload["concept_pair_config_path"]))
    return _build_gemma3_it_concept_direction_parity_case(
        config_path=config_path,
        concept_pair_path=concept_pair_path,
        calibration_surface=calibration_surface,
        parity_artifact_name=parity_artifact_name,
        reference_artifact_name=reference_artifact_name,
        session_name=session_name,
        require_cross_path_feature_overlap=require_cross_path_feature_overlap,
        require_gap_improvement=require_gap_improvement,
    )


def _load_gemma3_1b_it_orange_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_experiment_concept_direction_parity_case(
        config_name="gemma3_1b_it_local_color_fruit_orange.yaml",
        calibration_surface="orange",
        parity_artifact_name="gemma3_1b_it_orange",
        reference_artifact_name="gemma3_1b_it_orange_reference_graph_sanity",
        session_name="ct_gemma3_1b_it_orange",
        require_cross_path_feature_overlap=False,
        require_gap_improvement=False,
    )


def _load_gemma3_4b_it_ohio_2975_15708_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_experiment_concept_direction_parity_case(
        config_name="gemma3_4b_it_local_oqi_reasoning_oh_2975_15708.yaml",
        calibration_surface="ohio",
        parity_artifact_name="gemma3_4b_it_ohio_fs_2975_15708",
        reference_artifact_name="gemma3_4b_it_ohio_fs_2975_15708_reference_graph_sanity",
        session_name="ct_gemma3_4b_it_ohio_fs_2975_15708",
    )


def _load_gemma3_1b_it_orange_4973_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_experiment_concept_direction_parity_case(
        config_name="gemma3_1b_it_local_color_fruit_orange_4973.yaml",
        calibration_surface="orange",
        parity_artifact_name="gemma3_1b_it_orange_4973",
        reference_artifact_name="gemma3_1b_it_orange_4973_reference_graph_sanity",
        session_name="ct_gemma3_1b_it_orange_4973",
        require_cross_path_feature_overlap=False,
        require_gap_improvement=False,
    )


def _load_gemma3_1b_it_orange_fs_l10_n5_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_experiment_concept_direction_parity_case(
        config_name="gemma3_1b_it_local_color_fruit_orange_fs_l10_n5.yaml",
        calibration_surface="orange",
        parity_artifact_name="gemma3_1b_it_orange_fs_l10_n5",
        reference_artifact_name="gemma3_1b_it_orange_fs_l10_n5_reference_graph_sanity",
        session_name="ct_gemma3_1b_it_orange_fs_l10_n5",
        require_cross_path_feature_overlap=False,
        require_gap_improvement=False,
    )


def _load_gemma3_1b_it_orange_fs_l10_n5_s5_any_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_experiment_concept_direction_parity_case(
        config_name="gemma3_1b_it_local_color_fruit_orange_fs_l10_n5_s5_any.yaml",
        calibration_surface="orange",
        parity_artifact_name="gemma3_1b_it_orange_fs_l10_n5_s5_any",
        reference_artifact_name="gemma3_1b_it_orange_fs_l10_n5_s5_any_reference_graph_sanity",
        session_name="ct_gemma3_1b_it_orange_fs_l10_n5_s5_any",
        require_cross_path_feature_overlap=False,
        require_gap_improvement=True,
    )


@pytest.fixture(
    params=[
        pytest.param("cp_color_fruit_orange_gemma_it.yaml", id="orange_155_4973"),
    ]
)
def gemma3_it_concept_direction_parity_case(request) -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_it_concept_direction_parity_case(concept_pair_config_name=cast(str, request.param))


@pytest.fixture
def gemma3_4b_it_ohio_2975_15708_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_4b_it_ohio_2975_15708_concept_direction_parity_case()


@pytest.fixture
def gemma3_1b_it_orange_4973_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_1b_it_orange_4973_concept_direction_parity_case()


@pytest.fixture
def gemma3_1b_it_bat_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_1b_it_bat_concept_direction_parity_case()


@pytest.fixture
def gemma3_1b_it_orange_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_1b_it_orange_concept_direction_parity_case()


@pytest.fixture(
    params=[
        pytest.param(_load_gemma3_1b_it_orange_concept_direction_parity_case, id="orange"),
        pytest.param(_load_gemma3_1b_it_orange_fs_l10_n5_concept_direction_parity_case, id="orange_fs_l10_n5"),
    ]
)
def gemma3_1b_it_extended_concept_direction_parity_case(request) -> Gemma3ConceptDirectionParityCase:
    loader = cast(Any, request.param)
    return cast(Gemma3ConceptDirectionParityCase, loader())


@pytest.fixture(
    params=[
        pytest.param(_load_gemma3_1b_it_orange_fs_l10_n5_concept_direction_parity_case, id="orange_fs_l10_n5"),
    ]
)
def gemma3_1b_it_fs_l10_n5_concept_direction_parity_case(request) -> Gemma3ConceptDirectionParityCase:
    loader = cast(Any, request.param)
    return cast(Gemma3ConceptDirectionParityCase, loader())


@pytest.fixture
def gemma3_1b_it_orange_fs_l10_n5_s5_any_concept_direction_parity_case() -> Gemma3ConceptDirectionParityCase:
    return _load_gemma3_1b_it_orange_fs_l10_n5_s5_any_concept_direction_parity_case()


@pytest.fixture
def ct_nnsight_gemma3_case_session_factory(tmp_path):
    @contextmanager
    def _factory(case: Gemma3ConceptDirectionParityCase, run_name: str):
        with experiment_session(
            tmp_path,
            run_name,
            model_family="gemma3",
            model_variant=_resolve_case_model_variant(case),
            model_name=case.model_name,
            transcoder_set=case.transcoder_set,
            force_device="cpu",
            batch_size=case.batch_size,
            max_feature_nodes=case.max_feature_nodes,
        ) as (it_session, _module, _tokenizer):
            yield it_session

    return _factory


def _configure_debug_op_settings(module: Any, case: Gemma3ConceptDirectionParityCase) -> None:
    cfg = module.circuit_tracer_cfg
    cfg.model_name = case.model_name
    cfg.transcoder_set = case.transcoder_set
    cfg.dtype = torch.float32
    cfg.analysis_target_tokens = None
    cfg.target_token_ids = None
    cfg.max_feature_nodes = case.max_feature_nodes
    cfg.offload = "cpu"
    cfg.verbose = False
    cfg.batch_size = case.batch_size
    cfg.max_n_logits = 10
    cfg.desired_logit_prob = 0.95
    cfg.intervention_value_source = "top_feature_activation_values"
    cfg.intervention_scale_factor = case.intervention_scale_factor
    cfg.intervention_constrained_layers = list(range(module.replacement_model.cfg.n_layers))
    cfg.intervention_apply_activation_function = False
    cfg.intervention_freeze_attention = None
    cfg.intervention_sparse = False
    cfg.intervention_return_activations = False


def _build_case_feature_selection_spec(
    case: Gemma3ConceptDirectionParityCase,
    active_features: Any,
) -> Any | None:
    if case.constrained_feature_selection_refs is None:
        return None
    return _build_feature_selection_spec(_build_case_feature_selection_context(case), active_features)


def _build_perturbed_concept_direction_artifacts(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
    *,
    stage_result: ConceptDirectionStageResult,
    path_label: str,
    seed: int,
) -> tuple[ConceptDirectionParityArtifacts, dict[str, Any]]:
    perturbed_direction, perturbation_metadata = build_random_vector_perturbation(
        stage_result.concept_direction,
        scale=DEFAULT_RANDOM_PERTURBATION_SCALE,
        seed=seed,
    )
    perturbed_artifacts = _build_concept_direction_graph_artifacts(
        module,
        case,
        path_label=path_label,
        concept_direction=perturbed_direction,
        prompt_alignment_snapshots=stage_result.prompt_alignment_snapshots,
        extraction_snapshots=stage_result.extraction_snapshots,
        group_projection_states=stage_result.group_projection_states,
        group_ids=stage_result.group_ids,
    )
    return perturbed_artifacts, perturbation_metadata


def _build_graph_perturbation_control(
    *,
    base_artifacts: ConceptDirectionParityArtifacts,
    perturbed_artifacts: ConceptDirectionParityArtifacts,
    perturbation_metadata: dict[str, Any],
    base_label: str,
    perturbed_label: str,
) -> dict[str, Any]:
    feature_parity = compare_top_feature_sets(
        base_artifacts.top_feature_ids,
        perturbed_artifacts.top_feature_ids,
        left_scores=base_artifacts.top_feature_scores,
        right_scores=perturbed_artifacts.top_feature_scores,
        left_label=base_label,
        right_label=perturbed_label,
    )
    base_gap_delta = base_artifacts.post_gap - base_artifacts.pre_gap
    perturbed_gap_delta = perturbed_artifacts.post_gap - perturbed_artifacts.pre_gap
    return {
        "seed": int(perturbation_metadata["seed"]),
        "perturbation_scale": float(perturbation_metadata["scale"]),
        "direction_cosine_to_base": float(perturbation_metadata["cosine_to_base"]),
        "sampled_random_direction_cosine_to_base": float(
            perturbation_metadata["sampled_random_direction_cosine_to_base"]
        ),
        "perturbation_basis_cosine_to_base": float(perturbation_metadata["perturbation_basis_cosine_to_base"]),
        "feature_jaccard_vs_base": feature_parity.jaccard,
        "shared_score_cosine_vs_base": feature_parity.shared_score_cosine,
        "gap_delta_vs_base": float(perturbed_gap_delta - base_gap_delta),
        "feature_comparison": feature_parity.to_dict(),
        "perturbation_metadata": perturbation_metadata,
    }


def _build_reference_graph_target_artifact(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
    *,
    target_label: str,
    concept_direction: torch.Tensor,
    perturbation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = module.replacement_model
    tokenizer = model.tokenizer
    rendered_prompt = _render_debug_prompt(tokenizer, case.prompt, case.prompt_render_mode)
    graph_call_kwargs = _build_concept_direction_graph_call_kwargs(
        module,
        case,
        rendered_prompt=rendered_prompt,
        concept_direction=concept_direction,
    )
    serialized_graph_call_kwargs = _serialize_intervention_call_kwargs(
        {
            **graph_call_kwargs,
            "attribution_targets": _serialize_attribution_targets(graph_call_kwargs.get("attribution_targets")),
        }
    )
    graph = attribute(rendered_prompt, model, **graph_call_kwargs)
    analysis_backend = require_analysis_backend(module)
    node_influence_scores, _node_feature_rows = analysis_backend.compute_node_influence_scores(graph)
    signed_score_fn = getattr(analysis_backend, "compute_signed_node_influence_scores", None)
    _baseline_logits, activation_cache = module.replacement_model.get_activations(
        rendered_prompt,
        apply_activation_function=False,
    )
    active_feature_rows = graph.active_features.detach().cpu().reshape(-1, 3)
    activation_cache = torch.as_tensor(activation_cache, dtype=torch.float32).detach().cpu()
    activation_values = activation_cache[
        active_feature_rows[:, 0],
        active_feature_rows[:, 1],
        active_feature_rows[:, 2],
    ]
    feature_selection_inputs: dict[str, Any] = {
        "active_features": graph.active_features.detach().cpu(),
        "selected_features": graph.selected_features.detach().cpu(),
        "node_influence_scores": node_influence_scores,
        "activation_values": activation_values,
    }
    if callable(signed_score_fn):
        feature_selection_inputs["node_signed_influence_scores"] = (
            torch.as_tensor(signed_score_fn(graph), dtype=torch.float32).detach().cpu()
        )
    top_features_result, applied_feature_rows = _extract_top_features_with_optional_filter(
        module,
        _build_case_feature_selection_context(case),
        feature_selection_inputs,
        top_n=case.top_n,
    )
    top_feature_rows = [
        tuple(int(value) for value in row.tolist())
        for row in torch.as_tensor(top_features_result.top_feature_ids, dtype=torch.long).reshape(-1, 3)
    ]
    top_feature_scores = torch.as_tensor(top_features_result.top_feature_scores, dtype=torch.float32).reshape(-1)
    logit_probabilities = torch.as_tensor(graph.logit_probabilities, dtype=torch.float32).cpu().reshape(-1).tolist()
    serialized_targets = _serialize_attribution_targets(graph_call_kwargs.get("attribution_targets"))
    target_summary = serialized_targets[0] if serialized_targets else {}
    return {
        "path_label": target_label,
        "graph_call_kwargs": serialized_graph_call_kwargs,
        "attribution_targets": serialized_targets,
        "top_n": int(case.top_n),
        "requested_feature_selection": _serialize_constrained_feature_selection(
            case.constrained_feature_selection_refs
        ),
        "applied_feature_selection_rows": [list(row) for row in applied_feature_rows],
        "graph_input_tokens": _summarize_graph_input_tokens(
            tokenizer,
            rendered_prompt,
            cast(PromptRenderMode, case.prompt_render_mode),
            graph.input_tokens,
        ),
        "graph_result_input_tokens": torch.as_tensor(graph.input_tokens, dtype=torch.long).cpu().reshape(-1).tolist(),
        "selected_feature_count": int(len(graph.selected_features)),
        "active_feature_count": int(graph.active_features.shape[0]),
        "top_feature_ids": [list(row) for row in top_feature_rows],
        "top_feature_scores": [float(value) for value in top_feature_scores.tolist()],
        "logit_targets": [
            {
                "token_str": target.token_str,
                "vocab_idx": int(target.vocab_idx),
                "prob": float(logit_probabilities[index]) if index < len(logit_probabilities) else None,
            }
            for index, target in enumerate(graph.logit_targets)
        ],
        "direction_fingerprint": tensor_fingerprint(concept_direction),
        "direction_summary": {
            "norm": float(torch.linalg.vector_norm(torch.as_tensor(concept_direction, dtype=torch.float32)).item()),
            "target_tuple_token": target_summary.get("token_str"),
            "target_tuple_prob": target_summary.get("prob"),
        },
        "perturbation_metadata": perturbation_metadata,
    }


def _reference_graph_feature_comparison(
    report: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_payload = cast(dict[str, Any], report[left_label])
    right_payload = cast(dict[str, Any], report[right_label])
    return compare_top_feature_sets(
        cast(list[list[int]], left_payload.get("top_feature_ids", [])),
        cast(list[list[int]], right_payload.get("top_feature_ids", [])),
        left_scores=left_payload.get("top_feature_scores"),
        right_scores=right_payload.get("top_feature_scores"),
        left_label=left_label,
        right_label=right_label,
    ).to_dict()


def _build_reference_graph_sanity_report(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
    *,
    calibration_surface: str,
) -> dict[str, Any]:
    embed_stage = _compute_embed_concept_direction_stage(module, case)
    store_plain_stage = _compute_store_concept_direction_stage(module, case, context_enhanced=False)
    store_context_stage = _compute_store_concept_direction_stage(module, case, context_enhanced=True)

    reference_report: dict[str, Any] = {
        "report_kind": "reference_graph_sanity",
        "calibration_surface": calibration_surface,
        "mode": case.concept_direction_mode,
        "prompt": _render_debug_prompt(
            module.replacement_model.tokenizer,
            case.prompt,
            case.prompt_render_mode,
        ),
        "score_kind": "node_influence",
        "embed": _build_reference_graph_target_artifact(
            module,
            case,
            target_label="embed",
            concept_direction=embed_stage.concept_direction,
        ),
        "store_plain": _build_reference_graph_target_artifact(
            module,
            case,
            target_label="store_plain",
            concept_direction=store_plain_stage.concept_direction,
        ),
        "store_context": _build_reference_graph_target_artifact(
            module,
            case,
            target_label="store_context",
            concept_direction=store_context_stage.concept_direction,
        ),
    }

    perturbation_specs = (
        ("embed", embed_stage, "embed_random_perturbed", DEFAULT_RANDOM_PERTURBATION_SEED),
        ("store_plain", store_plain_stage, "store_plain_random_perturbed", DEFAULT_RANDOM_PERTURBATION_SEED + 1),
        (
            "store_context",
            store_context_stage,
            "store_context_random_perturbed",
            DEFAULT_RANDOM_PERTURBATION_SEED + 2,
        ),
    )
    direction_cosines = {
        "embed_vs_store_plain": cosine_similarity_value(
            embed_stage.concept_direction,
            store_plain_stage.concept_direction,
        ),
        "embed_vs_store_context": cosine_similarity_value(
            embed_stage.concept_direction,
            store_context_stage.concept_direction,
        ),
        "store_plain_vs_store_context": cosine_similarity_value(
            store_plain_stage.concept_direction,
            store_context_stage.concept_direction,
        ),
    }

    for base_label, stage_result, perturbed_label, seed in perturbation_specs:
        perturbed_direction, perturbation_metadata = build_random_vector_perturbation(
            stage_result.concept_direction,
            scale=DEFAULT_RANDOM_PERTURBATION_SCALE,
            seed=seed,
        )
        reference_report[perturbed_label] = _build_reference_graph_target_artifact(
            module,
            case,
            target_label=perturbed_label,
            concept_direction=perturbed_direction,
            perturbation_metadata=perturbation_metadata,
        )
        direction_cosines[f"{base_label}_vs_{perturbed_label}"] = float(perturbation_metadata["cosine_to_base"])

    reference_report["direction_cosines"] = direction_cosines
    reference_report["comparisons"] = {
        "embed_vs_store_plain": _reference_graph_feature_comparison(
            reference_report,
            left_label="embed",
            right_label="store_plain",
        ),
        "embed_vs_store_context": _reference_graph_feature_comparison(
            reference_report,
            left_label="embed",
            right_label="store_context",
        ),
        "store_plain_vs_store_context": _reference_graph_feature_comparison(
            reference_report,
            left_label="store_plain",
            right_label="store_context",
        ),
        "embed_vs_embed_random_perturbed": _reference_graph_feature_comparison(
            reference_report,
            left_label="embed",
            right_label="embed_random_perturbed",
        ),
        "store_plain_vs_store_plain_random_perturbed": _reference_graph_feature_comparison(
            reference_report,
            left_label="store_plain",
            right_label="store_plain_random_perturbed",
        ),
        "store_context_vs_store_context_random_perturbed": _reference_graph_feature_comparison(
            reference_report,
            left_label="store_context",
            right_label="store_context_random_perturbed",
        ),
    }
    return reference_report


def _build_debug_graph(
    module: Any,
    rendered_prompt: str,
    target_ids_tensor: torch.Tensor,
) -> tuple[Any, Graph, Any, dict[str, Any]]:
    graph_result = cast(
        Any,
        it.compute_attribution_graph(
            module,
            AnalysisBatch(prompts=[rendered_prompt], logit_target_ids=target_ids_tensor),
            batch=BatchEncoding(data={}),
            batch_idx=0,
            attribution_targets=target_ids_tensor.to(module.replacement_model.device),
        ),
    )

    assert torch.equal(
        torch.as_tensor(graph_result.logit_target_ids, dtype=torch.long).cpu(),
        target_ids_tensor.cpu(),
    ), "Graph target ids diverged from the requested debug ids"

    analysis_backend = require_analysis_backend(module)
    settings = analysis_backend.resolve_feature_intervention_settings(module)
    assert settings["constrained_layers"] == list(range(module.replacement_model.cfg.n_layers))
    assert settings["apply_activation_function"] is False

    graph = analysis_backend.hydrate_graph_from_batch(graph_result)
    graph_context = _build_graph_edge_validation_context(
        module.replacement_model,
        graph,
        selected_only=True,
    )
    return graph_result, graph, graph_context, settings


def _build_single_feature_top_features_result(
    graph_context: Any,
    feature_row: tuple[int, int, int],
) -> AnalysisBatch:
    layer, position, feature_id = feature_row
    activation_value = float(graph_context.activation_cache[layer, position, feature_id].item())
    return AnalysisBatch(
        top_feature_ids=torch.tensor([feature_row], dtype=torch.long),
        top_feature_scores=torch.tensor([0.0], dtype=torch.float32),
        top_feature_activation_values=torch.tensor([activation_value], dtype=torch.float32),
    )


def _maybe_preserve_debug_intervention_artifacts(
    *,
    constrained_feature_ref: str,
    case: Gemma3ConceptDirectionParityCase,
    rendered_prompt: str,
    graph: Graph,
    feature_row: tuple[int, int, int],
    interventions: list[tuple[int, int, int, float]],
    baseline_activation_cache: torch.Tensor,
    intervention_activation_cache: torch.Tensor,
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    graph_target_ids: list[int],
    report: Any,
    runtime_state: dict[str, Any] | None = None,
) -> Path | None:
    artifact_name = constrained_feature_ref.replace("/", "__").replace("-", "_")
    artifact_dir = resolve_artifact_output_dir(artifact_name=f"{artifact_name}")
    if artifact_dir is None:
        return None

    metadata = {
        "constrained_feature_ref": constrained_feature_ref,
        "feature_row": list(feature_row),
        "prompt_render_mode": case.prompt_render_mode,
        "requested_key_tokens": list(case.key_tokens),
        "graph_target_ids": list(graph_target_ids),
        "graph_target_tokens": [target.token_str for target in graph.logit_targets],
        "rendered_prompt": rendered_prompt,
        "model_name": case.model_name,
        "transcoder_set": case.transcoder_set,
        "runtime_state": runtime_state or {},
    }
    return save_preserved_intervention_artifacts(
        artifact_dir,
        graph=graph,
        feature_row=feature_row,
        interventions=interventions,
        baseline_activation_cache=baseline_activation_cache,
        intervention_activation_cache=intervention_activation_cache,
        baseline_logits=baseline_logits,
        intervention_logits=intervention_logits,
        activation_atol=FEATURE_EDGE_ACT_ATOL,
        activation_rtol=FEATURE_EDGE_ACT_RTOL,
        logit_atol=EDGE_LOGIT_ATOL,
        logit_rtol=EDGE_LOGIT_RTOL,
        report=report,
        metadata=metadata,
    )


def _assert_gemma3_1b_it_concept_direction_paths(
    ct_nnsight_gemma3_case_session_factory,
    case: Gemma3ConceptDirectionParityCase,
) -> None:
    with ct_nnsight_gemma3_case_session_factory(case, case.session_name) as it_session:
        module = cast(Any, it_session.module)
        _configure_gemma3_1b_concept_direction_parity_settings(module, case)
        _ensure_analysis_cfg(module, it.compute_attribution_graph)

        with clean_cuda(module.replacement_model):
            embed_stage = _compute_embed_concept_direction_stage(module, case)
            store_plain_stage = _compute_store_concept_direction_stage(
                module,
                case,
                context_enhanced=False,
            )
            store_context_stage = _compute_store_concept_direction_stage(
                module,
                case,
                context_enhanced=True,
            )
            embed_artifacts = _build_concept_direction_graph_artifacts(
                module,
                case,
                path_label=embed_stage.path_label,
                concept_direction=embed_stage.concept_direction,
                group_projection_states=embed_stage.group_projection_states,
                group_ids=embed_stage.group_ids,
                direction_stage_artifact=embed_stage.direction_stage_artifact,
            )
            store_plain_artifacts = _build_concept_direction_graph_artifacts(
                module,
                case,
                path_label=store_plain_stage.path_label,
                concept_direction=store_plain_stage.concept_direction,
                prompt_alignment_snapshots=store_plain_stage.prompt_alignment_snapshots,
                extraction_snapshots=store_plain_stage.extraction_snapshots,
                group_projection_states=store_plain_stage.group_projection_states,
                group_ids=store_plain_stage.group_ids,
                direction_stage_artifact=store_plain_stage.direction_stage_artifact,
            )
            store_context_artifacts = _build_concept_direction_graph_artifacts(
                module,
                case,
                path_label=store_context_stage.path_label,
                concept_direction=store_context_stage.concept_direction,
                prompt_alignment_snapshots=store_context_stage.prompt_alignment_snapshots,
                extraction_snapshots=store_context_stage.extraction_snapshots,
                group_projection_states=store_context_stage.group_projection_states,
                group_ids=store_context_stage.group_ids,
                direction_stage_artifact=store_context_stage.direction_stage_artifact,
            )
            embed_perturbed_artifacts, embed_perturbation_metadata = _build_perturbed_concept_direction_artifacts(
                module,
                case,
                stage_result=embed_stage,
                path_label="embed_random_perturbed",
                seed=DEFAULT_RANDOM_PERTURBATION_SEED,
            )
            (
                store_plain_perturbed_artifacts,
                store_plain_perturbation_metadata,
            ) = _build_perturbed_concept_direction_artifacts(
                module,
                case,
                stage_result=store_plain_stage,
                path_label="store_plain_random_perturbed",
                seed=DEFAULT_RANDOM_PERTURBATION_SEED + 1,
            )
            (
                store_context_perturbed_artifacts,
                store_context_perturbation_metadata,
            ) = _build_perturbed_concept_direction_artifacts(
                module,
                case,
                stage_result=store_context_stage,
                path_label="store_context_random_perturbed",
                seed=DEFAULT_RANDOM_PERTURBATION_SEED + 2,
            )
            reference_payloads = {
                "embed": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="embed",
                    concept_direction=embed_stage.concept_direction,
                ),
                "store_plain": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="store_plain",
                    concept_direction=store_plain_stage.concept_direction,
                ),
                "store_context": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="store_context",
                    concept_direction=store_context_stage.concept_direction,
                ),
                "embed_random_perturbed": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="embed_random_perturbed",
                    concept_direction=embed_perturbed_artifacts.concept_direction,
                    perturbation_metadata=embed_perturbation_metadata,
                ),
                "store_plain_random_perturbed": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="store_plain_random_perturbed",
                    concept_direction=store_plain_perturbed_artifacts.concept_direction,
                    perturbation_metadata=store_plain_perturbation_metadata,
                ),
                "store_context_random_perturbed": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="store_context_random_perturbed",
                    concept_direction=store_context_perturbed_artifacts.concept_direction,
                    perturbation_metadata=store_context_perturbation_metadata,
                ),
            }

    embed_vs_store_plain = compare_top_feature_sets(
        embed_artifacts.top_feature_ids,
        store_plain_artifacts.top_feature_ids,
        left_scores=embed_artifacts.top_feature_scores,
        right_scores=store_plain_artifacts.top_feature_scores,
        left_label="embed",
        right_label="store_plain",
    )
    embed_vs_store_context = compare_top_feature_sets(
        embed_artifacts.top_feature_ids,
        store_context_artifacts.top_feature_ids,
        left_scores=embed_artifacts.top_feature_scores,
        right_scores=store_context_artifacts.top_feature_scores,
        left_label="embed",
        right_label="store_context",
    )
    store_plain_vs_context = compare_top_feature_sets(
        store_plain_artifacts.top_feature_ids,
        store_context_artifacts.top_feature_ids,
        left_scores=store_plain_artifacts.top_feature_scores,
        right_scores=store_context_artifacts.top_feature_scores,
        left_label="store_plain",
        right_label="store_context",
    )
    direction_cosines = {
        "embed_vs_store_plain": float(
            torch.nn.functional.cosine_similarity(
                embed_artifacts.concept_direction.unsqueeze(0),
                store_plain_artifacts.concept_direction.unsqueeze(0),
            ).item()
        ),
        "embed_vs_store_context": float(
            torch.nn.functional.cosine_similarity(
                embed_artifacts.concept_direction.unsqueeze(0),
                store_context_artifacts.concept_direction.unsqueeze(0),
            ).item()
        ),
        "store_plain_vs_context": float(
            torch.nn.functional.cosine_similarity(
                store_plain_artifacts.concept_direction.unsqueeze(0),
                store_context_artifacts.concept_direction.unsqueeze(0),
            ).item()
        ),
        "embed_vs_embed_random_perturbed": float(embed_perturbation_metadata["cosine_to_base"]),
        "store_plain_vs_store_plain_random_perturbed": float(store_plain_perturbation_metadata["cosine_to_base"]),
        "store_context_vs_store_context_random_perturbed": float(store_context_perturbation_metadata["cosine_to_base"]),
    }
    perturbation_controls = {
        "embed": _build_graph_perturbation_control(
            base_artifacts=embed_artifacts,
            perturbed_artifacts=embed_perturbed_artifacts,
            perturbation_metadata=embed_perturbation_metadata,
            base_label="embed",
            perturbed_label="embed_random_perturbed",
        ),
        "store_plain": _build_graph_perturbation_control(
            base_artifacts=store_plain_artifacts,
            perturbed_artifacts=store_plain_perturbed_artifacts,
            perturbation_metadata=store_plain_perturbation_metadata,
            base_label="store_plain",
            perturbed_label="store_plain_random_perturbed",
        ),
        "store_context": _build_graph_perturbation_control(
            base_artifacts=store_context_artifacts,
            perturbed_artifacts=store_context_perturbed_artifacts,
            perturbation_metadata=store_context_perturbation_metadata,
            base_label="store_context",
            perturbed_label="store_context_random_perturbed",
        ),
    }
    report = {
        "report_kind": "concept_direction_parity",
        "experiment_name": case.experiment_name,
        "config_name": case.config_name,
        "calibration_surface": case.calibration_surface,
        "mode": case.concept_direction_mode,
        "top_n": int(case.top_n),
        "constrained_feature_selection": _serialize_constrained_feature_selection(
            case.constrained_feature_selection_refs
        ),
        "pipeline_state_artifact_file": "concept_direction_pipeline_state_artifacts.json",
        "reference_report_name": case.reference_artifact_name,
        "notebook_pipeline_artifact_name": case.notebook_pipeline_artifact_name,
        "direction_cosines": direction_cosines,
        "embed": {
            "top_feature_ids": [list(row) for row in embed_artifacts.top_feature_ids],
            "top_feature_scores": [float(value) for value in embed_artifacts.top_feature_scores],
            "pre_gap": embed_artifacts.pre_gap,
            "post_gap": embed_artifacts.post_gap,
            "group_a_projection_mean": embed_artifacts.group_a_projection_mean,
            "group_b_projection_mean": embed_artifacts.group_b_projection_mean,
        },
        "store_plain": {
            "top_feature_ids": [list(row) for row in store_plain_artifacts.top_feature_ids],
            "top_feature_scores": [float(value) for value in store_plain_artifacts.top_feature_scores],
            "pre_gap": store_plain_artifacts.pre_gap,
            "post_gap": store_plain_artifacts.post_gap,
            "group_a_projection_mean": store_plain_artifacts.group_a_projection_mean,
            "group_b_projection_mean": store_plain_artifacts.group_b_projection_mean,
            "prompt_alignment_count": len(store_plain_artifacts.prompt_alignment_snapshots),
            "prompt_alignment_snapshots": list(store_plain_artifacts.prompt_alignment_snapshots),
        },
        "store_context": {
            "top_feature_ids": [list(row) for row in store_context_artifacts.top_feature_ids],
            "top_feature_scores": [float(value) for value in store_context_artifacts.top_feature_scores],
            "pre_gap": store_context_artifacts.pre_gap,
            "post_gap": store_context_artifacts.post_gap,
            "group_a_projection_mean": store_context_artifacts.group_a_projection_mean,
            "group_b_projection_mean": store_context_artifacts.group_b_projection_mean,
            "prompt_alignment_count": len(store_context_artifacts.prompt_alignment_snapshots),
            "prompt_alignment_snapshots": list(store_context_artifacts.prompt_alignment_snapshots),
            "extraction_snapshot_count": len(store_context_artifacts.extraction_snapshots),
            "extraction_snapshots": list(store_context_artifacts.extraction_snapshots),
        },
        "comparisons": {
            "embed_vs_store_plain": embed_vs_store_plain.to_dict(),
            "embed_vs_store_context": embed_vs_store_context.to_dict(),
            "store_plain_vs_context": store_plain_vs_context.to_dict(),
        },
        "strict_feature_overlap_enforced": bool(case.require_cross_path_feature_overlap),
        "require_gap_improvement": bool(case.require_gap_improvement),
        "perturbation_controls": perturbation_controls,
    }
    _assert_reference_graph_payload_matches_direct_artifacts(
        embed_artifacts,
        cast(dict[str, Any], reference_payloads["embed"]),
        label="embed",
        report=report,
    )
    _assert_reference_graph_payload_matches_direct_artifacts(
        store_plain_artifacts,
        cast(dict[str, Any], reference_payloads["store_plain"]),
        label="store_plain",
        report=report,
    )
    _assert_reference_graph_payload_matches_direct_artifacts(
        store_context_artifacts,
        cast(dict[str, Any], reference_payloads["store_context"]),
        label="store_context",
        report=report,
    )
    _assert_reference_graph_payload_matches_direct_artifacts(
        embed_perturbed_artifacts,
        cast(dict[str, Any], reference_payloads["embed_random_perturbed"]),
        label="embed_random_perturbed",
        report=report,
    )
    _assert_reference_graph_payload_matches_direct_artifacts(
        store_plain_perturbed_artifacts,
        cast(dict[str, Any], reference_payloads["store_plain_random_perturbed"]),
        label="store_plain_random_perturbed",
        report=report,
    )
    _assert_reference_graph_payload_matches_direct_artifacts(
        store_context_perturbed_artifacts,
        cast(dict[str, Any], reference_payloads["store_context_random_perturbed"]),
        label="store_context_random_perturbed",
        report=report,
    )
    if os.environ.get(PRESERVE_ARTIFACTS_ENV) == "1":
        save_concept_direction_parity_report(
            report,
            artifact_name=case.parity_artifact_name,
        )
        save_concept_direction_pipeline_state_artifacts(
            {
                "experiment_name": case.experiment_name,
                "config_name": case.config_name,
                "calibration_surface": case.calibration_surface,
                "mode": case.concept_direction_mode,
                "top_n": int(case.top_n),
                "constrained_feature_selection": _serialize_constrained_feature_selection(
                    case.constrained_feature_selection_refs
                ),
                "embed": {
                    "direction": embed_artifacts.direction_stage_artifact,
                    "graph": embed_artifacts.graph_stage_artifact,
                },
                "store_plain": {
                    "direction": store_plain_artifacts.direction_stage_artifact,
                    "graph": store_plain_artifacts.graph_stage_artifact,
                },
                "store_context": {
                    "direction": store_context_artifacts.direction_stage_artifact,
                    "graph": store_context_artifacts.graph_stage_artifact,
                },
                "embed_random_perturbed": {
                    "graph": embed_perturbed_artifacts.graph_stage_artifact,
                    "perturbation_control": perturbation_controls["embed"],
                },
                "store_plain_random_perturbed": {
                    "graph": store_plain_perturbed_artifacts.graph_stage_artifact,
                    "perturbation_control": perturbation_controls["store_plain"],
                },
                "store_context_random_perturbed": {
                    "graph": store_context_perturbed_artifacts.graph_stage_artifact,
                    "perturbation_control": perturbation_controls["store_context"],
                },
                "graph_mode_comparisons": {
                    "embed": {
                        "base_vs_perturbed": perturbation_controls["embed"]["feature_comparison"],
                    },
                    "store_plain": {
                        "base_vs_perturbed": perturbation_controls["store_plain"]["feature_comparison"],
                    },
                    "store_context": {
                        "base_vs_perturbed": perturbation_controls["store_context"]["feature_comparison"],
                    },
                },
                "perturbation_controls": perturbation_controls,
            },
            artifact_name=case.parity_artifact_name,
        )

    for artifacts in (embed_artifacts, store_plain_artifacts, store_context_artifacts):
        assert torch.isclose(
            torch.linalg.vector_norm(artifacts.concept_direction),
            torch.tensor(1.0),
            atol=1e-4,
        ), json.dumps(report, indent=2, default=str)
        assert len(artifacts.top_feature_ids) > 0, json.dumps(report, indent=2, default=str)
        if case.concept_direction_mode == "single_group":
            assert artifacts.group_a_projection_mean > 0, json.dumps(report, indent=2, default=str)
        else:
            assert artifacts.group_b_projection_mean is not None
            assert artifacts.group_a_projection_mean > artifacts.group_b_projection_mean, json.dumps(
                report, indent=2, default=str
            )

    expected_example_count = len(case.group_a_entities) + len(case.group_b_entities)
    assert len(store_plain_artifacts.prompt_alignment_snapshots) == expected_example_count
    assert len(store_context_artifacts.prompt_alignment_snapshots) == expected_example_count
    assert len(store_context_artifacts.extraction_snapshots) == expected_example_count
    for snapshot_group in (
        store_plain_artifacts.prompt_alignment_snapshots,
        store_context_artifacts.prompt_alignment_snapshots,
    ):
        for snapshot in snapshot_group:
            assert snapshot["context_token_source"] == "probe_end", json.dumps(report, indent=2, default=str)
            assert snapshot["context_token_index"] == snapshot["probe_end_index"], json.dumps(
                report,
                indent=2,
                default=str,
            )
            assert snapshot["context_token_index"] < snapshot["answer_index"], json.dumps(report, indent=2, default=str)
    for extraction_snapshot in store_context_artifacts.extraction_snapshots:
        assert extraction_snapshot["context_source"] == "context_token_indices", json.dumps(
            report,
            indent=2,
            default=str,
        )

    for comparison in (embed_vs_store_plain, embed_vs_store_context, store_plain_vs_context):
        if case.require_cross_path_feature_overlap:
            assert comparison.shared_score_cosine is not None, json.dumps(report, indent=2, default=str)
            assert comparison.jaccard > 0.0, json.dumps(report, indent=2, default=str)
            assert comparison.shared_score_cosine > 0.0, json.dumps(report, indent=2, default=str)
        else:
            assert comparison.jaccard >= 0.0, json.dumps(report, indent=2, default=str)
            if comparison.shared_score_cosine is not None:
                assert -1.0 <= comparison.shared_score_cosine <= 1.0, json.dumps(
                    report,
                    indent=2,
                    default=str,
                )
    assert store_plain_artifacts.pre_gap == pytest.approx(embed_artifacts.pre_gap, abs=1e-6)
    assert store_context_artifacts.pre_gap == pytest.approx(embed_artifacts.pre_gap, abs=1e-6)
    for artifacts in (embed_artifacts, store_plain_artifacts, store_context_artifacts):
        if case.require_gap_improvement:
            assert artifacts.post_gap > artifacts.pre_gap, json.dumps(report, indent=2, default=str)
    assert direction_cosines["embed_vs_embed_random_perturbed"] < 0.2, json.dumps(report, indent=2, default=str)
    assert direction_cosines["store_plain_vs_store_plain_random_perturbed"] < 0.2, json.dumps(
        report,
        indent=2,
        default=str,
    )
    assert direction_cosines["store_context_vs_store_context_random_perturbed"] < 0.2, json.dumps(
        report,
        indent=2,
        default=str,
    )
    for control in perturbation_controls.values():
        feature_comparison = cast(dict[str, Any], control["feature_comparison"])
        shared_score_cosine = feature_comparison.get("shared_score_cosine")
        assert (
            float(feature_comparison["jaccard"]) < 0.999999
            or shared_score_cosine is None
            or float(shared_score_cosine) < 0.999999
        ), json.dumps(report, indent=2, default=str)
    if embed_vs_store_plain.shared_score_cosine is not None:
        assert embed_vs_store_plain.shared_score_cosine > 0, json.dumps(report, indent=2, default=str)
    if embed_vs_store_context.shared_score_cosine is not None:
        assert embed_vs_store_context.shared_score_cosine > 0, json.dumps(report, indent=2, default=str)


def _assert_gemma3_1b_it_reference_graph_sanity_case(
    ct_nnsight_gemma3_case_session_factory,
    case: Gemma3ConceptDirectionParityCase,
) -> None:
    with ct_nnsight_gemma3_case_session_factory(case, case.reference_artifact_name) as it_session:
        module = cast(Any, it_session.module)
        _configure_gemma3_1b_concept_direction_parity_settings(module, case)

        with clean_cuda(module.replacement_model):
            report = _build_reference_graph_sanity_report(
                module,
                case,
                calibration_surface=case.calibration_surface,
            )

    report["experiment_name"] = case.experiment_name
    report["config_name"] = case.config_name
    report["top_n"] = int(case.top_n)
    report["constrained_feature_selection"] = _serialize_constrained_feature_selection(
        case.constrained_feature_selection_refs
    )

    if os.environ.get(PRESERVE_ARTIFACTS_ENV) == "1":
        save_concept_direction_reference_graph_report(
            report,
            artifact_name=case.reference_artifact_name,
        )

    for label in (
        "embed",
        "store_plain",
        "store_context",
        "embed_random_perturbed",
        "store_plain_random_perturbed",
        "store_context_random_perturbed",
    ):
        payload = cast(dict[str, Any], report[label])
        assert payload["top_feature_ids"], json.dumps(report, indent=2, default=str)
        assert payload["top_feature_scores"], json.dumps(report, indent=2, default=str)
    assert report["direction_cosines"]["embed_vs_embed_random_perturbed"] < 0.2, json.dumps(
        report,
        indent=2,
        default=str,
    )
    assert report["direction_cosines"]["store_plain_vs_store_plain_random_perturbed"] < 0.2, json.dumps(
        report,
        indent=2,
        default=str,
    )
    assert report["direction_cosines"]["store_context_vs_store_context_random_perturbed"] < 0.2, json.dumps(
        report,
        indent=2,
        default=str,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_1b_it_concept_direction_paths(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_it_concept_direction_parity_case,
):
    _assert_gemma3_1b_it_concept_direction_paths(
        ct_nnsight_gemma3_case_session_factory,
        gemma3_it_concept_direction_parity_case,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_1b_it_extended_concept_direction_paths(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_1b_it_extended_concept_direction_parity_case,
):
    _assert_gemma3_1b_it_concept_direction_paths(
        ct_nnsight_gemma3_case_session_factory,
        gemma3_1b_it_extended_concept_direction_parity_case,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_1b_it_bat_reference_graph_sanity(
    cleanup_cuda,
    ct_nnsight_gemma3_it_session_factory,
    gemma3_1b_it_bat_concept_direction_parity_case,
):
    """Use upstream graph construction to compare bat concept-direction targets at the graph layer only."""

    case = gemma3_1b_it_bat_concept_direction_parity_case
    with ct_nnsight_gemma3_it_session_factory("ct_gemma3_1b_it_bat_reference_graph_sanity") as it_session:
        module = cast(Any, it_session.module)
        _configure_gemma3_1b_concept_direction_parity_settings(module, case)

        with clean_cuda(module.replacement_model):
            embed_stage = _compute_embed_concept_direction_stage(module, case)
            store_plain_stage = _compute_store_concept_direction_stage(module, case, context_enhanced=False)
            store_context_stage = _compute_store_concept_direction_stage(module, case, context_enhanced=True)
            embed_random_direction, embed_random_metadata = build_random_vector_perturbation(
                embed_stage.concept_direction,
                scale=DEFAULT_RANDOM_PERTURBATION_SCALE,
                seed=DEFAULT_RANDOM_PERTURBATION_SEED,
            )
            store_random_direction, store_random_metadata = build_random_vector_perturbation(
                store_context_stage.concept_direction,
                scale=DEFAULT_RANDOM_PERTURBATION_SCALE,
                seed=DEFAULT_RANDOM_PERTURBATION_SEED + 1,
            )

            report = {
                "report_kind": "reference_graph_sanity",
                "calibration_surface": "bat",
                "mode": case.concept_direction_mode,
                "prompt": _render_debug_prompt(
                    module.replacement_model.tokenizer,
                    case.prompt,
                    case.prompt_render_mode,
                ),
                "score_kind": "node_influence",
                "embed": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="embed",
                    concept_direction=embed_stage.concept_direction,
                ),
                "store_plain": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="store_plain",
                    concept_direction=store_plain_stage.concept_direction,
                ),
                "store_context": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="store_context",
                    concept_direction=store_context_stage.concept_direction,
                ),
                "embed_random_perturbed": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="embed_random_perturbed",
                    concept_direction=embed_random_direction,
                    perturbation_metadata=embed_random_metadata,
                ),
                "store_context_random_perturbed": _build_reference_graph_target_artifact(
                    module,
                    case,
                    target_label="store_context_random_perturbed",
                    concept_direction=store_random_direction,
                    perturbation_metadata=store_random_metadata,
                ),
            }

    report["direction_cosines"] = {
        "embed_vs_store_plain": cosine_similarity_value(
            embed_stage.concept_direction,
            store_plain_stage.concept_direction,
        ),
        "embed_vs_store_context": cosine_similarity_value(
            embed_stage.concept_direction,
            store_context_stage.concept_direction,
        ),
        "store_plain_vs_store_context": cosine_similarity_value(
            store_plain_stage.concept_direction,
            store_context_stage.concept_direction,
        ),
        "embed_vs_embed_random_perturbed": embed_random_metadata["cosine_to_base"],
        "store_context_vs_store_context_random_perturbed": store_random_metadata["cosine_to_base"],
    }
    report["comparisons"] = {
        "embed_vs_store_plain": _reference_graph_feature_comparison(
            report,
            left_label="embed",
            right_label="store_plain",
        ),
        "embed_vs_store_context": _reference_graph_feature_comparison(
            report,
            left_label="embed",
            right_label="store_context",
        ),
        "store_plain_vs_store_context": _reference_graph_feature_comparison(
            report,
            left_label="store_plain",
            right_label="store_context",
        ),
        "embed_vs_embed_random_perturbed": _reference_graph_feature_comparison(
            report,
            left_label="embed",
            right_label="embed_random_perturbed",
        ),
        "store_context_vs_store_context_random_perturbed": _reference_graph_feature_comparison(
            report,
            left_label="store_context",
            right_label="store_context_random_perturbed",
        ),
    }

    if os.environ.get(PRESERVE_ARTIFACTS_ENV) == "1":
        save_concept_direction_reference_graph_report(
            report,
            artifact_name="gemma3_1b_it_bat_reference_graph_sanity",
        )

    for label in (
        "embed",
        "store_plain",
        "store_context",
        "embed_random_perturbed",
        "store_context_random_perturbed",
    ):
        payload = cast(dict[str, Any], report[label])
        assert payload["top_feature_ids"], json.dumps(report, indent=2, default=str)
        assert payload["top_feature_scores"], json.dumps(report, indent=2, default=str)
        score_tensor = torch.tensor(payload["top_feature_scores"], dtype=torch.float32)
        assert torch.isfinite(score_tensor).all().item(), json.dumps(
            report,
            indent=2,
            default=str,
        )
        assert payload["direction_summary"]["norm"] == pytest.approx(1.0, abs=1e-5), json.dumps(
            report,
            indent=2,
            default=str,
        )

    assert report["direction_cosines"]["embed_vs_embed_random_perturbed"] < 0.2, json.dumps(
        report,
        indent=2,
        default=str,
    )
    assert report["direction_cosines"]["store_context_vs_store_context_random_perturbed"] < 0.2, json.dumps(
        report,
        indent=2,
        default=str,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_4b_it_ohio_reference_graph_sanity(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_4b_it_ohio_2975_15708_concept_direction_parity_case,
):
    _assert_gemma3_1b_it_reference_graph_sanity_case(
        ct_nnsight_gemma3_case_session_factory,
        gemma3_4b_it_ohio_2975_15708_concept_direction_parity_case,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_1b_it_orange_reference_graph_sanity(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_1b_it_orange_concept_direction_parity_case,
):
    _assert_gemma3_1b_it_reference_graph_sanity_case(
        ct_nnsight_gemma3_case_session_factory,
        gemma3_1b_it_orange_concept_direction_parity_case,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_1b_it_extended_reference_graph_sanity(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_1b_it_fs_l10_n5_concept_direction_parity_case,
):
    _assert_gemma3_1b_it_reference_graph_sanity_case(
        ct_nnsight_gemma3_case_session_factory,
        gemma3_1b_it_fs_l10_n5_concept_direction_parity_case,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
@pytest.mark.parametrize(
    ("constrained_feature_ref", "expected_feature_row"),
    [
        pytest.param(
            "gemma-3-1b-it/25-gemmascope-2-transcoder-16k/4973",
            (25, 27, 4973),
            id="feature_25_4973",
        ),
    ],
)
def test_analysis_backend_parity_feature_intervention_wrapper(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_1b_it_orange_4973_concept_direction_parity_case,
    constrained_feature_ref,
    expected_feature_row,
):
    case = gemma3_1b_it_orange_4973_concept_direction_parity_case
    with ct_nnsight_gemma3_case_session_factory(case, "ct_gemma3_1b_it_orange_4973_debug_op_session") as it_session:
        module = cast(Any, it_session.module)
        _configure_debug_op_settings(module, case)
        _ensure_analysis_cfg(module, it.compute_attribution_graph)

        with clean_cuda(module.replacement_model):
            tokenizer = module.replacement_model.tokenizer
            rendered_prompt = _render_debug_prompt(tokenizer, case.prompt, case.prompt_render_mode)
            graph_target_ids, _ = _resolve_debug_graph_target_ids(
                tokenizer,
                case.key_tokens,
                use_chat_template=case.prompt_render_mode != "plain",
            )
            target_ids_tensor = torch.tensor(graph_target_ids, dtype=torch.long)

            with module.replacement_model.zero_softcap():
                _, graph, graph_context, _ = _build_debug_graph(
                    module,
                    rendered_prompt,
                    target_ids_tensor,
                )
                _find_active_feature_index(graph_context.active_features, expected_feature_row)

                top_features_result = _build_single_feature_top_features_result(graph_context, expected_feature_row)
                summaries = _verify_wrapper_feature_interventions(
                    module,
                    rendered_prompt,
                    graph,
                    graph_context,
                    top_features_result,
                )

            assert len(summaries) == 1
            assert summaries[0].feature_row == expected_feature_row
            assert summaries[0].returned_activation_cache is True


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_feature_intervention_wrapper_sign_aware_top5_any_scaling(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_1b_it_orange_fs_l10_n5_s5_any_concept_direction_parity_case,
):
    case = gemma3_1b_it_orange_fs_l10_n5_s5_any_concept_direction_parity_case
    assert case.intervention_max_influence_norm_scale
    assert case.intervention_sign_aware_scale

    with ct_nnsight_gemma3_case_session_factory(case, case.session_name) as it_session:
        module = cast(Any, it_session.module)
        _configure_gemma3_1b_concept_direction_parity_settings(module, case)
        _ensure_analysis_cfg(module, it.compute_attribution_graph)

        with clean_cuda(module.replacement_model):
            embed_stage = _compute_embed_concept_direction_stage(module, case)
            artifacts = _build_concept_direction_graph_artifacts(
                module,
                case,
                path_label=embed_stage.path_label,
                concept_direction=embed_stage.concept_direction,
                group_projection_states=embed_stage.group_projection_states,
                group_ids=embed_stage.group_ids,
                direction_stage_artifact=embed_stage.direction_stage_artifact,
                validate_feature_edges=True,
            )

    graph_stage = cast(dict[str, Any], artifacts.graph_stage_artifact)
    feature_scores = torch.tensor(graph_stage["top_feature_scores"], dtype=torch.float32)
    activation_values = torch.tensor(graph_stage["top_feature_activation_values"], dtype=torch.float32)
    intervention_values = torch.tensor(graph_stage["intervention_values"], dtype=torch.float32)
    scale_factors = torch.tensor(graph_stage["intervention_scale_factors"], dtype=torch.float32)
    max_abs_score = feature_scores.abs().max().clamp_min(1e-12)
    expected_scale_factors = case.intervention_scale_factor * (feature_scores.abs() / max_abs_score)
    expected_values = feature_scores.sign() * activation_values.abs() * expected_scale_factors
    top_features = cast(list[list[int]], graph_stage["top_features"])
    wrapper_summaries = cast(list[dict[str, Any]], graph_stage["wrapper_feature_intervention_summaries"])

    assert len(feature_scores) == case.top_n
    assert torch.any(feature_scores > 0)
    assert torch.any(feature_scores < 0)
    assert torch.all(feature_scores.abs()[:-1] >= feature_scores.abs()[1:])
    assert_close(scale_factors, expected_scale_factors, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert_close(intervention_values, expected_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    nonzero_interventions = intervention_values != 0
    assert torch.all(
        torch.sign(intervention_values[nonzero_interventions]) == torch.sign(feature_scores[nonzero_interventions])
    )
    assert len(wrapper_summaries) == case.top_n
    for index, summary in enumerate(wrapper_summaries):
        assert summary["feature_row"] == top_features[index]
        assert summary["returned_activation_cache"] is True
        assert_close(
            torch.tensor([summary["intervention_value"]], dtype=torch.float32),
            intervention_values[index : index + 1],
            rtol=VALUE_RTOL,
            atol=VALUE_ATOL,
        )
    assert artifacts.post_gap > artifacts.pre_gap
    post_top_id = int(torch.argmax(artifacts.post_logits).item())
    first_target_variant_ids = {
        int(variant["token_id"]) for variant in cast(list[dict[str, Any]], graph_stage["target_token_variants"][0])
    }
    assert post_top_id in first_target_variant_ids, json.dumps(
        {
            "post_top_id": post_top_id,
            "target_ids": graph_stage["target_ids"],
            "target_token_variants": graph_stage["target_token_variants"],
            "pre_gap": artifacts.pre_gap,
            "post_gap": artifacts.post_gap,
            "top_features": top_features,
            "top_feature_scores": feature_scores.tolist(),
            "intervention_values": intervention_values.tolist(),
            "post_top_logits": graph_stage["post_top_logits"],
        },
        indent=2,
        default=str,
    )


@RUNIF(min_cuda_gpus=1, optional=True)
@pytest.mark.parametrize(
    ("constrained_feature_ref", "expected_feature_row"),
    [
        pytest.param(
            "gemma-3-1b-it/25-gemmascope-2-transcoder-16k/4973",
            (25, 27, 4973),
            id="feature_25_4973",
        ),
    ],
)
def test_analysis_backend_parity_selected_feature_adjacency_trace(
    cleanup_cuda,
    ct_nnsight_gemma3_case_session_factory,
    gemma3_1b_it_orange_4973_concept_direction_parity_case,
    constrained_feature_ref,
    expected_feature_row,
):
    case = gemma3_1b_it_orange_4973_concept_direction_parity_case
    with ct_nnsight_gemma3_case_session_factory(case, "ct_gemma3_1b_it_orange_4973_debug_trace_session") as it_session:
        module = cast(Any, it_session.module)
        _configure_debug_op_settings(module, case)
        _ensure_analysis_cfg(module, it.compute_attribution_graph)

        with clean_cuda(module.replacement_model):
            tokenizer = module.replacement_model.tokenizer
            rendered_prompt = _render_debug_prompt(tokenizer, case.prompt, case.prompt_render_mode)
            graph_target_ids, _ = _resolve_debug_graph_target_ids(
                tokenizer,
                case.key_tokens,
                use_chat_template=case.prompt_render_mode != "plain",
            )
            target_ids_tensor = torch.tensor(graph_target_ids, dtype=torch.long)

            with module.replacement_model.zero_softcap():
                graph_result, graph, graph_context, settings = _build_debug_graph(
                    module,
                    rendered_prompt,
                    target_ids_tensor,
                )
                top_features_result = _build_single_feature_top_features_result(graph_context, expected_feature_row)
                analysis_backend = require_analysis_backend(module)
                analysis_batch = AnalysisBatch(
                    prompts=[rendered_prompt],
                    top_feature_ids=top_features_result.top_feature_ids,
                    top_feature_scores=top_features_result.top_feature_scores,
                    top_feature_activation_values=top_features_result.top_feature_activation_values,
                    logit_target_ids=graph.logit_tokens.detach().cpu(),
                )
                interventions, _ = analysis_backend.build_feature_interventions(analysis_batch, settings)
                baseline_logits_raw, baseline_activation_cache = module.replacement_model.get_activations(
                    rendered_prompt,
                    apply_activation_function=False,
                )
                baseline_logits = (
                    last_token_logits(torch.as_tensor(baseline_logits_raw, dtype=torch.float32)).detach().cpu()
                )
                baseline_activation_cache = (
                    torch.as_tensor(
                        baseline_activation_cache,
                        dtype=torch.float32,
                    )
                    .detach()
                    .cpu()
                )
                wrapper_result = cast(
                    Any,
                    it.feature_intervention_forward(
                        module,
                        analysis_batch,
                        batch=cast(Any, None),
                        batch_idx=0,
                        prompt=rendered_prompt,
                        intervention_return_activations=True,
                    ),
                )
                assert getattr(wrapper_result, "intervention_activation_cache", None) is not None
                wrapper_post_logits = (
                    torch.as_tensor(
                        wrapper_result.post_intervention_logits,
                        dtype=torch.float32,
                    )
                    .detach()
                    .cpu()
                )
                wrapper_activation_cache = (
                    torch.as_tensor(
                        wrapper_result.intervention_activation_cache,
                        dtype=torch.float32,
                    )
                    .detach()
                    .cpu()
                )
                runtime_state = {
                    "module": snapshot_module_runtime_state(module),
                    "graph_op": {
                        "analysis_batch": snapshot_analysis_batch(
                            AnalysisBatch(prompts=[rendered_prompt], logit_target_ids=target_ids_tensor),
                            ("prompts", "logit_target_ids"),
                        ),
                        "call_kwargs": {
                            "attribution_targets": tensor_fingerprint(
                                target_ids_tensor.to(module.replacement_model.device)
                            )
                        },
                        "requested_graph_target_ids": [int(token_id) for token_id in graph_target_ids],
                        "result": {
                            "input_tokens": tensor_fingerprint(getattr(graph_result, "input_tokens", None)),
                            "active_features": tensor_fingerprint(getattr(graph_result, "active_features", None)),
                            "selected_features": tensor_fingerprint(getattr(graph_result, "selected_features", None)),
                            "selected_feature_rows": tensor_fingerprint(graph_context.active_features),
                            "adjacency_matrix": tensor_fingerprint(getattr(graph_result, "adjacency_matrix", None)),
                            "logit_target_ids": tensor_fingerprint(getattr(graph_result, "logit_target_ids", None)),
                            "logit_target_tokens": [target.token_str for target in graph.logit_targets],
                        },
                    },
                    "top_features_op": {
                        "result": snapshot_analysis_batch(
                            top_features_result,
                            ("top_feature_ids", "top_feature_scores", "top_feature_activation_values"),
                        )
                    },
                    "baseline_forward": {
                        "zero_softcap_enabled": True,
                        "apply_activation_function": False,
                        "graph_inputs": rendered_prompt,
                        "baseline_logits": tensor_fingerprint(baseline_logits),
                        "baseline_activation_cache": tensor_fingerprint(baseline_activation_cache),
                        "selected_feature_baseline_activation": float(
                            graph_context.activation_cache[
                                expected_feature_row[0], expected_feature_row[1], expected_feature_row[2]
                            ].item()
                        ),
                    },
                    "intervention_op": {
                        "analysis_batch": snapshot_analysis_batch(
                            analysis_batch,
                            (
                                "prompts",
                                "top_feature_ids",
                                "top_feature_scores",
                                "top_feature_activation_values",
                                "logit_target_ids",
                            ),
                        ),
                        "resolved_settings": settings,
                        "call_kwargs": analysis_backend.feature_intervention_call_kwargs(settings),
                        "interventions": [list(spec) for spec in interventions],
                        "result": {
                            "pre_intervention_logits": tensor_fingerprint(wrapper_result.pre_intervention_logits),
                            "post_intervention_logits": tensor_fingerprint(wrapper_post_logits),
                            "intervention_activation_cache": tensor_fingerprint(wrapper_activation_cache),
                            "intervention_specs_json": getattr(wrapper_result, "intervention_specs_json", None),
                            "intervention_config": getattr(wrapper_result, "intervention_config", None),
                        },
                    },
                }

                report = build_intervention_drift_report(
                    graph,
                    feature_row=expected_feature_row,
                    baseline_activation_cache=baseline_activation_cache,
                    intervention_activation_cache=wrapper_activation_cache,
                    baseline_logits=baseline_logits,
                    intervention_logits=wrapper_post_logits,
                    activation_atol=FEATURE_EDGE_ACT_ATOL,
                    activation_rtol=FEATURE_EDGE_ACT_RTOL,
                    logit_atol=EDGE_LOGIT_ATOL,
                    logit_rtol=EDGE_LOGIT_RTOL,
                )
                print(json.dumps(report.to_dict(), indent=2))

                artifact_dir = _maybe_preserve_debug_intervention_artifacts(
                    constrained_feature_ref=constrained_feature_ref,
                    case=case,
                    rendered_prompt=rendered_prompt,
                    graph=graph,
                    feature_row=expected_feature_row,
                    interventions=interventions,
                    baseline_activation_cache=baseline_activation_cache,
                    intervention_activation_cache=wrapper_activation_cache,
                    baseline_logits=baseline_logits,
                    intervention_logits=wrapper_post_logits,
                    graph_target_ids=graph_target_ids,
                    report=report,
                    runtime_state=runtime_state,
                )
                if artifact_dir is not None:
                    print(f"Preserved debug parity artifacts at: {artifact_dir}")

            assert report.feature_row == expected_feature_row
            assert report.total_feature_count == int(graph_context.active_features.size(0))
            assert report.logit_summary.total_logit_count == len(graph.logit_targets)
            if report.divergent_feature_count > 0:
                assert report.layer_with_max_divergence is not None, json.dumps(report.to_dict(), indent=2)


# ==================================================================================================
# Helpers migrated from tests/core/test_analysis_backend_parity.py
# These are used only by the concept-direction backend parity tests below.
# ==================================================================================================


def _configure_gemma3_1b_concept_direction_parity_settings(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
) -> None:
    cfg = module.circuit_tracer_cfg
    cfg.model_name = case.model_name
    cfg.transcoder_set = case.transcoder_set
    cfg.dtype = torch.float32
    cfg.analysis_target_tokens = None
    cfg.target_token_ids = None
    cfg.max_feature_nodes = case.max_feature_nodes
    cfg.offload = "cpu"
    cfg.verbose = False
    cfg.batch_size = case.batch_size
    cfg.max_n_logits = 10
    cfg.desired_logit_prob = 0.95
    cfg.intervention_value_source = "top_feature_activation_values"
    cfg.intervention_scale_factor = case.intervention_scale_factor
    cfg.intervention_max_influence_norm_scale = case.intervention_max_influence_norm_scale
    cfg.intervention_sign_aware_scale = case.intervention_sign_aware_scale
    cfg.intervention_constrained_layers = list(range(module.replacement_model.cfg.n_layers))
    cfg.intervention_apply_activation_function = False
    cfg.intervention_freeze_attention = None
    cfg.intervention_sparse = False
    cfg.intervention_return_activations = False


def _extract_classification_answer_choices(question: str) -> tuple[str, ...]:
    choices = tuple(re.findall(r'"([^"]+)"', question))
    assert choices, f"Unable to extract quoted classification answers from: {question}"
    return choices


def _build_classification_prompt_for_case(entity_name: str, question: str) -> str:
    return build_classification_prompt_text(entity_name, question)


def _iter_concept_direction_examples(
    case: Gemma3ConceptDirectionParityCase,
) -> tuple[tuple[str, str, int], ...]:
    examples: list[tuple[str, str, int]] = []
    examples.extend((entity_name, expected_answer, 0) for entity_name, expected_answer in case.group_a_entities)
    examples.extend((entity_name, expected_answer, 1) for entity_name, expected_answer in case.group_b_entities)
    return tuple(examples)


def _last_token_id(tokenizer: Any, token: str) -> int:
    encoded = tokenizer.encode(token, add_special_tokens=False)
    assert encoded, f"Unable to tokenize {token!r}"
    return int(encoded[-1])


def _project_group_separation(
    states: torch.Tensor,
    group_ids: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[float, float | None]:
    projections = torch.matmul(states.float(), direction.float())
    group_a_mean = float(projections[group_ids == 0].mean().item())
    if torch.any(group_ids == 1):
        return group_a_mean, float(projections[group_ids == 1].mean().item())
    return group_a_mean, None


def _resolve_case_concept_token_ids(
    tokenizer: Any,
    case: Gemma3ConceptDirectionParityCase,
) -> tuple[list[int], list[int]]:
    group_a_token_ids = [_last_token_id(tokenizer, token) for token in case.group_a_tokens]
    group_b_token_ids = [_last_token_id(tokenizer, token) for token in case.group_b_tokens]
    return group_a_token_ids, group_b_token_ids


def _build_case_feature_selection_context(case: Gemma3ConceptDirectionParityCase) -> Any:
    return SimpleNamespace(
        constrained_feature_selection_refs=case.constrained_feature_selection_refs,
        neuronpedia_model=case.neuronpedia_model,
        neuronpedia_set=case.neuronpedia_set,
    )


def _resolve_case_model_variant(case: Gemma3ConceptDirectionParityCase) -> str:
    model_name = case.model_name
    if model_name.endswith("gemma-3-1b-it"):
        return "1b_it"
    if model_name.endswith("gemma-3-4b-it"):
        return "4b_262k_it" if "width_262k" in case.transcoder_set else "4b_it"
    raise ValueError(f"Unsupported concept-direction parity model selection: {model_name!r}")


def _serialize_attribution_targets(targets: Any) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for target in cast(list[Any], list(targets or [])):
        if isinstance(target, CustomTarget):
            serialized.append(
                {
                    "kind": "CustomTarget",
                    "token_str": str(target.token_str),
                    "prob": float(target.prob),
                    "vec": tensor_fingerprint(torch.as_tensor(target.vec, dtype=torch.float32)),
                }
            )
            continue
        serialized.append({"kind": type(target).__name__, "value": str(target)})
    return serialized


def _build_concept_direction_graph_call_kwargs(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
    *,
    rendered_prompt: str,
    concept_direction: torch.Tensor,
) -> dict[str, Any]:
    tokenizer = module.replacement_model.tokenizer
    group_a_token_ids, group_b_token_ids = _resolve_case_concept_token_ids(tokenizer, case)
    analysis_backend = require_analysis_backend(module)
    attribution_targets = analysis_backend.build_concept_attribution_targets(
        module,
        rendered_prompt,
        concept_direction,
        case.concept_label,
        concept_group_a_token_ids=group_a_token_ids,
        concept_group_b_token_ids=group_b_token_ids,
        concept_direction_mode=case.concept_direction_mode,
    )
    assert attribution_targets, "Expected concept-direction attribution targets to be resolved"
    return {
        "attribution_targets": attribution_targets,
        "max_n_logits": 1,
        "desired_logit_prob": 1.0,
        "batch_size": case.batch_size,
        "max_feature_nodes": case.max_feature_nodes,
        "offload": "cpu",
        "verbose": False,
    }


def _assert_reference_graph_payload_matches_direct_artifacts(
    artifacts: ConceptDirectionParityArtifacts,
    reference_payload: Mapping[str, Any],
    *,
    label: str,
    report: Mapping[str, Any],
) -> None:
    direct_rows = [list(row) for row in artifacts.top_feature_ids]
    reference_rows = cast(list[list[int]], reference_payload.get("top_feature_ids", []))
    assertion_context = {
        "label": label,
        "direct_top_feature_ids": direct_rows,
        "reference_top_feature_ids": reference_rows,
        "direct_top_feature_scores": artifacts.top_feature_scores.detach().cpu().tolist(),
        "reference_top_feature_scores": list(reference_payload.get("top_feature_scores", [])),
        "report": report,
    }
    assert direct_rows == reference_rows, json.dumps(assertion_context, indent=2, default=str)
    assert_close(
        artifacts.top_feature_scores,
        torch.as_tensor(reference_payload.get("top_feature_scores", []), dtype=torch.float32),
        rtol=VALUE_RTOL,
        atol=VALUE_ATOL,
    )
    direct_graph_artifact = cast(dict[str, Any], artifacts.graph_stage_artifact)
    assert direct_graph_artifact["graph_result_input_tokens"] == reference_payload.get("graph_result_input_tokens"), (
        json.dumps(assertion_context, indent=2, default=str)
    )
    assert direct_graph_artifact["selected_feature_count"] == reference_payload.get("selected_feature_count"), (
        json.dumps(assertion_context, indent=2, default=str)
    )
    assert direct_graph_artifact["active_feature_count"] == reference_payload.get("active_feature_count"), json.dumps(
        assertion_context, indent=2, default=str
    )
    assert direct_graph_artifact["attribution_targets"] == reference_payload.get("attribution_targets"), json.dumps(
        assertion_context, indent=2, default=str
    )
    assert direct_graph_artifact["graph_call_kwargs"] == reference_payload.get("graph_call_kwargs"), json.dumps(
        assertion_context, indent=2, default=str
    )


def _validate_concept_direction_feature_intervention_wrappers(
    module: Any,
    rendered_prompt: str,
    graph: Graph,
    top_features_result: AnalysisBatch,
) -> list[dict[str, Any]]:
    analysis_backend = require_analysis_backend(module)
    settings = analysis_backend.resolve_feature_intervention_settings(module, {})
    feature_rows = torch.as_tensor(top_features_result.top_feature_ids, dtype=torch.long)
    feature_scores = torch.as_tensor(top_features_result.top_feature_scores, dtype=torch.float32)
    activation_values = torch.as_tensor(top_features_result.top_feature_activation_values, dtype=torch.float32)
    logit_target_ids = graph.logit_tokens.detach().cpu()
    baseline_logits_raw, baseline_activation_cache = module.replacement_model.get_activations(
        rendered_prompt,
        apply_activation_function=False,
    )
    baseline_logits = last_token_logits(torch.as_tensor(baseline_logits_raw, dtype=torch.float32)).detach().cpu()
    baseline_activation_cache = torch.as_tensor(baseline_activation_cache, dtype=torch.float32).detach().cpu()
    graph_adjacency = graph.adjacency_matrix.detach().float().cpu()
    graph_active_features = graph.active_features.detach().long().cpu()
    selected_feature_indices = graph.selected_features.detach().long().cpu()
    graph_selected_features = graph_active_features.index_select(0, selected_feature_indices)
    graph_feature_count = int(selected_feature_indices.numel())
    graph_logit_count = int(logit_target_ids.numel())
    baseline_demeaned_logits = baseline_logits[logit_target_ids] - baseline_logits.mean()

    wrapper_batch = AnalysisBatch(
        prompts=[rendered_prompt],
        top_feature_ids=feature_rows,
        top_feature_scores=feature_scores,
        top_feature_activation_values=activation_values,
        logit_target_ids=logit_target_ids,
    )
    interventions, _ = analysis_backend.build_feature_interventions(wrapper_batch, settings)
    wrapper_result = cast(
        Any,
        it.feature_intervention_forward(
            module,
            wrapper_batch,
            batch=cast(Any, None),
            batch_idx=0,
            prompt=rendered_prompt,
            intervention_return_activations=True,
        ),
    )
    assert getattr(wrapper_result, "intervention_activation_cache", None) is not None
    wrapper_pre = torch.as_tensor(wrapper_result.pre_intervention_logits, dtype=torch.float32).detach().cpu()
    wrapper_post = torch.as_tensor(wrapper_result.post_intervention_logits, dtype=torch.float32).detach().cpu()
    wrapper_activation_cache = (
        torch.as_tensor(wrapper_result.intervention_activation_cache, dtype=torch.float32).detach().cpu()
    )
    expected_feature_rows = [_feature_row_from_tensor(feature_row_tensor) for feature_row_tensor in feature_rows]
    assert_close(wrapper_pre, baseline_logits, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert wrapper_result.intervention_feature_ids == [feature_row[2] for feature_row in expected_feature_rows]
    assert wrapper_result.intervention_positions == [feature_row[1] for feature_row in expected_feature_rows]
    expected_intervention_values = torch.tensor(
        [intervention[3] for intervention in interventions],
        dtype=torch.float32,
    )
    assert_close(
        torch.tensor(wrapper_result.intervention_values, dtype=torch.float32),
        expected_intervention_values,
        rtol=VALUE_RTOL,
        atol=VALUE_ATOL,
    )
    feature_intervention_call_kwargs = analysis_backend.feature_intervention_call_kwargs(settings)
    feature_intervention_call_kwargs["return_activations"] = False
    expected_aggregate_feature_effects = torch.zeros(graph_feature_count, dtype=torch.float32)
    expected_aggregate_logit_effects = torch.zeros(graph_logit_count, dtype=torch.float32)

    summaries: list[dict[str, Any]] = []
    for index, feature_row_tensor in enumerate(feature_rows):
        feature_row = _feature_row_from_tensor(feature_row_tensor)
        single_post_logits_raw, _ = module.replacement_model.feature_intervention(
            rendered_prompt,
            [interventions[index]],
            **feature_intervention_call_kwargs,
        )
        single_post_logits = (
            last_token_logits(
                torch.as_tensor(single_post_logits_raw, dtype=torch.float32),
            )
            .detach()
            .cpu()
        )
        intervention_value = float(interventions[index][3])

        layer, position, feature_id = feature_row
        baseline_feature_activation = float(baseline_activation_cache[layer, position, feature_id].item())
        returned_cache_feature_activation = float(wrapper_activation_cache[layer, position, feature_id].item())
        requested_activation_delta = intervention_value - baseline_feature_activation
        assert abs(baseline_feature_activation) > 1e-12, json.dumps(
            {
                "feature_row": list(feature_row),
                "baseline_activation": baseline_feature_activation,
                "intervention_value": intervention_value,
            },
            indent=2,
        )

        graph_node_index = _find_active_feature_index(graph_selected_features, feature_row)
        raw_graph_feature_effects = graph_adjacency[:graph_feature_count, graph_node_index]
        raw_graph_logit_effects = graph_adjacency[-graph_logit_count:, graph_node_index]
        graph_effect_scale = requested_activation_delta / baseline_feature_activation
        expected_feature_effects = raw_graph_feature_effects * graph_effect_scale
        expected_logit_effects = raw_graph_logit_effects * graph_effect_scale
        expected_aggregate_feature_effects += expected_feature_effects
        expected_aggregate_logit_effects += expected_logit_effects
        single_demeaned_logits = single_post_logits[logit_target_ids] - single_post_logits.mean()
        actual_logit_effects = single_demeaned_logits - baseline_demeaned_logits
        nonzero_effects = (expected_logit_effects.abs() > EDGE_LOGIT_ATOL) & (
            actual_logit_effects.abs() > EDGE_LOGIT_ATOL
        )
        if torch.any(nonzero_effects):
            assert torch.all(
                torch.sign(actual_logit_effects[nonzero_effects]) == torch.sign(expected_logit_effects[nonzero_effects])
            ), json.dumps(
                {
                    "feature_row": list(feature_row),
                    "feature_score": float(feature_scores[index].item()),
                    "baseline_activation": baseline_feature_activation,
                    "requested_activation_delta": requested_activation_delta,
                    "intervention_value": intervention_value,
                    "graph_effect_scale": graph_effect_scale,
                    "raw_graph_logit_effects": raw_graph_logit_effects.tolist(),
                    "expected_graph_logit_effects": expected_logit_effects.tolist(),
                    "actual_graph_logit_effects": actual_logit_effects.tolist(),
                },
                indent=2,
            )

        summaries.append(
            {
                "feature_row": list(feature_row),
                "feature_score": float(feature_scores[index].item()),
                "baseline_activation": baseline_feature_activation,
                "returned_cache_activation": returned_cache_feature_activation,
                "returned_cache_delta": returned_cache_feature_activation - baseline_feature_activation,
                "requested_activation_delta": requested_activation_delta,
                "intervention_value": intervention_value,
                "graph_node_index": graph_node_index,
                "graph_effect_scale": graph_effect_scale,
                "expected_self_feature_effect": float(expected_feature_effects[graph_node_index].item()),
                "raw_graph_logit_effects": raw_graph_logit_effects.tolist(),
                "expected_graph_logit_effects": expected_logit_effects.tolist(),
                "actual_graph_logit_effects": actual_logit_effects.tolist(),
                "returned_activation_cache": True,
            }
        )
    for summary in summaries:
        graph_node_index = int(summary["graph_node_index"])
        expected_feature_delta = float(expected_aggregate_feature_effects[graph_node_index].item())
        actual_feature_delta = float(summary["returned_cache_delta"])
        expected_cache_activation = float(summary["baseline_activation"] + expected_feature_delta)
        summary["expected_returned_cache_activation"] = expected_cache_activation
        summary["expected_aggregate_feature_effect"] = expected_feature_delta
        if (
            abs(expected_feature_delta) > FEATURE_DELTA_SIGN_ATOL
            and abs(actual_feature_delta) > FEATURE_DELTA_SIGN_ATOL
        ):
            assert torch.sign(torch.tensor(actual_feature_delta)) == torch.sign(torch.tensor(expected_feature_delta)), (
                json.dumps(summary, indent=2)
            )
    wrapper_demeaned_logits = wrapper_post[logit_target_ids] - wrapper_post.mean()
    actual_aggregate_logit_effects = wrapper_demeaned_logits - baseline_demeaned_logits
    aggregate_nonzero_effects = (expected_aggregate_logit_effects.abs() > EDGE_LOGIT_ATOL) & (
        actual_aggregate_logit_effects.abs() > EDGE_LOGIT_ATOL
    )
    if torch.any(aggregate_nonzero_effects):
        assert torch.all(
            torch.sign(actual_aggregate_logit_effects[aggregate_nonzero_effects])
            == torch.sign(expected_aggregate_logit_effects[aggregate_nonzero_effects])
        ), json.dumps(
            {
                "top_features": [list(feature_row) for feature_row in expected_feature_rows],
                "feature_scores": feature_scores.tolist(),
                "intervention_values": expected_intervention_values.tolist(),
                "expected_aggregate_logit_effects": expected_aggregate_logit_effects.tolist(),
                "actual_aggregate_logit_effects": actual_aggregate_logit_effects.tolist(),
            },
            indent=2,
        )
    return summaries


def _build_concept_direction_graph_artifacts(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
    *,
    path_label: str,
    concept_direction: torch.Tensor,
    prompt_alignment_snapshots: tuple[dict[str, Any], ...] = (),
    extraction_snapshots: tuple[dict[str, Any], ...] = (),
    group_projection_states: torch.Tensor,
    group_ids: torch.Tensor,
    direction_stage_artifact: dict[str, Any] | None = None,
    validate_feature_edges: bool = False,
) -> ConceptDirectionParityArtifacts:
    tokenizer = module.replacement_model.tokenizer
    rendered_prompt = _render_debug_prompt(tokenizer, case.prompt, case.prompt_render_mode)
    target_ids, _ = _resolve_debug_graph_target_ids(
        tokenizer,
        case.target_tokens,
        use_chat_template=case.prompt_render_mode != "plain",
    )
    target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)
    group_a_token_ids, group_b_token_ids = _resolve_case_concept_token_ids(tokenizer, case)
    graph_call_kwargs = _build_concept_direction_graph_call_kwargs(
        module,
        case,
        rendered_prompt=rendered_prompt,
        concept_direction=concept_direction,
    )
    serialized_graph_call_kwargs = _serialize_intervention_call_kwargs(
        {
            **graph_call_kwargs,
            "attribution_targets": _serialize_attribution_targets(graph_call_kwargs.get("attribution_targets")),
        }
    )
    analysis_batch = AnalysisBatch(
        prompts=[rendered_prompt],
        concept_direction=concept_direction,
        concept_label=case.concept_label,
        concept_direction_mode=case.concept_direction_mode,
        concept_group_a_token_ids=group_a_token_ids,
        concept_group_b_token_ids=group_b_token_ids,
        logit_target_ids=target_ids_tensor,
    )
    graph_result = cast(
        Any,
        it.compute_attribution_graph(
            module,
            analysis_batch,
            batch=cast(Any, None),
            batch_idx=0,
            **graph_call_kwargs,
        ),
    )
    graph = require_analysis_backend(module).hydrate_graph_from_batch(graph_result) if validate_feature_edges else None
    influence_result = cast(Any, it.graph_node_influence(module, graph_result, batch=cast(Any, None), batch_idx=0))
    top_payload = dict(cast(Any, graph_result))
    top_payload.update(dict(cast(Any, influence_result)))
    top_features_result, applied_feature_rows = _extract_top_features_with_optional_filter(
        module,
        _build_case_feature_selection_context(case),
        top_payload,
        top_n=case.top_n,
    )
    intervention_result = cast(
        Any,
        it.feature_intervention_forward(
            module,
            AnalysisBatch(
                prompts=[rendered_prompt],
                top_feature_ids=top_features_result.top_feature_ids,
                top_feature_scores=top_features_result.top_feature_scores,
                top_feature_activation_values=top_features_result.top_feature_activation_values,
                logit_target_ids=torch.tensor([target_ids[0]], dtype=torch.long),
            ),
            batch=cast(Any, None),
            batch_idx=0,
        ),
    )
    pre_logits = intervention_result.pre_intervention_logits.float().cpu()
    post_logits = intervention_result.post_intervention_logits.float().cpu()
    top_feature_scores = torch.as_tensor(top_features_result.top_feature_scores, dtype=torch.float32).cpu()
    top_feature_activation_values = torch.as_tensor(
        top_features_result.top_feature_activation_values,
        dtype=torch.float32,
    ).cpu()
    expected_scale_factors = torch.full_like(top_feature_scores, float(case.intervention_scale_factor))
    if case.intervention_max_influence_norm_scale and top_feature_scores.numel() > 0:
        max_abs_score = top_feature_scores.abs().max().clamp_min(1e-12)
        expected_scale_factors = expected_scale_factors * (top_feature_scores.abs() / max_abs_score)
    expected_intervention_values = top_feature_activation_values * expected_scale_factors
    if case.intervention_sign_aware_scale:
        expected_intervention_values = (
            top_feature_scores.sign() * top_feature_activation_values.abs() * expected_scale_factors
        )
    assert_close(
        torch.as_tensor(intervention_result.intervention_scale_factors, dtype=torch.float32).cpu(),
        expected_scale_factors,
        rtol=VALUE_RTOL,
        atol=VALUE_ATOL,
    )
    assert_close(
        torch.as_tensor(intervention_result.intervention_values, dtype=torch.float32).cpu(),
        expected_intervention_values,
        rtol=VALUE_RTOL,
        atol=VALUE_ATOL,
    )
    group_a_projection_mean, group_b_projection_mean = _project_group_separation(
        group_projection_states,
        group_ids,
        concept_direction.float().cpu(),
    )
    top_feature_rows = tuple(
        _feature_row_from_tensor(torch.as_tensor(row)) for row in top_features_result.top_feature_ids
    )
    wrapper_summaries: list[dict[str, Any]] = []
    if validate_feature_edges:
        assert graph is not None
        wrapper_summaries = _validate_concept_direction_feature_intervention_wrappers(
            module,
            rendered_prompt,
            graph,
            top_features_result,
        )
    use_chat_template = case.prompt_render_mode != "plain"
    target_token_variants = [
        _resolve_debug_graph_target_token_variants(tokenizer, token, use_chat_template=use_chat_template)
        for token in case.target_tokens
    ]
    graph_stage_artifact = {
        "path_label": f"{path_label}_graph_pipeline",
        "graph_analysis_batch": snapshot_analysis_batch(
            analysis_batch,
            fields=(
                "concept_direction",
                "concept_label",
                "concept_direction_mode",
                "concept_group_a_token_ids",
                "concept_group_b_token_ids",
                "prompts",
                "logit_target_ids",
            ),
            max_items=32,
        ),
        "graph_call_kwargs": serialized_graph_call_kwargs,
        "attribution_targets": _serialize_attribution_targets(graph_call_kwargs.get("attribution_targets")),
        "graph_input_tokens": _summarize_graph_input_tokens(
            tokenizer,
            rendered_prompt,
            cast(PromptRenderMode, case.prompt_render_mode),
            graph_result.input_tokens,
        ),
        "graph_result_input_tokens": (
            torch.as_tensor(graph_result.input_tokens, dtype=torch.long).cpu().reshape(-1).tolist()
        ),
        "selected_feature_count": int(torch.as_tensor(graph_result.selected_features, dtype=torch.long).numel()),
        "active_feature_count": int(
            torch.as_tensor(graph_result.active_features, dtype=torch.long).reshape(-1, 3).shape[0]
        ),
        "target_ids": [int(token_id) for token_id in target_ids],
        "target_token_variants": target_token_variants,
        "top_n": int(case.top_n),
        "requested_feature_selection": _serialize_constrained_feature_selection(
            case.constrained_feature_selection_refs
        ),
        "applied_feature_selection_rows": [list(row) for row in applied_feature_rows],
        "top_features": [list(row) for row in top_feature_rows],
        "top_feature_scores": top_feature_scores.tolist(),
        "top_feature_activation_values": top_feature_activation_values.tolist(),
        "intervention_values": torch.as_tensor(intervention_result.intervention_values, dtype=torch.float32)
        .cpu()
        .tolist(),
        "intervention_scale_factors": torch.as_tensor(
            intervention_result.intervention_scale_factors,
            dtype=torch.float32,
        )
        .cpu()
        .tolist(),
        "expected_intervention_values": expected_intervention_values.tolist(),
        "expected_intervention_scale_factors": expected_scale_factors.tolist(),
        "pre_gap": float((pre_logits[target_ids[0]] - pre_logits[target_ids[1]]).item()),
        "post_gap": float((post_logits[target_ids[0]] - post_logits[target_ids[1]]).item()),
        "gap_delta": float(
            (post_logits[target_ids[0]] - post_logits[target_ids[1]]).item()
            - (pre_logits[target_ids[0]] - pre_logits[target_ids[1]]).item()
        ),
        "pre_logits_fingerprint": tensor_fingerprint(pre_logits),
        "post_logits_fingerprint": tensor_fingerprint(post_logits),
        "pre_top_logits": _summarize_top_logits(tokenizer, pre_logits),
        "post_top_logits": _summarize_top_logits(tokenizer, post_logits),
        "wrapper_feature_intervention_summaries": wrapper_summaries,
    }
    return ConceptDirectionParityArtifacts(
        path_label=path_label,
        concept_direction=concept_direction.float().cpu(),
        top_feature_ids=top_feature_rows,
        top_feature_scores=torch.as_tensor(top_features_result.top_feature_scores, dtype=torch.float32).cpu(),
        top_feature_activation_values=torch.as_tensor(
            top_features_result.top_feature_activation_values,
            dtype=torch.float32,
        ).cpu(),
        pre_logits=pre_logits,
        post_logits=post_logits,
        pre_gap=float((pre_logits[target_ids[0]] - pre_logits[target_ids[1]]).item()),
        post_gap=float((post_logits[target_ids[0]] - post_logits[target_ids[1]]).item()),
        group_a_projection_mean=group_a_projection_mean,
        group_b_projection_mean=group_b_projection_mean,
        prompt_alignment_snapshots=prompt_alignment_snapshots,
        extraction_snapshots=extraction_snapshots,
        direction_stage_artifact=direction_stage_artifact,
        graph_stage_artifact=graph_stage_artifact,
    )


def _compute_embed_concept_direction_stage(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
) -> ConceptDirectionStageResult:
    tokenizer = module.replacement_model.tokenizer
    embed_result = cast(
        Any,
        it.concept_direction(
            module,
            AnalysisBatch(
                concept_group_a=list(case.group_a_tokens),
                concept_group_b=list(case.group_b_tokens),
                concept_label=case.concept_label,
                concept_direction_mode=case.concept_direction_mode,
            ),
            batch=cast(Any, None),
            batch_idx=0,
        ),
    )
    embed_weight = module.model.get_input_embeddings().weight.detach().cpu().float()
    group_a_states = embed_weight[torch.tensor([_last_token_id(tokenizer, token) for token in case.group_a_tokens])]
    if case.group_b_tokens:
        group_b_states = embed_weight[torch.tensor([_last_token_id(tokenizer, token) for token in case.group_b_tokens])]
        group_projection_states = torch.cat([group_a_states, group_b_states], dim=0)
        group_ids = torch.tensor([0] * len(group_a_states) + [1] * len(group_b_states), dtype=torch.long)
    else:
        group_projection_states = group_a_states
        group_ids = torch.tensor([0] * len(group_a_states), dtype=torch.long)
    direction_geometry = compute_concept_direction_geometry(
        group_projection_states,
        group_ids,
        direction_mode=case.concept_direction_mode,
    )
    return ConceptDirectionStageResult(
        path_label="embed",
        concept_direction=torch.as_tensor(embed_result.concept_direction, dtype=torch.float32).detach().cpu(),
        group_projection_states=group_projection_states,
        group_ids=group_ids,
        direction_stage_artifact=build_concept_direction_stage_artifact(
            path_label="embed_direction",
            direction_mode=case.concept_direction_mode,
            direction=embed_result.concept_direction,
            tokenizer=tokenizer,
            group_a_token_ids=embed_result.concept_group_a_token_ids,
            group_b_token_ids=embed_result.concept_group_b_token_ids,
            geometry=direction_geometry,
        ),
    )


def _compute_store_concept_direction_stage(
    module: Any,
    case: Gemma3ConceptDirectionParityCase,
    *,
    context_enhanced: bool,
) -> ConceptDirectionStageResult:
    tokenizer = module.replacement_model.tokenizer
    model_backend = getattr(module, "_model_backend")
    device = next(module.model.parameters()).device
    answer_choices = _extract_classification_answer_choices(case.classification_question)

    extracted_batches: list[Any] = []
    prompt_alignment_snapshots: list[dict[str, Any]] = []
    prompt_alignment_artifacts: list[dict[str, Any]] = []
    extraction_snapshots: list[dict[str, Any]] = []
    context_extraction_artifacts: list[dict[str, Any]] = []
    example_logit_diffs: list[float] = []
    for entity_name, expected_answer, group_id in _iter_concept_direction_examples(case):
        raw_prompt = _build_classification_prompt_for_case(entity_name, case.classification_question)
        rendered_prompt = _render_debug_prompt(tokenizer, raw_prompt, case.prompt_render_mode)
        rendered_prompt_with_answer = f"{rendered_prompt}{expected_answer}"
        add_special_tokens = case.prompt_render_mode == "plain"
        probe_surface_text = normalize_prompt_entity_text(entity_name)
        prompt_alignment_snapshot = build_prompt_alignment_snapshot(
            tokenizer,
            rendered_prompt_with_answer,
            probe_text=probe_surface_text,
            answer_text=expected_answer,
            add_special_tokens=add_special_tokens,
        )
        context_token_index, context_token_source = resolve_prompt_alignment_context_index(prompt_alignment_snapshot)
        prompt_alignment_snapshot_dict = prompt_alignment_snapshot.to_dict()
        prompt_alignment_snapshot_dict["context_token_index"] = context_token_index
        prompt_alignment_snapshot_dict["context_token_source"] = context_token_source
        prompt_alignment_snapshots.append(prompt_alignment_snapshot_dict)

        enc_prompt = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            padding=False,
            add_special_tokens=add_special_tokens,
        )
        batch_dev_prompt = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in dict(enc_prompt).items()
        }
        enc_cache = tokenizer(
            rendered_prompt_with_answer,
            return_tensors="pt",
            padding=False,
            add_special_tokens=add_special_tokens,
        )
        batch_dev_cache = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in dict(enc_cache).items()
        }
        with torch.no_grad():
            prompt_logits, _ = model_backend.fwd_w_cache(
                model=module.model,
                batch=batch_dev_prompt,
                names_filter=case.store_concept_cache_key,
            )
            cache_logits, cache = model_backend.fwd_w_cache(
                model=module.model,
                batch=batch_dev_cache,
                names_filter=case.store_concept_cache_key,
            )

        cache_answer_index = int(prompt_alignment_snapshot.answer_index)
        assert int(prompt_logits.shape[1] - 1) == prompt_alignment_snapshot.answer_index - 1
        assert cache_answer_index < int(cache_logits.shape[1])
        prompt_alignment_artifacts.append(
            build_prompt_alignment_artifact(
                prompt_alignment_snapshot,
                probe_surface_text=probe_surface_text,
                cache_rendered_prompt=rendered_prompt_with_answer,
                cache_input_ids=enc_cache["input_ids"][0].tolist(),
                cache_input_tokens=tokenizer.convert_ids_to_tokens(enc_cache["input_ids"][0].tolist()),
                cache_answer_index=cache_answer_index,
                context_token_index=context_token_index,
                context_token_source=context_token_source,
            )
        )
        example_logits = prompt_logits[0, int(prompt_logits.shape[1] - 1)].float().cpu()
        expected_id = _last_token_id(tokenizer, expected_answer)
        alternative_answers = [choice for choice in answer_choices if choice != expected_answer]
        alternative_id = expected_id if not alternative_answers else _last_token_id(tokenizer, alternative_answers[0])
        example_logit_diff = float((example_logits[expected_id] - example_logits[alternative_id]).item())
        example_logit_diffs.append(example_logit_diff)
        analysis_batch = AnalysisBatch(
            cache={case.store_concept_cache_key: torch.as_tensor(cache[case.store_concept_cache_key]).detach().cpu()},
            answer_indices=torch.tensor([cache_answer_index], dtype=torch.long),
            context_token_indices=torch.tensor(
                [-1 if context_token_index is None else int(context_token_index)],
                dtype=torch.long,
            ),
            orig_labels=torch.tensor([group_id], dtype=torch.long),
            logit_diffs=torch.tensor([example_logit_diff], dtype=torch.float32),
            concept_group_a_label_ids=[0],
            concept_group_b_label_ids=[1] if case.group_b_entities else None,
            concept_group_a_name=case.group_a_name,
            concept_group_b_name=case.group_b_name,
            concept_cache_key=case.store_concept_cache_key,
            concept_correct_only=False,
            concept_weight_by_logit_diff=True,
        )
        source_batch = cast(
            Any,
            it.extract_concept_latent_state(
                module,
                analysis_batch,
                batch=cast(Any, None),
                batch_idx=0,
                context_enhanced=context_enhanced,
                context_scale=case.context_enhanced_scale,
                use_answer_state_as_basis=case.use_answer_state_as_basis,
            ),
        )
        if context_enhanced:
            extraction_snapshot = capture_context_enhanced_extraction_snapshot(
                source_batch,
                context_scale=case.context_enhanced_scale,
                use_answer_state_as_basis=case.use_answer_state_as_basis,
            )
            extraction_snapshots.append(extraction_snapshot.to_dict())
            context_extraction_artifacts.append(
                build_context_extraction_artifact(
                    extraction_snapshot,
                    prompt_alignment_artifact=prompt_alignment_artifacts[-1],
                )
            )
        extracted_batches.append(
            cast(
                Any,
                it.extract_concept_latent_examples(
                    module,
                    source_batch,
                    batch=cast(Any, None),
                    batch_idx=0,
                ),
            )
        )

    latent_state_rows = [batch.concept_latent_state for batch in extracted_batches]
    group_id_rows = [batch.concept_group_id for batch in extracted_batches]
    group_name_rows = [batch.concept_group_name for batch in extracted_batches]
    example_weight_rows = [batch.concept_example_weight for batch in extracted_batches]
    flattened_states, group_ids, example_weights, _ = _flatten_concept_store_rows(
        latent_state_rows,
        group_id_rows,
        group_name_rows,
        example_weight_rows,
    )
    direction_geometry = compute_concept_direction_geometry(
        flattened_states,
        group_ids,
        direction_mode=case.concept_direction_mode,
        example_weights=example_weights,
    )
    store_result = cast(
        Any,
        it.concept_direction(
            module,
            AnalysisBatch(
                concept_latent_state=latent_state_rows,
                concept_group_id=group_id_rows,
                concept_group_name=group_name_rows,
                concept_example_weight=example_weight_rows,
                concept_label=case.concept_label,
                concept_direction_mode=case.concept_direction_mode,
                concept_group_a_name=case.group_a_name,
                concept_group_b_name=case.group_b_name,
            ),
            batch=cast(Any, None),
            batch_idx=0,
        ),
    )
    direction_stage_artifact = build_concept_direction_stage_artifact(
        path_label="store_context_direction" if context_enhanced else "store_plain_direction",
        direction_mode=case.concept_direction_mode,
        direction=store_result.concept_direction,
        tokenizer=tokenizer,
        group_a_token_ids=[_last_token_id(tokenizer, token) for token in case.group_a_tokens],
        group_b_token_ids=[_last_token_id(tokenizer, token) for token in case.group_b_tokens],
        geometry=direction_geometry,
        prompt_examples=prompt_alignment_artifacts,
        example_logit_diffs=example_logit_diffs,
    ) | {
        "store_latent_extraction_mode": "context_enhanced" if context_enhanced else "answer_position_state",
        "context_enhanced_scale": case.context_enhanced_scale,
        "use_answer_state_as_basis": case.use_answer_state_as_basis,
        "context_enhanced_extraction": context_extraction_artifacts,
    }
    return ConceptDirectionStageResult(
        path_label="store_context" if context_enhanced else "store_plain",
        concept_direction=torch.as_tensor(store_result.concept_direction, dtype=torch.float32).detach().cpu(),
        group_projection_states=flattened_states,
        group_ids=group_ids,
        prompt_alignment_snapshots=tuple(prompt_alignment_snapshots),
        extraction_snapshots=tuple(extraction_snapshots),
        direction_stage_artifact=direction_stage_artifact,
    )
