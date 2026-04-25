from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import interpretune as it
import pytest
import torch
from circuit_tracer import Graph, ReplacementModel, attribute
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.graph import compute_node_influence
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.utils.demo_utils import get_unembed_vecs
from nnsight import save
from torch.testing import assert_close

from interpretune.analysis.backends import require_analysis_backend
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.analysis.ops.helpers import (
    apply_feature_selection_filter,
    last_token_logits,
)
from interpretune.config import AnalysisCfg, CircuitTracerConfig, NNsightConfig, init_analysis_cfgs
from tests import load_dotenv
from tests.analysis_resource_utils import clear_nnsight_test_state, serial_test_cleanup
from tests.configuration import config_modules
from tests.conftest import FixtPhase, clean_cuda, session_fixture_hook_exec
from tests.core.cfg_aliases import CircuitTracerNNsightGemma2, CircuitTracerNNsightGemma3
from tests.runif import RunIf


CONCEPT_DIRECTION_COSINE_MIN = 0.999
VALUE_RTOL = 1e-6
VALUE_ATOL = 1e-6
MIN_FREE_CUDA_BYTES_FOR_PARITY = 10 * 1024**3
TOKEN_EDGE_ACT_ATOL = 1e-3
TOKEN_EDGE_ACT_RTOL = 1e-3
FEATURE_EDGE_ACT_ATOL = 1e-3
FEATURE_EDGE_ACT_RTOL = 1e-5
EDGE_LOGIT_ATOL = 1e-5
EDGE_LOGIT_RTOL = 1e-3
GEMMA3_IT_PREFIX_TOKEN_IDS = (2, 105, 2364, 107)
RUNIF: Any = RunIf


@dataclass(frozen=True)
class SemanticInterventionParityCase:
    prompt: str
    capitals: list[str]
    states: list[str]
    label: str
    n_top: int


@dataclass(frozen=True)
class NativeSemanticBaseline:
    group_a_ids: list[int]
    top_features: list[tuple[int, int, int]]
    interventions: list[tuple[int, int, int, float]]
    activation_values: torch.Tensor
    pre_logits: torch.Tensor
    post_logits: torch.Tensor
    pre_gap: float
    post_gap: float
    concept_direction: torch.Tensor


@dataclass(frozen=True)
class OpSemanticBaseline:
    direction_mode: str
    group_a_token_ids: list[int]
    concept_direction: torch.Tensor
    top_feature_ids: list[tuple[int, int, int]]
    activation_values: torch.Tensor
    intervention_feature_ids: list[int]
    intervention_positions: list[int]
    intervention_values: torch.Tensor
    pre_logits: torch.Tensor
    post_logits: torch.Tensor
    pre_gap: float
    post_gap: float


@dataclass(frozen=True)
class Gemma3InstructionInterventionCase:
    prompt: str
    model_name: str
    transcoder_set: str
    pos_start: int
    token_position_limit: int
    error_layer_limit: int
    feature_sample_count: int
    reference_feature_sample_count: int
    intervention_scale_factor: float


@dataclass(frozen=True)
class Gemma3ConceptDirectionParityCase:
    experiment_name: str
    config_name: str
    calibration_surface: str
    parity_artifact_name: str
    reference_artifact_name: str
    notebook_pipeline_artifact_name: str | None
    session_name: str
    prompt: str
    prompt_render_mode: str
    target_tokens: tuple[str, str]
    key_tokens: tuple[str, ...]
    model_name: str
    transcoder_set: str
    neuronpedia_model: str
    neuronpedia_set: str
    max_feature_nodes: int
    batch_size: int
    intervention_scale_factor: float
    top_n: int
    concept_direction_mode: str
    group_a_tokens: tuple[str, ...]
    group_b_tokens: tuple[str, ...]
    group_a_entities: tuple[tuple[str, str], ...]
    group_b_entities: tuple[tuple[str, str], ...]
    group_a_name: str
    group_b_name: str
    concept_label: str
    classification_question: str
    store_concept_cache_key: str
    context_enhanced_scale: float
    constrained_feature_selection_refs: Any | None = None
    require_cross_path_feature_overlap: bool = True
    require_gap_improvement: bool = True


@dataclass(frozen=True)
class ConceptDirectionStageResult:
    path_label: str
    concept_direction: torch.Tensor
    group_projection_states: torch.Tensor
    group_ids: torch.Tensor
    prompt_alignment_snapshots: tuple[dict[str, Any], ...] = ()
    extraction_snapshots: tuple[dict[str, Any], ...] = ()
    direction_stage_artifact: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConceptDirectionParityArtifacts:
    path_label: str
    concept_direction: torch.Tensor
    top_feature_ids: tuple[tuple[int, int, int], ...]
    top_feature_scores: torch.Tensor
    top_feature_activation_values: torch.Tensor
    pre_logits: torch.Tensor
    post_logits: torch.Tensor
    pre_gap: float
    post_gap: float
    group_a_projection_mean: float
    group_b_projection_mean: float | None
    prompt_alignment_snapshots: tuple[dict[str, Any], ...] = ()
    extraction_snapshots: tuple[dict[str, Any], ...] = ()
    direction_stage_artifact: dict[str, Any] | None = None
    graph_stage_artifact: dict[str, Any] | None = None


@dataclass(frozen=True)
class GraphEdgeValidationContext:
    prompt: str
    adjacency_matrix: torch.Tensor
    active_features: torch.Tensor
    logit_tokens: torch.Tensor
    total_active_features: int
    activation_cache: torch.Tensor
    relevant_activations: torch.Tensor
    demeaned_relevant_logits: torch.Tensor


@dataclass(frozen=True)
class EdgeValidationSummary:
    label: str
    max_activation_abs_error: float
    max_logit_abs_error: float


@dataclass(frozen=True)
class WrapperFeatureInterventionSummary:
    feature_row: tuple[int, int, int]
    intervention_value: float
    edge_max_logit_abs_error: float
    returned_activation_cache: bool


@pytest.fixture
def semantic_intervention_parity_case() -> SemanticInterventionParityCase:
    return SemanticInterventionParityCase(
        prompt="Fact: the capital of the state containing Dallas is",
        capitals=["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"],
        states=["▁Texas", "▁California", "▁Washington", "▁Georgia"],
        label="Concept: Capitals − States",
        n_top=10,
    )


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


@pytest.fixture
def ct_nnsight_session_factory(tmp_path):
    def _build_test_cfg() -> CircuitTracerNNsightGemma2:
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
            except Exception:
                free_bytes = 0
            if free_bytes >= MIN_FREE_CUDA_BYTES_FOR_PARITY:
                return CircuitTracerNNsightGemma2()
            pytest.skip(
                "Semantic intervention parity requires at least "
                f"{MIN_FREE_CUDA_BYTES_FOR_PARITY / 1024**3:.1f} GiB of free CUDA memory"
            )

        pytest.skip("Semantic intervention parity requires CUDA memory but CUDA is unavailable")

    @contextmanager
    def _factory(run_name: str):
        session_dir = tmp_path / run_name
        session_dir.mkdir(parents=True, exist_ok=True)
        clear_nnsight_test_state(None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        load_dotenv()
        it_session = config_modules(_build_test_cfg(), run_name, {}, session_dir, {}, False)
        session_fixture_hook_exec(it_session, cast(FixtPhase, FixtPhase.setup))
        module = it_session.module
        assert module is not None
        replacement_model = cast(Any, module).replacement_model
        with serial_test_cleanup(it_session, module, replacement_model):
            yield it_session

    return _factory


@pytest.fixture
def ct_nnsight_gemma3_it_session_factory(tmp_path, gemma3_instruction_intervention_case):
    case = gemma3_instruction_intervention_case

    def _build_test_cfg() -> CircuitTracerNNsightGemma3:
        return CircuitTracerNNsightGemma3(
            device_type="cpu",
            nnsight_cfg=NNsightConfig(
                model_name=case.model_name,
                device_map="cpu",
                torch_dtype="float32",
                dispatch=True,
                attn_implementation="eager",
                tokenizer_kwargs={"padding_side": "left", "add_bos_token": True},
            ),
            circuit_tracer_cfg=CircuitTracerConfig(
                backend="nnsight",
                model_name=case.model_name,
                transcoder_set=case.transcoder_set,
                dtype=torch.float32,
                analysis_target_tokens=None,
                target_token_ids=None,
                max_feature_nodes=None,
                offload=None,
                verbose=False,
                batch_size=256,
                max_n_logits=10,
                desired_logit_prob=0.95,
            ),
        )

    @contextmanager
    def _factory(run_name: str):
        session_dir = tmp_path / run_name
        session_dir.mkdir(parents=True, exist_ok=True)
        clear_nnsight_test_state(None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        load_dotenv()
        it_session = config_modules(_build_test_cfg(), run_name, {}, session_dir, {}, False)
        session_fixture_hook_exec(it_session, cast(FixtPhase, FixtPhase.setup))
        module = it_session.module
        assert module is not None
        replacement_model = cast(Any, module).replacement_model
        with serial_test_cleanup(it_session, module, replacement_model):
            yield it_session

    return _factory


def _build_demo_semantic_target(model, case: SemanticInterventionParityCase) -> tuple[CustomTarget, list[int]]:
    assert len(case.capitals) == len(case.states), "Groups must have equal length for paired rejection"

    tokenizer = model.tokenizer
    ids_a = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in case.capitals]
    ids_b = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in case.states]
    vecs_a = get_unembed_vecs(model, ids_a, "nnsight")
    vecs_b = get_unembed_vecs(model, ids_b, "nnsight")

    residuals = []
    for vec_a, vec_b in zip(vecs_a, vecs_b):
        vec_a_f = vec_a.float()
        vec_b_f = vec_b.float()
        projection = (vec_a_f @ vec_b_f) / (vec_b_f @ vec_b_f) * vec_b_f
        residuals.append((vec_a_f - projection).to(vec_a.dtype))

    direction = torch.stack(residuals).mean(0)
    input_ids = model.ensure_tokenized(case.prompt)
    logits, _ = model.get_activations(input_ids)
    probs = torch.softmax(logits.squeeze(0)[-1], dim=-1)
    avg_prob = max(sum(probs[index].item() for index in ids_a) / len(ids_a), 1e-6)
    return CustomTarget(token_str=case.label, prob=avg_prob, vec=direction), ids_a


def _top_features_from_graph(graph, n_top: int) -> list[tuple[int, int, int]]:
    feature_rows, _ = _top_feature_rows_and_scores_from_graph(graph, n_top)
    return feature_rows


def _top_feature_rows_and_scores_from_graph(
    graph,
    n_top: int,
    *,
    feature_selection: Any | None = None,
) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)
    if n_features == 0:
        return [], torch.empty((0,), dtype=torch.float32)
    logit_weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)[:n_features].detach().cpu().float()
    selected_feature_indices = graph.selected_features.detach().to(device=graph.active_features.device)
    feature_row_tensor = graph.active_features.index_select(0, selected_feature_indices).detach().cpu()
    if feature_selection is not None:
        selection_mask = apply_feature_selection_filter(feature_row_tensor, feature_selection)
        if selection_mask.numel() == 0 or not bool(selection_mask.any().item()):
            return [], torch.empty((0,), dtype=torch.float32)
        node_influence = node_influence[selection_mask]
        feature_row_tensor = feature_row_tensor[selection_mask]
        n_features = int(feature_row_tensor.shape[0])
    ranked_indices = torch.argsort(node_influence, descending=True)[: min(n_top, n_features)]
    top_scores = node_influence.index_select(0, ranked_indices)
    feature_rows = [tuple(feature_row_tensor[index].tolist()) for index in ranked_indices.tolist()]
    return feature_rows, top_scores.detach().cpu().float()


def _gap_from_logits(logits: torch.Tensor, target_id: int, baseline_id: int) -> float:
    final_logits = logits if logits.ndim == 1 else logits.squeeze(0)[-1]
    return float((final_logits[target_id] - final_logits[baseline_id]).cpu().item())


def _build_native_baseline(it_session, case: SemanticInterventionParityCase) -> NativeSemanticBaseline:
    module = it_session.module
    native_target, native_group_a_ids = _build_demo_semantic_target(module.replacement_model, case)
    austin_id = module.replacement_model.tokenizer.encode("▁Austin", add_special_tokens=False)[-1]
    dallas_id = module.replacement_model.tokenizer.encode("▁Dallas", add_special_tokens=False)[-1]

    native_graph = module.generate_attribution_graph(case.prompt, attribution_targets=[native_target])
    native_top_features = _top_features_from_graph(native_graph, case.n_top)
    native_input_ids = module.replacement_model.ensure_tokenized(case.prompt)
    native_pre_logits, native_acts = module.replacement_model.get_activations(native_input_ids, sparse=True)
    native_interventions = [
        (layer, position, feature, 10.0 * float(native_acts[layer, position, feature].item()))
        for layer, position, feature in native_top_features
    ]
    native_post_logits, _ = module.replacement_model.feature_intervention(case.prompt, native_interventions)
    native_pre_final = native_pre_logits.detach().cpu().squeeze(0)[-1]
    native_post_final = native_post_logits.detach().cpu().squeeze(0)[-1]

    return NativeSemanticBaseline(
        group_a_ids=native_group_a_ids,
        top_features=native_top_features,
        interventions=native_interventions,
        activation_values=torch.tensor(
            [float(native_acts[layer, position, feature]) for layer, position, feature in native_top_features],
            dtype=torch.float32,
        ),
        pre_logits=native_pre_final.to(torch.float32),
        post_logits=native_post_final.to(torch.float32),
        pre_gap=_gap_from_logits(native_pre_final, austin_id, dallas_id),
        post_gap=_gap_from_logits(native_post_final, austin_id, dallas_id),
        concept_direction=native_target.vec.detach().cpu().to(torch.float32),
    )


def _build_op_baseline(it_session, case: SemanticInterventionParityCase) -> OpSemanticBaseline:
    module = it_session.module
    assert module is not None
    austin_id = module.replacement_model.tokenizer.encode("▁Austin", add_special_tokens=False)[-1]
    dallas_id = module.replacement_model.tokenizer.encode("▁Dallas", add_special_tokens=False)[-1]

    concept_op = DISPATCHER.get_op("concept_direction")
    graph_op = DISPATCHER.get_op("compute_attribution_graph")
    influence_op = DISPATCHER.get_op("graph_node_influence")
    top_features_op = DISPATCHER.get_op("extract_top_features")
    intervention_op = DISPATCHER.get_op("feature_intervention_forward")

    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = 10.0
    module.analysis_cfg = AnalysisCfg(target_op=graph_op, ignore_manual=True, save_tokens=False)
    if not module.analysis_cfg.applied_to(module):
        init_analysis_cfgs(module, [module.analysis_cfg])

    concept_result = cast(
        Any,
        concept_op(
            module,
            AnalysisBatch(
                concept_group_a=case.capitals,
                concept_group_b=case.states,
                concept_label=case.label,
                concept_direction_mode="paired_rejection",
            ),
            None,
            0,
        ),
    )
    graph_result = cast(
        Any,
        graph_op(
            module,
            AnalysisBatch(
                prompts=[case.prompt],
                concept_direction=concept_result.concept_direction,
                concept_label=concept_result.concept_label,
                concept_group_a_token_ids=concept_result.concept_group_a_token_ids,
                concept_group_b_token_ids=concept_result.concept_group_b_token_ids,
                concept_direction_mode=concept_result.concept_direction_mode,
            ),
            None,
            0,
        ),
    )
    influence_result = cast(Any, influence_op(module, graph_result, None, 0))
    top_features_payload = dict(cast(Any, graph_result))
    top_features_payload.update(dict(cast(Any, influence_result)))
    top_features_result = cast(
        Any,
        top_features_op(
            module,
            AnalysisBatch(**top_features_payload),
            None,
            0,
            top_n=case.n_top,
        ),
    )
    intervention_result = cast(
        Any,
        intervention_op(
            module,
            AnalysisBatch(
                prompts=[case.prompt],
                top_feature_ids=top_features_result.top_feature_ids,
                top_feature_scores=top_features_result.top_feature_scores,
                top_feature_activation_values=top_features_result.top_feature_activation_values,
                logit_target_ids=torch.tensor([austin_id], dtype=torch.long),
            ),
            None,
            0,
        ),
    )

    pre_final = intervention_result.pre_intervention_logits.float().cpu()
    post_final = intervention_result.post_intervention_logits.float().cpu()
    return OpSemanticBaseline(
        direction_mode=concept_result.concept_direction_mode,
        group_a_token_ids=concept_result.concept_group_a_token_ids,
        concept_direction=concept_result.concept_direction.float().cpu(),
        top_feature_ids=[tuple(row.tolist()) for row in top_features_result.top_feature_ids],
        activation_values=top_features_result.top_feature_activation_values.float().cpu(),
        intervention_feature_ids=intervention_result.intervention_feature_ids,
        intervention_positions=intervention_result.intervention_positions,
        intervention_values=torch.tensor(intervention_result.intervention_values, dtype=torch.float32),
        pre_logits=pre_final,
        post_logits=post_final,
        pre_gap=float((pre_final[austin_id] - pre_final[dallas_id]).item()),
        post_gap=float((post_final[austin_id] - post_final[dallas_id]).item()),
    )


def _ensure_analysis_cfg(module: Any, target_op: Any) -> None:
    module.analysis_cfg = AnalysisCfg(target_op=target_op, ignore_manual=True, save_tokens=False)
    if not module.analysis_cfg.applied_to(module):
        init_analysis_cfgs(module, [module.analysis_cfg])


@contextmanager
def _raw_gemma3_it_model(case: Gemma3InstructionInterventionCase):
    clear_nnsight_test_state(None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = ReplacementModel.from_pretrained(
        case.model_name,
        case.transcoder_set,
        dtype=torch.float32,
        backend="nnsight",
    )
    assert isinstance(model, NNSightReplacementModel)
    with serial_test_cleanup(model):
        yield cast(NNSightReplacementModel, model)


def _select_evenly_spaced_values(values: range | list[int], limit: int | None) -> list[int]:
    candidates = list(values)
    if limit is None or limit >= len(candidates):
        return candidates
    if limit <= 0 or not candidates:
        return []
    index_tensor = torch.linspace(0, len(candidates) - 1, steps=limit).round().to(torch.long)
    selected: list[int] = []
    seen: set[int] = set()
    for index in index_tensor.tolist():
        resolved = int(index)
        if resolved in seen:
            continue
        selected.append(candidates[resolved])
        seen.add(resolved)
    for value in candidates:
        if len(selected) >= limit:
            break
        if value not in selected:
            selected.append(value)
    return selected


def _format_feature_row(feature_row: tuple[int, int, int]) -> str:
    return f"(layer={feature_row[0]}, pos={feature_row[1]}, feature={feature_row[2]})"


def _feature_row_from_tensor(feature_row_tensor: torch.Tensor) -> tuple[int, int, int]:
    values = feature_row_tensor.detach().cpu().tolist()
    assert len(values) == 3
    return (int(values[0]), int(values[1]), int(values[2]))


def _find_active_feature_index(active_features: torch.Tensor, feature_row: tuple[int, int, int]) -> int:
    match_mask = (
        (active_features[:, 0] == feature_row[0])
        & (active_features[:, 1] == feature_row[1])
        & (active_features[:, 2] == feature_row[2])
    )
    matches = torch.nonzero(match_mask, as_tuple=True)[0]
    assert matches.numel() == 1, f"Expected exactly one active feature row match for {_format_feature_row(feature_row)}"
    return int(matches.item())


def _assert_gemma3_it_model_loaded(model: NNSightReplacementModel, prompt: str) -> None:
    tokenized = model.ensure_tokenized(prompt)
    tokens = tokenized[0] if tokenized.ndim > 1 else tokenized
    ignore_prefix = torch.tensor(GEMMA3_IT_PREFIX_TOKEN_IDS, dtype=tokens.dtype, device=tokens.device)
    assert isinstance(model, NNSightReplacementModel)
    assert getattr(model, "zero_positions", None) == slice(0, 4)
    assert tokens.size(0) >= 4 and torch.all(tokens[:4] == ignore_prefix), (
        "Gemma 3 IT prompt tokenization did not preserve the expected "
        f"<bos><start_of_turn>user\\n prefix: {tokens[:4].detach().cpu().tolist()}"
    )


def _build_graph_edge_validation_context(
    model: NNSightReplacementModel,
    graph: Graph,
    *,
    selected_only: bool = False,
) -> GraphEdgeValidationContext:
    prompt = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(device=model.device, dtype=model.dtype)
    active_features = graph.active_features.to(device=model.device)
    if selected_only and len(graph.selected_features) > 0:
        selected_indices = graph.selected_features.to(device=active_features.device, dtype=torch.long)
        active_features = active_features.index_select(0, selected_indices)
    logit_tokens = graph.logit_tokens.to(device=model.device)
    logits, activation_cache = model.get_activations(prompt, apply_activation_function=False)
    logits = logits.squeeze(0)
    relevant_activations = activation_cache[active_features[:, 0], active_features[:, 1], active_features[:, 2]]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()
    return GraphEdgeValidationContext(
        prompt=prompt,
        adjacency_matrix=adjacency_matrix,
        active_features=active_features,
        logit_tokens=logit_tokens,
        total_active_features=int(active_features.size(0)),
        activation_cache=activation_cache,
        relevant_activations=relevant_activations,
        demeaned_relevant_logits=demeaned_relevant_logits,
    )


def _assert_effect_matches_expected(
    *,
    label: str,
    context: GraphEdgeValidationContext,
    new_relevant_activations: torch.Tensor,
    new_logits: torch.Tensor,
    expected_effects: torch.Tensor,
    act_atol: float,
    act_rtol: float,
    logit_atol: float,
    logit_rtol: float,
) -> EdgeValidationSummary:
    last_logits = (new_logits if new_logits.dim() == 1 else new_logits[-1]).detach().float().cpu()
    new_relevant_activations = new_relevant_activations.detach().float().cpu()
    active_features = context.active_features.detach().cpu()
    logit_tokens = context.logit_tokens.detach().long().cpu()
    expected_effects = expected_effects.detach().float().cpu()
    relevant_activations = context.relevant_activations.detach().float().cpu()
    demeaned_relevant_logits = context.demeaned_relevant_logits.detach().float().cpu()

    new_relevant_logits = last_logits[logit_tokens]
    new_demeaned_relevant_logits = new_relevant_logits - last_logits.mean()

    expected_activation_difference = expected_effects[: context.total_active_features]
    expected_logit_difference = expected_effects[-len(logit_tokens) :]
    expected_activations = relevant_activations + expected_activation_difference
    expected_logits = demeaned_relevant_logits + expected_logit_difference

    act_error = (new_relevant_activations - expected_activations).abs().detach().float().cpu()
    logit_error = (new_demeaned_relevant_logits - expected_logits).abs().detach().float().cpu()
    act_top_indices = torch.topk(act_error, min(3, int(act_error.numel()))).indices.tolist()
    logit_top_indices = torch.topk(logit_error, min(3, int(logit_error.numel()))).indices.tolist()
    act_preview = "; ".join(
        f"{_format_feature_row(cast(tuple[int, int, int], _feature_row_from_tensor(active_features[index])))} "
        f"err={act_error[index].item():.3e}"
        for index in act_top_indices
    )
    logit_preview = "; ".join(
        f"token_id={int(logit_tokens[index].item())} err={logit_error[index].item():.3e}" for index in logit_top_indices
    )

    assert torch.allclose(
        new_relevant_activations,
        expected_activations,
        atol=act_atol,
        rtol=act_rtol,
    ), f"{label}: activation parity mismatch max={act_error.max().item():.3e}; top={act_preview}"
    assert torch.allclose(
        new_demeaned_relevant_logits,
        expected_logits,
        atol=logit_atol,
        rtol=logit_rtol,
    ), f"{label}: logit parity mismatch max={logit_error.max().item():.3e}; top={logit_preview}"

    return EdgeValidationSummary(
        label=label,
        max_activation_abs_error=float(act_error.max().item()),
        max_logit_abs_error=float(logit_error.max().item()),
    )


def _verify_token_and_error_edges(
    model: NNSightReplacementModel,
    graph: Graph,
    *,
    pos_start: int,
    token_position_limit: int | None = None,
    error_layer_limit: int | None = None,
    act_atol: float = TOKEN_EDGE_ACT_ATOL,
    act_rtol: float = TOKEN_EDGE_ACT_RTOL,
    logit_atol: float = EDGE_LOGIT_ATOL,
    logit_rtol: float = EDGE_LOGIT_RTOL,
) -> list[EdgeValidationSummary]:
    context = _build_graph_edge_validation_context(model, graph)
    attribution_ctx = model.setup_attribution(context.prompt)
    error_vectors = attribution_ctx.error_vectors
    token_vectors = attribution_ctx.token_vectors
    error_positions = _select_evenly_spaced_values(range(pos_start, error_vectors.size(1)), token_position_limit)
    token_positions = _select_evenly_spaced_values(range(pos_start, token_vectors.size(0)), token_position_limit)
    error_layers = _select_evenly_spaced_values(range(error_vectors.size(0)), error_layer_limit)
    summaries: list[EdgeValidationSummary] = []

    def _verify_intervention(
        expected_effects: torch.Tensor,
        intervention: Any,
        target_layer: int | str,
        label: str,
    ) -> None:
        _, freeze_fns = model.setup_intervention_with_freeze(
            context.prompt,
            constrained_layers=range(model.cfg.n_layers),  # type: ignore[arg-type]
        )
        _, activation_fn = model.get_activation_fn(apply_activation_function=False)

        with model.trace() as tracer:
            with tracer.invoke(context.prompt):
                pass

            direct_effects_barrier = tracer.barrier(2)

            with tracer.invoke():
                _, new_activation_cache = activation_fn()  # type: ignore[misc]
                new_activation_cache = save(new_activation_cache)  # type: ignore[assignment]

            for freeze_fn in freeze_fns:
                with tracer.invoke():
                    freeze_fn(direct_effects_barrier=direct_effects_barrier)

            with tracer.invoke():
                if target_layer == "embed":
                    intervention(model.embed_location)

                for layer, feature_output_loc in enumerate(model.feature_output_locs):
                    direct_effects_barrier()
                    if layer == target_layer:
                        intervention(feature_output_loc)

                new_logits = save(model.output.logits.squeeze(0))  # type: ignore[arg-type]

        new_relevant_activations = new_activation_cache[
            context.active_features[:, 0],
            context.active_features[:, 1],
            context.active_features[:, 2],
        ]
        summaries.append(
            _assert_effect_matches_expected(
                label=label,
                context=context,
                new_relevant_activations=new_relevant_activations,
                new_logits=new_logits,
                expected_effects=expected_effects,
                act_atol=act_atol,
                act_rtol=act_rtol,
                logit_atol=logit_atol,
                logit_rtol=logit_rtol,
            )
        )

    for error_node_layer in error_layers:
        for error_node_pos in error_positions:
            error_node_index = error_node_layer * error_vectors.size(1) + error_node_pos
            expected_effects = context.adjacency_matrix[:, context.total_active_features + error_node_index]

            def error_intervention(feature_output_loc, *, error_node_layer: int, error_node_pos: int):
                activations = feature_output_loc.output
                steering_vector = torch.zeros_like(activations)
                steering_vector[:, error_node_pos] += error_vectors[error_node_layer, error_node_pos]
                feature_output_loc.output = activations + steering_vector

            _verify_intervention(
                expected_effects,
                partial(error_intervention, error_node_layer=error_node_layer, error_node_pos=error_node_pos),
                error_node_layer,
                f"error_edge[layer={error_node_layer}, pos={error_node_pos}]",
            )

    total_error_nodes = error_vectors.size(0) * error_vectors.size(1)
    for token_pos in token_positions:
        expected_effects = context.adjacency_matrix[:, context.total_active_features + total_error_nodes + token_pos]

        def token_intervention(token_loc, *, token_pos: int):
            activations = token_loc.output
            steering_vector = torch.zeros_like(activations)
            steering_vector[:, token_pos] += token_vectors[token_pos]
            token_loc.output = activations + steering_vector

        _verify_intervention(
            expected_effects,
            partial(token_intervention, token_pos=token_pos),
            "embed",
            f"token_edge[pos={token_pos}]",
        )

    return summaries


def _verify_feature_edges_direct(
    model: NNSightReplacementModel,
    graph: Graph,
    *,
    feature_rows: list[tuple[int, int, int]] | None = None,
    n_samples: int | None = None,
    value_scale_factor: float = 2.0,
    selected_only: bool = False,
    act_atol: float = FEATURE_EDGE_ACT_ATOL,
    act_rtol: float = FEATURE_EDGE_ACT_RTOL,
    logit_atol: float = EDGE_LOGIT_ATOL,
    logit_rtol: float = EDGE_LOGIT_RTOL,
) -> list[EdgeValidationSummary]:
    context = _build_graph_edge_validation_context(model, graph, selected_only=selected_only)
    if feature_rows is None:
        sample_count = min(n_samples or int(context.active_features.size(0)), int(context.active_features.size(0)))
        chosen_nodes = torch.randperm(
            context.active_features.size(0),
            device=context.active_features.device,
        )[:sample_count]
    else:
        chosen_nodes = torch.tensor(
            [_find_active_feature_index(context.active_features, feature_row) for feature_row in feature_rows],
            device=context.active_features.device,
            dtype=torch.long,
        )

    summaries: list[EdgeValidationSummary] = []
    for chosen_node in chosen_nodes.tolist():
        layer, position, feature_id = (int(value) for value in context.active_features[chosen_node].tolist())
        old_activation = context.activation_cache[layer, position, feature_id]
        new_activation = float((old_activation * value_scale_factor).item())
        expected_effects = context.adjacency_matrix[:, chosen_node]
        new_logits, new_activation_cache = model.feature_intervention(
            context.prompt,
            [(layer, position, feature_id, new_activation)],
            constrained_layers=range(model.cfg.n_layers),  # type: ignore[arg-type]
            apply_activation_function=False,
        )
        new_relevant_activations = new_activation_cache[
            context.active_features[:, 0],
            context.active_features[:, 1],
            context.active_features[:, 2],
        ]
        summaries.append(
            _assert_effect_matches_expected(
                label=f"feature_edge[layer={layer}, pos={position}, feature={feature_id}]",
                context=context,
                new_relevant_activations=new_relevant_activations,
                new_logits=new_logits.squeeze(0),
                expected_effects=expected_effects,
                act_atol=act_atol,
                act_rtol=act_rtol,
                logit_atol=logit_atol,
                logit_rtol=logit_rtol,
            )
        )

    return summaries


def _configure_gemma3_it_op_settings(module: Any, case: Gemma3InstructionInterventionCase) -> None:
    cfg = module.circuit_tracer_cfg
    cfg.model_name = case.model_name
    cfg.transcoder_set = case.transcoder_set
    cfg.dtype = torch.float32
    cfg.analysis_target_tokens = None
    cfg.target_token_ids = None
    cfg.max_feature_nodes = None
    cfg.offload = None
    cfg.verbose = False
    cfg.batch_size = 256
    cfg.max_n_logits = 10
    cfg.desired_logit_prob = 0.95
    cfg.intervention_value_source = "top_feature_activation_values"
    cfg.intervention_scale_factor = case.intervention_scale_factor
    cfg.intervention_constrained_layers = list(range(module.replacement_model.cfg.n_layers))
    cfg.intervention_apply_activation_function = False
    cfg.intervention_freeze_attention = None
    cfg.intervention_sparse = False
    cfg.intervention_return_activations = False


def _verify_wrapper_feature_interventions(
    module: Any,
    prompt: str,
    graph: Graph,
    graph_context: GraphEdgeValidationContext,
    top_features_result: Any,
    *,
    require_direct_edge_parity: bool = True,
) -> list[WrapperFeatureInterventionSummary]:
    analysis_backend = require_analysis_backend(module)
    settings = analysis_backend.resolve_feature_intervention_settings(module)
    feature_rows = torch.as_tensor(top_features_result.top_feature_ids, dtype=torch.long)
    feature_scores = torch.as_tensor(top_features_result.top_feature_scores, dtype=torch.float32)
    activation_values = torch.as_tensor(top_features_result.top_feature_activation_values, dtype=torch.float32)
    logit_target_ids = graph.logit_tokens.detach().cpu()
    baseline_pre_logits, _ = module.replacement_model.get_activations(prompt)
    baseline_pre_final = last_token_logits(baseline_pre_logits.float()).cpu()
    summaries: list[WrapperFeatureInterventionSummary] = []

    for index, feature_row_tensor in enumerate(feature_rows):
        feature_row = cast(tuple[int, int, int], _feature_row_from_tensor(feature_row_tensor))
        analysis_batch = AnalysisBatch(
            prompts=[prompt],
            top_feature_ids=feature_row_tensor.unsqueeze(0),
            top_feature_scores=feature_scores[index : index + 1],
            top_feature_activation_values=activation_values[index : index + 1],
            logit_target_ids=logit_target_ids,
        )
        interventions, _ = analysis_backend.build_feature_interventions(analysis_batch, settings)
        wrapper_result = cast(
            Any,
            it.feature_intervention_forward(
                module,
                analysis_batch,
                batch=cast(Any, None),
                batch_idx=0,
                intervention_return_activations=require_direct_edge_parity,
            ),
        )
        wrapper_pre = wrapper_result.pre_intervention_logits.float().cpu()
        wrapper_post = wrapper_result.post_intervention_logits.float().cpu()

        assert_close(wrapper_pre, baseline_pre_final, rtol=VALUE_RTOL, atol=VALUE_ATOL)
        assert wrapper_result.intervention_feature_ids == [feature_row[2]]
        assert wrapper_result.intervention_positions == [feature_row[1]]
        assert_close(
            torch.tensor(wrapper_result.intervention_values, dtype=torch.float32),
            torch.tensor([interventions[0][3]], dtype=torch.float32),
            rtol=VALUE_RTOL,
            atol=VALUE_ATOL,
        )
        chosen_node = _find_active_feature_index(graph_context.active_features, feature_row)
        returned_activation_cache = getattr(wrapper_result, "intervention_activation_cache", None) is not None
        edge_max_logit_abs_error = float("nan")
        if require_direct_edge_parity:
            assert returned_activation_cache
            wrapper_activation_cache = (
                torch.as_tensor(
                    wrapper_result.intervention_activation_cache,
                    dtype=torch.float32,
                )
                .detach()
                .cpu()
            )
            active_feature_rows = graph_context.active_features.detach().cpu()
            wrapper_relevant_activations = wrapper_activation_cache[
                active_feature_rows[:, 0],
                active_feature_rows[:, 1],
                active_feature_rows[:, 2],
            ]
            edge_summary = _assert_effect_matches_expected(
                label=f"op_wrapper_feature_edge{_format_feature_row(feature_row)}",
                context=graph_context,
                new_relevant_activations=wrapper_relevant_activations,
                new_logits=wrapper_post,
                expected_effects=graph_context.adjacency_matrix[:, chosen_node],
                act_atol=FEATURE_EDGE_ACT_ATOL,
                act_rtol=FEATURE_EDGE_ACT_RTOL,
                logit_atol=EDGE_LOGIT_ATOL,
                logit_rtol=EDGE_LOGIT_RTOL,
            )
            edge_max_logit_abs_error = edge_summary.max_logit_abs_error
        summaries.append(
            WrapperFeatureInterventionSummary(
                feature_row=feature_row,
                intervention_value=float(interventions[0][3]),
                edge_max_logit_abs_error=edge_max_logit_abs_error,
                returned_activation_cache=returned_activation_cache,
            )
        )

    return summaries


@RUNIF(bf16_cuda=True, standalone=True)
def test_analysis_backend_parity_semantic_intervention_nnsight(
    cleanup_cuda,
    ct_nnsight_session_factory,
    semantic_intervention_parity_case,
):
    """Serial in-process parity between the native CT NNsight path and the analysis-op path.

    This mirrors the upstream circuit-tracer semantic intervention workflow but uses fresh Interpretune NNsight sessions
    for each phase so heavy state is released between the native and analysis-op baselines.
    """

    with ct_nnsight_session_factory("ct_native_parity_serial") as native_session:
        native_baseline = _build_native_baseline(native_session, semantic_intervention_parity_case)

    with ct_nnsight_session_factory("ct_op_parity_serial") as op_session:
        op_baseline = _build_op_baseline(op_session, semantic_intervention_parity_case)

    assert op_baseline.direction_mode == "paired_rejection"
    assert op_baseline.group_a_token_ids == native_baseline.group_a_ids

    concept_cosine = torch.nn.functional.cosine_similarity(
        op_baseline.concept_direction.unsqueeze(0),
        native_baseline.concept_direction.unsqueeze(0),
    ).item()
    assert concept_cosine > CONCEPT_DIRECTION_COSINE_MIN

    assert op_baseline.top_feature_ids == native_baseline.top_features
    assert_close(op_baseline.activation_values, native_baseline.activation_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert op_baseline.intervention_feature_ids == [feature for _, _, feature, _ in native_baseline.interventions]
    assert op_baseline.intervention_positions == [position for _, position, _, _ in native_baseline.interventions]
    assert_close(
        op_baseline.intervention_values,
        torch.tensor([value for _, _, _, value in native_baseline.interventions], dtype=torch.float32),
        rtol=VALUE_RTOL,
        atol=VALUE_ATOL,
    )
    assert_close(op_baseline.pre_logits, native_baseline.pre_logits, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert_close(op_baseline.post_logits, native_baseline.post_logits, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert native_baseline.post_gap > native_baseline.pre_gap
    assert op_baseline.post_gap > op_baseline.pre_gap


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_it_reference_intervention_graph(
    cleanup_cuda,
    gemma3_instruction_intervention_case,
):
    """Reference-only upstream-style Gemma 3 IT intervention graph check.

    This mirrors circuit-tracer's direct Gemma 3 IT NNsight path as closely as possible before any Interpretune
    session/config machinery is involved.
    """

    with _raw_gemma3_it_model(gemma3_instruction_intervention_case) as model:
        _assert_gemma3_it_model_loaded(model, gemma3_instruction_intervention_case.prompt)
        graph = attribute(gemma3_instruction_intervention_case.prompt, model)
        with model.zero_softcap():
            _verify_token_and_error_edges(
                model,
                graph,
                pos_start=gemma3_instruction_intervention_case.pos_start,
            )
            _verify_feature_edges_direct(
                model,
                graph,
                n_samples=gemma3_instruction_intervention_case.reference_feature_sample_count,
                value_scale_factor=gemma3_instruction_intervention_case.intervention_scale_factor,
            )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_it_direct_session_intervention_graph(
    cleanup_cuda,
    ct_nnsight_gemma3_it_session_factory,
    gemma3_instruction_intervention_case,
):
    """Interpretune session parity stage using direct circuit-tracer graph/intervention APIs."""

    with ct_nnsight_gemma3_it_session_factory("ct_gemma3_it_direct_session") as it_session:
        module = cast(Any, it_session.module)
        with clean_cuda(module.replacement_model):
            _assert_gemma3_it_model_loaded(module.replacement_model, gemma3_instruction_intervention_case.prompt)
            graph = attribute(gemma3_instruction_intervention_case.prompt, module.replacement_model)
            top_features = _top_features_from_graph(graph, gemma3_instruction_intervention_case.feature_sample_count)
            with module.replacement_model.zero_softcap():
                _verify_token_and_error_edges(
                    module.replacement_model,
                    graph,
                    pos_start=gemma3_instruction_intervention_case.pos_start,
                    token_position_limit=gemma3_instruction_intervention_case.token_position_limit,
                    error_layer_limit=gemma3_instruction_intervention_case.error_layer_limit,
                )
                _verify_feature_edges_direct(
                    module.replacement_model,
                    graph,
                    feature_rows=top_features,
                    value_scale_factor=gemma3_instruction_intervention_case.intervention_scale_factor,
                )


@RUNIF(min_cuda_gpus=1, optional=True)
def test_analysis_backend_parity_gemma3_it_op_intervention_graph(
    cleanup_cuda,
    ct_nnsight_gemma3_it_session_factory,
    gemma3_instruction_intervention_case,
):
    """Interpretune op-layer parity stage for Gemma 3 IT graph and feature interventions."""

    with ct_nnsight_gemma3_it_session_factory("ct_gemma3_it_op_session") as it_session:
        module = cast(Any, it_session.module)
        _configure_gemma3_it_op_settings(module, gemma3_instruction_intervention_case)
        _ensure_analysis_cfg(module, it.compute_attribution_graph)

        with clean_cuda(module.replacement_model):
            _assert_gemma3_it_model_loaded(module.replacement_model, gemma3_instruction_intervention_case.prompt)
            graph_result = cast(
                Any,
                it.compute_attribution_graph(
                    module,
                    AnalysisBatch(prompts=[gemma3_instruction_intervention_case.prompt]),
                    batch=None,
                    batch_idx=0,
                ),
            )
            graph = require_analysis_backend(module).hydrate_graph_from_batch(graph_result)
            graph_context = _build_graph_edge_validation_context(module.replacement_model, graph)
            influence_result = cast(Any, it.graph_node_influence(module, graph_result, batch=None, batch_idx=0))
            top_payload = dict(cast(Any, graph_result))
            top_payload.update(dict(cast(Any, influence_result)))
            top_features_result = cast(
                Any,
                it.extract_top_features(
                    module,
                    AnalysisBatch(**top_payload),
                    batch=None,
                    batch_idx=0,
                    top_n=gemma3_instruction_intervention_case.feature_sample_count,
                ),
            )

            expected_top_features = _top_features_from_graph(
                graph,
                gemma3_instruction_intervention_case.feature_sample_count,
            )
            actual_top_features = [tuple(row.tolist()) for row in torch.as_tensor(top_features_result.top_feature_ids)]
            assert actual_top_features == expected_top_features
            expected_activation_values = torch.tensor(
                [
                    float(graph_context.activation_cache[layer, position, feature_id].item())
                    for layer, position, feature_id in actual_top_features
                ],
                dtype=torch.float32,
            )
            assert_close(
                torch.as_tensor(top_features_result.top_feature_activation_values, dtype=torch.float32).cpu(),
                expected_activation_values,
                rtol=VALUE_RTOL,
                atol=VALUE_ATOL,
            )

            with module.replacement_model.zero_softcap():
                _verify_token_and_error_edges(
                    module.replacement_model,
                    graph,
                    pos_start=gemma3_instruction_intervention_case.pos_start,
                    token_position_limit=gemma3_instruction_intervention_case.token_position_limit,
                    error_layer_limit=gemma3_instruction_intervention_case.error_layer_limit,
                )
                _verify_feature_edges_direct(
                    module.replacement_model,
                    graph,
                    feature_rows=actual_top_features,
                    value_scale_factor=gemma3_instruction_intervention_case.intervention_scale_factor,
                )
                _verify_wrapper_feature_interventions(
                    module,
                    gemma3_instruction_intervention_case.prompt,
                    graph,
                    graph_context,
                    top_features_result,
                )
