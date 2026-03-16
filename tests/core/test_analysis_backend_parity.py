from __future__ import annotations

import json
import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import pytest
import torch
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.graph import compute_node_influence
from circuit_tracer.utils.demo_utils import get_unembed_vecs
from torch.testing import assert_close

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.config import AnalysisCfg, NNsightConfig, init_analysis_cfgs
from tests import load_dotenv
from tests.analysis_resource_utils import clear_nnsight_test_state, serial_test_cleanup
from tests.configuration import config_modules
from tests.conftest import FixtPhase, session_fixture_hook_exec
from tests.core.cfg_aliases import CircuitTracerNNsightGemma2
from tests.runif import RunIf


CONCEPT_DIRECTION_COSINE_MIN = 0.999
VALUE_RTOL = 1e-6
VALUE_ATOL = 1e-6
MIN_FREE_CUDA_BYTES_FOR_PARITY = 6 * 1024**3


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
def ct_nnsight_session_factory(tmp_path):
    def _build_test_cfg() -> CircuitTracerNNsightGemma2:
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
            except Exception:
                free_bytes = 0
            if free_bytes >= MIN_FREE_CUDA_BYTES_FOR_PARITY:
                return CircuitTracerNNsightGemma2()

        return CircuitTracerNNsightGemma2(
            device_type="cpu",
            nnsight_cfg=NNsightConfig(
                model_name="google/gemma-2-2b",
                device_map="cpu",
                torch_dtype="float32",
                dispatch=True,
                attn_implementation="eager",
                tokenizer_kwargs={"padding_side": "left", "add_bos_token": True},
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
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)
    logit_weights = torch.zeros(graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device)
    logit_weights[-n_logits:] = graph.logit_probabilities
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    _, top_idx = torch.topk(node_influence[:n_features], min(n_top, n_features))
    return [tuple(graph.active_features[graph.selected_features[index]].tolist()) for index in top_idx]


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
                concept_metadata=concept_result.concept_metadata,
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

    concept_metadata = concept_result.concept_metadata
    if isinstance(concept_metadata, str):
        concept_metadata = json.loads(concept_metadata)

    pre_final = intervention_result.pre_intervention_logits.float().cpu()
    post_final = intervention_result.post_intervention_logits.float().cpu()
    return OpSemanticBaseline(
        direction_mode=concept_metadata["direction_mode"],
        group_a_token_ids=concept_metadata["group_a_token_ids"],
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


@RunIf(bf16_cuda=True)
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
