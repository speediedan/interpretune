"""Synthetic analysis-backend intervention tests.

These currently exercise the circuit-tracer analysis backend because it is the only native analysis backend with
intervention support. The fixtures are intentionally backend-shaped so additional analysis backends can reuse the same
test surface later.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import torch
from datasets import Dataset, load_from_disk
from transformers import BatchEncoding

from interpretune.analysis.backends import FeatureSelectionSpec, InterventionDict
from interpretune.analysis.backends.circuit_tracer import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.analysis.core import AnalysisStore, schema_to_features
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.definitions import feature_intervention_forward_impl
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.analysis.ops.helpers import mean_target_logit_delta
from interpretune.config.circuit_tracer import CircuitTracerConfig


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "decoded prompt"


class _FakeReplacementModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[tuple[int, int, int, float]], dict[str, object]]] = []
        self.tokenizer = _FakeTokenizer()

    def get_activations(self, prompt: str | torch.Tensor, **kwargs):
        if isinstance(prompt, str):
            assert prompt == "Paris Austin"
        return torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32), None

    def feature_intervention(self, prompt: str, interventions, **kwargs):
        self.calls.append((prompt, list(interventions), dict(kwargs)))
        total_delta = sum(value for _, _, _, value in interventions)
        activation_cache = None
        if kwargs.get("return_activations"):
            activation_cache = torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0 + total_delta],
                    ],
                ],
                dtype=torch.float32,
            )
        return torch.tensor([[[0.1, 0.2, 0.3 + total_delta, 0.4]]], dtype=torch.float32), activation_cache


class _FakeModule:
    def __init__(self) -> None:
        self.replacement_model = _FakeReplacementModel()
        self.datamodule = type(
            "_DataModule",
            (),
            {
                "tokenizer": self.replacement_model.tokenizer,
                "itdm_cfg": type("_ITDMCfg", (), {"eval_batch_size": 1})(),
            },
        )()
        self._analysis_backend = DEFAULT_CT_ANALYSIS_BACKEND
        self.model = type(
            "_Model",
            (),
            {"tokenizer": type("_TokenizerMeta", (), {"vocab_size": 4, "model_max_length": 16})()},
        )()
        self.it_cfg = type(
            "_ITCfg",
            (),
            {
                "generative_step_cfg": type(
                    "_GenCfg", (), {"lm_generation_cfg": type("_LMGenCfg", (), {"max_new_tokens": 1})()}
                )(),
                "num_labels": 2,
                "entailment_mapping": {"a": 0, "b": 1},
            },
        )()
        self.circuit_tracer_cfg = CircuitTracerConfig(
            backend="transformerlens",
            intervention_scale_factor=2.0,
            intervention_sparse=True,
            intervention_return_activations=False,
            intervention_constrained_layers=[0, 1],
        )

    @property
    def analysis_backend(self):
        return self._analysis_backend


def test_feature_intervention_forward_impl_serializes_and_executes_interventions() -> None:
    module = _FakeModule()
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11], [0, 1, 7]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5, -0.25], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )

    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)

    assert module.replacement_model.calls == [
        (
            "Paris Austin",
            [(1, 2, 11, 1.0), (0, 1, 7, -0.5)],
            {"sparse": True, "return_activations": False, "constrained_layers": [0, 1]},
        )
    ]
    assert result.intervention_layers == [1, 0]
    assert result.intervention_positions == [2, 1]
    assert result.intervention_feature_ids == [11, 7]
    assert result.intervention_values == [1.0, -0.5]
    assert isinstance(result.feature_intervention_dict, InterventionDict)
    assert list(result.feature_intervention_dict.keys()) == [
        "feature.layer.1.position.2.id.11",
        "feature.layer.0.position.1.id.7",
    ]
    assert torch.isclose(
        result.feature_intervention_dict["feature.layer.1.position.2.id.11"][0].intervention_tensor,
        torch.tensor(1.0),
    )
    intervention_dict_payload = json.loads(result.feature_intervention_dict_json)
    assert intervention_dict_payload["feature.layer.0.position.1.id.7"][0]["intervention_tensor"] == -0.5
    assert torch.allclose(result.pre_intervention_logits, torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32))
    assert torch.allclose(result.post_intervention_logits, torch.tensor([0.1, 0.2, 0.8, 0.4], dtype=torch.float32))
    assert torch.isclose(result.logit_diff, torch.tensor(0.5, dtype=torch.float32))


def test_feature_intervention_store_round_trip_hydrates_intervention_specs(tmp_path) -> None:
    module = _FakeModule()
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )
    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)
    output_schema = cast(Any, DISPATCHER.get_op("feature_intervention_forward")).output_schema
    dataset = Dataset.from_dict(
        {
            "intervention_config": [result.intervention_config],
            "intervention_specs_json": [result.intervention_specs_json],
            "feature_intervention_dict_json": [result.feature_intervention_dict_json],
            "intervention_layers": [result.intervention_layers],
            "intervention_positions": [result.intervention_positions],
            "intervention_feature_ids": [result.intervention_feature_ids],
            "intervention_values": [result.intervention_values],
            "intervention_base_values": [result.intervention_base_values],
            "intervention_score_values": [result.intervention_score_values],
            "intervention_scale_factors": [result.intervention_scale_factors],
            "pre_intervention_logits": [result.pre_intervention_logits.tolist()],
            "post_intervention_logits": [result.post_intervention_logits.tolist()],
            "logit_diff": [float(result.logit_diff.item())],
        },
        features=schema_to_features(module, schema=output_schema),
    )

    save_path = tmp_path / "intervention_store"
    store = AnalysisStore(dataset=dataset, op_output_dataset_path=str(save_path))
    store.save_to_disk(str(save_path))
    reloaded = AnalysisStore(
        dataset=load_from_disk(str(save_path)),
        it_format_kwargs={"analysis_backend": DEFAULT_CT_ANALYSIS_BACKEND},
    )

    row = cast(dict[str, object], reloaded[0])
    assert row["intervention_specs"] == [(1, 2, 11, 1.0)]
    feature_intervention_dict = cast(InterventionDict, row["feature_intervention_dict"])
    assert list(feature_intervention_dict.keys()) == ["feature.layer.1.position.2.id.11"]
    assert torch.isclose(
        feature_intervention_dict["feature.layer.1.position.2.id.11"][0].intervention_tensor,
        torch.tensor(1.0),
    )
    assert torch.equal(cast(torch.Tensor, row["intervention_layers"]), torch.tensor([1], dtype=torch.int64))
    assert torch.allclose(cast(torch.Tensor, row["intervention_values"]), torch.tensor([1.0], dtype=torch.float32))
    assert torch.allclose(cast(torch.Tensor, row["intervention_scale_factors"]), torch.tensor([2.0]))
    assert "intervention_activation_cache" not in row


def test_feature_intervention_forward_impl_returns_intermediate_activation_cache() -> None:
    module = _FakeModule()
    module.circuit_tracer_cfg.intervention_return_activations = True
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )

    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)
    output_schema = cast(Any, DISPATCHER.get_op("feature_intervention_forward")).output_schema

    assert module.replacement_model.calls == [
        (
            "Paris Austin",
            [(1, 2, 11, 1.0)],
            {"sparse": True, "return_activations": True, "constrained_layers": [0, 1]},
        )
    ]
    assert torch.equal(
        cast(torch.Tensor, result.intervention_activation_cache),
        torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                ],
            ],
            dtype=torch.float32,
        ),
    )
    assert output_schema["intervention_activation_cache"].intermediate_only is True
    assert "intervention_activation_cache" not in schema_to_features(module, schema=output_schema)


def test_feature_intervention_forward_impl_accepts_tensor_prompt_kwarg() -> None:
    module = _FakeModule()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    analysis_batch = AnalysisBatch(
        top_feature_ids=torch.tensor([[1, 2, 11]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )

    feature_intervention_forward_impl(
        module,
        analysis_batch,
        batch=cast(BatchEncoding, None),
        batch_idx=0,
        prompt=prompt,
    )

    recorded_prompt, _, _ = module.replacement_model.calls[0]
    assert torch.equal(cast(torch.Tensor, recorded_prompt), prompt)


def test_feature_intervention_forward_impl_can_use_activation_values() -> None:
    module = _FakeModule()
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = 10.0
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11], [0, 1, 7]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5, -0.25], dtype=torch.float32),
        top_feature_activation_values=torch.tensor([0.25, -0.1], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )

    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)

    assert len(module.replacement_model.calls) == 1
    prompt, interventions, kwargs = module.replacement_model.calls[0]
    assert prompt == "Paris Austin"
    assert kwargs == {"sparse": True, "return_activations": False, "constrained_layers": [0, 1]}
    assert [(layer, position, feature) for layer, position, feature, _ in interventions] == [(1, 2, 11), (0, 1, 7)]
    assert torch.allclose(
        torch.tensor([value for _, _, _, value in interventions], dtype=torch.float32),
        torch.tensor([2.5, -1.0], dtype=torch.float32),
    )
    assert torch.allclose(torch.tensor(result.intervention_values, dtype=torch.float32), torch.tensor([2.5, -1.0]))


def test_feature_intervention_forward_impl_uses_score_sign_for_activation_values() -> None:
    module = _FakeModule()
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = 10.0
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11], [0, 1, 7]], dtype=torch.long),
        top_feature_scores=torch.tensor([-0.5, 0.25], dtype=torch.float32),
        top_feature_activation_values=torch.tensor([0.8, -0.4], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )

    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)

    assert len(module.replacement_model.calls) == 1
    _, interventions, _ = module.replacement_model.calls[0]
    assert torch.allclose(
        torch.tensor([value for _, _, _, value in interventions], dtype=torch.float32),
        torch.tensor([-8.0, 4.0], dtype=torch.float32),
    )
    assert torch.allclose(torch.tensor(result.intervention_base_values), torch.tensor([0.8, -0.4]))
    assert torch.allclose(torch.tensor(result.intervention_score_values), torch.tensor([-0.5, 0.25]))
    assert torch.allclose(torch.tensor(result.intervention_scale_factors), torch.tensor([10.0, 10.0]))


def test_feature_intervention_forward_impl_can_max_normalize_scale_by_signed_influence() -> None:
    module = _FakeModule()
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = 10.0
    module.circuit_tracer_cfg.intervention_max_influence_norm_scale = True
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11], [0, 1, 7], [1, 0, 9]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5, -0.25, 0.0], dtype=torch.float32),
        top_feature_activation_values=torch.tensor([2.0, 4.0, 7.0], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )

    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)

    assert len(module.replacement_model.calls) == 1
    _, interventions, _ = module.replacement_model.calls[0]
    assert torch.allclose(
        torch.tensor([value for _, _, _, value in interventions], dtype=torch.float32),
        torch.tensor([20.0, -20.0, 0.0], dtype=torch.float32),
    )
    assert torch.allclose(torch.tensor(result.intervention_base_values), torch.tensor([2.0, 4.0, 7.0]))
    assert torch.allclose(torch.tensor(result.intervention_score_values), torch.tensor([0.5, -0.25, 0.0]))
    assert torch.allclose(torch.tensor(result.intervention_scale_factors), torch.tensor([10.0, 5.0, 0.0]))


def test_feature_intervention_forward_impl_requires_scores_for_max_normalized_scaling() -> None:
    module = _FakeModule()
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_max_influence_norm_scale = True
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11]], dtype=torch.long),
        top_feature_activation_values=torch.tensor([2.0], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="top_feature_scores when max_influence_norm_scale is enabled"):
        feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)


def test_feature_intervention_forward_impl_can_select_and_intervene_on_mixed_sign_features() -> None:
    module = _FakeModule()
    module.circuit_tracer_cfg.intervention_scale_factor = 1.0
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        active_features=torch.tensor([[0, 0, 10], [1, 0, 12]], dtype=torch.long),
        node_logit_diff_gradient_scores=torch.tensor([0.4, -0.7], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
    )
    feature_selection = FeatureSelectionSpec(
        triples=[(0, 0, 10), (1, 0, 12)],
        score_source="gradient",
        score_sign="any",
        rank_by_abs=True,
    )

    result = feature_intervention_forward_impl(
        module,
        analysis_batch,
        batch=cast(BatchEncoding, None),
        batch_idx=0,
        top_n=2,
        feature_selection=feature_selection,
    )

    assert len(module.replacement_model.calls) == 1
    prompt, interventions, call_kwargs = module.replacement_model.calls[0]
    assert prompt == "Paris Austin"
    assert [(layer, position, feature_id) for layer, position, feature_id, _ in interventions] == [
        (1, 0, 12),
        (0, 0, 10),
    ]
    assert torch.allclose(
        torch.tensor([value for _, _, _, value in interventions], dtype=torch.float32),
        torch.tensor([-0.7, 0.4], dtype=torch.float32),
    )
    assert call_kwargs == {"sparse": True, "return_activations": False, "constrained_layers": [0, 1]}
    assert torch.equal(result.top_feature_ids, torch.tensor([[1, 0, 12], [0, 0, 10]], dtype=torch.long))
    assert torch.allclose(result.top_feature_scores, torch.tensor([-0.7, 0.4], dtype=torch.float32))
    assert torch.allclose(torch.tensor(result.intervention_values), torch.tensor([-0.7, 0.4]))
    assert list(result.feature_intervention_dict.keys()) == [
        "feature.layer.1.position.0.id.12",
        "feature.layer.0.position.0.id.10",
    ]


def test_feature_intervention_resolves_logit_target_ids_from_concept_groups() -> None:
    """When no explicit logit_target_ids, concept group token IDs are used as fallback."""
    module = _FakeModule()
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5], dtype=torch.float32),
        concept_group_a_token_ids=[2],
        concept_group_b_token_ids=[3],
    )

    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)

    # logit_diff should be computed over token IDs 2 and 3 (from concept groups)
    # pre_logits = [0.1, 0.2, 0.3, 0.4], post_logits = [0.1, 0.2, 0.3+delta, 0.4]
    # delta = 0.5 * 2.0 = 1.0 (scale_factor=2.0)
    # diff on id=2: (0.3+1.0 - 0.3) = 1.0; diff on id=3: (0.4 - 0.4) = 0.0
    # mean = (1.0 + 0.0) / 2 = 0.5
    assert torch.isclose(result.logit_diff, torch.tensor(0.5, dtype=torch.float32))


def test_feature_intervention_explicit_logit_target_ids_override_concept_groups() -> None:
    """Explicit logit_target_ids take precedence over concept group token IDs."""
    module = _FakeModule()
    analysis_batch = AnalysisBatch(
        prompts=["Paris Austin"],
        top_feature_ids=torch.tensor([[1, 2, 11]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5], dtype=torch.float32),
        logit_target_ids=torch.tensor([2], dtype=torch.long),
        concept_group_a_token_ids=[0],
        concept_group_b_token_ids=[1],
    )

    result = feature_intervention_forward_impl(module, analysis_batch, batch=cast(BatchEncoding, None), batch_idx=0)

    # logit_diff should use explicit logit_target_ids=[2], not concept group IDs [0, 1]
    # pre_logits = [0.1, 0.2, 0.3, 0.4], post_logits = [0.1, 0.2, 1.3, 0.4]
    # diff on id=2 only: (1.3 - 0.3) = 1.0
    assert torch.isclose(result.logit_diff, torch.tensor(1.0, dtype=torch.float32))


def test_mean_target_logit_delta_raises_on_virtual_ids() -> None:
    """Virtual IDs (>= vocab_size) in mean_target_logit_delta raise a descriptive error."""
    pre_logits = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    post_logits = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    virtual_ids = torch.tensor([4, 5], dtype=torch.long)  # vocab_size=4, so 4 and 5 are out of bounds

    with pytest.raises(ValueError, match="out-of-bounds indices"):
        mean_target_logit_delta(pre_logits, post_logits, virtual_ids)
