"""Synthetic analysis-backend intervention tests.

These currently exercise the circuit-tracer analysis backend because it is the only native analysis backend with
intervention support. The fixtures are intentionally backend-shaped so additional analysis backends can reuse the same
test surface later.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from datasets import Dataset, load_from_disk
from transformers import BatchEncoding

from interpretune.analysis.backends.circuit_tracer import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.analysis.core import AnalysisStore, schema_to_features
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.definitions import feature_intervention_forward_impl
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.config.circuit_tracer import CircuitTracerConfig


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "decoded prompt"


class _FakeReplacementModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[tuple[int, int, int, float]], dict[str, object]]] = []
        self.tokenizer = _FakeTokenizer()

    def get_activations(self, prompt: str):
        assert prompt == "Paris Austin"
        return torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32), None

    def feature_intervention(self, prompt: str, interventions, **kwargs):
        self.calls.append((prompt, list(interventions), dict(kwargs)))
        total_delta = sum(value for _, _, _, value in interventions)
        return torch.tensor([[[0.1, 0.2, 0.3 + total_delta, 0.4]]], dtype=torch.float32), None


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
            "intervention_layers": [result.intervention_layers],
            "intervention_positions": [result.intervention_positions],
            "intervention_feature_ids": [result.intervention_feature_ids],
            "intervention_values": [result.intervention_values],
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
    assert torch.equal(cast(torch.Tensor, row["intervention_layers"]), torch.tensor([1], dtype=torch.int64))
    assert torch.allclose(cast(torch.Tensor, row["intervention_values"]), torch.tensor([1.0], dtype=torch.float32))


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
