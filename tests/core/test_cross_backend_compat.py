"""Focused IG-4 cross-backend compatibility tests.

These tests stay intentionally lightweight while validating the composition
contract that matters for the circuit-tracer backend workstream:

- a native producer op can originate on a non-CT module surface
- AnalysisStore artifacts can be round-tripped and then consumed by CT ops
- CT consumer ops can read the same stored artifacts across both of their
  execution backends
- the analysis-level intervention op matches the native replacement-model
  intervention call on the nnsight-style source-of-truth path
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, load_from_disk

from circuit_tracer.attribution.targets import CustomTarget, LogitTarget
from circuit_tracer.graph import Graph
from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig

from interpretune.analysis.backends.circuit_tracer import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.analysis.core import AnalysisStore
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.definitions import (
    compute_attribution_graph_impl,
    concept_direction_impl,
    feature_intervention_forward_impl,
)
from interpretune.config.circuit_tracer import CircuitTracerConfig


class _FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {"Paris": 0, "London": 1, "Dallas": 2, "Austin": 3}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [self._vocab[token] for token in text.split() if token in self._vocab]}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        inverse_vocab = {value: key for key, value in self._vocab.items()}
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(inverse_vocab[int(token_id)] for token_id in token_ids if int(token_id) in inverse_vocab)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.decode([token_id], skip_special_tokens=False)


def _embed_weight() -> torch.Tensor:
    return torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )


class _TLLikeModel:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.W_E = _embed_weight()


class _NNsightLikeEmbeddings:
    def __init__(self) -> None:
        self.weight = _embed_weight()


class _NNsightLikeModel:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self._embeddings = _NNsightLikeEmbeddings()

    def get_input_embeddings(self):
        return self._embeddings


class _TLLikeProducerModule:
    def __init__(self) -> None:
        self.model = _TLLikeModel()
        self.datamodule = SimpleNamespace(tokenizer=self.model.tokenizer)


class _NNsightLikeProducerModule:
    def __init__(self) -> None:
        self.model = _NNsightLikeModel()
        self.datamodule = SimpleNamespace(tokenizer=self.model.tokenizer)


class _FakeReplacementModel:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.embed_weight = _embed_weight()
        self.calls: list[tuple[str, list[tuple[int, int, int, float]], dict[str, object]]] = []

    def get_activations(self, prompt: str):
        assert prompt == "Paris Austin"
        return torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32), None

    def feature_intervention(self, prompt: str, interventions, **kwargs):
        self.calls.append((prompt, list(interventions), dict(kwargs)))
        total_delta = sum(value for _, _, _, value in interventions)
        return torch.tensor([[[0.1, 0.2, 0.3 + total_delta, 0.4]]], dtype=torch.float32), None


def _make_graph(prompt: str) -> Graph:
    cfg = UnifiedConfig(
        n_layers=2,
        d_model=4,
        d_head=2,
        n_heads=2,
        d_mlp=8,
        d_vocab=32,
        tokenizer_name="fake-tokenizer",
        model_name="fake-model",
        original_architecture="FakeForCausalLM",
    )
    return Graph(
        input_string=prompt,
        input_tokens=torch.tensor([0, 3], dtype=torch.long),
        active_features=torch.tensor([[0, 0, 10], [1, 0, 12], [1, 1, 13]], dtype=torch.long),
        adjacency_matrix=torch.tensor(
            [
                [0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.4],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        cfg=cfg,
        selected_features=torch.tensor([0, 2], dtype=torch.long),
        activation_values=torch.tensor([0.25, 0.75], dtype=torch.float32),
        logit_targets=[LogitTarget("Dallas", 2), LogitTarget("Austin", 3)],
        logit_probabilities=torch.tensor([0.4, 0.6], dtype=torch.float32),
        scan="gemma",
        vocab_size=32,
    )


class _FakeCircuitTracerConsumerModule:
    def __init__(self, backend: str, input_store: AnalysisStore | None = None) -> None:
        self.replacement_model = _FakeReplacementModel()
        self._analysis_backend = DEFAULT_CT_ANALYSIS_BACKEND
        self.analysis_cfg = SimpleNamespace(input_store=input_store)
        self.datamodule = SimpleNamespace(
            tokenizer=self.replacement_model.tokenizer,
            itdm_cfg=SimpleNamespace(eval_batch_size=1),
        )
        self.model = SimpleNamespace(
            tokenizer=SimpleNamespace(vocab_size=4, model_max_length=16),
            get_input_embeddings=lambda: SimpleNamespace(weight=_embed_weight()),
        )
        self.circuit_tracer_cfg = CircuitTracerConfig(
            backend=backend,
            intervention_scale_factor=2.0,
            intervention_sparse=True,
            intervention_return_activations=False,
            intervention_constrained_layers=[0, 1],
        )
        self.generate_calls: list[dict[str, object]] = []

    @property
    def analysis_backend(self):
        return self._analysis_backend

    def generate_attribution_graph(self, prompt: str, **kwargs) -> Graph:
        self.generate_calls.append({"prompt": prompt, **kwargs})
        return _make_graph(prompt)


def _round_trip_store(tmp_path, name: str, dataset: Dataset, *, it_format_kwargs: dict | None = None) -> AnalysisStore:
    save_path = tmp_path / name
    store = AnalysisStore(dataset=dataset, op_output_dataset_path=str(save_path), it_format_kwargs=it_format_kwargs)
    store.save_to_disk(str(save_path))
    return AnalysisStore(dataset=load_from_disk(str(save_path)), it_format_kwargs=it_format_kwargs)


@pytest.mark.parametrize(
    ("module_factory", "expected_first_id"),
    [
        pytest.param(_TLLikeProducerModule, 2, id="transformerlens_native_producer"),
        pytest.param(_NNsightLikeProducerModule, 2, id="nnsight_native_producer"),
    ],
)
def test_concept_direction_supports_non_ct_native_modules(module_factory, expected_first_id) -> None:
    module = module_factory()

    result = concept_direction_impl(
        module,
        AnalysisBatch(concept_group_a=["Paris"], concept_group_b=["London"]),
        batch=None,
        batch_idx=0,
    )

    expected = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected)
    metadata = json.loads(result.concept_metadata)
    assert metadata["group_a_token_ids"] == [0]
    assert metadata["group_b_token_ids"] == [1]
    assert int(torch.argmax(torch.abs(result.concept_direction)).item()) == 0
    assert expected_first_id == 2


def test_transformerlens_store_round_trip_can_feed_ct_attribution_op(tmp_path) -> None:
    producer = _TLLikeProducerModule()
    produced = concept_direction_impl(
        producer,
        AnalysisBatch(concept_group_a=["Paris"], concept_group_b=["London"]),
        batch=None,
        batch_idx=0,
    )
    concept_store = _round_trip_store(
        tmp_path,
        "concept_store",
        Dataset.from_dict(
            {
                "concept_direction": [produced.concept_direction.tolist()],
                "concept_label": [produced.concept_label],
                "concept_metadata": [produced.concept_metadata],
            }
        ),
    )
    consumer = _FakeCircuitTracerConsumerModule(backend="nnsight", input_store=concept_store)

    result = compute_attribution_graph_impl(
        consumer,
        AnalysisBatch(prompts=["Paris Austin"]),
        batch=None,
        batch_idx=0,
        concept_target_top_k=2,
    )

    assert consumer.generate_calls, "compute_attribution_graph should invoke the CT producer"
    call = consumer.generate_calls[0]
    attribution_targets = call["attribution_targets"]
    assert isinstance(attribution_targets, list)
    assert len(attribution_targets) == 1
    assert isinstance(attribution_targets[0], CustomTarget)
    assert attribution_targets[0].token_str == "Paris -> London"

    metadata = json.loads(result.graph_metadata)
    assert metadata["concept_label"] == "Paris -> London"


@pytest.mark.parametrize("backend", ["transformerlens", "nnsight"])
def test_ct_feature_intervention_consumes_round_tripped_store_across_backends(tmp_path, backend: str) -> None:
    input_store = _round_trip_store(
        tmp_path,
        f"feature_store_{backend}",
        Dataset.from_dict(
            {
                "top_feature_ids": [[[1, 2, 11], [0, 1, 7]]],
                "top_feature_scores": [[0.5, -0.25]],
                "logit_target_ids": [[2]],
            }
        ),
    )
    consumer = _FakeCircuitTracerConsumerModule(backend=backend, input_store=input_store)

    result = feature_intervention_forward_impl(
        consumer,
        AnalysisBatch(prompts=["Paris Austin"]),
        batch=None,
        batch_idx=0,
    )

    assert consumer.replacement_model.calls == [
        (
            "Paris Austin",
            [(1, 2, 11, 1.0), (0, 1, 7, -0.5)],
            {"sparse": True, "return_activations": False, "constrained_layers": [0, 1]},
        )
    ]
    assert result.intervention_feature_ids == [11, 7]
    assert result.intervention_positions == [2, 1]
    assert torch.isclose(result.logit_diff, torch.tensor(0.5, dtype=torch.float32))
