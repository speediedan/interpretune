"""Synthetic analysis-backend graph serialization tests.

These currently target the circuit-tracer analysis backend, while keeping the test surface framed around generic
analysis-backend graph responsibilities so future backends can extend the same coverage.
"""

from __future__ import annotations

import torch
from datasets import Dataset, load_from_disk

from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph
from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig

from interpretune.analysis.backends.circuit_tracer import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.analysis.core import AnalysisStore, schema_to_features
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.analysis.ops.definitions import (
    compute_attribution_graph_impl,
    concept_direction_impl,
    extract_top_features_impl,
    graph_node_influence_impl,
    graph_prune_impl,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {"Paris": 0, "London": 1, "Dallas": 2, "Austin": 3}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [self._vocab[token] for token in text.split() if token in self._vocab]}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        inverse_vocab = {value: key for key, value in self._vocab.items()}
        return " ".join(inverse_vocab[int(token_id)] for token_id in token_ids if int(token_id) in inverse_vocab)


class _FakeReplacementModel:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.embed_weight = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )


class _FakeModule:
    def __init__(self, graph: Graph | None = None) -> None:
        self.replacement_model = _FakeReplacementModel()
        self.datamodule = type(
            "_DataModule",
            (),
            {
                "tokenizer": self.replacement_model.tokenizer,
                "itdm_cfg": type("_ITDMCfg", (), {"eval_batch_size": 1})(),
            },
        )()
        self.model = type(
            "_Model",
            (),
            {"tokenizer": type("_TokenizerMeta", (), {"vocab_size": 32, "model_max_length": 32})()},
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
        self._analysis_backend = DEFAULT_CT_ANALYSIS_BACKEND
        self._graph = graph

    def generate_attribution_graph(self, prompt: str, **kwargs) -> Graph:
        assert self._graph is not None
        return self._graph


def _make_graph() -> Graph:
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
        input_string="Paris Austin",
        input_tokens=torch.tensor([0, 3], dtype=torch.long),
        active_features=torch.tensor([[0, 0, 10], [1, 0, 12], [1, 1, 13]], dtype=torch.long),
        adjacency_matrix=torch.tensor(
            [
                [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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


def _graph_batch_from_graph(graph: Graph) -> AnalysisBatch:
    decomposed = compute_attribution_graph_impl(
        _FakeModule(graph=graph), AnalysisBatch(prompts=[graph.input_string]), None, 0
    )
    return AnalysisBatch(**decomposed)


def _rehydrate_graph(analysis_batch: AnalysisBatch) -> Graph:
    return DEFAULT_CT_ANALYSIS_BACKEND.hydrate_graph_from_batch(analysis_batch)


def test_concept_direction_impl_uses_embedding_difference() -> None:
    module = _FakeModule()
    analysis_batch = AnalysisBatch(concept_group_a=["Paris"], concept_group_b=["London"])

    result = concept_direction_impl(module, analysis_batch, batch=None, batch_idx=0)

    expected = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected)
    assert result.concept_label == "Paris -> London"


def test_compute_attribution_graph_impl_decomposes_graph() -> None:
    graph = _make_graph()
    module = _FakeModule(graph=graph)
    analysis_batch = AnalysisBatch(prompts=[graph.input_string])

    result = compute_attribution_graph_impl(module, analysis_batch, batch=None, batch_idx=0)
    restored_graph = _rehydrate_graph(result)

    assert torch.equal(restored_graph.adjacency_matrix, graph.adjacency_matrix)
    assert torch.equal(result.active_features, graph.active_features)
    assert result.logit_target_tokens == ["Dallas", "Austin"]
    assert torch.equal(result.logit_target_ids, graph.logit_token_ids.cpu())


def test_graph_prune_impl_round_trips_serialized_graph() -> None:
    graph = _make_graph()
    analysis_batch = _graph_batch_from_graph(graph)
    module = _FakeModule(graph=graph)

    result = graph_prune_impl(module, analysis_batch, batch=None, batch_idx=0, node_threshold=0.4, edge_threshold=0.4)
    restored_graph = _rehydrate_graph(result)

    assert restored_graph.adjacency_matrix.shape[0] <= graph.adjacency_matrix.shape[0]
    assert len(restored_graph.selected_features) <= len(graph.selected_features)
    assert result.graph_metadata is not None


def test_extract_top_features_impl_prefers_node_influence_scores() -> None:
    module = _FakeModule()
    analysis_batch = AnalysisBatch(
        active_features=torch.tensor([[0, 0, 10], [1, 1, 13]], dtype=torch.long),
        activation_values=torch.tensor([0.25, 0.75], dtype=torch.float32),
        node_influence_scores=torch.tensor([0.9, 0.1], dtype=torch.float32),
    )

    result = extract_top_features_impl(module, analysis_batch, batch=None, batch_idx=0, top_n=1)

    assert torch.equal(result.top_feature_ids, torch.tensor([[0, 0, 10]], dtype=torch.long))
    assert torch.allclose(result.top_feature_scores, torch.tensor([0.9], dtype=torch.float32))


def test_graph_node_influence_impl_returns_feature_rows() -> None:
    graph = _make_graph()
    analysis_batch = _graph_batch_from_graph(graph)
    module = _FakeModule(graph=graph)

    result = graph_node_influence_impl(module, analysis_batch, batch=None, batch_idx=0)

    assert result.node_feature_ids.shape == (2, 3)
    assert result.node_influence_scores.shape == (2,)
    assert torch.equal(result.node_feature_ids, graph.active_features.index_select(0, graph.selected_features))


def test_graph_fields_round_trip_through_analysis_store(tmp_path) -> None:
    graph = _make_graph()
    decomposed = _graph_batch_from_graph(graph)
    module = _FakeModule(graph=graph)
    dataset = Dataset.from_dict(
        {
            "input_string": [decomposed.input_string],
            "adjacency_matrix": [decomposed.adjacency_matrix.tolist()],
            "active_features": [decomposed.active_features.tolist()],
            "selected_features": [decomposed.selected_features.tolist()],
            "activation_values": [decomposed.activation_values.tolist()],
            "logit_target_ids": [decomposed.logit_target_ids.tolist()],
            "logit_target_tokens": [decomposed.logit_target_tokens],
            "logit_probabilities": [decomposed.logit_probabilities.tolist()],
            "input_tokens": [decomposed.input_tokens.tolist()],
            "graph_cfg_json": [decomposed.graph_cfg_json],
            "graph_scan_json": [decomposed.graph_scan_json],
            "graph_vocab_size": [decomposed.graph_vocab_size],
            "graph_metadata": [decomposed.graph_metadata],
        },
        features=schema_to_features(module, schema=DISPATCHER.get_op("compute_attribution_graph").output_schema),
    )
    save_path = tmp_path / "graph_store"
    store = AnalysisStore(
        dataset=dataset,
        op_output_dataset_path=str(save_path),
    )
    store.save_to_disk(str(save_path))
    reloaded = AnalysisStore(
        dataset=load_from_disk(str(save_path)),
        it_format_kwargs={"analysis_backend": DEFAULT_CT_ANALYSIS_BACKEND},
    )
    reloaded_row = reloaded[0]
    assert isinstance(reloaded_row["attribution_graph"], Graph)
    restored_graph = reloaded_row["attribution_graph"]

    assert torch.equal(restored_graph.adjacency_matrix, graph.adjacency_matrix)
    assert torch.equal(restored_graph.active_features, graph.active_features)
    assert torch.equal(restored_graph.logit_token_ids.cpu(), graph.logit_token_ids.cpu())
    assert [target.token_str for target in restored_graph.logit_targets] == [
        target.token_str for target in graph.logit_targets
    ]


def test_extract_top_features_impl_uses_selected_feature_mapping() -> None:
    graph = _make_graph()
    module = _FakeModule(graph=graph)
    analysis_batch = AnalysisBatch(
        active_features=graph.active_features,
        selected_features=graph.selected_features,
        node_influence_scores=torch.tensor([0.9, 0.1], dtype=torch.float32),
    )

    result = extract_top_features_impl(module, analysis_batch, batch=None, batch_idx=0, top_n=1)

    expected = graph.active_features.index_select(0, torch.tensor([0], dtype=torch.long))
    assert torch.equal(result.top_feature_ids, expected)
