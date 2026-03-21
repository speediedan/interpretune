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
from transformers import BatchEncoding

import interpretune as it
from circuit_tracer.attribution.targets import CustomTarget, LogitTarget
from circuit_tracer.graph import Graph
from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig

from interpretune.analysis import AnalysisInputs, execute_analysis_op
from interpretune.analysis.backends.circuit_tracer import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.analysis.core import AnalysisStore
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.definitions import (
    compute_attribution_graph_impl,
    concept_direction_impl,
    extract_concept_latent_examples_impl,
    feature_intervention_forward_impl,
)
from interpretune.config.circuit_tracer import CircuitTracerConfig
from interpretune.config import init_analysis_cfgs


class _FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {"Paris": 0, "London": 1, "Dallas": 2, "Austin": 3}
        self.pad_token_id = 0
        self.padding_side = "right"
        self.vocab_size = len(self._vocab)
        self.model_max_length = 16

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


class _FakeITSession(dict):
    def __init__(self, *, module, datamodule) -> None:
        super().__init__(module=module, datamodule=datamodule)
        self.module = module
        self.datamodule = datamodule


class _FakeConceptExtractionBackend:
    def fwd_w_cache_and_latent_models(self, model, batch, latent_model_handles, names_filter):
        del latent_model_handles, model
        labels = batch.get("labels")
        if labels is None:
            input_ids = batch["input"]
            labels = torch.tensor([0 if int(token_id) % 4 in (3, 0) else 1 for token_id in input_ids[:, 0]])
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(dtype=torch.long)
        batch_size = labels.shape[0]
        answer_logits = torch.zeros((batch_size, 2, 2), dtype=torch.float32)
        latent_rows = []
        for idx, label in enumerate(labels.tolist()):
            if label == 0:
                latent_rows.append([3.0, 0.0] if idx == 0 else [1.0, 1.0])
                answer_logits[idx, 1] = torch.tensor([3.0, 1.0] if idx == 0 else [1.0, 1.25], dtype=torch.float32)
            else:
                latent_rows.append([0.0, 4.0] if idx == 1 else [1.0, 3.0])
                answer_logits[idx, 1] = torch.tensor([1.0, 2.0] if idx == 1 else [1.0, 1.5], dtype=torch.float32)
        cache_key = "unembed.hook_in.hook_sae_input"
        cache_rows = [[[0.0, 0.0], row] for row in latent_rows]
        cache = {cache_key: torch.tensor(cache_rows, dtype=torch.float32)} if names_filter(cache_key) else {}
        return answer_logits, cache


class _FakeConceptExtractionDataModule:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.itdm_cfg = SimpleNamespace(eval_batch_size=2)
        self._batches = [
            BatchEncoding(
                {
                    "input": torch.tensor([[7, 8], [9, 10]], dtype=torch.long),
                    "labels": torch.tensor([0, 1], dtype=torch.long),
                }
            ),
            BatchEncoding(
                {
                    "input": torch.tensor([[11, 12], [13, 14]], dtype=torch.long),
                    "labels": torch.tensor([0, 1], dtype=torch.long),
                }
            ),
        ]

    def test_dataloader(self):
        return [BatchEncoding(dict(batch)) for batch in self._batches]

    def prepare_data(self, target_model=None):
        del target_model
        return None

    def setup(self, stage=None, target_model=None):
        del stage, target_model
        return None


class _FakeConceptExtractionModule:
    def __init__(self) -> None:
        self.model = _TLLikeModel()
        self.datamodule = _FakeConceptExtractionDataModule()
        self.it_cfg = SimpleNamespace(
            optimizer_init=False,
            generative_step_cfg=SimpleNamespace(lm_generation_cfg=SimpleNamespace(max_new_tokens=1)),
            num_labels=2,
            entailment_mapping=("Yes", "No"),
        )
        self._analysis_backend = DEFAULT_CT_ANALYSIS_BACKEND
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.current_epoch = 0
        self.global_step = 0
        self.sae_handles = [
            SimpleNamespace(
                cfg=SimpleNamespace(metadata=SimpleNamespace(hook_name="unembed.hook_in")),
                hook_dict={"hook_sae_input": object()},
            )
        ]
        self._model_backend = _FakeConceptExtractionBackend()
        self.analysis_cfg = None

    def batch_to_device(self, batch):
        return batch

    def labels_to_ids(self, labels):
        label_tensor = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        return label_tensor.clone(), label_tensor.clone()

    def standardize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def loss_fn(self, answer_logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        del answer_logits, label_ids
        return torch.tensor(0.0, dtype=torch.float32)

    def auto_prune_batch(self, batch, phase: str):
        del phase
        return batch

    def on_analysis_start(self):
        return None

    def on_analysis_epoch_end(self):
        return None

    def on_analysis_end(self):
        return None

    def setup(self, stage=None, datamodule=None):
        del stage, datamodule
        return None


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
    assert torch.allclose(result.concept_direction, expected, atol=1e-4)
    assert result.concept_group_a_token_ids == [0]
    assert result.concept_group_b_token_ids == [1]
    assert int(torch.argmax(torch.abs(result.concept_direction)).item()) == 0
    assert expected_first_id == 2


@pytest.mark.parametrize(
    "module_factory",
    [
        pytest.param(_TLLikeProducerModule, id="transformerlens"),
        pytest.param(_NNsightLikeProducerModule, id="nnsight"),
    ],
)
def test_concept_direction_direct_impl_uses_schema_default_mode(module_factory) -> None:
    module = module_factory()

    result = concept_direction_impl(
        module,
        AnalysisBatch(concept_group_a=["Paris"], concept_group_b=["London"]),
        batch=None,
        batch_idx=0,
    )

    assert result.concept_direction_mode == "mean_difference"


def test_extract_concept_latent_examples_filters_correct_rows_and_applies_logit_diff_weights() -> None:
    module = SimpleNamespace()
    analysis_batch = AnalysisBatch(
        cache={
            "unembed.hook_in.hook_sae_input": torch.tensor(
                [
                    [[0.0, 0.0], [3.0, 0.0]],
                    [[0.0, 0.0], [0.0, 4.0]],
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[0.0, 0.0], [1.0, 3.0]],
                ],
                dtype=torch.float32,
            )
        },
        answer_indices=torch.tensor([1, 1, 1, 1], dtype=torch.long),
        orig_labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        logit_diffs=torch.tensor([2.0, 1.0, -0.25, 0.5], dtype=torch.float32),
        concept_group_a_label_ids=[0],
        concept_group_b_label_ids=[1],
        concept_group_a_name="capital",
        concept_group_b_name="state",
        concept_weight_by_logit_diff=1,
    )

    result = extract_concept_latent_examples_impl(module, analysis_batch, batch=None, batch_idx=0)

    assert torch.equal(
        result.concept_latent_state,
        torch.tensor([[3.0, 0.0], [0.0, 4.0], [1.0, 3.0]], dtype=torch.float32),
    )
    assert torch.equal(result.concept_group_id, torch.tensor([0, 1, 1], dtype=torch.long))
    assert result.concept_group_name == ["capital", "state", "state"]
    assert torch.equal(result.concept_example_logit_diff, torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32))
    assert torch.equal(result.concept_example_weight, torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32))


def test_concept_direction_aggregates_round_tripped_synthetic_hf_dataset(tmp_path) -> None:
    synthetic_dataset = Dataset.from_dict(
        {
            "prompt": [
                "Paris : Is this a capital or a state?",
                "London : Is this a capital or a state?",
                "Dallas : Is this a capital or a state?",
                "Austin : Is this a capital or a state?",
            ],
            "label": ["capital", "capital", "state", "state"],
            "concept_latent_state": [[3.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 2.0]],
            "concept_group_id": [0, 0, 1, 1],
            "concept_group_name": ["capital", "capital", "state", "state"],
            "concept_example_weight": [2.0, 1.0, 1.0, 0.5],
        }
    )
    extracted_store = _round_trip_store(tmp_path, "synthetic_concept_examples", synthetic_dataset)
    producer = _TLLikeProducerModule()
    producer.analysis_cfg = SimpleNamespace(input_store=extracted_store, batch_inputs={}, run_inputs={})

    result = concept_direction_impl(
        producer,
        AnalysisBatch(concept_direction_mode="mean_difference"),
        batch=None,
        batch_idx=0,
    )

    expected = torch.tensor([8.0 / 3.0, -10.0 / 3.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected, atol=1e-4)
    assert result.concept_label == "capital -> state"

    aggregated_store = _round_trip_store(
        tmp_path,
        "aggregated_concept_direction",
        Dataset.from_dict(
            {
                "concept_direction": [result.concept_direction.tolist()],
                "concept_label": [result.concept_label],
                "concept_direction_mode": [result.concept_direction_mode],
                "concept_group_a_name": [result.concept_group_a_name],
                "concept_group_b_name": [result.concept_group_b_name],
            }
        ),
    )
    consumer = _FakeCircuitTracerConsumerModule(backend="nnsight", input_store=aggregated_store)

    graph_result = compute_attribution_graph_impl(
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
    assert attribution_targets[0].token_str == "capital -> state"

    metadata = json.loads(graph_result.graph_metadata)
    assert metadata["concept_label"] == "capital -> state"


def test_concept_direction_runner_store_workflow_aggregates_latent_examples(tmp_path) -> None:
    module = _FakeConceptExtractionModule()
    module.core_log_dir = tmp_path
    datamodule = module.datamodule
    extraction_inputs = SimpleNamespace(
        cache=[
            {"unembed.hook_in.hook_sae_input": torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]])},
            {"unembed.hook_in.hook_sae_input": torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]])},
        ],
        answer_indices=[torch.tensor([1, 1], dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)],
        orig_labels=[torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)],
        logit_diffs=[torch.tensor([2.0, 1.0], dtype=torch.float32), torch.tensor([-0.25, 0.5], dtype=torch.float32)],
    )
    analysis_cfg = it.AnalysisCfg(
        name="latent_example_rows",
        target_op=it.extract_concept_latent_examples,
        ignore_manual=True,
    )
    init_analysis_cfgs(module, [analysis_cfg], ignore_manual=True)

    concept_rows = []
    for batch_idx, batch in enumerate(datamodule.test_dataloader()):
        extracted = execute_analysis_op(
            module,
            batch,
            batch_idx,
            analysis_batch=AnalysisBatch(
                concept_group_a_label_ids=[0],
                concept_group_b_label_ids=[1],
                concept_group_a_name="capital",
                concept_group_b_name="state",
                concept_weight_by_logit_diff=True,
            ),
            analysis_cfg=analysis_cfg,
            analysis_inputs=AnalysisInputs(store=extraction_inputs),
        )
        for row_idx in range(int(extracted.concept_group_id.shape[0])):
            concept_rows.append(
                {
                    "concept_latent_state": extracted.concept_latent_state[row_idx].tolist(),
                    "concept_group_id": int(extracted.concept_group_id[row_idx].item()),
                    "concept_group_name": extracted.concept_group_name[row_idx],
                    "concept_example_logit_diff": float(extracted.concept_example_logit_diff[row_idx].item()),
                    "concept_example_weight": float(extracted.concept_example_weight[row_idx].item()),
                }
            )

    concept_store = AnalysisStore(
        dataset=Dataset.from_list(concept_rows), op_output_dataset_path=str(tmp_path / "rows")
    )
    assert len(concept_store.dataset) == 3

    aggregate_cfg = it.AnalysisCfg(
        target_op=it.concept_direction,
        input_store=concept_store,
        ignore_manual=True,
        save_tokens=False,
    )
    init_analysis_cfgs(module, [aggregate_cfg], ignore_manual=True)
    previous_analysis_cfg = module.analysis_cfg
    module.analysis_cfg = aggregate_cfg
    try:
        result = execute_analysis_op(
            module,
            BatchEncoding({}),
            0,
            analysis_batch=AnalysisBatch(concept_direction_mode="mean_difference"),
            analysis_cfg=aggregate_cfg,
        )
    finally:
        module.analysis_cfg = previous_analysis_cfg

    expected = torch.tensor([8.0 / 3.0, -11.0 / 3.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected, atol=1e-4)
    assert result.concept_label == "capital -> state"
    assert result.concept_direction_mode == "mean_difference"


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
                "concept_group_a_token_ids": [produced.concept_group_a_token_ids],
                "concept_group_b_token_ids": [produced.concept_group_b_token_ids],
                "concept_direction_mode": [produced.concept_direction_mode],
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


def test_concept_direction_prefers_run_scoped_inputs_over_row_store_values(tmp_path) -> None:
    module = _NNsightLikeProducerModule()
    input_store = _round_trip_store(
        tmp_path,
        "concept_store_with_rows",
        Dataset.from_dict(
            {
                "concept_group_a": [["Dallas"], ["Austin"]],
                "concept_group_b": [["London"], ["Paris"]],
            }
        ),
    )
    module.analysis_cfg = SimpleNamespace(
        input_store=input_store,
        batch_inputs={},
        run_inputs={"concept_group_a": ["Paris"], "concept_group_b": ["London"]},
    )

    result = concept_direction_impl(module, AnalysisBatch(), batch=None, batch_idx=1)

    expected = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected)
    assert result.concept_label == "Paris -> London"


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


@pytest.mark.parametrize(
    "module_factory",
    [
        pytest.param(_TLLikeProducerModule, id="transformerlens"),
        pytest.param(_NNsightLikeProducerModule, id="nnsight"),
    ],
)
def test_concept_direction_multi_batch_accumulation(tmp_path, module_factory) -> None:
    """Multiple dataloader batches produce per-batch direction rows that can be averaged."""
    module = module_factory()
    concept_batches = [
        {"concept_group_a": ["Paris"], "concept_group_b": ["London"]},
        {"concept_group_a": ["Dallas"], "concept_group_b": ["Austin"]},
    ]
    directions = []
    rows: dict[str, list] = {
        "concept_direction": [],
        "concept_label": [],
        "concept_group_a_token_ids": [],
        "concept_group_b_token_ids": [],
        "concept_direction_mode": [],
    }
    for batch_idx, groups in enumerate(concept_batches):
        result = concept_direction_impl(
            module,
            AnalysisBatch(**groups),
            batch=None,
            batch_idx=batch_idx,
        )
        directions.append(result.concept_direction)
        rows["concept_direction"].append(result.concept_direction.tolist())
        rows["concept_label"].append(result.concept_label)
        rows["concept_group_a_token_ids"].append(result.concept_group_a_token_ids)
        rows["concept_group_b_token_ids"].append(result.concept_group_b_token_ids)
        rows["concept_direction_mode"].append(result.concept_direction_mode)

    # Each batch produces a unit-norm direction
    for d in directions:
        assert torch.isclose(torch.linalg.vector_norm(d), torch.tensor(1.0)), "each per-batch direction should be unit"

    # Store round-trips all rows
    store = _round_trip_store(tmp_path, "multi_batch_concept", Dataset.from_dict(rows))
    assert len(store.dataset) == len(concept_batches)

    # Average + renormalize (mirroring the notebook pattern)
    stacked = torch.stack(directions)
    avg_dir = stacked.mean(dim=0)
    avg_dir = avg_dir / torch.linalg.vector_norm(avg_dir)
    assert torch.isclose(torch.linalg.vector_norm(avg_dir), torch.tensor(1.0))

    # The averaged direction should correlate positively with each per-batch direction
    for d in directions:
        cos = torch.nn.functional.cosine_similarity(avg_dir.unsqueeze(0), d.unsqueeze(0)).item()
        assert cos > 0, f"averaged direction should be positively aligned with each batch direction, got {cos}"
