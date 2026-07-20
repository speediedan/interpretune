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
import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, BatchEncoding

import interpretune as it
from circuit_tracer.attribution.targets import CustomTarget, LogitTarget
from circuit_tracer.graph import Graph
from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig

from interpretune.analysis import AnalysisInputs, execute_analysis_op
from interpretune.analysis.backends.circuit_tracer import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.analysis.core import AnalysisStore
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.definitions import (
    extract_concept_latent_state_impl,
    extract_concept_latent_examples_impl,
)
from interpretune.analysis.ops.helpers import _extract_concept_latent_state_from_cache
from interpretune.config.circuit_tracer import CircuitTracerConfig
from interpretune.config import init_analysis_cfgs
from tests import load_dotenv
from tests.nb_experiments.session import resolve_model_spec
from tests.nb_experiments.concept_direction.analysis.concept_direction_analysis import (
    build_prompt_alignment_snapshot,
    capture_context_enhanced_extraction_snapshot,
    compare_top_feature_sets,
    resolve_prompt_alignment_context_index,
)
from tests.runif import RunIf


RUNIF: Any = RunIf


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


@lru_cache(maxsize=1)
def _load_gemma3_1b_it_alignment_tokenizer() -> Any:
    load_dotenv()
    model_spec = resolve_model_spec("gemma3", "1b_it")
    auth_token = os.getenv("HF_GATED_PUBLIC_REPO_AUTH_KEY") or os.getenv("HF_TOKEN")
    tokenizer_kwargs = {"token": auth_token} if auth_token else {}
    return AutoTokenizer.from_pretrained(model_spec.model_name, **tokenizer_kwargs)


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
        self._analysis_backend = DEFAULT_CT_ANALYSIS_BACKEND


class _NNsightLikeProducerModule:
    def __init__(self) -> None:
        self.model = _NNsightLikeModel()
        self.datamodule = SimpleNamespace(tokenizer=self.model.tokenizer)
        self._analysis_backend = DEFAULT_CT_ANALYSIS_BACKEND


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
        cache_key = "unembed.hook_in.hook_sae_acts_post"
        cache_rows = [[[0.0, 0.0], row] for row in latent_rows]
        cache = {}
        if names_filter("unembed.hook_in.hook_sae_input"):
            cache["unembed.hook_in.hook_sae_input"] = torch.tensor(cache_rows, dtype=torch.float32)
        if names_filter(cache_key):
            cache[cache_key] = torch.tensor(cache_rows, dtype=torch.float32)
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

    result = it.concept_direction(
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
def test_concept_direction_public_op_uses_schema_default_mode(module_factory) -> None:
    module = module_factory()

    result = it.concept_direction(
        module,
        AnalysisBatch(concept_group_a=["Paris"], concept_group_b=["London"]),
        batch=None,
        batch_idx=0,
    )

    assert result.concept_direction_mode == "mean_difference"


@pytest.mark.parametrize(
    "module_factory",
    [
        pytest.param(_TLLikeProducerModule, id="transformerlens"),
        pytest.param(_NNsightLikeProducerModule, id="nnsight"),
    ],
)
def test_concept_direction_single_group_embed_supports_group_a_only(module_factory) -> None:
    module = module_factory()

    result = it.concept_direction(
        module,
        AnalysisBatch(concept_group_a=["Paris"], concept_direction_mode="single_group"),
        batch=None,
        batch_idx=0,
    )

    expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    assert torch.allclose(result.concept_direction, expected, atol=1e-6)
    assert result.concept_group_a_token_ids == [0]
    assert result.concept_group_b_token_ids == []
    assert result.concept_direction_mode == "single_group"


def test_extract_concept_latent_examples_filters_correct_rows_and_applies_logit_diff_weights() -> None:
    module = SimpleNamespace()
    analysis_batch = AnalysisBatch(
        cache={
            "unembed.hook_in.hook_sae_acts_post": torch.tensor(
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
        concept_cache_key="unembed.hook_in.hook_sae_acts_post",
        concept_weight_by_logit_diff=1,
    )

    source_batch = extract_concept_latent_state_impl(module, analysis_batch, batch=None, batch_idx=0)
    result = extract_concept_latent_examples_impl(module, source_batch, batch=None, batch_idx=0)

    assert torch.equal(
        result.concept_latent_state,
        torch.tensor([[3.0, 0.0], [0.0, 4.0], [1.0, 3.0]], dtype=torch.float32),
    )
    assert torch.equal(result.concept_group_id, torch.tensor([0, 1, 1], dtype=torch.long))
    assert result.concept_group_name == ["capital", "state", "state"]
    assert torch.equal(result.concept_example_logit_diff, torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32))
    assert torch.equal(result.concept_example_weight, torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32))


def test_extract_concept_latent_examples_allows_missing_group_b_labels() -> None:
    module = SimpleNamespace()
    analysis_batch = AnalysisBatch(
        cache={
            "unembed.hook_in.hook_sae_acts_post": torch.tensor(
                [
                    [[0.0, 0.0], [3.0, 0.0]],
                    [[0.0, 0.0], [2.0, 1.0]],
                ],
                dtype=torch.float32,
            )
        },
        answer_indices=torch.tensor([1, 1], dtype=torch.long),
        orig_labels=torch.tensor([0, 0], dtype=torch.long),
        logit_diffs=torch.tensor([2.0, 0.5], dtype=torch.float32),
        concept_group_a_label_ids=[0],
        concept_group_a_name="ohio_entities",
        concept_cache_key="unembed.hook_in.hook_sae_acts_post",
        concept_weight_by_logit_diff=1,
    )

    source_batch = extract_concept_latent_state_impl(module, analysis_batch, batch=None, batch_idx=0)
    result = extract_concept_latent_examples_impl(module, source_batch, batch=None, batch_idx=0)

    assert torch.equal(result.concept_group_id, torch.tensor([0, 0], dtype=torch.long))
    assert result.concept_group_name == ["ohio_entities", "ohio_entities"]
    assert torch.equal(result.concept_example_weight, torch.tensor([2.0, 0.5], dtype=torch.float32))


@RUNIF(standalone=True)
def test_prompt_alignment_snapshot_reports_probe_and_previous_tokens() -> None:
    tokenizer = _load_gemma3_1b_it_alignment_tokenizer()
    rendered_prompt = "Is this an Ohio entity or Indiana entity? Columbus : Ohio"
    input_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    snapshot = build_prompt_alignment_snapshot(
        tokenizer,
        rendered_prompt,
        probe_text="Columbus",
        answer_text="Ohio",
        add_special_tokens=False,
    )

    assert snapshot.input_ids == tuple(int(value) for value in input_ids)
    assert snapshot.input_tokens == tuple(str(token) for token in input_tokens)
    assert snapshot.probe_start_index == 9
    assert snapshot.probe_end_index == 9
    assert snapshot.answer_index == 11
    assert snapshot.answer_start_index == 11
    assert snapshot.answer_end_index == 11
    assert snapshot.answer_token_text == "▁Ohio"
    assert snapshot.previous_token_index == 10
    assert snapshot.previous_token_text == "▁:"
    assert snapshot.intervening_token_texts == ("▁:",)
    assert resolve_prompt_alignment_context_index(snapshot) == (9, "probe_end")


def test_context_enhanced_extraction_snapshot_matches_helper_projection() -> None:
    analysis_batch = AnalysisBatch(
        cache={
            "unembed.hook_in": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]],
                dtype=torch.float32,
            )
        },
        answer_indices=torch.tensor([2], dtype=torch.long),
        context_token_indices=torch.tensor([0], dtype=torch.long),
        concept_cache_key="unembed.hook_in",
    )

    snapshot = capture_context_enhanced_extraction_snapshot(analysis_batch, context_scale=2.0)
    latent_states, _ = _extract_concept_latent_state_from_cache(
        analysis_batch,
        context_enhanced=True,
        context_scale=2.0,
    )

    assert snapshot.answer_indices == (2,)
    assert snapshot.context_source == "context_token_indices"
    assert snapshot.context_indices == (0,)
    assert torch.allclose(snapshot.answer_states, torch.tensor([[1.0, 2.0, 0.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(snapshot.context_states, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(snapshot.dot_num, torch.tensor([[2.0]], dtype=torch.float32))
    assert torch.allclose(snapshot.dot_den, torch.tensor([[1.0]], dtype=torch.float32))
    assert torch.allclose(snapshot.final_latent_states, latent_states, atol=1e-5)
    assert snapshot.projection_basis == "context_state"


def test_context_enhanced_extraction_snapshot_matches_answer_basis_projection() -> None:
    analysis_batch = AnalysisBatch(
        cache={
            "unembed.hook_in": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]],
                dtype=torch.float32,
            )
        },
        answer_indices=torch.tensor([2], dtype=torch.long),
        context_token_indices=torch.tensor([1], dtype=torch.long),
        concept_cache_key="unembed.hook_in",
        use_answer_state_as_basis=True,
    )

    snapshot = capture_context_enhanced_extraction_snapshot(
        analysis_batch,
        context_scale=2.0,
        use_answer_state_as_basis=True,
    )
    latent_states, _ = _extract_concept_latent_state_from_cache(
        analysis_batch,
        context_enhanced=True,
        context_scale=2.0,
        use_answer_state_as_basis=True,
    )

    expected = torch.tensor([[2.2, 4.4, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(snapshot.projected_states, expected, atol=1e-5)
    assert torch.allclose(snapshot.final_latent_states, latent_states, atol=1e-5)
    assert snapshot.projection_basis == "answer_state"


def test_concept_direction_accepts_per_example_rows_without_manual_stacking() -> None:
    extracted_batches = [
        AnalysisBatch(
            concept_latent_state=torch.tensor([[3.0, 0.0]], dtype=torch.float32),
            concept_group_id=torch.tensor([0], dtype=torch.long),
            concept_group_name=["ohio_entities"],
            concept_example_weight=torch.tensor([2.0], dtype=torch.float32),
        ),
        AnalysisBatch(
            concept_latent_state=torch.tensor([[0.0, 4.0]], dtype=torch.float32),
            concept_group_id=torch.tensor([1], dtype=torch.long),
            concept_group_name=["indiana_entities"],
            concept_example_weight=torch.tensor([1.0], dtype=torch.float32),
        ),
    ]

    result = it.concept_direction(
        _TLLikeProducerModule(),
        AnalysisBatch(
            concept_latent_state=[batch.concept_latent_state for batch in extracted_batches],
            concept_group_id=[batch.concept_group_id for batch in extracted_batches],
            concept_group_name=[batch.concept_group_name for batch in extracted_batches],
            concept_example_weight=[batch.concept_example_weight for batch in extracted_batches],
            concept_direction_mode="mean_difference",
            concept_group_a_name="ohio_entities",
            concept_group_b_name="indiana_entities",
        ),
        batch=None,
        batch_idx=0,
    )

    expected = torch.tensor([3.0, -4.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected, atol=1e-6)
    assert result.concept_label == "ohio_entities -> indiana_entities"
    assert result.concept_direction_mode == "mean_difference"


def test_compare_top_feature_sets_reports_overlap_and_score_cosine() -> None:
    summary = compare_top_feature_sets(
        [(0, 0, 10), (1, 0, 12), (2, 0, 15)],
        [(1, 0, 12), (2, 0, 15), (3, 0, 20)],
        left_scores=torch.tensor([0.9, 0.5, 0.25], dtype=torch.float32),
        right_scores=torch.tensor([0.45, 0.2, 0.1], dtype=torch.float32),
        left_label="embed",
        right_label="store",
    )

    assert summary.shared == ((1, 0, 12), (2, 0, 15))
    assert summary.left_only == ((0, 0, 10),)
    assert summary.right_only == ((3, 0, 20),)
    assert summary.jaccard == pytest.approx(0.5)
    assert summary.shared_score_cosine is not None and summary.shared_score_cosine > 0


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

    result = it.concept_direction(
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

    graph_result = it.compute_attribution_graph(
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
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]])},
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]])},
        ],
        answer_indices=[torch.tensor([1, 1], dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)],
        orig_labels=[torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)],
        logit_diffs=[torch.tensor([2.0, 1.0], dtype=torch.float32), torch.tensor([-0.25, 0.5], dtype=torch.float32)],
    )
    analysis_cfg = it.AnalysisCfg(
        name="latent_example_rows",
        target_op=[it.extract_concept_latent_state, it.extract_concept_latent_examples],
        ignore_manual=True,
    )
    init_analysis_cfgs(module, analysis_cfg, ignore_manual=True)

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
                concept_cache_key="unembed.hook_in.hook_sae_acts_post",
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
    init_analysis_cfgs(module, aggregate_cfg, ignore_manual=True)
    previous_cfg = module.analysis_cfg
    result = execute_analysis_op(
        module,
        analysis_batch=AnalysisBatch(concept_direction_mode="mean_difference"),
        analysis_cfg=aggregate_cfg,
    )
    assert module.analysis_cfg is previous_cfg

    expected = torch.tensor([8.0 / 3.0, -11.0 / 3.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected, atol=1e-4)
    assert result.concept_label == "capital -> state"
    assert result.concept_direction_mode == "mean_difference"


def test_extract_concept_latent_examples_final_batch_is_directly_consumable(tmp_path) -> None:
    module = _FakeConceptExtractionModule()
    module.core_log_dir = tmp_path
    datamodule = module.datamodule
    extraction_inputs = SimpleNamespace(
        cache=[
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]])},
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]])},
        ],
        answer_indices=[torch.tensor([1, 1], dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)],
        orig_labels=[torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)],
        logit_diffs=[torch.tensor([2.0, 1.0], dtype=torch.float32), torch.tensor([-0.25, 0.5], dtype=torch.float32)],
    )
    analysis_cfg = it.AnalysisCfg(
        name="latent_example_rows",
        target_op=[it.extract_concept_latent_state, it.extract_concept_latent_examples],
        ignore_manual=True,
    )
    init_analysis_cfgs(module, analysis_cfg, ignore_manual=True)

    extracted_batches = []
    for batch_idx, batch in enumerate(datamodule.test_dataloader()):
        extracted_batches.append(
            execute_analysis_op(
                module,
                batch,
                batch_idx,
                analysis_batch=AnalysisBatch(
                    concept_group_a_label_ids=[0],
                    concept_group_b_label_ids=[1],
                    concept_group_a_name="capital",
                    concept_group_b_name="state",
                    concept_label="capital -> state",
                    concept_direction_mode="mean_difference",
                    concept_cache_key="unembed.hook_in.hook_sae_acts_post",
                    concept_weight_by_logit_diff=True,
                    concept_aggregate_output_mode="in_memory",
                ),
                analysis_cfg=analysis_cfg,
                analysis_inputs=AnalysisInputs(store=extraction_inputs),
            )
        )

    final_extracted = extracted_batches[-1]
    assert len(final_extracted.concept_latent_state_rows) == 2
    assert final_extracted.concept_latent_state_rows[0].shape == (2, 2)
    assert final_extracted.concept_group_id_rows[1].shape == (1,)

    result = it.concept_direction(module, final_extracted, batch=None, batch_idx=0)

    expected = torch.tensor([8.0 / 3.0, -11.0 / 3.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected, atol=1e-4)
    assert result.concept_label == "capital -> state"
    assert result.concept_direction_mode == "mean_difference"


def _run_chained_concept_direction_for_aggregate_mode(
    module,
    datamodule,
    extraction_inputs,
    *,
    aggregate_mode: str,
    shared_store,
    direction_mode: str = "mean_difference",
):
    """Drive extract_concept_latent_state + extract_concept_latent_examples + concept_direction per batch.

    Returns the list of analysis batches emitted by the chained op for each input batch.
    """
    analysis_cfg = it.AnalysisCfg(
        name=f"chained_concept_direction_{aggregate_mode}_{direction_mode}",
        target_op=[
            it.extract_concept_latent_state,
            it.extract_concept_latent_examples,
            it.concept_direction,
        ],
        ignore_manual=True,
    )
    init_analysis_cfgs(module, analysis_cfg, ignore_manual=True)

    emitted_batches = []
    for batch_idx, batch in enumerate(datamodule.test_dataloader()):
        emitted_batches.append(
            execute_analysis_op(
                module,
                batch,
                batch_idx,
                analysis_batch=AnalysisBatch(
                    concept_group_a_label_ids=[0],
                    concept_group_b_label_ids=[1],
                    concept_group_a_name="capital",
                    concept_group_b_name="state",
                    concept_label="capital -> state",
                    concept_direction_mode=direction_mode,
                    concept_cache_key="unembed.hook_in.hook_sae_acts_post",
                    concept_weight_by_logit_diff=True,
                    concept_aggregate_output_mode=aggregate_mode,
                ),
                analysis_cfg=analysis_cfg,
                analysis_inputs=AnalysisInputs(store=shared_store),
            )
        )
    return emitted_batches


def test_concept_direction_streaming_matches_in_memory_aggregate(tmp_path) -> None:
    """Explicit batched-vs-aggregate parity: streaming and in_memory must converge to the same direction."""
    expected = torch.tensor([8.0 / 3.0, -11.0 / 3.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)

    streaming_module = _FakeConceptExtractionModule()
    streaming_module.core_log_dir = tmp_path / "streaming"
    streaming_inputs = SimpleNamespace(
        cache=[
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]])},
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]])},
        ],
        answer_indices=[torch.tensor([1, 1], dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)],
        orig_labels=[torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)],
        logit_diffs=[torch.tensor([2.0, 1.0], dtype=torch.float32), torch.tensor([-0.25, 0.5], dtype=torch.float32)],
    )
    streaming_batches = _run_chained_concept_direction_for_aggregate_mode(
        streaming_module,
        streaming_module.datamodule,
        streaming_inputs,
        aggregate_mode="streaming",
        shared_store=streaming_inputs,
    )
    streaming_final = streaming_batches[-1]

    in_memory_module = _FakeConceptExtractionModule()
    in_memory_module.core_log_dir = tmp_path / "in_memory"
    in_memory_inputs = SimpleNamespace(
        cache=[
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]])},
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]])},
        ],
        answer_indices=[torch.tensor([1, 1], dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)],
        orig_labels=[torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)],
        logit_diffs=[torch.tensor([2.0, 1.0], dtype=torch.float32), torch.tensor([-0.25, 0.5], dtype=torch.float32)],
    )
    in_memory_batches = _run_chained_concept_direction_for_aggregate_mode(
        in_memory_module,
        in_memory_module.datamodule,
        in_memory_inputs,
        aggregate_mode="in_memory",
        shared_store=in_memory_inputs,
    )
    in_memory_final = in_memory_batches[-1]

    # Both modes must match the analytical expected mean-difference direction within tight tolerance.
    assert torch.allclose(streaming_final.concept_direction, expected, atol=1e-5)
    assert torch.allclose(in_memory_final.concept_direction, expected, atol=1e-5)
    # And the two modes must agree with each other to even tighter tolerance.
    assert torch.allclose(streaming_final.concept_direction, in_memory_final.concept_direction, atol=1e-6)

    assert streaming_final.concept_direction_mode == "mean_difference"
    assert in_memory_final.concept_direction_mode == "mean_difference"
    assert streaming_final.get("concept_aggregate_output_mode") == "streaming"
    assert streaming_final.concept_label == "capital -> state"
    assert in_memory_final.concept_label == "capital -> state"


def test_concept_direction_streaming_paired_rejection_matches_in_memory(tmp_path) -> None:
    """Paired-rejection parity: streaming pending-buffer impl must match the legacy in_memory math.

    Inputs are constructed with matched (group_a, group_b) counts per batch so that pair structure
    is preserved by both code paths. Pair index = stable iteration order across batches (matches
    legacy ``zip`` semantics).
    """
    # Two batches, each with one (group_a=label 0, group_b=label 1) pair, both correct.
    # Cache layout is (batch, seq=2, d_model=2); answer_indices=[1,1] selects the seq=1 row.
    #   batch 0: a=[3,0], b=[0,4]; weights from |logit_diffs| = [2.0, 1.0]
    #   batch 1: a=[1,1], b=[1,3]; weights from |logit_diffs| = [1.0, 0.5]
    cache_batches = [
        {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]])},
        {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]])},
    ]
    answer_indices = [torch.tensor([1, 1], dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)]
    orig_labels = [torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)]
    logit_diffs = [
        torch.tensor([2.0, 1.0], dtype=torch.float32),
        torch.tensor([1.0, 0.5], dtype=torch.float32),
    ]

    streaming_module = _FakeConceptExtractionModule()
    streaming_module.core_log_dir = tmp_path / "streaming"
    streaming_inputs = SimpleNamespace(
        cache=cache_batches,
        answer_indices=answer_indices,
        orig_labels=orig_labels,
        logit_diffs=logit_diffs,
    )
    streaming_batches = _run_chained_concept_direction_for_aggregate_mode(
        streaming_module,
        streaming_module.datamodule,
        streaming_inputs,
        aggregate_mode="streaming",
        shared_store=streaming_inputs,
        direction_mode="paired_rejection",
    )
    streaming_final = streaming_batches[-1]

    in_memory_module = _FakeConceptExtractionModule()
    in_memory_module.core_log_dir = tmp_path / "in_memory"
    in_memory_inputs = SimpleNamespace(
        cache=cache_batches,
        answer_indices=answer_indices,
        orig_labels=orig_labels,
        logit_diffs=logit_diffs,
    )
    in_memory_batches = _run_chained_concept_direction_for_aggregate_mode(
        in_memory_module,
        in_memory_module.datamodule,
        in_memory_inputs,
        aggregate_mode="in_memory",
        shared_store=in_memory_inputs,
        direction_mode="paired_rejection",
    )
    in_memory_final = in_memory_batches[-1]

    # Analytical expected:
    #   pair 0: a=[3,0], b=[0,4]; proj=(0/16)*b=[0,0]; residual=[3,0]; pair_w=(2+1)/2=1.5
    #   pair 1: a=[1,1], b=[1,3]; proj=(4/10)*b=[0.4,1.2]; residual=[0.6,-0.2]; pair_w=(1+0.5)/2=0.75
    #   sum = 1.5*[3,0] + 0.75*[0.6,-0.2] = [4.95, -0.15]; total_w = 2.25
    #   direction = [2.2, -0.0666...]; normalized below.
    expected = torch.tensor([4.95, -0.15], dtype=torch.float32) / torch.tensor(2.25, dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)

    assert torch.allclose(streaming_final.concept_direction, expected, atol=1e-5)
    assert torch.allclose(in_memory_final.concept_direction, expected, atol=1e-5)
    assert torch.allclose(streaming_final.concept_direction, in_memory_final.concept_direction, atol=1e-6)
    assert streaming_final.concept_direction_mode == "paired_rejection"
    assert in_memory_final.concept_direction_mode == "paired_rejection"
    assert streaming_final.get("concept_aggregate_output_mode") == "streaming"

    # Storage contract: streaming must populate paired-rejection running totals on the store and
    # leave no unmatched pending rows after the final batch (counts are balanced per batch).
    assert getattr(streaming_inputs, "concept_running_residual_sum", None) is not None
    assert float(getattr(streaming_inputs, "concept_running_pair_weight")) == pytest.approx(2.25, abs=1e-5)
    assert int(getattr(streaming_inputs, "concept_pending_a_states").shape[0]) == 0
    assert int(getattr(streaming_inputs, "concept_pending_b_states").shape[0]) == 0


def test_concept_direction_streaming_paired_rejection_handles_unmatched_batches(tmp_path) -> None:
    """Streaming paired-rejection must defer pair computation across batches when groups are skewed.

    Constructs two batches where the per-batch group counts are imbalanced, but the overall counts match. The pending-
    buffer mechanism should match (a, b) pairs across batch boundaries.
    """
    # batch 0: 1 group_a only (label=0). batch 1: 1 group_b only (label=1). Pair forms across batches.
    # Cache is (batch, seq=2, d_model=2); answer_indices=[1] selects seq=1.
    cache_batches = [
        {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])},
        {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [0.0, 5.0]]])},
    ]
    answer_indices = [torch.tensor([1], dtype=torch.long), torch.tensor([1], dtype=torch.long)]
    orig_labels = [torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)]
    logit_diffs = [torch.tensor([4.0], dtype=torch.float32), torch.tensor([2.0], dtype=torch.float32)]

    streaming_module = _FakeConceptExtractionModule()
    streaming_module.core_log_dir = tmp_path / "streaming"
    streaming_inputs = SimpleNamespace(
        cache=cache_batches,
        answer_indices=answer_indices,
        orig_labels=orig_labels,
        logit_diffs=logit_diffs,
    )
    streaming_batches = _run_chained_concept_direction_for_aggregate_mode(
        streaming_module,
        streaming_module.datamodule,
        streaming_inputs,
        aggregate_mode="streaming",
        shared_store=streaming_inputs,
        direction_mode="paired_rejection",
    )

    # After batch 0 (only group_a present), no pair has been formed yet -> direction is zero.
    assert torch.allclose(streaming_batches[0].concept_direction, torch.zeros(2), atol=1e-7)
    # After batch 1, pair (a=[2,0], b=[0,5]) forms: proj=(0)*b=[0,0]; residual=[2,0]; norm=[1,0].
    expected = torch.tensor([1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(streaming_batches[1].concept_direction, expected, atol=1e-6)
    # Pending buffers are drained.
    assert int(getattr(streaming_inputs, "concept_pending_a_states").shape[0]) == 0
    assert int(getattr(streaming_inputs, "concept_pending_b_states").shape[0]) == 0


def test_concept_direction_streaming_does_not_retain_latent_state_rows(tmp_path) -> None:
    """Streaming mode must skip in-memory row accumulation on extract_concept_latent_examples output."""
    module = _FakeConceptExtractionModule()
    module.core_log_dir = tmp_path
    extraction_inputs = SimpleNamespace(
        cache=[
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 4.0]]])},
            {"unembed.hook_in.hook_sae_acts_post": torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 3.0]]])},
        ],
        answer_indices=[torch.tensor([1, 1], dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)],
        orig_labels=[torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)],
        logit_diffs=[torch.tensor([2.0, 1.0], dtype=torch.float32), torch.tensor([-0.25, 0.5], dtype=torch.float32)],
    )

    streaming_batches = _run_chained_concept_direction_for_aggregate_mode(
        module,
        module.datamodule,
        extraction_inputs,
        aggregate_mode="streaming",
        shared_store=extraction_inputs,
    )

    for emitted in streaming_batches:
        # Streaming must NOT retain the per-batch row accumulators (memory-saving contract).
        assert emitted.get("concept_latent_state_rows") is None
        assert emitted.get("concept_group_id_rows") is None

    # And the shared store's running aggregate state must be populated.
    assert getattr(extraction_inputs, "concept_running_state_sum_a", None) is not None
    assert getattr(extraction_inputs, "concept_running_state_sum_b", None) is not None
    assert float(extraction_inputs.concept_running_weight_a) > 0.0
    assert float(extraction_inputs.concept_running_weight_b) > 0.0


def test_concept_direction_ignores_empty_store_rows(tmp_path) -> None:
    module = _FakeConceptExtractionModule()
    module.core_log_dir = tmp_path

    concept_store = AnalysisStore(
        dataset=Dataset.from_dict(
            {
                "concept_latent_state": [
                    [[3.0, 0.0]],
                    [],
                    [[1.0, 1.0], [1.0, 3.0]],
                ],
                "concept_group_id": [[0], [], [0, 1]],
                "concept_group_name": [["capital"], [], ["capital", "state"]],
                "concept_example_weight": [[2.0], [], [1.0, 0.5]],
            }
        ),
        op_output_dataset_path=str(tmp_path / "rows_with_empty_batches"),
    )

    aggregate_cfg = it.AnalysisCfg(
        target_op=it.concept_direction,
        input_store=concept_store,
        ignore_manual=True,
        save_tokens=False,
    )
    init_analysis_cfgs(module, aggregate_cfg, ignore_manual=True)

    result = execute_analysis_op(
        module,
        analysis_batch=AnalysisBatch(concept_direction_mode="mean_difference"),
        analysis_cfg=aggregate_cfg,
    )

    expected = torch.tensor([4.0 / 3.0, -8.0 / 3.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected, atol=1e-4)
    assert result.concept_label == "capital -> state"
    assert result.concept_direction_mode == "mean_difference"


def test_transformerlens_store_round_trip_can_feed_ct_attribution_op(tmp_path) -> None:
    producer = _TLLikeProducerModule()
    produced = it.concept_direction(
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

    result = it.compute_attribution_graph(
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

    result = it.concept_direction(module, AnalysisBatch(), batch=None, batch_idx=1)

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

    result = it.feature_intervention_forward(
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
        result = it.concept_direction(
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


# ---------------------------------------------------------------------------
# Concept-direction quality validation: embed-based vs store-based equivalence
# ---------------------------------------------------------------------------


class _EmbedStoreEquivalenceModule:
    """Module wired so the embed-based path returns vectors from a controlled 8-dim embedding."""

    def __init__(self, embed_weight: torch.Tensor, *, input_store: AnalysisStore | None = None) -> None:
        self._embed = embed_weight
        self.tokenizer = self._build_tokenizer()
        self.model = SimpleNamespace(
            tokenizer=self.tokenizer,
            W_E=embed_weight,
        )
        self.datamodule = SimpleNamespace(tokenizer=self.tokenizer)
        self._analysis_backend = DEFAULT_CT_ANALYSIS_BACKEND
        self.analysis_cfg = (
            SimpleNamespace(input_store=input_store, batch_inputs={}, run_inputs={})
            if input_store is not None
            else None
        )

    @staticmethod
    def _build_tokenizer():
        tok = _FakeTokenizer()
        tok._vocab = {
            "▁Austin": 0,
            "▁Sacramento": 1,
            "▁Olympia": 2,
            "▁Atlanta": 3,
            "▁Texas": 4,
            "▁California": 5,
            "▁Washington": 6,
            "▁Georgia": 7,
        }
        tok.vocab_size = 8
        return tok


def _build_capitals_states_embed_weight() -> torch.Tensor:
    """8-token embedding matrix with controlled geometry for capital/state distinction."""
    return torch.tensor(
        [
            # Capitals (group_a): positive in dims 0-3
            [3.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],  # ▁Austin
            [2.5, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0],  # ▁Sacramento
            [2.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],  # ▁Olympia
            [2.5, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],  # ▁Atlanta
            # States (group_b): positive in dims 4-7
            [0.5, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0],  # ▁Texas
            [0.0, 0.5, 0.0, 0.0, 2.5, 0.5, 1.0, 0.0],  # ▁California
            [0.0, 0.0, 0.5, 0.0, 2.0, 1.5, 0.5, 0.0],  # ▁Washington
            [0.0, 0.0, 0.0, 0.5, 2.5, 1.0, 0.0, 1.0],  # ▁Georgia
        ],
        dtype=torch.float32,
    )


def _compute_embed_paired_rejection(embed_weight: torch.Tensor, ids_a: list[int], ids_b: list[int]) -> torch.Tensor:
    """Reference implementation of paired_rejection on embedding vectors (mirrors parity test)."""
    group_a_embed = embed_weight[torch.tensor(ids_a)].float()
    group_b_embed = embed_weight[torch.tensor(ids_b)].float()
    residuals = []
    for embed_a, embed_b in zip(group_a_embed, group_b_embed):
        denom = torch.dot(embed_b, embed_b).clamp_min(1e-12)
        proj = (torch.dot(embed_a, embed_b) / denom) * embed_b
        residuals.append(embed_a - proj)
    direction = torch.stack(residuals).mean(dim=0)
    return direction / torch.linalg.vector_norm(direction)


def test_concept_direction_store_vs_embed_algebraic_equivalence(tmp_path) -> None:
    """When the store-based path receives embedding vectors as latent states, paired_rejection must produce the
    same direction as the direct embed-based path."""
    embed_weight = _build_capitals_states_embed_weight()
    capitals = ["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"]
    states = ["▁Texas", "▁California", "▁Washington", "▁Georgia"]

    # --- embed-based direction (direct path) ---
    module_embed = _EmbedStoreEquivalenceModule(embed_weight)
    embed_result = it.concept_direction(
        module_embed,
        AnalysisBatch(
            concept_group_a=capitals,
            concept_group_b=states,
            concept_direction_mode="paired_rejection",
        ),
        batch=None,
        batch_idx=0,
    )
    embed_direction = embed_result.concept_direction

    # --- store-based direction (feeding embedding vectors as latent states) ---
    tok = module_embed.tokenizer
    ids_a = [tok.get_vocab()[c] for c in capitals]
    ids_b = [tok.get_vocab()[s] for s in states]
    latent_states_a = embed_weight[torch.tensor(ids_a)]
    latent_states_b = embed_weight[torch.tensor(ids_b)]

    concept_store = _round_trip_store(
        tmp_path,
        "embed_as_latent",
        Dataset.from_dict(
            {
                "concept_latent_state": (latent_states_a.tolist() + latent_states_b.tolist()),
                "concept_group_id": [0] * len(capitals) + [1] * len(states),
                "concept_group_name": ["capital"] * len(capitals) + ["state"] * len(states),
                "concept_example_weight": [1.0] * (len(capitals) + len(states)),
            }
        ),
    )
    module_store = _EmbedStoreEquivalenceModule(embed_weight, input_store=concept_store)
    store_result = it.concept_direction(
        module_store,
        AnalysisBatch(concept_direction_mode="paired_rejection"),
        batch=None,
        batch_idx=0,
    )
    store_direction = store_result.concept_direction

    # --- reference direction ---
    ref_direction = _compute_embed_paired_rejection(embed_weight, ids_a, ids_b)

    # All three should agree
    cos_embed_ref = torch.nn.functional.cosine_similarity(
        embed_direction.unsqueeze(0), ref_direction.unsqueeze(0)
    ).item()
    cos_store_ref = torch.nn.functional.cosine_similarity(
        store_direction.unsqueeze(0), ref_direction.unsqueeze(0)
    ).item()
    cos_embed_store = torch.nn.functional.cosine_similarity(
        embed_direction.unsqueeze(0), store_direction.unsqueeze(0)
    ).item()

    assert cos_embed_ref > 0.999, f"embed vs reference cosine {cos_embed_ref}"
    assert cos_store_ref > 0.999, f"store vs reference cosine {cos_store_ref}"
    assert cos_embed_store > 0.999, f"embed vs store cosine {cos_embed_store}"

    # Both paths should be unit-normalized
    assert torch.isclose(torch.linalg.vector_norm(embed_direction), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(torch.linalg.vector_norm(store_direction), torch.tensor(1.0), atol=1e-5)

    # paired_rejection captures what's unique to group_a; group_a vectors should
    # project higher on average than group_b, but not every group_b vector need
    # be negative (shared components can keep some slightly positive).
    projs_a = torch.stack([torch.dot(embed_weight[i].float(), embed_direction) for i in ids_a])
    projs_b = torch.stack([torch.dot(embed_weight[i].float(), embed_direction) for i in ids_b])
    assert projs_a.mean() > projs_b.mean(), "capital tokens should project higher than state tokens on average"
    assert projs_a.mean() > 0, "capital mean projection should be positive"


def test_concept_direction_store_path_quality_with_distinct_latent_space(tmp_path) -> None:
    """When SAE activations inhabit a distinct space from embeddings, the store-based path must still produce a
    direction that correctly separates the two concept groups."""
    embed_weight = _build_capitals_states_embed_weight()

    # Synthetic SAE activations: intentionally different from embedding space
    # but with clear group structure (capitals vs states)
    capitals_latent = torch.tensor(
        [
            [5.0, 2.0, 0.0],  # ▁Austin
            [4.0, 3.0, 0.0],  # ▁Sacramento
            [4.5, 2.5, 0.0],  # ▁Olympia
            [5.0, 1.5, 0.5],  # ▁Atlanta
        ],
        dtype=torch.float32,
    )
    states_latent = torch.tensor(
        [
            [0.0, 2.0, 5.0],  # ▁Texas
            [0.0, 3.0, 4.0],  # ▁California
            [0.0, 2.5, 4.5],  # ▁Washington
            [0.5, 1.5, 5.0],  # ▁Georgia
        ],
        dtype=torch.float32,
    )

    concept_store = _round_trip_store(
        tmp_path,
        "sae_latent",
        Dataset.from_dict(
            {
                "concept_latent_state": (capitals_latent.tolist() + states_latent.tolist()),
                "concept_group_id": [0, 0, 0, 0, 1, 1, 1, 1],
                "concept_group_name": ["capital"] * 4 + ["state"] * 4,
                "concept_example_weight": [1.0] * 8,
            }
        ),
    )
    module = _EmbedStoreEquivalenceModule(embed_weight, input_store=concept_store)
    result = it.concept_direction(
        module,
        AnalysisBatch(concept_direction_mode="paired_rejection"),
        batch=None,
        batch_idx=0,
    )
    direction = result.concept_direction

    assert torch.isclose(torch.linalg.vector_norm(direction), torch.tensor(1.0), atol=1e-5)

    # paired_rejection captures what's unique to capitals; capitals should project
    # higher than states on average, and the mean gap should be substantial.
    projs_cap = torch.stack([torch.dot(row, direction) for row in capitals_latent])
    projs_st = torch.stack([torch.dot(row, direction) for row in states_latent])
    gap = projs_cap.mean() - projs_st.mean()
    assert gap > 0.5, f"mean projection gap should be substantial, got {gap}"
    assert projs_cap.mean() > 0, "capital mean projection should be positive"

    # dim-0 (the unique capital dimension) should dominate the direction
    abs_direction = direction.abs()
    assert abs_direction[0] > abs_direction[1], "dim-0 (capital-unique) should dominate over shared dim-1"
    assert abs_direction[0] > abs_direction[2], "dim-0 (capital-unique) should dominate over dim-2"


def test_concept_direction_single_group_store_supports_group_a_only(tmp_path) -> None:
    embed_weight = _build_capitals_states_embed_weight()
    latent_rows = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor([2.0, 1.0], dtype=torch.float32)
    concept_store = _round_trip_store(
        tmp_path,
        "single_group_latent",
        Dataset.from_dict(
            {
                "concept_latent_state": latent_rows.tolist(),
                "concept_group_id": [0, 0],
                "concept_group_name": ["ohio_city", "ohio_city"],
                "concept_example_weight": weights.tolist(),
            }
        ),
    )
    module = _EmbedStoreEquivalenceModule(embed_weight, input_store=concept_store)

    result = it.concept_direction(
        module,
        AnalysisBatch(concept_direction_mode="single_group", concept_group_a_name="ohio_city"),
        batch=None,
        batch_idx=0,
    )

    expected = ((latent_rows * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()).float()
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(result.concept_direction, expected, atol=1e-6)
    assert result.concept_direction_mode == "single_group"
    assert result.concept_label == "ohio_city"


def test_concept_direction_paired_rejection_separates_overlapping_groups(tmp_path) -> None:
    """paired_rejection on partially overlapping groups correctly extracts the discriminative signal."""
    embed_weight = _build_capitals_states_embed_weight()
    # Construct latent states with a large shared component and a small discriminative signal
    shared = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    signal_a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    signal_b = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    capitals_latent = torch.stack([shared + signal_a * (1 + i * 0.1) for i in range(4)])
    states_latent = torch.stack([shared + signal_b * (1 + i * 0.1) for i in range(4)])

    concept_store = _round_trip_store(
        tmp_path,
        "overlapping",
        Dataset.from_dict(
            {
                "concept_latent_state": (capitals_latent.tolist() + states_latent.tolist()),
                "concept_group_id": [0, 0, 0, 0, 1, 1, 1, 1],
                "concept_group_name": ["capital"] * 4 + ["state"] * 4,
                "concept_example_weight": [1.0] * 8,
            }
        ),
    )
    module = _EmbedStoreEquivalenceModule(embed_weight, input_store=concept_store)
    result = it.concept_direction(
        module,
        AnalysisBatch(concept_direction_mode="paired_rejection"),
        batch=None,
        batch_idx=0,
    )
    direction = result.concept_direction

    # paired_rejection should project out the shared component and pick up the discriminative signal
    # dim-0 (positive for capitals) and dim-2 (negative, projected out from states)
    assert direction[0] > 0, "paired_rejection should identify dim-0 as capital-discriminative"
    assert direction[2] < 0, "paired_rejection should identify dim-2 as state-discriminative (subtracted)"
    # dim-1 should be near-zero (shared component projected out)
    assert abs(float(direction[1])) < 0.1, f"shared dim-1 should be near-zero, got {direction[1]}"


def test_concept_direction_contextual_store_diverges_from_embed(tmp_path) -> None:
    """When the store receives contextual representations (not raw embeddings), the resulting direction diverges
    from the embed-based direction while each approach still correctly separates its own concept groups.

    This validates the empirical finding that contextual representations encode concept identity differently from static
    unembedding vectors, so direct substitution of store-based contextual directions for embed-based directions is
    expected to produce meaningfully different steering vectors.
    """
    embed_weight = _build_capitals_states_embed_weight()
    capitals = ["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"]
    states = ["▁Texas", "▁California", "▁Washington", "▁Georgia"]

    # --- embed-based direction ---
    module_embed = _EmbedStoreEquivalenceModule(embed_weight)
    embed_result = it.concept_direction(
        module_embed,
        AnalysisBatch(
            concept_group_a=capitals,
            concept_group_b=states,
            concept_direction_mode="paired_rejection",
        ),
        batch=None,
        batch_idx=0,
    )
    embed_direction = embed_result.concept_direction

    # --- contextual latent states: group structure preserved but geometry differs ---
    # These simulate what a model's hidden-state extraction would produce:
    # clear group separation, but the discriminative dimensions don't align with
    # the embedding matrix's layout.
    contextual_capitals = torch.tensor(
        [
            [0.5, 4.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0],  # ▁Austin
            [0.3, 3.5, 0.8, 1.2, 0.0, 0.0, 0.0, 0.0],  # ▁Sacramento
            [0.6, 3.8, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0],  # ▁Olympia
            [0.4, 4.2, 0.2, 1.1, 0.0, 0.0, 0.0, 0.0],  # ▁Atlanta
        ],
        dtype=torch.float32,
    )
    contextual_states = torch.tensor(
        [
            [0.5, 0.0, 0.0, 0.0, 4.0, 0.3, 1.0, 0.2],  # ▁Texas
            [0.3, 0.0, 0.0, 0.0, 3.5, 0.8, 1.2, 0.1],  # ▁California
            [0.6, 0.0, 0.0, 0.0, 3.8, 0.5, 0.8, 0.3],  # ▁Washington
            [0.4, 0.0, 0.0, 0.0, 4.2, 0.2, 1.1, 0.15],  # ▁Georgia
        ],
        dtype=torch.float32,
    )

    concept_store = _round_trip_store(
        tmp_path,
        "contextual_latent",
        Dataset.from_dict(
            {
                "concept_latent_state": (contextual_capitals.tolist() + contextual_states.tolist()),
                "concept_group_id": [0] * 4 + [1] * 4,
                "concept_group_name": ["capital"] * 4 + ["state"] * 4,
                "concept_example_weight": [1.0] * 8,
            }
        ),
    )
    module_store = _EmbedStoreEquivalenceModule(embed_weight, input_store=concept_store)
    store_result = it.concept_direction(
        module_store,
        AnalysisBatch(concept_direction_mode="paired_rejection"),
        batch=None,
        batch_idx=0,
    )
    store_direction = store_result.concept_direction

    # Both should be unit-normalized
    assert torch.isclose(torch.linalg.vector_norm(embed_direction), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(torch.linalg.vector_norm(store_direction), torch.tensor(1.0), atol=1e-5)

    # Directions should diverge: cosine well below 0.999 (the algebraic-equivalence
    # threshold). With real models this is ~0.15; the synthetic geometry here
    # produces an even starker divergence.
    cos_embed_store = torch.nn.functional.cosine_similarity(
        embed_direction.unsqueeze(0), store_direction.unsqueeze(0)
    ).item()
    assert cos_embed_store < 0.5, (
        f"contextual store direction should diverge from embed direction, but cosine was {cos_embed_store:.4f}"
    )

    # Each direction should still separate its own concept groups correctly.
    # Embed direction separates embedding vectors:
    tok = module_embed.tokenizer
    ids_a = [tok.get_vocab()[c] for c in capitals]
    ids_b = [tok.get_vocab()[s] for s in states]
    embed_projs_a = torch.stack([torch.dot(embed_weight[i].float(), embed_direction) for i in ids_a])
    embed_projs_b = torch.stack([torch.dot(embed_weight[i].float(), embed_direction) for i in ids_b])
    assert embed_projs_a.mean() > embed_projs_b.mean(), "embed direction should separate embedding vectors"

    # Store direction separates contextual vectors:
    store_projs_cap = torch.stack([torch.dot(row, store_direction) for row in contextual_capitals])
    store_projs_st = torch.stack([torch.dot(row, store_direction) for row in contextual_states])
    assert store_projs_cap.mean() > store_projs_st.mean(), "store direction should separate contextual vectors"


# -- Context-enhanced extraction mode tests --------------------------------------------------


def test_context_enhanced_projection_math() -> None:
    """Verify context-enhanced extraction projects scaled answer onto context direction via dot product."""
    # Synthetic cache: 1 example, 3 tokens, d_model=4
    # Use non-orthogonal answer/context so the projection is non-trivial.
    cache_tensor = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    answer_indices = torch.tensor([2], dtype=torch.long)
    batch = AnalysisBatch(cache={"unembed.hook_in": cache_tensor}, answer_indices=answer_indices)

    latent_states, _ = _extract_concept_latent_state_from_cache(batch, context_enhanced=True, context_scale=2.0)

    # answer = cache[0, 2] = [1, 2, 0, 0]; context = cache[0, prev=1] = [3, 4, 0, 0]
    # scaled_answer = 2.0 * [1, 2, 0, 0] = [2, 4, 0, 0]
    # dot(scaled, context) = 6 + 16 = 22
    # dot(context, context) = 9 + 16 = 25
    # projected = (22/25) * [3, 4, 0, 0] = [2.64, 3.52, 0, 0]
    expected = torch.tensor([[2.64, 3.52, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(latent_states, expected, atol=1e-5), f"Expected {expected}, got {latent_states}"


def test_context_enhanced_projection_uses_explicit_context_indices() -> None:
    cache_tensor = torch.tensor(
        [[[1.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 2.0, 0.0]]],
        dtype=torch.float32,
    )
    batch = AnalysisBatch(
        cache={"unembed.hook_in": cache_tensor},
        answer_indices=torch.tensor([2], dtype=torch.long),
        context_token_indices=torch.tensor([0], dtype=torch.long),
    )

    latent_states, _ = _extract_concept_latent_state_from_cache(batch, context_enhanced=True, context_scale=2.0)

    expected = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(latent_states, expected, atol=1e-5), f"Expected {expected}, got {latent_states}"


def test_context_enhanced_projection_default_scale() -> None:
    """With default scale=1.0 the dot-product projection formula still applies."""
    cache_tensor = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]],
        dtype=torch.float32,
    )
    answer_indices = torch.tensor([2], dtype=torch.long)
    batch = AnalysisBatch(cache={"unembed.hook_in": cache_tensor}, answer_indices=answer_indices)

    latent_states, _ = _extract_concept_latent_state_from_cache(batch, context_enhanced=True, context_scale=1.0)

    # answer = [50, 60]; context = [30, 40]
    # dot(answer, context) = 1500 + 2400 = 3900
    # dot(context, context) = 900 + 1600 = 2500
    # projected = (3900/2500) * [30, 40] = [46.8, 62.4]
    expected = torch.tensor([[46.8, 62.4]], dtype=torch.float32)
    assert torch.allclose(latent_states, expected, atol=1e-5)


def test_context_enhanced_projection_can_use_answer_state_as_basis() -> None:
    cache_tensor = torch.tensor(
        [[[1.0, 0.0], [3.0, 4.0], [1.0, 2.0]]],
        dtype=torch.float32,
    )
    batch = AnalysisBatch(cache={"unembed.hook_in": cache_tensor}, answer_indices=torch.tensor([2], dtype=torch.long))

    latent_states, _ = _extract_concept_latent_state_from_cache(
        batch,
        context_enhanced=True,
        context_scale=2.0,
        use_answer_state_as_basis=True,
    )

    expected = torch.tensor([[2.2, 4.4]], dtype=torch.float32)
    assert torch.allclose(latent_states, expected, atol=1e-5)


def test_context_enhanced_projection_skips_ans_idx_zero() -> None:
    """When the answer is at position 0 there is no preceding token — returns raw answer."""
    cache_tensor = torch.tensor(
        [[[5.0, 5.0], [6.0, 6.0]]],
        dtype=torch.float32,
    )
    answer_indices = torch.tensor([0], dtype=torch.long)
    batch = AnalysisBatch(cache={"unembed.hook_in": cache_tensor}, answer_indices=answer_indices)

    latent_states, _ = _extract_concept_latent_state_from_cache(batch, context_enhanced=True, context_scale=1.0)

    # answer at position 0 → no valid preceding position → raw answer returned
    expected = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
    assert torch.allclose(latent_states, expected)


def test_context_enhanced_projection_differs_from_default() -> None:
    """Context-enhanced produces a different latent than the raw answer-position extraction."""
    cache_tensor = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        dtype=torch.float32,
    )
    answer_indices = torch.tensor([2], dtype=torch.long)
    batch = AnalysisBatch(cache={"unembed.hook_in": cache_tensor}, answer_indices=answer_indices)

    default_states, _ = _extract_concept_latent_state_from_cache(batch, context_enhanced=False)
    enhanced_states, _ = _extract_concept_latent_state_from_cache(batch, context_enhanced=True, context_scale=1.0)

    assert not torch.equal(enhanced_states, default_states), (
        "Context-enhanced latent should differ from raw answer-position state"
    )


def test_context_enhanced_projection_multiple_examples() -> None:
    """Verify context-enhanced extraction handles a multi-example batch correctly."""
    # 2-example batch: each example has 3 tokens, d_model=2
    cache_tensor = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]],  # example 0
            [[0.5, 0.5], [1.0, 1.0], [4.0, 5.0]],  # example 1
        ],
        dtype=torch.float32,
    )
    answer_indices = torch.tensor([2, 2], dtype=torch.long)
    batch = AnalysisBatch(cache={"unembed.hook_in": cache_tensor}, answer_indices=answer_indices)

    latent_states, _ = _extract_concept_latent_state_from_cache(batch, context_enhanced=True, context_scale=1.0)

    # Example 0: answer=[2,3], context=[0,1]
    # dot([2,3],[0,1])=3, dot([0,1],[0,1])=1 → (3/1)*[0,1] = [0, 3]
    expected_0 = torch.tensor([0.0, 3.0], dtype=torch.float32)
    assert torch.allclose(latent_states[0], expected_0, atol=1e-5)

    # Example 1: answer=[4,5], context=[1,1]
    # dot([4,5],[1,1])=9, dot([1,1],[1,1])=2 → (9/2)*[1,1] = [4.5, 4.5]
    expected_1 = torch.tensor([4.5, 4.5], dtype=torch.float32)
    assert torch.allclose(latent_states[1], expected_1, atol=1e-5)
