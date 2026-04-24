from __future__ import annotations

from contextlib import contextmanager, nullcontext
import gzip
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch

from it_examples.utils.nb_ui_utils import get_topk
from interpretune.utils import DEFAULT_EXPLANATION_TYPE_NAME
from interpretune.utils.neuronpedia_explanations import NeuronpediaLocalExplanationStatus
from tests.concept_direction_approach_parity import concept_direction as concept_direction_module
from tests.concept_direction_approach_parity.concept_direction import (
    NotebookHarnessConfig,
    build_notebook_harness_config,
    execute_concept_latent_extraction_ops,
)
from tests.nb_experiment_harness import pipeline_patterns
from tests.nb_experiment_harness import nb_harness_utils as nb_harness_utils_module
from tests.nb_experiment_harness.nb_harness_utils import (
    _build_graph_analysis_inputs,
    _build_feature_selection_spec,
    _extract_top_features_with_optional_filter,
    _reduce_top_features_result_to_single_feature,
    _summarize_feature_row_deltas,
    _summarize_layer_error_rows,
    get_key_token_ids_and_labels,
    maybe_save_local_neuronpedia_graph,
    prepare_local_explanation_backfill,
    resolve_graph_target_tokens,
    resolve_target_tokens,
)
from tests.configuration import config_modules
from tests.nb_experiment_harness.session import build_test_cfg


class _StubTokenizer:
    def __init__(self, mapping: dict[str, int | list[int]]) -> None:
        self._mapping = mapping
        self._inverse: dict[int, str] = {}
        for key, value in mapping.items():
            token_ids = value if isinstance(value, list) else [value]
            if len(token_ids) == 1:
                self._inverse[int(token_ids[0])] = key

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        value = self._mapping[token]
        return list(value) if isinstance(value, list) else [value]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._inverse.get(token_id, f"<{token_id}>") for token_id in token_ids)


class _DisplayTokenizer:
    def __init__(self) -> None:
        self._tokens = {0: "▁Lansing", 1: "Columbus", 2: "▁Ohio"}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self._tokens[int(token_id)].replace("▁", "") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._tokens[int(token_id)]


_TEST_CONCEPT_PAIR_CONFIG_PATH = (
    Path(__file__).resolve().parent / "archived_cfgs" / "cp_capitals_states_gemma_it.yaml"
).resolve()


def _build_cfg(
    *,
    prompt_render_mode: Literal["plain", "apply_chat_template", "gemma_dataclass"],
    target_tokens: tuple[str, str],
    key_tokens_override: tuple[str, ...] = ("Austin", "Dallas"),
    analysis_mode: Literal["concept_pair", "explicit_embedding_difference", "debug_intervention_pipelines"] = "concept_pair",
    concept_direction_mode: Literal["mean_difference", "paired_rejection", "single_group"] = "paired_rejection",
    explicit_direction_tokens: tuple[str, str] | None = None,
    constrained_feature_selection_refs: tuple[str | tuple[str, str, int, int] | dict[str, object], ...] | None = None,
    enable_zero_softcap: bool = False,
    debug_session_surface_preset: Literal["notebook_default", "parity_surface"] = "notebook_default",
    direct_projection_interventions: dict[str, object] | None = None,
    direct_projection_intervention_hook_pattern: str | None = None,
    direct_projection_intervention_mode: str | None = None,
    direct_projection_intervention_scale_factor: float | None = None,
    direct_projection_intervention_use_intervention_tensor_as_basis: bool | None = None,
    use_localhost: bool = False,
    upload_local_graphs: bool = False,
    local_graph_slug_prefix: str | None = None,
    local_graph_upload_target: str = "localhost",
    local_graph_owner_username: str | None = None,
) -> NotebookHarnessConfig:
    return NotebookHarnessConfig(
        experiment_name="test",
        experiment_config_name="manual",
        model_family="gemma3",
        model_variant="1b_it",
        model_name="google/gemma-3-1b-it",
        transcoder_set="test-set",
        hf_model_head=None,
        neuronpedia_model="gemma-3-1b-it",
        neuronpedia_set="gemmascope-2-transcoder-16k",
        neuronpedia_base_url="https://www.neuronpedia.org",
        concept_pair_name=None,
        concept_pair_config_path=str(_TEST_CONCEPT_PAIR_CONFIG_PATH),
        prompt="Answer with only the missing city name.",
        prompt_render_mode=prompt_render_mode,
        target_tokens=target_tokens,
        target_token_ids=None,
        top_n=10,
        default_scale_factor=10.0,
        scale_factor_sweep=[10.0],
        ablation_n_list=[5],
        enable_sign_aware=False,
        force_device=None,
        work_root=Path("/tmp"),
        analysis_mode=analysis_mode,
        concept_direction_mode=concept_direction_mode,
        explicit_direction_tokens=explicit_direction_tokens,
        enable_zero_softcap=enable_zero_softcap,
        debug_session_surface_preset=debug_session_surface_preset,
        use_localhost=use_localhost,
        upload_local_graphs=upload_local_graphs,
        local_graph_slug_prefix=local_graph_slug_prefix,
        local_graph_upload_target=local_graph_upload_target,
        local_graph_owner_username=local_graph_owner_username,
        key_tokens_override=key_tokens_override,
        constrained_feature_selection_refs=cast(Any, constrained_feature_selection_refs),
        direct_projection_interventions=cast(Any, direct_projection_interventions),
        direct_projection_intervention_hook_pattern=direct_projection_intervention_hook_pattern,
        direct_projection_intervention_mode=direct_projection_intervention_mode,
        direct_projection_intervention_scale_factor=direct_projection_intervention_scale_factor,
        direct_projection_intervention_use_intervention_tensor_as_basis=(
            direct_projection_intervention_use_intervention_tensor_as_basis
        ),
    )


def test_resolve_target_tokens_strips_sentencepiece_prefix_in_chat_mode() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("▁Austin", "▁Dallas"))
    tokenizer = _StubTokenizer({"Austin": 107305, "Dallas": 85968})

    resolved_ids, resolved_tokens = resolve_target_tokens(cfg, tokenizer)

    assert resolved_tokens == ("Austin", "Dallas")
    assert resolved_ids == (107305, 85968)


def test_resolve_target_tokens_preserves_prefixed_tokens_in_plain_mode() -> None:
    cfg = _build_cfg(prompt_render_mode="plain", target_tokens=("▁Austin", "▁Dallas"))
    tokenizer = _StubTokenizer({"▁Austin": 24278, "▁Dallas": 26057})

    resolved_ids, resolved_tokens = resolve_target_tokens(cfg, tokenizer)

    assert resolved_tokens == ("▁Austin", "▁Dallas")
    assert resolved_ids == (24278, 26057)


def test_prepare_local_explanation_backfill_uses_requested_type_for_coverage(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.local_neuronpedia_db_url = "postgres://postgres:postgres@127.0.0.1:5433/postgres"

    observed: dict[str, object] = {}

    def fake_check_local_explanation_coverage(feature_refs, *, local_db_url=None, type_name=None):
        observed["local_db_url"] = local_db_url
        observed["type_name"] = type_name
        return [NeuronpediaLocalExplanationStatus(feature_ref=feature_refs[0], explanation_count=0)]

    def fake_populate_feature_cache_from_local_exports(feature_ref, *, export_roots, cache_dir=None):
        observed["prefetched_feature_url"] = feature_ref.feature_url
        return 3, tmp_path / "feature-1.jsonl.gz"

    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.check_local_explanation_coverage",
        fake_check_local_explanation_coverage,
    )
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils._populate_feature_cache_from_local_exports",
        fake_populate_feature_cache_from_local_exports,
    )
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils._resolve_local_export_roots",
        lambda roots: (tmp_path,),
    )

    result = prepare_local_explanation_backfill(cfg, [(0, 1)], cache_dir=tmp_path)

    assert observed["local_db_url"] == cfg.local_neuronpedia_db_url
    assert observed["type_name"] == DEFAULT_EXPLANATION_TYPE_NAME
    assert result.prefetch_statuses[0].cache_source == "local_export_cache"
    assert result.prefetch_statuses[0].activation_rows == 3


def test_notebook_harness_config_loads_chat_concept_pair_from_yaml() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))

    assert cfg.concept_pair_name == "capitals_states"
    assert cfg.concept_pair.group_a_tokens[:2] == ["Austin", "Sacramento"]
    assert cfg.concept_pair.classification_question.endswith('"Capital" or "State".')
    assert cfg.concept_pair_config_path is not None
    assert cfg.concept_pair_config_path.endswith("cp_capitals_states_gemma_it.yaml")


def test_notebook_harness_config_requires_experiment_key_tokens() -> None:
    try:
        _build_cfg(
            prompt_render_mode="apply_chat_template",
            target_tokens=("Austin", "Dallas"),
            key_tokens_override=(),
        )
    except ValueError as exc:
        assert "KEY_TOKENS" in str(exc)
    else:
        raise AssertionError("Expected NotebookHarnessConfig to require KEY_TOKENS.")


def test_explicit_embedding_difference_mode_requires_explicit_tokens() -> None:
    try:
        _build_cfg(
            prompt_render_mode="apply_chat_template",
            target_tokens=("Austin", "Dallas"),
            analysis_mode="explicit_embedding_difference",
        )
    except ValueError as exc:
        assert "explicit_direction_tokens" in str(exc)
    else:
        raise AssertionError("Expected explicit embedding-difference mode to require explicit_direction_tokens.")


def test_explicit_embedding_difference_mode_sets_expected_properties() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        analysis_mode="explicit_embedding_difference",
        explicit_direction_tokens=("Sacramento", "Austin"),
        enable_zero_softcap=True,
    )

    assert cfg.uses_explicit_embedding_difference
    assert not cfg.supports_store_direction
    assert cfg.analysis_concept_label == "Sacramento - Austin"
    assert cfg.enable_zero_softcap


def test_explicit_embedding_difference_mode_defaults_to_paired_rejection_direction_mode() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        analysis_mode="explicit_embedding_difference",
        explicit_direction_tokens=("Sacramento", "Austin"),
    )

    assert cfg.analysis_direction_mode_name == "paired_rejection"


def test_explicit_concept_direction_mode_override_flows_into_graph_analysis_inputs() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        analysis_mode="explicit_embedding_difference",
        concept_direction_mode="mean_difference",
        explicit_direction_tokens=("Sacramento", "Austin"),
    )

    analysis_batch, _call_kwargs = _build_graph_analysis_inputs(
        cfg,
        _StubTokenizer({"Austin": 1, "Dallas": 2}),
        "prompt",
        direction=torch.tensor([0.1, 0.2], dtype=torch.float32),
        group_a_ids=[1],
        group_b_ids=[2],
    )

    assert analysis_batch.concept_direction_mode == "mean_difference"


def test_build_notebook_harness_config_reads_nested_concept_direction_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    concept_pair_path = (Path(__file__).resolve().parent / "configs" / "cp_ohio_entities_gemma_it.yaml").resolve()
    config_path.write_text(
        "\n".join(
            [
                "EXPERIMENT_NAME: tmp_ohio_explicit",
                "MODEL:",
                "  family: gemma3",
                "  variant: 4b_it",
                "EXPERIMENT:",
                f"  concept_pair_config_path: {concept_pair_path}",
                "PROMPT:",
                '  text: "Answer with only the missing city name. Fact: the US state capital city closest to largest city in Michigan is"',
                "  render_mode: apply_chat_template",
                "  target_tokens: [Columbus, Indianapolis]",
                "  key_tokens: [Columbus, Detroit, Michigan, ▁Lansing, Indianapolis, Cleveland]",
                "ANALYSIS:",
                "  mode: explicit_embedding_difference",
                "  concept_direction_mode: mean_difference",
                "  explicit_direction_tokens: [Columbus, Indianapolis]",
                "NEURONPEDIA:",
                "  use_localhost: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg, _should_cleanup_work_root, _resolved_payload = build_notebook_harness_config(config_path)

    assert cfg.analysis_direction_mode_name == "mean_difference"


def test_notebook_harness_accepts_single_group_direction_mode() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        concept_direction_mode="single_group",
    )

    assert cfg.analysis_direction_mode_name == "single_group"


def test_execute_concept_latent_extraction_ops_uses_core_context_enhanced_path(monkeypatch) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    observed_kwargs: list[dict[str, object]] = []
    observed_context_indices: list[list[torch.Tensor]] = []
    original_state = torch.tensor([[9.0, 10.0]], dtype=torch.float32)

    def fake_execute_analysis_op(*args, **kwargs):
        del args
        observed_kwargs.append({key: kwargs[key] for key in ("context_enhanced", "context_scale")})
        observed_context_indices.append(list(kwargs["analysis_inputs"].store.context_token_indices))
        return SimpleNamespace(
            concept_latent_state=original_state.clone(),
            concept_group_id=torch.tensor([0], dtype=torch.long),
            concept_example_weight=torch.tensor([1.0], dtype=torch.float32),
            concept_group_name=["capital"],
        )

    monkeypatch.setattr(concept_direction_module, "execute_analysis_op", fake_execute_analysis_op)

    extracted = execute_concept_latent_extraction_ops(
        SimpleNamespace(),
        cfg,
        cached_batches=[{"unembed.hook_in": torch.tensor([[[1.0, 0.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32)}],
        answer_indices=[torch.tensor([2], dtype=torch.long)],
        context_token_indices=[torch.tensor([1], dtype=torch.long)],
        orig_labels=[torch.tensor([0], dtype=torch.long)],
        logit_diffs=[torch.tensor([1.0], dtype=torch.float32)],
        n_prompts=1,
        extraction_mode="context_enhanced",
    )

    assert observed_kwargs == [{"context_enhanced": True, "context_scale": cfg.context_enhanced_scale}]
    assert len(observed_context_indices) == 1
    assert torch.equal(observed_context_indices[0][0], torch.tensor([1], dtype=torch.long))
    assert torch.equal(extracted[0].concept_latent_state, original_state)


def test_execute_concept_latent_extraction_ops_uses_in_memory_rows_for_debug_artifacts(monkeypatch) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.debug_pipeline_state_artifacts = True
    observed_modes: list[str] = []

    def fake_execute_analysis_op(*args, **kwargs):
        del args
        analysis_batch = kwargs["analysis_batch"]
        observed_modes.append(str(analysis_batch.get("concept_aggregate_output_mode")))
        assert analysis_batch.get("concept_direction_mode") == "paired_rejection"
        return SimpleNamespace(
            concept_latent_state_rows=[torch.tensor([[1.0, 2.0]], dtype=torch.float32)],
            concept_group_id_rows=[torch.tensor([0], dtype=torch.long)],
            concept_group_name_rows=[["capital"]],
            concept_example_weight_rows=[torch.tensor([1.0], dtype=torch.float32)],
        )

    monkeypatch.setattr(concept_direction_module, "execute_analysis_op", fake_execute_analysis_op)

    extracted = execute_concept_latent_extraction_ops(
        SimpleNamespace(),
        cfg,
        cached_batches=[{"unembed.hook_in": torch.tensor([[[1.0, 0.0], [3.0, 4.0]]], dtype=torch.float32)}],
        answer_indices=[torch.tensor([1], dtype=torch.long)],
        context_token_indices=[torch.tensor([0], dtype=torch.long)],
        orig_labels=[torch.tensor([0], dtype=torch.long)],
        logit_diffs=[torch.tensor([1.0], dtype=torch.float32)],
        n_prompts=1,
    )

    assert observed_modes == ["in_memory"]
    assert torch.equal(extracted[0].concept_latent_state_rows[0], torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(extracted[0].concept_group_id_rows[0], torch.tensor([0], dtype=torch.long))


def test_execute_concept_latent_extraction_ops_uses_streaming_mode_without_debug_artifacts(monkeypatch) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    observed_modes: list[str] = []

    def fake_execute_analysis_op(*args, **kwargs):
        del args
        analysis_batch = kwargs["analysis_batch"]
        observed_modes.append(str(analysis_batch.get("concept_aggregate_output_mode")))
        assert analysis_batch.get("concept_direction_mode") == "paired_rejection"
        return SimpleNamespace(
            concept_latent_state_rows=None,
            concept_group_id_rows=None,
            concept_group_name_rows=None,
            concept_example_weight_rows=None,
            concept_direction=torch.zeros(2, dtype=torch.float32),
        )

    monkeypatch.setattr(concept_direction_module, "execute_analysis_op", fake_execute_analysis_op)

    extracted = execute_concept_latent_extraction_ops(
        SimpleNamespace(),
        cfg,
        cached_batches=[{"unembed.hook_in": torch.tensor([[[1.0, 0.0], [3.0, 4.0]]], dtype=torch.float32)}],
        answer_indices=[torch.tensor([1], dtype=torch.long)],
        context_token_indices=[torch.tensor([0], dtype=torch.long)],
        orig_labels=[torch.tensor([0], dtype=torch.long)],
        logit_diffs=[torch.tensor([1.0], dtype=torch.float32)],
        n_prompts=1,
    )

    assert observed_modes == ["streaming"]
    assert extracted[0].concept_latent_state_rows is None
    assert extracted[0].concept_group_id_rows is None


def test_compute_store_direction_runs_streaming_aggregation_incrementally(monkeypatch) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.debug_pipeline_state_artifacts = False

    class _FakeModel:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    tokenizer = _StubTokenizer(
        {
            "Austin": 101,
            "Dallas": 202,
            "Sacramento": 303,
            "California": 404,
            "Olympia": 505,
            "Washington": 606,
        }
    )

    def tolerant_encode(token: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        value = tokenizer._mapping.get(token, 999)
        return list(value) if isinstance(value, list) else [value]

    tokenizer.encode = tolerant_encode  # type: ignore[method-assign]

    fake_module = SimpleNamespace(
        model=_FakeModel(),
        replacement_model=SimpleNamespace(tokenizer=tokenizer),
        _model_backend=object(),
    )

    @contextmanager
    def fake_experiment_session(*args, **kwargs):
        del args, kwargs
        yield (None, fake_module, tokenizer)

    observed_calls: list[tuple[int, object]] = []
    shared_analysis_inputs = SimpleNamespace(store=SimpleNamespace())
    extracted_batches = [
        SimpleNamespace(
            concept_aggregate_output_mode="streaming",
            concept_latent_state_rows=None,
            concept_group_id_rows=None,
            concept_group_name_rows=None,
            concept_example_weight_rows=None,
            concept_latent_state=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            concept_group_id=torch.tensor([0], dtype=torch.long),
            concept_group_name=["capital"],
            concept_example_weight=torch.tensor([1.0], dtype=torch.float32),
        ),
        SimpleNamespace(
            concept_aggregate_output_mode="streaming",
            concept_latent_state_rows=None,
            concept_group_id_rows=None,
            concept_group_name_rows=None,
            concept_example_weight_rows=None,
            concept_latent_state=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            concept_group_id=torch.tensor([1], dtype=torch.long),
            concept_group_name=["state"],
            concept_example_weight=torch.tensor([1.0], dtype=torch.float32),
        ),
    ]

    monkeypatch.setattr(concept_direction_module, "experiment_session", fake_experiment_session)
    monkeypatch.setattr(concept_direction_module, "maybe_zero_softcap", lambda module, cfg: nullcontext())
    monkeypatch.setattr(concept_direction_module, "resolve_target_tokens", lambda cfg, tokenizer: ((101, 202), ("Austin", "Dallas")))
    monkeypatch.setattr(
        concept_direction_module,
        "construct_concept_pair_analysis_inputs",
        lambda *args, **kwargs: (
            [{"unembed.hook_in": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)}],
            [torch.tensor([0], dtype=torch.long)],
            [torch.tensor([0], dtype=torch.long)],
            [torch.tensor([0], dtype=torch.long)],
            [torch.tensor([1.0], dtype=torch.float32)],
            {"examples": [{"context_token_source": "probe_end"}]},
            [{"prompt_alignment_artifact": {}}],
        ),
    )

    def fake_execute_concept_latent_extraction_ops(*args, **kwargs):
        del args
        assert kwargs["return_analysis_inputs"] is True
        return extracted_batches, shared_analysis_inputs

    monkeypatch.setattr(
        concept_direction_module,
        "execute_concept_latent_extraction_ops",
        fake_execute_concept_latent_extraction_ops,
    )

    def fake_concept_direction(module, analysis_batch, batch, batch_idx, **kwargs):
        del module, batch
        observed_calls.append((batch_idx, kwargs.get("analysis_inputs")))
        return SimpleNamespace(concept_direction=torch.tensor([0.5, -0.5], dtype=torch.float32))

    monkeypatch.setattr(concept_direction_module.it, "concept_direction", fake_concept_direction)

    result = concept_direction_module.compute_store_direction(cfg)

    assert [batch_idx for batch_idx, _ in observed_calls] == [0, 1]
    assert all(analysis_inputs is shared_analysis_inputs for _, analysis_inputs in observed_calls)
    assert result["n_latent_rows"] == 0
    assert torch.equal(result["direction"], torch.tensor([0.5, -0.5], dtype=torch.float32))


def test_get_topk_preserves_tokenizer_special_markers() -> None:
    logits = torch.tensor([0.1, 0.2, 5.0], dtype=torch.float32)
    tokenizer = _DisplayTokenizer()

    topk = get_topk(logits, tokenizer, k=2)

    assert topk[0][0] == "▁Ohio"
    assert topk[1][0] == "Columbus"


def test_debug_mode_requires_single_constrained_feature() -> None:
    try:
        _build_cfg(
            prompt_render_mode="apply_chat_template",
            target_tokens=("Austin", "Dallas"),
            analysis_mode="debug_intervention_pipelines",
        )
    except ValueError as exc:
        assert "exactly one constrained feature" in str(exc)
    else:
        raise AssertionError("Expected debug mode to require exactly one constrained feature ref.")


def test_debug_mode_uses_key_tokens_as_graph_targets() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        key_tokens_override=("Austin", "Dallas", "Texas"),
        analysis_mode="debug_intervention_pipelines",
        constrained_feature_selection_refs=(("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),),
    )
    tokenizer = _StubTokenizer({"Austin": 107305, "Dallas": 85968, "Texas": 9191})

    target_ids, target_labels = resolve_graph_target_tokens(cfg, tokenizer)

    assert target_ids == [107305, 85968, 9191]
    assert target_labels == ["Austin", "Dallas", "Texas"]


def test_debug_mode_accepts_structured_single_constrained_feature() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        analysis_mode="debug_intervention_pipelines",
        constrained_feature_selection_refs=cast(
            Any,
            {
                "specific_features": [
                    ["gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866],
                ]
            },
        ),
    )

    assert cfg.constrained_feature_selection_refs is not None
    assert len(cfg.constrained_feature_selection_refs.specific_features) == 1


def test_debug_mode_accepts_structured_single_constrained_feature() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        analysis_mode="debug_intervention_pipelines",
        constrained_feature_selection_refs=cast(
            Any,
            {
                "specific_features": [
                    ["gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866],
                ]
            },
        ),
    )

    assert cfg.constrained_feature_selection_refs is not None
    assert len(cfg.constrained_feature_selection_refs.specific_features) == 1


def test_debug_mode_parity_surface_preserves_force_device_when_unset() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        analysis_mode="debug_intervention_pipelines",
        constrained_feature_selection_refs=(("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),),
        debug_session_surface_preset="parity_surface",
    )

    assert cfg.session_kwargs["force_device"] is None
    assert any(
        "preserving the configured or auto-selected device" in message for message in cfg.mode_warning_messages
    )


def test_concept_pair_mode_parity_surface_passes_through_session_kwargs() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        debug_session_surface_preset="parity_surface",
    )

    assert cfg.session_kwargs["debug_session_surface_preset"] == "parity_surface"
    assert any("Notebook sessions will use the parity-aligned session surface" in message for message in cfg.mode_warning_messages)


def test_notebook_harness_config_exposes_graph_upload_session_kwargs() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        use_localhost=True,
        upload_local_graphs=True,
        local_graph_slug_prefix="manual-graph-run",
    )

    assert cfg.session_kwargs["enable_neuronpedia_graph_upload"] is True
    assert cfg.session_kwargs["neuronpedia_graph_slug_prefix"] == "manual-graph-run"
    assert cfg.session_kwargs["neuronpedia_model"] == "gemma-3-1b-it"
    assert cfg.session_kwargs["neuronpedia_source_set"] == "gemmascope-2-transcoder-16k"


def test_resolve_neuronpedia_runtime_config_enables_localhost_for_graph_upload(monkeypatch) -> None:
    monkeypatch.setattr(
        concept_direction_module,
        "check_local_neuronpedia_services",
        lambda **kwargs: SimpleNamespace(webapp_available=True, db_available=True),
    )

    with pytest.warns(UserWarning, match="local graph upload"):
        runtime = concept_direction_module.resolve_neuronpedia_runtime_config(
            use_localhost=False,
            upload_local_graphs=True,
            local_webapp_url="http://localhost:3999",
        )

    assert runtime.use_localhost is True
    assert runtime.upload_local_graphs is True
    assert runtime.local_webapp_url == "http://localhost:3999"


def test_build_test_cfg_enables_neuronpedia_defaults_for_graph_upload() -> None:
    cfg = build_test_cfg(
        "gemma3",
        model_variant="1b_it",
        model_name="google/gemma-3-1b-it",
        transcoder_set="gemma",
        enable_neuronpedia_graph_upload=True,
        neuronpedia_graph_slug_prefix="graph-upload-test",
        neuronpedia_model="gemma-3-1b-it",
        neuronpedia_source_set="gemmascope-2-transcoder-16k",
    )

    assert cfg.neuronpedia_cfg.enabled is True
    assert cfg.neuronpedia_cfg.default_slug_prefix == "graph-upload-test"
    assert cfg.circuit_tracer_cfg.use_neuronpedia is True
    assert cfg.neuronpedia_cfg.default_metadata["info"]["neuronpedia_source_set"] == "gemmascope-2-transcoder-16k"
    assert cfg.neuronpedia_cfg.default_metadata["feature_details"]["neuronpedia_source_set"] == (
        "gemmascope-2-transcoder-16k"
    )


def test_cfg_only_session_preserves_neuronpedia_graph_upload_config(tmp_path: Path) -> None:
    cfg = build_test_cfg(
        "gemma3",
        model_variant="1b_it",
        model_name="google/gemma-3-1b-it",
        transcoder_set="gemma",
        enable_neuronpedia_graph_upload=True,
        neuronpedia_graph_slug_prefix="graph-upload-test",
        neuronpedia_model="gemma-3-1b-it",
        neuronpedia_source_set="gemmascope-2-transcoder-16k",
    )

    session_cfg = config_modules(cfg, "graph-upload-test", {}, tmp_path, {}, False, cfg_only=True)

    assert session_cfg.module_cfg.neuronpedia_cfg.enabled is True
    assert session_cfg.module_cfg.neuronpedia_cfg.default_slug_prefix == "graph-upload-test"
    assert session_cfg.module_cfg.circuit_tracer_cfg.use_neuronpedia is True


def test_maybe_save_local_neuronpedia_graph_uses_local_env_and_metadata(monkeypatch, tmp_path: Path) -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        use_localhost=True,
        upload_local_graphs=True,
        local_graph_slug_prefix="Local Graph Run",
    )
    cfg.work_root = tmp_path
    cfg.local_neuronpedia_webapp_url = "http://localhost:3999"

    observed: dict[str, Any] = {}
    monkeypatch.setenv("DEV_NEURONPEDIA_API_KEY", "dev-key")

    class _FakeNeuronpedia:
        def upload_graph_to_neuronpedia(self, graph_path, api_key=None):
            observed["upload_path"] = str(graph_path)
            observed["api_key"] = api_key
            observed["env_use_localhost"] = os.environ.get("USE_LOCALHOST")
            observed["env_local_webapp_url"] = os.environ.get("LOCAL_NEURONPEDIA_WEBAPP_URL")
            return SimpleNamespace(url="http://localhost:3999/gemma-3-1b-it/graph?slug=test")

    class _FakeModule:
        neuronpedia = _FakeNeuronpedia()

        def save_graph(self, *, graph, output_path, slug=None, custom_metadata=None, use_neuronpedia=None):
            del graph
            observed["slug"] = slug
            observed["custom_metadata"] = custom_metadata
            observed["use_neuronpedia"] = use_neuronpedia
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            graph_path = output_dir / f"{slug}.json"
            graph_path.write_text("{}", encoding="utf-8")
            return graph_path

    artifact = maybe_save_local_neuronpedia_graph(
        cfg,
        _FakeModule(),
        object(),
        phase_label="Embed Pipeline",
        rendered_prompt="The capital of Texas is",
        graph_target_tokens=("Austin", "Dallas"),
        graph_target_ids=[1, 2],
        extra_metadata={"info": {"scale_factor": 10.0}},
    )

    assert artifact is not None
    assert artifact["uploaded"] is True
    assert artifact["graph_url"] == "http://localhost:3999/gemma-3-1b-it/graph?slug=test"
    assert observed["api_key"] == "dev-key"
    assert observed["env_use_localhost"] == "true"
    assert observed["env_local_webapp_url"] == "http://localhost:3999"
    assert observed["use_neuronpedia"] is True
    assert observed["custom_metadata"]["feature_details"]["neuronpedia_source_set"] == "gemmascope-2-transcoder-16k"
    assert observed["custom_metadata"]["info"]["neuronpedia_source_set"] == "gemmascope-2-transcoder-16k"
    assert observed["custom_metadata"]["info"]["scale_factor"] == 10.0
    assert "embed-pipeline" in observed["slug"]


def test_maybe_save_local_neuronpedia_graph_public_upload_syncs_local_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        use_localhost=True,
        upload_local_graphs=True,
        local_graph_slug_prefix="Local Graph Run",
        local_graph_upload_target="public_then_sync_local",
        local_graph_owner_username="speediedan",
    )
    cfg.work_root = tmp_path
    cfg.local_neuronpedia_webapp_url = "http://localhost:3999"

    observed: dict[str, Any] = {}
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "prod-key")
    monkeypatch.setenv("USE_LOCALHOST", "true")

    class _FakeModule:
        neuronpedia = object()

        def save_graph(self, *, graph, output_path, slug=None, custom_metadata=None, use_neuronpedia=None):
            del graph
            observed["slug"] = slug
            observed["custom_metadata"] = custom_metadata
            observed["use_neuronpedia"] = use_neuronpedia
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            graph_path = output_dir / f"{slug}.json"
            graph_path.write_text("{}", encoding="utf-8")
            return graph_path

    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils._upload_graph_for_public_then_sync_local",
        lambda graph_path, *, api_key, public_base_url=None: (
            observed.update(
                upload_path=str(graph_path),
                api_key=api_key,
                env_use_localhost=os.environ.get("USE_LOCALHOST"),
                env_local_webapp_url=os.environ.get("LOCAL_NEURONPEDIA_WEBAPP_URL"),
            )
            or SimpleNamespace(
                graph_metadata=SimpleNamespace(
                    model_id="gemma-3-1b-it",
                    slug="test-public-sync",
                    json_url="https://storage.neuronpedia.org/user-graphs/test-public-sync.json",
                    url="https://neuronpedia.org/gemma-3-1b-it/graph?slug=test-public-sync",
                    url_embed="https://neuronpedia.org/gemma-3-1b-it/graph?slug=test-public-sync&embed=true",
                ),
                public_saved_to_db=True,
                public_save_to_db_error=None,
            )
        ),
    )

    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.sync_graph_metadata_to_local_dev",
        lambda **kwargs: {
            "model_id": "gemma-3-1b-it",
            "slug": "test-public-sync",
            "username": kwargs["username"],
            "graph_json_url": "https://storage.neuronpedia.org/user-graphs/test-public-sync.json",
            "public_graph_url": "https://neuronpedia.org/gemma-3-1b-it/graph?slug=test-public-sync",
            "local_graph_url": "http://localhost:3999/gemma-3-1b-it/graph?slug=test-public-sync",
            "local_graph_embed_url": "http://localhost:3999/gemma-3-1b-it/graph?slug=test-public-sync&embed=true",
            "graph_metadata_id": "graph-row-id",
            "source_set_name": "gemmascope-2-transcoder-16k",
        },
    )

    artifact = maybe_save_local_neuronpedia_graph(
        cfg,
        _FakeModule(),
        object(),
        phase_label="Embed Pipeline",
        rendered_prompt="The capital of Texas is",
        graph_target_tokens=("Austin", "Dallas"),
        graph_target_ids=[1, 2],
    )

    assert artifact is not None
    assert artifact["uploaded"] is True
    assert artifact["upload_target"] == "public_then_sync_local"
    assert artifact["synced_to_local"] is True
    assert artifact["graph_url"] == "http://localhost:3999/gemma-3-1b-it/graph?slug=test-public-sync"
    assert artifact["public_graph_url"] == "https://neuronpedia.org/gemma-3-1b-it/graph?slug=test-public-sync"
    assert observed["api_key"] == "prod-key"
    assert observed["env_use_localhost"] == "false"
    assert observed["env_local_webapp_url"] is None


def test_maybe_save_local_neuronpedia_graph_public_upload_falls_back_to_local_sync_when_public_save_fails(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        use_localhost=True,
        upload_local_graphs=True,
        local_graph_slug_prefix="Local Graph Run",
        local_graph_upload_target="public_then_sync_local",
        local_graph_owner_username="speediedan",
    )
    cfg.work_root = tmp_path
    cfg.local_neuronpedia_webapp_url = "http://localhost:3999"

    class _FakeModule:
        neuronpedia = object()

        def save_graph(self, *, graph, output_path, slug=None, custom_metadata=None, use_neuronpedia=None):
            del graph, custom_metadata, use_neuronpedia
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            graph_path = output_dir / f"{slug}.json"
            graph_path.write_text("{}", encoding="utf-8")
            return graph_path

    monkeypatch.setenv("NEURONPEDIA_API_KEY", "prod-key")
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils._upload_graph_for_public_then_sync_local",
        lambda graph_path, *, api_key, public_base_url=None: SimpleNamespace(
            graph_metadata=SimpleNamespace(
                model_id="gemma-3-1b-it",
                slug="test-public-sync-fallback",
                json_url="https://storage.neuronpedia.org/user-graphs/test-public-sync-fallback.json",
                url=None,
                url_embed=None,
            ),
            public_saved_to_db=False,
            public_save_to_db_error="HTTP 500: Failed to generate signed URL",
        ),
    )
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.sync_graph_metadata_to_local_dev",
        lambda **kwargs: {
            "model_id": "gemma-3-1b-it",
            "slug": "test-public-sync-fallback",
            "username": kwargs["username"],
            "graph_json_url": "https://storage.neuronpedia.org/user-graphs/test-public-sync-fallback.json",
            "public_graph_url": None,
            "local_graph_url": "http://localhost:3999/gemma-3-1b-it/graph?slug=test-public-sync-fallback",
            "local_graph_embed_url": (
                "http://localhost:3999/gemma-3-1b-it/graph?slug=test-public-sync-fallback&embed=true"
            ),
            "graph_metadata_id": "graph-row-id",
            "source_set_name": "gemmascope-2-transcoder-16k",
        },
    )

    artifact = maybe_save_local_neuronpedia_graph(
        cfg,
        _FakeModule(),
        object(),
        phase_label="Embed Pipeline",
        rendered_prompt="The capital of Texas is",
        graph_target_tokens=("Austin", "Dallas"),
        graph_target_ids=[1, 2],
    )

    assert artifact is not None
    assert artifact["uploaded"] is True
    assert artifact["synced_to_local"] is True
    assert artifact["public_saved_to_db"] is False
    assert artifact["public_save_to_db_error"] == "HTTP 500: Failed to generate signed URL"
    assert artifact["graph_url"] == "http://localhost:3999/gemma-3-1b-it/graph?slug=test-public-sync-fallback"
    assert artifact["public_graph_url"] is None


def test_sync_graph_metadata_to_local_dev_uses_public_json_url_and_upsert(monkeypatch) -> None:
    graph_metadata = SimpleNamespace(
        id="graph-row-id",
        model_id="gemma-3-1b-it",
        slug="test-public-sync",
        prompt_tokens=["Hello"],
        prompt="Hello",
        title_prefix="",
        json_url="https://storage.neuronpedia.org/user-graphs/test-public-sync.json",
        url="https://neuronpedia.org/gemma-3-1b-it/graph?slug=test-public-sync",
        url_embed="https://neuronpedia.org/gemma-3-1b-it/graph?slug=test-public-sync&embed=true",
    )
    observed: dict[str, Any] = {"queries": []}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return gzip.compress(
                json.dumps(
                    {
                        "metadata": {
                            "info": {"neuronpedia_source_set": "remote/gemmascope-2-transcoder-16k"},
                        }
                    }
                ).encode("utf-8")
            )

    class _FakeCursor:
        def execute(self, query, params=None):
            observed["queries"].append((query, params))
            self._last_query = query

        def fetchone(self):
            if 'SELECT id FROM "User"' in self._last_query:
                return ("local-user-id",)
            if 'SELECT name FROM public."SourceSet"' in self._last_query:
                return ("gemmascope-2-transcoder-16k",)
            if 'RETURNING id, "modelId", slug, url, "sourceSetName"' in self._last_query:
                return (
                    "graph-row-id",
                    "gemma-3-1b-it",
                    "test-public-sync",
                    "https://storage.neuronpedia.org/user-graphs/test-public-sync.json",
                    "gemmascope-2-transcoder-16k",
                )
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeConnection:
        def cursor(self):
            return _FakeCursor()

        def commit(self):
            observed["committed"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("tests.nb_experiment_harness.nb_harness_utils.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.resolve_local_neuronpedia_db_url",
        lambda local_db_url=None: "postgres://postgres:postgres@127.0.0.1:5433/postgres",
    )
    monkeypatch.setattr("tests.nb_experiment_harness.nb_harness_utils.urlopen", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.psycopg.connect",
        lambda *args, **kwargs: _FakeConnection(),
    )
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.load_dotenv",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setenv("LOCAL_NEURONPEDIA_WEBAPP_URL", "http://localhost:3999")

    from tests.nb_experiment_harness.nb_harness_utils import sync_graph_metadata_to_local_dev

    summary = sync_graph_metadata_to_local_dev(username="speediedan", graph_metadata=graph_metadata)

    assert summary["graph_json_url"] == "https://storage.neuronpedia.org/user-graphs/test-public-sync.json"
    assert summary["local_graph_url"] == "http://localhost:3999/gemma-3-1b-it/graph?slug=test-public-sync"
    assert summary["source_set_name"] == "gemmascope-2-transcoder-16k"
    assert observed["committed"] is True


def test_sync_graph_metadata_to_local_dev_generates_local_id_when_public_metadata_id_missing(monkeypatch) -> None:
    graph_metadata = SimpleNamespace(
        id=None,
        model_id="gemma-3-1b-it",
        slug="test-public-sync-missing-id",
        prompt_tokens=["Hello"],
        prompt="Hello",
        title_prefix="",
        json_url="https://storage.neuronpedia.org/user-graphs/test-public-sync-missing-id.json",
        url=None,
        url_embed=None,
    )
    observed: dict[str, Any] = {"queries": []}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return gzip.compress(
                json.dumps(
                    {
                        "metadata": {
                            "info": {"neuronpedia_source_set": "remote/gemmascope-2-transcoder-16k"},
                        }
                    }
                ).encode("utf-8")
            )

    class _FakeCursor:
        def execute(self, query, params=None):
            observed["queries"].append((query, params))
            self._last_query = query

        def fetchone(self):
            if 'SELECT id FROM "User"' in self._last_query:
                return ("local-user-id",)
            if 'SELECT name FROM public."SourceSet"' in self._last_query:
                return ("gemmascope-2-transcoder-16k",)
            if 'RETURNING id, "modelId", slug, url, "sourceSetName"' in self._last_query:
                return (
                    "local-sync-generated",
                    "gemma-3-1b-it",
                    "test-public-sync-missing-id",
                    "https://storage.neuronpedia.org/user-graphs/test-public-sync-missing-id.json",
                    "gemmascope-2-transcoder-16k",
                )
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeConnection:
        def cursor(self):
            return _FakeCursor()

        def commit(self):
            observed["committed"] = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.urlopen",
        lambda request, timeout=60: _FakeResponse(),
    )
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.resolve_local_neuronpedia_db_url",
        lambda url=None: "postgres://db",
    )
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.psycopg.connect",
        lambda *args, **kwargs: _FakeConnection(),
    )
    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.load_dotenv",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setenv("LOCAL_NEURONPEDIA_WEBAPP_URL", "http://localhost:3999")
    for env_key in (
        "POSTGRES_PRISMA_URL",
        "POSTGRES_PASSWORD",
        "POSTGRES_USER",
        "POSTGRES_URL_NON_POOLING",
        "POSTGRES_DB",
    ):
        monkeypatch.setenv(env_key, "")

    summary = nb_harness_utils_module.sync_graph_metadata_to_local_dev(
        username="speediedan",
        graph_metadata=graph_metadata,
    )

    insert_query, insert_params = next(
        (query, params)
        for query, params in observed["queries"]
        if 'INSERT INTO public."GraphMetadata"' in query
    )
    assert insert_query
    assert str(insert_params[0]).startswith("local-sync-")
    assert summary["graph_metadata_id"] == "local-sync-generated"
    assert observed["committed"] is True


def test_build_graph_analysis_inputs_supports_generic_graph_kwargs_without_concept_direction() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))

    analysis_batch, call_kwargs = _build_graph_analysis_inputs(
        cfg,
        _StubTokenizer({"Austin": 107305, "Dallas": 85968}),
        "prompt",
        direction=None,
        group_a_ids=None,
        group_b_ids=None,
        attribution_targets=["Austin", "Dallas"],
        graph_call_kwargs={"max_n_logits": 5},
        analysis_batch_kwargs={"logit_target_ids": torch.tensor([107305, 85968], dtype=torch.long)},
    )

    assert analysis_batch.prompts == ["prompt"]
    assert analysis_batch.logit_target_ids.tolist() == [107305, 85968]
    assert call_kwargs["attribution_targets"] == ["Austin", "Dallas"]
    assert call_kwargs["max_n_logits"] == 5


def test_get_key_token_ids_and_labels_tracks_prefixed_and_bare_chat_variants() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        key_tokens_override=("Austin", "Texas"),
    )
    tokenizer = _StubTokenizer({" Austin": 24278, "Austin": 107305, " Texas": 69033, "Texas": 9191})

    key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)

    assert key_ids == [24278, 107305, 69033, 9191]
    assert key_labels == ["▁Austin", "Austin", "▁Texas", "Texas"]


def test_get_key_token_ids_and_labels_uses_last_chat_subtoken_label_for_split_token() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        key_tokens_override=("Sacramento",),
    )
    tokenizer = _StubTokenizer(
        {
            " Sacramento": 56183,
            "Sacramento": [89354, 36580],
            "ramento": 36580,
        }
    )

    key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)

    assert key_ids == [56183, 36580]
    assert key_labels == ["▁Sacramento", "ramento"]


def test_build_graph_analysis_inputs_moves_debug_attribution_targets_to_requested_device() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        key_tokens_override=("Austin", "Dallas", "Texas"),
        analysis_mode="debug_intervention_pipelines",
        constrained_feature_selection_refs=(("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),),
    )
    tokenizer = _StubTokenizer({"Austin": 107305, "Dallas": 85968, "Texas": 9191})

    analysis_batch, call_kwargs = _build_graph_analysis_inputs(
        cfg,
        tokenizer,
        "prompt",
        direction=None,
        group_a_ids=None,
        group_b_ids=None,
        attribution_target_device="meta",
    )

    assert analysis_batch.logit_target_ids.device.type == "cpu"
    assert analysis_batch.logit_target_ids.tolist() == [107305, 85968, 9191]
    assert call_kwargs["attribution_targets"].device.type == "meta"


def test_build_feature_selection_spec_matches_layer_and_feature_id_across_positions() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.constrained_feature_selection_refs = cast(Any, ("gemma-3-1b-it/21-gemmascope-2-transcoder-16k/3866",))

    active_features = torch.tensor(
        [
            [21, 12, 3866],
            [21, 13, 3866],
            [21, 12, 7777],
            [20, 12, 3866],
        ],
        dtype=torch.long,
    )

    spec = _build_feature_selection_spec(cfg, active_features)

    assert spec is not None
    assert spec.triples == [(21, 12, 3866), (21, 13, 3866)]


def test_build_feature_selection_spec_accepts_explicit_tuple_refs() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.constrained_feature_selection_refs = cast(
        Any,
        (("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),),
    )

    active_features = torch.tensor(
        [
            [21, 12, 3866],
            [21, 13, 3866],
            [21, 13, 7777],
        ],
        dtype=torch.long,
    )

    spec = _build_feature_selection_spec(cfg, active_features)

    assert spec is not None
    assert spec.triples == [(21, 12, 3866), (21, 13, 3866)]


def test_build_feature_selection_spec_preserves_missing_feature_refs_with_activation_override() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        constrained_feature_selection_refs=(
            {
                "ref": ("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),
                "activation_value": 4.5,
            },
        ),
    )

    active_features = torch.tensor(
        [
            [20, 11, 3866],
            [21, 12, 7777],
        ],
        dtype=torch.long,
    )

    with pytest.warns(UserWarning, match="synthesize candidate rows"):
        spec = _build_feature_selection_spec(cfg, active_features)

    assert spec is not None
    assert spec.triples == []
    assert spec.layer_feature_pairs == [(21, 3866)]
    assert spec.activation_overrides == {(21, 3866): 4.5}


def test_build_feature_selection_spec_supports_structured_feature_selection_mapping() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.constrained_feature_selection_refs = cast(
        Any,
        {
            "specific_features": [
                [21, 12, 3866],
                "gemma-3-1b-it/21-gemmascope-2-transcoder-16k/7777",
            ],
            "layer_slices": [(10, None)],
            "position_slices": [(0, 2)],
        },
    )
    cfg.__post_init__()

    active_features = torch.tensor(
        [
            [9, 0, 111],
            [10, 1, 222],
            [11, 2, 333],
            [21, 12, 3866],
            [21, 13, 7777],
        ],
        dtype=torch.long,
    )

    spec = _build_feature_selection_spec(cfg, active_features)

    assert spec is not None
    assert spec.layers == [10, 11, 21]
    assert spec.positions == [0, 1]
    assert spec.triples == [(21, 12, 3866), (21, 13, 7777)]
    assert spec.layer_feature_pairs == [(21, 7777)]


def test_structured_constrained_feature_selection_serializes_full_mapping() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.constrained_feature_selection_refs = cast(
        Any,
        {
            "specific_features": [
                {
                    "ref": ("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),
                    "activation_value": 4.5,
                },
                [21, 12, 7777],
            ],
            "layer_slices": [(10, None)],
            "position_slices": [(0, 10)],
        },
    )
    cfg.__post_init__()

    serialized = nb_harness_utils_module._serialize_constrained_feature_selection(cfg.constrained_feature_selection_refs)

    assert serialized == {
        "specific_features": [
            {
                "ref": ["gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866],
                "activation_value": 4.5,
            },
            [21, 12, 7777],
        ],
        "layer_slices": [[10, None]],
        "position_slices": [[0, 10]],
    }


def test_build_feature_selection_spec_supports_structured_feature_selection_mapping() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        constrained_feature_selection_refs=cast(
            Any,
            {
                "specific_features": [
                    [21, 12, 3866],
                    "gemma-3-1b-it/21-gemmascope-2-transcoder-16k/7777",
                ],
                "layer_slices": [(10, None)],
                "position_slices": [(0, 2)],
            },
        ),
    )

    active_features = torch.tensor(
        [
            [9, 0, 111],
            [10, 1, 222],
            [11, 2, 333],
            [21, 12, 3866],
            [21, 13, 7777],
        ],
        dtype=torch.long,
    )

    spec = _build_feature_selection_spec(cfg, active_features)

    assert spec is not None
    assert spec.layers == [10, 11, 21]
    assert spec.positions == [0, 1]
    assert spec.triples == [(21, 12, 3866), (21, 13, 7777)]
    assert spec.layer_feature_pairs == [(21, 7777)]


def test_structured_constrained_feature_selection_serializes_full_mapping() -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        constrained_feature_selection_refs=cast(
            Any,
            {
                "specific_features": [
                    {
                        "ref": ("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),
                        "activation_value": 4.5,
                    },
                    [21, 12, 7777],
                ],
                "layer_slices": [(10, None)],
                "position_slices": [(0, 10)],
            },
        ),
    )

    serialized = nb_harness_utils_module._serialize_constrained_feature_selection(cfg.constrained_feature_selection_refs)

    assert serialized == {
        "specific_features": [
            {
                "ref": ["gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866],
                "activation_value": 4.5,
            },
            [21, 12, 7777],
        ],
        "layer_slices": [[10, None]],
        "position_slices": [[0, 10]],
    }


def test_run_direct_projection_pipeline_injects_concept_direction_for_metadata_only_interventions(monkeypatch) -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        direct_projection_interventions={
            "blocks.0.hook_in": {
                "mode": "project",
                "scale_factor": 10.0,
                "use_intervention_tensor_as_basis": True,
            }
        },
    )
    direction = torch.tensor([0.1, 0.2], dtype=torch.float32)
    observed: dict[str, object] = {}

    def fake_shared_run(cfg_arg, label, *, scale_factor, build_analysis_batch):
        observed["cfg"] = cfg_arg
        observed["label"] = label
        observed["scale_factor"] = scale_factor
        observed["analysis_batch"] = build_analysis_batch("prompt", 11, 22, scale_factor)
        return {"ok": True}

    monkeypatch.setattr(concept_direction_module, "_shared_run_direct_projection_pipeline", fake_shared_run)

    result = concept_direction_module.run_direct_projection_pipeline(cfg, direction, "direct_proj", scale_factor=5.0)

    analysis_batch = cast(Any, observed["analysis_batch"])
    assert result == {"ok": True}
    assert observed["label"] == "direct_proj"
    assert observed["scale_factor"] == 5.0
    assert torch.equal(analysis_batch.concept_direction, direction)
    assert analysis_batch.logit_target_ids.tolist() == [11]
    assert analysis_batch.concept_group_a_token_ids == [11]
    assert analysis_batch.concept_group_b_token_ids == [22]
    assert list(analysis_batch.interventions.keys()) == ["blocks.0.hook_in"]
    explicit_spec = analysis_batch.interventions["blocks.0.hook_in"]
    assert explicit_spec["mode"] == "project"
    assert explicit_spec["scale_factor"] == 10.0
    assert explicit_spec["use_intervention_tensor_as_basis"] is True
    assert torch.equal(explicit_spec["intervention_tensor"], direction)


def test_run_direct_projection_pipeline_uses_legacy_shorthand_without_explicit_interventions(monkeypatch) -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        direct_projection_intervention_hook_pattern="blocks.0.hook_in",
        direct_projection_intervention_mode="project",
        direct_projection_intervention_scale_factor=10.0,
        direct_projection_intervention_use_intervention_tensor_as_basis=True,
    )
    direction = torch.tensor([0.1, 0.2], dtype=torch.float32)
    observed: dict[str, object] = {}

    def fake_shared_run(cfg_arg, label, *, scale_factor, build_analysis_batch):
        observed["cfg"] = cfg_arg
        observed["label"] = label
        observed["scale_factor"] = scale_factor
        observed["analysis_batch"] = build_analysis_batch("prompt", 11, 22, scale_factor)
        return {"ok": True}

    monkeypatch.setattr(concept_direction_module, "_shared_run_direct_projection_pipeline", fake_shared_run)

    result = concept_direction_module.run_direct_projection_pipeline(cfg, direction, "direct_proj", scale_factor=5.0)

    analysis_batch = cast(Any, observed["analysis_batch"])
    assert result == {"ok": True}
    assert observed["label"] == "direct_proj"
    assert observed["scale_factor"] == 5.0
    assert torch.equal(analysis_batch.concept_direction, direction)
    assert analysis_batch.logit_target_ids.tolist() == [11]
    assert analysis_batch.concept_group_a_token_ids == [11]
    assert analysis_batch.concept_group_b_token_ids == [22]
    assert getattr(analysis_batch, "interventions", None) is None
    assert analysis_batch.concept_cache_key == cfg.store_concept_cache_key
    assert analysis_batch.intervention_hook_pattern == "blocks.0.hook_in"
    assert analysis_batch.intervention_mode == "project"
    assert analysis_batch.intervention_scale_factor == 10.0
    assert analysis_batch.intervention_use_intervention_tensor_as_basis is True


def test_run_direct_projection_pipeline_builds_explicit_intervention_mapping_for_tensor_specs(monkeypatch) -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        direct_projection_interventions={
            "blocks.0.hook_in": {
                "mode": "project",
                "scale_factor": 10.0,
                "use_intervention_tensor_as_basis": True,
                "intervention_tensor": torch.tensor([3.0, 4.0], dtype=torch.float32),
            }
        },
    )
    direction = torch.tensor([0.1, 0.2], dtype=torch.float32)
    observed: dict[str, object] = {}

    def fake_shared_run(cfg_arg, label, *, scale_factor, build_analysis_batch):
        observed["cfg"] = cfg_arg
        observed["label"] = label
        observed["scale_factor"] = scale_factor
        observed["analysis_batch"] = build_analysis_batch("prompt", 11, 22, scale_factor)
        return {"ok": True}

    monkeypatch.setattr(concept_direction_module, "_shared_run_direct_projection_pipeline", fake_shared_run)

    result = concept_direction_module.run_direct_projection_pipeline(cfg, direction, "direct_proj", scale_factor=5.0)

    analysis_batch = cast(Any, observed["analysis_batch"])
    assert result == {"ok": True}
    assert observed["label"] == "direct_proj"
    assert observed["scale_factor"] == 5.0
    assert torch.equal(analysis_batch.concept_direction, direction)
    assert analysis_batch.logit_target_ids.tolist() == [11]
    assert analysis_batch.concept_group_a_token_ids == [11]
    assert analysis_batch.concept_group_b_token_ids == [22]
    assert list(analysis_batch.interventions.keys()) == ["blocks.0.hook_in"]
    explicit_spec = analysis_batch.interventions["blocks.0.hook_in"]
    assert explicit_spec["mode"] == "project"
    assert explicit_spec["scale_factor"] == 10.0
    assert explicit_spec["use_intervention_tensor_as_basis"] is True
    assert torch.equal(explicit_spec["intervention_tensor"], torch.tensor([3.0, 4.0], dtype=torch.float32))


def test_pipeline_serializer_supports_structured_constrained_feature_refs() -> None:
    raw_ref = concept_direction_module.ConstrainedFeatureSelectionRef(
        ref=("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),
        activation_value=4.5,
    )

    serialized = pipeline_patterns._serialize_constrained_feature_selection_ref(raw_ref)

    assert serialized == {
        "ref": ["gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866],
        "activation_value": 4.5,
    }


def test_extract_top_features_with_optional_filter_passes_feature_selection_to_op(monkeypatch) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.constrained_feature_selection_refs = cast(
        Any,
        (("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),),
    )

    top_payload = {
        "active_features": torch.tensor([[21, 12, 3866]], dtype=torch.long),
        "node_influence_scores": torch.tensor([0.5], dtype=torch.float32),
    }
    fake_result = SimpleNamespace(
        top_feature_ids=torch.tensor([[21, 12, 3866], [9, 4, 9999]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5, 0.4], dtype=torch.float32),
        top_feature_activation_values=torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    observed_feature_selection: dict[str, Any] = {}

    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.it.extract_top_features",
        lambda *args, **kwargs: observed_feature_selection.update(feature_selection=kwargs.get("feature_selection"))
        or fake_result,
    )

    result, applied = _extract_top_features_with_optional_filter(object(), cfg, top_payload, top_n=10)

    assert applied == [(21, 12, 3866)]
    assert observed_feature_selection["feature_selection"] is not None
    assert list(observed_feature_selection["feature_selection"].triples) == [(21, 12, 3866)]
    assert result is fake_result


def test_extract_top_features_with_optional_filter_leaves_op_result_unchanged(monkeypatch) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.constrained_feature_selection_refs = cast(
        Any,
        (("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),),
    )

    top_payload = {
        "active_features": torch.tensor([[21, 12, 3866]], dtype=torch.long),
        "node_influence_scores": torch.tensor([0.5], dtype=torch.float32),
    }
    fake_result = SimpleNamespace(
        top_feature_ids=torch.tensor([[9, 4, 9999]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.4], dtype=torch.float32),
        top_feature_activation_values=torch.tensor([2.0], dtype=torch.float32),
    )

    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.it.extract_top_features",
        lambda *args, **kwargs: fake_result,
    )

    result, applied = _extract_top_features_with_optional_filter(object(), cfg, top_payload, top_n=10)

    assert applied == [(21, 12, 3866)]
    assert result is fake_result
    assert torch.equal(result.top_feature_ids, torch.tensor([[9, 4, 9999]], dtype=torch.long))
    assert torch.allclose(result.top_feature_scores, torch.tensor([0.4], dtype=torch.float32))
    assert torch.allclose(result.top_feature_activation_values, torch.tensor([2.0], dtype=torch.float32))


def test_extract_top_features_with_optional_filter_matches_direct_feature_selection_result() -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.constrained_feature_selection_refs = cast(
        Any,
        (("gemma-3-1b-it", "gemmascope-2-transcoder-16k", 21, 3866),),
    )

    top_payload = {
        "active_features": torch.tensor([[21, 12, 3866], [9, 4, 9999]], dtype=torch.long),
        "node_influence_scores": torch.tensor([0.5, 0.4], dtype=torch.float32),
        "activation_values": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }

    feature_selection = _build_feature_selection_spec(cfg, top_payload["active_features"])
    assert feature_selection is not None

    direct_result = nb_harness_utils_module.it.extract_top_features(
        object(),
        nb_harness_utils_module.it.AnalysisBatch(**top_payload),
        cast(Any, None),
        0,
        top_n=10,
        feature_selection=feature_selection,
    )
    result, applied = _extract_top_features_with_optional_filter(object(), cfg, top_payload, top_n=10)

    assert applied == [(21, 12, 3866)]
    assert torch.equal(result.top_feature_ids, direct_result.top_feature_ids)
    assert torch.allclose(result.top_feature_scores, direct_result.top_feature_scores)
    assert torch.allclose(result.top_feature_activation_values, direct_result.top_feature_activation_values)


def test_reduce_top_features_result_to_single_feature_prefers_largest_abs_score() -> None:
    result = SimpleNamespace(
        top_feature_ids=torch.tensor([[21, 12, 3866], [21, 13, 3866], [21, 14, 3866]], dtype=torch.long),
        top_feature_scores=torch.tensor([0.5, -0.9, 0.4], dtype=torch.float32),
        top_feature_activation_values=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    )

    reduced, candidate_count = _reduce_top_features_result_to_single_feature(result)

    assert candidate_count == 3
    assert torch.equal(reduced.top_feature_ids, torch.tensor([[21, 13, 3866]], dtype=torch.long))
    assert torch.allclose(reduced.top_feature_scores, torch.tensor([-0.9], dtype=torch.float32))
    assert torch.allclose(reduced.top_feature_activation_values, torch.tensor([2.0], dtype=torch.float32))


def test_summarize_feature_row_deltas_orders_rows_by_abs_error() -> None:
    feature_rows = torch.tensor(
        [
            [21, 12, 3866],
            [33, 34, 398],
            [31, 34, 721],
        ],
        dtype=torch.long,
    )
    baseline = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    post = torch.tensor([15.0, 5.0, 45.0], dtype=torch.float32)
    expected_delta = torch.tensor([5.0, -10.0, 2.0], dtype=torch.float32)

    rows = _summarize_feature_row_deltas(feature_rows, baseline, post, expected_delta, top_k=2)

    assert [entry["row"] for entry in rows] == [[31, 34, 721], [33, 34, 398]]
    assert rows[0]["abs_error"] == 13.0
    assert rows[1]["abs_error"] == 5.0
    assert not rows[1]["sign_mismatch"]


def test_summarize_layer_error_rows_aggregates_layer_statistics() -> None:
    feature_rows = torch.tensor(
        [
            [21, 12, 3866],
            [21, 13, 9999],
            [33, 34, 398],
        ],
        dtype=torch.long,
    )
    baseline = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    post = torch.tensor([15.0, 40.0, 28.0], dtype=torch.float32)
    expected_delta = torch.tensor([5.0, 1.0, 4.0], dtype=torch.float32)

    summaries = _summarize_layer_error_rows(feature_rows, baseline, post, expected_delta, top_k=5)

    assert summaries[0]["layer"] == 21
    assert summaries[0]["feature_count"] == 2
    assert summaries[0]["max_abs_error"] == 19.0
    assert summaries[0]["sign_mismatch_count"] == 0
    assert summaries[1]["layer"] == 33
    assert summaries[1]["sign_mismatch_count"] == 1


def test_run_scale_sweep_suppresses_internal_output_on_success(monkeypatch, capsys) -> None:
    cfg = _build_cfg(prompt_render_mode="apply_chat_template", target_tokens=("Austin", "Dallas"))
    cfg.scale_factor_sweep = [2.0, 5.0]

    class _Tokenizer:
        def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
            mapping = {"Austin": 1, "Dallas": 2}
            return [mapping[token]]

    class _Session:
        def __enter__(self):
            return (None, object(), _Tokenizer())

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pipeline_patterns, "experiment_session", lambda *args, **kwargs: _Session())
    monkeypatch.setattr(pipeline_patterns, "resolve_target_tokens", lambda cfg, tokenizer: ((1, 2), ("Austin", "Dallas")))
    monkeypatch.setattr(pipeline_patterns, "get_key_token_ids_and_labels", lambda cfg, tokenizer: ([1, 2], ["Austin", "Dallas"]))
    monkeypatch.setattr(pipeline_patterns, "render_prompt", lambda prompt, tokenizer, mode: prompt)
    monkeypatch.setattr(pipeline_patterns, "configure_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_patterns, "maybe_zero_softcap", lambda module, cfg: nullcontext())

    def _fake_compute_attribution_graph(*args, **kwargs):
        print("noisy stdout from attribution graph")
        sys.stderr.write("noisy stderr from attribution graph\n")
        return {"graph": 1}

    def _fake_graph_node_influence(*args, **kwargs):
        print("noisy stdout from node influence")
        return {"influence": 1}

    def _fake_feature_intervention_forward(*args, **kwargs):
        print("noisy stdout from feature intervention")
        sys.stderr.write("noisy stderr from feature intervention\n")
        return SimpleNamespace(
            pre_intervention_logits=torch.tensor([0.0, 2.0, 1.0]),
            post_intervention_logits=torch.tensor([0.0, 3.5, 0.5]),
        )

    monkeypatch.setattr(pipeline_patterns.it, "compute_attribution_graph", _fake_compute_attribution_graph)
    monkeypatch.setattr(pipeline_patterns.it, "graph_node_influence", _fake_graph_node_influence)
    monkeypatch.setattr(
        pipeline_patterns,
        "_extract_top_features_with_optional_filter",
        lambda *args, **kwargs: (
            SimpleNamespace(
                top_feature_ids=torch.tensor([11]),
                top_feature_scores=torch.tensor([0.5]),
                top_feature_activation_values=torch.tensor([0.25]),
            ),
            [],
        ),
    )
    monkeypatch.setattr(pipeline_patterns.it, "feature_intervention_forward", _fake_feature_intervention_forward)

    results = pipeline_patterns.run_scale_sweep(
        cfg,
        build_graph_analysis_inputs=lambda tokenizer, rendered_prompt: (SimpleNamespace(), {}),
    )

    captured = capsys.readouterr()

    assert len(results) == 2
    assert captured.out == ""
    assert captured.err == ""


def test_pipeline_run_direction_probes_allows_missing_group_b(monkeypatch) -> None:
    cfg = _build_cfg(
        prompt_render_mode="apply_chat_template",
        target_tokens=("Austin", "Dallas"),
        concept_direction_mode="single_group",
    )
    cfg.concept_pair.group_a_tokens = ["Columbus", "Cleveland"]
    cfg.concept_pair.group_b_tokens = []

    class _Tokenizer:
        def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            mapping = {"Columbus": 0, "Cleveland": 1}
            return [mapping[token]]

    class _Session:
        def __enter__(self):
            return (None, object(), _Tokenizer())

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Backend:
        def get_embedding_weight(self, module: object) -> torch.Tensor:
            del module
            return torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            )

    monkeypatch.setattr(pipeline_patterns, "experiment_session", lambda *args, **kwargs: _Session())
    monkeypatch.setattr("interpretune.analysis.backends.circuit_tracer.CircuitTracerAnalysisBackend", _Backend)

    results = pipeline_patterns.run_direction_probes(
        cfg,
        embed_direction=torch.tensor([1.0, 0.5], dtype=torch.float32),
        store_direction=torch.tensor([0.25, 1.0], dtype=torch.float32),
    )

    assert [row["group"] for row in results["Embed"]["rows"]] == ["A", "A"]
    assert results["Embed"]["mean_a"] == pytest.approx(0.75)
    assert results["Store"]["mean_a"] == pytest.approx(0.625)
    assert results["Embed"]["mean_b"] is None
    assert results["Store"]["mean_b"] is None