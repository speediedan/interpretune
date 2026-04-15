from __future__ import annotations

from contextlib import nullcontext
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
from tests.nb_experiment_harness.nb_harness_utils import (
    _build_graph_analysis_inputs,
    _build_feature_selection_spec,
    _extract_top_features_with_optional_filter,
    _reduce_top_features_result_to_single_feature,
    _summarize_feature_row_deltas,
    _summarize_layer_error_rows,
    get_key_token_ids_and_labels,
    prepare_local_explanation_backfill,
    resolve_graph_target_tokens,
    resolve_target_tokens,
)


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
        concept_pair_config_path="cp_capitals_states_gemma_it.yaml",
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
    original_state = torch.tensor([[9.0, 10.0]], dtype=torch.float32)

    def fake_execute_analysis_op(*args, **kwargs):
        del args
        observed_kwargs.append({key: kwargs[key] for key in ("context_enhanced", "context_scale")})
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
        orig_labels=[torch.tensor([0], dtype=torch.long)],
        logit_diffs=[torch.tensor([1.0], dtype=torch.float32)],
        n_prompts=1,
        extraction_mode="context_enhanced",
    )

    assert observed_kwargs == [{"context_enhanced": True, "context_scale": cfg.context_enhanced_scale}]
    assert torch.equal(extracted[0].concept_latent_state, original_state)


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


def test_extract_top_features_with_optional_filter_post_filters_nonmatching_rows(monkeypatch) -> None:
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

    monkeypatch.setattr(
        "tests.nb_experiment_harness.nb_harness_utils.it.extract_top_features",
        lambda *args, **kwargs: fake_result,
    )

    result, applied = _extract_top_features_with_optional_filter(object(), cfg, top_payload, top_n=10)

    assert applied == [(21, 12, 3866)]
    assert torch.equal(result.top_feature_ids, torch.tensor([[21, 12, 3866]], dtype=torch.long))
    assert torch.allclose(result.top_feature_scores, torch.tensor([0.5], dtype=torch.float32))
    assert torch.allclose(result.top_feature_activation_values, torch.tensor([1.0], dtype=torch.float32))


def test_extract_top_features_with_optional_filter_returns_empty_when_requested_feature_missing(monkeypatch) -> None:
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
    assert result.top_feature_ids.shape == (0, 3)
    assert result.top_feature_scores.shape == (0,)
    assert result.top_feature_activation_values.shape == (0,)


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
    monkeypatch.setattr(
        pipeline_patterns,
        "_build_graph_analysis_inputs",
        lambda *args, **kwargs: (SimpleNamespace(), {}),
    )
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
        direction=torch.tensor([0.1, 0.2]),
        group_a_ids=[1],
        group_b_ids=[2],
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