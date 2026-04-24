from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING, Any, cast

import torch

hf_hub_utils: Any = None
try:
    import huggingface_hub.utils as _hf_hub_utils
except ImportError:  # pragma: no cover - huggingface_hub is part of the test env
    pass
else:
    hf_hub_utils = _hf_hub_utils

import interpretune as it  # noqa: E402
import interpretune.analysis  # noqa: E402
from interpretune.analysis.backends import require_analysis_backend  # noqa: E402
from interpretune.analysis.ops.helpers import last_token_logits  # noqa: E402

from tests.nb_experiment_harness.nb_harness_utils import (  # noqa: E402
    _build_classification_prompt,
    _build_graph_analysis_inputs,
    _build_key_token_candidates,
    _build_prompt_batch,
    _extract_top_features_with_optional_filter,
    _get_prompt_debugger,
    _match_feature_row_index,
    _maybe_preserve_debug_intervention_artifacts,
    _reduce_top_features_result_to_single_feature,
    _resolve_model_layer_count,
    _serialize_constrained_feature_selection,
    _serialize_constrained_feature_selection_ref as _serialize_feature_selection_ref,
    _serialize_intervention_call_kwargs,
    _summarize_feature_row_deltas,
    _summarize_graph_input_tokens,
    _summarize_layer_error_rows,
    _summarize_logit_delta_rows,
    _summarize_same_feature_rows,
    _tokenize_rendered_prompt,
    _topk_token_summaries,
    configure_analysis,
    feature_ids_to_tuples,
    get_key_token_ids_and_labels,
    maybe_save_local_neuronpedia_graph,
    maybe_zero_softcap,
    phase_run_name,
    render_prompt,
    render_prompt_variants,
    resolve_graph_target_tokens,
    resolve_target_tokens,
    resolve_key_tokens,
    summarize_gap,
    tensor_to_cpu,
)
from tests.nb_experiment_harness.session import experiment_session  # noqa: E402
from tests.parity_analysis.intervention_drift_analysis import (  # noqa: E402
    build_intervention_drift_report,
    snapshot_analysis_batch,
    snapshot_module_runtime_state,
    tensor_fingerprint,
)
from tests.parity_analysis.concept_direction_parity_analysis import (  # noqa: E402
    build_prompt_alignment_artifact,
    build_prompt_alignment_snapshot,
    normalize_prompt_entity_text,
    resolve_prompt_alignment_context_index,
)

if TYPE_CHECKING:
    from tests.concept_direction_approach_parity.concept_direction import NotebookHarnessConfig


are_progress_bars_disabled = (
    cast(Callable[..., bool], hf_hub_utils.are_progress_bars_disabled) if hf_hub_utils is not None else None
)
disable_progress_bars = (
    cast(Callable[..., None], hf_hub_utils.disable_progress_bars) if hf_hub_utils is not None else None
)
enable_progress_bars = (
    cast(Callable[..., None], hf_hub_utils.enable_progress_bars) if hf_hub_utils is not None else None
)


NULL_BATCH: Any = None
GraphAnalysisInputBuilder = Callable[[Any, str], tuple[Any, dict[str, Any]]]


@contextmanager
def _capture_internal_notebook_output() -> Any:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    previous_disable_level = logging.root.manager.disable
    progress_bars_were_disabled = True
    try:
        if disable_progress_bars is not None and are_progress_bars_disabled is not None:
            progress_bars_were_disabled = are_progress_bars_disabled()
            if not progress_bars_were_disabled:
                disable_progress_bars()
        logging.disable(logging.INFO)
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            yield
    except Exception:
        captured_stdout = stdout_buffer.getvalue()
        captured_stderr = stderr_buffer.getvalue()
        if captured_stdout:
            print(captured_stdout, end="")
        if captured_stderr:
            sys.stderr.write(captured_stderr)
        raise
    finally:
        logging.disable(previous_disable_level)
        if enable_progress_bars is not None and not progress_bars_were_disabled:
            enable_progress_bars()


def _serialize_constrained_feature_selection_ref(raw_ref: Any) -> str | list[Any] | dict[str, Any]:
    serialized = _serialize_feature_selection_ref(raw_ref)
    return cast(str | list[Any] | dict[str, Any], serialized)


def _serialize_requested_constrained_feature_selection(cfg: NotebookHarnessConfig) -> list[Any] | dict[str, Any] | None:
    return _serialize_constrained_feature_selection(cfg.constrained_feature_selection_refs)


def run_initial_sanity_check(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    """Run initial sanity check for the configured intervention prompt and return logit analysis.

    Uses ``cfg.prompt`` (which incorporates PROMPT_OVERRIDE from YAML configs)
    rather than the concept pair's hardcoded prompts.
    """
    raw_prompt = cfg.prompt

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "initial_sanity_check"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered = render_prompt(raw_prompt, tokenizer, cfg.prompt_render_mode)
        enc = _build_prompt_batch(tokenizer, rendered, cfg.prompt_render_mode, module.device)

        with maybe_zero_softcap(module, cfg), torch.inference_mode():
            gen_out = module.model.generate(
                **enc,
                max_new_tokens=1,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )

        gen_text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
        first_logits = gen_out.logits[0][0]
        probs = torch.softmax(first_logits.float(), dim=-1)

        key_analysis: list[dict[str, Any]] = []
        for candidate in _build_key_token_candidates(cfg, tokenizer):
            token_id = int(candidate["token_id"])
            entry: dict[str, Any] = dict(candidate)
            entry["logit"] = float(first_logits[token_id].item())
            entry["prob"] = float(probs[token_id].item())
            key_analysis.append(entry)

        # Sort key tokens by logit magnitude descending
        key_analysis.sort(key=lambda e: abs(e.get("logit", 0.0)), reverse=True)

        top_id = int(first_logits.argmax(dim=-1).item())
        top_token = tokenizer.decode([top_id])
        top_logit = float(first_logits[top_id].item())
        top_prob = float(probs[top_id].item())

        return {
            "prompt_style": "chat" if cfg.use_chat_template else "plain",
            "rendered_prompt": rendered[:400],
            "generated_text": gen_text,
            "key_tokens": key_analysis,
            "top1_token": top_token,
            "top1_id": top_id,
            "top1_logit": top_logit,
            "top1_prob": top_prob,
        }


def collect_baseline_path_debug(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    """Compare the notebook sanity-check path against replacement-model baseline logits."""

    raw_prompt = cfg.prompt
    generation_kwargs = {
        "max_new_tokens": 1,
        "do_sample": False,
        "output_logits": True,
        "return_dict_in_generate": True,
    }

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "baseline_path_debug"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(raw_prompt, tokenizer, cfg.prompt_render_mode)
        prompt_debug = _get_prompt_debugger(module).collect_prompt_debug_info(
            raw_prompt,
            rendered_sequences=rendered_prompt,
            add_special_tokens=cfg.prompt_render_mode == "plain",
        )[0]

        prompt_batch = _build_prompt_batch(tokenizer, rendered_prompt, cfg.prompt_render_mode, module.device)
        prompt_input_ids = cast(torch.Tensor, prompt_batch["input_ids"])[0]

        with maybe_zero_softcap(module, cfg), torch.inference_mode():
            forward_out = module.model(**prompt_batch)
            forward_logits = forward_out.logits if hasattr(forward_out, "logits") else forward_out
            forward_last = forward_logits[0, -1].float().cpu()
            gen_out = module.model.generate(**prompt_batch, **generation_kwargs)
            generate_first = gen_out.logits[0][0].float().cpu()

            replacement_string = (
                last_token_logits(module.replacement_model.get_activations(rendered_prompt)[0]).float().cpu()
            )
            replacement_tokens = (
                last_token_logits(module.replacement_model.get_activations(prompt_input_ids)[0]).float().cpu()
            )

        return {
            "prompt_render_mode": cfg.prompt_render_mode,
            "generation_kwargs": generation_kwargs,
            "prompt_debug": prompt_debug,
            "render_variants": render_prompt_variants(raw_prompt, tokenizer),
            "generated_text": tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True),
            "baseline_sources": {
                "forward_last": _topk_token_summaries(forward_last, tokenizer),
                "generate_first": _topk_token_summaries(generate_first, tokenizer),
                "replacement_from_string": _topk_token_summaries(replacement_string, tokenizer),
                "replacement_from_tokens": _topk_token_summaries(replacement_tokens, tokenizer),
            },
            "max_abs_diffs": {
                "forward_vs_generate": float((forward_last - generate_first).abs().max().item()),
                "forward_vs_replacement_string": float((forward_last - replacement_string).abs().max().item()),
                "forward_vs_replacement_tokens": float((forward_last - replacement_tokens).abs().max().item()),
                "generate_vs_replacement_string": float((generate_first - replacement_string).abs().max().item()),
                "generate_vs_replacement_tokens": float((generate_first - replacement_tokens).abs().max().item()),
            },
        }


def _build_render_variant_equalities(
    render_variants: dict[str, str | None],
    render_variant_token_ids: dict[str, list[int]],
) -> dict[str, bool | None]:
    """Build equality comparisons between render variants, using None when apply_chat_template is unavailable."""
    has_chat = render_variants.get("apply_chat_template") is not None
    return {
        "apply_chat_template_vs_dataclass": (
            render_variants["apply_chat_template"] == render_variants["gemma_dataclass"] if has_chat else None
        ),
        "apply_chat_template_vs_dataclass_token_ids": (
            render_variant_token_ids["apply_chat_template"] == render_variant_token_ids["gemma_dataclass"]
            if has_chat
            else None
        ),
    }


def run_tokenizer_verification(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "tokenizer_verification"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        render_variants = render_prompt_variants(cfg.prompt, tokenizer)
        selected_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        add_special_tokens = cfg.prompt_render_mode == "plain"
        enc = tokenizer(selected_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        resolved_target_ids, resolved_target_tokens = resolve_target_tokens(cfg, tokenizer)
        render_variant_token_ids = {
            mode_name: _tokenize_rendered_prompt(tokenizer, rendered_prompt, mode_name)
            for mode_name, rendered_prompt in render_variants.items()
            if rendered_prompt is not None
        }
        report: dict[str, Any] = {
            "groups": {},
            "key_tokens": {},
            "prompt_token_count": int(enc["input_ids"].shape[-1]),
            "module_type": type(module).__name__,
            "prompt_render_mode": cfg.prompt_render_mode,
            "render_variants": render_variants,
            "render_variant_token_ids": render_variant_token_ids,
            "render_variant_tokens": {
                mode_name: tokenizer.convert_ids_to_tokens(token_ids)
                for mode_name, token_ids in render_variant_token_ids.items()
            },
            "selected_prompt_preview": selected_prompt[:400],
            "selected_prompt_token_ids": enc["input_ids"][0].tolist(),
            "selected_prompt_tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist()),
            "render_variant_equalities": _build_render_variant_equalities(render_variants, render_variant_token_ids),
            "target_tokens": {
                "group_a": {"id": resolved_target_ids[0], "decoded": resolved_target_tokens[0]},
                "group_b": {"id": resolved_target_ids[1], "decoded": resolved_target_tokens[1]},
            },
        }
        for label, tokens in [
            (cfg.concept_pair.group_a_name, cfg.concept_pair.group_a_tokens),
            (cfg.concept_pair.group_b_name, cfg.concept_pair.group_b_tokens),
        ]:
            entries = []
            for token in tokens:
                ids = tokenizer.encode(token, add_special_tokens=False)
                entries.append({"token": token, "ids": ids, "decoded": tokenizer.decode(ids)})
            report["groups"][label] = entries
        for token in resolve_key_tokens(cfg):
            ids = tokenizer.encode(token, add_special_tokens=False)
            report["key_tokens"][token] = {"ids": ids, "decoded": tokenizer.decode(ids)}
        report["resolved_key_token_candidates"] = _build_key_token_candidates(cfg, tokenizer)
        return report


def run_pipeline(
    cfg: NotebookHarnessConfig,
    label: str,
    *,
    scale_factor: float,
    top_n: int,
    build_graph_analysis_inputs: GraphAnalysisInputBuilder,
) -> dict[str, Any]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("run_pipeline is not available in debug_intervention_pipelines mode")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, f"{label}_pipeline_{scale_factor}x"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        (target_a_id, target_b_id), resolved_target_tokens = resolve_target_tokens(cfg, tokenizer)
        key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        configure_analysis(module, it.compute_attribution_graph, scale_factor)
        analysis_batch, graph_call_kwargs = build_graph_analysis_inputs(tokenizer, rendered_prompt)
        graph_artifact = None
        with maybe_zero_softcap(module, cfg):
            graph_result = cast(
                Any,
                it.compute_attribution_graph(
                    module,
                    analysis_batch,
                    NULL_BATCH,
                    0,
                    **graph_call_kwargs,
                ),
            )
            if cfg.upload_local_graphs:
                analysis_backend = require_analysis_backend(module)
                graph = analysis_backend.hydrate_graph_from_batch(graph_result)
                graph_artifact = maybe_save_local_neuronpedia_graph(
                    cfg,
                    module,
                    graph,
                    phase_label=f"{label}_pipeline_{scale_factor}x",
                    rendered_prompt=rendered_prompt,
                    graph_target_tokens=resolved_target_tokens,
                    graph_target_ids=[target_a_id, target_b_id],
                    extra_metadata={"info": {"direction_label": label, "scale_factor": float(scale_factor)}},
                )
            influence_result = cast(Any, it.graph_node_influence(module, graph_result, NULL_BATCH, 0))
            top_payload = dict(cast(Any, graph_result))
            top_payload.update(dict(cast(Any, influence_result)))
            top_features_result, applied_feature_filter_triples = _extract_top_features_with_optional_filter(
                module,
                cfg,
                top_payload,
                top_n=top_n,
            )
            intervention_result = cast(
                Any,
                it.feature_intervention_forward(
                    module,
                    it.AnalysisBatch(
                        prompts=[rendered_prompt],
                        top_feature_ids=top_features_result.top_feature_ids,
                        top_feature_scores=top_features_result.top_feature_scores,
                        top_feature_activation_values=top_features_result.top_feature_activation_values,
                        logit_target_ids=torch.tensor([target_a_id], dtype=torch.long),
                    ),
                    NULL_BATCH,
                    0,
                ),
            )
        pre_logits = tensor_to_cpu(intervention_result.pre_intervention_logits)
        post_logits = tensor_to_cpu(intervention_result.post_intervention_logits)
        pre_gap, post_gap, gap_delta = summarize_gap(pre_logits, post_logits, target_a_id, target_b_id)
        pre_top = torch.topk(pre_logits.float(), 5)
        post_top = torch.topk(post_logits.float(), 5)
        result = {
            "label": label,
            "scale_factor": scale_factor,
            "top_n": top_n,
            "pre_logits": pre_logits,
            "post_logits": post_logits,
            "pre_gap": pre_gap,
            "post_gap": post_gap,
            "gap_delta": gap_delta,
            "feature_ids": feature_ids_to_tuples(top_features_result.top_feature_ids),
            "feature_scores": tensor_to_cpu(top_features_result.top_feature_scores).tolist(),
            "feature_activations": tensor_to_cpu(top_features_result.top_feature_activation_values),
            "target_a_id": target_a_id,
            "target_b_id": target_b_id,
            "target_a_tok": resolved_target_tokens[0],
            "target_b_tok": resolved_target_tokens[1],
            "key_ids": key_ids,
            "key_labels": key_labels,
            "rendered_prompt": rendered_prompt,
            "rendered_prompt_token_ids": _tokenize_rendered_prompt(tokenizer, rendered_prompt, cfg.prompt_render_mode),
            "pre_top_tokens": [tokenizer.decode([int(token_id)]) for token_id in pre_top.indices.tolist()],
            "pre_top_token_ids": [int(token_id) for token_id in pre_top.indices.tolist()],
            "post_top_tokens": [tokenizer.decode([int(token_id)]) for token_id in post_top.indices.tolist()],
            "post_top_token_ids": [int(token_id) for token_id in post_top.indices.tolist()],
            "requested_constrained_feature_selection": _serialize_requested_constrained_feature_selection(cfg),
            "applied_feature_filter_triples": applied_feature_filter_triples,
            "graph_artifact": graph_artifact,
        }
        if cfg.debug_pipeline_state_artifacts:
            result["debug_pipeline_state_artifacts"] = {
                "path_label": f"{label}_graph_pipeline",
                "scale_factor": float(scale_factor),
                "graph_analysis_batch": snapshot_analysis_batch(
                    analysis_batch,
                    fields=(
                        "concept_direction",
                        "concept_direction_mode",
                        "concept_label",
                        "concept_group_a",
                        "concept_group_b",
                        "concept_group_a_token_ids",
                        "concept_group_b_token_ids",
                        "prompts",
                        "logit_target_ids",
                        "context_token_indices",
                    ),
                    max_items=32,
                ),
                "graph_call_kwargs": _serialize_intervention_call_kwargs(graph_call_kwargs),
                "graph_input_tokens": _summarize_graph_input_tokens(
                    tokenizer,
                    rendered_prompt,
                    cfg.prompt_render_mode,
                    graph_result.input_tokens,
                ),
                "graph_result_input_tokens": tensor_to_cpu(graph_result.input_tokens).reshape(-1).tolist(),
                "top_features": feature_ids_to_tuples(top_features_result.top_feature_ids),
                "top_feature_scores": tensor_to_cpu(top_features_result.top_feature_scores).tolist(),
                "top_feature_activation_values": tensor_to_cpu(
                    top_features_result.top_feature_activation_values
                ).tolist(),
                "pre_gap": pre_gap,
                "post_gap": post_gap,
                "gap_delta": gap_delta,
                "pre_logits_fingerprint": tensor_fingerprint(pre_logits),
                "post_logits_fingerprint": tensor_fingerprint(post_logits),
            }
        return result


def run_direct_projection_pipeline(
    cfg: NotebookHarnessConfig,
    label: str,
    *,
    scale_factor: float,
    build_analysis_batch: Callable[[str, int, int, float], Any],
) -> dict[str, Any]:
    """Run a direct ``model_fwd_intervention`` notebook pipeline.

    The shared helper owns prompt rendering, tokenization, session setup, and summary formatting.
    Experiment modules provide a ``build_analysis_batch`` callback that injects any experiment-specific
    intervention payloads, such as concept-direction tensors or concept-group metadata.
    """
    if cfg.is_debug_intervention_mode:
        raise ValueError("run_direct_projection_pipeline is not available in debug_intervention_pipelines mode")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, f"{label}_direct_proj_{scale_factor}x"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        (target_a_id, target_b_id), resolved_target_tokens = resolve_target_tokens(cfg, tokenizer)
        key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        add_special_tokens = cfg.prompt_render_mode == "plain"
        enc = tokenizer(rendered_prompt, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens)
        device = next(module.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict(enc).items()}
        configure_analysis(module, it.model_fwd_intervention, scale_factor)
        analysis_batch = build_analysis_batch(rendered_prompt, target_a_id, target_b_id, scale_factor)
        with maybe_zero_softcap(module, cfg):
            intervention_result = cast(
                Any,
                it.model_fwd_intervention(
                    module,
                    analysis_batch,
                    cast(Any, batch),
                    0,
                ),
            )
        pre_logits = tensor_to_cpu(intervention_result.pre_intervention_logits)
        post_logits = tensor_to_cpu(intervention_result.post_intervention_logits)

        pre_gap, post_gap, gap_delta = summarize_gap(pre_logits, post_logits, target_a_id, target_b_id)
        pre_top = torch.topk(pre_logits.float(), 5)
        post_top = torch.topk(post_logits.float(), 5)
        return {
            "label": label,
            "scale_factor": scale_factor,
            "top_n": 0,  # No feature selection in direct projection
            "pre_logits": pre_logits,
            "post_logits": post_logits,
            "pre_gap": pre_gap,
            "post_gap": post_gap,
            "gap_delta": gap_delta,
            "feature_ids": [],  # No features used
            "feature_scores": [],
            "feature_activations": torch.tensor([]),
            "target_a_id": target_a_id,
            "target_b_id": target_b_id,
            "target_a_tok": resolved_target_tokens[0],
            "target_b_tok": resolved_target_tokens[1],
            "key_ids": key_ids,
            "key_labels": key_labels,
            "rendered_prompt": rendered_prompt,
            "rendered_prompt_token_ids": _tokenize_rendered_prompt(tokenizer, rendered_prompt, cfg.prompt_render_mode),
            "pre_top_tokens": [tokenizer.decode([int(token_id)]) for token_id in pre_top.indices.tolist()],
            "pre_top_token_ids": [int(token_id) for token_id in pre_top.indices.tolist()],
            "post_top_tokens": [tokenizer.decode([int(token_id)]) for token_id in post_top.indices.tolist()],
            "post_top_token_ids": [int(token_id) for token_id in post_top.indices.tolist()],
            "intervention_type": "direct_projection",
        }


def run_direct_projection_scale_sweep(
    cfg: NotebookHarnessConfig,
    *,
    build_pipeline: Callable[[float], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run a scale sweep using an experiment-specific direct-projection pipeline wrapper."""
    return [build_pipeline(scale_factor) for scale_factor in cfg.scale_factor_sweep]


def build_classification_prompt_examples(cfg: NotebookHarnessConfig, tokenizer: Any) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    add_special_tokens = cfg.prompt_render_mode == "plain"
    for group_name, rows in (
        (cfg.concept_pair.group_a_name, cfg.concept_pair.group_a_entities),
        (cfg.concept_pair.group_b_name, cfg.concept_pair.group_b_entities),
    ):
        for entity_name, expected_answer in rows:
            raw_prompt = _build_classification_prompt(entity_name, cfg.concept_pair.classification_question)
            rendered_prompt = render_prompt(raw_prompt, tokenizer, cfg.prompt_render_mode)
            rendered_prompt_with_answer = f"{rendered_prompt}{expected_answer}"
            probe_surface_text = normalize_prompt_entity_text(entity_name)
            snapshot = build_prompt_alignment_snapshot(
                tokenizer,
                rendered_prompt_with_answer,
                probe_text=probe_surface_text,
                answer_text=expected_answer,
                add_special_tokens=add_special_tokens,
            )
            rendered_prompt_token_ids = _tokenize_rendered_prompt(tokenizer, rendered_prompt, cfg.prompt_render_mode)
            if len(rendered_prompt_token_ids) != snapshot.answer_index:
                raise ValueError(
                    "Prompt alignment expected the answer token span to begin immediately after the rendered prompt "
                    f"({len(rendered_prompt_token_ids)} vs {snapshot.answer_index})"
                )
            context_token_index, context_token_source = resolve_prompt_alignment_context_index(snapshot)
            scoring_index = len(rendered_prompt_token_ids) - 1
            cache_answer_index = int(snapshot.answer_index)
            prompt_alignment_artifact = build_prompt_alignment_artifact(
                snapshot,
                probe_surface_text=probe_surface_text,
                cache_rendered_prompt=rendered_prompt_with_answer,
                cache_input_ids=snapshot.input_ids,
                cache_input_tokens=snapshot.input_tokens,
                cache_answer_index=cache_answer_index,
                context_token_index=context_token_index,
                context_token_source=context_token_source,
            )
            examples.append(
                {
                    "entity_name": entity_name,
                    "probe_surface_text": probe_surface_text,
                    "expected_answer": expected_answer,
                    "group_name": group_name,
                    "raw_prompt": raw_prompt,
                    "rendered_prompt": rendered_prompt,
                    "rendered_prompt_with_answer": rendered_prompt_with_answer,
                    "rendered_prompt_token_ids": rendered_prompt_token_ids,
                    "scoring_index": scoring_index,
                    "cache_answer_index": cache_answer_index,
                    "context_token_index": context_token_index,
                    "context_token_source": context_token_source,
                    "prompt_alignment_snapshot": snapshot,
                    "prompt_alignment_artifact": prompt_alignment_artifact,
                }
            )
    return examples


def build_all_prompts(cfg: NotebookHarnessConfig, tokenizer: Any) -> list[tuple[str, str, str]]:
    return [
        (example["rendered_prompt"], example["expected_answer"], example["group_name"])
        for example in build_classification_prompt_examples(cfg, tokenizer)
    ]


def _score_expected_answer(
    tokenizer: Any,
    example_logits: torch.Tensor,
    expected_answer: str,
    target_a_id: int,
    target_b_id: int,
) -> tuple[int, float, bool, int, list[int], list[str]]:
    expected_id = tokenizer.encode(expected_answer, add_special_tokens=False)[-1]
    other_id = target_b_id if expected_id == target_a_id else target_a_id
    topk_ids = torch.topk(example_logits, 10).indices.tolist()
    topk_tokens = [tokenizer.decode([token_id]) for token_id in topk_ids]
    correct = expected_id in topk_ids
    rank = topk_ids.index(expected_id) if correct else -1
    logit_diff = float((example_logits[expected_id] - example_logits[other_id]).item())
    return expected_id, logit_diff, correct, rank, topk_ids, topk_tokens


def run_scale_sweep(
    cfg: NotebookHarnessConfig,
    *,
    build_graph_analysis_inputs: GraphAnalysisInputBuilder,
) -> list[dict[str, Any]]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("run_scale_sweep is not available in debug_intervention_pipelines mode")

    results = []
    for scale_factor in cfg.scale_factor_sweep:
        with _capture_internal_notebook_output():
            with experiment_session(
                cfg.work_root,
                phase_run_name(cfg.experiment_name, f"scale_sweep_{scale_factor}x"),
                **cfg.session_kwargs,
            ) as (_, module, tokenizer):
                (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)
                key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)
                rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
                configure_analysis(module, it.compute_attribution_graph, scale_factor)
                analysis_batch, graph_call_kwargs = build_graph_analysis_inputs(tokenizer, rendered_prompt)
                graph_artifact = None
                with maybe_zero_softcap(module, cfg):
                    graph_result = cast(
                        Any,
                        it.compute_attribution_graph(
                            module,
                            analysis_batch,
                            NULL_BATCH,
                            0,
                            **graph_call_kwargs,
                        ),
                    )
                    if cfg.upload_local_graphs:
                        analysis_backend = require_analysis_backend(module)
                        graph = analysis_backend.hydrate_graph_from_batch(graph_result)
                        graph_artifact = maybe_save_local_neuronpedia_graph(
                            cfg,
                            module,
                            graph,
                            phase_label=f"scale_sweep_{scale_factor}x",
                            rendered_prompt=rendered_prompt,
                            graph_target_tokens=None,
                            graph_target_ids=[target_a_id, target_b_id],
                            extra_metadata={"info": {"scale_factor": float(scale_factor), "sweep": True}},
                        )
                    influence_result = cast(Any, it.graph_node_influence(module, graph_result, NULL_BATCH, 0))
                    top_payload = dict(cast(Any, graph_result))
                    top_payload.update(dict(cast(Any, influence_result)))
                    top_features_result, applied_feature_filter_triples = _extract_top_features_with_optional_filter(
                        module,
                        cfg,
                        top_payload,
                        top_n=cfg.top_n,
                    )
                    intervention_result = cast(
                        Any,
                        it.feature_intervention_forward(
                            module,
                            it.AnalysisBatch(
                                prompts=[rendered_prompt],
                                top_feature_ids=top_features_result.top_feature_ids,
                                top_feature_scores=top_features_result.top_feature_scores,
                                top_feature_activation_values=top_features_result.top_feature_activation_values,
                                logit_target_ids=torch.tensor([target_a_id], dtype=torch.long),
                            ),
                            NULL_BATCH,
                            0,
                        ),
                    )
                pre_logits = tensor_to_cpu(intervention_result.pre_intervention_logits)
                post_logits = tensor_to_cpu(intervention_result.post_intervention_logits)
                pre_gap, post_gap, gap_delta = summarize_gap(pre_logits, post_logits, target_a_id, target_b_id)
            results.append(
                {
                    "scale_factor": float(scale_factor),
                    "pre_logits": pre_logits,
                    "post_logits": post_logits,
                    "pre_gap": pre_gap,
                    "post_gap": post_gap,
                    "gap_delta": gap_delta,
                    "key_ids": key_ids,
                    "key_labels": key_labels,
                    "requested_constrained_feature_selection": _serialize_requested_constrained_feature_selection(cfg),
                    "applied_feature_filter_triples": applied_feature_filter_triples,
                    "graph_artifact": graph_artifact,
                }
            )
    return results


def collect_feature_pool(
    cfg: NotebookHarnessConfig,
    *,
    top_n: int,
    build_graph_analysis_inputs: GraphAnalysisInputBuilder,
) -> dict[str, Any]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("collect_feature_pool is not available in debug_intervention_pipelines mode")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "feature_pool"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)
        key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        configure_analysis(module, it.compute_attribution_graph, 0.0)
        analysis_batch, graph_call_kwargs = build_graph_analysis_inputs(tokenizer, rendered_prompt)
        with maybe_zero_softcap(module, cfg):
            graph_result = cast(
                Any,
                it.compute_attribution_graph(
                    module,
                    analysis_batch,
                    NULL_BATCH,
                    0,
                    **graph_call_kwargs,
                ),
            )
            influence_result = cast(Any, it.graph_node_influence(module, graph_result, NULL_BATCH, 0))
            top_payload = dict(cast(Any, graph_result))
            top_payload.update(dict(cast(Any, influence_result)))
            feature_result, applied_feature_filter_triples = _extract_top_features_with_optional_filter(
                module,
                cfg,
                top_payload,
                top_n=top_n,
            )
        return {
            "feature_ids": tensor_to_cpu(feature_result.top_feature_ids.to(torch.float32)).to(torch.long),
            "feature_scores": tensor_to_cpu(feature_result.top_feature_scores),
            "feature_activations": tensor_to_cpu(feature_result.top_feature_activation_values),
            "target_a_id": target_a_id,
            "target_b_id": target_b_id,
            "key_ids": key_ids,
            "key_labels": key_labels,
            "requested_constrained_feature_selection": _serialize_requested_constrained_feature_selection(cfg),
            "applied_feature_filter_triples": applied_feature_filter_triples,
        }


def run_ablations(
    cfg: NotebookHarnessConfig,
    feature_pool: dict[str, Any],
    pre_logits_ref: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[int, torch.Tensor]]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("run_ablations is not available in debug_intervention_pipelines mode")

    abl_groups = {
        "baseline": {
            label: float(torch.softmax(pre_logits_ref.float(), dim=-1)[token_id].item())
            for label, token_id in zip(feature_pool["key_labels"][:3], feature_pool["key_ids"][:3])
        }
    }
    abl_logit_diffs = {
        "baseline": float(pre_logits_ref[feature_pool["target_a_id"]] - pre_logits_ref[feature_pool["target_b_id"]])
    }
    results: dict[int, torch.Tensor] = {}
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "progressive_ablation"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        with maybe_zero_softcap(module, cfg):
            for n_value in cfg.ablation_n_list:
                if n_value > int(feature_pool["feature_ids"].shape[0]):
                    continue
                intervention_result = cast(
                    Any,
                    it.feature_intervention_forward(
                        module,
                        it.AnalysisBatch(
                            prompts=[rendered_prompt],
                            top_feature_ids=feature_pool["feature_ids"][:n_value],
                            top_feature_scores=feature_pool["feature_scores"][:n_value],
                            top_feature_activation_values=feature_pool["feature_activations"][:n_value] * 0.0,
                            logit_target_ids=torch.tensor([feature_pool["target_a_id"]], dtype=torch.long),
                        ),
                        NULL_BATCH,
                        0,
                    ),
                )
                post_logits = tensor_to_cpu(intervention_result.post_intervention_logits)
                probs = torch.softmax(post_logits, dim=-1)
                label = f"top-{n_value}"
                abl_groups[label] = {
                    name: float(probs[token_id].item())
                    for name, token_id in zip(feature_pool["key_labels"][:3], feature_pool["key_ids"][:3])
                }
                abl_logit_diffs[label] = float(
                    post_logits[feature_pool["target_a_id"]] - post_logits[feature_pool["target_b_id"]]
                )
                results[n_value] = post_logits
    return abl_groups, abl_logit_diffs, results


def run_sign_aware(
    cfg: NotebookHarnessConfig,
    feature_pool: dict[str, Any],
    pre_logits_ref: torch.Tensor,
) -> dict[str, Any]:
    if cfg.is_debug_intervention_mode:
        raise ValueError("run_sign_aware is not available in debug_intervention_pipelines mode")

    feature_ids = feature_pool["feature_ids"]
    feature_scores = feature_pool["feature_scores"]
    feature_activations = feature_pool["feature_activations"]
    positive_mask = feature_activations > 0
    negative_mask = feature_activations < 0
    result: dict[str, Any] = {
        "positive_features": feature_ids[positive_mask],
        "negative_features": feature_ids[negative_mask],
        "positive_scores": feature_scores[positive_mask],
        "negative_scores": feature_scores[negative_mask],
        "positive_activations": feature_activations[positive_mask],
        "negative_activations": feature_activations[negative_mask],
        "messages": [],
    }
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "sign_aware"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        with maybe_zero_softcap(module, cfg):
            if len(result["positive_features"]) > 0:
                n_pos = min(cfg.top_n, len(result["positive_features"]))
                pos_intervention = cast(
                    Any,
                    it.feature_intervention_forward(
                        module,
                        it.AnalysisBatch(
                            prompts=[rendered_prompt],
                            top_feature_ids=result["positive_features"][:n_pos],
                            top_feature_scores=result["positive_scores"][:n_pos],
                            top_feature_activation_values=result["positive_activations"][:n_pos]
                            * cfg.default_scale_factor,
                            logit_target_ids=torch.tensor([feature_pool["target_a_id"]], dtype=torch.long),
                        ),
                        NULL_BATCH,
                        0,
                    ),
                )
                result["positive_post_logits"] = tensor_to_cpu(pos_intervention.post_intervention_logits)
            else:
                result["messages"].append(
                    "No positive-activation features were available for the current feature pool."
                )
            if len(result["negative_features"]) > 0:
                n_neg = min(cfg.top_n, len(result["negative_features"]))
                neg_intervention = cast(
                    Any,
                    it.feature_intervention_forward(
                        module,
                        it.AnalysisBatch(
                            prompts=[rendered_prompt],
                            top_feature_ids=result["negative_features"][:n_neg],
                            top_feature_scores=result["negative_scores"][:n_neg],
                            top_feature_activation_values=result["negative_activations"][:n_neg] * 0.0,
                            logit_target_ids=torch.tensor([feature_pool["target_a_id"]], dtype=torch.long),
                        ),
                        cast(Any, None),
                        0,
                    ),
                )
                result["negative_post_logits"] = tensor_to_cpu(neg_intervention.post_intervention_logits)
            else:
                result["messages"].append(
                    "No negative-activation features were available for the current feature pool."
                )
    result["pre_logits_ref"] = pre_logits_ref
    return result


def run_debug_intervention_validation(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    if not cfg.is_debug_intervention_mode:
        raise ValueError("run_debug_intervention_validation is only available in debug_intervention_pipelines mode")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "debug_intervention_validation"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        (target_a_id, target_b_id), resolved_target_tokens = resolve_target_tokens(cfg, tokenizer)
        key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer, include_bare_variants=False)
        graph_target_ids, graph_target_tokens = resolve_graph_target_tokens(cfg, tokenizer)

        configure_analysis(module, it.compute_attribution_graph, cfg.default_scale_factor)
        analysis_batch, graph_call_kwargs = _build_graph_analysis_inputs(
            cfg,
            tokenizer,
            rendered_prompt,
            direction=None,
            group_a_ids=None,
            group_b_ids=None,
            attribution_target_device=module.replacement_model.device,
        )

        with maybe_zero_softcap(module, cfg):
            analysis_backend = require_analysis_backend(module)
            graph_result = cast(
                Any,
                it.compute_attribution_graph(
                    module,
                    analysis_batch,
                    NULL_BATCH,
                    0,
                    **graph_call_kwargs,
                ),
            )
            graph = analysis_backend.hydrate_graph_from_batch(graph_result)
            graph_artifact = maybe_save_local_neuronpedia_graph(
                cfg,
                module,
                graph,
                phase_label="debug_intervention_validation",
                rendered_prompt=rendered_prompt,
                graph_target_tokens=graph_target_tokens,
                graph_target_ids=graph_target_ids,
                extra_metadata={"info": {"debug_intervention_validation": True}},
            )
            influence_result = cast(Any, it.graph_node_influence(module, graph_result, NULL_BATCH, 0))
            top_payload = dict(cast(Any, graph_result))
            top_payload.update(dict(cast(Any, influence_result)))
            top_features_result, applied_feature_filter_triples = _extract_top_features_with_optional_filter(
                module,
                cfg,
                top_payload,
                top_n=cfg.top_n,
            )
            top_features_result, selected_feature_candidate_count = _reduce_top_features_result_to_single_feature(
                top_features_result
            )
            selected_feature_rows = top_features_result.top_feature_ids.to(torch.long)

            active_features = tensor_to_cpu(torch.as_tensor(graph_result.active_features, dtype=torch.long)).to(
                torch.long
            )
            selected_feature_indices = (
                tensor_to_cpu(torch.as_tensor(getattr(graph_result, "selected_features", []), dtype=torch.long))
                .reshape(-1)
                .to(torch.long)
            )
            graph_feature_rows = (
                active_features.index_select(0, selected_feature_indices)
                if selected_feature_indices.numel() > 0
                else active_features
            )
            adjacency_matrix = tensor_to_cpu(torch.as_tensor(graph_result.adjacency_matrix, dtype=torch.float32))
            logit_target_ids = (
                tensor_to_cpu(torch.as_tensor(graph_result.logit_target_ids, dtype=torch.long))
                .reshape(-1)
                .to(torch.long)
            )
            graph_result_target_tokens = [
                str(token) for token in getattr(graph_result, "logit_target_tokens", graph_target_tokens)
            ]
            requested_graph_target_ids = [int(token_id) for token_id in graph_target_ids]
            actual_graph_target_ids = [int(token_id) for token_id in logit_target_ids.tolist()]
            selected_feature = tensor_to_cpu(selected_feature_rows[0]).to(torch.long)
            selected_graph_index = _match_feature_row_index(graph_feature_rows, selected_feature)
            graph_inputs = getattr(graph_result, "input_tokens", rendered_prompt)
            input_diagnostics = _summarize_graph_input_tokens(
                tokenizer,
                rendered_prompt,
                cfg.prompt_render_mode,
                graph_inputs,
            )

            baseline_logits_raw, baseline_activation_cache = module.replacement_model.get_activations(
                graph_inputs,
                apply_activation_function=False,
            )
            baseline_logits = last_token_logits(baseline_logits_raw).float().cpu()
            baseline_activation_cache_t = torch.as_tensor(baseline_activation_cache).detach().cpu().float()

            layer, position, feature_id = (int(value) for value in selected_feature.tolist())
            baseline_activation = float(baseline_activation_cache_t[layer, position, feature_id].item())
            n_layers = _resolve_model_layer_count(module)
            wrapper_inputs = it.AnalysisBatch(
                top_feature_ids=top_features_result.top_feature_ids,
                top_feature_scores=top_features_result.top_feature_scores,
                top_feature_activation_values=top_features_result.top_feature_activation_values,
                logit_target_ids=logit_target_ids,
                prompts=[rendered_prompt],
            )
            wrapper_settings = analysis_backend.resolve_feature_intervention_settings(
                module,
                {"intervention_return_activations": True},
            )
            interventions, _ = analysis_backend.build_feature_interventions(wrapper_inputs, wrapper_settings)
            wrapper_result = cast(
                Any,
                it.feature_intervention_forward(
                    module,
                    wrapper_inputs,
                    NULL_BATCH,
                    0,
                    prompt=graph_inputs,
                    intervention_return_activations=True,
                ),
            )
            wrapper_pre_logits = tensor_to_cpu(
                torch.as_tensor(wrapper_result.pre_intervention_logits, dtype=torch.float32)
            ).reshape(-1)
            wrapper_post_logits = tensor_to_cpu(
                torch.as_tensor(wrapper_result.post_intervention_logits, dtype=torch.float32)
            ).reshape(-1)
            wrapper_activation_cache = getattr(wrapper_result, "intervention_activation_cache", None)
            if wrapper_activation_cache is None:
                raise ValueError("feature_intervention_forward did not return intervention_activation_cache")
            post_activation_cache_t = torch.as_tensor(wrapper_activation_cache).detach().cpu().float()
            post_logits = wrapper_post_logits
            intervention_value = float(interventions[0][3])
            wrapper_pre_gap, wrapper_post_gap, wrapper_gap_delta = summarize_gap(
                wrapper_pre_logits,
                wrapper_post_logits,
                target_a_id,
                target_b_id,
            )
            wrapper_intervention_specs = json.loads(getattr(wrapper_result, "intervention_specs_json", "[]"))
            wrapper_intervention_config = json.loads(getattr(wrapper_result, "intervention_config", "{}"))
            wrapper_path_diagnostics = {
                "resolved_settings": {
                    "scale_factor": float(wrapper_settings["scale_factor"]),
                    "value_source": str(wrapper_settings["value_source"]),
                    "value": wrapper_settings["value"],
                    "constrained_layers": wrapper_settings["constrained_layers"],
                    "freeze_attention": wrapper_settings["freeze_attention"],
                    "apply_activation_function": wrapper_settings["apply_activation_function"],
                    "sparse": bool(wrapper_settings["sparse"]),
                    "return_activations": bool(wrapper_settings["return_activations"]),
                },
                "call_kwargs": _serialize_intervention_call_kwargs(
                    analysis_backend.feature_intervention_call_kwargs(wrapper_settings)
                ),
                "intervention_specs": wrapper_intervention_specs,
                "intervention_config": wrapper_intervention_config,
                "pre_gap": wrapper_pre_gap,
                "post_gap": wrapper_post_gap,
                "gap_delta": wrapper_gap_delta,
                "matches_validation_path_settings": {
                    "constrained_layers": wrapper_settings["constrained_layers"] == list(range(n_layers)),
                    "apply_activation_function": wrapper_settings["apply_activation_function"] is False,
                },
                "returned_activation_cache": True,
                "max_abs_pre_logit_diff_vs_baseline_path": float(
                    (wrapper_pre_logits - baseline_logits).abs().max().item()
                ),
                "key_logits_post": [float(wrapper_post_logits[token_id].item()) for token_id in key_ids],
            }
            runtime_state = {
                "module": snapshot_module_runtime_state(module),
                "graph_op": {
                    "analysis_batch": snapshot_analysis_batch(analysis_batch, ("prompts", "logit_target_ids")),
                    "call_kwargs": _serialize_intervention_call_kwargs(graph_call_kwargs),
                    "requested_graph_target_ids": requested_graph_target_ids,
                    "requested_graph_target_tokens": [str(token) for token in graph_target_tokens],
                    "result": {
                        "input_tokens": tensor_fingerprint(getattr(graph_result, "input_tokens", None)),
                        "active_features": tensor_fingerprint(getattr(graph_result, "active_features", None)),
                        "selected_features": tensor_fingerprint(getattr(graph_result, "selected_features", None)),
                        "selected_feature_rows": tensor_fingerprint(graph_feature_rows),
                        "adjacency_matrix": tensor_fingerprint(getattr(graph_result, "adjacency_matrix", None)),
                        "logit_target_ids": tensor_fingerprint(logit_target_ids),
                        "logit_target_tokens": graph_result_target_tokens,
                    },
                },
                "top_features_op": {
                    "applied_feature_filter_triples": [list(triple) for triple in applied_feature_filter_triples],
                    "selected_feature_candidate_count": int(selected_feature_candidate_count),
                    "result": snapshot_analysis_batch(
                        top_features_result,
                        ("top_feature_ids", "top_feature_scores", "top_feature_activation_values"),
                    ),
                },
                "baseline_forward": {
                    "zero_softcap_enabled": bool(cfg.enable_zero_softcap),
                    "apply_activation_function": False,
                    "graph_inputs": tensor_fingerprint(graph_inputs)
                    if isinstance(graph_inputs, torch.Tensor)
                    else str(graph_inputs),
                    "input_diagnostics": input_diagnostics,
                    "baseline_logits": tensor_fingerprint(baseline_logits),
                    "baseline_activation_cache": tensor_fingerprint(baseline_activation_cache_t),
                    "selected_feature_baseline_activation": float(baseline_activation),
                },
                "intervention_op": {
                    "analysis_batch": snapshot_analysis_batch(
                        wrapper_inputs,
                        (
                            "prompts",
                            "top_feature_ids",
                            "top_feature_scores",
                            "top_feature_activation_values",
                            "logit_target_ids",
                        ),
                    ),
                    "resolved_settings": wrapper_path_diagnostics["resolved_settings"],
                    "call_kwargs": wrapper_path_diagnostics["call_kwargs"],
                    "interventions": [list(spec) for spec in interventions],
                    "result": {
                        "pre_intervention_logits": tensor_fingerprint(wrapper_pre_logits),
                        "post_intervention_logits": tensor_fingerprint(wrapper_post_logits),
                        "intervention_activation_cache": tensor_fingerprint(post_activation_cache_t),
                        "intervention_specs": wrapper_intervention_specs,
                        "intervention_config": wrapper_intervention_config,
                    },
                    "diagnostics": wrapper_path_diagnostics,
                },
            }

        relevant_activations = baseline_activation_cache_t[
            graph_feature_rows[:, 0], graph_feature_rows[:, 1], graph_feature_rows[:, 2]
        ]
        new_relevant_activations = post_activation_cache_t[
            graph_feature_rows[:, 0], graph_feature_rows[:, 1], graph_feature_rows[:, 2]
        ]
        relevant_logits = baseline_logits[logit_target_ids]
        new_relevant_logits = post_logits[logit_target_ids]
        demeaned_relevant_logits = relevant_logits - baseline_logits.mean()
        new_demeaned_relevant_logits = new_relevant_logits - post_logits.mean()

        expected_effects = adjacency_matrix[:, selected_graph_index]
        effect_multiplier = 1.0
        expected_activation_difference = expected_effects[: graph_feature_rows.shape[0]]
        expected_logit_difference = expected_effects[-len(logit_target_ids) :]

        activation_prediction = relevant_activations + expected_activation_difference
        logit_prediction = demeaned_relevant_logits + expected_logit_difference
        activation_abs_error = (new_relevant_activations - activation_prediction).abs()
        logit_abs_error = (new_demeaned_relevant_logits - logit_prediction).abs()
        actual_activation_difference = new_relevant_activations - relevant_activations
        actual_logit_difference = new_demeaned_relevant_logits - demeaned_relevant_logits

        activation_error_rows = _summarize_feature_row_deltas(
            graph_feature_rows,
            relevant_activations,
            new_relevant_activations,
            expected_activation_difference,
            top_k=cfg.debug_validation_top_k,
            ranking="abs_error",
        )
        expected_effect_rows = _summarize_feature_row_deltas(
            graph_feature_rows,
            relevant_activations,
            new_relevant_activations,
            expected_activation_difference,
            top_k=cfg.debug_validation_top_k,
            ranking="expected_delta",
        )
        actual_effect_rows = _summarize_feature_row_deltas(
            graph_feature_rows,
            relevant_activations,
            new_relevant_activations,
            expected_activation_difference,
            top_k=cfg.debug_validation_top_k,
            ranking="actual_delta",
        )
        activation_layer_summary = _summarize_layer_error_rows(
            graph_feature_rows,
            relevant_activations,
            new_relevant_activations,
            expected_activation_difference,
            top_k=cfg.debug_validation_top_k,
        )
        logit_error_rows = _summarize_logit_delta_rows(
            logit_target_ids.tolist(),
            list(graph_target_tokens),
            relevant_logits,
            new_relevant_logits,
            demeaned_relevant_logits,
            new_demeaned_relevant_logits,
            expected_logit_difference,
            top_k=cfg.debug_validation_top_k,
        )
        same_feature_rows_in_graph = _summarize_same_feature_rows(
            graph_feature_rows,
            relevant_activations,
            new_relevant_activations,
            expected_activation_difference,
            layer=layer,
            feature_id=feature_id,
        )
        selected_feature_self_effect = {
            "graph_index": int(selected_graph_index),
            "row": [layer, position, feature_id],
            "baseline_activation": float(relevant_activations[selected_graph_index].item()),
            "predicted_activation": float(activation_prediction[selected_graph_index].item()),
            "post_activation": float(new_relevant_activations[selected_graph_index].item()),
            "expected_delta": float(expected_activation_difference[selected_graph_index].item()),
            "actual_delta": float(actual_activation_difference[selected_graph_index].item()),
            "abs_error": float(activation_abs_error[selected_graph_index].item()),
            "signed_error": float(
                (new_relevant_activations[selected_graph_index] - activation_prediction[selected_graph_index]).item()
            ),
        }
        activation_sign_mismatch_count = int(
            ((expected_activation_difference * actual_activation_difference) < 0.0).sum().item()
        )
        logit_sign_mismatch_count = int(((expected_logit_difference * actual_logit_difference) < 0.0).sum().item())

        drift_report = build_intervention_drift_report(
            graph,
            feature_row=tuple(int(value) for value in selected_feature.tolist()),
            baseline_activation_cache=baseline_activation_cache_t,
            intervention_activation_cache=post_activation_cache_t,
            baseline_logits=baseline_logits,
            intervention_logits=post_logits,
            activation_atol=cfg.debug_validation_act_atol,
            activation_rtol=cfg.debug_validation_act_rtol,
            logit_atol=cfg.debug_validation_logit_atol,
            logit_rtol=cfg.debug_validation_logit_rtol,
        )
        activation_passed = drift_report.divergent_feature_count == 0
        logit_passed = drift_report.logit_summary.divergent_logit_count == 0

        pre_gap, post_gap, gap_delta = summarize_gap(baseline_logits, post_logits, target_a_id, target_b_id)
        selected_feature_score = float(tensor_to_cpu(top_features_result.top_feature_scores)[0].item())
        selected_feature_activation = float(tensor_to_cpu(top_features_result.top_feature_activation_values)[0].item())
        key_logits_pre = [float(baseline_logits[token_id].item()) for token_id in key_ids]
        key_logits_post = [float(post_logits[token_id].item()) for token_id in key_ids]
        artifact_dir = _maybe_preserve_debug_intervention_artifacts(
            cfg,
            graph=graph,
            feature_row=tuple(int(value) for value in selected_feature.tolist()),
            interventions=interventions,
            baseline_activation_cache=baseline_activation_cache_t,
            intervention_activation_cache=post_activation_cache_t,
            baseline_logits=baseline_logits,
            intervention_logits=post_logits,
            graph_target_ids=requested_graph_target_ids,
            graph_target_tokens=graph_target_tokens,
            selected_feature_score=selected_feature_score,
            selected_feature_activation=selected_feature_activation,
            report=drift_report,
            runtime_state=runtime_state,
        )

        return {
            "rendered_prompt": rendered_prompt,
            "selected_feature": tuple(int(value) for value in selected_feature.tolist()),
            "selected_feature_candidate_count": selected_feature_candidate_count,
            "selected_feature_score": selected_feature_score,
            "selected_feature_activation": selected_feature_activation,
            "selected_feature_graph_index": selected_graph_index,
            "graph_target_ids": requested_graph_target_ids,
            "graph_target_tokens": graph_target_tokens,
            "graph_result_target_ids": actual_graph_target_ids,
            "graph_result_target_tokens": graph_result_target_tokens,
            "graph_summary": {
                "active_feature_count": int(active_features.shape[0]),
                "selected_feature_count": int(graph_feature_rows.shape[0]),
                "selected_feature_index_count": int(selected_feature_indices.numel()),
                "adjacency_shape": [int(dim) for dim in adjacency_matrix.shape],
                "requested_graph_target_count": len(requested_graph_target_ids),
                "graph_target_count": int(logit_target_ids.numel()),
                "graph_targets_match_requested": actual_graph_target_ids == requested_graph_target_ids,
            },
            "input_diagnostics": input_diagnostics,
            "requested_constrained_feature_selection": _serialize_requested_constrained_feature_selection(cfg),
            "applied_feature_filter_triples": applied_feature_filter_triples,
            "target_a_id": target_a_id,
            "target_b_id": target_b_id,
            "target_a_tok": resolved_target_tokens[0],
            "target_b_tok": resolved_target_tokens[1],
            "pre_gap": pre_gap,
            "post_gap": post_gap,
            "gap_delta": gap_delta,
            "baseline_activation": baseline_activation,
            "intervention_value": intervention_value,
            "effect_multiplier": effect_multiplier,
            "selected_feature_self_effect": selected_feature_self_effect,
            "same_feature_rows_in_graph": same_feature_rows_in_graph,
            "activation_passed": activation_passed,
            "logit_passed": logit_passed,
            "all_passed": activation_passed and logit_passed,
            "activation_max_abs_error": float(activation_abs_error.max().item()),
            "logit_max_abs_error": float(logit_abs_error.max().item()),
            "activation_mean_abs_error": float(activation_abs_error.mean().item()),
            "logit_mean_abs_error": float(logit_abs_error.mean().item()),
            "activation_sign_mismatch_count": activation_sign_mismatch_count,
            "logit_sign_mismatch_count": logit_sign_mismatch_count,
            "activation_error_rows": activation_error_rows,
            "activation_layer_summary": activation_layer_summary,
            "expected_effect_rows": expected_effect_rows,
            "actual_effect_rows": actual_effect_rows,
            "logit_error_rows": logit_error_rows,
            "drift_report": drift_report.to_dict(top_feature_count=cfg.debug_validation_top_k),
            "artifact_dir": None if artifact_dir is None else str(artifact_dir),
            "graph_artifact": graph_artifact,
            "runtime_state": runtime_state,
            "intervention_paths": {
                "adjacency_validation": {
                    "intervention": {
                        "layer": layer,
                        "position": position,
                        "feature_id": feature_id,
                        "value": intervention_value,
                    },
                    "call_kwargs": {
                        "constrained_layers": {"kind": "range", "start": 0, "stop": n_layers, "step": 1},
                        "apply_activation_function": False,
                    },
                    "pre_gap": pre_gap,
                    "post_gap": post_gap,
                    "gap_delta": gap_delta,
                },
                "feature_intervention_forward": wrapper_path_diagnostics,
            },
            "key_ids": key_ids,
            "key_labels": key_labels,
            "key_logits_pre": key_logits_pre,
            "key_logits_post": key_logits_post,
            "validation_tolerances": {
                "act_atol": cfg.debug_validation_act_atol,
                "act_rtol": cfg.debug_validation_act_rtol,
                "logit_atol": cfg.debug_validation_logit_atol,
                "logit_rtol": cfg.debug_validation_logit_rtol,
            },
            "debug_validation_top_k": int(cfg.debug_validation_top_k),
            "debug_validation_raise_on_failure": bool(cfg.debug_validation_raise_on_failure),
        }


def run_direction_probes(
    cfg: NotebookHarnessConfig,
    embed_direction: torch.Tensor,
    store_direction: torch.Tensor,
) -> dict[str, Any]:
    if not cfg.supports_store_direction:
        raise ValueError("run_direction_probes is only available in concept_pair mode")

    from interpretune.analysis.backends.circuit_tracer import CircuitTracerAnalysisBackend

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg.experiment_name, "direction_probes"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        backend = CircuitTracerAnalysisBackend()
        embedding_weight = backend.get_embedding_weight(module).float().detach()
        unembed = embedding_weight.T if embedding_weight.shape[0] > embedding_weight.shape[1] else embedding_weight
        probe_results: dict[str, Any] = {}
        for label, direction in [("Embed", embed_direction), ("Store", store_direction)]:
            direction_dev = direction.to(unembed.device)
            rows = []
            group_a_projections = []
            group_b_projections = []
            for token in cfg.concept_pair.group_a_tokens:
                token_id = tokenizer.encode(token, add_special_tokens=False)[-1]
                projection = float(torch.dot(unembed[:, token_id].float(), direction_dev).item())
                group_a_projections.append(projection)
                rows.append({"token": token, "projection": projection, "group": "A"})
            for token in cfg.concept_pair.group_b_tokens:
                token_id = tokenizer.encode(token, add_special_tokens=False)[-1]
                projection = float(torch.dot(unembed[:, token_id].float(), direction_dev).item())
                group_b_projections.append(projection)
                rows.append({"token": token, "projection": projection, "group": "B"})
            mean_b = None if not group_b_projections else sum(group_b_projections) / len(group_b_projections)
            probe_results[label] = {
                "rows": rows,
                "mean_a": sum(group_a_projections) / len(group_a_projections),
                "mean_b": mean_b,
            }
        return probe_results
