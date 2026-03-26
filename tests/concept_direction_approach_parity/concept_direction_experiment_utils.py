from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch

import interpretune as it
import interpretune.analysis  # noqa: F401
from interpretune.config import AnalysisCfg, init_analysis_cfgs

from it_examples.example_prompt_configs import GemmaPromptConfig
from tests.concept_direction_approach_parity.concept_direction_approach_experimentation import (
    CONCEPT_PAIRS,
    ConceptPair,
    _build_classification_prompt,
    _chattify,
)
from tests.concept_direction_approach_parity.experiment_resource_utils import (
    experiment_session,
    feature_ids_to_tuples,
    tensor_to_cpu,
)


PromptRenderMode = Literal["plain", "apply_chat_template", "gemma_dataclass"]


@dataclass
class NotebookHarnessConfig:
    experiment_name: str
    experiment_config_name: str
    model_family: str
    model_variant: str
    model_name: str
    transcoder_set: str
    neuronpedia_model: str
    neuronpedia_set: str
    concept_pair_name: str
    prompt: str
    prompt_render_mode: PromptRenderMode
    target_tokens: tuple[str, str] | None
    target_token_ids: tuple[int, int] | None
    top_n: int
    default_scale_factor: float
    scale_factor_sweep: list[float]
    ablation_n_list: list[int]
    enable_sign_aware: bool
    force_device: str | None
    work_root: Any
    concept_pair: ConceptPair = field(init=False)

    def __post_init__(self) -> None:
        self.concept_pair = CONCEPT_PAIRS[self.concept_pair_name]
        if self.target_tokens is not None:
            if len(self.target_tokens) != 2:
                raise ValueError("target_tokens must contain exactly two tokens")
            self.target_tokens = (str(self.target_tokens[0]), str(self.target_tokens[1]))
        if self.target_token_ids is not None:
            if len(self.target_token_ids) != 2:
                raise ValueError("target_token_ids must contain exactly two token ids")
            self.target_token_ids = (int(self.target_token_ids[0]), int(self.target_token_ids[1]))
        if self.target_tokens is None and self.target_token_ids is None:
            raise ValueError("either target_tokens or target_token_ids must be provided")
        self.scale_factor_sweep = [float(value) for value in self.scale_factor_sweep]
        self.ablation_n_list = [int(value) for value in self.ablation_n_list]

    @property
    def use_chat_template(self) -> bool:
        return self.prompt_render_mode != "plain"

    @property
    def chat_template_method(self) -> str:
        if self.prompt_render_mode == "gemma_dataclass":
            return "gemma_dataclass"
        if self.prompt_render_mode == "apply_chat_template":
            return "apply_chat_template"
        return "plain"


def phase_run_name(cfg: NotebookHarnessConfig, label: str) -> str:
    cleaned = label.lower().replace(" ", "_").replace("/", "_")
    return f"{cfg.experiment_name}_{cleaned}"


def render_prompt(prompt: str, tokenizer: Any, mode: PromptRenderMode) -> str:
    if mode == "plain":
        return prompt
    chat_method = "gemma_dataclass" if mode == "gemma_dataclass" else "apply_chat_template"
    return _chattify(prompt, tokenizer, chat_method)


def render_prompt_variants(prompt: str, tokenizer: Any) -> dict[str, str]:
    gemma_cfg = GemmaPromptConfig()
    return {
        "plain": prompt,
        "apply_chat_template": gemma_cfg.apply_chat_template_fn(
            tokenizer,
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        ),
        "gemma_dataclass": gemma_cfg.model_chat_template_fn(prompt, tokenization_pattern="gemma-chat"),
    }


def _tokenize_rendered_prompt(tokenizer: Any, rendered_prompt: str, mode: PromptRenderMode | str) -> list[int]:
    add_special_tokens = mode == "plain"
    return cast(list[int], tokenizer(rendered_prompt, add_special_tokens=add_special_tokens)["input_ids"])


def resolve_target_tokens(cfg: NotebookHarnessConfig, tokenizer: Any) -> tuple[tuple[int, int], tuple[str, str]]:
    if cfg.target_tokens is not None:
        resolved_ids = tuple(tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.target_tokens)
        return cast(tuple[int, int], resolved_ids), cfg.target_tokens
    assert cfg.target_token_ids is not None
    decoded_tokens = tuple(tokenizer.decode([token_id]) for token_id in cfg.target_token_ids)
    return cfg.target_token_ids, cast(tuple[str, str], decoded_tokens)


def get_key_token_ids_and_labels(cfg: NotebookHarnessConfig, tokenizer: Any) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    labels: list[str] = []
    for token in cfg.concept_pair.key_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded:
            ids.append(encoded[-1])
            labels.append(token)
    return ids, labels


def summarize_gap(pre_logits: torch.Tensor, post_logits: torch.Tensor, target_a_id: int, target_b_id: int) -> tuple[float, float, float]:
    pre_gap = float((pre_logits[target_a_id] - pre_logits[target_b_id]).item())
    post_gap = float((post_logits[target_a_id] - post_logits[target_b_id]).item())
    return pre_gap, post_gap, post_gap - pre_gap


def configure_analysis(module: Any, graph_op: Any, scale_factor: float) -> None:
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = scale_factor
    module.analysis_cfg = AnalysisCfg(target_op=graph_op, ignore_manual=True, save_tokens=False)
    if not module.analysis_cfg.applied_to(module):
        init_analysis_cfgs(module, [module.analysis_cfg])


def run_reasoning_sanity_check(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    """Probe multi-hop reasoning with the Dallas→Austin prompt and return logit analysis."""
    concept_pair = cfg.concept_pair
    if cfg.use_chat_template and concept_pair.chat_intervention_prompt:
        raw_prompt = concept_pair.chat_intervention_prompt
    else:
        raw_prompt = concept_pair.intervention_prompt

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "reasoning_sanity_check"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, tokenizer):
        rendered = render_prompt(raw_prompt, tokenizer, cfg.prompt_render_mode)
        add_special = cfg.prompt_render_mode == "plain"
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=add_special)
        enc = {k: v.to(module.device) for k, v in enc.items()}

        with torch.inference_mode():
            gen_out = module.model.generate(
                **enc, max_new_tokens=5, do_sample=False, output_logits=True, return_dict_in_generate=True,
            )

        gen_text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
        first_logits = gen_out.logits[0][0]
        probs = torch.softmax(first_logits.float(), dim=-1)

        key_labels = ["Austin", "Dallas", "Texas"]
        key_analysis: list[dict[str, Any]] = []
        for label in key_labels:
            ids = tokenizer.encode(label, add_special_tokens=False)
            tid = ids[0] if ids else None
            entry: dict[str, Any] = {"label": label, "token_id": tid}
            if tid is not None:
                entry["logit"] = float(first_logits[tid].item())
                entry["prob"] = float(probs[tid].item())
            key_analysis.append(entry)

        top_id = int(first_logits.argmax(dim=-1).item())
        top_token = tokenizer.decode([top_id])
        top_prob = float(probs[top_id].item())

        return {
            "prompt_style": "chat" if cfg.use_chat_template else "plain",
            "rendered_prompt": rendered[:400],
            "generated_text": gen_text,
            "key_tokens": key_analysis,
            "top1_token": top_token,
            "top1_id": top_id,
            "top1_prob": top_prob,
        }


def run_tokenizer_verification(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "tokenizer_verification"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, tokenizer):
        render_variants = render_prompt_variants(cfg.prompt, tokenizer)
        selected_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        add_special_tokens = cfg.prompt_render_mode == "plain"
        enc = tokenizer(selected_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        resolved_target_ids, resolved_target_tokens = resolve_target_tokens(cfg, tokenizer)
        render_variant_token_ids = {
            mode_name: _tokenize_rendered_prompt(tokenizer, rendered_prompt, mode_name)
            for mode_name, rendered_prompt in render_variants.items()
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
            "render_variant_equalities": {
                "apply_chat_template_vs_dataclass": render_variants["apply_chat_template"] == render_variants["gemma_dataclass"],
                "apply_chat_template_vs_dataclass_token_ids": (
                    render_variant_token_ids["apply_chat_template"] == render_variant_token_ids["gemma_dataclass"]
                ),
            },
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
        for token in cfg.concept_pair.key_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            report["key_tokens"][token] = {"ids": ids, "decoded": tokenizer.decode(ids)}
        return report


def compute_embed_direction(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "embed_direction"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, _):
        embed_result = cast(
            Any,
            it.concept_direction(
                module,
                it.AnalysisBatch(
                    concept_group_a=cfg.concept_pair.group_a_tokens,
                    concept_group_b=cfg.concept_pair.group_b_tokens,
                    concept_label=cfg.concept_pair.concept_label,
                    concept_direction_mode="paired_rejection",
                ),
                None,
                0,
            ),
        )
        return {
            "direction": tensor_to_cpu(embed_result.concept_direction),
            "group_a_ids": list(embed_result.concept_group_a_token_ids),
            "group_b_ids": list(embed_result.concept_group_b_token_ids),
        }


def run_pipeline(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    label: str,
    *,
    scale_factor: float,
    top_n: int,
    group_a_ids: list[int],
    group_b_ids: list[int],
) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, f"{label}_pipeline_{scale_factor}x"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, tokenizer):
        (target_a_id, target_b_id), resolved_target_tokens = resolve_target_tokens(cfg, tokenizer)
        key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        configure_analysis(module, it.compute_attribution_graph, scale_factor)
        graph_result = cast(
            Any,
            it.compute_attribution_graph(
                module,
                it.AnalysisBatch(
                    prompts=[rendered_prompt],
                    concept_direction=direction,
                    concept_label=cfg.concept_pair.concept_label,
                    concept_direction_mode="paired_rejection",
                    concept_group_a_token_ids=group_a_ids,
                    concept_group_b_token_ids=group_b_ids,
                ),
                None,
                0,
            ),
        )
        influence_result = cast(Any, it.graph_node_influence(module, graph_result, None, 0))
        top_payload = dict(cast(Any, graph_result))
        top_payload.update(dict(cast(Any, influence_result)))
        top_features_result = cast(
            Any,
            it.extract_top_features(module, it.AnalysisBatch(**top_payload), None, 0, top_n=top_n),
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
                None,
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
        }


def build_all_prompts(cfg: NotebookHarnessConfig, tokenizer: Any) -> list[tuple[str, str, str]]:
    prompts: list[tuple[str, str, str]] = []
    for entity_name, expected_answer in cfg.concept_pair.group_a_entities:
        raw = _build_classification_prompt(entity_name, cfg.concept_pair.classification_question)
        prompts.append((render_prompt(raw, tokenizer, cfg.prompt_render_mode), expected_answer, cfg.concept_pair.group_a_name))
    for entity_name, expected_answer in cfg.concept_pair.group_b_entities:
        raw = _build_classification_prompt(entity_name, cfg.concept_pair.classification_question)
        prompts.append((render_prompt(raw, tokenizer, cfg.prompt_render_mode), expected_answer, cfg.concept_pair.group_b_name))
    return prompts


def compute_store_direction(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "store_direction"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, tokenizer):
        model_backend = getattr(module, "_model_backend", None)
        device = next(module.model.parameters()).device
        all_prompts = build_all_prompts(cfg, tokenizer)
        latent_states: list[torch.Tensor] = []
        prediction_info = {"examples": [], "n_correct": 0}
        for prompt_text, expected_answer, group in all_prompts:
            add_special_tokens = cfg.prompt_render_mode == "plain"
            enc = tokenizer(prompt_text, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens)
            batch_dev = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in dict(enc).items()}
            with torch.no_grad():
                logits, cache = model_backend.fwd_w_cache(model=module.model, batch=batch_dev, names_filter="unembed.hook_in")
            last_pos = logits.shape[1] - 1
            example_logits = logits[0, last_pos]
            cache_tensor = torch.as_tensor(cache["unembed.hook_in"])
            latent_states.append(tensor_to_cpu(cache_tensor[0, last_pos]))
            topk = torch.topk(example_logits, 10).indices.tolist()
            topk_tokens = [tokenizer.decode([token_id]) for token_id in topk]
            expected_id = tokenizer.encode(expected_answer, add_special_tokens=False)[-1]
            correct = expected_id in topk
            rank = topk.index(expected_id) if correct else -1
            if correct:
                prediction_info["n_correct"] += 1
            prediction_info["examples"].append(
                {
                    "group": group,
                    "expected": expected_answer,
                    "correct": correct,
                    "rank": rank,
                    "top1": topk_tokens[0] if topk_tokens else None,
                    "top5": topk_tokens[:5],
                    "prompt": prompt_text,
                    "input_ids": enc["input_ids"][0].tolist(),
                    "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist()),
                    "attention_mask": enc["attention_mask"][0].tolist() if "attention_mask" in enc else None,
                }
            )
        stacked = torch.stack(latent_states)
        n_a = len(cfg.concept_pair.group_a_entities)
        n_b = len(cfg.concept_pair.group_b_entities)
        group_ids = torch.cat([torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)])
        group_names = ([cfg.concept_pair.group_a_name] * n_a) + ([cfg.concept_pair.group_b_name] * n_b)
        store_result = cast(
            Any,
            it.concept_direction(
                module,
                it.AnalysisBatch(
                    concept_latent_state=[stacked],
                    concept_group_id=[group_ids],
                    concept_group_name=[group_names],
                    concept_example_weight=[torch.ones(len(stacked), dtype=torch.float32)],
                    concept_label=cfg.concept_pair.concept_label,
                    concept_direction_mode="paired_rejection",
                    concept_group_a_name=cfg.concept_pair.group_a_name,
                    concept_group_b_name=cfg.concept_pair.group_b_name,
                ),
                None,
                0,
            ),
        )
        return {
            "direction": tensor_to_cpu(store_result.concept_direction),
            "group_a_ids": [tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_a_tokens],
            "group_b_ids": [tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_b_tokens],
            "prediction_info": prediction_info,
            "n_total": len(all_prompts),
        }


def run_scale_sweep(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    group_a_ids: list[int],
    group_b_ids: list[int],
) -> list[dict[str, Any]]:
    results = []
    for scale_factor in cfg.scale_factor_sweep:
        with experiment_session(
            cfg.work_root,
            phase_run_name(cfg, f"scale_sweep_{scale_factor}x"),
            model_family=cfg.model_family,
            model_name=cfg.model_name,
            transcoder_set=cfg.transcoder_set,
            force_device=cfg.force_device,
        ) as (_, module, tokenizer):
            (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)
            key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)
            rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
            configure_analysis(module, it.compute_attribution_graph, scale_factor)
            graph_result = cast(
                Any,
                it.compute_attribution_graph(
                    module,
                    it.AnalysisBatch(
                        prompts=[rendered_prompt],
                        concept_direction=direction,
                        concept_label=cfg.concept_pair.concept_label,
                        concept_direction_mode="paired_rejection",
                        concept_group_a_token_ids=group_a_ids,
                        concept_group_b_token_ids=group_b_ids,
                    ),
                    None,
                    0,
                ),
            )
            influence_result = cast(Any, it.graph_node_influence(module, graph_result, None, 0))
            top_payload = dict(cast(Any, graph_result))
            top_payload.update(dict(cast(Any, influence_result)))
            top_features_result = cast(
                Any,
                it.extract_top_features(module, it.AnalysisBatch(**top_payload), None, 0, top_n=cfg.top_n),
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
                    None,
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
                }
            )
    return results


def collect_feature_pool(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    group_a_ids: list[int],
    group_b_ids: list[int],
    *,
    top_n: int,
) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "feature_pool"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, tokenizer):
        (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)
        key_ids, key_labels = get_key_token_ids_and_labels(cfg, tokenizer)
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
        configure_analysis(module, it.compute_attribution_graph, 0.0)
        graph_result = cast(
            Any,
            it.compute_attribution_graph(
                module,
                it.AnalysisBatch(
                    prompts=[rendered_prompt],
                    concept_direction=direction,
                    concept_label=cfg.concept_pair.concept_label,
                    concept_direction_mode="paired_rejection",
                    concept_group_a_token_ids=group_a_ids,
                    concept_group_b_token_ids=group_b_ids,
                ),
                None,
                0,
            ),
        )
        influence_result = cast(Any, it.graph_node_influence(module, graph_result, None, 0))
        top_payload = dict(cast(Any, graph_result))
        top_payload.update(dict(cast(Any, influence_result)))
        feature_result = cast(
            Any,
            it.extract_top_features(module, it.AnalysisBatch(**top_payload), None, 0, top_n=top_n),
        )
        return {
            "feature_ids": tensor_to_cpu(feature_result.top_feature_ids.to(torch.float32)).to(torch.long),
            "feature_scores": tensor_to_cpu(feature_result.top_feature_scores),
            "feature_activations": tensor_to_cpu(feature_result.top_feature_activation_values),
            "target_a_id": target_a_id,
            "target_b_id": target_b_id,
            "key_ids": key_ids,
            "key_labels": key_labels,
        }


def run_ablations(
    cfg: NotebookHarnessConfig,
    feature_pool: dict[str, Any],
    pre_logits_ref: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[int, torch.Tensor]]:
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
        phase_run_name(cfg, "progressive_ablation"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
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
                    None,
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
            abl_logit_diffs[label] = float(post_logits[feature_pool["target_a_id"]] - post_logits[feature_pool["target_b_id"]])
            results[n_value] = post_logits
    return abl_groups, abl_logit_diffs, results


def run_sign_aware(cfg: NotebookHarnessConfig, feature_pool: dict[str, Any], pre_logits_ref: torch.Tensor) -> dict[str, Any]:
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
        phase_run_name(cfg, "sign_aware"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
    ) as (_, module, tokenizer):
        rendered_prompt = render_prompt(cfg.prompt, tokenizer, cfg.prompt_render_mode)
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
                        top_feature_activation_values=result["positive_activations"][:n_pos] * cfg.default_scale_factor,
                        logit_target_ids=torch.tensor([feature_pool["target_a_id"]], dtype=torch.long),
                    ),
                    None,
                    0,
                ),
            )
            result["positive_post_logits"] = tensor_to_cpu(pos_intervention.post_intervention_logits)
        else:
            result["messages"].append("No positive-activation features were available for the current feature pool.")
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
                    None,
                    0,
                ),
            )
            result["negative_post_logits"] = tensor_to_cpu(neg_intervention.post_intervention_logits)
        else:
            result["messages"].append("No negative-activation features were available for the current feature pool.")
    result["pre_logits_ref"] = pre_logits_ref
    return result


def run_direction_probes(cfg: NotebookHarnessConfig, embed_direction: torch.Tensor, store_direction: torch.Tensor) -> dict[str, Any]:
    from interpretune.analysis.backends.circuit_tracer import CircuitTracerAnalysisBackend

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "direction_probes"),
        model_family=cfg.model_family,
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_set,
        force_device=cfg.force_device,
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
            probe_results[label] = {
                "rows": rows,
                "mean_a": sum(group_a_projections) / len(group_a_projections),
                "mean_b": sum(group_b_projections) / len(group_b_projections),
            }
        return probe_results