from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

import torch

import interpretune as it
import interpretune.analysis
from interpretune.analysis.execution import execute_analysis_op
from interpretune.analysis.ops.helpers import AnalysisInputs
from interpretune.config import AnalysisCfg, init_analysis_cfgs

from it_examples.example_prompt_configs import GemmaPromptConfig
from tests.concept_direction_approach_parity.experiment_resource_utils import (
    experiment_session,
    feature_ids_to_tuples,
    tensor_to_cpu,
)


PromptRenderMode = Literal["plain", "apply_chat_template", "gemma_dataclass"]
StoreLatentExtractionMode = Literal["answer_position_state", "context_enhanced"]


CLASSIFICATION_QUESTION_V3 = 'Is this a Capital or a State? Answer with one word: " Capital" or " State".'
KEY_TOKENS_TO_INSPECT = ["▁Austin", "▁Dallas", "▁Texas", "▁Capital", "▁State", "▁capital", "▁state", "▁City", "▁city"]


@dataclass
class ConceptPair:
    """A pair of concept groups for direction computation and intervention."""

    name: str
    description: str
    group_a_tokens: list[str]
    group_b_tokens: list[str]
    group_a_entities: list[tuple[str, str]]
    group_b_entities: list[tuple[str, str]]
    group_a_name: str
    group_b_name: str
    concept_label: str
    classification_question: str
    intervention_prompt: str
    key_tokens: list[str]
    chat_intervention_prompt: str | None = None


CAPITALS_STATES = ConceptPair(
    name="capitals_states",
    description="US state capitals vs states",
    group_a_tokens=["▁Austin", "▁Sacramento", "▁Olympia", "▁Atlanta"],
    group_b_tokens=["▁Texas", "▁California", "▁Washington", "▁Georgia"],
    group_a_entities=[
        ("Austin", "Capital"),
        ("Sacramento", "Capital"),
        ("Olympia", "Capital"),
        ("Atlanta", "Capital"),
    ],
    group_b_entities=[
        ("Texas", "State"),
        ("California", "State"),
        ("Washington", "State"),
        ("Georgia", "State"),
    ],
    group_a_name="capitals",
    group_b_name="states",
    concept_label="Concept: Capitals − States",
    classification_question=CLASSIFICATION_QUESTION_V3,
    intervention_prompt="Fact: the capital of the state containing Dallas is",
    key_tokens=KEY_TOKENS_TO_INSPECT,
    chat_intervention_prompt=(
        "Answer with only the missing city name. Fact: the capital of the state containing Dallas is"
    ),
)

DOG_CAT = ConceptPair(
    name="dog_cat",
    description="Dog breeds vs cat breeds (simpler concept pair for sanity check)",
    group_a_tokens=["▁Labrador", "▁Poodle", "▁Beagle", "▁Bulldog"],
    group_b_tokens=["▁Siamese", "▁Persian", "▁Tabby", "▁Sphynx"],
    group_a_entities=[
        ("Labrador", "Dog"),
        ("Poodle", "Dog"),
        ("Beagle", "Dog"),
        ("Bulldog", "Dog"),
    ],
    group_b_entities=[
        ("Siamese", "Cat"),
        ("Persian", "Cat"),
        ("Tabby", "Cat"),
        ("Sphynx", "Cat"),
    ],
    group_a_name="dogs",
    group_b_name="cats",
    concept_label="Concept: Dogs − Cats",
    classification_question='Is this a Dog or a Cat breed? Answer with one word: " Dog" or " Cat".',
    intervention_prompt="My favorite kind of common four-legged domestic pet is the",
    key_tokens=["▁Dog", "▁Cat", "▁dog", "▁cat", "▁Labrador", "▁Siamese", "▁puppy", "▁kitten"],
    chat_intervention_prompt=(
        'Answer with only the missing animal type (e.g. " Dog", " Cat").'
        " My favorite kind of common four-legged domestic pet is the"
    ),
)

CAT_DOG = ConceptPair(
    name="cat_dog",
    description="Cat breeds vs dog breeds (reversed direction: Cat − Dog)",
    group_a_tokens=["▁Siamese", "▁Persian", "▁Tabby", "▁Sphynx"],
    group_b_tokens=["▁Labrador", "▁Poodle", "▁Beagle", "▁Bulldog"],
    group_a_entities=[
        ("Siamese", "Cat"),
        ("Persian", "Cat"),
        ("Tabby", "Cat"),
        ("Sphynx", "Cat"),
    ],
    group_b_entities=[
        ("Labrador", "Dog"),
        ("Poodle", "Dog"),
        ("Beagle", "Dog"),
        ("Bulldog", "Dog"),
    ],
    group_a_name="cats",
    group_b_name="dogs",
    concept_label="Concept: Cats − Dogs",
    classification_question='Is this a Cat or a Dog breed? Answer with one word: " Cat" or " Dog".',
    intervention_prompt="My favorite kind of common four-legged domestic pet is the",
    key_tokens=["▁Cat", "▁Dog", "▁cat", "▁dog", "▁Siamese", "▁Labrador", "▁kitten", "▁puppy"],
    chat_intervention_prompt=(
        'Answer with only the missing animal type (e.g. " Cat", " Dog").'
        " My favorite kind of common four-legged domestic pet is the"
    ),
)

CONCEPT_PAIRS: dict[str, ConceptPair] = {
    "capitals_states": CAPITALS_STATES,
    "dog_cat": DOG_CAT,
    "cat_dog": CAT_DOG,
}


def _build_classification_prompt(entity_name: str, question: str) -> str:
    """Build a classification-style prompt for a single entity."""
    return f"{question} {entity_name} : "


def _chattify_apply_chat_template(prompt: str, tokenizer: Any) -> str:
    """Wrap a prompt in the model's chat template using tokenizer.apply_chat_template."""
    cfg = GemmaPromptConfig()
    return cfg.apply_chat_template_fn(tokenizer, prompt, tokenize=False, add_generation_prompt=True)


def _chattify_gemma_dataclass(prompt: str) -> str:
    """Wrap a prompt using the GemmaPromptConfig dataclass approach."""
    cfg = GemmaPromptConfig()
    return cfg.model_chat_template_fn(prompt, tokenization_pattern="gemma-chat")


def _chattify(prompt: str, tokenizer: Any, method: str = "apply_chat_template") -> str:
    """Apply chat template using the configured method."""
    if method == "gemma_dataclass":
        return _chattify_gemma_dataclass(prompt)
    return _chattify_apply_chat_template(prompt, tokenizer)


@dataclass
class NotebookHarnessConfig:
    experiment_name: str
    experiment_config_name: str
    model_family: str
    model_variant: str
    model_name: str
    transcoder_set: str
    hf_model_head: str | None
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
    batch_size: int | None = None
    max_feature_nodes: int | None = None
    key_tokens_override: tuple[str, ...] | None = None
    store_latent_extraction_mode: StoreLatentExtractionMode = "answer_position_state"
    context_enhanced_scale: float = 1.0
    store_concept_cache_key: str = "unembed.hook_in"
    store_concept_correct_only: bool = False
    store_weight_by_logit_diff: bool = True
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

    @property
    def session_kwargs(self) -> dict[str, Any]:
        """Common kwargs passed to every ``experiment_session`` call."""
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "transcoder_set": self.transcoder_set,
            "hf_model_head": self.hf_model_head,
            "force_device": self.force_device,
            "batch_size": self.batch_size,
            "max_feature_nodes": self.max_feature_nodes,
        }


def phase_run_name(cfg: NotebookHarnessConfig, label: str) -> str:
    cleaned = label.lower().replace(" ", "_").replace("/", "_")
    return f"{cfg.experiment_name}_{cleaned}"


def render_prompt(prompt: str, tokenizer: Any, mode: PromptRenderMode) -> str:
    if mode == "plain":
        return prompt
    chat_method = "gemma_dataclass" if mode == "gemma_dataclass" else "apply_chat_template"
    return _chattify(prompt, tokenizer, chat_method)


def render_prompt_variants(prompt: str, tokenizer: Any) -> dict[str, str | None]:
    gemma_cfg = GemmaPromptConfig()
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None
    return {
        "plain": prompt,
        "apply_chat_template": gemma_cfg.apply_chat_template_fn(
            tokenizer,
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_chat_template
        else None,
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


def get_key_token_ids_and_labels(
    cfg: NotebookHarnessConfig,
    tokenizer: Any,
    *,
    include_bare_variants: bool = True,
) -> tuple[list[int], list[str]]:
    """Resolve key token IDs and labels, optionally including bare (non-▁-prefixed) variants.

    SentencePiece tokenizers encode ``" Austin"`` → ``▁Austin`` (token 24278), but in chat
    mode the model may predict bare ``Austin`` (token 107305). Setting
    *include_bare_variants* (default ``True``) ensures both forms are tracked.
    """
    ids: list[int] = []
    labels: list[str] = []
    seen_ids: set[int] = set()
    source_tokens = cfg.key_tokens_override if cfg.key_tokens_override is not None else cfg.concept_pair.key_tokens
    for token in source_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded:
            tid = encoded[-1]
            if tid not in seen_ids:
                ids.append(tid)
                labels.append(token)
                seen_ids.add(tid)
        if include_bare_variants and token.startswith("▁"):
            bare = token.lstrip("▁")
            bare_encoded = tokenizer.encode(bare, add_special_tokens=False)
            if bare_encoded:
                bare_tid = bare_encoded[-1]
                if bare_tid not in seen_ids:
                    ids.append(bare_tid)
                    labels.append(bare)
                    seen_ids.add(bare_tid)
    return ids, labels


def summarize_gap(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    target_a_id: int,
    target_b_id: int,
) -> tuple[float, float, float]:
    pre_gap = float((pre_logits[target_a_id] - pre_logits[target_b_id]).item())
    post_gap = float((post_logits[target_a_id] - post_logits[target_b_id]).item())
    return pre_gap, post_gap, post_gap - pre_gap


def configure_analysis(module: Any, graph_op: Any, scale_factor: float) -> None:
    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = scale_factor
    module.analysis_cfg = AnalysisCfg(target_op=graph_op, ignore_manual=True, save_tokens=False)
    if not module.analysis_cfg.applied_to(module):
        init_analysis_cfgs(module, [module.analysis_cfg])


def run_initial_sanity_check(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    """Run initial sanity check for the configured intervention prompt and return logit analysis.

    Uses ``cfg.prompt`` (which incorporates PROMPT_OVERRIDE from YAML configs)
    rather than the concept pair's hardcoded prompts.
    """
    concept_pair = cfg.concept_pair
    raw_prompt = cfg.prompt

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "initial_sanity_check"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        rendered = render_prompt(raw_prompt, tokenizer, cfg.prompt_render_mode)
        add_special = cfg.prompt_render_mode == "plain"
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=add_special)
        enc = {k: v.to(module.device) for k, v in enc.items()}

        with torch.inference_mode():
            gen_out = module.model.generate(
                **enc,
                max_new_tokens=5,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )

        gen_text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
        first_logits = gen_out.logits[0][0]
        probs = torch.softmax(first_logits.float(), dim=-1)

        # Build key_labels dynamically from the concept pair's key_tokens.
        # Space-prefix ensures SentencePiece encodes to a single token (e.g. ▁Austin)
        # rather than splitting bare "Austin" into sub-tokens.
        # Also include bare variants (without ▁ prefix) since chat-mode models may
        # predict different token IDs for bare vs prefixed forms.
        seen_ids: set[int] = set()
        key_analysis: list[dict[str, Any]] = []
        source_tokens = cfg.key_tokens_override if cfg.key_tokens_override is not None else concept_pair.key_tokens
        for tok in source_tokens:
            # ▁-prefixed token (space prefix encoding)
            encode_form = f" {tok.lstrip('▁')}"
            label = tok.lstrip("▁")
            ids = tokenizer.encode(encode_form, add_special_tokens=False)
            tid = ids[0] if ids else None
            if tid is not None and tid not in seen_ids:
                entry: dict[str, Any] = {"label": f"▁{label}", "token_id": tid}
                entry["logit"] = float(first_logits[tid].item())
                entry["prob"] = float(probs[tid].item())
                key_analysis.append(entry)
                seen_ids.add(tid)
            # Bare token variant (for chat-mode predictions)
            if tok.startswith("▁"):
                bare = tok.lstrip("▁")
                bare_ids = tokenizer.encode(bare, add_special_tokens=False)
                bare_tid = bare_ids[-1] if bare_ids else None
                if bare_tid is not None and bare_tid not in seen_ids:
                    bare_entry: dict[str, Any] = {"label": bare, "token_id": bare_tid}
                    bare_entry["logit"] = float(first_logits[bare_tid].item())
                    bare_entry["prob"] = float(probs[bare_tid].item())
                    key_analysis.append(bare_entry)
                    seen_ids.add(bare_tid)

        # Sort key tokens by logit magnitude descending
        key_analysis.sort(key=lambda e: abs(e.get("logit", 0.0)), reverse=True)

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
        phase_run_name(cfg, "tokenizer_verification"),
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
        for token in cfg.concept_pair.key_tokens:
            ids = tokenizer.encode(token, add_special_tokens=False)
            report["key_tokens"][token] = {"ids": ids, "decoded": tokenizer.decode(ids)}
        return report


def compute_embed_direction(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "embed_direction"),
        **cfg.session_kwargs,
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
        **cfg.session_kwargs,
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


def run_direct_projection_pipeline(
    cfg: NotebookHarnessConfig,
    direction: torch.Tensor,
    label: str,
    *,
    scale_factor: float,
) -> dict[str, Any]:
    """Run a direct residual-stream projection intervention (bypasses feature selection).

    Instead of going through the full attribution graph → feature selection → feature
    intervention pipeline, this uses ``it.model_fwd_intervention`` to add
    the scaled concept direction vector *directly* to the residual stream via the model
    backend's ``fwd_w_intervention`` method.

    Returns a result dict compatible with ``run_pipeline``'s output format.
    """
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, f"{label}_direct_proj_{scale_factor}x"),
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
        intervention_result = cast(
            Any,
            it.model_fwd_intervention(
                module,
                it.AnalysisBatch(
                    prompts=[rendered_prompt],
                    concept_direction=direction,
                    concept_cache_key=cfg.store_concept_cache_key,
                    direction_scale_factor=scale_factor,
                    logit_target_ids=torch.tensor([target_a_id], dtype=torch.long),
                    concept_group_a_token_ids=[target_a_id],
                    concept_group_b_token_ids=[target_b_id],
                ),
                batch,
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
    direction: torch.Tensor,
) -> list[dict[str, Any]]:
    """Run a scale sweep using direct residual-stream projection (no feature selection)."""
    results = []
    for scale_factor in cfg.scale_factor_sweep:
        result = run_direct_projection_pipeline(
            cfg,
            direction,
            "direct_proj_sweep",
            scale_factor=scale_factor,
        )
        results.append(result)
    return results


def build_all_prompts(cfg: NotebookHarnessConfig, tokenizer: Any) -> list[tuple[str, str, str]]:
    prompts: list[tuple[str, str, str]] = []
    for entity_name, expected_answer in cfg.concept_pair.group_a_entities:
        raw = _build_classification_prompt(entity_name, cfg.concept_pair.classification_question)
        prompts.append(
            (render_prompt(raw, tokenizer, cfg.prompt_render_mode), expected_answer, cfg.concept_pair.group_a_name)
        )
    for entity_name, expected_answer in cfg.concept_pair.group_b_entities:
        raw = _build_classification_prompt(entity_name, cfg.concept_pair.classification_question)
        prompts.append(
            (render_prompt(raw, tokenizer, cfg.prompt_render_mode), expected_answer, cfg.concept_pair.group_b_name)
        )
    return prompts


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


def compute_store_direction_manual(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "store_direction_manual"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        model_backend = getattr(module, "_model_backend", None)
        assert model_backend is not None, "experiment session module must expose _model_backend"
        device = next(module.model.parameters()).device
        all_prompts = build_all_prompts(cfg, tokenizer)
        (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)
        latent_states: list[torch.Tensor] = []
        prediction_info = {"examples": [], "n_correct": 0}
        for prompt_text, expected_answer, group in all_prompts:
            add_special_tokens = cfg.prompt_render_mode == "plain"
            enc = tokenizer(prompt_text, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens)
            batch_dev = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in dict(enc).items()
            }
            with torch.no_grad():
                logits, cache = model_backend.fwd_w_cache(
                    model=module.model,
                    batch=batch_dev,
                    names_filter="unembed.hook_in",
                )
            last_pos = logits.shape[1] - 1
            example_logits = logits[0, last_pos]
            cache_tensor = torch.as_tensor(cache["unembed.hook_in"])
            latent_states.append(tensor_to_cpu(cache_tensor[0, last_pos]))
            expected_id, _, correct, rank, topk_ids, topk_tokens = _score_expected_answer(
                tokenizer,
                example_logits,
                expected_answer,
                target_a_id,
                target_b_id,
            )
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
                    "top10_ids": topk_ids,
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
            "group_a_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_a_tokens
            ],
            "group_b_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_b_tokens
            ],
            "prediction_info": prediction_info,
            "n_total": len(all_prompts),
        }


def construct_concept_pair_analysis_inputs(
    cfg: NotebookHarnessConfig,
    module: Any,
    tokenizer: Any,
    model_backend: Any,
    device: Any,
    target_a_id: int,
    target_b_id: int,
) -> tuple[
    list[dict[str, torch.Tensor]],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    dict[str, Any],
    list[tuple[str, str, str]],
]:
    """Build per-example forward-pass caches, answer indices, labels, and logit diffs.

    For each entity prompt in the concept pair, runs a forward pass with caching, scores
    the expected answer, and collects the cached activations and metadata needed by
    ``execute_concept_latent_extraction_ops``.

    Returns:
        (cached_batches, answer_indices, orig_labels, logit_diffs, prediction_info, all_prompts)
    """
    all_prompts = build_all_prompts(cfg, tokenizer)

    cached_batches: list[dict[str, torch.Tensor]] = []
    answer_indices: list[torch.Tensor] = []
    orig_labels: list[torch.Tensor] = []
    logit_diffs: list[torch.Tensor] = []
    prediction_info: dict[str, Any] = {"examples": [], "n_correct": 0}

    for prompt_text, expected_answer, group in all_prompts:
        add_special_tokens = cfg.prompt_render_mode == "plain"
        enc = tokenizer(prompt_text, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens)
        batch_dev = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in dict(enc).items()
        }
        with torch.no_grad():
            logits, cache = model_backend.fwd_w_cache(
                model=module.model,
                batch=batch_dev,
                names_filter=cfg.store_concept_cache_key,
            )

        last_pos = int(logits.shape[1] - 1)
        example_logits = logits[0, last_pos]
        expected_id, example_logit_diff, correct, rank, topk_ids, topk_tokens = _score_expected_answer(
            tokenizer,
            example_logits,
            expected_answer,
            target_a_id,
            target_b_id,
        )
        if correct:
            prediction_info["n_correct"] += 1

        group_id = 0 if group == cfg.concept_pair.group_a_name else 1
        cached_batches.append(
            {cfg.store_concept_cache_key: torch.as_tensor(cache[cfg.store_concept_cache_key]).detach().cpu()}
        )
        answer_indices.append(torch.tensor([last_pos], dtype=torch.long))
        orig_labels.append(torch.tensor([group_id], dtype=torch.long))
        logit_diffs.append(torch.tensor([example_logit_diff], dtype=torch.float32))
        prediction_info["examples"].append(
            {
                "group": group,
                "expected": expected_answer,
                "expected_token_id": expected_id,
                "correct": correct,
                "rank": rank,
                "top1": topk_tokens[0] if topk_tokens else None,
                "top5": topk_tokens[:5],
                "top10_ids": topk_ids,
                "logit_diff": example_logit_diff,
                "prompt": prompt_text,
                "input_ids": enc["input_ids"][0].tolist(),
                "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist()),
                "attention_mask": enc["attention_mask"][0].tolist() if "attention_mask" in enc else None,
            }
        )

    return cached_batches, answer_indices, orig_labels, logit_diffs, prediction_info, all_prompts


def _apply_context_enhanced_projection(
    extracted_batches: list[Any],
    cached_batches: list[dict[str, torch.Tensor]],
    answer_indices: list[torch.Tensor],
    cache_key: str,
    context_scale: float = 1.0,
) -> list[Any]:
    """Project answer-position latent states into previous-token context.

    For each extracted batch, retrieves the context vector from the token immediately
    preceding the answer position, scales the answer-position latent state by
    *context_scale*, then adds the scaled answer state to the context vector.  The
    resulting "context-enhanced" latent replaces the original ``concept_latent_state`` in
    the batch.

    This grounding step attempts to disentangle the answer-token representation from
    senses of the token that are not specific to the prompt context (e.g. the multiple
    senses of "Capital" that are not related to capital-city semantics).
    """
    for batch_idx, batch in enumerate(extracted_batches):
        if batch.concept_latent_state is None:
            continue

        cache_tensor = torch.as_tensor(cached_batches[batch_idx][cache_key])
        ans_idx = int(answer_indices[batch_idx].reshape(-1)[0])

        if ans_idx < 1:
            continue

        prev_idx = ans_idx - 1
        if cache_tensor.dim() >= 3:
            context_state = cache_tensor[0, prev_idx].detach().cpu().float()
        else:
            continue

        answer_state = batch.concept_latent_state
        if answer_state.dim() == 2:
            answer_state = answer_state.squeeze(0)

        projected = context_state + context_scale * answer_state
        batch.concept_latent_state = projected.unsqueeze(0)

    return extracted_batches


def execute_concept_latent_extraction_ops(
    module: Any,
    cfg: NotebookHarnessConfig,
    cached_batches: list[dict[str, torch.Tensor]],
    answer_indices: list[torch.Tensor],
    orig_labels: list[torch.Tensor],
    logit_diffs: list[torch.Tensor],
    n_prompts: int,
    *,
    extraction_mode: StoreLatentExtractionMode = "answer_position_state",
) -> list[Any]:
    """Execute ``extract_concept_latent_state`` and ``extract_concept_latent_examples`` ops.

    Wraps the AnalysisCfg setup, AnalysisInputs construction, and per-batch op execution
    loop.  The *extraction_mode* parameter selects which variant of latent extraction to
    use:

    - ``"answer_position_state"``: Default — extracts the hidden state at the answer
      token position.
    - ``"context_enhanced"``: Extracts the answer-position state *and* the immediately
      preceding token's context, then projects the scaled answer state into that context.
      The scale factor is read from ``cfg.context_enhanced_scale`` (default 1.0).
    """
    if extraction_mode not in ("answer_position_state", "context_enhanced"):
        raise ValueError(f"Unsupported extraction_mode: {extraction_mode}")

    extraction_cfg = AnalysisCfg(
        name="concept_latent_rows",
        target_op=[it.extract_concept_latent_state, it.extract_concept_latent_examples],
        ignore_manual=True,
    )
    analysis_inputs = AnalysisInputs(
        store=SimpleNamespace(
            cache=cached_batches,
            answer_indices=answer_indices,
            orig_labels=orig_labels,
            logit_diffs=logit_diffs,
        )
    )

    extracted_batches: list[Any] = []
    for batch_idx in range(n_prompts):
        extracted_batches.append(
            execute_analysis_op(
                module,
                batch=None,
                batch_idx=batch_idx,
                analysis_batch=it.AnalysisBatch(
                    concept_group_a_label_ids=[0],
                    concept_group_b_label_ids=[1],
                    concept_group_a_name=cfg.concept_pair.group_a_name,
                    concept_group_b_name=cfg.concept_pair.group_b_name,
                    concept_cache_key=cfg.store_concept_cache_key,
                    concept_correct_only=cfg.store_concept_correct_only,
                    concept_weight_by_logit_diff=cfg.store_weight_by_logit_diff,
                ),
                analysis_cfg=extraction_cfg,
                analysis_inputs=analysis_inputs,
            )
        )

    if extraction_mode == "context_enhanced":
        extracted_batches = _apply_context_enhanced_projection(
            extracted_batches,
            cached_batches,
            answer_indices,
            cfg.store_concept_cache_key,
            context_scale=cfg.context_enhanced_scale,
        )

    return extracted_batches


def prepare_extracted_concept_example_tensors(
    extracted_batches: list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Prepare tensors from extracted AnalysisBatch rows for ``it.concept_direction``.

    This intermediate step is needed because ``it.extract_concept_latent_examples`` emits
    one AnalysisBatch *per example* (each containing a single-row latent state, group id,
    weight, and group name).  The ``it.concept_direction`` op, however, expects a *single*
    stacked tensor of all concept latent states along with parallel group-id, weight, and
    group-name sequences — the format produced when the two ops are chained via the
    AnalysisStore in a standard analysis session (where the store accumulates rows
    automatically across batches).

    In experiment-harness code we run the ops manually in a per-example loop, so the
    store-level accumulation does not happen.  This function bridges that gap by
    concatenating the per-example rows into the stacked format ``it.concept_direction``
    requires.

    .. note::

       *IG-7 follow-up:* Investigate whether ``extract_concept_latent_examples`` and
       ``concept_direction`` can be adapted (or an additional intermediate op added) so
       their input/output contracts allow direct pipelining via a composite op alias,
       eliminating the need for this manual stacking step.

    Returns:
        (stacked_latent_states, group_ids, example_weights, group_name_rows)
    """
    latent_rows = [batch.concept_latent_state for batch in extracted_batches if batch.concept_latent_state is not None]
    group_id_rows = [batch.concept_group_id for batch in extracted_batches if batch.concept_group_id is not None]
    weight_rows = [
        batch.concept_example_weight for batch in extracted_batches if batch.concept_example_weight is not None
    ]
    group_name_rows: list[str] = []
    for batch in extracted_batches:
        if batch.concept_group_name is not None:
            group_name_rows.extend(list(batch.concept_group_name))

    stacked = torch.cat([tensor_to_cpu(row) for row in latent_rows], dim=0)
    group_ids = torch.cat([tensor_to_cpu(row) for row in group_id_rows], dim=0)
    example_weights = torch.cat([tensor_to_cpu(row) for row in weight_rows], dim=0)
    return stacked, group_ids, example_weights, group_name_rows


def compute_store_direction(cfg: NotebookHarnessConfig) -> dict[str, Any]:
    if cfg.store_latent_extraction_mode not in ("answer_position_state", "context_enhanced"):
        raise ValueError(f"Unsupported store_latent_extraction_mode: {cfg.store_latent_extraction_mode}")

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "store_direction"),
        **cfg.session_kwargs,
    ) as (_, module, tokenizer):
        model_backend = getattr(module, "_model_backend", None)
        assert model_backend is not None, "experiment session module must expose _model_backend"
        device = next(module.model.parameters()).device
        (target_a_id, target_b_id), _ = resolve_target_tokens(cfg, tokenizer)

        cached_batches, answer_indices, orig_labels, logit_diffs, prediction_info, all_prompts = (
            construct_concept_pair_analysis_inputs(
                cfg,
                module,
                tokenizer,
                model_backend,
                device,
                target_a_id,
                target_b_id,
            )
        )

        extracted_batches = execute_concept_latent_extraction_ops(
            module,
            cfg,
            cached_batches,
            answer_indices,
            orig_labels,
            logit_diffs,
            len(all_prompts),
            extraction_mode=cfg.store_latent_extraction_mode,
        )

        stacked, group_ids, example_weights, group_name_rows = prepare_extracted_concept_example_tensors(
            extracted_batches,
        )

        store_result = cast(
            Any,
            it.concept_direction(
                module,
                it.AnalysisBatch(
                    concept_latent_state=[stacked],
                    concept_group_id=[group_ids],
                    concept_group_name=[group_name_rows],
                    concept_example_weight=[example_weights],
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
            "group_a_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_a_tokens
            ],
            "group_b_ids": [
                tokenizer.encode(token, add_special_tokens=False)[-1] for token in cfg.concept_pair.group_b_tokens
            ],
            "prediction_info": prediction_info,
            "n_total": len(all_prompts),
            "n_latent_rows": int(stacked.shape[0]),
            "manual_reference_fn": "compute_store_direction_manual",
            "store_latent_extraction_mode": cfg.store_latent_extraction_mode,
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
            **cfg.session_kwargs,
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
        **cfg.session_kwargs,
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
        **cfg.session_kwargs,
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
        **cfg.session_kwargs,
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


def run_direction_probes(
    cfg: NotebookHarnessConfig,
    embed_direction: torch.Tensor,
    store_direction: torch.Tensor,
) -> dict[str, Any]:
    from interpretune.analysis.backends.circuit_tracer import CircuitTracerAnalysisBackend

    with experiment_session(
        cfg.work_root,
        phase_run_name(cfg, "direction_probes"),
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
            probe_results[label] = {
                "rows": rows,
                "mean_a": sum(group_a_projections) / len(group_a_projections),
                "mean_b": sum(group_b_projections) / len(group_b_projections),
            }
        return probe_results
