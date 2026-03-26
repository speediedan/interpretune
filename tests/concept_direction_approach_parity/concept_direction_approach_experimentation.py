#!/usr/bin/env python3
"""Concept-direction approach experimentation: embed-based vs store-based construction.

This is a manual debugging and analysis tool, intentionally not part of the normal pytest or
coverage flows. Use the regular parity tests for regression detection.

**V3 (2025-03-24)**: Prompts reformulated to explicitly include expected answer tokens
("Capital" / "State") so the model sees them in context. Expected answers now use capitalized
forms matching the model's natural preference.  Key-token logit analysis (Austin, Dallas, Texas)
added. Configurable concept pairs (not just capitals/states). Direction reversal testing.
GemmaPromptConfig-based chat template option. Datetime-stamped log file output.

Approach comparison:
  - **Embed-based**: Uses raw embedding/unembedding weight vectors for concept tokens.
    Fast, no model forward pass required. Matches the existing test_analysis_backend_parity.py.
  - **Store-based**: Runs model forward on classification prompts, caches ``unembed.hook_in``
    activations (pre-logit d_model space), extracts per-example latent states at the answer
    position, then computes concept_direction via ``paired_rejection``.

Both directions are compared via:
  1. Cosine similarity of the direction vectors
  2. Attribution graph overlap (Jaccard of top-N features)
  3. Intervention effect comparison (pre/post logit gaps)
  4. Key-token logit analysis (detailed per-token logit inspection)

**Architecture note**: The circuit-tracer NNsight module uses its own ``TranscoderSet``
(not SAELens SAE objects), so the ``fwd_w_cache_and_latent_models`` path (which requires
``sae_handles``) is not available. We use ``fwd_w_cache`` with ``unembed.hook_in`` to
capture pre-logit residual-stream activations from a plain forward pass. Transcoder splicing
happens only in the attribution graph computation.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

# ---------------------------------------------------------------------------
# Interpretune imports
# ---------------------------------------------------------------------------
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.dispatcher import DISPATCHER
from interpretune.config import AnalysisCfg, init_analysis_cfgs
from it_examples.example_prompt_configs import GemmaPromptConfig

# ---------------------------------------------------------------------------
# Test-infrastructure imports (fully qualified from tests package)
# ---------------------------------------------------------------------------
from tests import load_dotenv
from tests.analysis_resource_utils import clear_nnsight_test_state, serial_test_cleanup
from tests.configuration import config_modules
from tests.conftest import FixtPhase, session_fixture_hook_exec
from tests.core.cfg_aliases import CircuitTracerNNsightGemma2
from tests.core.test_analysis_backend_parity import (
    SemanticInterventionParityCase,
)

# ---------------------------------------------------------------------------
# Model variant registry
# ---------------------------------------------------------------------------
MODEL_VARIANTS: dict[str, str] = {
    "it": "google/gemma-2-2b-it",
    "base": "google/gemma-2-2b",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = "Fact: the capital of the state containing Dallas is"
DEFAULT_TOP_N = 10
CACHE_KEY = "unembed.hook_in"

# V3 classification prompt: explicitly names both expected answers to prime the model.
# The model should see "Capital" and "State" as viable answer tokens.
CLASSIFICATION_QUESTION_V3 = 'Is this a Capital or a State? Answer with one word: "Capital" or "State".'

# Key tokens to inspect in logit analysis (beyond the target pair)
KEY_TOKENS_TO_INSPECT = ["▁Austin", "▁Dallas", "▁Texas", "▁Capital", "▁State", "▁capital", "▁state", "▁City", "▁city"]


# ---------------------------------------------------------------------------
# Concept pair definitions
# ---------------------------------------------------------------------------
@dataclass
class ConceptPair:
    """A pair of concept groups for direction computation and intervention."""

    name: str
    description: str
    # For embed-based direction: SentencePiece tokens (with leading ▁)
    group_a_tokens: list[str]
    group_b_tokens: list[str]
    # For store-based direction: entity display names and expected answers
    group_a_entities: list[tuple[str, str]]  # (display_name, expected_answer)
    group_b_entities: list[tuple[str, str]]
    # Labels
    group_a_name: str
    group_b_name: str
    concept_label: str
    # Classification question template
    classification_question: str
    # Intervention prompt
    intervention_prompt: str
    # Key tokens to inspect (logit analysis)
    key_tokens: list[str]
    chat_intervention_prompt: str | None = None


# Pre-defined concept pairs
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
    key_tokens=["▁Austin", "▁Dallas", "▁Texas", "▁Capital", "▁State",
                "▁capital", "▁state", "▁City", "▁city"],
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
    classification_question='Is this a Dog or a Cat breed? Answer with one word: "Dog" or "Cat".',
    intervention_prompt="My favorite kind of common four-legged domestic pet is the",
    key_tokens=["▁Dog", "▁Cat", "▁dog", "▁cat", "▁Labrador", "▁Siamese",
                "▁puppy", "▁kitten"],
)

CONCEPT_PAIRS: dict[str, ConceptPair] = {
    "capitals_states": CAPITALS_STATES,
    "dog_cat": DOG_CAT,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DirectionResult:
    """Result from a single concept_direction computation."""

    approach: str
    direction_vector: torch.Tensor
    direction_mode: str
    group_a_token_ids: list[int] | None = None
    group_b_token_ids: list[int] | None = None
    n_examples: int | None = None
    correct_count: int | None = None


@dataclass
class KeyTokenLogits:
    """Detailed logit values for key tokens at a given pipeline stage."""

    token_name: str
    token_id: int
    logit_value: float
    rank: int  # rank among all vocab tokens


@dataclass
class InterventionResult:
    """Full pipeline result: direction → graph → features → intervention."""

    approach: str
    direction_mode: str
    top_feature_ids: list[tuple[int, ...]]
    pre_gap: float
    post_gap: float
    pre_logits: torch.Tensor
    post_logits: torch.Tensor
    # V3: per-token logit inspection
    pre_key_token_logits: list[KeyTokenLogits] = field(default_factory=list)
    post_key_token_logits: list[KeyTokenLogits] = field(default_factory=list)


@dataclass
class ParityComparison:
    """Comparison of two concept_direction approaches."""

    cosine_similarity: float
    feature_jaccard: float
    embed_pre_gap: float
    embed_post_gap: float
    store_pre_gap: float
    store_post_gap: float
    n_shared_features: int
    n_total_features: int


@dataclass
class HarnessConfig:
    """All tuneable harness parameters, populated from CLI args."""

    model_variant: str = "it"
    concept_pair_name: str = "capitals_states"
    prompt: str | None = None  # None = use concept pair default
    top_n: int = DEFAULT_TOP_N
    skip_intervention: bool = False
    use_chat_template: bool = True
    chat_template_method: str = "apply_chat_template"  # "apply_chat_template" or "gemma_dataclass"
    test_direction_reversal: bool = False
    output: str | None = None
    log_dir: str | None = None  # None = concept_direction_approach_parity folder
    # Automatically set in __post_init__
    model_name: str = field(init=False)
    concept_pair: ConceptPair = field(init=False)

    def __post_init__(self):
        self.model_name = MODEL_VARIANTS.get(self.model_variant, MODEL_VARIANTS["it"])
        self.concept_pair = CONCEPT_PAIRS[self.concept_pair_name]
        # Default: chat template on for IT models, off for base
        if self.model_variant == "base":
            self.use_chat_template = False
        if self.prompt is None:
            self.prompt = resolve_intervention_prompt(self.concept_pair, use_chat_template=self.use_chat_template)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._file = open(log_path, "w")  # noqa: SIM115
        self._stdout = sys.stdout

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout


def _make_log_path(harness_cfg: HarnessConfig) -> Path:
    """Generate a datetime-stamped log file path."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    if harness_cfg.log_dir:
        log_dir = Path(harness_cfg.log_dir)
    else:
        log_dir = Path(__file__).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"experiment_{harness_cfg.model_variant}_{harness_cfg.concept_pair_name}_{timestamp}.log"
    return log_dir / filename


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _build_classification_prompt(entity_name: str, question: str) -> str:
    """Build a classification-style prompt for a single entity."""
    return f"{entity_name} : {question}"


def resolve_intervention_prompt(concept_pair: ConceptPair, *, use_chat_template: bool) -> str:
    """Resolve the default intervention prompt for plain vs chat-rendered runs."""
    if use_chat_template and concept_pair.chat_intervention_prompt is not None:
        return concept_pair.chat_intervention_prompt
    return concept_pair.intervention_prompt


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


def _build_all_prompts(
    tokenizer: Any,
    harness_cfg: HarnessConfig,
) -> list[tuple[str, str, str]]:
    """Build full prompt list: (final_prompt, expected_answer, group).

    Returns list of (prompt_text, expected_token, group_label) tuples.
    """
    cp = harness_cfg.concept_pair
    prompts: list[tuple[str, str, str]] = []
    for entity_name, expected_answer in cp.group_a_entities:
        raw = _build_classification_prompt(entity_name, cp.classification_question)
        if harness_cfg.use_chat_template:
            final = _chattify(raw, tokenizer, harness_cfg.chat_template_method)
        else:
            final = raw
        prompts.append((final, expected_answer, cp.group_a_name))
    for entity_name, expected_answer in cp.group_b_entities:
        raw = _build_classification_prompt(entity_name, cp.classification_question)
        if harness_cfg.use_chat_template:
            final = _chattify(raw, tokenizer, harness_cfg.chat_template_method)
        else:
            final = raw
        prompts.append((final, expected_answer, cp.group_b_name))
    return prompts


# ---------------------------------------------------------------------------
# Model and session helpers
# ---------------------------------------------------------------------------


def _build_test_cfg(harness_cfg: HarnessConfig):
    """Build a CircuitTracerNNsightGemma2 config, optionally for the IT model."""
    cfg = CircuitTracerNNsightGemma2(
        phase="test",
        device_type="cuda" if torch.cuda.is_available() else "cpu",
    )
    assert cfg.circuit_tracer_cfg is not None
    assert cfg.nnsight_cfg is not None
    cfg.circuit_tracer_cfg.model_name = harness_cfg.model_name
    cfg.nnsight_cfg.model_name = harness_cfg.model_name
    return cfg


@contextmanager
def _session_factory(tmp_path: Path, run_name: str, harness_cfg: HarnessConfig):
    """Create and yield a fully-initialized ITSession, then clean up."""
    session_dir = tmp_path / run_name
    session_dir.mkdir(parents=True, exist_ok=True)
    clear_nnsight_test_state(None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    load_dotenv()
    it_session = config_modules(_build_test_cfg(harness_cfg), run_name, {}, session_dir, {}, False)
    session_fixture_hook_exec(it_session, cast(FixtPhase, FixtPhase.setup))
    module = it_session.module
    assert module is not None
    replacement_model = cast(Any, module).replacement_model
    with serial_test_cleanup(it_session, module, replacement_model):
        yield it_session


# ---------------------------------------------------------------------------
# Tokenizer verification
# ---------------------------------------------------------------------------


def verify_tokenizer(module: Any, concept_pair: ConceptPair) -> dict[str, Any]:
    """Verify that the tokenizer handles concept target tokens correctly."""
    tokenizer = module.replacement_model.tokenizer
    results: dict[str, Any] = {"group_a": {}, "group_b": {}, "key_tokens": {}}

    for token in concept_pair.group_a_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        results["group_a"][token] = {"token_ids": ids, "decoded": decoded, "last_id": ids[-1]}

    for token in concept_pair.group_b_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        results["group_b"][token] = {"token_ids": ids, "decoded": decoded, "last_id": ids[-1]}

    # All expected answer tokens + key inspection tokens
    answer_tokens = set()
    for _, expected in concept_pair.group_a_entities + concept_pair.group_b_entities:
        answer_tokens.add(expected)
        answer_tokens.add(expected.lower())
    for kt in concept_pair.key_tokens:
        # Strip leading ▁ for non-SP encoding check
        answer_tokens.add(kt.replace("▁", ""))

    for target in sorted(answer_tokens):
        ids = tokenizer.encode(target, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        results["key_tokens"][target] = {"token_ids": ids, "decoded": decoded}

    return results


# ---------------------------------------------------------------------------
# Key-token logit analysis
# ---------------------------------------------------------------------------


def _extract_key_token_logits(
    logits: torch.Tensor,
    tokenizer: Any,
    key_tokens: list[str],
) -> list[KeyTokenLogits]:
    """Extract logit values and ranks for key tokens from a logit vector."""
    # Sort all logits to get ranks
    sorted_vals, sorted_indices = torch.sort(logits, descending=True)
    rank_map = {int(idx): rank for rank, idx in enumerate(sorted_indices.tolist())}

    results = []
    for token_str in key_tokens:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if not ids:
            continue
        tid = ids[-1]
        logit_val = float(logits[tid].item())
        rank = rank_map.get(tid, -1)
        results.append(KeyTokenLogits(
            token_name=token_str,
            token_id=tid,
            logit_value=logit_val,
            rank=rank,
        ))
    return results


def _print_key_token_table(
    label: str,
    pre_ktl: list[KeyTokenLogits],
    post_ktl: list[KeyTokenLogits],
):
    """Print a table of key token logit values, pre- and post-intervention."""
    print(f"\n  {label} Key-Token Logit Analysis:")
    print(f"  {'Token':<15} {'Pre Logit':>10} {'Pre Rank':>9} {'Post Logit':>11} {'Post Rank':>10} {'Δ Logit':>9}")
    print(f"  {'-'*15} {'-'*10} {'-'*9} {'-'*11} {'-'*10} {'-'*9}")
    post_map = {ktl.token_name: ktl for ktl in post_ktl}
    for pre in pre_ktl:
        post = post_map.get(pre.token_name)
        if post:
            delta = post.logit_value - pre.logit_value
            print(
                f"  {pre.token_name:<15} {pre.logit_value:>10.4f} {pre.rank:>9d}"
                f" {post.logit_value:>11.4f} {post.rank:>10d} {delta:>+9.4f}"
            )


# ---------------------------------------------------------------------------
# Embed-based concept_direction (existing approach, static token embeddings)
# ---------------------------------------------------------------------------


def build_embed_direction(module: Any, case: SemanticInterventionParityCase) -> DirectionResult:
    """Build concept_direction using the embed-based approach (from token embeddings)."""
    concept_op = DISPATCHER.get_op("concept_direction")
    result = cast(
        Any,
        concept_op(
            module,
            AnalysisBatch(
                concept_group_a=case.capitals,
                concept_group_b=case.states,
                concept_label=case.label,
                concept_direction_mode="paired_rejection",
            ),
            None,
            0,
        ),
    )
    return DirectionResult(
        approach="embed",
        direction_vector=result.concept_direction.float().cpu(),
        direction_mode=result.concept_direction_mode,
        group_a_token_ids=list(result.concept_group_a_token_ids),
        group_b_token_ids=list(result.concept_group_b_token_ids),
    )


# ---------------------------------------------------------------------------
# Store-based concept_direction (classification prompts → latent states)
# ---------------------------------------------------------------------------


def _run_individual_forward_passes(
    module: Any,
    harness_cfg: HarnessConfig,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Run each classification prompt individually and collect latent states.

    Returns:
        Tuple of (prediction_info, latent_states_list) where latent_states_list
        contains one [d_model] tensor per prompt (from ``unembed.hook_in`` at the
        last token position).
    """
    model_backend = getattr(module, "_model_backend", None)
    if model_backend is None:
        raise ValueError("Module needs a model backend for fwd_w_cache")

    tokenizer = module.replacement_model.tokenizer
    device = next(module.model.parameters()).device

    all_prompts = _build_all_prompts(tokenizer, harness_cfg)

    results: dict[str, Any] = {"examples": [], "all_correct": True, "n_correct": 0}
    latent_states: list[torch.Tensor] = []

    for idx, (prompt_text, expected_answer, group) in enumerate(all_prompts):
        add_special = not harness_cfg.use_chat_template
        enc = tokenizer(prompt_text, return_tensors="pt", padding=False, add_special_tokens=add_special)
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict(enc).items()}

        with torch.no_grad():
            logits, cache = model_backend.fwd_w_cache(
                model=module.model,
                batch=batch_dev,
                names_filter=CACHE_KEY,
            )

        last_pos = logits.shape[1] - 1
        example_logits = logits[0, last_pos]

        # Extract latent state at last position
        cache_tensor = torch.as_tensor(cache[CACHE_KEY])
        latent = cache_tensor[0, last_pos].detach().cpu().float()
        latent_states.append(latent)

        # Prediction quality
        top_k = 10
        topk_vals, topk_ids_t = torch.topk(example_logits, top_k)
        topk_ids = topk_ids_t.tolist()
        topk_tokens = [tokenizer.decode([tid]) for tid in topk_ids]

        expected_id = tokenizer.encode(expected_answer, add_special_tokens=False)[-1]
        correct = expected_id in topk_ids
        rank = topk_ids.index(expected_id) if correct else -1

        if correct:
            results["n_correct"] += 1
        else:
            results["all_correct"] = False

        results["examples"].append({
            "idx": idx,
            "group": group,
            "prompt": prompt_text[:80] + ("..." if len(prompt_text) > 80 else ""),
            "expected": expected_answer,
            "expected_id": expected_id,
            "correct": correct,
            "rank": rank,
            "top5_tokens": topk_tokens[:5],
            "top5_ids": topk_ids[:5],
        })

    return results, latent_states


def _sanity_check_latent_states(
    latent_states: list[torch.Tensor],
    n_group_a: int,
) -> dict[str, Any]:
    """Run basic sanity checks on the extracted pre-unembed latent states."""
    stacked = torch.stack(latent_states)
    norms = torch.linalg.vector_norm(stacked, dim=-1)

    # Pairwise cosine similarity
    normed = stacked / norms.unsqueeze(-1)
    cosine_matrix = normed @ normed.T

    # Mean within-group and between-group cosines
    cap_cap = cosine_matrix[:n_group_a, :n_group_a]
    state_state = cosine_matrix[n_group_a:, n_group_a:]
    cap_state = cosine_matrix[:n_group_a, n_group_a:]

    info = {
        "mean_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "all_finite": bool(torch.isfinite(stacked).all()),
        "mean_cosine_within_group_a": float(
            cap_cap.triu(diagonal=1).sum() / max(cap_cap.triu(diagonal=1).nonzero().shape[0], 1)
        ),
        "mean_cosine_within_group_b": float(
            state_state.triu(diagonal=1).sum() / max(state_state.triu(diagonal=1).nonzero().shape[0], 1)
        ),
        "mean_cosine_between_groups": float(cap_state.mean()),
    }
    return info


def build_store_direction(
    module: Any,
    case: SemanticInterventionParityCase,
    harness_cfg: HarnessConfig,
) -> tuple[DirectionResult, dict[str, Any], dict[str, Any]]:
    """Build concept_direction using the store-based approach.

    Pipeline:
    1. Build classification prompts with explicit expected answer tokens
    2. Run model forward individually per prompt (avoids padding artifacts)
    3. Extract pre-unembed latent states from ``unembed.hook_in``
    4. Sanity-check the latent states
    5. Compute concept_direction via ``paired_rejection``
    """
    cp = harness_cfg.concept_pair
    prediction_info, latent_state_list = _run_individual_forward_passes(module, harness_cfg)
    sanity_info = _sanity_check_latent_states(latent_state_list, len(cp.group_a_entities))

    latent_states = torch.stack(latent_state_list)
    n_a = len(cp.group_a_entities)
    n_b = len(cp.group_b_entities)
    n_total = n_a + n_b

    group_ids = torch.cat([
        torch.zeros(n_a, dtype=torch.long),
        torch.ones(n_b, dtype=torch.long),
    ])
    group_names = ([cp.group_a_name] * n_a) + ([cp.group_b_name] * n_b)

    concept_op = DISPATCHER.get_op("concept_direction")
    ab = AnalysisBatch(
        concept_latent_state=[latent_states],
        concept_group_id=[group_ids],
        concept_group_name=[group_names],
        concept_example_weight=[torch.ones(n_total, dtype=torch.float32)],
        concept_label=case.label,
        concept_direction_mode="paired_rejection",
        concept_group_a_name=cp.group_a_name,
        concept_group_b_name=cp.group_b_name,
    )
    result = cast(Any, concept_op(module, ab, None, 0))

    # Resolve token IDs from the tokenizer
    tokenizer = module.replacement_model.tokenizer
    group_a_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in case.capitals]
    group_b_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in case.states]

    return (
        DirectionResult(
            approach="store",
            direction_vector=result.concept_direction.float().cpu(),
            direction_mode=result.concept_direction_mode,
            group_a_token_ids=group_a_ids,
            group_b_token_ids=group_b_ids,
            n_examples=n_total,
            correct_count=prediction_info["n_correct"],
        ),
        prediction_info,
        sanity_info,
    )


# ---------------------------------------------------------------------------
# Full pipeline: direction → graph → influence → features → intervention
# ---------------------------------------------------------------------------


def _get_target_token_ids(module: Any, concept_pair: ConceptPair) -> tuple[int, int]:
    """Get the target token IDs for intervention gap measurement.

    For capitals_states: Austin vs Dallas.
    For other pairs: first group_a token vs an appropriate contrast token.
    """
    tokenizer = module.replacement_model.tokenizer
    if concept_pair.name == "capitals_states":
        austin_id = tokenizer.encode("▁Austin", add_special_tokens=False)[-1]
        dallas_id = tokenizer.encode("▁Dallas", add_special_tokens=False)[-1]
        return austin_id, dallas_id
    elif concept_pair.name == "dog_cat":
        dog_id = tokenizer.encode("▁dog", add_special_tokens=False)[-1]
        cat_id = tokenizer.encode("▁cat", add_special_tokens=False)[-1]
        return dog_id, cat_id
    else:
        # Fallback: first token from each group
        a_id = tokenizer.encode(concept_pair.group_a_tokens[0], add_special_tokens=False)[-1]
        b_id = tokenizer.encode(concept_pair.group_b_tokens[0], add_special_tokens=False)[-1]
        return a_id, b_id


def run_full_pipeline(
    module: Any,
    case: SemanticInterventionParityCase,
    direction: torch.Tensor,
    approach: str,
    harness_cfg: HarnessConfig,
    group_a_token_ids: list[int] | None = None,
    group_b_token_ids: list[int] | None = None,
) -> InterventionResult:
    """Run the full CT analysis pipeline using a precomputed concept_direction."""
    cp = harness_cfg.concept_pair
    target_a_id, target_b_id = _get_target_token_ids(module, cp)
    tokenizer = module.replacement_model.tokenizer

    graph_op = DISPATCHER.get_op("compute_attribution_graph")
    influence_op = DISPATCHER.get_op("graph_node_influence")
    top_features_op = DISPATCHER.get_op("extract_top_features")
    intervention_op = DISPATCHER.get_op("feature_intervention_forward")

    module.circuit_tracer_cfg.intervention_value_source = "top_feature_activation_values"
    module.circuit_tracer_cfg.intervention_scale_factor = 10.0
    module.analysis_cfg = AnalysisCfg(target_op=graph_op, ignore_manual=True, save_tokens=False)
    if not module.analysis_cfg.applied_to(module):
        init_analysis_cfgs(module, [module.analysis_cfg])

    graph_result = cast(
        Any,
        graph_op(
            module,
            AnalysisBatch(
                prompts=[case.prompt],
                concept_direction=direction,
                concept_label=case.label,
                concept_direction_mode="paired_rejection",
                concept_group_a_token_ids=group_a_token_ids,
                concept_group_b_token_ids=group_b_token_ids,
            ),
            None,
            0,
        ),
    )
    influence_result = cast(Any, influence_op(module, graph_result, None, 0))
    top_features_payload = dict(cast(Any, graph_result))
    top_features_payload.update(dict(cast(Any, influence_result)))
    top_features_result = cast(
        Any,
        top_features_op(module, AnalysisBatch(**top_features_payload), None, 0, top_n=case.n_top),
    )
    intervention_result = cast(
        Any,
        intervention_op(
            module,
            AnalysisBatch(
                prompts=[case.prompt],
                top_feature_ids=top_features_result.top_feature_ids,
                top_feature_scores=top_features_result.top_feature_scores,
                top_feature_activation_values=top_features_result.top_feature_activation_values,
                logit_target_ids=torch.tensor([target_a_id], dtype=torch.long),
            ),
            None,
            0,
        ),
    )

    pre_final = intervention_result.pre_intervention_logits.float().cpu()
    post_final = intervention_result.post_intervention_logits.float().cpu()

    # V3: Key-token logit analysis
    pre_ktl = _extract_key_token_logits(pre_final, tokenizer, cp.key_tokens)
    post_ktl = _extract_key_token_logits(post_final, tokenizer, cp.key_tokens)

    return InterventionResult(
        approach=approach,
        direction_mode="paired_rejection",
        top_feature_ids=[tuple(row.tolist()) for row in top_features_result.top_feature_ids],
        pre_gap=float((pre_final[target_a_id] - pre_final[target_b_id]).item()),
        post_gap=float((post_final[target_a_id] - post_final[target_b_id]).item()),
        pre_logits=pre_final,
        post_logits=post_final,
        pre_key_token_logits=pre_ktl,
        post_key_token_logits=post_ktl,
    )


# ---------------------------------------------------------------------------
# Direction consistency probes
# ---------------------------------------------------------------------------


def _probe_direction_consistency(
    embed_dir: DirectionResult,
    store_dir: DirectionResult,
    module: Any,
    concept_pair: ConceptPair,
) -> dict[str, Any]:
    """Probe whether the concept direction is internally consistent.

    Checks:
    1. Alignment: does the direction point from group_b → group_a in embed space?
    2. Sign consistency: do both directions agree on which end is "capital" vs "state"?
    3. Projection test: project group_a and group_b tokens onto the direction — are they separated?
    """
    tokenizer = module.replacement_model.tokenizer
    model = module.model

    # Get unembedding matrix — try multiple paths used by HookedTransformer / NNsight
    unembed = None
    # Path 1: HookedTransformer top-level W_U property
    w_u = getattr(model, "W_U", None)
    if isinstance(w_u, torch.Tensor):
        unembed = w_u.float().detach()  # shape: (d_model, d_vocab)
    # Path 2: model.unembed.W_U
    if unembed is None:
        unembed_mod = getattr(model, "unembed", None)
        if unembed_mod is not None:
            w_u = getattr(unembed_mod, "W_U", None)
            if isinstance(w_u, torch.Tensor):
                unembed = w_u.float().detach()
    # Path 3: use the embedding weight from the CT backend (transpose of embed for tied weights)
    if unembed is None:
        try:
            from interpretune.analysis.backends.circuit_tracer import CircuitTracerAnalysisBackend
            backend = CircuitTracerAnalysisBackend()
            embed_w = backend.get_embedding_weight(module).float().detach()
            # embed_w is typically (vocab, d_model) — transpose to (d_model, vocab)
            if embed_w.shape[0] > embed_w.shape[1]:
                unembed = embed_w.T  # (d_model, vocab)
            else:
                unembed = embed_w  # already (d_model, vocab)
        except Exception:
            pass

    if unembed is None:
        return {"error": "Could not find unembedding matrix"}

    probes: dict[str, Any] = {}

    for dir_result in [embed_dir, store_dir]:
        approach = dir_result.approach
        d = dir_result.direction_vector.to(unembed.device)

        # Project group_a and group_b token embeddings onto the direction
        a_projections = []
        b_projections = []

        for token in concept_pair.group_a_tokens:
            tid = tokenizer.encode(token, add_special_tokens=False)[-1]
            # unembed is [d_model, vocab] → transpose to get token vectors
            token_vec = unembed[:, tid].float()
            proj = float(torch.dot(token_vec, d).item())
            a_projections.append(proj)

        for token in concept_pair.group_b_tokens:
            tid = tokenizer.encode(token, add_special_tokens=False)[-1]
            token_vec = unembed[:, tid].float()
            proj = float(torch.dot(token_vec, d).item())
            b_projections.append(proj)

        mean_a = sum(a_projections) / len(a_projections)
        mean_b = sum(b_projections) / len(b_projections)

        probes[approach] = {
            "group_a_projections": a_projections,
            "group_b_projections": b_projections,
            "mean_group_a_projection": mean_a,
            "mean_group_b_projection": mean_b,
            "separation": mean_a - mean_b,
            "direction_points_a_minus_b": mean_a > mean_b,
        }

    # Cross-check: do both directions agree on which end is which?
    embed_sign = probes["embed"]["direction_points_a_minus_b"]
    store_sign = probes["store"]["direction_points_a_minus_b"]
    probes["sign_agreement"] = embed_sign == store_sign
    probes["potential_reversal"] = not probes["sign_agreement"]

    return probes


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_directions(
    embed_result: DirectionResult,
    store_result: DirectionResult,
    embed_intervention: InterventionResult | None = None,
    store_intervention: InterventionResult | None = None,
) -> ParityComparison:
    """Compare embed-based and store-based concept directions."""
    cosine = torch.nn.functional.cosine_similarity(
        embed_result.direction_vector.unsqueeze(0),
        store_result.direction_vector.unsqueeze(0),
    ).item()

    feature_jaccard = 0.0
    n_shared = 0
    n_total = 0
    embed_pre_gap = embed_post_gap = store_pre_gap = store_post_gap = 0.0

    if embed_intervention and store_intervention:
        embed_set = set(embed_intervention.top_feature_ids)
        store_set = set(store_intervention.top_feature_ids)
        n_shared = len(embed_set & store_set)
        n_total = len(embed_set | store_set)
        feature_jaccard = n_shared / n_total if n_total > 0 else 0.0

        embed_pre_gap = embed_intervention.pre_gap
        embed_post_gap = embed_intervention.post_gap
        store_pre_gap = store_intervention.pre_gap
        store_post_gap = store_intervention.post_gap

    return ParityComparison(
        cosine_similarity=cosine,
        feature_jaccard=feature_jaccard,
        embed_pre_gap=embed_pre_gap,
        embed_post_gap=embed_post_gap,
        store_pre_gap=store_pre_gap,
        store_post_gap=store_post_gap,
        n_shared_features=n_shared,
        n_total_features=n_total,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-variant", default="it", choices=list(MODEL_VARIANTS),
                        help="Model variant: 'it' for gemma-2-2b-it, 'base' for gemma-2-2b (default: it)")
    parser.add_argument("--concept-pair", default="capitals_states", choices=list(CONCEPT_PAIRS),
                        help="Concept pair to test (default: capitals_states)")
    parser.add_argument("--prompt", default=None,
                        help="Override intervention prompt (default: from concept pair)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--skip-intervention", action="store_true", help="Skip full pipeline comparison")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Disable chat template even for IT model")
    parser.add_argument(
        "--chat-template-method",
        default="apply_chat_template",
        choices=["apply_chat_template", "gemma_dataclass"],
        help="Chat template method (default: apply_chat_template)",
    )
    parser.add_argument("--test-direction-reversal", action="store_true",
                        help="Also test with negated concept_direction to check for reversal issues")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log directory (default: concept_direction_approach_parity folder)")
    args = parser.parse_args()

    harness_cfg = HarnessConfig(
        model_variant=args.model_variant,
        concept_pair_name=args.concept_pair,
        prompt=args.prompt,
        top_n=args.top_n,
        skip_intervention=args.skip_intervention,
        chat_template_method=args.chat_template_method,
        test_direction_reversal=args.test_direction_reversal,
        output=args.output,
        log_dir=args.log_dir,
    )
    if args.no_chat_template:
        harness_cfg.use_chat_template = False

    # Set up datetime-stamped log file
    log_path = _make_log_path(harness_cfg)
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        _run_experiment(harness_cfg)
    finally:
        logger.close()
        print(f"\nLog saved to: {log_path}", file=sys.__stdout__)


def _run_experiment(harness_cfg: HarnessConfig):
    """Run the full experiment suite."""
    cp = harness_cfg.concept_pair

    case = SemanticInterventionParityCase(
        prompt=cast(str, harness_cfg.prompt),
        capitals=cp.group_a_tokens,
        states=cp.group_b_tokens,
        label=cp.concept_label,
        n_top=harness_cfg.top_n,
    )

    print(f"\n{'='*80}")
    print("Concept Direction Approach Experimentation - V3")
    print(f"{'='*80}")
    print(f"Model: {harness_cfg.model_name}")
    print(f"Model variant: {harness_cfg.model_variant}")
    print(f"Concept pair: {cp.name} ({cp.description})")
    print(f"Chat template: {harness_cfg.use_chat_template} (method: {harness_cfg.chat_template_method})")
    print(f"Intervention prompt: {harness_cfg.prompt}")
    print(f"Classification question: {cp.classification_question}")
    print(f"Top-N features: {harness_cfg.top_n}")
    print(f"Test direction reversal: {harness_cfg.test_direction_reversal}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # --- Phase 1: Build embed-based direction ---
        print(f"\n{'='*80}")
        print("Phase 1: Embed-based concept_direction")
        print(f"{'='*80}")

        with _session_factory(tmp_path, "embed_direction", harness_cfg) as session:
            print("\n--- Tokenizer verification ---")
            tok_info = verify_tokenizer(session.module, cp)
            for group_name, group_data in tok_info.items():
                print(f"\n{group_name}:")
                for token, info in group_data.items():
                    print(f"  {token}: ids={info['token_ids']}, decoded='{info['decoded']}'")

            embed_dir = build_embed_direction(session.module, case)
            print(f"\nEmbed direction: norm={torch.linalg.vector_norm(embed_dir.direction_vector):.6f}")
            print(f"  mode={embed_dir.direction_mode}")

            embed_intervention = None
            if not harness_cfg.skip_intervention:
                print("\n--- Running full embed pipeline (graph → features → intervention) ---")
                embed_intervention = run_full_pipeline(
                    session.module, case, embed_dir.direction_vector, "embed", harness_cfg,
                    group_a_token_ids=embed_dir.group_a_token_ids,
                    group_b_token_ids=embed_dir.group_b_token_ids,
                )
                print(f"  Top features: {embed_intervention.top_feature_ids}")
                _print_gap_result("Embed", embed_intervention, cp)
                _print_key_token_table("Embed", embed_intervention.pre_key_token_logits,
                                       embed_intervention.post_key_token_logits)

        # --- Phase 2: Build store-based direction ---
        print(f"\n{'='*80}")
        print("Phase 2: Store-based concept_direction (classification prompts)")
        print(f"{'='*80}")

        with _session_factory(tmp_path, "store_direction", harness_cfg) as session:
            store_dir, prediction_info, sanity_info = build_store_direction(
                session.module, case, harness_cfg,
            )
            print(f"\nStore direction: norm={torch.linalg.vector_norm(store_dir.direction_vector):.6f}")
            print(f"  mode={store_dir.direction_mode}")
            print(f"  n_examples={store_dir.n_examples}")
            print(f"  correct_predictions={store_dir.correct_count}/{store_dir.n_examples}")

            print("\n--- Latent state sanity checks ---")
            for k, v in sanity_info.items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

            print("\n--- Model predictions on classification prompts ---")
            for ex in prediction_info["examples"]:
                status = f"rank={ex['rank']}" if ex["correct"] else "MISS"
                print(f"  [{ex['group']:>10}] {ex['prompt']}")
                print(f"              expected={ex['expected']!r} {status}  top5={ex['top5_tokens']}")

            store_intervention = None
            if not harness_cfg.skip_intervention:
                print("\n--- Running full store pipeline (graph → features → intervention) ---")
                store_intervention = run_full_pipeline(
                    session.module, case, store_dir.direction_vector, "store", harness_cfg,
                    group_a_token_ids=store_dir.group_a_token_ids,
                    group_b_token_ids=store_dir.group_b_token_ids,
                )
                print(f"  Top features: {store_intervention.top_feature_ids}")
                _print_gap_result("Store", store_intervention, cp)
                _print_key_token_table("Store", store_intervention.pre_key_token_logits,
                                       store_intervention.post_key_token_logits)

            # --- Direction consistency probes ---
            print(f"\n{'='*80}")
            print("Phase 2b: Direction Consistency Probes")
            print(f"{'='*80}")
            probes = _probe_direction_consistency(embed_dir, store_dir, session.module, cp)
            _print_direction_probes(probes)

            # --- Phase 2c: Direction reversal test ---
            reversed_intervention = None
            if harness_cfg.test_direction_reversal and not harness_cfg.skip_intervention:
                print(f"\n{'='*80}")
                print("Phase 2c: Direction Reversal Test (negated embed direction)")
                print(f"{'='*80}")
                negated_direction = -embed_dir.direction_vector
                reversed_intervention = run_full_pipeline(
                    session.module, case, negated_direction, "embed_negated", harness_cfg,
                    group_a_token_ids=embed_dir.group_a_token_ids,
                    group_b_token_ids=embed_dir.group_b_token_ids,
                )
                _print_gap_result("Embed-Negated", reversed_intervention, cp)
                _print_key_token_table("Embed-Negated", reversed_intervention.pre_key_token_logits,
                                       reversed_intervention.post_key_token_logits)

        # --- Phase 3: Compare ---
        print(f"\n{'='*80}")
        print("Phase 3: Comparison")
        print(f"{'='*80}")

        comparison = compare_directions(embed_dir, store_dir, embed_intervention, store_intervention)

        print(f"\n  Cosine similarity:    {comparison.cosine_similarity:.6f}")
        n_total_examples = len(prediction_info["examples"])
        print(f"  Correct predictions:  {prediction_info['n_correct']}/{n_total_examples}")
        if embed_intervention and store_intervention:
            print(f"  Feature Jaccard:      {comparison.feature_jaccard:.4f}")
            print(f"  Shared features:      {comparison.n_shared_features}/{comparison.n_total_features}")
            embed_delta = comparison.embed_post_gap - comparison.embed_pre_gap
            store_delta = comparison.store_post_gap - comparison.store_pre_gap
            print(
                f"  Embed  pre/post gap:  {comparison.embed_pre_gap:.4f}"
                f" / {comparison.embed_post_gap:.4f} (Δ={embed_delta:+.4f})"
            )
            print(
                f"  Store  pre/post gap:  {comparison.store_pre_gap:.4f}"
                f" / {comparison.store_post_gap:.4f} (Δ={store_delta:+.4f})"
            )

        if reversed_intervention:
            rev_delta = reversed_intervention.post_gap - reversed_intervention.pre_gap
            print(
                f"\n  Embed-negated pre/post gap: {reversed_intervention.pre_gap:.4f}"
                f" / {reversed_intervention.post_gap:.4f} (Δ={rev_delta:+.4f})"
            )

        # --- Phase 4: Agent hypothesis - Embed intervention with store token IDs ---
        if not harness_cfg.skip_intervention and store_dir.group_a_token_ids and embed_dir.group_a_token_ids:
            # Hypothesis: the token ID list used in graph computation matters.
            # The embed approach uses SentencePiece-encoded IDs while the store approach
            # resolves them separately. Check if using store's token IDs with embed's direction
            # (or vice versa) changes the result.
            print(f"\n{'='*80}")
            print("Phase 4: Agent Hypothesis — Cross Token-ID Test")
            print("  Testing: embed direction + store token IDs")
            print(f"{'='*80}")

            # Re-use the last session for this
            with _session_factory(tmp_path, "cross_token_test", harness_cfg) as session:
                cross_intervention = run_full_pipeline(
                    session.module, case, embed_dir.direction_vector, "embed_w_store_ids", harness_cfg,
                    group_a_token_ids=store_dir.group_a_token_ids,
                    group_b_token_ids=store_dir.group_b_token_ids,
                )
                cross_delta = cross_intervention.post_gap - cross_intervention.pre_gap
                print(f"  Pre-gap:  {cross_intervention.pre_gap:.4f}")
                print(f"  Post-gap: {cross_intervention.post_gap:.4f}")
                print(f"  Gap Δ:    {cross_delta:+.4f}")
                _print_key_token_table("Cross-Token-ID", cross_intervention.pre_key_token_logits,
                                       cross_intervention.post_key_token_logits)

                # Check if token IDs actually differ
                ids_match = (embed_dir.group_a_token_ids == store_dir.group_a_token_ids and
                             embed_dir.group_b_token_ids == store_dir.group_b_token_ids)
                print(f"\n  Token IDs match between approaches: {ids_match}")
                if not ids_match:
                    print(f"    Embed group_a IDs: {embed_dir.group_a_token_ids}")
                    print(f"    Store group_a IDs: {store_dir.group_a_token_ids}")
                    print(f"    Embed group_b IDs: {embed_dir.group_b_token_ids}")
                    print(f"    Store group_b IDs: {store_dir.group_b_token_ids}")

        # --- Output JSON ---
        output = {
            "experiment_version": "V3",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model_name": harness_cfg.model_name,
            "model_variant": harness_cfg.model_variant,
            "concept_pair": cp.name,
            "use_chat_template": harness_cfg.use_chat_template,
            "chat_template_method": harness_cfg.chat_template_method,
            "classification_question": cp.classification_question,
            "prompt": case.prompt,
            "group_a_tokens": cp.group_a_tokens,
            "group_b_tokens": cp.group_b_tokens,
            "top_n": case.n_top,
            "tokenizer_info": tok_info,
            "embed_direction_norm": float(torch.linalg.vector_norm(embed_dir.direction_vector)),
            "store_direction_norm": float(torch.linalg.vector_norm(store_dir.direction_vector)),
            "cosine_similarity": comparison.cosine_similarity,
            "model_predictions": prediction_info,
            "latent_sanity": sanity_info,
            "direction_probes": probes,
        }
        if embed_intervention and store_intervention:
            output.update({
                "feature_jaccard": comparison.feature_jaccard,
                "n_shared_features": comparison.n_shared_features,
                "n_total_features": comparison.n_total_features,
                "embed_pre_gap": comparison.embed_pre_gap,
                "embed_post_gap": comparison.embed_post_gap,
                "store_pre_gap": comparison.store_pre_gap,
                "store_post_gap": comparison.store_post_gap,
                "embed_key_token_logits": {
                    "pre": [{"token": k.token_name, "logit": k.logit_value, "rank": k.rank}
                            for k in embed_intervention.pre_key_token_logits],
                    "post": [{"token": k.token_name, "logit": k.logit_value, "rank": k.rank}
                             for k in embed_intervention.post_key_token_logits],
                },
                "store_key_token_logits": {
                    "pre": [{"token": k.token_name, "logit": k.logit_value, "rank": k.rank}
                            for k in store_intervention.pre_key_token_logits],
                    "post": [{"token": k.token_name, "logit": k.logit_value, "rank": k.rank}
                             for k in store_intervention.post_key_token_logits],
                },
            })
        if reversed_intervention:
            rev_delta = reversed_intervention.post_gap - reversed_intervention.pre_gap
            output["reversed_embed_pre_gap"] = reversed_intervention.pre_gap
            output["reversed_embed_post_gap"] = reversed_intervention.post_gap
            output["reversed_embed_delta"] = rev_delta

        if harness_cfg.output:
            output_path = Path(harness_cfg.output)
            output_path.write_text(json.dumps(output, indent=2, default=str))
            print(f"\nResults written to {output_path}")
        else:
            print("\n--- JSON output ---")
            print(json.dumps(output, indent=2, default=str))


def _print_gap_result(label: str, intervention: InterventionResult, cp: ConceptPair):
    """Print the gap result for an intervention."""
    delta = intervention.post_gap - intervention.pre_gap
    target_desc = "Austin−Dallas" if cp.name == "capitals_states" else f"{cp.group_a_name}−{cp.group_b_name}"
    print(f"  Pre-gap ({target_desc}):  {intervention.pre_gap:.4f}")
    print(f"  Post-gap ({target_desc}): {intervention.post_gap:.4f}")
    print(f"  Gap Δ: {delta:+.4f}")


def _print_direction_probes(probes: dict[str, Any]):
    """Print direction consistency probe results."""
    if "error" in probes:
        print(f"  Error: {probes['error']}")
        return

    for approach in ["embed", "store"]:
        if approach not in probes:
            continue
        p = probes[approach]
        print(f"\n  {approach.upper()} direction probes:")
        print(f"    Mean group_a projection: {p['mean_group_a_projection']:.6f}")
        print(f"    Mean group_b projection: {p['mean_group_b_projection']:.6f}")
        print(f"    Separation (a − b):      {p['separation']:.6f}")
        print(f"    Direction points a > b:   {p['direction_points_a_minus_b']}")
        print(f"    Group A projections: {[f'{x:.4f}' for x in p['group_a_projections']]}")
        print(f"    Group B projections: {[f'{x:.4f}' for x in p['group_b_projections']]}")

    print(f"\n  Sign agreement: {probes.get('sign_agreement', 'N/A')}")
    if probes.get("potential_reversal"):
        print("  ⚠️  POTENTIAL DIRECTION REVERSAL DETECTED!")
        print("  The embed and store directions disagree on which end is group_a vs group_b.")


if __name__ == "__main__":
    main()
