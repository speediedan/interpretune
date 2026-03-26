# Concept Direction Approach Parity

## Overview

This directory contains tools for comparing two approaches to building concept_direction vectors
used in circuit-tracer semantic interventions:

1. **Embed-based** (existing, validated): Uses raw embedding/unembedding weight vectors
   for capital/state tokens. No model forward pass required. This approach is validated in
   `tests/core/test_analysis_backend_parity.py` and matches the upstream circuit-tracer's
   `demo_utils.get_unembed_vecs()` approach.

2. **Store-based** (new, dataset-driven): Runs the model forward on classification-style
   prompts (e.g., "Austin : Is this a capital or a state? Answer with only one word:"),
   caches `unembed.hook_in` activations (pre-logit d_model space), extracts per-example
   latent states at the answer position, then computes concept_direction via
   `paired_rejection` on those latent states.

## Architecture Notes

The circuit-tracer NNsight module uses its own `TranscoderSet` (not SAELens SAE objects),
so the `fwd_w_cache_and_latent_models` path (which requires `sae_handles` from
`_SAEHandleMixin`) is **not available**. We use `fwd_w_cache` with `unembed.hook_in` to
capture pre-logit residual-stream activations from a plain forward pass. Transcoder splicing
happens only during the attribution graph computation.

## Theoretical Differences

The two approaches operate in the same vector space (d_model) but capture different things:

- **Embed-based**: The raw embedding vector for a token (e.g., "▁Austin") represents the
  static, context-independent representation of that token.

- **Store-based**: The unembed.hook_in activation at the answer position represents the
  model's contextual representation after processing the full prompt through all transformer
  layers. For classification prompts, this captures the model's internal "about to predict
  capital vs state" distinction.

## Empirical Results

### V2: gemma-2-2b-it with classification prompts (2025-03-23)

**Setup**: Instruction-tuned model with chat template, classification prompts
("X : Is this a capital or a state? Answer with only one word:").

#### Direction Vector Comparison

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.111 |
| Both direction norms | 1.0 |
| Model predictions correct | 4/8 (states only, lowercase "state" at ranks 2-7) |

The model uses capitalized tokens ("State", "Capital", "City") rather than the lowercase
expected tokens ("state", "capital"). If matching case-insensitively:
- "Capital" appears in top-3 for all 4 capital cities
- "State" appears as rank 0 for all 4 states

#### Latent State Sanity Checks

| Metric | Value |
|--------|-------|
| Mean norm | 226.86 |
| Min/max norm | 222.69 / 233.79 |
| All finite | True |
| Mean cosine within capitals | 0.981 |
| Mean cosine within states | 0.988 |
| Mean cosine between groups | 0.980 |

Within-group cosines are slightly higher than between-group (0.981/0.988 vs 0.980),
confirming the latent states encode a small but measurable capital-vs-state distinction.

#### Full Pipeline Comparison

| Metric | Embed-based | Store-based |
|--------|-------------|-------------|
| Pre-intervention gap (Austin−Dallas) | 3.09 | 3.09 |
| Post-intervention gap (Austin−Dallas) | 1.03 | **−1.34** |
| Gap delta | −2.06 | **−4.43** |
| Top-10 feature count | 10 | 10 |
| Feature Jaccard overlap | 0.25 (4/16 shared) |

#### Key Findings (V2)

- **Store-based direction is dramatically stronger**: The store-based intervention produces
  a gap delta of −4.43 vs the embed's −2.06, and it **reverses the logit gap**, pushing
  the model from preferring Austin (+3.09) to preferring Dallas (−1.34).

- Both approaches now steer in the same general direction (reducing the Austin−Dallas gap),
  unlike V1 where they produced opposite effects. The IT model + classification prompts
  produce a more coherent concept direction that aligns with the attribution graph framework.

- The low cosine similarity (0.111) between the two directions is consistent with V1 (0.149)
  and reflects a fundamental difference between static token embeddings and contextual
  pre-logit representations.

### V1: gemma-2-2b base with factual-completion prompts (2025-03-22)

**Setup**: Base model (no chat template), factual-completion prompts
("The capital of Texas is" → "Austin").

#### Direction Vector Comparison

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.149 |
| Both direction norms | 1.0 |
| Model predictions correct | 8/8 (all rank 0) |

#### Full Pipeline Comparison

| Metric | Embed-based | Store-based |
|--------|-------------|-------------|
| Pre-intervention gap (Austin−Dallas) | 2.57 | 2.57 |
| Post-intervention gap (Austin−Dallas) | 5.55 | 0.99 |
| Gap delta | **+2.97** | **−1.58** |

**V1 Critical Observation**: The embed-based direction steered toward Austin (+2.97) while
the store-based direction steered away from Austin (−1.58), indicating the factual-completion
latent states encoded the concept differently than the attribution framework expected.

### V3: 2×2 Experimentation Matrix (2025-03-23)

**Setup**: Both models, two concept pairs (capitals/states + dog/cat), classification prompts
with explicit answer tokens, datetime-stamped logs, key-token logit analysis.

#### Results Matrix

| Model | Concept Pair | Embed Δ | Store Δ | Cosine Sim | Jaccard | Predictions |
|-------|-------------|---------|---------|------------|---------|-------------|
| Base  | capitals/states | **+2.97** | +0.005 | 0.239 | 0.250 | 0/8 |
| Base  | dog/cat | −4.34 | −1.92 | 0.184 | 0.176 | 0/8 |
| IT    | capitals/states | −2.06 | −2.22 | 0.161 | 0.250 | 8/8 |
| IT    | dog/cat | +0.25 | **−3.59** | 0.048 | 0.250 | 8/8 |

#### V3 Key Findings

1. **Negated direction = identical results**: Root cause is `abs_()` in
   `compute_partial_influences` (circuit_tracer/graph.py:390). Feature selection is
   sign-agnostic by design; interventions use feature activation values, not the direction.

2. **Intervention effect is concept-specific, not model-specific**: The embed approach
   increases the gap for base+capitals (+2.97) but decreases it for base+dog/cat (−4.34).
   The direction of effect depends on which features the concept selects and their
   activation values, not on whether the model is instruction-tuned.

3. **Broad logit disruption at scale_factor=10**: Key-token logit analysis shows all tokens
   dropping by 5–20 logit points under embed intervention. This is not targeted concept
   steering — it is broad suppression from amplifying features beyond the model's
   natural operating regime.

4. **Transcoder mismatch is a contributing factor**: Gemma transcoders are trained on the
   base model. When used with the IT model, they inject base-model-calibrated activation
   patterns that can disrupt instruction-tuning-aligned representations.

See [V3_ANALYSIS.md](V3_ANALYSIS.md) for the full analysis.

## Practical Implications

1. **CT feature selection is sign-agnostic**: Direction reversal tests are inherently
   inconclusive for the CT pipeline because `compute_partial_influences` uses absolute
   influence values. Features that oppose a concept are selected equally to those that
   promote it.

2. **Scale factor sensitivity**: `scale_factor=10` produces interventions in the 10–20 logit
   range, beyond the model's natural operating regime. Lower scale factors may produce
   qualitatively different (more targeted) effects.

3. **Store-based directions are not reliably "better"**: They produce stronger but more
   disruptive effects. The classification-based latent representations select different
   features than embed, but amplifying them doesn't necessarily produce more targeted
   interventions.

4. **For future work**: Scale factor sweeps, sign-aware feature selection, and IT-specific
   transcoders are the most promising avenues for improving intervention precision.

## Files

- `concept_direction_approach_experimentation.py`: V3 experimentation harness with 2×2
  model/concept matrix, key-token logit analysis, direction reversal tests, and
  datetime-stamped log output.
- `compare_approaches.py`: V1/V2 harness (predecessor). Supports `--model-variant it|base`,
  `--no-chat-template`, `--skip-intervention`, `--output`.
- `V3_ANALYSIS.md`: Comprehensive V3 analysis document with full results and findings.
- `README.md`: This file.
- `experiment_*.log`: Timestamped experiment logs from V3 runs.

## Usage

```bash
cd /home/speediedan/repos/interpretune
source /mnt/cache/speediedan/.venvs/it_latest/bin/activate

# V3: Run specific model + concept pair experiments
python tests/concept_direction_approach_parity/concept_direction_approach_experimentation.py \
  --model-variant base --concept-pair capitals_states
python tests/concept_direction_approach_parity/concept_direction_approach_experimentation.py \
  --model-variant it --concept-pair dog_cat

# V1/V2: Legacy harness
python -m tests.concept_direction_approach_parity.compare_approaches --model-variant base
python -m tests.concept_direction_approach_parity.compare_approaches --model-variant it
```

## Comparison Metrics

1. **Cosine similarity** of the direction vectors (0.05–0.24 across experiments)
2. **Feature Jaccard**: Overlap of top-N attributed features from the graph (~0.18–0.25)
3. **Pre/post intervention gap delta**: Change in concept token logit gap
4. **Key-token logit analysis**: Per-token pre/post logit and rank changes (V3)
5. **Direction reversal test**: Negated vs normal direction comparison (V3)
6. **Latent state sanity checks**: Norms, finiteness, within- vs between-group cosines
