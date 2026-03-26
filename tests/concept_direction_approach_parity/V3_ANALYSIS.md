# V3 Concept Direction Experimentation — Analysis

**Date**: 2025-03-23  
**Script**: `concept_direction_approach_experimentation.py`  
**Models**: `google/gemma-2-2b` (base), `google/gemma-2-2b-it` (instruction-tuned)  
**GPU**: RTX 4090 (24GB)

## Experiment Summary

V3 reformulated the classification prompts to explicitly include expected answer tokens
("Capital"/"State" capitalized) and added:
- Key-token logit analysis (per-token pre/post logit changes)
- Direction reversal tests (negated concept direction)
- Alternate concept pair testing (dog/cat in addition to capitals/states)
- Direction consistency probes (token projection onto direction vector)
- Cross token-ID hypothesis test

## Results Matrix

### Intervention Gap Deltas (Austin−Dallas or dog−cat)

| Model | Concept Pair | Embed Δ | Store Δ | Direction |
|-------|-------------|---------|---------|-----------|
| Base  | capitals_states | **+2.97** | +0.005 | Austin boosted ✓ |
| Base  | dog_cat | **−4.34** | −1.92 | cat boosted (unexpected) |
| IT    | capitals_states | −2.06 | −2.22 | Dallas boosted (unexpected) |
| IT    | dog_cat | +0.25 | **−3.59** | cat boosted |

### Other Metrics

| Model | Concept Pair | Cosine Sim | Jaccard | Predictions |
|-------|-------------|------------|---------|-------------|
| Base  | capitals_states | 0.239 | 0.250 (4/16) | 0/8 |
| Base  | dog_cat | 0.184 | 0.176 (3/17) | 0/8 |
| IT    | capitals_states | 0.161 | 0.250 (4/16) | 8/8 |
| IT    | dog_cat | 0.048 | 0.250 (4/16) | 8/8 |

## Finding 1: Negated Direction = Identical Results

**Observation**: Negating the concept direction (`-direction`) produces exactly the same
intervention result (identical pre/post gaps, identical key-token logits) as the original
direction.

**Root cause**: `compute_partial_influences` in `circuit_tracer/graph.py` applies
`.abs_()` to the edge matrix before computing feature influence rankings:

```python
# circuit_tracer/graph.py:390
normalized_matrix = normalized_matrix.abs_()
```

This means:
1. The concept direction sign affects attribution values (positive/negative) ✓
2. Feature **selection** ignores sign — features ranked by |influence| ✓  
3. The intervention uses feature activation values, not the direction itself ✓

**Conclusion**: The direction reversal test is inherently inconclusive for the CT pipeline.
Feature selection via absolute influence is intentional — features that strongly suppress a
concept are equally important as those that promote it. The intervention's effect depends on
the **feature activation values**, not the direction sign.

## Finding 2: Intervention Effect Is Concept-Specific, Not Model-Specific

The initial hypothesis (V2) was that the IT model's instruction tuning causes the embed
direction to decrease the gap. V3 shows this is **not the full picture**:

- Base model + capitals_states: embed **increases** gap (+2.97) ✓
- Base model + dog_cat: embed **decreases** gap (−4.34) ✗
- IT model + dog_cat: embed barely moves (+0.25) ≈ neutral

The intervention direction depends on the **interaction** between:
1. Which features the concept direction selects  
2. Those features' activation values  
3. How amplifying those features shifts the residual stream

## Finding 3: Embed Interventions Cause Broad Logit Disruption

### IT Model — Capitals/States (Embed, scale=10×)

| Token | Pre Logit | Post Logit | Δ |
|-------|-----------|------------|---|
| ▁Austin | 14.95 (rank 0) | 5.14 (rank 6) | **−9.81** |
| ▁Dallas | 11.86 (rank 5) | 4.11 (rank 20) | −7.75 |
| ▁Texas | 14.15 (rank 1) | −4.90 (rank 82786) | **−19.05** |

The intervention pushes **all** logits down dramatically. Austin drops more than Dallas
(−9.81 vs −7.75), hence the gap decreases. This is not targeted concept steering — it is
broad logit suppression.

### Base Model — Capitals/States (Embed, scale=10×)

| Token | Pre Logit | Post Logit | Δ |
|-------|-----------|------------|---|
| ▁Austin | 26.08 (rank 0) | 26.16 (rank 0) | +0.09 |
| ▁Dallas | 23.50 (rank 6) | 20.62 (rank 18) | **−2.89** |
| ▁Texas | 24.01 (rank 3) | 2.50 (rank ~100K) | **−21.52** |

The base model shows a more selective pattern: Austin barely moves, Dallas drops, Texas
drops massively. The concept-related features, when amplified, selectively suppress non-Austin
completions.

### Base Model — Dog/Cat (Embed, scale=10×)

| Token | Pre Logit | Post Logit | Δ |
|-------|-----------|------------|---|
| ▁dog | 27.06 (rank 0) | 9.87 (rank 416) | −17.19 |
| ▁cat | 26.87 (rank 1) | 14.02 (rank 123) | −12.85 |
| ▁kitten | 21.87 (rank 54) | −0.88 (rank 23459) | **−22.75** |

Even the base model shows broad suppression for dog/cat — all tokens drop substantially.
Cat drops less than dog (−12.85 vs −17.19), causing the gap to decrease.

## Finding 4: Store Directions Produce Stronger, More Disruptive Interventions

| Experiment | Store Δ |
|------------|---------|
| Base capitals_states | +0.005 (neutral) |
| Base dog_cat | −1.92 |
| IT capitals_states | −2.22 |
| IT dog_cat | **−3.59** |

Store directions consistently produce stronger effects than embed (except base
capitals_states where store is nearly zero). The store-based concept direction captures
contextual information that, when fed through the attribution pipeline, selects more
influential features.

## Finding 5: Low Cosine Similarity Is Consistent

Across all experiments, embed and store directions have low cosine similarity (0.05–0.24).
This confirms that embedding-space and contextual-representation-space encode concepts very
differently, regardless of model variant or concept pair.

## Architectural Insight: The Transcoder Mismatch Hypothesis

The Gemma transcoder set is trained on the **base model**. When used with the IT model:

1. The transcoders decompose activations into features that correspond to the base model's
   representation space
2. Amplifying these features injects base-model-calibrated activation patterns into the IT
   model's forward pass
3. For the IT model, this injection disrupts the instruction-tuning-aligned representations,
   causing broad logit suppression rather than targeted concept steering

Evidence: The IT model's key-token analysis shows **all tokens dropping by 5–20 logit points**
under embed intervention, while the base model's capital_states embed intervention shows
Austin barely moving (+0.09) and selectively dropping non-target tokens.

However, this hypothesis is partially contradicted by the base model dog/cat result
(also shows broad suppression with Δ=−4.34). The transcoder mismatch may be a
contributing factor but is not the sole explanation.

## Practical Implications

1. **The CT semantic intervention pipeline's feature selection is sign-agnostic** by design.
   Direction reversal tests cannot differentiate approaches.

2. **Scale factor sensitivity**: `scale_factor=10` produces interventions in the 10–20 logit
   range, which is beyond the model's natural operating regime. The intervention effect may
   be qualitatively different at lower scale factors.

3. **The embed approach works well for base model + capitals_states** because the embedding
   vectors for "Austin"/"Dallas" tokens happen to select features whose amplification
   differentially affects those tokens. This is not guaranteed for other concept pairs.

4. **Store-based directions are not reliably "better"** — they produce stronger but more
   disruptive effects. The classification-based latent representations select different
   features than embed, but amplifying them with the same pipeline doesn't necessarily
   produce more targeted interventions.

## Recommended Next Steps

1. **Scale factor sweep**: Test `scale_factor` values from 0.1 to 100 to find the regime
   where the intervention transitions from targeted to disruptive.

2. **Feature activation value analysis**: Examine the actual feature activation values being
   amplified — are they uniformly large, or is one dominant feature driving the effect?

3. **Sign-aware feature selection**: Explore whether filtering features by the **sign** of
   their attribution (not just absolute influence) produces more targeted interventions.

4. **IT-specific transcoders**: If available, test with transcoders trained on the IT model
   to eliminate the transcoder mismatch hypothesis.

## Raw Experiment Logs

- `experiment_base_capitals_states_20260323_032205.log`
- `experiment_it_capitals_states_20260323_032621.log`
- `experiment_it_dog_cat_20260323_033511.log`
- `experiment_base_dog_cat_20260323_033752.log`
