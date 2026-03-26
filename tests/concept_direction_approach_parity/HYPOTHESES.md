# Concept Direction Experiment — Hypothesis Tracker

**Started**: 2025-06-26  
**Context**: V3 CLI experiments revealed unexpected intervention behaviors;  
these notebook experiments systematically investigate root causes.

---

## Hypothesis 1: Scale Factor Regime Transition

**Statement**: There exists a scale factor threshold below which the CT intervention
produces targeted concept steering, and above which it produces broad logit disruption.

**Motivation**: V3 used `scale_factor=10`, which produced 10–20 logit shifts — well
beyond the model's natural operating regime. The base model capitals_states result
(+2.97 gap delta) may be an artifact of a narrow sweet spot rather than genuine
concept steering.

**Test**: Scale factor sweep [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0].
Measure gap delta and inspect whether all key-token logits move in the same direction
(disruption) vs. only the target token moving (steering).

**Status**: [ ] Not started  
**Outcome**: _pending_

---

## Hypothesis 2: Sign-Agnostic Feature Selection Dilutes Signal

**Statement**: The `abs_()` in `compute_partial_influences` (graph.py:390) causes
the top-N features to include features that **oppose** the concept direction alongside
those that promote it. Intervening on this mixed set partially cancels the intended
effect.

**Motivation**: If half the top-10 features promote group A and half promote group B,
then amplifying all of them produces a near-zero net effect. The base model
capitals_states Δ=+2.97 may be the result of an asymmetric feature split
rather than consistent alignment.

**Test**: Classify top-2N features by activation sign. Run separate interventions:
(a) amplify only positive-activation features, (b) ablate only negative-activation
features, (c) original mixed set. Compare gap deltas.

**Status**: [ ] Not started  
**Outcome**: _pending_

---

## Hypothesis 3: Transcoder Mismatch Causes IT Model Disruption

**Statement**: Gemma transcoders are trained on the **base model**. When applied to the
IT model, they decompose activations into features calibrated for base-model
representations, and amplifying these features injects base-model patterns into the IT
model's forward pass, causing broad logit suppression.

**Motivation**: V3 showed IT model capitals_states embed intervention produces
**all key-token logits dropping 5–20 points** (broad suppression), while base model
shows selective suppression. This asymmetry aligns with transcoder mismatch.

**Test**: 
1. Run gemma3 PT model with PT-trained transcoders → expect base-like behavior
2. Run gemma3 IT model with IT-trained transcoders → if mismatch is the cause,
   IT+IT transcoders should show more targeted interventions than gemma2 IT+base transcoders
3. (If feasible) Run gemma3 IT model with PT transcoders → should show disruption

**Status**: [ ] Not started  
**Outcome**: _pending_

---

## Hypothesis 4: Store Direction Selects More Influential but Less Aligned Features

**Statement**: The store-based concept direction produces stronger gap deltas because
it selects inherently more influential features (higher activation magnitudes), but
these features are not necessarily more aligned with the concept direction.

**Motivation**: V3 showed store directions produce stronger but more disruptive effects
(e.g., dog_cat IT: store Δ=−3.59 vs embed Δ=+0.25). The low cosine similarity (0.05–0.24)
between embed and store directions suggests they select very different feature sets.

**Test**: Compare top-N feature sets from embed vs. store: (a) mean absolute activation
value, (b) mean influence score, (c) Jaccard overlap, (d) activation sign distribution.
If store features have higher mean activation but more mixed sign, this hypothesis holds.

**Status**: [ ] Not started  
**Outcome**: _pending_

---

## Hypothesis 5: Progressive Ablation Reveals Causal Feature Density

**Statement**: The number of features needed to materially shift the gap (ablation
inflection point) indicates whether the concept is encoded in a few dominant features
or distributed across many.

**Motivation**: If ablating the top-5 features produces ~80% of the top-100 ablation
effect, the concept is concentrated. If the effect is roughly linear in N, it is
distributed. This affects whether targeted interventions are even feasible.

**Test**: Progressive ablation with N = [5, 10, 25, 50, 100]. Plot gap delta vs. N.
Measure slope to determine concentration.

**Status**: [ ] Not started  
**Outcome**: _pending_

---

## Hypothesis 6: Concept Pair Determines Intervention Direction, Not Model

**Statement**: V2 hypothesized IT instruction tuning causes gap decrease. V3 refuted
this — the base model with dog_cat also shows gap decrease (Δ=−4.34). The concept
pair's token-embedding geometry determines which direction the intervention pushes.

**Motivation**: Different concept pairs have different token-embedding structures.
"capitals_states" has geographically related tokens (Austin/Dallas are both Texas
cities), while "dog_cat" has categorically related tokens. The embedding geometry
determines which features are selected and their relative activation patterns.

**Test**: Compare direction consistency probes across concept pairs and models.
If group A/B token projections onto the concept direction have consistent sign
patterns within a concept pair (regardless of model), this hypothesis holds.

**Status**: [ ] Not started  
**Outcome**: _pending_

---

## Hypothesis 7: Gemma3 1B Models Show Same Patterns as Gemma2 2B

**Statement**: The phenomena observed in Gemma2 2B (sign-agnostic selection, broad
disruption at high scale factors, low embed-store cosine similarity) are model-size
and tokenizer-independent artifacts of the CT pipeline architecture.

**Motivation**: If the same patterns appear in a different model family (Gemma3 1B)
with different tokenizers and vocabulary, the root causes are in the pipeline
(abs_(), transcoder decomposition, feature intervention mechanism) rather than
model-specific representation properties.

**Test**: Run the full experiment suite on Gemma3 1B PT and IT models. Compare
gap deltas, cosine similarities, and sign distributions to Gemma2 2B results.

**Status**: [ ] Not started  
**Outcome**: _pending_

---

## Experiment Execution Log

| Date | Model | Concept | Experiment | Key Result | Notes |
|------|-------|---------|-----------|------------|-------|
| _pending_ | | | | | |

---

## Cross-Reference

- **V3 Analysis**: [V3_ANALYSIS.md](V3_ANALYSIS.md) — full V3 results and architectural insights
- **Implementation Plan**: See `ct_backend_implementation_plan.md` for session logs
- **Root Cause**: `abs_()` in `circuit_tracer/graph.py:390`
- **Gemma2 Notebook**: `concept_direction_experiment_gemma2.ipynb`
- **Gemma3 Notebook**: `concept_direction_experiment_gemma3.ipynb`
