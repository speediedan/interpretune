# Concept Direction Quality Analysis

This document tracks the quality of concept-direction vectors derived from different
extraction approaches, using Neuronpedia feature interpretations and direction-probe
separations as diagnostic tools.

## Model and SAE Configuration

| Property | Value |
|----------|-------|
| Model | gemma-3-4b-it |
| SAE width | 262,144 (262k) |
| SAE set | `gemmascope-2-transcoder-262k` |
| Layers available | 0–33 (34 total) |
| Concept pair | capitals_states (Austin ↔ Dallas) |
| Prompt | standard (chat template, "Answer with only the missing city name. The capital of Texas is") |

> **Note:** The 16k-width transcoders used in the main experiment harness have limited
> Neuronpedia coverage across only 12 layers. The 262k-width set (used in the baseline
> notebook `gemma3_4b_262k_it_capitals_states_20260330_173050.ipynb`) provides the richer
> feature library used for the analysis below.

---

## Top Features by Direction Approach (262k Width)

### Embed Direction (Token-Embedding Difference)

Top higher-layer features aligned with the embed-space concept direction:

| Layer/Feature | Neuronpedia Link | Interpretation | Notes |
|:-------------|:----------------|:---------------|:------|
| 5/445 | [gemma-3-4b-it/5-gemmascope-2-transcoder-262k/445](https://www.neuronpedia.org/gemma-3-4b-it/5-gemmascope-2-transcoder-262k/445) | "what is the capital of" | Directly task-relevant |
| 10/168247 | [gemma-3-4b-it/10-gemmascope-2-transcoder-262k/168247](https://www.neuronpedia.org/gemma-3-4b-it/10-gemmascope-2-transcoder-262k/168247) | "increased" | Semantically ambiguous |
| 22/8872 | [gemma-3-4b-it/22-gemmascope-2-transcoder-262k/8872](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-transcoder-262k/8872) | "Texas locations" | Geographically relevant |
| 23/13341 | [gemma-3-4b-it/23-gemmascope-2-transcoder-262k/13341](https://www.neuronpedia.org/gemma-3-4b-it/23-gemmascope-2-transcoder-262k/13341) | "capital cities" | Directly task-relevant |
| 25/1387 | [gemma-3-4b-it/25-gemmascope-2-transcoder-262k/1387](https://www.neuronpedia.org/gemma-3-4b-it/25-gemmascope-2-transcoder-262k/1387) | "cities" | Broadly relevant |
| 25/3015 | [gemma-3-4b-it/25-gemmascope-2-transcoder-262k/3015](https://www.neuronpedia.org/gemma-3-4b-it/25-gemmascope-2-transcoder-262k/3015) | "cities" | Broadly relevant |

**Summary:** 4 of 6 top embed features are directly or geographically relevant to the
capital/state distinction. The embed direction selects a semantically coherent feature set.

### Store Direction (Answer-Position Latent State Difference)

Top features aligned with the store-space (answer-position activation) concept direction:

| Layer/Feature | Neuronpedia Link | Interpretation | Notes |
|:-------------|:----------------|:---------------|:------|
| 10/168247 | [gemma-3-4b-it/10-gemmascope-2-transcoder-262k/168247](https://www.neuronpedia.org/gemma-3-4b-it/10-gemmascope-2-transcoder-262k/168247) | "increased" | Shared with embed; semantically ambiguous |
| 25/3015 | [gemma-3-4b-it/25-gemmascope-2-transcoder-262k/3015](https://www.neuronpedia.org/gemma-3-4b-it/25-gemmascope-2-transcoder-262k/3015) | "cities" | Shared with embed; broadly relevant |
| 28/30980 | [gemma-3-4b-it/28-gemmascope-2-transcoder-262k/30980](https://www.neuronpedia.org/gemma-3-4b-it/28-gemmascope-2-transcoder-262k/30980) | "word beginnings" | Not task-relevant |
| 31/40039 | [gemma-3-4b-it/31-gemmascope-2-transcoder-262k/40039](https://www.neuronpedia.org/gemma-3-4b-it/31-gemmascope-2-transcoder-262k/40039) | "symbols" | Not task-relevant |

**Summary:** Only 2 of 4 top store features overlap with embed (10/168247, 25/3015).
The store-specific features (28/30980, 31/40039) are surface-level/positional rather
than semantically relevant. This is consistent with the poor store-direction probe
separation observed across experiments.

### Feature Overlap (Jaccard)

| Config | Feature Jaccard (embed ∩ store) |
|--------|:------------------------------:|
| capitals_states | 0.667 |
| cat_dog | 0.429 |
| oqi_indirect | 0.818 |

Despite moderate-to-high Jaccard overlap, the non-overlapping store features are
semantically weaker, explaining why the store direction produces qualitatively
different intervention behavior.

---

## Direction Probe Separation

Direction probes project a fixed set of concept-relevant tokens onto the direction
vector and measure mean-group separation. Higher separation indicates the direction
vector meaningfully distinguishes Group A (targets) from Group B (controls) in
activation space.

### Capitals_States Probes

#### Embed Direction

| Token | Projection | Group |
|:------|----------:|:-----:|
| ▁Austin | 0.6073 | A |
| ▁Sacramento | 0.5790 | A |
| ▁Olympia | 0.5868 | A |
| ▁Atlanta | 0.5665 | A |
| ▁Texas | 0.1004 | B |
| ▁California | 0.0629 | B |
| ▁Washington | 0.2209 | B |
| ▁Georgia | 0.0719 | B |

**Mean A: 0.5849 | Mean B: 0.1141 | Separation: 0.4709**

The embed direction cleanly separates capitals from states, with all
Group A projections > 0.5 and all Group B projections < 0.25.

#### Store Direction

| Token | Projection | Group |
|:------|----------:|:-----:|
| ▁Austin | −0.0399 | A |
| ▁Sacramento | −0.0249 | A |
| ▁Olympia | 0.0406 | A |
| ▁Atlanta | −0.0345 | A |
| ▁Texas | 0.0027 | B |
| ▁California | 0.0179 | B |
| ▁Washington | −0.0561 | B |
| ▁Georgia | 0.0335 | B |

**Mean A: −0.0147 | Mean B: −0.0005 | Separation: −0.0142**

The store direction has **no meaningful separation** — projections are scattered
around zero with no consistent group distinction. This explains:
- Direct projection Δ ≈ 0 for capitals_states
- Store pipeline effects come entirely from feature-mediated amplification, not direction alignment

### Cat_Dog Probes

| Metric | Embed | Store |
|--------|------:|------:|
| Separation | 0.484 | 0.102 |

Cat_dog shows modest store separation (0.102), explaining why direct projection
produces a nonzero effect (+3.00) for this concept pair but not for capitals_states.

---

## V12: Context-Enhanced vs Standard Extraction Mode

V12 introduced the **context-enhanced** store extraction mode (`STORE_LATENT_EXTRACTION_MODE:
context_enhanced`) which applies a `CONTEXT_ENHANCED_SCALE` weighting. The goal was to
test whether richer contextual weighting would improve store direction quality.

### Comparison (All Metrics at Scale 10×)

| Config | Mode | Embed Δ | Store Δ | Direct Proj Δ | Cosine Sim | Jaccard | Predictions |
|--------|:----:|--------:|--------:|--------------:|-----------:|--------:|:-----------:|
| capitals_states | standard | −7.00 | −8.38 | −0.13 | −0.035 | 0.667 | 8/8 |
| capitals_states | ctx_enhanced | −7.00 | −8.38 | −0.13 | −0.032 | 0.667 | 8/8 |
| cat_dog | standard | +3.50 | −0.94 | +3.00 | 0.073 | 0.429 | 8/8 |
| cat_dog | ctx_enhanced | +3.50 | −0.94 | +3.00 | 0.070 | 0.429 | 8/8 |
| oqi_indirect | standard | +8.69 | +8.69 | +0.00 | −0.035 | 0.818 | 8/8 |
| oqi_indirect | ctx_enhanced | +8.69 | +8.69 | +0.00 | −0.032 | 0.818 | 8/8 |

### Probe Separation Comparison

| Config | Mode | Embed Sep | Store Sep |
|--------|:----:|----------:|----------:|
| capitals_states | standard | 0.4709 | −0.0142 |
| capitals_states | ctx_enhanced | 0.4709 | −0.0142 |
| cat_dog | standard | 0.4839 | 0.1020 |
| cat_dog | ctx_enhanced | 0.4839 | 0.1014 |
| oqi_indirect | standard | 0.4709 | −0.0142 |
| oqi_indirect | ctx_enhanced | 0.4709 | −0.0142 |

### Assessment

Context-enhanced extraction produces **near-identical results** to standard extraction:
- Cosine similarity differs by ~0.003 (well within noise)
- Feature Jaccard is identical across all configs
- All gap deltas (embed, store, direct projection) are identical
- Probe separations differ by at most 0.0006 (cat_dog store: 0.1020 vs 0.1014)
- Prediction accuracy is identical (8/8 everywhere)

**Conclusion:** At the current `CONTEXT_ENHANCED_SCALE=10.0`, the context-enhanced
mode does not measurably improve store direction quality for the tested concept pairs.
The store direction weakness is a geometric property of the answer-position latent
space, not an artifact of the extraction weighting scheme.

---

## Ohio Prompt-Anchor Reruns

The Ohio notebook turned out to be more sensitive to prompt framing than to the nominal
"Ohio cities vs Indiana cities" label. The earlier failed run
`gemma3_4b_it_local_oqi_reasoning_oh_20260414_170839.ipynb` was caused by the harness
passing the high-level `explicit_embedding_difference` mode into the low-level
`it.concept_direction(...)` call. After separating `ANALYSIS.concept_direction_mode`
from `ANALYSIS.mode`, two reruns make the prompt effect clear:

| Config | Notebook | Prompt anchor | Baseline top-1 | Embed gap change | Dominant intervention motion |
|--------|----------|---------------|----------------|------------------|------------------------------|
| `gemma3_4b_it_local_oqi_reasoning_oh.yaml` | `gemma3_4b_it_local_oqi_reasoning_oh_20260414_180140.ipynb` | nearest-capital question anchored on `Michigan` / `largest city in Michigan` | `Indianapolis` (97.665%), `Columbus` only `3.28e-04` | `-8.0 -> -68.5` (`Δ = -60.5`) | `Michigan +172.5`, `▁Lansing +157.25`, `Indianapolis +85.75`, `Columbus +25.25` |
| `gemma3_4b_it_local_oqi_reasoning_oh_cleveland_anchor.yaml` | `gemma3_4b_it_local_oqi_reasoning_oh_cleveland_anchor_20260414_180925.ipynb` | direct Ohio anchor via `state containing Cleveland` | `Columbus` (100.0%), `Ohio` and `Cleveland` already outrank `Indianapolis` / `Indiana` | `29.25 -> 151.5` (`Δ = +122.25`) | `▁Columbus +177.5`, `Columbus +172.0`, `Cleveland +110.0`, `Ohio +86.0`, while `Indiana -31.69`, `▁Indiana -36.94` |

### Why the Original Ohio Notebook Looks Michigan/Indiana-Heavy

The base Ohio prompt never names Ohio. Instead, it asks the model to solve a
nearest-capital query conditioned on `largest city in Michigan`, which strongly invites
a `Michigan -> Detroit -> nearby capital` reasoning path before any intervention is
applied. The baseline logits already show that the model is living in that circuit:
`Indianapolis` dominates the answer distribution, while `▁Lansing`, `Michigan`, and
`Cleveland` all appear among the strongest tracked tokens.

The embed intervention then pushes that same circuit harder rather than recovering an
Ohio-specific one. In the rerun notebook `...180140.ipynb`, the largest key-token gains
go to `Michigan` and `▁Lansing`, not to `Columbus`. Even the explicit direct projection
step strengthens the original Indiana answer (`Indianapolis` rises from 97.665% to
99.883%) instead of moving probability mass toward Ohio.

By contrast, changing only the prompt anchor to `state containing Cleveland` immediately
flips the baseline and the intervention semantics. In `...180925.ipynb`, the baseline
answer is `Columbus`, the tracked Ohio tokens (`Ohio`, `Cleveland`) are salient before
intervention, and the embed step amplifies `Columbus/Cleveland/Ohio` while suppressing
`Indiana` terms. That is strong evidence that the earlier Michigan-heavy behavior was
primarily prompt contamination, not a failure of the `Columbus - Indianapolis` explicit
direction itself.

### Additional Observations from `...202642.ipynb`

The later Ohio rerun `gemma3_4b_it_local_oqi_reasoning_oh_20260414_202642.ipynb` makes
the embed/store split more concrete. Both construction modes were using
`paired_rejection`, but they differed in the source of the direction vector:

- The embed direction applied `paired_rejection` directly to the two token groups in
   embedding space, and its top features included the Ohio-state feature
   `(25, 27, 15708)`.
- The store direction applied `paired_rejection` to saved activations coming from the
   harness's then-current context-enhanced path, and its top features included the
   Indiana-state feature `(26, 30, 3591)` instead.

That split lines up with the qualitative behavior in the same notebook:

- The embed attribution graph does include Ohio-related features in addition to city
   features, but capital-city features are more numerous than the Ohio-state features, so
   amplification mostly returns capital-city tokens rather than a clean Ohio-only state
   identity.
- The answer-position store direction remains much less stable. Its attribution graph
   surfaces an Indiana-state feature, and the store-direction consistency probes show only
   weak separation between the two concepts. A plausible explanation is that the store
   direction is still capturing the dominant `Indianapolis` answer context from the base
   prompt. Notably, `paired_rejection` does appear to remove most of the direct `city`
   features from that store attribution, but it does not remove the Indiana-state feature.
- Direct projection demonstrates that the constructed concept direction can still steer
   strongly toward Ohio. In that step the model effectively ignores the rest of the
   original prompt and returns almost entirely Ohio-related tokens in the top-k output.

### Post-Fix Reruns (`...113258.ipynb` and `...115348.ipynb`)

After the context-enhanced extraction path was fixed so the projection happens inside
`_extract_concept_latent_state_from_cache(...)` rather than through the harness-side
`_apply_context_enhanced_projection(...)`, the paired Ohio rerun
`gemma3_4b_it_local_oqi_reasoning_oh_20260415_113258.ipynb` changed in one important
way: the embed path stayed effectively the same, but the store path no longer showed the
old Indiana-heavy behavior. The top store features changed from the earlier
`(32, 30, 110), (28, 30, 5565), (26, 30, 3591), (3, 30, 69), (0, 26, 144)` mix to
`(0, 26, 144), (0, 5, 443), (0, 4, 826), (0, 28, 66), (25, 30, 60)`, so the explicit
Indiana-state feature `(26, 30, 3591)` disappeared. However, that did not make the
store direction more semantically useful. Its direction-probe separation dropped from
`0.1395` to `0.0350`, and the store direct-projection intervention collapsed from
`Δ = +67.25` in `...202642.ipynb` to only `Δ = +8.3066`. The corrected interpretation is
that the earlier harness-side reprojection was amplifying the store direction much more
than it was clarifying it.

The new `single_group` Ohio-city rerun exposed one remaining notebook-harness bug on the
first attempt: `tests/nb_experiment_harness/pipeline_patterns.py::run_direction_probes`
still assumed group B was non-empty and raised `ZeroDivisionError` when the notebook
reached the probe cell. After fixing that helper and rerunning, the clean notebook
`gemma3_4b_it_local_oqi_reasoning_oh_single_group_20260415_115348.ipynb` completed
successfully. Its embed direction now behaves as expected for a group-A-only concept:
`Columbus`, `Cleveland`, and `Cincinnati` all score strongly positive, with
`Mean A = 0.7523`, `Mean B = n/a`, and `Separation = n/a`. The store direction also now
reports `Mean B = n/a` cleanly without crashing, which confirms the end-to-end
single-group execution path works. But the actual store signal remains extremely weak:
`Mean A = -0.0011`, direct projection only gives `Δ = +6.6836`, and the top store
features remain a mixed set `[(32, 30, 110), (27, 30, 1791), (0, 26, 144),
(28, 30, 5565), (25, 30, 60)]` rather than a clearly Ohio-specific state identity.

One smaller but useful confirmation from the same reruns is that the top-k token display
fix is visible in the notebook output itself. Tokens such as `▁Lansing` now keep their
SentencePiece marker in the rendered tables instead of being shown as the decoded plain
text form.

### Working Hypotheses

1. The Michigan prompt is probing a Midwest distance-to-capital circuit, not a clean
   Ohio-vs-Indiana city-membership circuit.
2. The explicit `Columbus - Indianapolis` direction is usable, but only when the prompt
   already points at Ohio-relevant evidence.
3. Store-direction comparisons should be deferred until the prompt-side confound is
   removed; otherwise the analysis mixes prompt retrieval failure with direction-quality
   failure.

### Recommended Follow-Ups

1. Add symmetric, within-state prompts such as `state containing Cleveland` vs
   `state containing Hammond` so the relation type is matched across Ohio and Indiana.
2. Add a state-identity concept pair (`Ohio`, `Columbus`, `Cleveland`, `Cincinnati`
   vs `Indiana`, `Indianapolis`, `Carmel`, `Bloomington`) for debugging state-membership
   behavior directly rather than nearest-capital reasoning.
3. Keep using explicit mode when debugging prompt framing, then reintroduce
   concept-pair/store-direction comparisons only after the prompt anchor is aligned.

---

## Outstanding Questions

1. **Would larger context-enhanced scales or alternative weighting schemes improve separation?**
   The current test used `CONTEXT_ENHANCED_SCALE=10.0`. More aggressive weighting
   or alternative approaches (e.g., attention-weighted extraction, multi-position
   aggregation) may yet produce better store directions.

2. **Are there concept pairs where the store direction naturally outperforms embed?**
   All tested pairs show embed dominance. Identifying a counter-example would help
   understand the geometric conditions for useful store directions.

3. **What explains the semantic quality gap in top features?**
   The embed direction selects interpretable, task-relevant features while the store
   direction selects surface-level/positional features. Understanding why requires
   analysis of how the SAE encoder maps activation-space directions to feature space.
