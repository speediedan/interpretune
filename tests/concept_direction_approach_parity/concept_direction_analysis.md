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
