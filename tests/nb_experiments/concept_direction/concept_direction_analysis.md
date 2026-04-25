# Concept Direction Quality Analysis

This document tracks the quality of concept-direction vectors derived from different
extraction approaches, using Neuronpedia feature interpretations and direction-probe
separations as diagnostic tools.

> **Implementation guardrail:** The analysis notebooks and helper modules should not become an
> independent fork of the production op behavior. Keep `src/interpretune/analysis/ops/definitions.py`
> and `tests/nb_experiments/concept_direction/analysis/concept_direction_latent_dynamics.py` aligned,
> especially `concept_direction_impl` and `_paired_rejection_payload`. If the latent-dynamics module
> remains in tree, it needs rigorous parity tests so report-only changes cannot silently drift from the
> op contract.

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
first attempt: `tests/nb_experiments/pipeline_patterns.py::run_direction_probes`
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

### 2026-04-19 Gemma 3.1B Prompt-Aligned Probe-End Rerun

The latest `gemma-3-1b-it` Ohio rerun moved the context-token policy out of test-only
instrumentation and into the real store extraction path. The concept-direction harness now
computes an explicit `context_token_index` from the prompt-alignment snapshot and threads
that value through the core extraction op contract. The executed notebook
`gemma3_1b_it_local_oqi_reasoning_oh_20260419_114405.ipynb` records that policy directly in
its final summary:

- `store_context_token_source_counts = {"probe_end": 6}`
- representative cache-space mappings such as `probe_end_index=25 -> cache_answer_index=31 -> answer_index=32`
- no remaining cases where the live notebook path falls back to the old turn-boundary token

That change improved the downstream notebook parity only slightly relative to baseline
`gemma3_1b_it_local_oqi_reasoning_oh_20260417_171549.ipynb`:

| Notebook | Embed Δ | Store Δ | Cosine Sim | Jaccard | Predictions |
|----------|--------:|--------:|-----------:|--------:|:-----------:|
| `...171549.ipynb` | `-64.875` | `-28.375` | `-0.0592208` | `0.1764706` (`3/17`) | `6/6` |
| `...114405.ipynb` | `-64.875` | `-25.375` | `-0.0586897` | `0.25` (`4/16`) | `6/6` |

The direction-feature overlap improved modestly and the cosine moved only in the fourth
decimal place, but the store intervention still did not move materially closer to the
embed path on the main downstream metric. The absolute gap between embed and store deltas
actually widened slightly (`36.5 -> 39.5`). That matters because it rules out the most
obvious context-token hypothesis: using the probe/entity token instead of the turn-boundary
newline is now implemented and observable, but it is not enough to explain the remaining
store-vs-embed divergence.

The same conclusion shows up in the direct-op parity reports regenerated after the change.
For the Gemma 3.1B Ohio two-group and single-group cases, embed/store/plain/store/context
still have:

- exact top-feature ID agreement (`Jaccard = 1.0`)
- shared-score cosine ~= `1.0`
- identical `pre_gap` / `post_gap` summaries

So the remaining discrepancy is still a magnitude/calibration issue rather than a feature
ranking issue. The store direction is landing on the same salient features but with a
different latent-space scale and a meaningfully different downstream intervention effect.

### 2026-04-19 Artifact Notebook Extension, Successful Shortlist Scan, and Bat Follow-Up

The parity-analysis notebook now renders two views that were previously easy to miss:

1. preserved direct-op direction geometry, including raw norms, latent-row norms, example
   weights, paired-rejection residuals, and SHA fingerprints
2. notebook-debug pipeline artifacts that do not have a paired direct-op parity report

The refreshed direct-op rerun sharpens the diagnosis further. The preserved graph-stage
summary is still perfectly aligned for both Gemma 3.1B concept-direction test variants, but
the upstream direction geometry is not:

| Artifact view | Embed raw norm | Store raw norm(s) | Direction cosine(s) | Downstream overlap summary |
|---------------|---------------:|------------------:|--------------------:|----------------------------|
| preserved direct-op `gemma3_1b_it_two_group` | `0.592790` | `53.009087` (`store_plain`), `188.381836` (`store_context`) | `-0.0381` / `0.0706` / `0.0192` | graph-stage Jaccard `1.0`, shared-score cosine `1.0`, identical gap delta `-47.308847` |
| preserved direct-op `gemma3_1b_it_single_group` | `0.645849` | `125.866814` (`store_plain`), `330.705688` (`store_context`) | `0.0126` / `0.0593` / `0.3914` | graph-stage Jaccard `1.0`, shared-score cosine `1.0`, identical gap delta `-47.308847` |
| notebook-debug `gemma3_1b_it_ohio_notebook_debug` | `0.592790` | `170.347778` (`store_direction`) | `0.076709` | feature Jaccard `0.333333`, shared-score cosine `0.858457`, embed/store gap deltas `-64.875` vs `-52.875` |
| notebook-debug `gemma3_1b_it_bat_notebook_debug` | `0.528009` | `209.447205` (`store_direction`) | `0.067216` | feature Jaccard `0.666667`, shared-score cosine `0.990623`, embed/store gap deltas `+57.9375` vs `+19.5625` |

So the preserved direct-op graph summary is still best understood as a masking layer: it can
stay perfectly aligned even while the pre-graph directions differ substantially in norm and
remain close to orthogonal. That is exactly why the Ohio notebook continues to look
materially different once the full pipeline is exercised.

The shortlist workflow also moved forward in the same session. The local inference service
was restored, the request auth mismatch was fixed by sending `X-SECRET-KEY` alongside the
bearer token, and both inference-side scan scripts completed successfully for the filtered
`orange`, `hammer`, and `bat` shortlist.

The saved JSON outputs showed an important distinction:

1. the shortlist classification prompts are all dominated by the same generic top features
   (`14/.../457`, `11/.../12`, `13/.../256`, `8/.../91`, `13/.../613`)
2. the saved scan rows currently report `has_local_explanation = false` for those features
3. the hypothesis-query scans, not the classification scans, provide the discriminative signal

Combining the saved scan JSON with the prompt-answer harness results gives the current
candidate ranking:

| Candidate | Top-1 / Top-2 answer | Hypothesis-query readout | Takeaway |
|-----------|----------------------|--------------------------|----------|
| `orange` (color vs fruit) | `color` `0.9884` / `fruit` `0.0075` | still overlaps the generic shortlist-heavy feature set | usable ambiguity candidate, but still fairly lexical/generic |
| `hammer` (tool vs weapon) | `tool` `0.9983` / `Tool` `0.0005` | shows some distinctive object-use features | too clean for the next calibration pass |
| `bat` (bird vs mammal) | `mammal` `0.7908` / `bird` `0.1999` | exposes the richest hypothesis-only set, including `22/.../10136`, `23/.../7092`, `25/.../413`, `25/.../937`, `7/.../377`, `7/.../468` | best next follow-up |

That led to a dedicated `bat` concept-pair config and notebook config, followed by a fresh
Ohio rerun (`gemma3_1b_it_local_oqi_reasoning_oh_20260419_193722.ipynb`), a new `bat`
notebook run (`gemma3_1b_it_local_bird_mammal_bat_20260419_193941.ipynb`), and a refreshed
parity-analysis notebook execution. The Ohio notebook remains anti-target for both embed and
store, but the `bat` notebook is the first shortlist follow-up where both paths steer the
intended class gap in the positive direction, with substantially higher feature overlap than
the Ohio run even though the underlying direction cosine is still low.

#### Upstream Reference Graph and Random Perturbation Controls

The refreshed 1B artifact set now includes two new diagnostics that were missing from the
earlier pass:

1. an upstream-only reference graph sanity report for the `bat` surface
2. a large random-vector perturbation control in the notebook-debug artifacts for both `bat`
   and Ohio

The direct-op `bat` reference report confirms that upstream graph construction is more stable
than the raw direction cosines would suggest, but not stable enough to erase the embed/store
split:

| Bat reference comparison | Jaccard | Shared-score cosine | Direction cosine | Shared top features |
|--------------------------|--------:|--------------------:|-----------------:|---------------------|
| embed vs `store_plain` | `0.428571` | `0.962254` | `-0.033594` | 6 |
| embed vs `store_context` | `0.666667` | `0.989558` | `0.065248` | 8 |
| `store_plain` vs `store_context` | `0.538462` | `0.921478` | `0.173271` | 7 |
| embed vs embed-random-perturbed | `0.428571` | `0.996317` | `0.099504` | 6 |
| `store_context` vs store-random-perturbed | `0.538462` | `0.943407` | `0.099504` | 7 |

The shared `bat` reference rows still center on a small common core:
`[0, 28, 276]`, `[0, 26, 363]`, `[0, 24, 441]`, `[0, 25, 276]`, and `[0, 27, 19]`. The main
top-5 difference is that embed keeps `[0, 4, 5392]` while the store-context path instead lifts
`[0, 28, 420]` into the shared frontier.

The notebook perturbation controls reinforce the same story. Even after forcing the perturbed
direction cosine down to `0.099504`, the graph selections remain partly stable:

| Notebook artifact | Base embed/store Jaccard | Embed perturb Jaccard | Store perturb Jaccard | Embed `Δgap_vs_base` | Store `Δgap_vs_base` |
|-------------------|-------------------------:|----------------------:|----------------------:|---------------------:|---------------------:|
| `gemma3_1b_it_bat_notebook_debug` | `0.666667` | `0.428571` | `0.538462` | `-47.8125` | `+2.4375` |
| `gemma3_1b_it_ohio_notebook_debug` | `0.333333` | `0.333333` | `0.333333` | `+17.5` | `-1.0` |

Interpretation:

1. The graph layer is materially more stable than the raw embed/store direction cosine, which
   helps explain why the direct-op graph summaries can still look healthier than the full
   notebook pipeline.
2. That stability is surface-dependent. `bat` keeps much more shared graph structure than Ohio,
   but the positive `bat` store path still moves the target gap far less than embed.
3. The perturbation control now rules out a simplistic "completely different direction means
   completely different graph" story. The next debugging step should focus on scale and weight
   calibration rather than only on feature-ID overlap.

### 2026-04-20 Ohio Direct-Op Fix and Structured Feature-Selection Reruns

The top-priority Ohio direct-op failure turned out to be a reference-side mismatch rather than a
remaining embed-vs-store graph-construction bug. After aligning the Ohio direct-op reference path
with the corrected target configuration and ranking behavior, the preserved Gemma 3.1B Ohio parity
tests now validate direct-op vs reference agreement for both the two-group and single-group cases.
The large random-vector perturbation control is also no longer inert: the preserved reports still
show the expected `~0.099504` direction cosine to the base path, but the perturbed graph selections
and post-gap summaries are now distinct from the unperturbed ones instead of being byte-for-byte
identical.

That matters because it changes how to interpret the remaining Ohio mismatch. The failing test was
not telling us that the embed/store paths were secretly identical and the reference helper was fine.
It was telling us that the test had been over-asserting cross-path equality after the reference path
was fixed. The corrected Ohio direct-op parity gate now treats exact direct-op vs reference parity
as the hard invariant for each path, while only requiring partial overlap and consistent gap
reduction across embed/store variants.

The same session also landed the shared constrained-feature-selection refactor in the notebook
harness. `constrained_feature_selection` now supports structured `FeatureSelectionSpec`-style
payloads via `specific_features`, `layer_slices`, and `position_slices`, and slice bounds now use
numeric value semantics so a config entry like `layer_slices: [[10, null]]` really means "layers
greater than or equal to 10". Two new notebook configs exercised that support directly:

- `gemma3_1b_it_local_oqi_reasoning_oh_fs_l10_n5.yaml`
- `gemma3_1b_it_local_color_fruit_orange_fs_l10_n5.yaml`

Both reruns applied the intended constraint shape correctly: all extracted rows came from layers
`>= 10`, and the harness returned only the top 5 constrained features per path. The important part
is what happened next: the high-layer filter did **not** restore embed/store parity. It made the
remaining divergence easier to localize.

| Surface | Config | Feature Jaccard | Shared features | Embed `Δgap` | Store `Δgap` | Embed layers | Store layers |
|---------|--------|----------------:|----------------:|-------------:|-------------:|--------------|--------------|
| Ohio | baseline | `0.333333` | `5/15` | `-64.875` | `-52.875` | `0,16,17,18,19,20,25` | `0,16,17,25` |
| Ohio | `fs_l10_n5` | `0.111111` | `1/9` | `-46.875` | `-54.875` | `17,18,19,20,25` | `16,17,24,25` |
| Orange | baseline | `0.333333` | `5/15` | `+5.5` | `+3.0` | `0,1,16,20,25` | `0,3,5` |
| Orange | `fs_l10_n5` | `0.0` | `0/10` | `+15.625` | `+1.625` | `16,20,25` | `12,16,17,25` |

Three conclusions follow from those reruns:

1. The structured constrained-feature-selection path is working as intended. The notebook harness,
   serializer, and backend filter all honor the new layer-slice specification.
2. Removing the obvious layer-0 rows does not collapse embed/store divergence. On both Ohio and
   orange, overlap becomes sparser rather than denser once the selection is restricted to later
   layers.
3. The remaining notebook drift is therefore not just a low-layer lexical-feature problem. Even in
   the high-layer-only slice, the store path lifts a different later-layer frontier than embed.

That puts the next debugging step in a narrower box. The open issue is no longer "why does the
reference helper disagree with direct-op?" and it is no longer "does the notebook parser really
apply a layer slice?" The remaining question is why the later-layer store direction keeps surfacing
different rows and materially different intervention deltas even when the experiment is forced onto
the same high-layer band as embed.

### Refreshed Analysis Notebook

The full rerun of `tests/nb_experiments/concept_direction/analysis/concept_direction_analysis.ipynb` makes the new
boundary clearer. Path-vs-reference parity is now exact on the refreshed Ohio and orange artifacts,
including the new `fs_l10_n5` surfaces, so the remaining mismatch is squarely about cross-path
geometry rather than report-generation drift.

For the direct-op artifacts, the strongest recurring pattern is high shared-score cosine paired with
near-orthogonal normalized directions and extreme raw-scale inflation:

- Ohio two-group: embed/store-plain/store-context shared-score cosine stays high
   (`0.971491` / `0.861413`), but the corresponding normalized direction cosines remain tiny
   (`-0.038099`, `0.070584`, `0.019183`) while raw norms jump from `0.592790` to `53.009087`
   and then `188.381836`.
- Ohio `fs_l10_n5`: the high-layer filter still leaves only `3` shared embed/store-plain rows and
   `1` shared embed/store-context row, even though the surviving rows remain numerically aligned
   (`0.991121` and `1.000000` shared-score cosine).
- Orange baseline: embed/store-plain/store-context shared-score cosine stays similarly high
   (`0.985395` / `0.980590`), but normalized direction cosines are still only `0.009717`,
   `0.043186`, and `0.140010`, while raw norms jump from `0.592765` to `60.012154` and then
   `259.368958`.
- Orange `fs_l10_n5`: the high-layer slice is the starkest case. Embed and store-plain keep only
   `1` shared row, embed and store-context keep `0`, and yet path-vs-reference parity still holds.

The notebook-debug artifacts remain consistent with that direct-op picture instead of contradicting
it. Ohio stays low-cosine and anti-target (`0.076709`, Jaccard `0.333333`, gap deltas `-64.875`
vs `-52.875`), while orange stays low-cosine but target-consistent (`0.066321`, Jaccard
`0.333333`, gap deltas `+5.5` vs `+3.0`). The large random-vector control is also active on the
refreshed surfaces rather than inert: the parity notebook now shows the expected `~0.099504`
direction cosine together with changed feature sets for both direct-op and notebook-debug artifacts.

### Orange `fs_l10_n5` Latent Dynamics

The separate latent notebook `tests/nb_experiments/concept_direction/analysis/concept_direction_latent_dynamics_analysis.ipynb`
also now runs cleanly through papermill with a real top `parameters` cell, clickable Neuronpedia
feature links, and best-available explanation text in the feature tables. The old UMAP
spectral-initialization fallback is gone; the only remaining UMAP warning on this surface is the
expected `n_jobs` override that comes from pinning `random_state`.

- For `context_enhanced_scale=10.0`, the mean Color norms rise from `138.01` answer-state /
   `103.42` context-state to `357.81` projected/selected-store-state, while the mean Fruit norms
   rise from `148.82` / `110.42` to `417.85`. `selected_store_state_source` is
   `projected_context_state` for every example and `projected_selected_state_delta_norm` stays
   `0.0`, so the selected store state is literally the projected context state on this surface,
   not a later fallback.
- The code path makes that equality explicit. In
   `tests/nb_experiments/concept_direction/analysis/concept_direction_analysis.py`,
   `projected_states` is computed as the scaled projection onto `context_states`, and
   `final_latent_states` only diverges from it on the `answer_state_fallback` branch when the
   context index is invalid. On the orange notebook-debug artifact every snapshot uses
   `context_token_indices`, so there is no such fallback branch in play.
- The slight `projected_context_state` vs `selected_store_state` separation that can appear in the
   latent-dynamics UMAP should therefore be read as a visualization artifact, not as a genuine
   latent mismatch. `tests/nb_experiments/concept_direction/analysis/concept_direction_latent_dynamics.py`
   appends both stage labels as separate rows before projection, so the plot renders two labeled
   stages even when the underlying vectors are identical.
- The new signed `expected_answer_logit_margin` view resolves the earlier orange `logit_diff`
   confusion. All four Fruit prompts remain strongly positive, and red/green are positive once the
   fixed `Fruit - Color` gap is flipped into the expected-answer frame, but blue and yellow remain
   anti-target (`-14.125` and `-0.5`). The score path is therefore not globally inverted; two of
   the four Color exemplars are still genuinely misaligned.
- The precise hash collapse is now between
   `store_context_enhanced_paired_rejection_direction` and
   `store_context_enhanced_paired_rejection_reconstruction_direction`, which both resolve to
   `bc11607d`. `store_answer_position_paired_rejection_direction` remains distinct at `ec185c69`,
   so the context-enhanced path is still replaying the same paired-rejection residual rather than
   rotating onto the answer-position store axis.
- The later-layer context-enhanced frontier remains
   `[(13,27,10777), (11,27,688), (17,27,11442), (12,25,3), (16,25,400)]`. That matches the
   refreshed direct-op parity notebook, where orange `fs_l10_n5` still reports embed/store-plain
   Jaccard `0.111111`, embed/store-context Jaccard `0.0`, store-plain/store-context Jaccard
   `0.428571`, shared-score cosine `1.0`, and raw norms `0.592765 -> 60.012154 -> 259.368958`.
- A supplemental `context_enhanced_scale=1.0` rerun did not restore store-plain parity. It kept
   the same context-enhanced frontier, changed the normalized hash from `bc11607d` to `38ebd49a`,
   and shrank the raw context-enhanced direction norm from `252.238266` to `25.223827`. The exact
   comparison script shows the same `3/7` shared top features with the answer-position store
   frontier at both scales (Jaccard `0.428571`), the same context-vs-answer cosine
   (`0.143060848`), and an effective context-vs-context cosine of `1.0` between the scale-`10.0`
   and scale-`1.0` directions. The per-example projected/selected-store-state norms contracted by
   the same factor (`357.81 -> 35.78` for Color, `417.85 -> 41.79` for Fruit). The most defensible
   reading is that lowering the scale mostly contracts magnitude; it does not rotate the context-
   enhanced store direction onto the answer-position store axis.
- That also answers the narrower parity question between `projected_states` and
   `final_latent_states`: there is no residual parity gap left to restore between those two on the
   orange surface, because they already coincide exactly. The remaining non-parity is between the
   context-enhanced store direction and the answer-position / reference-store directions. Changing
   the scale from `10.0` to `1.0` contracts the same later-layer replay, but it does not rotate
   that replay onto the answer-position store axis.
- A follow-up orange-only notebook-triad check resolved the apparent `embed` vs
   `embed_reference` mismatch on `fs_l10_n5`. On the current preserved `/tmp` artifacts,
   `gemma3_1b_it_orange_fs_l10_n5_notebook_debug.embed_pipeline.top_features`,
   `gemma3_1b_it_orange_fs_l10_n5.embed.top_feature_ids`, and
   `gemma3_1b_it_orange_fs_l10_n5_reference_graph_sanity.embed.top_feature_ids` are identical
   (top rows begin `(25,27,1316), (25,27,765), (16,27,155), (17,26,481), (23,27,725)`).
- The earlier notebook-triad discrepancy was therefore a reference-resolution state issue, not an
   orange embed-path regression. The analysis notebook previously fell back to broad surface-based
   matching for `*_notebook_debug` artifacts; it now first tries the exact derived
   `*_reference_graph_sanity` artifact name, which deterministically selects
   `gemma3_1b_it_orange_fs_l10_n5_reference_graph_sanity` when that preserved reference artifact is
   present.

### 2026-04-21 Interpretation Update

#### Scale vs directional shift

The strongest recurring pattern is no longer "shared features vs different features". It is
"similar high-salience frontier, different normalized geometry, very different raw scale". Across
Ohio and orange, embed/store shared-score cosine can stay high while normalized direction cosine
stays near zero and store raw norms inflate by two to three orders of magnitude. That means the
remaining drift is better explained as a calibration problem in how the store rows are weighted and
aggregated than as evidence that the store path is discovering a wholly unrelated concept.

#### Store plain vs context-enhanced precision

`store_plain` should now be treated as the cleaner diagnostic baseline. It shows the answer-position
signal before the extra context-enhanced reprojection amplifies it. `store_context` is still useful,
but mostly as a stress test for whether contextual weighting adds semantic precision. On the current
Ohio and orange reruns it usually adds magnitude much more reliably than it adds alignment. The key
question is therefore not whether context-enhanced extraction makes the store vector bigger, but
whether it makes the downstream motion more embed-like. On the current surfaces, it generally does
not. The orange scale-`1.0` follow-up sharpens that point: reducing the scale contracts the raw and
projected norms by about `10x`, but it leaves the later-layer frontier intact, preserves the same
`0.143060848` context-vs-answer cosine, and still does not recover answer-position parity.

#### Paired-Residual Replay and Latent-Dynamics Interpretation

The orange `fs_l10_n5` latent notebook makes that precision issue explicit. The normalized hash
collapse is between `store_context_enhanced_paired_rejection_direction` and its explicit
reconstruction stage, not between the context-enhanced path and the answer-position store path.
Combined with projected and selected store states landing on the same coordinates for every example,
the projection step is best understood as injecting more magnitude into an already-selected later-
layer residual, not rotating the state onto the embed direction.

#### How to read Ohio shared-score cosine

The Ohio reruns are the clearest warning against over-reading shared-score cosine. A value near
`1.0` says the overlapping rows are scored similarly once they survive selection. It does **not**
say the full normalized directions agree, and it does **not** guarantee similar intervention
behavior. Ohio still shows high shared-score cosine alongside low normalized cosine, large store
norm inflation, and materially different gap motion. The most defensible reading is that both paths
still touch the same broad Midwest/capital circuit while weighting that circuit very differently.

#### Parity target update

The parity target should no longer be "maximize embed/store feature overlap". The refreshed random
perturbation controls show that partially stable graph selections can survive even when the control
direction is forced down to the low-cosine regime, so feature overlap by itself is too weak a goal.
The stronger target is:

1. Pivot our focus away from maximizing embed versus store_context surviving feature set alignment to instead focus on quantifying and documenting the differing behavior and latent space dynamics underlying it as well as the differing contextual utility of each approach.
2. Understand and exploit cross-path differences in normalized geometry and raw-scale inflation.
3. Prefer surfaces where downstream gap motion stays aligned after those geometry checks, rather than
   treating high Jaccard or shared-score cosine as success on their own.

### Working Hypotheses

1. The Ohio prompt is still probing a Midwest distance-to-capital circuit, not a clean
   Ohio-vs-Indiana city-membership circuit.
2. The remaining store-direction divergence is not primarily a prompt-alignment bug; the refreshed
    direct-op parity notebook and the orange latent pass both point to a scale/calibration issue in
    how stored latent rows are weighted, normalized, or aggregated into the final concept direction.
   On the orange `fs_l10_n5` surface, context-enhanced extraction still replays the same later-
   layer paired-rejection frontier, and lowering the scale from `10.0` to `1.0` mostly contracts
   that replay instead of moving it onto the answer-position store axis.
3. Ambiguous taxonomic prompts like `bat` appear to be a better calibration surface than
   the Ohio prompt because they preserve more shared top features while still exposing the
   low-cosine embed/store geometry underneath.
4. High-layer-only feature constraints do not restore parity, which argues that the residual
   drift is not just caused by generic layer-0 / prompt-formatting rows dominating the top-k
   frontier.
5. The same broader-embed / narrower-store-context split now appears on both Ohio and orange,
    so the residual drift is not specific to the Ohio prompt semantics.

### Recommended Follow-Ups

1. Use `gemma3_1b_it_local_bird_mammal_bat.yaml` as the current notebook-default calibration
   surface for the next embed-vs-store pass rather than spending another iteration on Ohio
   prompt wording first.
2. Compare pre-normalization store latent-state norms, per-example weights, and paired-
   rejection residuals across Ohio, orange, and `bat`, and explicitly include the orange
   `context_enhanced_scale=1.0` versus `10.0` comparison before changing the extraction contract.
3. Use the new large random-vector perturbation control to compare graph stability against the
   pre-graph norm/weight summaries; the control is now implemented in both the direct-op and
   notebook artifact paths.
4. Revisit explicit-marker or state-membership Ohio prompts only after the scale/calibration
   investigation establishes why the current store directions remain low-cosine despite the
   much stronger top-feature overlap now visible in the `bat` follow-up.
5. Add a narrowed `specific_features` follow-up on the new `fs_l10_n5` surfaces using the shared
   later-layer survivors directly, rather than only comparing full unconstrained frontiers.
6. Inspect the two orange Color exemplars that remain anti-target in the signed-margin view
   (`blue` and `yellow`) before treating orange as a solved calibration surface.

### 2026-04-24 Ohio 4B parity closeout and fresh full regeneration

The remaining Ohio 4B direct-op blocker turned out to be a test expectation bug rather than a
new backend or graph-construction regression. The preserved Ohio path defines the monitored gap
as `Columbus - Indianapolis`, so a successful Ohio intervention should increase that gap. The
failing parity assertion was still checking the opposite inequality. After correcting that
semantic expectation, the focused Ohio slice is green again:

- `test_analysis_backend_parity_gemma3_1b_it_concept_direction_paths[two_group]` passed in
  `1107.29s`
- `test_analysis_backend_parity_gemma3_1b_it_extended_reference_graph_sanity[ohio_fs_l10_n5]`
  passed in `557.09s`
- the narrowed 3-test Ohio subset passed in `2226.97s`

That matters because it re-establishes the intended invariant for the preserved 4B Ohio artifacts:
the embed path is allowed to be strongly pro-Ohio when the gap is computed as
`Columbus - Indianapolis`, and the test gate now matches the actual stored metric instead of
implicitly assuming an `Indianapolis - Columbus` interpretation.

The full fresh launcher run also completed cleanly from a new artifact generation rooted at
`/tmp/it_concept_direction_analysis_artifacts`.

1. The launcher archived the previous root to
   `/tmp/it_concept_direction_analysis_artifacts_20260424_203259`.
2. The full 5-test reference/parity gate passed before notebook execution.
3. All four requested experiment notebooks completed successfully:
   `gemma3_1b_it_local_color_fruit_orange_20260424_211426.ipynb`,
   `gemma3_1b_it_local_color_fruit_orange_155_4973_20260424_212158.ipynb`,
   `gemma3_1b_it_local_color_fruit_orange_fs_l10_n5_20260424_212931.ipynb`, and
   `gemma3_4b_it_local_oqi_reasoning_oh_2975_15708_20260424_213707.ipynb`.
4. The final analysis notebook execution also finished successfully and wrote
   `/tmp/it_concept_direction_experiments/analysis/concept_direction_analysis_20260424_203259.ipynb`.
5. No notebook-level `Traceback`, `CellExecutionError`, or `output_type = "error"` entries were
   present in either the final Ohio notebook or the generated analysis notebook.

The refreshed artifact root now includes both the parity/reference views and the notebook-debug
views needed for the next calibration pass:

- `gemma3_4b_it_two_group/`
- `gemma3_4b_it_ohio_reference_graph_sanity_two_group/`
- `gemma3_4b_it_ohio_fs_2975_15708_reference_graph_sanity/`
- `gemma3_4b_it_ohio_fs_2975_15708_notebook_debug/`
- the matching orange reference/notebook-debug surfaces

So the current state is materially better than the earlier Ohio handoff: the direct-op Ohio 4B
slice is repaired, the fresh notebook/analysis generation path has been revalidated end to end,
and the latest generated analysis notebook can now be used as the authoritative comparison surface
for the preserved `/tmp` artifacts from this cycle.

---

## Outstanding Questions

1. **Would larger context-enhanced scales or alternative weighting schemes improve separation?**
   The current reruns now bracket both `CONTEXT_ENHANCED_SCALE=10.0` and `1.0`. The smaller scale
   mostly contracts magnitude without restoring answer-position parity, but it is still unclear
   whether an intermediate scale, adaptive weighting, or a different aggregation rule would improve
   separation.

2. **Are there concept pairs where the store direction naturally outperforms embed?**
   All tested pairs show embed dominance. Identifying a counter-example would help
   understand the geometric conditions for useful store directions.

3. **What explains the semantic quality gap in top features?**
   The embed direction selects interpretable, task-relevant features while the store
   direction selects surface-level/positional features. Understanding why requires
   analysis of how the SAE encoder maps activation-space directions to feature space.
