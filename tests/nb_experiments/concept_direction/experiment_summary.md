# Concept Direction Experiment — Synthesized Summary

This document distils everything learned across twelve experiment waves (V1–V12) into a
single living reference (per-wave summaries were consolidated into this document,
2026-04-25). Start here for current understanding.

---

## 1. Synthesized Understanding

### What the experiments test

Two methods for computing a **concept direction** vector in the circuit-tracer (CT)
semantic intervention pipeline, plus diagnostic variants:

| Approach | Source | Forward pass? | Space | Wave |
|----------|--------|---------------|-------|------|
| **Embed-based** (a) | Raw token embedding/unembedding weight vectors | No | d_model | V1+ |
| **Store-based** (b) | `extract_concept_latent_state` / `extract_concept_latent_examples` op pipeline — caches `unembed.hook_in` activations at the answer position, then computes direction via `paired_rejection` | Yes | d_model | V1+ |
| **Direct projection** (c) | Same store direction, applied directly to residual stream at `unembed.hook_in` — bypasses CT feature selection | — (uses cached direction) | d_model | V11 |
| **Context-enhanced store** (d,e) | Context-enhanced extraction averages activations across all positions weighted by attention to the answer token, then mean-pool across prompts. Uses `STORE_LATENT_EXTRACTION_MODE: context_enhanced` with configurable `CONTEXT_ENHANCED_SCALE`. Produces variants (d) = feature pipeline, (e) = direct projection. | Yes | d_model | V12 |

The embed-based direction is validated in
`tests/core/test_analysis_backend_parity.py` and matches the upstream circuit-tracer
`demo_utils.get_unembed_vecs()` path. The goal is to make the store-based direction
produce interventions that are at least as semantically useful as embed, since the
store-based path generalises to concepts that lack a clean single-token embedding.

### What we know

1. **Embed and store select materially different feature subspaces.**
   Cosine similarity between the two normalised directions is consistently near zero
   (range 0.02–0.24, with V8 op-driven capitals_states values all < 0.08). Jaccard
   overlap on selected top-N features is low to moderate (0.11–0.82). No single config
   shows complete convergence; the V3–V6 gemma3_it dog_cat Jaccard=1.0 was an artifact
   of invalid key_labels (see §1 item 9). Corrected V9 value is 0.333.

2. **For PT models, embed tends to reinforce the target gap; store diverges by concept pair.**
   In capitals_states, embed reinforces Austin-over-Dallas (embed Δ = +2.97, +3.25) while
   store erodes it (store Δ = −0.65, −0.88 in V8). In dog_cat (V9), the pattern is more
   complex: gemma-2-2b PT shows both directions negative (embed Δ = −4.34, store Δ =
   −6.34); gemma-3-1b PT and gemma-3-4b PT show embed negative but store strongly positive
   (+11.25, +4.06). PT models consistently score 0/8 on classification prompts, so the
   store direction is derived from weak-supervision states regardless of concept pair.

3. **For IT models, both directions tend to steer anti-target or near-zero.**
   gemma-2-2b-it capitals_states (V5) shows mild positive embed Δ (+0.14) but near-zero
   store Δ (+0.05) under `apply_chat_template`. gemma-3-4b-it capitals_states (V8) is the
   starkest case: embed Δ = −7.56, store Δ = −15.13, yet prompt-level predictions are 8/8
   correct. V9 dog_cat IT results are milder: gemma-2-2b-it (embed Δ = −0.21, store Δ =
   −0.13), gemma-3-1b-it (embed Δ = −2.06, store Δ = −2.81), gemma-3-4b-it (embed Δ =
   −1.25, store Δ = −0.63). Correct classification is not sufficient for a useful
   intervention direction.

4. **CT feature selection is sign-agnostic (`abs_()`).**
   `compute_partial_influences` (circuit_tracer/graph.py:390) uses absolute influence
   values, so features that oppose the concept direction are selected equally to those
   that promote it. Direction-negation tests are inherently inconclusive under this
   design. This is the single most important architectural constraint on interpreting
   all experimental results.

5. **Scale factor 10 operates outside the model's natural regime.**
   At scale_factor=10, interventions produce 10–20 logit shifts. Lower scales (2.0)
   show more selective effects on gemma-2-2b PT but the scale-sweep phase has not yet
   been systematically aggregated across configs.

6. **Prompt rendering matters but does not rescue store.**
   `apply_chat_template` and `gemma_dataclass` produce identical tokenisation for Gemma 3 IT.
   For Gemma 2 IT, switching render mode changes intervention magnitude (embed Δ from +0.11
   to +1.09) but does not fix the fundamental store-direction problem.

7. **Gemma 3 4B works via Gemma3ForConditionalGeneration.**
   After hook-mapping and multimodal-backbone NNsight fixes, both 4B PT and 4B IT run
   end-to-end. Treat this as targeted text-only support, not blanket VLM compatibility.

8. **V9 dog_cat results are now valid and confirm model-size/family effects.**
   With corrected key_labels and chat_intervention_prompt, the dog_cat concept pair shows:
   (a) the same IT 8/8 vs PT 0/8 dichotomy as capitals_states; (b) store gap deltas vary
   wildly by model family (gemma-2-2b PT: −6.34, gemma-3-1b PT: +11.25, gemma-3-4b PT:
   +4.06); (c) feature Jaccard for 4B PT dog_cat is notably high (0.818) compared to
   capitals_states (0.667), suggesting the simpler concept may use more overlapping features.

9. **V9 262k transcoder variants show minimal difference from 16k for the same model.**
   Cosine similarity and Jaccard are identical between 16k and 262k for the same
   model+concept (these depend on model internals, not transcoder width). Gap deltas
   differ slightly: 262k capitals_states PT embed Δ = +3.13 vs 16k = +3.25; store Δ =
   −0.75 vs −0.88. The wider transcoder captures similar concept structure with marginal
   quantitative differences. The ~93 GB VMPeak is virtual address space from safetensors
   mmap, not physical memory (VmHWM=12.7 GB, VmRSS=6.1 GB). The ~51 min runtime is
   dominated by per-layer lazy W_enc/W_dec disk reads during
   `compute_attribution_components()` (34 layers × 1.34 GB each via safetensors).

10. **Neuronpedia transcoder dashboards have partial layer coverage for some variants.**
    Verified via neuronpedia.org/available-resources: gemma-3-4b-it has both
    `gemmascope-2-transcoder-16k` (first 12 layers only) and `gemmascope-2-transcoder-262k`
    (layers 0-33, full coverage). No transcoder sets exist for gemma-3-1b-it (only a few
    layers of res-16k). gemma-2-2b has `gemmascope-transcoder-16k` (all layers, loaded).
    The 16k layer gap for gemma-3-4b-it means feature dashboard inspection is limited to
    layers 0-11 unless 262k dashboards are used.

11. **V10 cat_dog direction reversal is approximately symmetric for PT models.**
    gemma-2-2b PT shows the cleanest symmetry: dog_cat embed Δ = −4.34 vs cat_dog
    embed Δ = +4.26; store Δ = −6.34 vs +5.06. gemma-3-4b-pt store is perfectly
    symmetric (±4.06) but embed is not (+2.25 vs −0.13). IT models show small,
    near-zero Δ in both directions. The 8/8 IT vs 0/8 PT classification dichotomy
    holds for cat_dog as it did for dog_cat.

12. **OQ-I: IT amplification saturation is prompt-specific, not model-inherent.**
    gemma-3-4b-it's standard capitals_states shows extreme anti-target saturation
    (embed Δ = −7.56, store Δ = −15.13). The indirect JFK prompt **reverses the sign**
    to strongly positive (embed Δ = +4.63, store Δ = +9.44). The reasoning-chain prompt
    reduces magnitude to near-zero (embed Δ = −0.38, store Δ = −1.69). gemma-2-2b-it
    shows minimal sensitivity across all three prompt variants (all Δ within ±0.38).

13. **OQ-I: different prompts engage different feature subspaces (Jaccard drops).**
    Feature Jaccard for the indirect prompt drops sharply (gemma-3-4b-it: 0.333 → 0.111)
    while cosine similarity between embed/store direction vectors remains unchanged
    (−0.035 for all three prompt variants). The direction vectors themselves are stable;
    it is the feature selection that changes with prompt framing.

14. **V11: Direct projection reveals that pipeline effects are mostly feature-mediated.**
    Applying the store direction vector directly to the residual stream at `unembed.hook_in`
    (bypassing circuit-tracer feature selection) produces meaningful deltas only when the
    direction has intrinsic geometric alignment. For cat_dog (store probe separation 0.102),
    direct projection Δ = +3.00 — comparable to embed pipeline Δ = +3.50. For
    capitals_states (store probe separation −0.014), direct projection Δ ≈ 0, yet the
    feature-mediated pipeline Δ = −7 to −15. The pipeline's power comes from feature
    selection and amplification, not from the direction vector's raw alignment.

15. **V11: Feature mediation can invert the direction signal.**
    Cat_dog store direct projection is +3.00 (pro-target), but the store pipeline is
    −0.94 (anti-target). The feature selection mechanism selects features whose mixed-sign
    activations, when amplified, produce a net effect opposite to the underlying direction.

16. **V11: OQ-I indirect embed/store convergence confirmed with higher Jaccard.**
    V11 oqi_indirect shows identical embed and store deltas (+8.69) with Jaccard = 0.818
    (highest in the V11 wave). V10 had already shown the indirect prompt reverses the
    sign; V11 confirms it also drives embed/store feature selection into near-convergence.

17. **V12: Context-enhanced extraction is numerically equivalent to standard answer-position extraction.**
    Context-enhanced mode (`STORE_LATENT_EXTRACTION_MODE: context_enhanced`,
    `CONTEXT_ENHANCED_SCALE: 10.0`) averages activations across all positions weighted by
    attention to the answer token, then mean-pools across prompts. Across all three configs
    (cat_dog, capitals_states, oqi_indirect), context-enhanced variants produce cosine
    similarity, Jaccard, probe separation, and gap deltas that differ from standard by
    ≤ 0.003 in cosine and < 0.001 in probe separation. This means the answer-position
    activation already dominates the attention-weighted average, so enriching with context
    does not change the resulting direction geometry.

18. **V12: The store direction's weakness is geometric, not an extraction-mode artifact.**
    Context-enhanced extraction was designed to test OQ-B by enriching the store direction
    with contextual information beyond the answer position. Its equivalence with standard
    extraction (finding 17) confirms that the store direction's low probe separation
    (capitals_states: −0.014) and near-zero direct projection Δ are properties of the
    underlying concept geometry, not of how activations are extracted. The store direction
    captures something about the model's representation, but that something is not aligned
    with the target token distinction.

19. **V12: Five-approach framework validates complementarity of embed and store methods.**
    Running all five approaches (a–e) for three configs confirms: embed-based directions
    have strong geometric alignment (probe separation 0.47–0.48, direct projection Δ up to
    +3.00 for cat_dog) while store-based directions capture feature-mediated effects that
    can diverge from or even invert the raw direction signal. The cat_dog concept pair
    shows the most informative store direction (probe separation 0.10), while
    capitals_states and oqi_indirect show near-zero store probe separation despite large
    feature-mediated pipeline deltas. This suggests that the store pipeline's value is in
    feature discovery rather than direction alignment.

### What we do not yet know

See §2 (Outstanding Questions) for the full list. The highest-priority unknowns are:

- Whether sign-aware feature subsets or lower scale factors can recover targeted steering
  from the store direction.
- Why PT models give 0/8 on classification prompts and whether continuation-style prompt
  reformulation would improve store-direction supervision.
- What specifically about the standard capitals_states prompt drives gemma-3-4b-it into
  the anti-target feature regime, given that the indirect prompt eliminates the saturation
  entirely (V10 OQ-I finding).
- Whether direct projection of the **embed** direction (not just store) shows the same
  strong alignment that pipeline-mediated embed interventions exhibit. V11 only tested
  store-direction direct projection.
- **V12 addition:** Whether alternative extraction strategies beyond attention-weighted
  context averaging (e.g., layer-specific extraction, multi-position aggregation with
  different pooling schemes, or contrastive extraction across prompt pairs) can produce
  a store direction with better geometric alignment. The V12 context-enhanced equivalence
  closes one avenue but leaves open more aggressive extraction reformulations.

---

## 2. Outstanding Questions

### Active

| ID | Question | Related legacy IDs | Priority | Primary configs |
|----|----------|--------------------|----------|-----------------|
| OQ-A | **Why do PT models give 0/8 on classification prompts, and would continuation-style prompts produce better store supervision?** Classification framing asks the model to categorise, but PT models are optimised for factual completion. The store direction is derived from weak-supervision states. Reformulating as factual completions or cloze-style templates may produce latent states aligned with the model's natural prediction distribution and improve store-direction quality. | OQ1, H1, OQ-H | High | gemma2_pt_capitals_states, gemma3_4b_pt_capitals_states |
| OQ-B | **How much of the observed behaviour comes from sign-agnostic ranking (`abs_()`) vs the underlying feature semantics? Does the raw direction vector have geometric alignment with the target distinction?** Would positive-only amplification or negative-only ablation change the gap more cleanly than the mixed-sign intervention? **V11 update:** Direct projection (bypassing feature selection entirely) shows the store direction has meaningful raw alignment only for cat_dog (Δ = +3.00, probe separation 0.102). For capitals_states (Δ ≈ 0, probe separation −0.014) and oqi_indirect (Δ = 0), the pipeline effect is entirely feature-mediated. Feature mediation can also invert the direction signal: cat_dog store pipeline Δ = −0.94 vs direct projection Δ = +3.00. **V12 update:** Context-enhanced extraction (attention-weighted average over all positions) produces equivalent metrics to standard answer-position extraction across all configs (cosine diff ≤ 0.003, probe sep diff < 0.001), confirming the store direction's weakness is geometric, not an extraction-mode artifact. **Remaining:** test embed-direction direct projection and sign-aware subsets. | OQ5, H2 | High | all active configs |
| OQ-C | **Is there a scale factor regime where store-based interventions produce targeted steering rather than broad disruption?** V3 showed scale_factor=10 is in the destructive regime; sweeps at 0.5–5.0 have not been aggregated. | OQ1 (partial), H1 | High | gemma2_pt_capitals_states, gemma3_4b_pt_capitals_states, gemma3_4b_it_capitals_states |
| OQ-E | **Does transcoder mismatch (base-trained transcoders on IT models) cause the IT suppression pattern?** Gemma 2 IT interventions drop all key-token logits 5–20 points; Gemma 3 IT is milder. Are IT-matched transcoders the fix, or is this inherent to the pipeline? | OQ3, H3 | Medium | gemma2_it_capitals_states, gemma3_it_capitals_states, gemma3_4b_it_capitals_states |
| OQ-F | **Are store features more influential but less aligned than embed features?** Store directions often produce larger magnitude Δ with low cosine overlap. Quantify: mean activation magnitude, influence score, sign distribution of store vs embed top-N features. | OQ4, H4 | Medium | all active configs |
| OQ-G | **What is the causal feature density for each config?** Progressive ablation (top-5/10/25/50/100) slope indicates whether the concept is concentrated in a few features or distributed. Determines whether targeted intervention is even feasible. | H5 | Medium | gemma2_pt_capitals_states, gemma3_4b_pt_capitals_states |
| OQ-I | **Is the IT model amplification saturation hypothesis real, and can softcap-aware analysis, reasoning-chain prompts, or alternative prompts mitigate it?** gemma-3-4b-it shows store Δ = −15.13 (capitals_states) despite 8/8 correct predictions. **V10 update:** saturation is prompt-specific, not model-inherent. Indirect JFK prompt reverses sign to +9.44 store Δ; reasoning-chain reduces magnitude to −1.69. Feature Jaccard drops from 0.333 to 0.111 for both alternative prompts, indicating different feature subspaces. Cosine sim is unchanged (−0.035). gemma-2-2b-it shows no sensitivity to prompt variant (all Δ within ±0.38). **V11 update:** oqi_indirect now shows embed Δ = store Δ = +8.69 (identical), Jaccard = 0.818 (highest observed). Direct projection Δ = 0 for both standard and indirect prompts, confirming that the prompt does not change the raw direction geometry — only the feature selection. Scale sweep is non-monotonic: −3.00 (2×), −4.75 (5×), +8.69 (10×), +1.13 (20×), +4.00 (50×). **Remaining:** understand what triggers the standard prompt's anti-target regime and the indirect prompt's non-monotonic scale behavior. | New (item 4b) | Medium (was High) | gemma3_4b_it_capitals_states, gemma2_it_capitals_states |

### Answered / Resolved

| ID | Question | Resolution | Wave |
|----|----------|------------|------|
| AQ-1 | Are `apply_chat_template` and `gemma_dataclass` render modes equivalent for Gemma 3 IT? | **Yes.** V4 confirmed identical tokenisation and SUMMARY_RECORDs for gemma3_it_capitals_states under both modes. The `gemma_dataclass` configs have been de-prioritised. | V4 |
| AQ-2 | Does direction negation change the intervention? | **No, by design.** `abs_()` in `compute_partial_influences` makes feature selection sign-agnostic. Negating the direction selects the same features with the same activations. | V3 |
| AQ-3 | Can Gemma 2 IT configs run within 24GB VRAM? | **Yes, with BATCH_SIZE=128.** Default 1024 caused OOM; 128 resolves it for all gemma-2-2b-it configs. | V5 |
| AQ-4 | Can Gemma 3 4B models run in the concept-direction pipeline? | **Yes.** After Gemma3ForConditionalGeneration hook-mapping and multimodal NNsight fixes, both 4B PT and 4B IT run end-to-end. The 4B PT benchmark was kept out of the active suite; the notebook run is a comparison artifact. | V7/V8 |
| AQ-5 | Are results reproducible across waves? | **Yes.** V5 and V6 produce identical SUMMARY_RECORDs for matching configs. | V5/V6 |
| AQ-6 | Does gemma-3-1b-it dog/cat show complete feature convergence (Jaccard=1.0)? | **No — the V3–V6 Jaccard=1.0 was an artifact of invalid key_labels.** V9 corrected the dog_cat pipeline bugs (hardcoded Austin/Dallas/Texas key_labels, missing chat_intervention_prompt). The true gemma-3-1b-it dog_cat Jaccard is 0.333 with cosine_sim=0.129 — materially different feature subspaces, consistent with all other configs. | V9 |

### De-prioritised

| ID | Question | Reason | Wave |
|----|----------|--------|------|
| DQ-1 | Can gemma-3-1b-pt solve Austin-Dallas? | **No — baseline top-1 is "the", not "Austin".** The 1B PT model cannot distinguish US state capitals from generic completions. The embed Δ = +6.06 is misleading (distribution collapse, not steering). | V4/V7 |
| DQ-2 | Is gemma-3-1b-it useful for concept-direction work? | **Low value.** Marginal amplification effects, and no neuronpedia dashboards exist for 1B transcoders, blocking feature inspection. | V4/V5 |
| DQ-3 | Does `gemma_dataclass` render mode produce different results from `apply_chat_template`? | **Identical for Gemma 3 IT** (AQ-1). For Gemma 2 IT the dataclass path changes the intervention regime materially, but since the tokenizer path is the standard HF convention, the dataclass variant is de-prioritised. | V4 |
| DQ-4 | Should the dog/cat store divergence across families be investigated before capitals/states is understood? | **No, but dog_cat is now a valid secondary diagnostic.** V9 corrected the data integrity issues (AQ-6). Dog/cat shows the same PT 0/8 vs IT 8/8 pattern as capitals_states and provides useful cross-concept comparison, but capitals_states remains the primary target. | V4/V8/V9 |

---

## 3. Results Tables

### Capitals/States — Active Models (V8 op-driven values where available)

| Model | Variant | Baseline 1st token | Austin prob | Dallas prob | Baseline gap | Embed Δ | Store Δ | Cosine sim | Jaccard | Predictions | Wave | Artifact |
|-------|---------|-------------------|-------------|-------------|-------------|---------|---------|------------|---------|-------------|------|----------|
| gemma-2-2b | PT | Austin | 0.4095 | 0.0312 | +2.57 | **+2.97** | **−0.65** | 0.078 | 0.111 | 0/8 | V8 | gemma2_pt_capitals_states_20260327_175111.ipynb |
| gemma-2-2b-it | IT | Austin | — | — | — | +0.14 | +0.05 | 0.161 | 0.111 | 8/8 | V5 | gemma2_it_capitals_states (V5 wave) |
| gemma-3-4b-pt | PT | Austin | 0.5328 | 0.1189 | +1.50 | **+3.25** | **−0.88** | 0.024 | 0.667 | 0/8 | V8 | gemma3_4b_pt_capitals_states_20260327_175809.ipynb |
| gemma-3-4b-it | IT | Austin | — | — | — | −7.56 | **−15.13** | −0.035 | 0.333 | 8/8 | V8 | gemma3_4b_it_capitals_states_20260327_181944.ipynb |
| gemma-3-4b-it | IT | Austin | — | — | — | −7.00 | −8.38 | −0.035 | 0.667 | 8/8 | V12 | v12_gemma3_4b_it_capitals_states |
| gemma-3-4b-it (ctx_enh) | IT | Austin | — | — | — | −7.00 | −8.38 | −0.032 | 0.667 | 8/8 | V12 | v12_gemma3_4b_it_capitals_states_ctx_enhanced |
**Key pattern:** PT embed helps, PT store hurts. IT predictions are correct but IT interventions
steer anti-target — especially the 4B IT store direction (−15.13).

### Capitals/States — 262k Width (V9)

| Model | Variant | Baseline 1st token | Embed Δ | Store Δ | Cosine sim | Jaccard | Predictions | Wave | Note |
|-------|---------|-------------------|---------|---------|------------|---------|-------------|------|------|
| gemma-3-4b-pt | PT | Austin | +3.13 | −0.75 | 0.024 | 0.667 | 0/8 | V9 | 262k transcoder, similar to 16k |
| gemma-3-4b-it | IT | Austin | −6.84 | −12.58 | −0.035 | 0.333 | 8/8 | V9 | 262k transcoder, similar to 16k |

### Capitals/States — De-prioritised Models

| Model | Variant | Baseline 1st token | Embed Δ | Store Δ | Cosine sim | Predictions | Wave | Note |
|-------|---------|-------------------|---------|---------|------------|-------------|------|------|
| gemma-3-1b-pt | PT | the | +6.06 | +0.38 | 0.271 | 0/8 | V4/V7 | Weak baseline; embed Δ is distribution collapse |
| gemma-3-1b-it | IT | — | +0.38 | −0.88 | 0.165 | 8/8 | V4 | Marginal effects; no NP dashboards |

### Dog/Cat — 16k Width (V9 — corrected data)

| Model | Variant | Baseline 1st token | Embed Δ | Store Δ | Cosine sim | Jaccard | Predictions | Wave |
|-------|---------|-------------------|---------|---------|------------|---------|-------------|------|
| gemma-2-2b | PT | Dog | −4.34 | −6.34 | 0.169 | 0.250 | 0/8 | V9 |
| gemma-2-2b-it | IT | Dog | −0.21 | −0.13 | 0.177 | 0.250 | 8/8 | V9 |
| gemma-3-1b-pt | PT | Dog | −3.00 | +11.25 | 0.240 | 0.667 | 0/8 | V9 |
| gemma-3-1b-it | IT | Dog | −2.06 | −2.81 | 0.129 | 0.333 | 8/8 | V9 |
| gemma-3-4b-pt | PT | Dog | +2.25 | +4.06 | 0.225 | 0.818 | 0/8 | V9 |
| gemma-3-4b-it | IT | Dog | −1.25 | −0.63 | 0.115 | 0.250 | 8/8 | V9 |

### Dog/Cat — 262k Width (V9)

| Model | Variant | Baseline 1st token | Embed Δ | Store Δ | Cosine sim | Jaccard | Predictions | Wave |
|-------|---------|-------------------|---------|---------|------------|---------|-------------|------|
| gemma-3-4b-pt | PT | Dog | +2.81 | +6.25 | 0.225 | 0.818 | 0/8 | V9 |
| gemma-3-4b-it | IT | Dog | +0.25 | −0.88 | 0.115 | 0.250 | 8/8 | V9 |

**Note:** V3–V6 dog_cat results were invalid due to hardcoded Austin/Dallas/Texas
key_labels and missing chat_intervention_prompt for IT variants (see AQ-6). All values
above are from the corrected V9 pipeline.

### Cat/Dog (reversed direction) — 16k Width (V10/V11)

Cat_dog reverses the concept direction from dog_cat: the direction vector points
Cat→Dog, so positive Δ = intervention reinforces Cat-over-Dog.

| Model | Variant | Embed Δ | Store Δ | Cosine sim | Jaccard | Predictions | Wave |
|-------|---------|--------:|--------:|-----------:|--------:|:-----------:|------|
| gemma-2-2b | PT | +4.26 | +5.06 | 0.155 | 0.176 | 0/8 | V10 |
| gemma-2-2b-it | IT | +0.70 | −0.02 | 0.160 | 0.429 | 8/8 | V10 |
| gemma-3-4b-pt | PT | −0.13 | −4.06 | 0.127 | 0.818 | 0/8 | V10 |
| gemma-3-4b-it | IT | +3.50 | −0.94 | 0.073 | 0.429 | 8/8 | V11 |
| gemma-3-4b-it (ctx_enh) | IT | +3.50 | −0.94 | 0.070 | 0.429 | 8/8 | V12 |

### Cat/Dog — 262k Width (V10)

| Model | Variant | Embed Δ | Store Δ | Cosine sim | Jaccard | Predictions | Wave |
|-------|---------|--------:|--------:|-----------:|--------:|:-----------:|------|
| gemma-3-4b-it | IT | +3.25 | +4.88 | 0.073 | 0.333 | 8/8 | V10 |

### OQ-I Probes — Capitals/States IT with Alternative Prompts (V10)

| Config | Model | Prompt type | Target tokens | Embed Δ | Store Δ | Cosine sim | Jaccard | Predictions | Wave |
|--------|-------|------------|---------------|--------:|--------:|-----------:|--------:|:-----------:|------|
| gemma2_it (V5 baseline) | gemma-2-2b-it | standard | Austin/Dallas | +0.14 | +0.05 | 0.161 | 0.111 | 8/8 | V5 |
| gemma2_it_oqi_indirect | gemma-2-2b-it | indirect (JFK) | Austin/Dallas | −0.11 | +0.18 | 0.175 | 0.053 | 8/8 | V10 |
| gemma2_it_oqi_reasoning | gemma-2-2b-it | reasoning chain | Sacramento/Austin | −0.31 | −0.38 | 0.176 | 0.111 | 8/8 | V10 |
| gemma3_4b_it (V8 baseline) | gemma-3-4b-it | standard | Austin/Dallas | −7.56 | −15.13 | −0.035 | 0.333 | 8/8 | V8 |
| gemma3_4b_it_oqi_indirect | gemma-3-4b-it | indirect (JFK) | Austin/Dallas | +4.63 | +9.44 | −0.035 | 0.111 | 8/8 | V10 |
| gemma3_4b_it_oqi_indirect | gemma-3-4b-it | indirect (JFK) | Austin/Dallas | +8.69 | +8.69 | −0.035 | 0.818 | 8/8 | V11 |
| gemma3_4b_it_oqi_indirect (ctx_enh) | gemma-3-4b-it | indirect (JFK) | Austin/Dallas | +8.69 | +8.69 | −0.032 | 0.818 | 8/8 | V12 |
| gemma3_4b_it_oqi_reasoning | gemma-3-4b-it | reasoning chain | Sacramento/Austin | −0.38 | −1.69 | −0.044 | 0.111 | 8/8 | V10 |

**Key OQ-I finding:** gemma-3-4b-it's extreme anti-target saturation (store Δ = −15.13)
is prompt-specific. The indirect prompt reverses it to +9.44; the reasoning-chain prompt
reduces it to −1.69. Feature Jaccard drops from 0.333 to 0.111, indicating substantially
different feature subspaces are engaged. gemma-2-2b-it shows no sensitivity.
V11 update: oqi_indirect embed/store deltas converge to +8.69 (identical), Jaccard rises
to 0.818. V10→V11 delta shifts are due to bug fixes (issues 1, 5).

### Direct Projection — Store Direction (V11/V12)

Direct projection applies the store concept-direction vector directly to the residual
stream at `unembed.hook_in`, bypassing circuit-tracer feature selection entirely.

| Config | Model | Pre Gap | Post Gap | DP Δ | Pipeline Embed Δ | Pipeline Store Δ | Store Probe Sep | Wave |
|--------|-------|--------:|---------:|-----:|-----------------:|-----------------:|----------------:|------|
| cat_dog | gemma-3-4b-it | −9.75 | −6.75 | **+3.00** | +3.50 | −0.94 | 0.102 | V11 |
| cat_dog (ctx_enh) | gemma-3-4b-it | −9.75 | −6.75 | **+3.00** | +3.50 | −0.94 | 0.101 | V12 |
| capitals_states | gemma-3-4b-it | +27.88 | +27.75 | **−0.13** | −7.00 | −8.38 | −0.014 | V11 |
| capitals_states (ctx_enh) | gemma-3-4b-it | +27.88 | +27.75 | **−0.13** | −7.00 | −8.38 | −0.014 | V12 |
| oqi_indirect | gemma-3-4b-it | −11.25 | −11.25 | **+0.00** | +8.69 | +8.69 | −0.014 | V11 |
| oqi_indirect (ctx_enh) | gemma-3-4b-it | −11.25 | −11.25 | **+0.00** | +8.69 | +8.69 | −0.014 | V12 |

**Key finding:** Direct projection Δ correlates with store probe separation, not with
pipeline Δ magnitude. The pipeline's large effects on capitals_states and oqi_indirect
are entirely feature-mediated — the raw direction has no alignment with the target
distinction at the residual-stream level. **V12 update:** context-enhanced extraction
produces equivalent direct projection results (identical DP Δ), confirming the
direction's geometric properties are extraction-mode invariant.

---

## 4. Next Steps (Experimental Priorities)

### V12 Wave — Complete ✅

V12 introduced context-enhanced extraction mode (`STORE_LATENT_EXTRACTION_MODE:
context_enhanced`) and a five-approach framework (embed+features, store+features,
store+direct_projection, ctx_enhanced_store+features, ctx_enhanced_store+direct_projection).
All 6 experiments (gemma-3-4b-it × {cat_dog, capitals_states, oqi_indirect} × {standard,
ctx_enhanced}) completed successfully. Also fixed a `batch=None` tokenization bug in
`run_direct_projection_pipeline`. Key finding: context-enhanced extraction is numerically
equivalent to standard answer-position extraction (cosine diff ≤ 0.003, probe sep diff
< 0.001), confirming the store direction's weakness is geometric, not an extraction-mode
artifact. Results are in §3 (the V12 per-wave summary was consolidated into this
document, 2026-04-25).
See also [concept_direction_analysis.md](concept_direction_analysis.md) for detailed
feature analysis with Neuronpedia dashboard links and probe separation tables.

### V11 Wave — Complete ✅

V11 included 8 harness quality fixes and a new direct projection store-direction variant
that bypasses circuit-tracer feature selection entirely. All 3 experiments (gemma-3-4b-it ×
{cat_dog, capitals_states, oqi_indirect}) completed 27/27 cells. Key findings: direct
projection is informative only when the direction has intrinsic geometric alignment
(cat_dog); feature mediation can invert the direction signal; OQ-I indirect embed/store
deltas converge to identical values with Jaccard = 0.818. Results are in §3 (the V11
per-wave summary was consolidated into this document, 2026-04-25).

### V10 Wave — Complete ✅

V10 tested direction asymmetry (cat_dog) and began the OQ-I investigation with
reasoning-chain and indirect prompts. All 9 experiments (5 cat_dog + 4 OQ-I) ran
to completion. Key findings: PT direction reversal is approximately symmetric;
IT amplification saturation is prompt-specific, not model-inherent. Results are in §3
(the V10 per-wave summary was consolidated into this document, 2026-04-25).

### V9 Wave — Complete ✅

V9 addressed the dog_cat data integrity issues and added 262k transcoder variants.
All 10 experiments ran to completion. Results are in §3 (the V9 per-wave summary was
consolidated into this document, 2026-04-25).

### Subsequent priorities (post-V12)

These are ordered by expected diagnostic value:

1. **Continuation-style prompt reformulation for PT store** (OQ-A).
   Design factual-completion prompt variants for the store-direction pipeline to test
   whether the 0/8 classification failure is the root cause of the weak PT store signal.

2. **Understand gemma-3-4b-it standard prompt anti-target regime** (OQ-I follow-up).
   V10 showed the saturation is prompt-specific: indirect prompt reverses it, reasoning-
   chain eliminates it. Investigate which tokens/positions in the standard prompt drive
   the anti-target feature selection. Test additional prompt variants to map the boundary
   between positive and negative Δ. Consider whether the standard prompt should be
   replaced for 4B IT experiments.

3. **Sign-aware feature subsets and embed-direction direct projection** (OQ-B).
   Classify selected features by activation sign; run positive-only amplification and
   negative-only ablation to test whether mixed-sign cancellation is the store-direction's
   primary failure mode. V11 showed that feature mediation can invert the direction signal
   (cat_dog store pipeline Δ = −0.94 vs direct projection Δ = +3.00), making sign-aware
   analysis a higher priority. Also extend direct projection to the embed direction to
   test whether embed has stronger raw geometric alignment. Investigate sign-aware CT
   feature selection options in `circuit_tracer/graph.py`.

4. **Scale-factor sweep aggregation** (OQ-C).
   The sweep data exists in per-notebook outputs but is not in SUMMARY_RECORDs. Add
   sweep-best-scale and inflection-point metrics to the record format and re-aggregate.

5. **Neuronpedia source set layer coverage improvement.**
   `gemmascope-2-transcoder-16k` only covers the first 12 layers for gemma-3-4b-it.
   Investigate whether updated source sets with full layer coverage (all 33 layers) are
   available or can be requested. Until then, use 262k dashboards for full-layer
   feature inspection on gemma-3-4b-it.

6. **262k transcoder optimisation (de-prioritised).**
   262k experiments work but are slow (~51 min vs ~4 min for 16k) due to per-layer lazy
   W_enc/W_dec disk reads (34 layers × 1.34 GB each via safetensors). Memory is NOT a
   constraint: VMPeak 93 GB is mmap virtual address space, not physical (VmHWM=12.7 GB,
   system has 62 GiB). Core metric patterns (cosine_sim, Jaccard) match 16k. Defer to
   when the wider transcoder provides differentiated signal worth the runtime cost.

---

## 5. Conventions

- **Embed Δ** = (post-intervention gap) − (pre-intervention gap) for the embed direction.
  For capitals_states: positive = intervention reinforces the Austin-over-Dallas lead.
  For dog_cat: positive = intervention reinforces the Dog-over-Cat lead.
  For cat_dog: positive = intervention reinforces the Cat-over-Dog lead.
- **Store Δ** = same, for the store direction.
- **Cosine sim** = cosine similarity between normalised embed and store direction vectors.
- **Jaccard** = Jaccard index on the top-N feature sets selected by each direction.
- **Predictions** = store-direction prompt-conditioned classification accuracy (X/8).
- Per-wave detail (V1–V12), legacy hypotheses, and the legacy observation log were
  consolidated into this document in the 2026-04-25 consolidation; the standalone
  `archived_analysis/` files no longer exist.
