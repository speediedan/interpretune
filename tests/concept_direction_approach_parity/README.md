# Concept Direction Approach Parity

## Ultimate Objective

Build an `AnalysisStore`-based `concept_direction` op for the proven capital-states difference
that approximates the embed-based approach (or is at least demonstrably useful) and that works
not only in gemma-2-2b (current config prefix: `gemma2_pt_*`) but also gemma-2-2b-it,
gemma-3-1b-it, and the validated Gemma 3 4B PT text-only path.

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

Current naming note: the non-IT Gemma 2 configs are now named `gemma2_pt_*` for consistency
with Gemma 3 PT. Historical notebooks and older analysis notes may still refer to
`gemma2_base_*`, but they target the same `google/gemma-2-2b` model.

Prompt note: `CLASSIFICATION_QUESTION_V3` now explicitly asks for one-word answers
`" Capital"` or `" State"`, which makes the PT capitals/states runs easier to compare
directly across model sizes.

## Architecture Notes

Core harness infrastructure now lives in `tests/nb_experiment_harness/`.
This directory keeps the concept-direction experiment definitions, concept-pair YAMLs, and
analysis-specific notebook logic, while shared launcher/bootstrap/session/config code is owned by
the shared harness package.

The circuit-tracer NNsight module uses its own `TranscoderSet` (not SAELens SAE objects),
so the `fwd_w_cache_and_latent_models` path (which requires `sae_handles` from
`_SAEHandleMixin`) is **not available**. We use `fwd_w_cache` with `unembed.hook_in` to
capture pre-logit residual-stream activations from a plain forward pass. Transcoder splicing
happens only during the attribution graph computation.

## Notebook Modes

`concept_direction_template.ipynb` now supports three analysis modes via the
`ANALYSIS_MODE` notebook/YAML parameter:

- `concept_pair`: the original embed-direction vs store-direction comparison.
- `explicit_embedding_difference`: computes the primary direction directly from
   `EXPLICIT_DIRECTION_TOKENS`, then runs the regular attribution-graph and intervention
   pipeline without the store-direction phases.
- `debug_intervention_pipelines`: bypasses concept-direction generation entirely and runs a
   single-feature intervention validation pass modeled on circuit-tracer's Gemma 3 NNsight
   checks.

### Explicit Embedding-Difference Mode

Use `ANALYSIS_MODE: explicit_embedding_difference` together with
`EXPLICIT_DIRECTION_TOKENS: [Sacramento, Austin]` when you want the harness to derive a
direction from a direct token-embedding difference instead of the concept-pair registry.

This mode still supports:

- the embed-based attribution-graph pipeline
- direct residual-stream projection
- scale sweeps, progressive ablations, and sign-aware analysis

This mode skips the concept-pair-only phases:

- store-direction extraction
- embed/store comparison summary
- direction consistency probes

### Debug Intervention Mode

Use `ANALYSIS_MODE: debug_intervention_pipelines` for single-feature pipeline validation.
This mode is intended to answer a narrower question than the normal notebook: whether the
selected feature intervention matches the adjacency-matrix prediction under the same
conditions used by the circuit-tracer validation tests.

Expected settings for this mode:

- exactly one entry in `CONSTRAINED_FEATURE_SELECTION_LIST`
- `ENABLE_ZERO_SOFTCAP: true` for Gemma 3-style validation parity
- `DEFAULT_SCALE_FACTOR: 2.0` unless you are intentionally probing a different multiplier
- `DEBUG_SESSION_SURFACE_PRESET: parity_surface` when the notebook debug run should mirror the standalone circuit-tracer validation surface while preserving GPU execution when available

Keep that preset scoped to `ANALYSIS_MODE: debug_intervention_pipelines`. The normal localhost-backed concept-pair configs should stay on the notebook-default session surface; targeted OQI reruns showed that enabling `parity_surface` on non-debug runs materially worsened the embed/store deltas without fixing the remaining dependency-level drift.

Behavior differences in this mode:

- graph targets come from the configured `KEY_TOKENS`
- the notebook skips Phases 6-10's normal analyses and replaces Phase 6 with a dedicated
   debug validation report
- validation uses `apply_activation_function=False`, `constrained_layers=range(n_layers)`,
   and compares observed activation/logit deltas against the selected adjacency column

Additional references:

- `intervention_pipeline_debugging_mode.md` documents the debug-mode config surface and the `parity_surface` preset.
- `../parity_analysis/intervention_graph_parity_testing.md` documents the preserved-artifact parity workflow and the manual ablation tooling.

The new config
`tests/concept_direction_approach_parity/archived_cfgs/gemma3_4b_it_local_oqi_reasoning_single_fs_di.yaml`
is the historical single-feature debug entry point for the local OQI reasoning prompt. That
config still targets the originally requested `23/2313` feature, which is inactive for the
current prompt.

The current active high-layer debug configs are:

- `tests/concept_direction_approach_parity/archived_cfgs/gemma3_4b_it_local_oqi_reasoning_single_fs_di_60.yaml`
- `tests/concept_direction_approach_parity/archived_cfgs/gemma3_4b_it_local_oqi_reasoning_single_fs_di_60_full_graph.yaml`
- `tests/concept_direction_approach_parity/archived_cfgs/gemma3_4b_it_local_oqi_reasoning_single_fs_di_60_no_softcap.yaml`

These follow-up configs target `gemma-3-4b-it/25-gemmascope-2-transcoder-16k/60`, chosen from
the normal non-debug OQI run because it is the highest-ranked active feature above layer 20 in
the embed pipeline.

### Latest Debug Investigation: Gemma 3 4B IT OQI Feature 25/60 (2026-04-10)

The most useful high-layer candidates from the current normal OQI run were:

- Embed top features: `(25, 34, 60)` at `1.56e-08`, `(33, 34, 1061)` at `1.02e-08`,
   `(31, 34, 721)` at `9.82e-09`, `(23, 34, 2748)` at `8.67e-09`
- Store top features: `(33, 34, 1061)` at `1.84e-08`, `(31, 34, 721)` at `1.70e-08`,
   `(3, 34, 69)` at `1.56e-08`

Three completed notebooks now capture the focused `25/60` investigation:

- `generated_experiments/gemma3_4b_it_local_oqi_reasoning_single_fs_di_60_20260410_225906.ipynb`
- `generated_experiments/gemma3_4b_it_local_oqi_reasoning_single_fs_di_60_full_graph_20260410_230326.ipynb`
- `generated_experiments/gemma3_4b_it_local_oqi_reasoning_single_fs_di_60_no_softcap_20260410_230725.ipynb`

Key findings from those runs:

- Baseline `25/60` debug validation still fails, but the selected feature's self-row is exact:
   predicted and observed deltas are both `0.0` for `(25, 34, 60)`.
- The dominant residual has moved off the selected feature and into downstream layer-33 nodes.
   In the baseline/no-softcap runs the worst row is `(33, 34, 304)` with
   `expected_delta=-24.125`, `actual_delta=-56.0`, and `abs_error=31.875`.
- The `no_softcap` variant is numerically identical to the baseline notebook for the tracked
   diagnostics, so softcap handling is not driving the remaining mismatch.
- The `full_graph` variant increases retained feature nodes from `8192` to `20000`, but that
   does not resolve the failure. The max activation error rises to `34.25`, the max logit error
   rises to `0.2793`, and the top residual shifts to `(33, 34, 198)`.
- Compared with the earlier active fallback feature `23/2748`, the `25/60` runs reduce the max
   activation error (`31.875` vs `38.125`) but lose the previous logit pass. The common pattern is
   that the selected feature itself remains exact while downstream layer-33 rows drift.

### Latest Localhost Follow-up: Non-Debug Rerun on Current Dependency State (2026-04-12)

After validating that `DEBUG_SESSION_SURFACE_PRESET: parity_surface` should remain debug-only, the
requested localhost-backed non-debug configs were rerun with the notebook-default session surface:

- `generated_experiments/gemma3_4b_it_local_oqi_reasoning_20260412_173925.ipynb`
- `generated_experiments/gemma3_4b_it_local_oqi_reasoning_single_fs_20260412_174857.ipynb`
- `generated_experiments/gemma3_4b_it_local_cat_dog_20260412_175750.ipynb`
- `generated_experiments/gemma3_4b_it_local_capitals_states_20260412_180711.ipynb`
- `generated_experiments/gemma3_1b_it_local_capitals_states_20260412_181555.ipynb`

Observed summary deltas on the current environment:

- `gemma3_4b_it_local_oqi_reasoning`: embed `-37.75`, store `-30.875`, Jaccard `0.4286`, predictions `8/8`
- `gemma3_4b_it_local_oqi_reasoning_single_fs`: embed `-53.0`, store `-53.0`, Jaccard `1.0`, predictions `8/8`
- `gemma3_4b_it_local_cat_dog`: embed `-4.25`, store `-38.75`, Jaccard `0.3333`, predictions `8/8`
- `gemma3_4b_it_local_capitals_states`: embed `0.0`, store `0.0`, Jaccard `1.0`, predictions `8/8`
- `gemma3_1b_it_local_capitals_states`: embed `+14.5`, store `+14.5`, Jaccard `0.4286`, predictions `8/8`

Compared with the last clean 2026-04-09 artifacts, only the 4B capitals_states rerun stayed near its
old neutral behavior. The OQI reasoning run regressed from embed/store `+17.3125 / +3.3594` to
`-37.75 / -30.875`, the single-feature OQI run regressed from `+0.8125 / 0.0` to `-53.0 / -53.0`,
the 4B cat_dog run regressed from `+7.75 / 0.0` to `-4.25 / -38.75`, and the 1B capitals_states run
flipped from `-3.375 / -6.9375` to `+14.5 / +14.5`.

The important conclusion is that removing `parity_surface` from the non-debug configs was the
correct Interpretune-side fix, but it does not restore the earlier baselines. The remaining drift is
consistent with the current dirty `circuit-tracer` checkout, where the NNsight trace path now
canonicalizes string prompts into token tensors before tracing. Treat the 2026-04-12 localhost
artifacts as current-environment references, not as replacements for the 2026-04-09 parity baselines.

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

### V4–V6: Full Matrix with Gemma 3, Resource Management Fix (2026-03-27)

**Setup**: All 8 baseline configs executed via the notebook launcher. V4 was the first full-matrix run.
V5/V6 reran previously-failed configs after resource management fixes (batch_size=128 for Gemma 2 IT).

#### Combined Results Matrix

| Model | Concept Pair | Embed Δ | Store Δ | Cosine Sim | Jaccard | Predictions | Wave |
|-------|-------------|---------|---------|------------|---------|-------------|------|
| gemma-2-2b (base) | capitals/states | +2.97 | +0.005 | 0.239 | — | — | V4 |
| gemma-2-2b (base) | dog/cat | −4.34 | −1.92 | 0.184 | — | — | V4 |
| gemma-2-2b-it | capitals/states | +0.141 | +0.047 | 0.161 | 0.111 | 8/8 | V5 |
| gemma-2-2b-it | dog/cat | +0.250 | −0.340 | 0.048 | 0.250 | 8/8 | V5/V6 |
| gemma-3-1b-pt | capitals/states | +6.06 | +0.375 | 0.271 | — | — | V4 |
| gemma-3-1b-pt | dog/cat | −3.00 | +11.63 | 0.290 | — | — | V4 |
| gemma-3-1b-it | capitals/states | +0.375 | −0.875 | 0.165 | 0.333 | 8/8 | V4 |
| gemma-3-1b-it | dog/cat | +3.730 | +3.730 | 0.063 | 1.000 | 8/8 | V5/V6 |

#### V4–V6 Key Findings

1. **Batch size fix resolves Gemma 2 IT OOM.** Setting `BATCH_SIZE: 128` (vs default 1024)
   allows gemma-2-2b-it configs to complete within 24GB VRAM. See `resource_management.md`.

2. **Gemma 2 IT effects are mild under chat template.** Both capitals/states (embed Δ=+0.141)
   and dog/cat (embed Δ=+0.250) show small positive deltas, unlike V3's stronger negative
   effects under `gemma_dataclass` render mode.

3. **Gemma 3 IT dog/cat has complete feature convergence.** Jaccard=1.0 — embed and store
   directions select identical features, producing identical deltas (+3.730). This is unique
   across all configs.

4. **Cross-family PT/non-IT pattern reproduces.** Both Gemma 2 PT (historically labeled
   `base`) and Gemma 3 PT show strong embed-direction gap improvement for capitals/states
   with near-zero store influence.

5. **Dog/cat store direction diverges across families.** Gemma 3 PT dog/cat has store Δ=+11.63
   while Gemma 2 PT has store Δ=−1.92. Largest cross-family divergence in the matrix.

6. **Results are reproducible.** V5 and V6 produce identical SUMMARY_RECORDs for matching
   configs (same embed Δ, store Δ, cosine sim, Jaccard, predictions).

### V7: Historical PT Capitals/States Comparison (2026-03-27)

The filtered PT-only wave gives the cleanest current read on the non-IT capitals/states task.

| Config | Baseline first token | Austin prob | Dallas prob | Austin−Dallas gap | Embed Δ | Store Δ | Cos sim | Jaccard | Interpretation |
|---|---|---|---|---|---|---|---|---|---|
| `gemma2_pt_capitals_states` | `Austin` | `0.4095` | `0.0312` | `+2.5734` | `+2.9734` | `-1.7183` | `0.2098` | `0.1765` | Cleanest 2B PT intervention target. Embed widens a correct Austin lead; store gives most of it back. |
| `gemma3_pt_capitals_states` | `the` | `0.0156` | `0.0745` | `-1.5625` | `+6.0625` | `+0.3750` | `0.2895` | `0.4286` | Weak 1B PT baseline. Embed flips the gap, but only by collapsing the answer distribution; store barely helps and does not fix the wrong answer. |
| `gemma3_4b_pt_capitals_states` | `Austin` | `0.5328` | `0.1189` | `+1.5000` | `+3.2500` | `-0.8750` | `0.2970` | `0.6667` | Strongest raw PT baseline and best current Gemma 3 PT candidate. Embed helps moderately; store again compresses the Austin lead. |

Key readouts:

- Austin probability is the most stable cross-model comparison. On that metric the ordering is `gemma3_4b_pt` > `gemma2_pt` >> `gemma3_pt`.
- All three PT runs still report `Predictions: 0/8` for the store-direction prompt set, even with the explicit `" Capital"` / `" State"` wording. The store pipeline is still learning directions from prompt-conditioned states that do not robustly solve the task.
- The large `+6.0625` embed delta on `gemma3_pt` is not a win by itself. The accompanying notebook and the manual observation log show that the model starts producing nonsense completions when that direction is applied. The usable PT interventions are therefore `gemma2_pt` and `gemma3_4b_pt`, not `gemma3_pt`.

See [experimental_summaries.md](experimental_summaries.md) for full observation log and analysis.

### V8: Op-Driven Store Rerun + Gemma 3 4B IT (2026-03-27)

V8 reran the PT capitals/states notebooks after replacing the handwritten store-direction path with the
real `extract_concept_latent_state` + `extract_concept_latent_examples` pipeline, then added the new
`gemma3_4b_it_capitals_states` config. The 4B PT RTE/BoolQ benchmark YAML remains removed from the active
benchmark suite; the 4B PT notebook here is retained only as a concept-direction comparison artifact.

| Config | Artifact | Embed Δ | Store Δ | Cos sim | Jaccard | Predictions | Readout |
|---|---|---|---|---|---|---|---|
| `gemma2_pt_capitals_states` | `gemma2_pt_capitals_states_20260327_175111.ipynb` | `+2.9734` | `-0.6506` | `0.0784` | `0.1111` | `0/8` | PT baseline still favors Austin under embed, but the op-driven store path now weakens that gap instead of helping. |
| `gemma3_4b_pt_capitals_states` | `gemma3_4b_pt_capitals_states_20260327_175809.ipynb` | `+3.2500` | `-0.8750` | `0.0244` | `0.6667` | `0/8` | Strong 4B PT raw baseline, moderate embed gain, and the same PT store failure mode: prompt-conditioned states do not solve the task and compress the Austin lead. |
| `gemma3_4b_it_capitals_states` | `gemma3_4b_it_capitals_states_20260327_181944.ipynb` | `-7.5625` | `-15.1250` | `-0.0347` | `0.3333` | `8/8` | First successful 4B IT capitals/states run. Both directions steer away from the Austin-over-Dallas target gap, and the store direction is much stronger despite near-zero directional agreement with embed. |

#### V8 Key Findings

1. **The op-driven PT store path is still not task-aligned.** Both PT reruns keep `prediction_correct=0/8`, and both
   store directions reduce the Austin-minus-Dallas gap. The implementation is now aligned with the real analysis op
   pipeline, but the prompt-conditioned latent states are still weak supervision for this task.

2. **Gemma 3 4B PT and Gemma 2 PT now show the same qualitative split.** Embed improves the Austin lead; store erodes
   it. The absolute magnitudes differ, but the regime is consistent across the two viable PT notebook baselines.

3. **Gemma 3 4B IT closes the architecture bring-up gap but not the steering gap.** The new config runs end-to-end and
   preserves `8/8` prompt-conditioned classification accuracy, yet both embed and store interventions move in the wrong
   direction for the Austin-minus-Dallas target gap. Store is stronger than embed here, but stronger means more
   anti-target, not more semantically useful.

4. **Embed/store agreement is extremely low in every V8 run.** Cosine similarity stays near zero (`0.0784`, `0.0244`,
   `-0.0347`), reinforcing the earlier conclusion that contextual answer-position states are selecting a materially
   different feature subspace than the token-embedding direction.

5. **The remaining 4B IT blocker was display-layer, not analysis-layer.** The first run completed analysis and failed
   only when rendering the ablation chart. After hardening `display_ablation_chart()` against non-finite values and
   `tight_layout()` failures, the notebook completed successfully.

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

5. **Model compatibility**: Pure CausalLM support is still the easy path, but Gemma 3 4B PT/IT
   now works in the validated benchmark path and in both 4B capitals/states notebook runs by routing
   through `Gemma3ForConditionalGeneration`, adding multimodal hook mapping, and teaching the
   NNsight cache logic to traverse nested `language_model` backbones. Treat that as targeted
   multimodal-text support, not blanket VLM compatibility.

## Resource Management

GPU memory management is critical for running larger models (Gemma 2 2B-IT) within 24GB VRAM.
Key parameters:

- **`BATCH_SIZE`**: Circuit-tracer attribution batch size. The default inherited from the
   shared cfg aliases is 256; 128 works reliably for 2B IT-style configs and 64 is a safer
   starting point for 4B IT configs. Set via YAML config.
- **`MAX_FEATURE_NODES`**: Maximum feature nodes for attribution. Default 8192 is fine for most configs.
- **`PYTORCH_CUDA_ALLOC_CONF`**: Set `expandable_segments:True` to reduce allocator fragmentation.

See [resource_management.md](resource_management.md) for full details and [/docs/resource_management.md](/home/speediedan/repos/interpretune/docs/resource_management.md) for the user-facing guide.

## Files

- `concept_direction.py`: Concept-direction-specific config/runtime surface — `NotebookHarnessConfig`,
   concept-pair loading, direction computation, and local explanation preparation.
- `concept_direction_template.ipynb`: Parameterized concept-direction notebook template.
- `../nb_experiment_harness/nb_harness_utils.py`: shared notebook utility helpers used across
   experiment templates.
- `../nb_experiment_harness/pipeline_patterns.py`: shared notebook phase runners used by the
   concept-direction template.
- `../nb_experiment_harness/nb_experiment_launcher.py`: shared Papermill launcher for running
   parameterized experiment notebooks across YAML configs with timestamped output.
- `../nb_experiment_harness/README.md`: shared launcher/bootstrap/config/session harness notes.
- `archived_analysis/`: Historical notes and results from the retired standalone experimentation wave.
- `compare_approaches.py`: V1/V2 harness (predecessor). Supports `--model-variant it|base`,
  `--no-chat-template`, `--skip-intervention`, `--output`.
- `V3_ANALYSIS.md`: Comprehensive V3 analysis document with full results and findings.
- `HYPOTHESES.md`: Tracked hypotheses (H1–H7) for systematic investigation.
- `experimental_summaries.md`: Outstanding questions, observation log, and V4–V6 results.
- `resource_management.md`: Developer-facing GPU memory management guide.
- `configs/`: YAML config files for parameterized notebook experiments. Includes
  `archived_cfgs/` for deprecated `gemma_dataclass` render mode configs.
- `generated_experiments/`: Timestamped executed notebook outputs.
- `README.md`: This file.
- `experiment_*.log`: Timestamped experiment logs from V3 runs.

## Usage

### Running parameterized experiments (V4+, recommended)

Use the shared notebook harness launcher directly. Configs may use nested sections plus `EXTENDS`
inheritance; the notebook receives the resolved config path rather than a flattened papermill
parameter expansion.

```bash
cd /home/speediedan/repos/interpretune
source /mnt/cache/speediedan/.venvs/it_latest/bin/activate

# Run a single config
python tests/nb_experiment_harness/nb_experiment_launcher.py \
   --notebook tests/concept_direction_approach_parity/concept_direction_template.ipynb \
   tests/concept_direction_approach_parity/configs/gemma3_1b_it_local_capitals_states.yaml

# Run all configs in the configs/ directory
python tests/nb_experiment_harness/nb_experiment_launcher.py \
   --notebook tests/concept_direction_approach_parity/concept_direction_template.ipynb --all-configs

# Run only the PT capitals/states wave
python tests/nb_experiment_harness/nb_experiment_launcher.py \
    --notebook tests/concept_direction_approach_parity/concept_direction_template.ipynb \
   --config-pattern '.*pt_capitals_states\\.yaml'

# Prepare only (copy notebook + archive config, don't execute)
python tests/nb_experiment_harness/nb_experiment_launcher.py \
   --notebook tests/concept_direction_approach_parity/concept_direction_template.ipynb --all-configs --prepare-only

# Custom timeout and kernel
python tests/nb_experiment_harness/nb_experiment_launcher.py \
   --notebook tests/concept_direction_approach_parity/concept_direction_template.ipynb \
    tests/concept_direction_approach_parity/configs/gemma3_1b_it_local_capitals_states.yaml \
  --timeout 3600 --kernel-name it_latest
```

Each run writes timestamped artifacts under `generated_experiments/`, including the executed
notebook plus `.source.yaml` and `.resolved.yaml` snapshots of the config used for that run.

### Historical V3 notes

The standalone V3 experimentation script has been retired.
Historical analysis notes remain under `archived_analysis/`.

## Comparison Metrics

1. **Cosine similarity** of the direction vectors (0.05–0.24 across experiments)
2. **Feature Jaccard**: Overlap of top-N attributed features from the graph (~0.18–0.25)
3. **Pre/post intervention gap delta**: Change in concept token logit gap
4. **Key-token logit analysis**: Per-token pre/post logit and rank changes (V3)
5. **Direction reversal test**: Negated vs normal direction comparison (V3)
6. **Latent state sanity checks**: Norms, finiteness, within- vs between-group cosines
