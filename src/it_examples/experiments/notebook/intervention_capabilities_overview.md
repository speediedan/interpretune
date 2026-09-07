# Intervention Capabilities Overview

**Date:** 2026-07-11 (Phase 7 / 7c amendments; see `EXPERIMENT_STATUS.md` "7c Amendments" §2);
J-space status refreshed 2026-09-01

A high-level map of interpretune's current intervention/steering surface as exercised by the
`concept_direction` experiment family and the circuit-tracer demo notebooks. Deeper reference:
`docs/interpretune_intervention_apis.md` (API contract), `docs/interpretune_intervention_apis.md`
cross-references, and the test anchors listed at the end.

## The two intervention paths at a glance

| | **Embed path** (hook-tensor interventions) | **Store path** (CT feature interventions) |
|---|---|---|
| What is adjusted | A residual-stream-space tensor added/projected/replaced at model hook points (last token) | Individual transcoder **feature activations** `(layer, position, feature_id) -> value` |
| Entry op | `model_fwd_intervention` (aliases `direction_intervention`, `direct_concept_direction_intervention`) | `feature_intervention_forward` (alias `ct_feature_intervention`) |
| Core primitive | `InterventionSpec` / `InterventionDict` (`interpretune.analysis.backends`) | canonical CT tuples built by `CircuitTracerAnalysisBackend.build_feature_interventions` |
| Executed by | `ModelBackend.fwd_w_intervention` — identical math on TransformerLens **and** NNsight backends | `ReplacementModel.feature_intervention` (circuit-tracer model) |
| Typical source tensor | `concept_direction` op output (store-latent or embed-difference) | `extract_top_features` output (attribution-ranked, sign-aware-selected) |

Both paths emit `pre_intervention_logits`, `post_intervention_logits`, and `logit_diff` into the
`AnalysisStore`, so steering effect sizes are directly comparable across paths.

## Embed-path controls (`InterventionSpec`)

- `mode`: `replace` | `add` (`input + tensor * scale_factor`) | `project` (project onto tensor span) |
  `patch` (lens-coordinate swap: takes a `(2, d_model)` pair of directions, reads the activation's
  coordinates along the pair via the pseudoinverse and exchanges them, `h <- h + V(sigma(c) - c)`,
  leaving the orthogonal component untouched; basis-agnostic, so J-lens rows and embed-basis
  concept poles are both valid pairs). The pair and the model must share a residual basis: see the
  TransformerLens weight-processing caveat in `docs/interpretune_intervention_apis.md`.
- `scale_factor`: amplification for `add`/`project`
- `use_intervention_tensor_as_basis`: basis-direction toggle for `project`
- Hook targeting: explicit `interventions` dicts or shorthand
  (`intervention_hook_pattern` + `intervention_mode`/`intervention_scale_factor`/...), with
  alias-aware pattern expansion (`blocks.{i}.hook_in` ↔ `hook_resid_pre`, `unembed.hook_in`, ...)
  and wildcard/per-hook tensor splitting. When no tensor is given the op falls back to the batch's
  `concept_direction` with mode `add`.
- SAE/latent sub-hook targets are supported via `use_latent_models`/`sae_handles`
  (act_input / hidden_pre / feature_acts / sae_error / sae_output sub-hooks on both backends).
- An op declares the configurations it needs through `required_intervention_modes` and
  `required_position_scopes`, checked against the backend's `InterventionSupport` record before
  anything runs. Without them an unsupported mode surfaces only when the backend refuses it, after
  every earlier op in a composition has already executed, and a published collection advertises
  nothing a consumer can check short of running it.

## Store-path controls (`CircuitTracerConfig.intervention_*` + per-call overrides)

- `intervention_value_source`: `top_feature_scores` | `top_feature_activation_values` | `constant`
  (with `intervention_value` for the constant case) — the per-feature base value.
- `intervention_scale_factor`: base scalar amplification.
- `intervention_max_influence_norm_scale`: per-feature amplification by
  `abs(score) / max(abs(score))` (influence-normalized scaling).
- `intervention_sign_aware_scale` (default on): the applied value carries the **sign of the
  feature's attribution score** — the pinned formula (verified by the orange s5_any test) is
  `value = sign(score) * abs(activation) * (scale_factor * abs(score)/max_abs_score)`.
- Execution knobs forwarded to circuit-tracer: `sparse`, `return_activations`,
  `constrained_layers`, `freeze_attention`, `apply_activation_function`.

## Feature selection (`FeatureSelectionSpec`)

Structured, sign-aware selection feeding `extract_top_features` (alias `ct_top_features`):

- **Filters** (OR semantics): `layers`, `positions`, `feature_ids`, `layer_slice`,
  `position_slice`, `triples` (`(layer, pos, fid)`), `layer_feature_pairs`.
- **Ranking**: `score_source` (`influence`, `signed_influence`; `gradient`-family planned),
  `score_sign` (`any` | `positive` | `negative`), `rank_by_abs`, top-n.
- `activation_overrides` for pinned per-feature activation values; constrained selection
  synthesizes rows for requested-but-missing features so pinned lineages remain runnable.

This is the machinery behind the orange `fs_l10_n5` lineage (layers ≥ 10, top-5, sign variants).

## Concept-direction pipeline (op composition)

`concept_direction` (alias `semantic_direction`) produces a normalized direction from either:
- **store latents** (answer-position latent-state differences; modes `mean_difference`,
  `paired_rejection`, `single_group`; `streaming` or `in_memory` aggregation), or
- **embed fallback** (token-group embedding differences) when no latent rows exist.

Registered composites chain the full flow:

- `attribution_from_concept` = `concept_direction . compute_attribution_graph .
  graph_node_influence . extract_top_features`
- `intervention_from_features` = `feature_intervention_forward`
- `intervention_from_concept` = the full five-op pipeline (concept direction → attribution graph →
  node influence → top features → feature intervention)

All are callable as `it.<name>(...)` via the lazy op dispatcher.

## Verified steering anchors (expected vs realized effects)

- `src/it_examples/tests/notebook/concept_direction/test_concept_direction_backend_parity.py::`
  `test_analysis_backend_parity_feature_intervention_wrapper_sign_aware_top5_any_scaling` — pins
  the sign-aware/max-norm scaling formula AND the realized steering outcome (post-intervention gap
  > pre-gap; post-intervention argmax lands in the target-token variant set) on the orange
  `fs_l10_n5_s5_any` config.
- The graph-edge expectation tests in the same file compare adjacency-predicted aggregate
  feature/logit effects against realized demeaned logit deltas (sign agreement asserted).
- `tests/core/test_model_backend_parity.py::TestDirectionInterventionBackendParity` — embed-path
  `model_fwd_intervention` parity across TL/NNsight (logits change, scale-factor monotonicity,
  pre/post/logit-diff cross-backend parity).

These anchors are why the `orange` (color-vs-fruit) example is the demo substrate of choice: its
expected-vs-realized activation effects are already CI-verified.

## Local explanations today, streamable shared dashboards next (Wave 2)

The local steering demo's flow (`ct_concept_steering_demo_local_np.ipynb`:
`feature_tuples_to_feature_refs` + `ensure_local_feature_explanations` with
`generate_missing=True`) currently writes explanations into
the **local Neuronpedia DB** only. The Wave 2 intention for this workstream is to migrate these
local-only explanations and custom dashboards to **user Hugging Face Hub uploaded/cached streamable
dashboards**, making them easily shareable across researchers without central Neuronpedia DB
modification (or consumer-side local-DB imports). Until then, treat locally generated explanations
as a maintainer/local-DB-developer capability (exercised by the steering demo's local mode and its
optional service-gated tests).

## Jacobian-space (J-lens) integration: shipped and remaining (2026-09-01)

The J-space technique (Gurnee et al. 2026, https://transformer-circuits.pub/2026/workspace/) is
tracked as [interpretune#225](https://github.com/speediedan/interpretune/issues/225) (the in-tree
half) and [interpretune#273](https://github.com/speediedan/interpretune/issues/273) (a separately
published, genuinely non-bundled op collection, `speediedan/jlens_steering_ops`, private until the
[#261](https://github.com/speediedan/interpretune/issues/261) public flip).

**Shipped (the write path):**

- `model_fwd_intervention` mode `patch` (above), validated three ways in
  `tests/core/test_jlens_patch_validation.py`: an eager-reference check of the traced mechanics, a
  float64 convergence check against the true per-prompt Jacobian-vector product, and a magnitude
  sweep (0.25 to 8.0) that pins where first-order behavior departs. No freezing is needed for the
  mechanics check; the module docstring records why.
- The collection's `jlens_concept_patch_pair` op builds the `(2, d_model)` J-lens pole pair for a
  concept token pair from the pre-fitted `neuronpedia/jacobian-lens` artifacts, folding the final
  norm's elementwise scale into the vectors (`v_c = (W_U[c] * s) @ J_l`, the direction the paper's
  own readout computes); the composite `jlens_patch_intervention` chains it into the bundled
  intervention op. Both steering demos carry a J-space section (4b) executed on GPU: the swap flips
  the orange example at scale 1.0 on gemma-2-2b (layer 24) and gemma-3-1b-it (layer 21), on both
  backends.
- `interpretune.analysis.optools.resolve_unembed_and_norm_scale`: the sanctioned seam for the
  unembed matrix and final-norm scale per model family (HF gemma `1 + weight`, other HF RMSNorms
  `weight`, TransformerLens `ln_final.w` as stored).
- The folding decision is measured, not assumed: folding is essential on gemma-3-1b-it (unfolded
  vectors never flip) and roughly 3x weaker at late layers on gemma-2-2b (both flip), because a
  uniform norm scale cancels exactly in patch mode and only its anisotropy, through its alignment
  with where the model stores the task contrast, can matter. `jlens_apply_final_norm` therefore
  stays a per-model option. The grounding is
  [interpretune#330](https://github.com/speediedan/interpretune/issues/330).

**Remaining (the read path and the integrations), all tracked on #225 unless noted:**

- A lens loader/config and the readout ops (`jlens_read`, `jlens_concept_probe`,
  `jlens_sparse_inventory`) behind the backend seam. Nothing in-tree can produce a J-lens readout
  today; the experimental probes in `concept_direction_analysis.md` wait on this.
- A `basis="jlens"` selector on `concept_direction` (there is no `basis=` selector today; embed and
  store are two parallel pipelines, and `use_answer_state_as_basis` is a different axis).
- Per-feature J-space signatures as the third profile in the Phase-6 decoupling tooling.
- The `clamp` and `reject` intervention modes (the paper's coordinate clamping and top-k J-space
  ablation).
- The embed-basis `patch` comparison on the Direct-hook path (concept-axis-only swap vs naive add).
- J-lens subspace attribution graphs
  ([#338](https://github.com/speediedan/interpretune/issues/338)) and level-3 validation of the
  production pair on the real gemma artifacts
  ([#339](https://github.com/speediedan/interpretune/issues/339)).
