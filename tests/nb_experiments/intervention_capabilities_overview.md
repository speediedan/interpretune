# Intervention Capabilities Overview

**Date:** 2026-07-11 (Phase 7 / 7c amendments — see `EXPERIMENT_STATUS.md` "7c Amendments" §2)

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

- `mode`: `replace` | `add` (`input + tensor * scale_factor`) | `project` (project onto tensor span)
- `scale_factor`: amplification for `add`/`project`
- `use_intervention_tensor_as_basis`: basis-direction toggle for `project`
- Hook targeting: explicit `interventions` dicts or shorthand
  (`intervention_hook_pattern` + `intervention_mode`/`intervention_scale_factor`/...), with
  alias-aware pattern expansion (`blocks.{i}.hook_in` ↔ `hook_resid_pre`, `unembed.hook_in`, ...)
  and wildcard/per-hook tensor splitting. When no tensor is given the op falls back to the batch's
  `concept_direction` with mode `add`.
- SAE/latent sub-hook targets are supported via `use_latent_models`/`sae_handles`
  (act_input / hidden_pre / feature_acts / sae_error / sae_output sub-hooks on both backends).

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

- `tests/nb_experiments/concept_direction/test_concept_direction_backend_parity.py::`
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

The steering demo's `DASHBOARD_MODE="local"` flow (`feature_tuples_to_feature_refs` +
`ensure_local_feature_explanations` with `generate_missing=True`) currently writes explanations into
the **local Neuronpedia DB** only. The Wave 2 intention for this workstream is to migrate these
local-only explanations and custom dashboards to **user Hugging Face Hub uploaded/cached streamable
dashboards**, making them easily shareable across researchers without central Neuronpedia DB
modification (or consumer-side local-DB imports). Until then, treat locally generated explanations
as a maintainer/local-DB-developer capability (exercised by the steering demo's local mode and its
optional service-gated tests).

## Planned: Jacobian-space (J-lens) integration

Jacobian-lens support (read/probe/sparse-inventory ops, a `jlens` concept-direction basis,
lens-coordinate `patch` interventions alongside `add`/`project`, per-feature J-space signatures in
the decoupling tooling, and shareable J-space artifacts co-designed with AnalysisStore hub support)
is scoped and tracked in [interpretune#225](https://github.com/speediedan/interpretune/issues/225)
(post-PR-wave, Wave-2-adjacent; pairs with
[interpretune#124](https://github.com/speediedan/interpretune/issues/124)).
