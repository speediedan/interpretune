# Interpretune Intervention APIs

Interpretune intervenes on two different objects, at two levels, and every name in this area says which.

## Two levels, one rule

| Level | Capability member | Object acted on | Entry op | Executed by | Support record |
|---|---|---|---|---|---|
| Model backend | `ModelBackendCapability.ACTIVATION_INTERVENTION` | a residual-stream tensor at a vocabulary point (the **embed path**) | `model_fwd_intervention` | `ModelBackend.fwd_w_intervention`, identical math on TransformerLens and NNsight | `InterventionSupport` (modes, position scopes) |
| Analysis backend | `AnalysisBackendCapability.FEATURE_INTERVENTION` | latent feature activations `(layer, position, feature_id) -> value` (the **store path**) | `feature_intervention_forward` | `ReplacementModel.feature_intervention` through `CircuitTracerAnalysisBackend` | `FeatureInterventionSupport` (value sources, constrainable layers, returned activations) |

The table in `src/it_examples/experiments/notebook/intervention_capabilities_overview.md` is the definition both
enums cite; this note is the API contract beneath it.

**The rule, stated once for both levels.** A capability member names a protocol surface (a `Supports*` group of
methods) and answers only whether the backend implements it. Which configurations of that surface the backend
honours is a typed support record on the protocol, never a member and never an implicit default: a backend that
implements `fwd_w_intervention` for `add` and `project` but not `patch` declares the surface and a record that
says so, and a gate refuses `patch` by name before anything runs. The same rule covers the surfaces that are not
interventions: `CaptureSupport` says which vocabulary points a model backend can capture, and
`AttributionGraphSupport` says what graph construction requires of the model (the modeling module's own eager
attention bound at the attention call site, a fact the configured `eager` does not establish because another
library can replace that function for the whole process).

**Basis is a configuration, not a capability.** Where an intervention is defined in a basis (the J-lens work's
paper basis versus the norm-aware folded basis; the two are not interchangeable and are not related by a
coefficient), the basis is a declared configuration of the intervention surface: stated on the op's spec, carried
by the backend's record as honoured or not, and refused by name when unstated or unhonoured. A silent default is
exactly the failure the results' non-interchangeability names.

## Current surfaces

- `ModelBackend.fwd_w_intervention(...)` is the model-level API for hook-tensor (embed-path) interventions,
  implemented with identical last-token math by both the TransformerLens and NNsight model backends (including
  SAE/latent sub-hook targets via `use_latent_models`/`sae_handles`). Its record, `InterventionSupport`, declares
  the modes and position scopes; an op declares what it needs through `required_intervention_modes` and
  `required_position_scopes`, and the gate compares the two before the op runs.
- `ReplacementModel.feature_intervention(...)` is the model-level primitive the analysis-level store path executes
  on. Both circuit-tracer backend implementations consume canonical intervention tuples of the form
  `(layer, position, feature_idx, value)` and support passthrough controls such as constrained layers, sparse
  activation capture and optional activation return. `CircuitTracerAnalysisBackend` resolves the settings and
  refuses, through `FeatureInterventionSupport`, a value source it does not honour, a constant source without a
  value, a layer constraint or an activation return it cannot provide.

At this layer interpretune delegates to backend-native steering surfaces rather than inventing a second execution
mechanism. The closest analogs are TransformerLens hook-driven flows such as `run_with_hooks(...)` and NNsight
tracing that mutates activations inside a trace.

`feature_intervention_forward` is the analysis-level op. It consumes `top_feature_ids` plus value inputs from
`AnalysisStore`, constructs canonical intervention tuples, runs a clean forward pass and an intervened forward pass
through `module.replacement_model`, and stores Arrow-safe intervention summaries back into `AnalysisStore`.

`model_fwd_intervention` is the model-level op for direct tensor interventions. It accepts either explicit
`interventions` / `interventions_json` mappings or the shorthand `intervention_hook_pattern`, `intervention_mode`,
`intervention_scale_factor` and `intervention_use_intervention_tensor_as_basis` fields; explicit mappings take
precedence. Concept-direction notebook experiments use this surface for direct-projection phases, including
configurations that inject the computed `concept_direction` as the `intervention_tensor` for a non-default hook
such as `blocks.0.hook_in` in `project` mode.

`model_fwd_intervention` dispatches four modes: `replace`, `add`, `project`, and `patch`. The `patch` mode (the
J-space write) takes exactly two direction vectors stacked on a leading axis and swaps the activation's
coordinates along that pair, leaving the component orthogonal to the pair untouched. Mechanically it is
basis-agnostic: any pair of directions can serve as the patch pair, so embed-basis concept poles are as valid a
source as Jacobian lens rows; which basis produced the pair is the configuration the paragraph above requires the
op to state.

> **The pair and the model must share a residual basis, and TransformerLens weight processing changes it.**
> `HookedTransformer.from_pretrained` defaults to folding LayerNorm and centering weights, which rewrites the
> residual stream's geometry at every `hook_resid_*` point: a patch pair built from unprocessed weights (which is
> what lens artifacts and the circuit-tracer `ReplacementModel` path use) then swaps in the wrong plane. Measured
> on gpt2: the same pair's intervention delta disagrees between backends by a **0.948 relative gap** under default
> processing, versus <10% with `from_pretrained_no_processing`. Load unprocessed when applying patch pairs on TL, or
> build the pair from the processed model's own weights; never mix. (Pinned by
> `tests/core/test_jlens_patch_validation.py`.)

## Op-level entry points and composites

The registered analysis ops (all callable as `it.<name>(...)`):

- `concept_direction` (alias `semantic_direction`): builds a normalized direction from store latents
  (modes `mean_difference` / `paired_rejection` / `single_group`; `streaming` or `in_memory` aggregation)
  with an embed-difference fallback when no latent rows exist.
- `compute_attribution_graph` (`ct_graph`), `graph_node_influence` (`ct_node_influence`),
  `extract_top_features` (`ct_top_features`), `feature_intervention_forward` (`ct_feature_intervention`),
  `model_fwd_intervention` (`direction_intervention` / `direct_concept_direction_intervention`).
- Composites: `attribution_from_concept` (direction -> graph -> influence -> top features),
  `intervention_from_features`, and `intervention_from_concept` (the full five-op pipeline).

## Feature selection (`FeatureSelectionSpec`)

`extract_top_features` accepts a structured `FeatureSelectionSpec` with OR-semantics filters (`layers`,
`positions`, `feature_ids`, `layer_slice`, `position_slice`, `triples`, `layer_feature_pairs`), ranking
controls (`score_source` — `influence` / `signed_influence`, `gradient`-family planned; `score_sign` —
`any` / `positive` / `negative`; `rank_by_abs`), and `activation_overrides` for pinned per-feature
activation values. This is the sign-aware selection surface exercised by the orange `fs_l10_n5` lineage.

## Active store-path scaling behavior

These are ACTIVE runtime controls (not just config candidates): `intervention_value_source`
(`top_feature_scores` | `top_feature_activation_values` | `constant`), `intervention_scale_factor`,
`intervention_max_influence_norm_scale` (per-feature `abs(score)/max(abs(score))` amplification), and
`intervention_sign_aware_scale` (default on). The pinned combined formula (verified by
`test_analysis_backend_parity_feature_intervention_wrapper_sign_aware_top5_any_scaling`) is
`value = sign(score) * abs(activation) * (scale_factor * abs(score)/max_abs_score)`.

## Config split: what exists and what is still planned

`FeatureInterventionSupport` is now the analysis-level configuration space as a record: value sources, whether
layers can be constrained, whether activations can be returned. What is still planned is the split of the
`CircuitTracerConfig` fields themselves, which keep both levels' knobs at the top level behind `intervention_`
prefixes:

- a shared model-level intervention config dataclass reusable by `model_fwd_intervention`,
  `resolve_interventions(...)` and both model backends: `hook_pattern(s)`, `interventions`, `mode`, `scale_factor`,
  `use_intervention_tensor_as_basis`, `intervention_tensor`, and the basis where one applies, with the
  non-prefixed spellings canonical and legacy prefixed spellings normalized at the op boundary;
- a circuit-tracer analysis-level intervention config dataclass for `value`, `value_source`, `sign_aware_scale`,
  `max_influence_norm_scale`, `constrained_layers`, `freeze_attention`, `apply_activation_function`, `sparse` and
  `return_activations`, the fields the record already describes.

Until that lands, `resolve_feature_intervention_settings` reads the prefixed fields and the record is the check.

## Hook pattern contract

Prefer canonical TransformerBridge-style hook names in new intervention configs, even when the current backend still
accepts older HookedTransformer aliases. The current portable subset and its legacy aliases are documented in
[intervention_hook_pattern_support.md](intervention_hook_pattern_support.md).

In practice this means new configs should prefer names such as `blocks.{i}.hook_in`, `blocks.{i}.hook_out`,
`blocks.{i}.attn.hook_out`, `blocks.{i}.attn.o.hook_in`, and `unembed.hook_in`. The intervention pattern expander now
tries supported canonical and legacy spellings in both directions before backend resolution.

## Direct projection config pattern

The concept-direction notebook configs now expose a notebook-facing `ANALYSIS.direct_projection` section. The most
expressive form mirrors `resolve_interventions(...)` by supplying an explicit intervention mapping whose values are
valid `InterventionSpec`-style payloads. The experiment wrapper injects the runtime `concept_direction` tensor when an
`intervention_tensor` is omitted.

Example:

```yaml
ANALYSIS:
    direct_projection:
        interventions:
            blocks.0.hook_in:
                mode: project
                scale_factor: 10.0
                use_intervention_tensor_as_basis: true
```

When an explicit `interventions` mapping is not supplied, the notebook wrapper falls back to the shorthand fields and
lets `resolve_interventions(...)` derive the final payload using its standard precedence rules.

## Current storage contract

The analysis op stores both a compact JSON payload and primitive Arrow-safe columns:

- `intervention_config`: JSON-serialized config summary
- `intervention_specs_json`: JSON-serialized canonical tuple payload
- `intervention_layers`, `intervention_positions`, `intervention_feature_ids`, `intervention_values`: primitive summary columns for downstream filtering and inspection
- `feature_intervention_dict` / `feature_intervention_dict_json`: hydrate-ready intervention mapping payloads
- `intervention_base_values`, `intervention_scale_factors`, `intervention_score_values`: per-feature scaling provenance
- `pre_intervention_logits`, `post_intervention_logits`, `logit_diff`: forward-only comparison outputs
- `intervention_activation_cache` (optional): captured activations when `return_activations` is enabled

The circuit-tracer analysis backend hydrates `intervention_specs_json` back into canonical tuple lists when an
`AnalysisStore` row or batch is formatted with `analysis_backend=DEFAULT_CT_ANALYSIS_BACKEND`.

## Missing-feature constrained selection

`extract_top_features` now supports constrained selections that refer to `(layer, feature_id)` pairs not present in
the original attribution graph rows. When a requested feature is missing, the op synthesizes candidate rows across the
observed positions for that layer, or across all observed positions when the layer has no active rows. Synthetic rows
inherit score baselines from same-layer rows when available and fall back to global means otherwise.

For activation-derived interventions, the op also carries forward optional activation overrides keyed by
`(layer, feature_id)`. When an override is not provided, synthetic rows use the mean activation of same-layer active
rows when available and the global mean activation otherwise. This lets `feature_intervention_forward` intervene on
requested features that were absent from the original graph without degrading to a zero activation heuristic.

When constrained feature selection is active, top-feature ranking now preserves at least one highest-scoring row per
requested `(layer, feature_id)` pair before returning the final top-feature payload. This prevents repeated positions
for one requested feature from crowding out a second requested feature before downstream intervention tuple
construction.

## Scope boundary

Interpretune currently exposes forward-only intervention analysis at the analysis-op layer.

`ReplacementModel.feature_intervention_generate(...)` remains a model-level API and is not yet exposed as a first-class
interpretune analysis op. That follow-up should land as a separate generation-oriented intervention task once the
forward-only storage, config, and validation story is stable.

## Testing guidance

Cross-backend parity for intervention behavior should follow the same resource-aware testing direction already used for other expensive analysis flows:

- prefer shared extraction/caching helpers from `tests/analysis_resource_utils.py`
- prefer focused parity comparisons over blanket standalone isolation
- reserve standalone-heavy GPU coverage for cases where backend/runtime constraints actually require it
