# Interpretune Intervention APIs

This note captures the current intervention split between model-level intervention mechanisms and interpretune's
analysis-level intervention ops.

## Current surfaces

There are two distinct model-facing surfaces (a high-level tour with the full control matrix lives in
`tests/nb_experiments/intervention_capabilities_overview.md`):

- `ModelBackend.fwd_w_intervention(...)` is the shared model-level API for **hook-tensor ("embed"-path)
  interventions**, implemented with identical last-token math by both the TransformerLens and NNsight model
  backends (including SAE/latent sub-hook targets via `use_latent_models`/`sae_handles`).
- `ReplacementModel.feature_intervention(...)` is the model-level API for **circuit-tracer feature ("store"-path)
  interventions**. Both circuit-tracer backend implementations consume canonical intervention tuples of the form
  `(layer, position, feature_idx, value)` and support overlapping passthrough controls such as constrained layers,
  sparse activation capture, optional activation return, and backend-specific execution details.

At this layer, interpretune is delegating to backend-native steering surfaces rather than inventing a second execution
mechanism. The closest analogs are TransformerLens hook-driven intervention flows such as `run_with_hooks(...)` and
NNsight tracing/model-steering patterns that mutate activations inside a trace.

`feature_intervention_forward` is the current interpretune analysis-level API. It consumes `top_feature_ids` plus
value inputs from `AnalysisStore`, constructs canonical intervention tuples, runs a clean forward pass and an
intervened forward pass through `module.replacement_model`, and stores Arrow-safe intervention summaries back into
`AnalysisStore`.

`model_fwd_intervention` is the current hook-level API for direct tensor interventions. It accepts either explicit
`interventions` / `interventions_json` mappings or the shorthand `intervention_hook_pattern`, `intervention_mode`,
`intervention_scale_factor`, and `intervention_use_intervention_tensor_as_basis` fields. Explicit mappings take
precedence. Concept-direction notebook experiments now use this surface for direct-projection phases, including
configurations that inject the computed `concept_direction` as the `intervention_tensor` for a non-default hook such
as `blocks.0.hook_in` in `project` mode.

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

## Planned config split (aspirational — NOT yet implemented)

As of 2026-07-11, `CircuitTracerConfig` still keeps all `intervention_*` knobs at the top level; the split
below remains the target design, not the current state.

The current `CircuitTracerConfig` still mixes shared model-level intervention knobs with circuit-tracer-specific
analysis-level feature-intervention settings by keeping them all at the top level behind `intervention_` prefixes.
The next refactor should split that surface into:

- a shared model-level intervention config dataclass that can be reused by `model_fwd_intervention`,
	`resolve_interventions(...)`, and both the TransformerLens and NNsight model backends
- a circuit-tracer analysis-level intervention config dataclass for feature-selection and
	`ReplacementModel.feature_intervention(...)`-specific controls

### Model-level intervention config candidates

These options are backend-agnostic because they describe how a resolved hook intervention should be applied after the
hook target is known, regardless of whether execution happens through TransformerLens or NNsight:

- `hook_pattern` or `hook_patterns`: the user-facing hook selector before wildcard expansion
- `interventions`: explicit resolved-or-resolvable intervention mapping payloads
- `mode`: canonical replacement/add/project selector aligned with `InterventionSpec.mode`
- `scale_factor`: canonical scalar aligned with `InterventionSpec.scale_factor`
- `use_intervention_tensor_as_basis`: canonical projection-basis selector aligned with
	`InterventionSpec.use_intervention_tensor_as_basis`
- `intervention_tensor`: optional shorthand tensor when the caller is not supplying an explicit mapping

The important canonicalization rule is that the shared dataclass should prefer the non-prefixed field names above.
For example, `use_intervention_tensor_as_basis` should become the single canonical config name in the shared
model-level dataclass rather than preserving both `use_intervention_tensor_as_basis` and
`intervention_use_intervention_tensor_as_basis` as parallel long-term surfaces. Any legacy prefixed spellings should
be normalized at the op/config boundary rather than carried deeper into backend execution code.

### Circuit-tracer analysis-level config candidates

These options are not general hook-intervention controls. They are specific to the analysis-layer feature-selection
and `ReplacementModel.feature_intervention(...)` flow and should live in a circuit-tracer-specific intervention config
object instead of the shared model-level one:

- `value` and `value_source`
- `sign_aware_scale`
- `max_influence_norm_scale`
- `constrained_layers`
- `freeze_attention`
- `apply_activation_function`
- `sparse`
- `return_activations`

This split keeps the shared model-backend contract focused on `InterventionSpec`-style hook semantics while letting
the circuit-tracer analysis backend own its feature-intervention execution details.

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
