# Interpretune Intervention APIs

This note captures the current intervention split between model-level intervention mechanisms and interpretune's
analysis-level intervention ops.

## Current surfaces

`ReplacementModel.feature_intervention(...)` is the current model-level API shared by the upstream TransformerLens and
NNsight circuit-tracer backends. Both implementations consume canonical intervention tuples of the form
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
- `pre_intervention_logits`, `post_intervention_logits`, `logit_diff`: forward-only comparison outputs

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
