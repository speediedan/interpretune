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

## Current storage contract

The analysis op stores both a compact JSON payload and primitive Arrow-safe columns:

- `intervention_config`: JSON-serialized config summary
- `intervention_specs_json`: JSON-serialized canonical tuple payload
- `intervention_layers`, `intervention_positions`, `intervention_feature_ids`, `intervention_values`: primitive summary columns for downstream filtering and inspection
- `pre_intervention_logits`, `post_intervention_logits`, `logit_diff`: forward-only comparison outputs

The circuit-tracer analysis backend hydrates `intervention_specs_json` back into canonical tuple lists when an
`AnalysisStore` row or batch is formatted with `analysis_backend=DEFAULT_CT_ANALYSIS_BACKEND`.

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
