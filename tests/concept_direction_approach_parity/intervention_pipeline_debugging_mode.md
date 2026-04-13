# Intervention Pipeline Debugging Mode

Use `ANALYSIS_MODE: debug_intervention_pipelines` when the goal is to validate one selected
feature against the attribution graph's predicted deltas rather than to run the full concept-pair
analysis notebook.

## Required Config Surface

- Set exactly one entry in `CONSTRAINED_FEATURE_SELECTION_LIST`.
- Keep `ENABLE_ZERO_SOFTCAP: true` for Gemma 3-style parity runs.
- Prefer `DEFAULT_SCALE_FACTOR: 2.0` unless you are intentionally probing a different intervention magnitude.
- Set `DEBUG_SESSION_SURFACE_PRESET: parity_surface` when you want the notebook debug run to mirror the standalone circuit-tracer validation surface.

Do not reuse that preset for the normal localhost concept-pair configs. The parity-aligned surface is a debug validation tool, not a generally better notebook surface, and non-debug OQI reruns regressed further when it was enabled.

## What `parity_surface` Changes

The preset is applied inside `experiment_resource_utils.py` and forwarded by `NotebookHarnessConfig.session_kwargs`.
When selected, it pushes the debug run toward the standalone parity surface by:

- preserving the configured `FORCE_DEVICE`, or the auto-selected CUDA device when `FORCE_DEVICE` is unset
- setting NNsight attention to `eager`
- using float32 for the NNsight and circuit-tracer debug path
- clearing default `analysis_target_tokens` and `target_token_ids`
- using CPU offload and quieter circuit-tracer logging
- moving debug `attribution_targets` onto the replacement-model device so the graph call matches the standalone helper

## Validation Path

`run_debug_intervention_validation()` replaces the notebook's normal late-phase analysis with a focused parity report.
That report compares:

- predicted activation deltas from the selected adjacency column
- observed post-intervention activation deltas
- predicted demeaned-logit deltas
- observed demeaned-logit deltas

## Guardrails

- `tests/parity_analysis/oqi_debug_session_ablation.py` is the manual parity probe for session-surface debugging.
- `tests/parity_analysis/test_oqi_debug_session_ablation.py` verifies that the ablation helper preserves `DEBUG_SESSION_SURFACE_PRESET` without silently reintroducing a notebook-only CPU override.
- Historical large validation errors such as `activation_max_abs_error=34.5` and `logit_max_abs_error=0.1855621337890625` came from runs that resolved to `notebook_default`, not from the corrected helper path.

## Operational Note

The parity surface is still more conservative than the default notebook surface because it switches the debug path to eager attention, float32, cleared target defaults, and CPU offload. It now preserves GPU execution when available, which is the preferred mode for reproducing the standalone OQI parity checks.