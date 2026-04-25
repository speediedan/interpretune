# Intervention Graph Parity Testing

This note covers the manual workflow for debugging graph/intervention mismatches between the notebook debug path and the standalone circuit-tracer parity tests.

## Primary Tools

- `tests/nb_experiments/concept_direction/analysis/oqi_debug_session_ablation.py`: runs the notebook-side debug validation under targeted session-surface overrides and compares the result against preserved standalone artifacts.
- `tests/nb_experiments/concept_direction/analysis/intervention_drift_analysis.py`: fingerprints tensors and summarizes adjacency, baseline-logit, and activation-cache drift.
- `tests/nb_experiments/concept_direction/analysis/intervention_drift_analysis.ipynb`: interactive follow-up notebook for inspecting preserved artifacts.

## Preserve Artifacts First

When comparing surfaces, preserve artifacts from the run you trust before changing anything else.
The current scripts use:

- `IT_PARITY_PRESERVE_ARTIFACTS=1`
- `IT_PARITY_PRESERVE_ARTIFACT_DIR=tests/nb_experiments/concept_direction/analysis/artifacts`

These artifacts make it possible to compare:

- graph input tokens
- selected feature rows
- adjacency matrices
- baseline logits
- baseline activation caches

## Recommended Workflow

1. Start from the preserved debug YAML config under `tests/nb_experiments/concept_direction/archived_cfgs/`.
2. Confirm the config builder preserves `DEBUG_SESSION_SURFACE_PRESET` without forcing a notebook-only CPU fallback when `FORCE_DEVICE` is unset.
3. Reproduce the notebook path with the exact launcher command before changing presets by hand.
4. Compare the pre-intervention surface first: input tokens, graph targets, selected feature set, adjacency matrix, baseline logits, and baseline activation cache.
5. Only inspect intervention deltas after the pre-intervention surface matches closely enough to rule out session-construction drift.

The current OQI GPU validation command is:

```bash
python tests/nb_experiments/nb_experiment_launcher.py \
	--notebook tests/nb_experiments/concept_direction/concept_direction_template.ipynb \
	tests/nb_experiments/concept_direction/archived_cfgs/gemma3_4b_it_local_oqi_reasoning_single_fs_di_60.yaml
```

## Current Regression Guard

`_build_notebook_cfg()` in `oqi_debug_session_ablation.py` now forwards `DEBUG_SESSION_SURFACE_PRESET` into `NotebookHarnessConfig`.
That prevents manual ablation runs from silently reverting to `notebook_default` or reintroducing the old notebook-only CPU override.
The focused regression test lives in `tests/nb_experiments/concept_direction/analysis/test_oqi_debug_session_ablation.py`.
