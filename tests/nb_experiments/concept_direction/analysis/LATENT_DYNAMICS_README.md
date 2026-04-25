# Latent Dynamics Generated Artifacts

This directory stores timestamped executed outputs for the latent-dynamics notebook at
`tests/nb_experiments/concept_direction/analysis/concept_direction_latent_dynamics_analysis.ipynb`.

Current policy:

- Treat generated notebooks here as disposable execution artifacts, not source notebooks.
- Keep only successful runs that correspond to the current base notebook and current report schema.
- Remove stale papermill failures rather than preserving them as reference artifacts.

Implementation guardrail:

- The latent-dynamics helpers intentionally mirror parts of the production concept-direction op logic in
  `src/interpretune/analysis/ops/definitions.py`. In particular, keep
  `concept_direction_impl` aligned with the local `concept_direction_latent_dynamics.py` direction builders,
  including `_paired_rejection_payload`. If this module remains as a long-lived analysis surface, add and
  maintain rigorous parity tests so exploratory reporting code does not drift away from the op contract.

To regenerate the orange latent-dynamics artifact, use the latent-dynamics launcher. The only required
argument is the experiment config YAML (or its basename relative to `tests/nb_experiments/concept_direction/analysis/configs`). The
launcher defaults to `PROJECTION_METHOD=umap`, calculates the timestamped output filename automatically,
and writes the notebook plus `.source.yaml` / `.resolved.yaml` companions into
`/tmp/it_concept_direction_experiments/analysis/`.

Store-context-enhanced trajectory notes:

- `projected_context_state_mean_difference_direction` is the direction built from the context-enhanced projected states.
- `store_context_enhanced_paired_rejection_unnormalized_direction` is the raw paired-rejection residual before normalization.
- `store_context_enhanced_paired_rejection_direction` is the normalized direction actually compared against the other stage directions.
- `store_context_enhanced_paired_rejection_reconstruction_direction` is the reconstruction-space variant derived from the same store-context-enhanced snapshot.
- In the per-example records, `selected_store_state` is the state actually used for the store trajectory. When a valid context token exists, `selected_store_state_source` is `projected_context_state`; otherwise it falls back to the answer-position state.

```bash
python tests/nb_experiments/concept_direction/analysis/latent_dynamics_launcher.py \
  gemma3_1b_it_latent_dynamics_color_fruit_orange_fs_l10_n5.yaml
```

Optional overrides map directly to the current papermill parameters and execution options:

```bash
python tests/nb_experiments/concept_direction/analysis/latent_dynamics_launcher.py \
  gemma3_1b_it_latent_dynamics_color_fruit_orange_fs_l10_n5.yaml \
  --projection-method pca \
  --stage-top-n 7 \
  --output-dir /tmp/it_concept_direction_experiments/analysis \
  --timeout 7200 \
  --kernel-name python3
```

When the base notebook or the latent report payload changes, regenerate the artifact instead of trying to keep
older generated notebooks in sync manually.
