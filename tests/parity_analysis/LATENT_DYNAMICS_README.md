# Latent Dynamics Generated Artifacts

This directory stores timestamped executed outputs for the latent-dynamics notebook at
`tests/parity_analysis/concept_direction_latent_dynamics_analysis.ipynb`.

Current policy:

- Treat generated notebooks here as disposable execution artifacts, not source notebooks.
- Keep only successful runs that correspond to the current base notebook and current report schema.
- Remove stale papermill failures rather than preserving them as reference artifacts.

To regenerate the orange latent-dynamics artifact, use the latent-dynamics launcher. The only required
argument is the experiment config YAML (or its basename relative to `tests/parity_analysis/configs`). The
launcher defaults to `PROJECTION_METHOD=umap`, calculates the timestamped output filename automatically,
and writes the notebook plus `.source.yaml` / `.resolved.yaml` companions into
`tests/parity_analysis/generated_experiments/`.

```bash
python tests/parity_analysis/latent_dynamics_launcher.py \
  gemma3_1b_it_latent_dynamics_color_fruit_orange_fs_l10_n5.yaml
```

Optional overrides map directly to the current papermill parameters and execution options:

```bash
python tests/parity_analysis/latent_dynamics_launcher.py \
  gemma3_1b_it_latent_dynamics_color_fruit_orange_fs_l10_n5.yaml \
  --projection-method pca \
  --stage-top-n 7 \
  --output-dir tests/parity_analysis/generated_experiments \
  --timeout 7200 \
  --kernel-name python3
```

When the base notebook or the latent report payload changes, regenerate the artifact instead of trying to keep
older generated notebooks in sync manually.
