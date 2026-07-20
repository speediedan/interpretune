# Concept Direction Analysis Launcher

This directory contains the analysis notebook and launcher used to regenerate the concept-direction
comparison surface from a clean set of reference-test reports and notebook experiment outputs.

Treat generated notebooks under `/tmp/it_concept_direction_experiments/analysis/` as execution
artifacts. The source notebook is `concept_direction_analysis.ipynb`; timestamped outputs exist so a
completed analysis run can be inspected later without re-running long model jobs.

## Default Run

From the Interpretune repository root:

```bash
python tests/nb_experiments/concept_direction/analysis/concept_direction_analysis_launcher.py
```

By default, the launcher reads `default_analysis_experiment_set.yaml`, archives any existing artifact
root at `/tmp/it_concept_direction_analysis_artifacts`, regenerates reference-test reports, runs the
configured notebook experiments, then executes `concept_direction_analysis.ipynb` into
`/tmp/it_concept_direction_experiments/analysis/`.

The default manifest currently controls:

- `reference_tests`: pytest node IDs for the reference/parity reports that must be generated before
  notebook execution.
- `notebook_configs`: experiment config YAML files passed to `tests/nb_experiments/nb_experiment_launcher.py`.
- `notebook_path`, `config_dir`, and `notebook_output_dir`: where experiment notebooks are read from
  and written to.
- `analysis_notebook`, `analysis_output_dir`, and `analysis_output_stem`: the final analysis notebook
  source and timestamped output naming.

## Common Modes

Use a custom manifest when narrowing the analysis surface:

```bash
python tests/nb_experiments/concept_direction/analysis/concept_direction_analysis_launcher.py \
  --experiment-set tests/nb_experiments/concept_direction/analysis/default_analysis_experiment_set.yaml
```

Reuse an existing artifact root without re-running tests or experiment notebooks:

```bash
python tests/nb_experiments/concept_direction/analysis/concept_direction_analysis_launcher.py \
  --use-existing \
  --artifact-root /tmp/it_concept_direction_analysis_artifacts
```

Prepare reference and experiment artifacts but skip the final analysis notebook:

```bash
python tests/nb_experiments/concept_direction/analysis/concept_direction_analysis_launcher.py \
  --skip-analysis-notebook
```

Render experiment notebooks only, without executing their model cells:

```bash
python tests/nb_experiments/concept_direction/analysis/concept_direction_analysis_launcher.py \
  --prepare-only
```

Long runs can override papermill execution controls:

```bash
python tests/nb_experiments/concept_direction/analysis/concept_direction_analysis_launcher.py \
  --timeout 7200 \
  --kernel-name python3
```

## Outputs

The launcher sets `IT_CONCEPT_DIRECTION_PARITY_OUTPUT_DIR` so reference reports are written under the
selected artifact root instead of the older standalone default. It also sets
`CONCEPT_DIRECTION_ANALYSIS_GENERATION`, which lets the analysis notebook prefer reports from the
current generation rather than stale reports from prior runs.

Experiment notebooks are written under `/tmp/it_concept_direction_experiments/` unless the manifest
overrides `notebook_output_dir`. The final executed analysis notebook is written under
`/tmp/it_concept_direction_experiments/analysis/` with a timestamped
`concept_direction_analysis_*.ipynb` filename.

Use `--use-existing` only when the artifact root is known to contain a coherent reference-test and
notebook-experiment set. For a new comparison pass, prefer the default fresh run so stale artifacts do
not silently influence the analysis.
