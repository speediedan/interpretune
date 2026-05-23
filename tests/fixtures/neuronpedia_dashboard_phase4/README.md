# Phase 4 Neuronpedia Dashboard Baseline Fixtures

These fixtures preserve the detached-baseline artifact and import contract used to validate the current-code
`legacy_json_cpu` compatibility path. The active baseline is the latest published Phase 4 detached-baseline row from
`distributed-insight/project_admin/PRs/neuronpedia/source-set-enhancements/phase4_summary_results_for_pr.md`:

- RTE: `SD-7886eaa/SL-3eea6552/NP-5a33f17/IT-fc69a14`, `65129` imported `Activation` rows,
  `33.42s` average batch time, `919.2` features/min, and `48.78s` import wall time.
- Monology: `SD-7886eaa/SL-3eea6552/NP-5a33f17/IT-fc69a14`, `137331` imported `Activation` rows,
  `45.82s` average batch time, `1341.0` features/min, and `61.37s` import wall time.

## When To Regenerate

Regenerate `import_tolerance_baselines.json` when one of these changes intentionally updates the detached preserved
baseline contract:

- the preserved SAEDashboard baseline commit changes from `7886eaa`;
- the preserved Neuronpedia importer/exporter baseline commit changes from `5a33f17`;
- the preserved SAELens dependency baseline changes from `3eea6552`;
- the Interpretune harness commit used to run the detached baseline changes and produces a new published row in
  `phase4_summary_results_for_pr.md`;
- the retained prompt artifacts, layer/batch bounds, or import mode for the detached baseline change.

Do not regenerate this fixture from current-code legacy or `lazy_gpu` rows. Those are candidates compared against this
baseline, not the baseline itself.

## Regeneration Process

1. Run the detached preserved-baseline RTE and Monology reduced presets through the Interpretune profiling harness using
   the exact layer-9 bounded commands recorded in `phase4_summary_results_for_pr.md`.
2. Confirm the run lineages. The SAEDashboard, SAELens, and Neuronpedia commits must be the intended detached baseline
   commits; the Interpretune commit should be the current harness commit used for the published rerun.
3. Copy the published `avg_batch_seconds`, `throughput_features_per_min`, `import_wall_seconds`,
   `imported_activation_rows`, and `session_root` values into `import_tolerance_baselines.json`.
4. Keep `activation_rows_max_abs_delta` at `0` for detached baseline versus current-code legacy unless a documented
   upstream-only behavior change explicitly changes the old path contract.
5. Mirror the same contract values into the SAEDashboard and Neuronpedia upstream fixture manifests used by their
   preserved-baseline contract tests.
6. Run the focused fixture/contract tests before committing:

```bash
cd /home/speediedan/repos/interpretune
source /mnt/cache/speediedan/.venvs/it_latest/bin/activate
python -m pytest tests/core/test_neuronpedia_dashboard_pipeline.py -k phase4_import_tolerance_fixture -q
```
