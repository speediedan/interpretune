# Scalable Dashboards (Wave 1): reviewer benchmark evidence, `full_20260829`

> Links here open in the same tab: GitHub's markdown sanitizer strips `target="_blank"`
> (verified, only `rel="nofollow"` survives), so new-tab behavior cannot be set from the source.
> Ctrl/Cmd-click to open alongside.

Regenerable evidence backing the performance claims in the coordinated Wave 1 PR set
([tracking umbrella](https://github.com/speediedan/interpretune/issues/231)).

| Artifact | What it is |
| --- | --- |
| **[benchmark_summary.md](benchmark_summary.md)** | **Start here.** All summary tables: per-scenario primary benchmark, substage timings, DB-import substage, resource peaks, activation-row parity, n-prompt scaling, rendered natively by GitHub. |
| [dashboard_benchmark_diagram.mmd](dashboard_benchmark_diagram.mmd) | Unified flow diagram (Mermaid; GitHub renders it inline). |
| [manifest.json](manifest.json) | Exact four-repo lineage, invocation, GPU/device metadata. The suite refuses to package with dirty repos. |
| `tables/`, [extracted_data.json](extracted_data.json) | Extracted markdown tables and the raw extraction the charts are plotted from. |
| **[Executed profiling notebook (renders in-browser)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/speediedan/interpretune/main/docs/benchmark_artifacts/full_20260829/dashboard_profiling_20260829_171106.html)** | Charts and data only, code cells hidden. Opens rendered via htmlpreview; the [raw file](dashboard_profiling_20260829_171106.html) is also committed here. |
| Complete package incl. per-leg raw profiling output | `.tar.gz` on the same [release](https://github.com/speediedan/interpretune/releases/tag/scalable-dashboards-wave1-evidence-20260829) (roughly 25 MB of raw output, regenerable rather than committed). |

## Environment

RTX 4090 (24 GiB), CUDA 13 / torch 2.13.0+cu130, gemma-3-1b-it, 262k-width transcoders. Lineage
`SD-7a0266f / SL-v6.49.1 / NP-9c043341 / IT-c573842`, zero dirty repos. All 17 legs reached their
batch target with no leg warnings.

`SL-v6.49.1` is a version, not a commit, because sae-lens resolves to the released wheel rather
than a checkout: the fork pin was retired in #241. Stamping the release is the honest record, since
stamping a retired fork's HEAD would name code this run never imported.

**These are consumer-GPU numbers.** The columnar lane is GPU-batched, so its advantage grows with the
feature axis and with available VRAM; larger devices should exceed what is measured here. Only basic
multi-GPU generation has been explored so far; see the
[multi-GPU generation notes](https://interpretune.readthedocs.io/en/latest/usage/neuronpedia_dashboard_pipeline.html#basic-multi-gpu-generation-scope-example-configuration-and-limitations).

## Reproducing

`scripts/setup_dashboard_benchmark_env.py` prepares everything (four repos, preserved-baseline
worktrees, integrated venv, prompt datasets) in one guided, non-destructive command; then
`scripts/run_dashboard_benchmark_suite.py --mode full`. See
[dashboard_benchmark_suite_usage.md](../../../scripts/dashboard_benchmark_suite_usage.md) and the
reproduction quickstart in the [umbrella](https://github.com/speediedan/interpretune/issues/231).

## Relationship to the retired `full_20260812` package

This package supersedes `full_20260812`, which is retired from the tree once the PR citations move
here. Both measured the same code paths on the same hardware; this one re-measures them at
`IT-c573842` on sae-lens 6.49.1.

Two things to know when comparing against its published numbers:

1. **The headline ratio moved up, and the numerator is why.** Monology 4096x256 reads 24.6x here
   (25,334 vs 1,030 f/min) against 22.9x before (23,906 vs 1,042). The legacy denominator is flat
   between runs (down 1.2%); the columnar numerator is 6.0% faster. That is run-to-run variance of
   the same order its predecessor documented in the opposite direction, when 24.3x moved to 22.9x on
   a roughly 5% columnar swing. **Treat the columnar figure as carrying variance of that order**
   rather than reading either move as a change in the code.
2. **The RTE headline shape changed, deliberately.** Earlier bodies led with RTE 512x128, the
   smallest and weakest shape in the sweep, which understated the lane by more than 3x. The RTE
   headline is now 2048x128 at 12.4x (10,748 vs 866 f/min). 512x128 is still reported, and remains
   the acceptance and parity shape, but it is no longer quoted as the headline.

Activation-row parity is unchanged in character: detached-vs-current legacy rows match within
0.25% at the largest shape, and the preserved-baseline comparison is reported per scenario in
`tables/parity_*.md`.
