# Scalable Dashboards (Wave 1) — reviewer benchmark evidence, `full_20260812`

> Links here open in the same tab: GitHub's markdown sanitizer strips `target="_blank"`
> (verified — only `rel="nofollow"` survives), so new-tab behavior cannot be set from the source.
> Ctrl/Cmd-click to open alongside.

Regenerable evidence backing the performance claims in the coordinated Wave 1 PR set
([tracking umbrella](https://github.com/speediedan/interpretune/issues/231)).

| Artifact | What it is |
| --- | --- |
| **[benchmark_summary.md](benchmark_summary.md)** | **Start here.** All summary tables — per-scenario primary benchmark, substage timings, DB-import substage, resource peaks, activation-row parity, n-prompt scaling — rendered natively by GitHub. |
| [dashboard_benchmark_diagram.mmd](dashboard_benchmark_diagram.mmd) | Unified flow diagram (Mermaid; GitHub renders it inline). |
| [manifest.json](manifest.json) | Exact four-repo lineage, invocation, GPU/device metadata. The suite refuses to package with dirty repos. |
| `tables/`, [extracted_data.json](extracted_data.json) | Extracted markdown tables and the raw extraction the charts are plotted from. |
| **[Executed profiling notebook (renders in-browser)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/speediedan/interpretune/main/docs/benchmark_artifacts/full_20260812/dashboard_profiling_20260813_051303.html)** | Charts and data only, code cells hidden. Opens rendered via htmlpreview; the [raw file](dashboard_profiling_20260813_051303.html) is also committed here, and the same HTML is attached to the release for offline use. |
| Complete package incl. per-leg raw profiling output | `.tar.gz` on the same [release](https://github.com/speediedan/interpretune/releases/tag/scalable-dashboards-wave1-evidence-20260812) (~24 MB of raw output, regenerable rather than committed). |

## Environment

RTX 4090 (24 GiB), CUDA 13 / torch 2.13.0+cu130, gemma-3-1b-it, 262k-width transcoders. Lineage
`SD-823ae3b / SL-v6.49.0 / NP-9c043341 / IT-dfee9e8`, zero dirty repos. All 17 legs reached their
batch target with no leg warnings.

`SL-v6.49.0` is a version, not a commit, because sae-lens now resolves to the released wheel rather
than a checkout — the fork pin was retired in #241. Stamping the release is the honest record: the
retired fork is still on disk, and stamping its HEAD would name code this run never imported.

**These are consumer-GPU numbers.** The columnar lane is GPU-batched, so its advantage grows with the
feature axis and with available VRAM — larger devices should exceed what is measured here. Only basic
multi-GPU generation has been explored so far; see the
[multi-GPU generation notes](https://interpretune.readthedocs.io/en/latest/usage/neuronpedia_dashboard_pipeline.html#basic-multi-gpu-generation-scope-example-configuration-and-limitations).

## Reproducing

`scripts/setup_dashboard_benchmark_env.py` prepares everything (four repos, preserved-baseline
worktrees, integrated venv, prompt datasets) in one guided, non-destructive command; then
`scripts/run_dashboard_benchmark_suite.py --mode full`. See
[dashboard_benchmark_suite_usage.md](../../../scripts/dashboard_benchmark_suite_usage.md) and the
reproduction quickstart in the [umbrella](https://github.com/speediedan/interpretune/issues/231).

## Relationship to `full_20260727`

This package supersedes [`full_20260727`](../full_20260727/), which is retained until the PR
citations move over. Both measure the same code paths; this one re-measures them on the current
dependency stack (sae-lens 6.49.0 released wheel, torch 2.13.0+cu130) at `IT-dfee9e8`.

Two differences worth knowing when comparing the two:

1. **Parity is cleaner here.** Detached-vs-current legacy activation rows now match exactly on every
   batch of both scenarios — 0 mismatched features, 0 value-bearing mismatches. The predecessor
   carried a handful of raw-row wobbles (dead features whose zero-tie fill-row count moves by ±1).
2. **The headline ratio moved down, and the numerator is why.** Monology 4096x256 reads 22.9x here
   (23,906 vs 1,042 f/min) against 24.3x before (25,218 vs 1,037). The legacy denominator is flat
   between runs; the columnar numerator is ~5% slower. That is a real difference in the measurement,
   not a re-baselining — treat the columnar figure as carrying run-to-run variance of that order.

`full_20260727`'s own manifest additionally misreports its lineage and mode: it was re-stamped with
`--from-existing`, which (before the fix shipped alongside this package) overwrote `repo_heads` with
the re-stamp-time checkout and reset `mode` to the `threeway` default. Its `benchmark_summary.md`
lineage table inherits the same values. The measurement lineage of record for that package is
`SD-1c36394/SL-990b6b37/NP-f8b13ef2/IT-9bf79d4`.
