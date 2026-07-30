# Scalable Dashboards (Wave 1) — reviewer benchmark evidence, `full_20260727`

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
| **[Executed profiling notebook (renders in-browser)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/speediedan/interpretune/streamlined-streamable-dashboard-generation-phase-1/docs/benchmark_artifacts/full_20260727/dashboard_profiling_20260727_212938.html)** | Charts and data only, code cells hidden. Opens rendered via htmlpreview; the [raw file](dashboard_profiling_20260727_212938.html) is also committed here, and the same HTML is attached to the release for offline use. |
| Complete package incl. per-leg raw profiling output | `.tar.gz` on the same [release](https://github.com/speediedan/interpretune/releases/tag/scalable-dashboards-wave1-evidence-20260727) (~23 MB of raw output, regenerable rather than committed). |

## Environment

RTX 4090 (24 GiB), CUDA 13 / torch 2.13.0+cu130, gemma-3-1b-it, 262k-width transcoders. Lineage
`SD-40bdc62 / SL-978a9654 / NP-6484b342 / IT-eeb9745`, zero dirty repos.

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
