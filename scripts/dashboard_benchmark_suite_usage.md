# Dashboard Benchmark Suite Usage

`scripts/run_dashboard_benchmark_suite.py` runs Neuronpedia dashboard generation benchmark waves through
`scripts/profile_neuronpedia_dashboard_generation.py` and packages regenerable reviewer artifacts: extracted
markdown tables, a unified Mermaid flow diagram (`dashboard_benchmark_diagram.mmd`), an executed papermill
profiling notebook (+ HTML export), and a top-level `benchmark_summary.md` linking everything.

## Prerequisites

- `it_latest` venv active (papermill, nbformat, matplotlib available), all four editable repos installed.
- Local Neuronpedia Postgres reachable (default `postgres://postgres:postgres@127.0.0.1:5433/postgres`).
- Detached baseline worktrees present for 3-way mode (`SAEDashboard-7886eaa` + patched siblings).
- **Reproducibility policy**: commit all outstanding changes in SAEDashboard, SAELens, neuronpedia, and
  interpretune before a reviewer 3-way regeneration. The script refuses to package with dirty trees unless
  `--allow-dirty` is passed (disposable diagnostic runs only).

```bash
cd ~/repos/interpretune
source /mnt/cache/speediedan/.venvs/it_latest/bin/activate
```

## Mode 1: Full 3-way, 2-scenario benchmark (accepted-benchmark flags)

Runs 6 legs sequentially (RTE + Monology x detached legacy / current legacy / columnar_gpu) with the accepted
Phase 4/5 flag set (jemalloc, layer 9, 4 batches / 1 warmup, import-stage profiling, perf logging,
rolling-coefficient threads=8), then packages the reviewer artifacts:

```bash
python scripts/run_dashboard_benchmark_suite.py \
  --mode threeway \
  --session-root /tmp/np_dashboard_generation_profiles/threeway_$(date +%Y%m%d) \
  --local-db-url "postgres://postgres:postgres@127.0.0.1:5433/postgres"
```

Restrict to one scenario with `--scenarios rte` (or `monology`). Forward additional child-pipeline flags to all
legs with repeated `--dashboard-extra-arg=<flag>`.

## Mode 2: 2-way scaling sweep (current legacy vs columnar_gpu)

Excludes the detached baseline and sweeps batch-shape configs (`features:prompts`):

```bash
python scripts/run_dashboard_benchmark_suite.py \
  --mode scaling \
  --scenarios monology \
  --config 1024:256 --config 2048:256 --config 2048:512 \
  --session-root /tmp/np_dashboard_generation_profiles/scaling_$(date +%Y%m%d)
```

Each config gets its own variant directory (`legacy_monology_2048x512`, `columnar_monology_2048x512`, ...);
the profiling notebook groups the configs per path so scaling characteristics can be compared as configs
accumulate.

**Naming**: variant directories and presets use the canonical path names — `detached_legacy_*` (preserved pre-PR
baseline), `legacy_*` (in-tree deprecated legacy path), `columnar_*` (current columnar_gpu path). The retired
phase-era preset names (`phase3-legacy-*`, `phase4-current-legacy-*`, `phase3-lazy-*`) remain valid aliases in
`profile_neuronpedia_dashboard_generation.py`, and packaging still classifies artifact roots produced under the
old naming.

When `--config` is omitted in scaling (or full) mode, the per-scenario **default sweep**
(`DEFAULT_SCALING_CONFIGS`) is used. As of Phase 6.2 the defaults sweep the **feature axis** — the dimension on
which `columnar_gpu` throughput actually scales — at 2x and 4x the accepted reviewer baseline feature count
(RTE baseline `512x128`, Monology baseline `1024x256`):

| Scenario | Default sweep configs |
| --- | --- |
| RTE | `1024:128`, `2048:128` |
| Monology | `2048:256`, `4096:256` |

The prior conservative prompt-axis shapes (RTE `512:256`, Monology `1024:512`) were retired after two full waves
confirmed prompt-doubling is throughput-neutral on both paths, and RTE `512:256` columnar peaked at **23.9 GiB
GPU** (effectively at the device ceiling on the RTX 4090 host). Treat 256 prompts x 319-token contexts as the
columnar RTE prompt ceiling; scale RTE on the feature axis (or add `--primary-acts-batch-size`) rather than the
prompt axis. Widen the sweep deliberately (explicit `--config`s) only after new shapes are proven on the target
host.

## Mode 3: Combined run (`--mode full`)

Runs Mode 1 and Mode 2 in one session: all six 3-way legs at the accepted baseline shapes, then the 2-way default
scaling sweeps (or explicit `--config`s) for each scenario. The baseline-shape 3-way legs double as the baseline
group in the notebook's config-grouped scaling chart, so the default sweep does not re-run the baseline shapes.

```bash
python scripts/run_dashboard_benchmark_suite.py \
  --mode full \
  --session-root /tmp/np_dashboard_generation_profiles/full_$(date +%Y%m%d)
```

With the default sweeps this executes 14 sequential legs (6 threeway + 4 RTE scaling + 4 Monology scaling);
budget roughly 1.5-2.5 hours on the reference RTX 4090 host. Run it under `nohup`/background and monitor the log.

Concrete first full-suite run (2026-07-02, dirty-tree diagnostic while the suite scripts were still uncommitted):

```bash
cd ~/repos/interpretune && source /mnt/cache/speediedan/.venvs/it_latest/bin/activate
nohup python scripts/run_dashboard_benchmark_suite.py \
  --mode full \
  --session-root /tmp/np_dashboard_generation_profiles/full_20260702 \
  --package-root /tmp/dashboard_benchmark_packages/full_20260702 \
  --allow-dirty \
  > /tmp/full_suite_20260702.log 2>&1 &
```

The packaged reviewer artifacts land in `/tmp/dashboard_benchmark_packages/full_20260702/` (summary, diagram,
executed notebook + HTML, tables, raw per-leg copies) — add that folder to the workspace to review iteratively.

## Mode 4: Package from an existing artifact root (no benchmark execution)

Regenerate the reviewer package (tables, diagram, notebook, summary) from any prior artifact root laid out as
`<root>/<variant>/<timestamp>/<preset>/result.json`:

```bash
python scripts/run_dashboard_benchmark_suite.py \
  --from-existing /tmp/np_dashboard_generation_profiles/phase5_current_lineage_20260701 \
  --package-root /tmp/dashboard_benchmark_package_demo \
  --allow-dirty          # only when the working trees are intentionally dirty (diagnostic package)
```

## Useful flags

| Flag | Purpose |
| --- | --- |
| `--timing-mode {steady-state,all-batches}` | Exclude warmup batches (default) or average all batches |
| `--rolling-coefficient-substage` | Include deep `rolling_*` sub-substage aggregation when instrumented |
| `--skip-parity` | Skip the per-feature activation row parity pass (parses all legacy batch JSONs) |
| `--skip-notebook` | Skip papermill execution/HTML export |
| `--target-batches` / `--summary-warmup-batches` / `--layer` | Benchmark shape (defaults 4 / 1 / 9) |
| `--rolling-threads` | `--runner-rolling-coefficient-num-threads` value for every leg (default 8) |
| `--run-tag` | Suffix tag baked into child run names |

## Package layout

When `--package-root` is omitted, packages default to `/tmp/dashboard_benchmark_packages/<session_root_name>` so
successive packages accumulate in one reviewable location.

```
<package_root>/
├── benchmark_summary.md            # top-level reviewer summary (links to all assets)
├── manifest.json                   # lineage (repo heads + dirty flags), run parameters, source root
├── extracted_data.json             # machine-readable payload consumed by the notebook
├── dashboard_benchmark_diagram.mmd # unified Mermaid flow diagram
├── dashboard_profiling_<ts>.ipynb  # executed papermill notebook
├── dashboard_profiling_<ts>.html   # HTML export
├── tables/{primary,substage,import,resource,parity}_<scenario>.md
└── raw/<variant>/                  # per-leg result.json / runner_perf_events.json / import_stage_profile.json
```

## Extraction semantics

- `stage_timing` events inherit their batch from the most recent preceding `batch_total` event; steady-state
  means exclude batches below `--summary-warmup-batches` (this reproduces the published Phase 5 tables exactly).
- Summary-table substage values use last-event-per-batch semantics, matching the documented extraction process
  and all previously published tables. For once-per-batch stages that is the per-batch value; for per-minibatch
  stages (`rolling_coefficient_update`) it is the final (partial) minibatch's wall time. Per-event means and
  per-batch sums are additionally captured in `extracted_data.json` and used for diagram annotations.
- The `batch_total` row uses the profiling harness `avg_batch_seconds` (detected batch intervals).
- E2E features/min = `(features_per_batch x batches) * 60 / (avg_batch_s * batches + import_wall_s)`.
