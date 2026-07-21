# Dashboard Benchmark Suite Usage

`scripts/run_dashboard_benchmark_suite.py` runs Neuronpedia dashboard generation benchmark waves through
`scripts/profile_neuronpedia_dashboard_generation.py` and packages regenerable reviewer artifacts: extracted
markdown tables, a unified Mermaid flow diagram (`dashboard_benchmark_diagram.mmd`), an executed papermill
profiling notebook (+ HTML export), and a top-level `benchmark_summary.md` linking everything.

## Prerequisites

- The interpretune environment active (papermill, nbformat, matplotlib available), all four editable repos installed.
- Local Neuronpedia Postgres reachable (default `postgres://postgres:postgres@127.0.0.1:5433/postgres`).
- Detached baseline worktrees present for 3-way mode (`SAEDashboard-7886eaa` + patched siblings).
- The four benchmark prompt datasets present under `${IT_NP_CACHE}` (`pretokenized/` + `legacy_pretokenized/`); they
  are not published to the HF Hub — regenerate them per
  ["Regenerating the benchmark prompt datasets"](../docs/neuronpedia_dashboard_pipeline.md#regenerating-the-benchmark-prompt-datasets).
- **Reproducibility policy**: commit all outstanding changes in SAEDashboard, SAELens, neuronpedia, and
  interpretune before a reviewer 3-way regeneration. The script refuses to package with dirty trees unless
  `--allow-dirty` is passed (disposable diagnostic runs only).

```bash
cd ~/repos/interpretune
source <your-venv>/bin/activate
```

Host-specific locations are resolved from the environment with portable defaults:

- `IT_NP_CACHE` is the Neuronpedia cache root; **`IT_NP_CACHE` defaults to `$HF_HOME/interpretune/neuronpedia`**.
- `IT_BENCH_PYTHON` / `IT_BENCH_PY_SPY` select the benchmark interpreter and py-spy binary; they default to the
  running interpreter and its sibling `py-spy`.
- Repo roots default to `~/repos/<repo>` and can be overridden with `SAEDASHBOARD_REPO_ROOT`, `SAELENS_REPO_ROOT`,
  and `NEURONPEDIA_UTILS_ROOT`.
- `IT_NP_PROFILE_CACHE_DEVICE` picks the path whose filesystem is used for IO utilization sampling; it defaults to
  the Neuronpedia cache root.

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

## Mode 2: 2-way scaling sweep (legacy vs columnar_gpu)

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
confirmed prompt-doubling is throughput-neutral on both paths, and RTE `512:256` columnar peaked effectively at
the reference host's device-memory ceiling (**23.9 GiB**). Treat 256 prompts x 319-token contexts as the
columnar RTE prompt ceiling on that class of hardware; scale RTE on the feature axis (or add
`--primary-acts-batch-size`) rather than the prompt axis. Widen the sweep deliberately (explicit `--config`s)
only after new shapes are proven on the target host.

## Mode 3: Combined run (`--mode full`) — THE unified reviewer artifact, one command

Runs Mode 1, Mode 2, AND the canonical n-prompt scaling sweep (Mode 5) in one session, producing a single
self-contained reviewer package: all six 3-way legs at the accepted baseline shapes, the 2-way default
feature-axis scaling sweeps (or explicit `--config`s) per scenario, then the Monology n-prompt sweep
(`{2490, 4096, 24576}` at `4096x256`, columnar only, on `--prompt-sweep-layer` — default 12 — under the
opt-in reduced-peak-memory flags, since the largest total-prompt points can exceed device memory on some
layer/hardware combinations without them). The baseline-shape 3-way legs double as the
baseline group in the notebook's config-grouped scaling chart, so the default sweep does not re-run the
baseline shapes. The n-prompt data lands in its own summary section (fully config-annotated columns) and
the notebook's prompt-scaling charts; disable with `--no-prompt-sweep`.

```bash
python scripts/run_dashboard_benchmark_suite.py \
  --mode full \
  --session-root /tmp/np_dashboard_generation_profiles/full_$(date +%Y%m%d)
```

With the default sweeps this executes 17 sequential legs (6 threeway + 4 RTE scaling + 4 Monology scaling +
3 n-prompt sweep); budget roughly 2-3 hours on the reference host (the largest n-prompt sweep leg alone is
~20-30 minutes). Run it under `nohup`/background and monitor with `robust_benchmark_monitor.sh`.

Concrete first full-suite run (2026-07-02, dirty-tree diagnostic while the suite scripts were still uncommitted):

```bash
cd ~/repos/interpretune && source <your-venv>/bin/activate
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

## Mode 5: Prompt-dimension scaling sweep

**The default n-prompt sweep runs automatically inside `--mode full`** (Mode 3): Monology
`{2490, 4096, 24576}` at `4096x256`, columnar only, on a single configurable layer
(`--prompt-sweep-layer`, default 12), with the opt-in reduced-peak-memory flags applied to the sweep
legs only. Tune with `--prompt-sweep-config <features:prompts:nprompts>` (repeatable) or skip with
`--no-prompt-sweep`. This section documents the standalone/scaling-mode form for targeted studies
(e.g. stratifying early vs dense layers).

Sweeps the **total prompt count** (not the feature axis) to characterize the columnar `feature_statistics` /
`activation_histogram` packaging cliff. Config spec gains an optional third field —
`features:prompts:nprompts` (or `label:features:prompts:nprompts`), where `prompts` is the forward-pass
minibatch and `nprompts` is the total prompt count (overrides the preset default). Per-point peak GPU
memory (`cuda_reserved_gib`) is captured for the memory-vs-timing view.

The default Monology scaling presets are pinned to a 2490-prompt pretokenized set, so exceeding 2490 needs
`--monology-sweep`:

- **`--monology-sweep pretok`** (recommended) — columnar reads the largest built pretokenized
  `concat_<N>` set (reproducible, HF-independent for prompts, tokenization excluded from the
  measurement; e.g. `concat_24576` covers the default sweep). One set serves all smaller subsets, so build
  it ONCE (see `docs/neuronpedia_dashboard_pipeline.md` "Prompt-dimension scaling sweep sets"); the
  legacy leg streams the same first-N prompts (it cannot `load_from_disk`). Builds self-record their
  cost in `pretokenization_run.json` beside the dataset (surfaced in the artifact diagram/summary).
- **`--monology-sweep streaming`** — both paths use `load_dataset` over `monology/pile-uncopyrighted`;
  use it for sizes beyond the largest built pretokenized set.

Because the cliff is **layer-dependent** (denser layers hit it at smaller total-prompt counts), a
single-layer curve understates the OOM ceiling set by the densest layer — stratify with a sparse + a
dense layer for ceiling studies.

The **default scaling sweep is `{2490, 4096, 24576}`** (see `monology_262k_production_run.md` for how the
points were chosen). The largest points typically require the opt-in peak-memory mitigation flags below
AND a covering `concat_<N>` pretokenized set (the sweep preset resolves to the largest built `concat_<N>`
automatically). Monitor long runs with
`distributed-insight/.../source-set-enhancements/robust_benchmark_monitor.sh <suite.log>` so per-leg
failures / stalls / HF-403 storms are reported immediately instead of idling on an apparent hang (raise
`FAIL_FRAC` when a sweep intentionally probes OOM boundaries — an expected-OOM leg otherwise trips the
DEGENERATE verdict).

**Clean-room columnar-only sweeps (`--path-tags columnar`):** scaling mode runs the legacy legs first;
isolate the path under test for peak-memory measurements near the ceiling. (NOTE: peak GPU memory is NOT
necessarily monotonic in n_prompts on the default path — the runner keeps the whole activation tensor
GPU-resident while it fits a fixed device-staging byte budget, so a total-prompt count whose tensor sits
exactly at the budget can peak HIGHER (and even OOM) where a larger count spills to host staging and
completes at a lower peak. The opt-in byte-budget flag below removes that discontinuity.)

**Mitigated (opt-in reduced-peak) sweep** — bit-identical outputs; lets memory-constrained layer/hardware
combinations fit large total-prompt counts (substitute the dense layer of interest for `<L>`):

```bash
python scripts/run_dashboard_benchmark_suite.py \
  --mode scaling --scenarios monology --monology-sweep pretok --layer <L> --path-tags columnar \
  --config 4096:256:2490 --config 4096:256:4096 --config 4096:256:24576 \
  --dashboard-extra-arg=--runner-columnar-max-staged-acts-bytes=0 \
  --dashboard-extra-arg=--runner-columnar-row-chunk-size=16 \
  --skip-parity \
  --session-root /tmp/np_dashboard_generation_profiles/promptdim_L<L>_mitigated_$(date +%Y%m%d) \
  --run-tag promptdim-L<L>-mit
```

(`--skip-parity` because a single-path package has no legacy/columnar pair to compare.
`--runner-columnar-max-staged-acts-bytes=0` forces host staging of the acts matrix;
`--runner-columnar-row-chunk-size` bounds the per-chunk packaging AND batched-selection transients.)

## Useful flags

| Flag | Purpose |
| --- | --- |
| `--monology-sweep {pretok,streaming}` | §3d prompt-dimension sweep presets (exceed the 2490 pretok cap) |
| `--prompt-sweep / --no-prompt-sweep` | Include the canonical n-prompt sweep legs (default: on in full mode) |
| `--prompt-sweep-layer` / `--prompt-sweep-config` | n-prompt sweep layer (default 12) and specs (repeatable) |
| `--path-tags {det,legacy,columnar}` | Restrict legs to these path tags (clean-room single-path sweeps) |
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
