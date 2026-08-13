# Dashboard Benchmark Summary

Generated: 2026-08-13T05:13:03+00:00

Source artifact root: `/tmp/np_dashboard_generation_profiles/full_20260812`
Timing mode: steady-state (warmup batches excluded: 1)
Coordination PR: [Scalable Dashboards coordination PR](https://github.com/speediedan/interpretune/issues/231)

## Lineage

| Path | Lineage | Dirty repos |
| --- | --- | --- |
| Preserved baseline | `SD-7886eaa+benchmark_patches/SL-3eea6552/NP-5a33f17/IT-dfee9e8` | none |
| legacy | `SD-823ae3b/SL-v6.49.0/NP-9c043341/IT-dfee9e8` | none |
| columnar_gpu | `SD-823ae3b/SL-v6.49.0/NP-9c043341/IT-dfee9e8` | none |

## Generation commands

Suite invocation that produced these artifacts:

```bash
python scripts/run_dashboard_benchmark_suite.py --mode full --session-root /tmp/np_dashboard_generation_profiles/full_20260812 --package-root /tmp/dashboard_benchmark_packages/full_20260812 --run-tag full-unified-20260812 --coordination-pr-url https://github.com/speediedan/interpretune/issues/231 --local-db-url postgres://postgres:postgres@127.0.0.1:5433/postgres
```
Prompts: the accepted-shape presets use their pretokenized caches (built once via the commands in `docs/neuronpedia_dashboard_pipeline.md` § Pretokenize dashboard datasets); not re-pretokenized this run.

## Linked Assets

- Unified flow diagram: [dashboard_benchmark_diagram.mmd](dashboard_benchmark_diagram.mmd)
- Profiling notebook: [dashboard_profiling_20260813_051303.ipynb](dashboard_profiling_20260813_051303.ipynb)
- Notebook HTML export: [dashboard_profiling_20260813_051303.html](dashboard_profiling_20260813_051303.html)

## RTE

### Primary Benchmark

| Variant | Config | Avg batch s | Features/min | Import wall s | Import load s | Import act s | Activation rows | E2E features/min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 35.91 | 855.5 | 48.2 | 19.1 | 27.2 | 65,131 | 640.7 |
| legacy | 512x128 | 35.32 | 869.9 | 46.2 | 19.0 | 25.8 | 65,131 | 655.5 |
| legacy | 1024x128 | 68.35 | 898.9 | 91.5 | 39.1 | 49.9 | 130,392 | 673.5 |
| legacy | 2048x128 | 142.11 | 864.7 | 194.9 | 77.5 | 112.6 | 258,602 | 643.9 |
| columnar_gpu | 512x128 | 9.75 | 3150.7 | 13.8 | 0.7 | 9.4 | 65,186 | 2326.9 |
| columnar_gpu | 1024x128 | 10.32 | 5954.8 | 24.4 | 1.3 | 18.6 | 130,479 | 3744.9 |
| columnar_gpu | 2048x128 | 10.90 | 11273.6 | 47.3 | 2.5 | 38.5 | 258,745 | 5405.2 |

### Substage Timings (steady-state)

| Stage | Preserved baseline 512x128 s | legacy 512x128 s | legacy 1024x128 s | legacy 2048x128 s | columnar_gpu 512x128 s | columnar_gpu 1024x128 s | columnar_gpu 2048x128 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 13.594 | 14.271 | 25.146 | 54.341 | 8.750 | 8.859 | 9.222 |
| feature_statistics_packaging | 0.834 | 0.823 | 1.599 | 3.173 | 0.011 | 0.019 | 0.036 |
| logits_histogram_packaging | 10.522 | 9.754 | 19.948 | 41.863 | 0.024 | 0.046 | 0.093 |
| activation_histogram_packaging | 0.146 | 0.142 | 0.297 | 1.155 | 0.047 | 0.091 | 0.171 |
| sequence_packaging | 5.432 | 5.194 | 11.172 | 22.654 | 0.337 | 0.635 | 1.291 |
| rolling_coefficient_update | 0.326 | 0.323 | 0.548 | 1.237 | 0.001 | 0.001 | 0.001 |
| batch_total | 35.908 | 35.316 | 68.351 | 142.108 | 9.750 | 10.318 | 10.900 |

### DB Import Substage

| Variant | Config | Import wall s | Conversion s | Activation load s | Activation import s | Imported rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 48.2 | 10.7 | 19.1 | 27.2 | 65,131 |
| legacy | 512x128 | 46.2 | 10.7 | 19.0 | 25.8 | 65,131 |
| legacy | 1024x128 | 91.5 | 21.7 | 39.1 | 49.9 | 130,392 |
| legacy | 2048x128 | 194.9 | 42.7 | 77.5 | 112.6 | 258,602 |
| columnar_gpu | 512x128 | 13.8 | 0.0 | 0.7 | 9.4 | 65,186 |
| columnar_gpu | 1024x128 | 24.4 | 0.0 | 1.3 | 18.6 | 130,479 |
| columnar_gpu | 2048x128 | 47.3 | 0.0 | 2.5 | 38.5 | 258,745 |

### Resource Peaks

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 6.17 | 25.45 | 11,282 | 11,292 | 8.8 | 100 |
| legacy | 512x128 | 7.83 | 27.28 | 10,198 | 10,208 | 6.2 | 100 |
| legacy | 1024x128 | 8.71 | 26.39 | 12,098 | 12,108 | 5.2 | 100 |
| legacy | 2048x128 | 10.51 | 29.86 | 18,798 | 18,808 | 4.6 | 100 |
| columnar_gpu | 512x128 | 3.51 | 22.95 | 13,604 | 13,614 | 81.8 | 100 |
| columnar_gpu | 1024x128 | 3.78 | 23.08 | 14,584 | 14,594 | 81.2 | 100 |
| columnar_gpu | 2048x128 | 4.33 | 23.64 | 16,244 | 16,254 | 77.1 | 100 |

### Activation Row Parity (detached vs current legacy)

| Batch | Det rows | Cur rows | Match | Mismatched features | Value-bearing mismatches |
| --- | ---: | ---: | --- | ---: | ---: |
| 0 | 16275 | 16275 | MATCH | 0 | 0 |
| 1 | 16181 | 16181 | MATCH | 0 | 0 |
| 2 | 16243 | 16243 | MATCH | 0 | 0 |
| 3 | 16432 | 16432 | MATCH | 0 | 0 |

**100.00% raw per-feature match / 100.00% value-bearing match across 2048 feature-batches (0 raw, 0 value-bearing mismatches)**

## Monology

### Primary Benchmark

| Variant | Config | Avg batch s | Features/min | Import wall s | Import load s | Import act s | Activation rows | E2E features/min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 1024x256 | 47.22 | 1301.2 | 59.0 | 24.1 | 32.3 | 137,329 | 991.4 |
| legacy | 1024x256 | 46.66 | 1316.9 | 59.4 | 24.4 | 32.9 | 137,329 | 998.7 |
| legacy | 2048x256 | 95.70 | 1284.0 | 118.6 | 48.7 | 65.8 | 271,721 | 980.3 |
| legacy | 4096x256 | 235.87 | 1041.9 | 245.2 | 103.6 | 133.8 | 546,237 | 827.0 |
| columnar_gpu | 1024x256 | 5.15 | 11924.4 | 25.8 | 1.5 | 20.0 | 137,085 | 5296.1 |
| columnar_gpu | 2048x256 | 6.85 | 17932.9 | 49.7 | 2.9 | 40.7 | 271,228 | 6377.4 |
| columnar_gpu | 4096x256 | 10.28 | 23905.7 | 98.1 | 5.7 | 84.1 | 544,871 | 7058.5 |

### Substage Timings (steady-state)

| Stage | Preserved baseline 1024x256 s | legacy 1024x256 s | legacy 2048x256 s | legacy 4096x256 s | columnar_gpu 1024x256 s | columnar_gpu 2048x256 s | columnar_gpu 4096x256 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 10.409 | 10.779 | 22.427 | 57.183 | 3.956 | 4.664 | 5.700 |
| feature_statistics_packaging | 1.494 | 1.513 | 2.893 | 5.624 | 0.026 | 0.049 | 0.096 |
| logits_histogram_packaging | 21.128 | 20.167 | 42.054 | 110.440 | 0.047 | 0.092 | 0.183 |
| activation_histogram_packaging | 0.223 | 0.222 | 0.941 | 2.262 | 0.114 | 0.206 | 0.512 |
| sequence_packaging | 7.584 | 7.610 | 15.805 | 36.279 | 0.586 | 1.154 | 2.849 |
| rolling_coefficient_update | 0.740 | 0.765 | 1.623 | 4.251 | 0.001 | 0.001 | 0.001 |
| batch_total | 47.217 | 46.657 | 95.698 | 235.872 | 5.152 | 6.852 | 10.280 |

### DB Import Substage

| Variant | Config | Import wall s | Conversion s | Activation load s | Activation import s | Imported rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 1024x256 | 59.0 | 14.5 | 24.1 | 32.3 | 137,329 |
| legacy | 1024x256 | 59.4 | 14.6 | 24.4 | 32.9 | 137,329 |
| legacy | 2048x256 | 118.6 | 29.9 | 48.7 | 65.8 | 271,721 |
| legacy | 4096x256 | 245.2 | 62.2 | 103.6 | 133.8 | 546,237 |
| columnar_gpu | 1024x256 | 25.8 | 0.0 | 1.5 | 20.0 | 137,085 |
| columnar_gpu | 2048x256 | 49.7 | 0.0 | 2.9 | 40.7 | 271,228 |
| columnar_gpu | 4096x256 | 98.1 | 0.0 | 5.7 | 84.1 | 544,871 |

### Resource Peaks

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 1024x256 | 5.90 | 25.22 | 10,418 | 10,428 | 4.1 | 100 |
| legacy | 1024x256 | 7.95 | 27.31 | 9,192 | 9,202 | 3.5 | 48 |
| legacy | 2048x256 | 8.66 | 28.07 | 11,376 | 11,386 | 3.4 | 100 |
| legacy | 4096x256 | 17.11 | 36.48 | 17,374 | 17,384 | 3.2 | 100 |
| columnar_gpu | 1024x256 | 3.55 | 22.97 | 12,960 | 12,970 | 70.8 | 100 |
| columnar_gpu | 2048x256 | 3.92 | 23.38 | 14,202 | 14,212 | 46.9 | 100 |
| columnar_gpu | 4096x256 | 4.66 | 24.14 | 17,662 | 17,672 | 50.4 | 100 |

### Activation Row Parity (detached vs current legacy)

| Batch | Det rows | Cur rows | Match | Mismatched features | Value-bearing mismatches |
| --- | ---: | ---: | --- | ---: | ---: |
| 0 | 34133 | 34133 | MATCH | 0 | 0 |
| 1 | 34381 | 34381 | MATCH | 0 | 0 |
| 2 | 34399 | 34399 | MATCH | 0 | 0 |
| 3 | 34416 | 34416 | MATCH | 0 | 0 |

**100.00% raw per-feature match / 100.00% value-bearing match across 4096 feature-batches (0 raw, 0 value-bearing mismatches)**

## N-prompt scaling (Monology)

Prompt-dimension scaling sweep on **layer 12** (single configurable layer; column headings carry the full config as `features x fwd-minibatch x total-prompts`). Swept configs: 4096x256x2490, 4096x256x4096, 4096x256x24576. Columnar path only, run under the opt-in reduced-peak-memory flags (`--runner-columnar-max-staged-acts-bytes=0`, `--runner-columnar-row-chunk-size=16`) that the 24,576-prompt point requires — outputs are bit-identical; only peak GPU memory and speed move. NOTE: a single-layer curve understates the OOM ceiling set by the densest layer (layer-density adds several GiB to the packaging working set).

### Primary Benchmark (n-prompt sweep)

| Variant | Config | Avg batch s | Features/min | Import wall s | Import load s | Import act s | Activation rows | E2E features/min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| columnar_gpu | 4096x256x24576 | 235.36 | 1044.2 | 101.4 | 5.5 | 87.2 | 513,747 | 942.6 |
| columnar_gpu | 4096x256x2490 | 28.22 | 8708.3 | 96.8 | 5.2 | 82.9 | 492,756 | 4687.6 |
| columnar_gpu | 4096x256x4096 | 43.34 | 5670.9 | 94.8 | 5.3 | 81.1 | 496,703 | 3666.0 |

### Substage Timings (n-prompt sweep, steady-state)

| Stage | columnar_gpu 4096x256x24576 s | columnar_gpu 4096x256x2490 s | columnar_gpu 4096x256x4096 s |
| --- | ---: | ---: | ---: |
| activation_and_encode_total | 84.431 | 10.461 | 15.781 |
| feature_statistics_packaging | 60.679 | 5.973 | 10.001 |
| logits_histogram_packaging | 0.181 | 0.183 | 0.183 |
| activation_histogram_packaging | 62.843 | 6.345 | 10.902 |
| sequence_packaging | 24.113 | 4.217 | 5.436 |
| rolling_coefficient_update | 0.001 | 0.001 | 0.001 |
| batch_total | 235.358 | 28.221 | 43.337 |

### Resource Peaks (n-prompt sweep)

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| columnar_gpu | 4096x256x24576 | 27.98 | 47.28 | 22,246 | 22,256 | 28.8 | 100 |
| columnar_gpu | 4096x256x2490 | 6.15 | 25.62 | 15,526 | 15,536 | 24.6 | 100 |
| columnar_gpu | 4096x256x4096 | 7.77 | 27.26 | 16,426 | 16,436 | 28.1 | 100 |

Pretokenization (one-time): 8.587 s for 2,490 prompts x 128 tokens (recorded 2026-07-24T02:44:38+00:00).

## Regeneration

One command regenerates this full artifact (all benchmark legs, the n-prompt scaling sweep, tables, diagram, and notebook):

```bash
python scripts/run_dashboard_benchmark_suite.py --mode full
```

To re-package from an existing artifact root without re-running benchmarks:

```bash
python scripts/run_dashboard_benchmark_suite.py --from-existing <artifact_root> --package-root <out_dir>
```

See `scripts/dashboard_benchmark_suite_usage.md` for full usage, including live 3-way and scaling modes.
