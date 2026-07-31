# Dashboard Benchmark Summary

Generated: 2026-07-27T21:29:38+00:00

Source artifact root: `/tmp/np_dashboard_generation_profiles/full_20260727`
Timing mode: steady-state (warmup batches excluded: 1)
Coordination PR: [Scalable Dashboards coordination PR](https://github.com/speediedan/interpretune/issues/231)

## Lineage

| Path | Lineage | Dirty repos |
| --- | --- | --- |
| Preserved baseline | `SD-7886eaa+benchmark_patches/SL-3eea6552/NP-5a33f17/IT-eeb9745` | none |
| legacy | `SD-40bdc62/SL-978a9654/NP-6484b342/IT-eeb9745` | none |
| columnar_gpu | `SD-40bdc62/SL-978a9654/NP-6484b342/IT-eeb9745` | none |

## Generation commands

Suite invocation that produced these artifacts:

```bash
python scripts/run_dashboard_benchmark_suite.py --from-existing /tmp/np_dashboard_generation_profiles/full_20260727 --package-root /tmp/dashboard_benchmark_packages/full_20260727_stamped --coordination-pr-url https://github.com/speediedan/interpretune/issues/231
```
Prompts: the accepted-shape presets use their pretokenized caches (built once via the commands in `docs/neuronpedia_dashboard_pipeline.md` § Pretokenize dashboard datasets); not re-pretokenized this run.

## Linked Assets

- Unified flow diagram: [dashboard_benchmark_diagram.mmd](dashboard_benchmark_diagram.mmd)
- Profiling notebook: [dashboard_profiling_20260727_212938.ipynb](dashboard_profiling_20260727_212938.ipynb)
- Notebook HTML export: [dashboard_profiling_20260727_212938.html](dashboard_profiling_20260727_212938.html)

## RTE

### Primary Benchmark

| Variant | Config | Avg batch s | Features/min | Import wall s | Import load s | Import act s | Activation rows | E2E features/min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 34.84 | 881.7 | 49.6 | 20.6 | 27.7 | 65,131 | 650.2 |
| legacy | 512x128 | 36.06 | 851.9 | 49.5 | 20.4 | 27.8 | 65,131 | 634.2 |
| legacy | 1024x128 | 68.11 | 902.1 | 98.3 | 41.2 | 54.6 | 130,392 | 663.0 |
| legacy | 2048x128 | 142.47 | 862.5 | 207.7 | 82.1 | 121.0 | 258,602 | 632.1 |
| columnar_gpu | 512x128 | 9.78 | 3139.7 | 13.4 | 0.7 | 9.0 | 65,186 | 2339.6 |
| columnar_gpu | 1024x128 | 10.39 | 5915.0 | 23.9 | 1.3 | 18.2 | 130,479 | 3757.8 |
| columnar_gpu | 2048x128 | 10.95 | 11224.9 | 45.8 | 2.4 | 36.6 | 258,745 | 5489.4 |

### Substage Timings (steady-state)

| Stage | Preserved baseline 512x128 s | legacy 512x128 s | legacy 1024x128 s | legacy 2048x128 s | columnar_gpu 512x128 s | columnar_gpu 1024x128 s | columnar_gpu 2048x128 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 13.221 | 13.922 | 24.267 | 53.637 | 8.685 | 8.835 | 9.123 |
| feature_statistics_packaging | 0.858 | 0.848 | 1.637 | 3.158 | 0.010 | 0.019 | 0.036 |
| logits_histogram_packaging | 10.534 | 10.236 | 20.348 | 43.378 | 0.024 | 0.046 | 0.092 |
| activation_histogram_packaging | 0.141 | 0.142 | 0.297 | 1.134 | 0.045 | 0.094 | 0.165 |
| sequence_packaging | 5.200 | 5.152 | 11.068 | 22.499 | 0.326 | 0.639 | 1.262 |
| rolling_coefficient_update | 0.313 | 0.332 | 0.571 | 1.222 | 0.001 | 0.001 | 0.001 |
| batch_total | 34.842 | 36.061 | 68.106 | 142.470 | 9.784 | 10.387 | 10.947 |

### DB Import Substage

| Variant | Config | Import wall s | Conversion s | Activation load s | Activation import s | Imported rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 49.6 | 11.9 | 20.6 | 27.7 | 65,131 |
| legacy | 512x128 | 49.5 | 11.9 | 20.4 | 27.8 | 65,131 |
| legacy | 1024x128 | 98.3 | 24.3 | 41.2 | 54.6 | 130,392 |
| legacy | 2048x128 | 207.7 | 48.2 | 82.1 | 121.0 | 258,602 |
| columnar_gpu | 512x128 | 13.4 | 0.0 | 0.7 | 9.0 | 65,186 |
| columnar_gpu | 1024x128 | 23.9 | 0.0 | 1.3 | 18.2 | 130,479 |
| columnar_gpu | 2048x128 | 45.8 | 0.0 | 2.4 | 36.6 | 258,745 |

### Resource Peaks

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 7.97 | 21.57 | 11,282 | 11,292 | 4.7 | 100 |
| legacy | 512x128 | 7.33 | 22.92 | 10,198 | 10,208 | 6.0 | 100 |
| legacy | 1024x128 | 8.19 | 22.05 | 12,098 | 12,108 | 4.5 | 100 |
| legacy | 2048x128 | 10.48 | 25.77 | 18,798 | 18,808 | 4.5 | 100 |
| columnar_gpu | 512x128 | 3.49 | 19.25 | 13,604 | 13,614 | 78.4 | 100 |
| columnar_gpu | 1024x128 | 3.77 | 19.06 | 14,584 | 14,594 | 75.3 | 100 |
| columnar_gpu | 2048x128 | 4.34 | 19.64 | 16,244 | 16,254 | 75.8 | 100 |

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
| Preserved baseline | 1024x256 | 47.46 | 1294.5 | 63.7 | 25.8 | 35.3 | 137,329 | 969.4 |
| legacy | 1024x256 | 47.99 | 1280.3 | 63.9 | 25.5 | 35.9 | 137,329 | 960.4 |
| legacy | 2048x256 | 95.51 | 1286.6 | 122.5 | 52.1 | 66.4 | 271,721 | 974.2 |
| legacy | 4096x256 | 237.02 | 1036.9 | 242.0 | 107.2 | 127.1 | 546,237 | 826.0 |
| columnar_gpu | 1024x256 | 5.17 | 11874.2 | 24.7 | 1.5 | 18.6 | 137,085 | 5418.3 |
| columnar_gpu | 2048x256 | 6.88 | 17852.3 | 48.1 | 2.9 | 39.0 | 271,228 | 6498.4 |
| columnar_gpu | 4096x256 | 9.75 | 25217.7 | 93.1 | 5.6 | 79.3 | 544,871 | 7440.8 |

### Substage Timings (steady-state)

| Stage | Preserved baseline 1024x256 s | legacy 1024x256 s | legacy 2048x256 s | legacy 4096x256 s | columnar_gpu 1024x256 s | columnar_gpu 2048x256 s | columnar_gpu 4096x256 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 10.559 | 10.573 | 22.133 | 56.849 | 3.946 | 4.657 | 5.738 |
| feature_statistics_packaging | 1.494 | 1.513 | 2.918 | 5.629 | 0.026 | 0.049 | 0.095 |
| logits_histogram_packaging | 21.271 | 21.153 | 42.413 | 111.825 | 0.046 | 0.092 | 0.182 |
| activation_histogram_packaging | 0.221 | 0.218 | 0.979 | 2.322 | 0.116 | 0.210 | 0.507 |
| sequence_packaging | 7.571 | 7.520 | 15.684 | 35.890 | 0.581 | 1.159 | 2.729 |
| rolling_coefficient_update | 0.743 | 0.740 | 1.585 | 4.238 | 0.001 | 0.001 | 0.001 |
| batch_total | 47.462 | 47.990 | 95.507 | 237.019 | 5.174 | 6.883 | 9.746 |

### DB Import Substage

| Variant | Config | Import wall s | Conversion s | Activation load s | Activation import s | Imported rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 1024x256 | 63.7 | 16.3 | 25.8 | 35.3 | 137,329 |
| legacy | 1024x256 | 63.9 | 16.1 | 25.5 | 35.9 | 137,329 |
| legacy | 2048x256 | 122.5 | 32.7 | 52.1 | 66.4 | 271,721 |
| legacy | 4096x256 | 242.0 | 66.8 | 107.2 | 127.1 | 546,237 |
| columnar_gpu | 1024x256 | 24.7 | 0.0 | 1.5 | 18.6 | 137,085 |
| columnar_gpu | 2048x256 | 48.1 | 0.0 | 2.9 | 39.0 | 271,228 |
| columnar_gpu | 4096x256 | 93.1 | 0.0 | 5.6 | 79.3 | 544,871 |

### Resource Peaks

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 1024x256 | 5.89 | 21.20 | 10,418 | 10,428 | 3.5 | 100 |
| legacy | 1024x256 | 7.35 | 22.86 | 9,192 | 9,202 | 3.6 | 100 |
| legacy | 2048x256 | 9.12 | 24.07 | 11,376 | 11,386 | 3.2 | 100 |
| legacy | 4096x256 | 17.06 | 32.75 | 17,374 | 17,384 | 2.9 | 100 |
| columnar_gpu | 1024x256 | 3.60 | 18.98 | 12,742 | 12,752 | 68.5 | 100 |
| columnar_gpu | 2048x256 | 3.86 | 19.20 | 14,202 | 14,212 | 63.8 | 100 |
| columnar_gpu | 4096x256 | 4.62 | 19.96 | 17,662 | 17,672 | 50.5 | 100 |

### Activation Row Parity (detached vs current legacy)

| Batch | Det rows | Cur rows | Match | Mismatched features | Value-bearing mismatches |
| --- | ---: | ---: | --- | ---: | ---: |
| 0 | 34133 | 34133 | MATCH | 0 | 0 |
| 1 | 34381 | 34381 | MATCH | 0 | 0 |
| 2 | 34399 | 34399 | MATCH | 0 | 0 |
| 3 | 34416 | 34416 | MATCH | 0 | 0 |

**100.00% raw per-feature match / 100.00% value-bearing match across 4096 feature-batches (0 raw, 0 value-bearing mismatches)**

## N-prompt scaling (Monology)

Prompt-dimension scaling sweep on **layer 12** (single configurable layer; column headings carry the full config as `features x fwd-minibatch x total-prompts`). Swept configs: 4096x256x2490, 4096x256x4096, 4096x256x24576. Columnar path only, run under the opt-in reduced-peak-memory flags (`see leg logs`) that the 24,576-prompt point requires — outputs are bit-identical; only peak GPU memory and speed move. NOTE: a single-layer curve understates the OOM ceiling set by the densest layer (layer-density adds several GiB to the packaging working set).

### Primary Benchmark (n-prompt sweep)

| Variant | Config | Avg batch s | Features/min | Import wall s | Import load s | Import act s | Activation rows | E2E features/min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| columnar_gpu | 4096x256x24576 | 231.09 | 1063.5 | 94.4 | 5.5 | 80.5 | 513,747 | 964.9 |
| columnar_gpu | 4096x256x2490 | 28.86 | 8514.2 | 91.1 | 5.1 | 77.6 | 492,756 | 4758.4 |
| columnar_gpu | 4096x256x4096 | 42.76 | 5747.5 | 92.8 | 5.2 | 79.2 | 496,703 | 3726.1 |

### Substage Timings (n-prompt sweep, steady-state)

| Stage | columnar_gpu 4096x256x24576 s | columnar_gpu 4096x256x2490 s | columnar_gpu 4096x256x4096 s |
| --- | ---: | ---: | ---: |
| activation_and_encode_total | 84.193 | 10.470 | 16.004 |
| feature_statistics_packaging | 58.577 | 6.029 | 9.931 |
| logits_histogram_packaging | 0.182 | 0.181 | 0.181 |
| activation_histogram_packaging | 62.041 | 6.750 | 10.336 |
| sequence_packaging | 23.041 | 4.227 | 5.465 |
| rolling_coefficient_update | 0.001 | 0.001 | 0.001 |
| batch_total | 231.094 | 28.865 | 42.759 |

### Resource Peaks (n-prompt sweep)

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| columnar_gpu | 4096x256x24576 | 27.95 | 43.46 | 22,246 | 22,256 | 28.7 | 100 |
| columnar_gpu | 4096x256x2490 | 6.20 | 21.52 | 15,526 | 15,536 | 27.3 | 100 |
| columnar_gpu | 4096x256x4096 | 7.86 | 23.22 | 16,426 | 16,436 | 25.2 | 100 |

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
