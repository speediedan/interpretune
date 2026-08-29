# Dashboard Benchmark Summary

Generated: 2026-08-29T17:11:06+00:00

Source artifact root: `/tmp/np_dashboard_generation_profiles/full_20260829`
Timing mode: steady-state (warmup batches excluded: 1)
Coordination PR: [Scalable Dashboards coordination PR](https://github.com/speediedan/interpretune/issues/231)

## Lineage

| Path | Lineage | Dirty repos |
| --- | --- | --- |
| Preserved baseline | `SD-7886eaa+benchmark_patches/SL-3eea6552/NP-789942ed/IT-c573842` | none |
| legacy | `SD-7a0266f/SL-v6.49.1/NP-9c043341/IT-c573842` | none |
| columnar_gpu | `SD-7a0266f/SL-v6.49.1/NP-9c043341/IT-c573842` | none |

## Generation commands

Suite invocation that produced these artifacts:

```bash
python scripts/run_dashboard_benchmark_suite.py --from-existing /tmp/np_dashboard_generation_profiles/full_20260829 --package-root /tmp/dashboard_benchmark_packages/full_20260829 --coordination-pr-url https://github.com/speediedan/interpretune/issues/231
```
Prompts: the accepted-shape presets use their pretokenized caches (built once via the commands in `docs/neuronpedia_dashboard_pipeline.md` § Pretokenize dashboard datasets); not re-pretokenized this run.

## Linked Assets

- Unified flow diagram: [dashboard_benchmark_diagram.mmd](dashboard_benchmark_diagram.mmd)
- Profiling notebook: [dashboard_profiling_20260829_171106.ipynb](dashboard_profiling_20260829_171106.ipynb)
- Notebook HTML export: [dashboard_profiling_20260829_171106.html](dashboard_profiling_20260829_171106.html)

## RTE

### Primary Benchmark

| Variant | Config | Avg batch s | Features/min | Import wall s | Import load s | Import act s | Activation rows | E2E features/min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 35.22 | 872.1 | 49.2 | 19.2 | 28.9 | 65,131 | 646.3 |
| legacy | 512x128 | 35.23 | 871.9 | 47.5 | 19.0 | 27.4 | 65,131 | 652.3 |
| legacy | 1024x128 | 67.65 | 908.2 | 93.6 | 38.6 | 53.2 | 130,392 | 674.8 |
| legacy | 2048x128 | 141.88 | 866.1 | 198.2 | 76.2 | 118.9 | 258,602 | 641.9 |
| columnar_gpu | 512x128 | 9.16 | 3355.0 | 12.1 | 0.7 | 7.9 | 65,186 | 2521.3 |
| columnar_gpu | 1024x128 | 10.29 | 5970.5 | 22.3 | 1.3 | 16.8 | 130,479 | 3871.8 |
| columnar_gpu | 2048x128 | 11.43 | 10748.0 | 43.7 | 2.3 | 35.1 | 258,745 | 5496.5 |

### Substage Timings (steady-state)

| Stage | Preserved baseline 512x128 s | legacy 512x128 s | legacy 1024x128 s | legacy 2048x128 s | columnar_gpu 512x128 s | columnar_gpu 1024x128 s | columnar_gpu 2048x128 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 13.247 | 13.746 | 24.384 | 53.755 | 8.666 | 8.796 | 9.124 |
| feature_statistics_packaging | 0.850 | 0.834 | 1.656 | 3.161 | 0.011 | 0.019 | 0.039 |
| logits_histogram_packaging | 10.274 | 9.900 | 20.373 | 42.698 | 0.024 | 0.046 | 0.093 |
| activation_histogram_packaging | 0.142 | 0.143 | 0.301 | 1.100 | 0.045 | 0.089 | 0.163 |
| sequence_packaging | 5.212 | 5.120 | 11.085 | 22.557 | 0.296 | 0.573 | 1.196 |
| rolling_coefficient_update | 0.322 | 0.307 | 0.569 | 1.252 | 0.001 | 0.001 | 0.001 |
| batch_total | 35.224 | 35.232 | 67.649 | 141.875 | 9.156 | 10.291 | 11.433 |

### DB Import Substage

| Variant | Config | Import wall s | Conversion s | Activation load s | Activation import s | Imported rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 49.2 | 10.5 | 19.2 | 28.9 | 65,131 |
| legacy | 512x128 | 47.5 | 10.5 | 19.0 | 27.4 | 65,131 |
| legacy | 1024x128 | 93.6 | 21.4 | 38.6 | 53.2 | 130,392 |
| legacy | 2048x128 | 198.2 | 42.6 | 76.2 | 118.9 | 258,602 |
| columnar_gpu | 512x128 | 12.1 | 0.0 | 0.7 | 7.9 | 65,186 |
| columnar_gpu | 1024x128 | 22.3 | 0.0 | 1.3 | 16.8 | 130,479 |
| columnar_gpu | 2048x128 | 43.7 | 0.0 | 2.3 | 35.1 | 258,745 |

### Resource Peaks

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 512x128 | 7.82 | 14.47 | 11,282 | 11,296 | 6.6 | 100 |
| legacy | 512x128 | 7.75 | 14.39 | 10,198 | 10,212 | 4.6 | 100 |
| legacy | 1024x128 | 11.15 | 16.07 | 12,098 | 12,112 | 6.0 | 76 |
| legacy | 2048x128 | 12.54 | 19.18 | 18,798 | 18,812 | 4.5 | 100 |
| columnar_gpu | 512x128 | 5.52 | 12.14 | 13,604 | 13,618 | 82.2 | 100 |
| columnar_gpu | 1024x128 | 5.81 | 12.42 | 14,584 | 14,598 | 81.5 | 100 |
| columnar_gpu | 2048x128 | 6.40 | 13.02 | 16,244 | 16,258 | 80.0 | 100 |

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
| Preserved baseline | 1024x256 | 47.15 | 1303.1 | 59.5 | 24.7 | 33.2 | 137,329 | 990.6 |
| legacy | 1024x256 | 46.04 | 1334.5 | 59.1 | 24.6 | 32.8 | 137,329 | 1010.3 |
| legacy | 2048x256 | 95.04 | 1293.0 | 116.4 | 48.9 | 64.6 | 271,721 | 989.9 |
| legacy | 4096x256 | 238.53 | 1030.3 | 235.6 | 101.7 | 128.6 | 546,237 | 826.3 |
| columnar_gpu | 1024x256 | 5.14 | 11944.2 | 23.5 | 1.4 | 17.8 | 137,085 | 5576.6 |
| columnar_gpu | 2048x256 | 6.84 | 17966.3 | 45.7 | 2.7 | 37.0 | 271,228 | 6730.9 |
| columnar_gpu | 4096x256 | 9.70 | 25333.7 | 89.0 | 5.3 | 75.6 | 544,871 | 7692.5 |

### Substage Timings (steady-state)

| Stage | Preserved baseline 1024x256 s | legacy 1024x256 s | legacy 2048x256 s | legacy 4096x256 s | columnar_gpu 1024x256 s | columnar_gpu 2048x256 s | columnar_gpu 4096x256 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| activation_and_encode_total | 10.158 | 10.504 | 22.189 | 57.793 | 3.955 | 4.512 | 5.596 |
| feature_statistics_packaging | 1.502 | 1.482 | 2.914 | 5.648 | 0.026 | 0.049 | 0.095 |
| logits_histogram_packaging | 21.219 | 19.825 | 42.124 | 111.839 | 0.046 | 0.092 | 0.182 |
| activation_histogram_packaging | 0.221 | 0.215 | 0.955 | 2.253 | 0.114 | 0.205 | 0.507 |
| sequence_packaging | 7.617 | 7.573 | 15.752 | 36.622 | 0.564 | 1.122 | 2.676 |
| rolling_coefficient_update | 0.716 | 0.713 | 1.610 | 4.247 | 0.001 | 0.001 | 0.001 |
| batch_total | 47.150 | 46.041 | 95.037 | 238.529 | 5.144 | 6.839 | 9.701 |

### DB Import Substage

| Variant | Config | Import wall s | Conversion s | Activation load s | Activation import s | Imported rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 1024x256 | 59.5 | 14.8 | 24.7 | 33.2 | 137,329 |
| legacy | 1024x256 | 59.1 | 14.8 | 24.6 | 32.8 | 137,329 |
| legacy | 2048x256 | 116.4 | 29.9 | 48.9 | 64.6 | 271,721 |
| legacy | 4096x256 | 235.6 | 61.0 | 101.7 | 128.6 | 546,237 |
| columnar_gpu | 1024x256 | 23.5 | 0.0 | 1.4 | 17.8 | 137,085 |
| columnar_gpu | 2048x256 | 45.7 | 0.0 | 2.7 | 37.0 | 271,228 |
| columnar_gpu | 4096x256 | 89.0 | 0.0 | 5.3 | 75.6 | 544,871 |

### Resource Peaks

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved baseline | 1024x256 | 9.09 | 15.70 | 10,418 | 10,432 | 5.3 | 100 |
| legacy | 1024x256 | 8.48 | 15.11 | 9,192 | 9,206 | 3.2 | 92 |
| legacy | 2048x256 | 10.76 | 17.38 | 11,376 | 11,390 | 3.2 | 100 |
| legacy | 4096x256 | 19.17 | 25.74 | 17,374 | 17,388 | 3.3 | 100 |
| columnar_gpu | 1024x256 | 6.39 | 12.23 | 12,960 | 12,974 | 72.6 | 100 |
| columnar_gpu | 2048x256 | 5.99 | 12.54 | 14,202 | 14,216 | 67.8 | 100 |
| columnar_gpu | 4096x256 | 6.77 | 13.30 | 17,662 | 17,676 | 58.1 | 100 |

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
| columnar_gpu | 4096x256x24576 | 234.29 | 1049.0 | 87.3 | 5.1 | 74.1 | 513,747 | 959.5 |
| columnar_gpu | 4096x256x2490 | 28.15 | 8729.4 | 83.8 | 4.8 | 71.0 | 492,756 | 5004.9 |
| columnar_gpu | 4096x256x4096 | 42.57 | 5772.9 | 86.4 | 5.3 | 73.3 | 496,703 | 3829.8 |

### Substage Timings (n-prompt sweep, steady-state)

| Stage | columnar_gpu 4096x256x24576 s | columnar_gpu 4096x256x2490 s | columnar_gpu 4096x256x4096 s |
| --- | ---: | ---: | ---: |
| activation_and_encode_total | 85.330 | 10.220 | 15.731 |
| feature_statistics_packaging | 60.563 | 5.924 | 9.871 |
| logits_histogram_packaging | 0.182 | 0.183 | 0.182 |
| activation_histogram_packaging | 61.990 | 6.565 | 10.597 |
| sequence_packaging | 23.081 | 4.156 | 5.371 |
| rolling_coefficient_update | 0.001 | 0.001 | 0.001 |
| batch_total | 234.291 | 28.153 | 42.571 |

### Resource Peaks (n-prompt sweep)

| Variant | Config | Max tree RSS GiB | Max host used GiB | Max GPU proc MiB | Max GPU dev MiB | Avg GPU util % (steady) | Max GPU util % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| columnar_gpu | 4096x256x24576 | 29.84 | 36.25 | 22,246 | 22,260 | 28.5 | 100 |
| columnar_gpu | 4096x256x2490 | 8.18 | 14.74 | 15,526 | 15,540 | 23.2 | 100 |
| columnar_gpu | 4096x256x4096 | 9.73 | 16.28 | 16,426 | 16,440 | 26.9 | 100 |

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
