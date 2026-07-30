# Benchmark baseline patches (vendored)

These are the audited patches `scripts/setup_dashboard_benchmark_env.py` applies on top of the
clean SAEDashboard preserved-baseline commit (`7886eaa`) to build the detached comparison
worktree used by the three-way dashboard benchmark (lineage label `SD-7886eaa+benchmark_patches`).
Apply order and per-patch classification/rationale follow below.

---

# Detached Baseline (7886eaa) Patch Set — Classification and Rationale

Date: 2026-06-06

The detached legacy baseline (SAEDashboard 7886eaa) requires several patches to support parity
benchmarking against the current legacy path. The patches are organized by category and applied in
dependency order on top of the clean 7886eaa commit.

## Patch Order (apply sequentially)

1. `saedashboard-7886eaa-profiling.patch` — Profiling/logging infrastructure (largest patch)
2. `saedashboard-7886eaa-activation-significance-floor.patch` — Functional: activation floor filtering
3. `saedashboard-7886eaa-no-shuffle-tokens.patch` — Functional: deterministic token ordering
4. `saedashboard-7886eaa-shared-tokens-file.patch` — Functional: shared pretokenized dataset support

## Patch 1: Profiling Infrastructure

Status: **Profiling only — no functional impact**

Adds `[runner_perf]` per-stage timing instrumentation across all dashboard generation stages.
This is the largest patch and must be applied first because subsequent patches may extend
config classes that were modified here.

### Files changed:
- `sae_dashboard/perf_logging.py` (new) — `log_perf_event`, `timed_stage`, `elapsed_timer`,
  `tensor_runtime_metadata`, `temporary_torch_num_threads`, CPU/IO/thread/rusage snapshot utilities
- `sae_dashboard/feature_data_generator.py` — `timed_stage` wrappers around: primary_acts_device_transfer,
  feature_encode, rolling_coefficient_update, feature_acts_pad_and_cpu_transfer, activation_cache_load,
  activation_capture; `temporary_torch_num_threads` for rolling; extra kwargs passthrough
  (`minibatch_index`, `feature_count`, `token_shape`) for perf context; `all_features_acts = None`
  in non-TopK branch (linter-only); `profile_feature_data` local variable
- `sae_dashboard/neuronpedia/neuronpedia_runner.py` — `timed_stage` around conversion/serialization;
  `elapsed_timer` and `log_perf_event` for disk_write and batch_total; performance config
  passthrough to `SaeVisConfig`; CLI args for `--log-performance`, `--profile-rolling-substages`,
  `--rolling-coefficient-num-threads`
- `sae_dashboard/neuronpedia/neuronpedia_runner_config.py` — fields: `log_performance`,
  `profile_rolling_substages`, `rolling_coefficient_num_threads`
- `sae_dashboard/sae_vis_data.py` — fields: `log_performance`, `profile_rolling_substages`,
  `perf_batch`, `rolling_coefficient_num_threads`
- `sae_dashboard/sae_vis_runner.py` — `timed_stage` wrappers around all dashboard stages:
  activation_and_encode_total, logits_projection, feature_statistics_packaging,
  feature_table_packaging, logits_histogram_packaging, activation_histogram_packaging,
  logits_table_packaging, sequence_packaging; ignore_tokens_mask computation moved
  outside per-feature loop for perf measurement separation; feature loop refactored
  from single loop to stage-separated loops
- `sae_dashboard/utils_fns.py` — `timed_stage` wrappers in `RollingCorrCoef.update()`
  for x/y materialization, accumulator init, sum updates; extra `perf_enabled`, `perf_label`,
  `perf_context` kwargs; `perf_logging` import

### Config fields added (only active when `log_performance=True`):
- `NeuronpediaRunnerConfig`: `log_performance`, `profile_rolling_substages`, `rolling_coefficient_num_threads`
- `SaeVisConfig`: `log_performance`, `profile_rolling_substages`, `perf_batch`, `rolling_coefficient_num_threads`

### Rationale:
Profiling instrumentation is required to compare substage timings between detached legacy
and current legacy paths. All changes are gated behind `log_performance=True`; when this flag
is `False` (default), no `timed_stage` overhead is incurred and behavior is identical.

### Non-profiling incidental changes within this patch:
- `self.device = torch.device(device)` (was `self.device = device`) in `RollingCorrCoef.__init__`:
  ensures device is a `torch.device`, not a string. This prevents downstream bugs when the
  device is used in tensor creation. Classified as a bugfix.
- `all_features_acts = None` in the non-TopK else branch: purely linter hygiene.
- Ignore tokens mask refactoring: moved from inside per-feature loop to before the loop.
  Functionally equivalent — the mask does not depend on individual features.

---

## Patch 2: Activation Significance Floor

Status: **Non-profiling functional change**

### Files changed:
- `sae_dashboard/neuronpedia/neuronpedia_runner.py` — CLI arg `--activation-significance-floor`
- `sae_dashboard/neuronpedia/neuronpedia_runner_config.py` — field `activation_significance_floor: float = 0.0`
- `sae_dashboard/sae_vis_data.py` — field `activation_significance_floor: float = 0.0`
- `sae_dashboard/sae_vis_runner.py` — `nonzero_feat_acts = masked_feat_acts[masked_feat_acts > significance_floor]`
  replaces `masked_feat_acts > 0`; `getattr(self.cfg, "activation_significance_floor", 0.0)`

### Rationale:
The current legacy path uses an `activation_significance_floor` to filter near-zero
activations. This patch adds the same parameter to the detached legacy path so both
paths apply identical activation filtering during histogram computation. Without this
patch, the detached legacy uses a hardcoded floor of 0, which can include
sub-threshold activations that the current path filters out, causing bin edge shifts
and different sequence selections.

---

## Patch 3: No-Shuffle Tokens

Status: **Non-profiling functional change**

### Files changed:
- `sae_dashboard/neuronpedia/neuronpedia_runner.py` — CLI arg `--no-shuffle-tokens`
  (maps to `shuffle_tokens=False`); config passthrough `shuffle_tokens=args.shuffle_tokens`

### Rationale:
The original `get_tokens()` uses `torch.randperm()` without a fixed seed (line ~1071
of the original), causing each process invocation to get a different token order.
With different token order → different minibatch composition → different cross-prompt
attention → different residual stream → different SAE output → different batch JSON.

Setting `shuffle_tokens=False` uses dataset order in both paths, producing identical
model inputs and enabling true parity comparison. This is the root cause fix for
the ~50 frac_nonzero diffs observed in detached legacy self-parity testing.

---

## Patch 4: Shared Tokens File

Status: **Non-profiling functional change**

### Files changed:
- `sae_dashboard/neuronpedia/neuronpedia_runner.py` — CLI arg `--shared-tokens-file`;
  `get_tokens()` modified to check `shared_tokens_file` first, loading tokens directly
  from the pre-built file when provided (bypassing token generation from dataset)
- `sae_dashboard/neuronpedia/neuronpedia_runner_config.py` — field `shared_tokens_file: Optional[str] = None`

### Rationale:
The original `get_tokens()` generates `tokens_2490.pt` from the legacy JSONL dataset
(`train.jsonl`), which produces 2486 prompts (via internal deduplication). The current
legacy path uses a pre-built shared `tokens_2490.pt` with 2488 prompts. The 2-prompt
difference (including the "Brooklyn Borough Hall" example at dataset index 2486) means
the detached legacy sees different prompt sets → different batch composition →
different model forward outputs → different SAE activations.

This patch allows the pipeline to stage a shared tokens file (identical to what the
current legacy path uses) into the output directory, ensuring both paths process
exactly the same prompts.
