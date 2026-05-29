# Neuronpedia Dashboard Pipeline

This note documents the supported Interpretune path for generating SAEDashboard outputs, converting them to Neuronpedia export bundles, and importing them into a local Neuronpedia database.

## Purpose

Use `interpretune.utils.neuronpedia_dashboard_pipeline` when you want one file-backed, resumable pipeline that:

1. runs SAEDashboard generation for a layer range
2. converts each completed layer into Neuronpedia export format
3. imports the converted bundle into a local Neuronpedia Postgres database
4. emits enough diagnostics to distinguish stalls, kills, and conversion/import seams
5. can replay existing export bundles into the local DB without regenerating dashboards

This replaces the earlier ad hoc shell resume flow that depended on `tee` and separate manual conversion/import steps.

## Required environment

The current working setup on this host is:

```bash
PYTHON=/mnt/cache/speediedan/.venvs/it_latest/bin/python
HF_HOME=/mnt/cache_extended/speediedan/.cache/huggingface
HF_DATASETS_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/datasets
HF_HUB_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/hub
IT_NP_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia
```

For local DB imports, the pipeline resolves the DB URL from the Neuronpedia local env files, but the explicit localhost URL currently in use is:

```bash
postgres://postgres:postgres@127.0.0.1:5433/postgres
```

## YAML configs and launcher

The pipeline now supports `--config <path.yaml>`, and the recommended operational path is to pair that with
`scripts/launch_neuronpedia_dashboard_pipeline.py`.

Config files can use `EXTENDS` inheritance and keep launcher-only settings separate from pipeline values:

```yaml
EXTENDS: ./gemmascope-2-rte-base.yaml

pipeline:
  model_name: gemma-3-1b-it
  neuronpedia_source_set_id: gemmascope-2-transcoder-262k-rte
  run_name_suffix: context319-full-prompts
  n_features_per_batch: 512
  n_prompts_in_forward_pass: 128
  primary_acts_batch_size: 64
  archive_partial_dirs: false
  import_to_local_db: false

launcher:
  background: true
  log_path: /tmp/dashboard.launcher.log
  env:
    HF_HOME: /mnt/cache_extended/speediedan/.cache/huggingface
```

Notes:

1. `pipeline` uses the same snake_case names as `NeuronpediaDashboardPipelineConfig`.
2. Prefer positive boolean keys in YAML such as `archive_partial_dirs`, `resume_from_existing_logs`, and `import_to_local_db`.
3. Explicit CLI flags always override config-file values. This applies both when calling the pipeline directly and when using the launcher script.
4. `launcher` is ignored by the pipeline itself and is only read by `scripts/launch_neuronpedia_dashboard_pipeline.py`.
5. When you need a fresh generation lineage for the same `model_name` plus `neuronpedia_source_set_id`, set `run_name_suffix` or override `--run-name-suffix`. That changes the run directory and default logs without forcing a fake source-set id or a different export target.
6. If you accidentally point a resumed launch at a fully completed lineage, the pipeline now warns that the requested layer range is already complete and tells you to use `--run-name-suffix`, `--run-root`, or `--no-resume`.

The shared configs for the current RTE flows live under:

```text
/home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs
```

Current single-worker fallback pattern for the context-`319` full-prompt transcoder restart:

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  /home/speediedan/repos/interpretune/scripts/launch_neuronpedia_dashboard_pipeline.py \
  --config /home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs/gemmascope-2-transcoder-262k-rte-gpu1.yaml \
  --run-name-suffix context319-full-prompts
```

Current two-worker launch pattern for the same run namespace:

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  /home/speediedan/repos/interpretune/scripts/launch_neuronpedia_dashboard_pipeline.py \
  --config /home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs/gemmascope-2-transcoder-262k-rte-gpu01.yaml \
  --run-name-suffix context319-full-prompts
```

The launcher backgrounds the process by default when `launcher.background: true`, prints the resolved launcher log,
prints the pipeline log, and prints the exact `tail -f` command to monitor progress. Use `--foreground` when you want
to keep the process attached to the terminal.

Use `--dry-run` to inspect the resolved command or worker fanout without launching anything. `--print-command` prints
the command as part of launch output, so it is not a dry-run mode.

You can still override individual values at launch time:

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  /home/speediedan/repos/interpretune/scripts/launch_neuronpedia_dashboard_pipeline.py \
  --config /home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs/gemmascope-2-clt-262k-rte.yaml \
  --start-layer 3 \
  --start-batch 10
```

### Multi-worker launcher mode

For simple multi-GPU dashboard generation, keep shared generation identity and output settings in `pipeline`, then put
per-process GPU overrides under `launcher.workers`. The launcher starts one pipeline process per worker, writes a
worker-specific launcher log, and writes a manifest named `launcher.workers.<timestamp>.json` into the run directory.

Only these worker-level overrides are accepted:

```text
cuda_visible_devices
start_layer
end_layer
n_features_per_batch
n_prompts_in_forward_pass
primary_acts_batch_size
heartbeat_seconds
stall_timeout_seconds
layer_lock_stale_seconds
runner_log_performance
runner_torch_profile
runner_torch_profile_dir
```

That allowlist is intentional: workers may differ in device placement, layer start/end overrides, and memory/monitoring
shape, but they must share the same source set, prompt cache, run namespace, and conversion/import policy. Feature width
can vary by worker, but partial-layer resume is only allowed when the existing layer `run_settings.json` matches that
worker's resolved `n_features_per_batch`. Pin `start_layer` on heterogeneous workers when resuming existing partial
layers so the worker that owns the matching feature width reaches that partial first.

Example two-worker shape for the current context-`319` full-prompt run:

```yaml
pipeline:
  run_name_suffix: context319-full-prompts
  n_features_per_batch: 512
  n_prompts_in_forward_pass: 64
  primary_acts_batch_size: 16
  enable_layer_locks: true

launcher:
  monitor: true
  monitor_heartbeat_seconds: 60
  workers:
    - id: gpu1
      cuda_visible_devices: "1"
      start_layer: 3
      n_features_per_batch: 512
      n_prompts_in_forward_pass: 64
      primary_acts_batch_size: 16
    - id: gpu0
      cuda_visible_devices: "0"
      start_layer: 4
      n_features_per_batch: 1024
      n_prompts_in_forward_pass: 256
      primary_acts_batch_size: 64
```

    Set `launcher.monitor: true` or pass `--monitor` to start a detached supervisor process alongside the workers. The
    monitor behaves like a small service for the current launcher manifest: it watches each expected worker, emits
    `MONITOR_HEARTBEAT` lines with worker config, batch progress, process snapshots, GPU snapshots, and restart state, and
    only exits when every expected worker has either completed its requested layer range or exhausted its OOM restart
    budget.

    OOM restart policy is deliberately conservative and per worker:

    1. First OOM observed in that worker's new log segment: halve `primary_acts_batch_size` and restart that worker.
    2. Second OOM observed for the adjusted worker config: halve `n_prompts_in_forward_pass` and restart that worker.
    3. Third OOM observed: log `MONITOR_OOM_DISABLED` with a message that two automatic mitigations have already been
       attempted, then stop restarting that worker so it cannot enter an infinite OOM loop.

    The monitor keeps supervising any other worker that is still running or restartable. OOM detection is log-segment based:
    the launcher records each worker log's byte offset at launch time, and the monitor searches only new text after that
    offset for `CUDA out of memory`, `out of memory`, or `torch.OutOfMemoryError`. Every automatic restart appends to the
    same worker pipeline log and writes a `launcher.<worker>.monitor.<timestamp>.log` file. Use `--no-monitor` to disable a
    config-file monitor setting, or `--monitor-foreground` when debugging the monitor loop itself.

    The fastest measured context-`319` pair remains GPU1 at `512x64 / acts16` plus GPU0 at `1024x256 / acts64`, about
    `1440.9` combined features/min. The shared production config now keeps that lower-residency pair under the OOM monitor.
    The more aggressive `1024x512 / acts256` GPU0 shape still reaches `cuda_max_allocated_gib=16.43` without OOM and now
    lands around `75.0s` on the second batch (`~819 features/min`), which is much better than the earlier `92.3s`
    (`~665 features/min`) result but still not a clear throughput win over `1024x256 / acts64`. Use that aggressive shape
    when the goal is to validate higher VRAM residency; keep `1024x256 / acts64` when wall-clock throughput is the priority.

For the current run namespace, layers `0-2` were already completed with the earlier `256`-feature shape and can be
kept. Any partially written `256`-feature layer must be archived or restarted from batch `0` before launching this
heterogeneous config, otherwise the pipeline will stop on the run-settings mismatch.

The pipeline uses per-layer lock files under `<run_directory>/layer_locks/layer_<n>.lock`. Each lock records the PID,
worker id, CUDA-visible device string, layer number, creation time, and pipeline log path. If a worker is stopped with a
normal process kill, another worker or a later restart will remove the stale lock when the recorded PID is no longer
alive or when `layer_lock_stale_seconds` has elapsed.

Worker pipeline logs are deliberately separated:

```text
run.gpu0.resume-0-25.log
run.gpu1.resume-0-25.log
launcher.gpu0.<timestamp>.log
launcher.gpu1.<timestamp>.log
launcher.workers.<timestamp>.json
monitor.<timestamp>.log
```

Stop one worker by killing its process group. For example, if the launcher reports `PID 1373792` for `gpu1`:

```bash
kill -TERM -1373792
```

### Clean stop and single-worker restart

For the current multi-worker RTE production flow, killing only the detached monitor is not sufficient. The monitor will
stop supervising, but any already-running worker pipeline and child SAEDashboard runner processes will keep going until
their own process groups are terminated.

Use this sequence when GPU0 must be freed for development or profiling while GPU1 continues production work:

1. Identify the monitor and worker process groups:

```bash
ps -eo pid,ppid,pgid,etimes,cmd | grep -E \
  "launch_neuronpedia_dashboard_pipeline|interpretune\.utils\.neuronpedia_dashboard_pipeline|sae_dashboard\.neuronpedia\.neuronpedia_runner|gemmascope-2-transcoder-262k-rte" \
  | grep -v grep
```

2. Stop the detached monitor process group first so it cannot auto-restart a worker:

```bash
kill -TERM -<monitor_pgid>
```

3. Stop each worker process group that should exit. This also terminates the child `sae_dashboard.neuronpedia.neuronpedia_runner`
   process for that worker:

```bash
kill -TERM -<gpu0_worker_pgid>
kill -TERM -<gpu1_worker_pgid>
```

4. Verify that the workers are really gone before reusing a GPU:

```bash
ps -eo pid,ppid,pgid,etimes,cmd | grep -E \
  "interpretune\.utils\.neuronpedia_dashboard_pipeline|sae_dashboard\.neuronpedia\.neuronpedia_runner|gemmascope-2-transcoder-262k-rte" \
  | grep -v grep

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
```

5. Check `layer_locks/` and remove only stale locks whose recorded PID is no longer alive. Example lock payloads are JSON and
   include `pid`, `worker_id`, `cuda_visible_devices`, and `pipeline_log_path`.

```bash
for file in <run_directory>/layer_locks/*.lock; do
  echo "=== $file ==="
  cat "$file"
done

rm -f <run_directory>/layer_locks/*.lock
```

6. Restart only the desired worker without the monitor. For the current production namespace, GPU1 can be resumed directly
   with the same config and `worker_id`, but GPU0 should remain stopped for development or profiling:

```bash
nohup /mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --config /home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs/gemmascope-2-transcoder-262k-rte-gpu01.yaml \
  --run-name-suffix context319-full-prompts \
  --worker-id gpu1 \
  --enable-layer-locks \
  --cuda-visible-devices 1 \
  --start-layer 10 \
  --n-features-per-batch 512 \
  --n-prompts-in-forward-pass 64 \
  --primary-acts-batch-size 16 \
  > <run_directory>/restart.gpu1.<timestamp>.launch.log 2>&1 &
```

Important resume caveat:

- Partial-layer resume is only safe when the existing `run_settings.json` matches the worker's resolved `n_features_per_batch`.
- In the current `context319-full-prompts` namespace, layer `9` was left with partial `1024`-feature GPU0 output, so a
  `512`-feature GPU1 worker could not resume it and exited with the explicit run-settings mismatch error.
- The correct recovery was to leave layer `9` for GPU0 work and restart GPU1 from layer `10`, which already matches the
  `512`-feature worker shape.

The other worker continues because it has its own process group, child runner, log file, and layer lock. To stop the
whole two-worker run, terminate each worker process group separately.

## Standard launch example

This is the current Target B pattern for `gemma-3-1b-it` `gemmascope-2-transcoder-16k`:

```bash
RUN_ROOT="/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs"
RUN_DIR="${RUN_ROOT}/gemma-3-1b-it_gemmascope-2-transcoder-16k"
LAUNCH_LOG="${RUN_DIR}/run.resume-24-25.launch.log"

mkdir -p "${RUN_DIR}"

nohup /mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --model-name gemma-3-1b-it \
  --model-layers 26 \
  --sae-set gemma-scope-2-1b-it-transcoders-all \
  --neuronpedia-source-set-id gemmascope-2-transcoder-16k \
  --neuronpedia-source-set-description 'Transcoder - 16k' \
  --creator-name 'Google DeepMind' \
  --release-id gemma-scope-2 \
  --release-title 'Gemma Scope 2' \
  --release-url https://huggingface.co/mwhanna/gemma-scope-2-1b-it \
  --hf-weights-repo-id mwhanna/gemma-scope-2-1b-it \
  --hf-weights-path-template 'transcoder_all/layer_{layer}_width_16k_l0_small_affine' \
  --hook-point hook_mlp_in \
  --prompts-huggingface-dataset-path monology/pile-uncopyrighted \
  --start-layer 24 \
  --end-layer 25 \
  --sae-path-template 'layer_{layer}_width_16k_l0_small_affine' \
  --python-executable /mnt/cache/speediedan/.venvs/it_latest/bin/python \
  --cuda-visible-devices 0 \
  --heartbeat-seconds 60 \
  --stall-timeout-seconds 1800 \
  --use-skip-transcoder \
  > "${LAUNCH_LOG}" 2>&1 &
```

## Bridge + pretokenized dataset example

The pipeline supports `TransformerBridge` for the runner subprocess and can point the runner at a prebuilt local
HuggingFace dataset with an `input_ids` column. RTE prompt construction happens in Interpretune before launch; the
SAEDashboard runner only receives generic dataset paths.

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --model-name gemma-3-1b-it \
  --model-layers 26 \
  --sae-set gemma-scope-2-1b-it-transcoders-all \
  --neuronpedia-source-set-id gemmascope-2-transcoder-262k-rte \
  --neuronpedia-source-set-description 'Transcoder - 262k (RTE pilot)' \
  --creator-name 'Google DeepMind' \
  --release-id gemma-scope-2 \
  --release-title 'Gemma Scope 2' \
  --release-url https://huggingface.co/google/gemma-scope-2-1b-it \
  --hf-weights-repo-id google/gemma-scope-2-1b-it \
  --hf-weights-path-template 'transcoder_all/layer_{layer}_width_262k_l0_small_affine' \
  --hook-point hook_mlp_in \
  --prompts-huggingface-dataset-path aps/super_glue \
  --prompts-huggingface-dataset-config-name rte \
  --prompts-huggingface-dataset-split train \
  --prompts-pretokenized-dataset-path /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts \
  --model-wrapper bridge \
  --bridge-enable-compatibility-mode \
  --runner-log-resource-snapshots \
  --runner-log-hook-aliases \
  --start-layer 0 \
  --end-layer 0 \
  --start-batch 0 \
  --end-batch 0 \
  --sae-path-template 'layer_{layer}_width_262k_l0_small_affine' \
  --python-executable /mnt/cache/speediedan/.venvs/it_latest/bin/python \
  --cuda-visible-devices 0 \
  --use-skip-transcoder
```

Key extra controls in this mode:

1. `--model-wrapper bridge` switches the runner from the legacy HookedTransformer path to `SAETransformerBridge`.
2. `--bridge-enable-compatibility-mode` restores the legacy hook aliases (`blocks.<n>.hook_mlp_in`, `hook_mlp_out`) expected by the current dashboard metadata.
3. `--prompts-dataset-mode` makes the prompt artifact contract explicit:
  - `load_dataset` means the runner reads raw prompt rows or tokenized Hub/local datasets through `load_dataset(...)` style inputs;
  - `load_from_disk` means the pipeline uses a local `save_to_disk()` / `load_from_disk()` prompt cache;
  - `legacy_jsonl` means the legacy runner consumes a deprecated local `load_dataset("json", data_files=...)` export with `<split>.jsonl` plus `sae_lens.json`.
  The config default is `load_dataset`. When `--prompts-pretokenized-dataset-path` is set, the pipeline resolves that
  invocation to `load_from_disk` because the supplied path names a `save_to_disk()` artifact, except for the
  `legacy` runner which continues to stay on the `load_dataset`/`legacy_jsonl` side of the contract. When the
  dataset path already points at a legacy `<split>.jsonl` plus `sae_lens.json` export, the pipeline still resolves that
  path to deprecated `legacy_jsonl` compatibility automatically. `legacy` must not be paired with
  `load_from_disk`.
4. `--prompts-pretokenized-dataset-path` points the runner at the local SAELens-compatible prompt cache. For raw text datasets, use `--prompts-dataset-text-field` with a prebuilt dataset that already has the dashboard prompt text.
5. `--start-batch` and `--end-batch` make small, reproducible batch probes possible without forking a second pipeline path.
6. `--runner-log-resource-snapshots` and `--runner-log-hook-aliases` provide lightweight migration diagnostics without touching the outer pipeline heartbeats.
7. `--runner-log-performance` enables `[runner_perf]` lines from SAEDashboard with per-batch CPU snapshots, process IO deltas, disk write throughput, and wall/CUDA timings for activation capture, feature encode, logits/statistics packaging, sequence packaging, JSON serialization, and writes.
8. `--runner-torch-profile` writes short-run Torch profiler traces under `torch_profiles/` in the layer output tree, or under `--runner-torch-profile-dir` when provided. Keep it to one- or two-batch probes; it adds material overhead.
9. `--prompts-pretokenized-dataset-path` can point at a local HuggingFace dataset saved with an `input_ids` column. The runner still uses the Bridge model path, but SAELens skips raw text tokenization when building the activation-store token cache.

Current validated result for this shape:

1. default-dataset Bridge smoke on `monology/pile-uncopyrighted`: completed and wrote `batch-0.json`
2. RTE Bridge batch-`0` probe on all `2490` train prompts: completed with exit status `0`, `1:12.99` end-to-end wall time, about `24s` of actual batch execution after token setup, about `19.8 GiB` max RSS, and `3.04 GiB` max CUDA allocation
3. RTE Bridge `512x128` one-batch profile with the pretokenized prompt dataset: completed with exit status `0`, `62.5s` end-to-end wall time, and the token generation profile share dropped from about `29%` to under `1%`

### Pretokenize dashboard datasets

Build the local prompt caches once before a long run. SAEDashboard now provides the generic dashboard pretokenization
harness and prompt artifact writer. Interpretune keeps task-specific rendering, such as the RTE/BoolQ custom module,
and passes it to SAEDashboard with `--custom-dataset-module`.

RTE full-prompt cache command of record:

```bash
HF_HOME=/mnt/cache_extended/speediedan/.cache/huggingface \
HF_DATASETS_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/datasets \
HF_HUB_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/hub \
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  -m sae_dashboard.neuronpedia.prompt_pretokenization \
  --dataset-path aps/super_glue \
  --dataset-name rte \
  --dataset-split train \
  --tokenizer-name google/gemma-3-1b-it \
  --context-size 128 \
  --custom-dataset-module it_examples.utils.dashboard_pretokenization_rte \
  --windowing-mode max-prompt-pad \
  --no-shuffle \
  --output-dir /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts \
  --force
```

Monology cache command of record:

```bash
HF_HOME=/mnt/cache_extended/speediedan/.cache/huggingface \
HF_DATASETS_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/datasets \
HF_HUB_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/hub \
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  -m sae_dashboard.neuronpedia.prompt_pretokenization \
  --dataset-path monology/pile-uncopyrighted \
  --dataset-split train \
  --tokenizer-name google/gemma-3-1b-it \
  --context-size 128 \
  --column-name text \
  --use-chat-formatting \
  --streaming \
  --num-proc 1 \
  --no-shuffle \
  --max-tokenized-rows 2490 \
  --output-dir /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_2490 \
  --force
```

  Windowing modes now belong to SAEDashboard rather than the Interpretune custom module:

  1. `concatenate` keeps the legacy SAELens packed-window behavior and remains the streaming-friendly choice for dense
    corpora such as Monology.
  2. `filter-truncate` keeps only prompts that already satisfy the configured width and truncates them to that width. It
    is still a packed-mode contract, not the example-aligned padded contract used for RTE full-prompt caches.
  3. `max-prompt-pad` preserves one row per rendered prompt, pads every row to the longest observed prompt, and emits an
    `attention_mask` sidecar so later bucketing can recover each prompt's effective length. This is the current RTE
    full-prompt mode.
  4. `fixed-context-pad` also preserves one row per prompt, but pads to the configured `context-size` and rejects any
    prompt that would exceed that width.

  Prompt-length metadata and bucket scheduling now share one contract:

  1. `effective_context_size` is the saved row width after windowing. For `max-prompt-pad` it equals the longest prompt
    observed in the materialized dataset; for packed modes and `fixed-context-pad` it equals the configured
    `--context-size`.
  2. `prompt_lengths` records the per-prompt pre-padding lengths only for example-aligned padded modes. Packed modes do
    not keep a 1:1 prompt-length sidecar because the saved rows no longer represent whole prompts.
  3. `shared_tokens_file` is the staged `tokens_<n_prompts_total>.pt` tensor that every layer run reuses instead of
    rebuilding prompt tokens in each layer directory.
  4. The staged effective-length sidecar is derived from `attention_mask` when the dataset keeps one row per prompt.
    That sidecar is what the runner uses for prompt-length bucketing and dynamic batch scaling.
  5. `prompt_bucket_schedule_file` is an explicit JSON schedule artifact. `auto_prompt_bucket_schedule` builds the same
    schedule directly from the staged effective-length sidecar when no explicit file is supplied.
  6. `prompt_bucket_ceilings` are optional explicit inclusive bucket ceilings. When left empty, the runner now derives a
    small ceiling set from prompt-length quantiles in the staged effective-length distribution instead of relying on a
    hardcoded global default.
  7. Streaming currently remains limited to the packed windowing family. Example-aligned pad-enabled modes
    (`max-prompt-pad` and `fixed-context-pad`) are intentionally still a future streaming follow-up because they require
    prompt-by-prompt length accounting and final-width resolution before the artifact contract can be finalized.

The current RTE full-prompt cache is:

```text
/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts
```

The current Monology dense cache is:

```text
/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_pile_uncopyrighted_context128_concat_2490
```

The RTE custom module still maps the `aps/super_glue:rte` train split through Interpretune's composable prompt
dataclasses from `1b_it_lightning_ct_ns_zs_test.yaml`, renders prompts with
`RTEBoolqGemmaPromptConfig.apply_chat_template_fn(..., add_generation_prompt=True)`, and pads each complete rendered
prompt to the measured maximum length without truncation. The saved `sae_lens.json` now keeps those RTE-specific fields
under a nested `custom` block while leaving the top-level metadata close to the upstream SAELens schema.

Monology does not need a custom module. It uses the upstream SAELens `pretokenize_dataset(...)` path directly with
`streaming=True`, `use_chat_formatting=True`, and a `2490`-row cap on the materialized token windows.

When `--prompts-pretokenized-dataset-path` is set, the pipeline now also materializes a shared prompt tensor cache once before the layer loop:

```text
/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts/tokens_2490.pt
```

Current behavior for future runs:

1. The pipeline builds `tokens_<n_prompts_total>.pt` once from the saved HuggingFace dataset.
2. The runner stages that file into each layer output directory as the local `tokens_<n_prompts_total>.pt` expected by `get_tokens()`.
3. The stage uses a symlink when possible and falls back to a plain copy if symlinks are unavailable.
4. The staged tensor stays on CPU until the actual feature/model forward path needs it, so per-layer token regeneration is no longer required.

For the current RTE cache, `tokens_2490.pt` has shape `(2490, 319)`. It preserves two exact duplicate rows because they
represent distinct RTE examples; the matching production config sets `deduplicate_shared_prompt_tokens: false` and
`strict_shared_prompt_count: true`.

### Full RTE generation-only rollout

The current command of record for the full `gemmascope-2-transcoder-262k-rte` generation run is the Bridge + structured-dataset path below. This is intentionally a generation-only launch: local layer-`0` import and explanation generation have already been validated, so the long-running `0-25` job keeps `--skip-local-db-import` to avoid coupling dashboard generation to local Postgres capacity.

```bash
HF_HOME=/mnt/cache_extended/speediedan/.cache/huggingface \
HF_DATASETS_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/datasets \
HF_HUB_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/hub \
IT_NP_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --model-name gemma-3-1b-it \
  --model-layers 26 \
  --sae-set gemma-scope-2-1b-it-transcoders-all \
  --neuronpedia-source-set-id gemmascope-2-transcoder-262k-rte \
  --neuronpedia-source-set-description 'Transcoder - 262k (RTE)' \
  --creator-name 'Google DeepMind' \
  --release-id gemma-scope-2 \
  --release-title 'Gemma Scope 2' \
  --release-url https://huggingface.co/google/gemma-scope-2-1b-it \
  --hf-weights-repo-id google/gemma-scope-2-1b-it \
  --hf-weights-path-template 'transcoder_all/layer_{layer}_width_262k_l0_small_affine' \
  --hook-point hook_mlp_in \
  --prompts-huggingface-dataset-path aps/super_glue \
  --prompts-huggingface-dataset-config-name rte \
  --prompts-huggingface-dataset-split train \
  --prompts-pretokenized-dataset-path /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts \
  --model-wrapper bridge \
  --bridge-enable-compatibility-mode \
  --runner-log-resource-snapshots \
  --start-layer 0 \
  --end-layer 25 \
  --start-batch 0 \
  --n-prompts-total 2490 \
  --n-tokens-in-prompt 319 \
  --n-features-per-batch 512 \
  --n-prompts-in-forward-pass 512 \
  --no-deduplicate-shared-prompt-tokens \
  --strict-shared-prompt-count \
  --no-archive-partials \
  --sae-path-template 'layer_{layer}_width_262k_l0_small_affine' \
  --python-executable /mnt/cache/speediedan/.venvs/it_latest/bin/python \
  --cuda-visible-devices 0 \
  --heartbeat-seconds 60 \
  --stall-timeout-seconds 1800 \
  --use-skip-transcoder \
  --skip-local-db-import \
  --run-root /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs
```

### Smaller-GPU background restart command

Use the command below for the current background restart when you want to run the 262k RTE generation path on the 8 GiB RTX 2070 SUPER while freeing the 24 GiB GPU. This is the command of record for the new no-truncation context-`319` full-prompt cache, not the old context-`128` comparison lineage.

Keep the knobs distinct when reasoning about this path: `n_features_per_batch` controls how many features are processed in a dashboard batch, `n_prompts_in_forward_pass` maps to SAEDashboard's logical prompt minibatch, and `primary_acts_batch_size` only chunks the internal activation-capture forwards inside each logical minibatch. After the latest SAEDashboard memory pass, the context-`319` GPU1 config can use `512x64 / acts16`; older `256x64 / acts16` notes are the conservative pre-memory-pass fallback.

```bash
cd ~/repos/interpretune && \
act it_latest && \
python scripts/launch_neuronpedia_dashboard_pipeline.py \
  --config /home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs/gemmascope-2-transcoder-262k-rte-gpu1.yaml \
  --run-name-suffix context319-full-prompts
```

The `--run-name-suffix context319-full-prompts` piece is what keeps the new full-prompt restart out of the old completed unsuffixed run namespace. The active GPU1 YAML now defaults to that same suffix, but keeping it explicit in the command makes the fresh lineage obvious in shell history and protects future ad hoc overrides.

Do not use the unsuffixed run directory below for the new full-prompt restart:

```text
/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/gemma-3-1b-it_gemmascope-2-transcoder-262k-rte
```

That unsuffixed namespace contains the earlier completed lineage and its `run.resume-0-25.log` markers. Reusing it will trigger completed-layer skips rather than a fresh generation start.

The launcher prints the exact `tail -f` target automatically. With the suffixed context-`319` launch, the logs now live under:

```text
/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/gemma-3-1b-it_gemmascope-2-transcoder-262k-rte_context319-full-prompts/launcher.<worker>.<timestamp>.log
```

The worker pipeline logs of record are:

```text
/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/gemma-3-1b-it_gemmascope-2-transcoder-262k-rte_context319-full-prompts/run.gpu1.resume-0-25.log
/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/gemma-3-1b-it_gemmascope-2-transcoder-262k-rte_context319-full-prompts/run.gpu0.resume-0-25.log
```

Resolved GPU1 settings in the config:

```yaml
run_name_suffix: context319-full-prompts
prompts_pretokenized_dataset_path: /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts
n_tokens_in_prompt: 319
n_features_per_batch: 512
n_prompts_in_forward_pass: 64
primary_acts_batch_size: 16
deduplicate_shared_prompt_tokens: false
strict_shared_prompt_count: true
cuda_visible_devices: "1"
```

The two-worker config keeps those conservative GPU1 settings and lets GPU0 use a larger prompt-forward shape:

```yaml
workers:
  - id: gpu1
    cuda_visible_devices: "1"
    start_layer: 3
    n_features_per_batch: 512
    n_prompts_in_forward_pass: 64
    primary_acts_batch_size: 16
  - id: gpu0
    cuda_visible_devices: "0"
    start_layer: 4
    n_features_per_batch: 1024
    n_prompts_in_forward_pass: 256
    primary_acts_batch_size: 64
```

The selected GPU1 probe was `512x64 / acts16` on layer `5`, batches `0-2`: `cuda_max_allocated_gib=4.96`, steady `cuda_reserved_gib=4.54-4.55`, and heartbeat GPU process memory about `5150 MiB`. The throughput-first GPU0 rerun was `1024x256 / acts64` on layer `6`, batches `0-2`: `cuda_max_allocated_gib=5.91`, steady `cuda_reserved_gib=4.17-4.18`, and about `819-826` features/min. A later aggressive GPU0 rerun at `1024x512 / acts256` still reached `cuda_max_allocated_gib=16.43` without OOM and improved to about `75.0s` on batch `1` (`~819 features/min`), but it only reached near-parity while carrying much higher VRAM residency.

Current tuning notes for this RTE Bridge path:

1. The old `48` / `64` prompt shape-mismatch blocker is fixed in the SAEDashboard Bridge wrapper.
2. The context-`128` tuning table remains useful for relative CPU/dashboard behavior, but the context-`319` full-prompt cache must be treated as a new memory regime.
3. On the 24 GiB card, `1024x256 / acts64` remains the shared monitored GPU0 throughput shape, while `1024x512 / acts256` is the optional high-VRAM validation shape that now reaches near-parity but still is not a clear win. On the 8 GiB card, use `512x64 / acts16` for the full-prompt cache.
4. Pretokenization is still worth using for full restarts because it removes repeated raw text tokenization and makes prompt coverage auditable before generation starts.
5. Do not use the `2048` band for this Bridge path unless memory headroom is irrelevant. Earlier optimized `2048x128` probes consumed high host and GPU memory while underperforming the best `512` rows.
6. Future runs no longer create a nested `layer_<n>/google/...` leaf when `model_id` contains `/`. The runner now sanitizes `google/gemma-3-1b-it` into a single path component like `google_gemma-3-1b-it_...` before building the leaf output directory.
7. The shared token cache stays CPU-backed through `before_feature_run_0`; the decisive GPU jump belongs to `SaeVisRunner.run(...)`, not token materialization.
8. Treat GPU1 as an independent worker target, not as a way to split a single runner. Two concurrent dashboard processes duplicate model/SAE residency and CPU packaging work, so monitor host RSS and cache IO if GPU0 and GPU1 are used at the same time.
9. A fresh context-`319` launch should start in a suffixed run directory and log `START layer=<n>` immediately, not a sequence of completed-layer skips from the old unsuffixed lineage. If you see the new pipeline warning that the requested layer range is already complete, you are still pointed at a finished lineage.
10. The 2026-05-02 two-worker validation assigned `layer_0` to GPU0 and `layer_1` to GPU1, wrote new batch artifacts from both workers, stopped GPU1 independently while GPU0 continued through another batch, cleaned stale locks on restart, and relaunched the two-worker config successfully.
11. Torch profiler traces are written as `batch-<n>.trace.json`; dashboard leaf discovery deliberately matches only exact `batch-<n>.json` files so those traces do not get mistaken for generated dashboard batches.

### Operational troubleshooting for GPU1 background runs

The 2026-05-01 production `gemmascope-2-transcoder-262k-rte` GPU1 run exposed a few operator lessons that are easy to misread from the logs.

1. A long pause after `Layer=<n> generation completed` is not automatically another generation stall. In the real layer-`19` incident, the runner child had already exited and the pipeline then spent about `9` minutes in the CPU-only conversion path before it finally logged `Converted layer=19 ...` and `DONE layer=19`. During that window there was no active GPU worker and no new `batch-*.json`, but the pipeline was still healthy.
2. The later layer-`20` and layer-`21` failures were real runner OOMs, but they were not in the Interpretune DB/import path. The traceback came from SAEDashboard / SAELens inside `SaeVisRunner.run(...) -> encoder.fold_W_dec_norm()`, where PyTorch tried to allocate another `1.12 GiB` after `batch-0.json` had already been written. The recent `neuronpedia_db_utils.py` extraction was not on that stack, and neither conversion nor local DB import had been reached yet.
3. A clean relaunch from the same YAML config is still the right first recovery step when the runner dies after writing some batches. The pipeline resumes completed layers from the log markers, and the SAEDashboard worker skips existing `batch-*.json` files inside the active layer directory. In the layer-`20` recovery, the restarted run skipped the existing `batch-0.json` and advanced through `batch-1`, `batch-2`, and into `batch-3` without changing the config.
4. If the same post-`batch-0.json` OOM reproduces on a later layer and the stack is still `fold_W_dec_norm()`, reduce `--n-prompts-in-forward-pass` by one power of two before touching `--primary-acts-batch-size`. `fold_W_dec_norm()` runs before the next internal activation-capture forward starts, so `primary_acts_batch_size` does not attack the failing allocation directly; lowering the outer logical minibatch is what buys back the missing pre-batch VRAM headroom.
5. Do not launch a second fresh lineage on GPU1 while another GPU1 dashboard worker is still resident. A later 2026-05-02 fresh-namespace smoke proved the new suffixing avoids old-log reuse, but it failed before `batch-0.json` when a separate live layer-`25` GPU1 worker was already holding about `4.4 GiB`; the second Bridge model init then OOMed during `hf_model.to(device)`.

Recommended response sequence for this run family:

1. Tail the pipeline log first. If the last new line is `Layer=<n> generation completed`, allow time for conversion before assuming the run is dead.
2. If the log shows `torch.OutOfMemoryError` or the parent process disappears while a worker continues briefly, treat that as a failed run. Stop the orphaned worker and relaunch the same config with `scripts/launch_neuronpedia_dashboard_pipeline.py`.
3. Only retune the config if the same seam reproduces cleanly after a fresh relaunch. For post-`batch-0.json` `fold_W_dec_norm()` OOMs, reduce `--n-prompts-in-forward-pass` first. Reserve `--primary-acts-batch-size` reductions for forward-time OOMs in `get_model_acts(...)`, `TransformerLensWrapper.forward(...)`, or `self.encoder.encode(...)` before the batch output is written.

### Import existing export bundles into the local DB

Use `--import-only-local-db` when dashboards were already generated with `--skip-local-db-import` and you later want to import the existing export bundles into the local Neuronpedia DB without rerunning generation.

```bash
LOCAL_NEURONPEDIA_DB_URL='postgres://postgres:postgres@127.0.0.1:5433/postgres' \
/mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --model-name gemma-3-1b-it \
  --model-layers 26 \
  --sae-set gemma-scope-2-1b-it-transcoders-all \
  --neuronpedia-source-set-id gemmascope-2-transcoder-262k-rte \
  --neuronpedia-source-set-description 'Transcoder - 262k (RTE)' \
  --creator-name 'Google DeepMind' \
  --release-id gemma-scope-2 \
  --release-title 'Gemma Scope 2' \
  --release-url https://huggingface.co/google/gemma-scope-2-1b-it \
  --hf-weights-repo-id google/gemma-scope-2-1b-it \
  --hf-weights-path-template 'transcoder_all/layer_{layer}_width_262k_l0_small_affine' \
  --hook-point hook_mlp_in \
  --prompts-huggingface-dataset-path aps/super_glue \
  --start-layer 10 \
  --end-layer 10 \
  --sae-path-template 'layer_{layer}_width_262k_l0_small_affine' \
  --import-only-local-db
```

Notes:

1. `--import-only-local-db` is the complement to `--skip-local-db-import`; do not combine them.
2. In this mode the pipeline skips generation and conversion entirely, finds an existing export bundle under `export_root/<model_name>/<layer>-<source_set>*`, and imports it directly.
3. The mode does not rely on completed-layer log markers, so it is safe for backfilling local DB rows after a generation-only run.

### Temporary conversion CUDA debug mode

The pipeline also has a temporary troubleshooting flag, `--conversion-cuda-debug`, which records PyTorch CUDA allocator snapshots around the generation-to-conversion seam and at key points inside `convert-saedashboard-to-neuronpedia-export.py`.

Artifacts are written under:

```text
<run_root>/<run_name>/conversion_cuda_snapshots/layer_<n>/
```

This instrumentation was added for the 2026-04-30 `gemmascope-2-transcoder-262k-rte` diagnosis and should be treated as debug-only until we decide whether it belongs in a future PR.

Current finding from the layer-`10` diagnostic run: the pipeline process held `0.0 GiB` PyTorch-managed CUDA allocated and reserved at every conversion checkpoint, and every saved snapshot contained zero allocator segments. The converter therefore does not explain the previously suspected multi-GiB CUDA growth; that apparent spike comes from the runner-side handoff window or other non-PyTorch CUDA context residency, not from `convert-saedashboard-to-neuronpedia-export.py` itself.

### Profiling the RTE dashboard path

Use the profiling harness when a future run needs function-level attribution or resource-only batch-shape comparison:

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  /home/speediedan/repos/interpretune/scripts/profile_neuronpedia_dashboard_generation.py \
  --config primary-512x512:512:512 \
  --target-batches 3
```

Add `--no-py-spy` for clean throughput/resource measurements. Keep py-spy enabled for attribution runs; the harness now sends `SIGINT` first so speedscope output is flushed, and it aggregates every subprocess profile from `py-spy --subprocesses`. Use `--cuda-visible-devices 1` and `--primary-acts-batch-size 64` for the smaller-GPU feasibility probe.

The original `512x128` pretokenized Bridge profile showed the steady-state bottleneck was CPU-side SAEDashboard dashboard packaging, not IO or GPU compute:

| Config | Avg batch s | Features/min | Max RSS GiB | Max GPU MiB | Avg batch GPU util | Cache IO util |
| --- | --- | --- | --- | --- | --- | --- |
| `512x128` | `29.4` | `1044` | `4.92` | `5902` | about `1-2%` | about `0.1-0.2%` |
| `512x256` | `28.9` | `1063` | `5.00` | `7542` | about `1-2%` | about `0.2%` |
| `1024x128` | `57.2` | `1075` | `6.89` | `6282` | about `1-2%` | about `0.1%` |

The top `py-spy` leaf is `HistogramData.from_data(...)` in SAEDashboard's `utils_fns.py`, called from the logits-histogram branch of `SaeVisRunner.run(...)`; `utils_fns.py` accounts for about `61%` of all aggregated samples. The next optimization target is therefore SAEDashboard's per-feature histogram/statistics/sequence packaging path, not another feature-batch or prompt-forward sweep. A fuller write-up is tracked in `/home/speediedan/repos/distributed-insight/project_admin/interpretune/design/circuit-tracer-backend/new_dashboard_exploration/dashboard_generation_profiling.md`.

After the first optimization pass, the best clean `512x512` probe measured `11.3s` average batch time and `2715` features/min. The matching py-spy attribution run measured `2469` features/min and shifted the next visible hotspot to sequence packaging (`package_sequences_data(...)` and `get_indices_dict(...)`), while JSON/converter work remained secondary. The generation-only projection is now about `1.7-1.9` days for all `26` layers before conversion/import/explanation overhead.

The successful RTX 2070 SUPER probe used:

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  /home/speediedan/repos/interpretune/scripts/profile_neuronpedia_dashboard_generation.py \
  --config gpu1-512x256-mf64-logitscpu:512:256 \
  --layer 10 \
  --target-batches 2 \
  --session-root /tmp/np_dashboard_generation_profiles_gpu1_logitcpu \
  --run-root /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs_gpu1_logitcpu_probe \
  --primary-acts-batch-size 64 \
  --cuda-visible-devices 1 \
  --max-tree-rss-gib 24
```

Result: `target_reached`, `2` batches, `13.5s` average batch time, `2271` features/min, max tree RSS `4.73 GiB`, and max GPU process memory `5564 MiB`. A `KeyboardInterrupt` at the end of the layer log is expected for this probe because the profiling harness intentionally terminates the child process after it observes the target batch count.

## CLT + structured dataset example

The same pipeline now supports the current GemmaScope2 CLT layout for `google/gemma-scope-2-1b-it`, where each layer lives in a shared local directory as `config.json` plus `params_layer_<n>.safetensors`.

First, prefetch the CLT directory so every layer file is present before the long run reaches later layers:

```bash
CLT_SNAPSHOT=$(/mnt/cache/speediedan/.venvs/it_latest/bin/python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download('google/gemma-scope-2-1b-it', allow_patterns=['clt/width_262k_l0_medium_affine/*']))
PY
)

CLT_DIR="${CLT_SNAPSHOT}/clt/width_262k_l0_medium_affine"
```

### Layer-0 smoke

Use this small smoke command to validate generation, conversion, and local DB import on one batch:

```bash
LOCAL_NEURONPEDIA_DB_URL='postgres://postgres:postgres@127.0.0.1:5433/postgres' \
/mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --model-name gemma-3-1b-it \
  --model-layers 26 \
  --sae-set gemma-scope-2-1b-it-clt-all \
  --neuronpedia-source-set-id gemmascope-2-clt-262k-rte \
  --neuronpedia-source-set-description 'CLT - 262k (RTE smoke)' \
  --creator-name 'Google DeepMind' \
  --release-id gemma-scope-2 \
  --release-title 'Gemma Scope 2' \
  --release-url https://huggingface.co/google/gemma-scope-2-1b-it \
  --hf-weights-repo-id google/gemma-scope-2-1b-it \
  --hf-weights-path-template 'clt/width_262k_l0_medium_affine/params_layer_{layer}.safetensors' \
  --hf-model-path google/gemma-3-1b-it \
  --hook-point hook_mlp_in \
  --prompts-huggingface-dataset-path aps/super_glue \
  --prompts-huggingface-dataset-config-name rte \
  --prompts-huggingface-dataset-split train \
  --prompts-pretokenized-dataset-path /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts \
  --model-wrapper bridge \
  --bridge-enable-compatibility-mode \
  --runner-log-resource-snapshots \
  --start-layer 0 \
  --end-layer 0 \
  --start-batch 0 \
  --end-batch 0 \
  --n-prompts-total 64 \
  --n-tokens-in-prompt 319 \
  --n-features-per-batch 8 \
  --no-deduplicate-shared-prompt-tokens \
  --strict-shared-prompt-count \
  --sae-path-template "${CLT_DIR}" \
  --python-executable /mnt/cache/speediedan/.venvs/it_latest/bin/python \
  --cuda-visible-devices 0 \
  --use-clt
```

The config-backed equivalent for the full rollout lives at:

```text
/home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs/gemmascope-2-clt-262k-rte.yaml
```

Launch it with:

```bash
/mnt/cache/speediedan/.venvs/it_latest/bin/python \
  /home/speediedan/repos/interpretune/scripts/launch_neuronpedia_dashboard_pipeline.py \
  --config /home/speediedan/repos/distributed-insight/project_admin/interpretune/admin_scripts/neuronpedia_dashboard_generation/configs/gemmascope-2-clt-262k-rte.yaml
```

Validated result for this smoke path:

1. layer-`0` dashboard output wrote `batch-0.json`
2. the pipeline converted `/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports/gemma-3-1b-it/0-gemmascope-2-clt-262k-rte`
3. the local import summary was `SourceSet=1`, `Source=1`, `Neuron=8`, `Activation=256`

### Full RTE rollout

This is the current command of record for the full `0-25` CLT RTE rollout:

```bash
LOCAL_NEURONPEDIA_DB_URL='postgres://postgres:postgres@127.0.0.1:5433/postgres' \
/mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --model-name gemma-3-1b-it \
  --model-layers 26 \
  --sae-set gemma-scope-2-1b-it-clt-all \
  --neuronpedia-source-set-id gemmascope-2-clt-262k-rte \
  --neuronpedia-source-set-description 'CLT - 262k (RTE)' \
  --creator-name 'Google DeepMind' \
  --release-id gemma-scope-2 \
  --release-title 'Gemma Scope 2' \
  --release-url https://huggingface.co/google/gemma-scope-2-1b-it \
  --hf-weights-repo-id google/gemma-scope-2-1b-it \
  --hf-weights-path-template 'clt/width_262k_l0_medium_affine/params_layer_{layer}.safetensors' \
  --hf-model-path google/gemma-3-1b-it \
  --hook-point hook_mlp_in \
  --prompts-huggingface-dataset-path aps/super_glue \
  --prompts-huggingface-dataset-config-name rte \
  --prompts-huggingface-dataset-split train \
  --prompts-pretokenized-dataset-path /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts \
  --model-wrapper bridge \
  --bridge-enable-compatibility-mode \
  --runner-log-resource-snapshots \
  --start-layer 0 \
  --end-layer 25 \
  --start-batch 0 \
  --n-prompts-total 2490 \
  --n-tokens-in-prompt 319 \
  --n-features-per-batch 128 \
  --n-prompts-in-forward-pass 32 \
  --no-deduplicate-shared-prompt-tokens \
  --strict-shared-prompt-count \
  --no-archive-partials \
  --sae-path-template "${CLT_DIR}" \
  --python-executable /mnt/cache/speediedan/.venvs/it_latest/bin/python \
  --cuda-visible-devices 0 \
  --use-clt
```

Validated runtime envelope for the corresponding direct layer-`0` one-batch probe on the real `2490`-prompt workload:

1. `0:40.88` wall time
2. `7,004,248 kB` max RSS
3. `cuda_max_allocated_gib=4.35`
4. `post_batch_0` RSS `2.26 GiB`

That probe used `float32` model weights in the direct runner. The pipeline keeps `model_dtype=bfloat16` by default, so the full job has additional GPU headroom beyond the measured probe.

The first full layer-`1` CLT attempt later exposed a deeper mixed-precision seam: the Bridge path supplies `bfloat16` activations while the local CLT weights stay `float32`, so `CLTLayerWrapper.encode()` now casts activations to the CLT weight dtype before `F.linear(...)`. The exact failed layer-`1` repro is now past the old crash point, and the full `0-25` run has been resumed from layer `1` under the same `128/32` shape.

## Restarting a paused or killed run

For batch-level resume inside a layer, use the exact same pipeline command and keep these constraints:

1. Always include `--no-archive-partials`. Without it, the outer pipeline renames the partial `layer_<n>` directory and destroys the inner runner's `batch-*.json` skip state.
2. Keep the same `run_directory`, `neuronpedia-source-set-id`, and `sae-path-template`.
3. Re-run the same command after the pause; do not advance `--start-batch` manually.
4. Keep the same `run.resume-<start>-<end>.log` file. The pipeline now parses timestamped `DONE layer=<n>` lines there and skips already completed layers before it relaunches the inner runner.
5. Validate the restart by checking the pipeline log for preserved-batch skips followed by the next missing batch.

Example validation commands:

```bash
grep -E 'Skipping Batch #9|Running Batch #10' \
  /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/ \
  gemma-3-1b-it_gemmascope-2-clt-262k-rte/run.resume-0-25.log

find /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/ \
  gemma-3-1b-it_gemmascope-2-clt-262k-rte/layer_0 -name 'batch-*.json' | wc -l
```

Current validated behavior on this host:

1. the first launch was stopped after the first `10` completed layer-`0` batch files
2. the restart log showed `Skipping Batch #8`, `Skipping Batch #9`, then `Running Batch #10`
3. layer `0` later completed and imported locally, but the first full layer-`1` attempt then failed on a `bfloat16` vs `float32` matmul inside the CLT wrapper
4. after fixing that dtype seam, the next relaunch logged `Skipping already completed layer=0 based on existing logs.` and resumed at `START layer=1`

## Local explanation note

When generating explanations against locally imported custom source sets, pass `local_db_url=` explicitly to the explanation helper in this environment. The layer-`0` `gemmascope-2-transcoder-262k-rte` and `gemmascope-2-clt-262k-rte` dashboards both now support `np_moe-max-act-logits`, but that required one helper-side fix: if cached activation batches are missing for a local custom source set, `interpretune.utils.neuronpedia_explanations` now falls back to the feature API payload's embedded `activations` rows instead of hard-failing.

## Model metadata guardrail

The local import path does more than features and explanations. `import_neuronpedia_export_bundle_local_db(...)` also imports
bundle-scoped metadata tables such as `model.jsonl` and `release.jsonl`. Those inserts use `ON CONFLICT DO NOTHING`, which
means a correct existing local `Model` row is preserved, but a missing row in a fresh local DB is seeded directly from the
bundle metadata.

That import seam was the real cause of the earlier broken local Gemma graph rows. Older Neuronpedia utility exports were
still emitting `layers: 0` and omitting `defaultGraphSourceSetName`, so importing one of those stale bundles could recreate
a `Model` row that broke `/graph` even though graph upload itself was fine.

Current supported fix:

1. the converter now requires `--model-layers` and writes both `defaultSourceSetName` and `defaultGraphSourceSetName`
2. the checked-in `gemma-3-1b-it` export examples under `neuronpedia_utils/exports/` have been refreshed to match
   `gemmascope-2-transcoder-16k`

If your local DB was already seeded from a pre-fix bundle, repair that `Model` row once in the live DB. After that, stick
to regenerated or refreshed bundles rather than re-importing older `model.jsonl` and `release.jsonl` artifacts.

## What the pipeline now handles automatically

### Nested dashboard leaf resolution

SAEDashboard layer directories do not always place `batch-*.json` files directly under `layer_<n>`. For the Gemma Target B run, the actual converter input for layer `23` was:

```text
.../layer_23/google/gemma-3-1b-it_gemma-scope-2-1b-it-transcoders-all_blocks.23.hook_mlp_in_16384
```

The converter script expects the directory that directly contains `batch-*.json`, not the layer root. The Interpretune pipeline now resolves that leaf directory automatically before conversion.

### Export-root recovery

After conversion, the pipeline resolves the expected export root from the real batch metadata and falls back across candidate export directories if the direct path is absent.

### Diagnostics

At every heartbeat, the pipeline logs:

1. output directory file count
2. output directory byte count
3. `ps` snapshot for the inner runner
4. `nvidia-smi` snapshot for GPU memory/process state

On stall or nonzero exit, it also logs a kernel snapshot via `dmesg`.

## Monitoring

To follow a live run, inspect the file-backed launch log:

```bash
tail -f /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/gemma-3-1b-it_gemmascope-2-transcoder-16k/run.resume-24-25.launch.log
```

Healthy heartbeat lines look like:

```text
INFO Heartbeat output_dir=... files=2 bytes=1639 elapsed_without_growth=60.1s pid=... ps=... gpu=...
```

If bytes are static for too long, use the logged `ps`, GPU, and kernel snapshots before assuming the job is stuck in Python logic.

## Local import validation

If you want to validate an already-converted bundle directly, use the Python helper with the explicit localhost Postgres URL:

The helper is now a handoff boundary rather than the owner of Neuronpedia table mechanics. Interpretune requires
`neuronpedia_utils.local_db_import` for local dashboard import and import-mode benchmark functionality, then adds only
Interpretune-local URL resolution for localhost Docker Postgres URLs. The current `it_latest` environment has been
validated with an editable no-deps install of the local Neuronpedia utility package.

```bash
cd "${IT_REPO_DIR}"
source "${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate"

python - <<'PY'
from pathlib import Path
from interpretune.utils import import_neuronpedia_export_bundle_local_db

bundle = Path(
  '${NEURONPEDIA_REPO_DIR}/utils/neuronpedia-utils/neuronpedia_utils/exports/'
    'gemma-3-1b-it/23-gemmascope-2-transcoder-16k'
)
summary = import_neuronpedia_export_bundle_local_db(
    bundle,
    local_db_url='postgres://postgres:postgres@127.0.0.1:5433/postgres',
)
print(summary)
PY
```

For import timing, prefer one no-rollback benchmark mode per clean database or schema so binary `COPY FROM STDIN` timings
are not distorted by rollback bookkeeping. The current benchmark modes to run separately are `activation_arrow_copy`,
`activation_parquet_copy`, and, if reset cost is acceptable, `activation_arrow_jsonb`, `activation_parquet_jsonb`, and
legacy `jsonl`.

The first clean-schema timing pass used `/tmp/np_local_import_benchmark_subset_20260511`, a three-shard real export subset
with generated activation Arrow/Parquet sidecars. All modes inserted `20,240/20,240` activation rows. COPY reduced the
activation insert phase to about `2.5s`, versus about `6.5s` for JSONB, but the tiny regenerated sample remained
load/alignment dominated: `activation_arrow_copy` took `9.82s` process wall, `activation_parquet_copy` took `10.22s`, and
legacy compressed `jsonl` took `9.19s`. Treat that as a functional gate and sizing clue, not a final production import
throughput result.

If you are validating a bundle outside this repo, inspect its `model.jsonl` and `release.jsonl` first. A pre-fix bundle can
still seed stale `Model` metadata into an otherwise fresh local DB.

## How notebook explanation backfill uses dashboard exports

The dashboard/import pipeline and the localhost concept-direction notebook now share the same export artifacts.

The explanation path now behaves like a layered cache hierarchy:

1. `Explanation` rows in the local Neuronpedia DB
2. feature-specific Interpretune cache under `IT_NP_CACHE/.../feature-activations/feature-<index>.jsonl.gz`
3. local Neuronpedia export batches under `<neuronpedia_repo>/.../exports/<model>/<layer-source-set>/activations/batch-*.jsonl.gz`
4. remote public activation-batch fallback downloaded into `IT_NP_CACHE/.../activations/batch-<n>.jsonl.gz`

That layering matters because the notebook uses different recovery steps depending on what already exists.

### Retrieval-only case: explanation already exists locally

If `check_local_explanation_coverage(...)` finds an `Explanation` row for the target `(modelId, layer, index)` tuple, the pipeline does not generate anything.

The path is:

1. notebook computes candidate feature refs
2. local DB coverage check returns `has_local_explanation=True`
3. generation is skipped for that feature
4. the webapp renders the stored `Explanation.description`

No export scan, feature-cache write, or remote download happens in this case.

### Local-source case: explanation missing, activations already available locally

If the explanation is missing but the activation source data is already available, the notebook stays entirely local.

If the feature cache already exists:

1. `load_cached_feature_activations(...)` reads `feature-<index>.jsonl.gz`
2. prompt construction and explanation generation run immediately
3. the new explanation is inserted into the local DB
4. the feature-specific cache file is deleted after successful insert

If only the local export batches exist:

1. `prepare_local_explanation_backfill(...)` scans `activations/batch-*.jsonl.gz`
2. it filters the rows for the exact feature index
3. it writes a feature-specific cache file under `feature-activations/`
4. explanation generation consumes that reduced cache
5. after successful insert, the feature-specific cache file is removed automatically

This is the preferred localhost flow because it reuses the exact export bundle produced by the dashboard pipeline.

### Remote fallback case: explanation missing and no local activations are present

If neither the local export bundle nor the feature cache has the needed activations, explanation generation falls back automatically to the public dataset endpoint.

Current implementation:

1. `load_cached_feature_activations(...)` checks local feature cache and local batch cache first
2. if neither exists, it derives candidate public batch URLs from `candidate_public_activation_batch_urls(...)`
3. `download_url_to_path(...)` downloads the batch automatically to:
  - `IT_NP_CACHE/v1/<model>/<layer-source-set>/activations/batch-<n>.jsonl.gz`
4. the downloaded batch is filtered to the requested feature rows
5. generation continues without any manual user action

Important detail:

- the current fallback remote base is `https://neuronpedia-datasets.s3.amazonaws.com`
- the downloaded file lands in the durable Interpretune batch cache, not in `/tmp`
- this is automatic inside explanation generation; the notebook user does not need a separate preload step

The relevant pieces are:

1. dashboard generation creates Neuronpedia export bundles under:
  - `${NEURONPEDIA_REPO_DIR}/utils/neuronpedia-utils/neuronpedia_utils/exports/<model>/<layer-source-set>`
2. each bundle includes `activations/batch-*.jsonl.gz`
3. the localhost concept-direction notebook calls `prepare_local_explanation_backfill(...)` before `ensure_local_feature_explanations(...)`
4. that helper searches the local export `activations/` batches for the exact feature index and writes a feature-specific Interpretune cache artifact under:
  - `IT_NP_CACHE/v1/<model>/<layer-source-set>/feature-activations/feature-<index>.jsonl.gz`
5. the explanation-generation helper then reads those cached rows to build/import explanations into the local DB

This is now the preferred local path because exported activation batch numbering is not guaranteed to match the older public-dataset batch-number heuristic. Searching the local export bundle by exact feature index is more reliable for localhost runs.

### Cleanup behavior after successful insert

The explanation pipeline now treats feature-specific cache files as transient staging artifacts.

After `generate_explanation_artifact(...)` successfully inserts the explanation into the local DB:

1. `cleanup_feature_activation_cache(...)` checks whether the activations came from `feature-activations/feature-<index>.jsonl.gz`
2. if so, it deletes that file and removes the empty parent directory when possible
3. batch caches under `IT_NP_CACHE/.../activations/batch-<n>.jsonl.gz` are kept so later features from the same batch do not trigger another download

That means:

- feature caches are disposable
- batch caches are reusable
- the dashboard export bundle remains the long-lived local source of truth

### Manual cache warm-up checklist

If a localhost notebook reports explanation cache misses:

1. verify the export bundle exists for the relevant layer/source-set
2. verify `activations/batch-*.jsonl.gz` is present in that bundle
3. rerun the notebook with `CHECK_LOCAL_EXPLANATION_COVERAGE=true` and `GENERATE_MISSING_LOCAL_EXPLANATIONS=true`
4. inspect the notebook’s `local_explanation_prefetch` JSON block
5. confirm the derived `feature-activations/feature-<index>.jsonl.gz` file exists under `IT_NP_CACHE`

For manual inspection on this host:

```bash
find "${NEURONPEDIA_REPO_DIR}/utils/neuronpedia-utils/neuronpedia_utils/exports/gemma-3-1b-it/19-gemmascope-2-transcoder-16k/activations" -maxdepth 1 -name 'batch-*.jsonl.gz' | head

ls -l /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/v1/gemma-3-1b-it/19-gemmascope-2-transcoder-16k/feature-activations/
```

If you need to force regeneration of previously inserted Copilot-generated explanations before a rerun, delete those rows from the local DB first and then rerun the localhost config:

```sql
DELETE FROM "Explanation"
WHERE "modelId" = 'gemma-3-1b-it'
  AND layer LIKE '%-gemmascope-2-transcoder-16k'
  AND notes LIKE '%interpretune-github-copilot-cli%';
```

## Current operational caution

The pipeline is currently running from the shared `it_latest` environment because the documented Neuronpedia-specific environment on this host is incomplete. That is acceptable for active recovery, but long-running dashboard jobs should eventually move into a dedicated environment with the Neuronpedia and SAEDashboard dependencies pinned independently from the main Interpretune dev environment.
