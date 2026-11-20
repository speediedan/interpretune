# Configuration Guidance

Practical guidance for composing interpretune configurations. See
{doc}`usage/session_module_datamodule_usage` for the session/config API itself.

## Keep the composition space flat

Interpretune deliberately keeps configuration composition **flat**: configs compose by explicit
inclusion/override rather than deep implicit inheritance hierarchies. When authoring new configs,
prefer adding a sibling config (or a small override file) over introducing new nesting levels —
flatness is what keeps the composition space legible and greppable as the config surface grows.

## Group configs by model, then experiment

Configuration files can technically be composed arbitrarily, but the convention used throughout
the examples — and recommended for your own experiments — is grouping by **model level** (shared
per-model substrate settings) and **experiment level** (task/experiment-specific settings layered
on a model config). See `tests/benchmarks/configs/` for the reference layout
(e.g. `rte_boolq/gemma2/...`, `rte_boolq/gemma3/...`).

## TL-native configs: `model_name_or_path` can be arbitrary

When instantiating a config-only TransformerLens model via interpretune config, the ITLens
config's `model_name_or_path` can be set to an arbitrary identifier — the TL pretrained-model
resolution is driven by the TL config's own model selection, so `model_name_or_path` functions as
a label rather than a required HF-hub pointer in that mode.

## CLI: `auto_comp_cfg` is not supported via jsonargparse

The `auto_comp_cfg` mechanism (automatic composition-config resolution,
`interpretune.config.shared.AutoCompConfig`) is **not supported through the CLI**: jsonargparse's
signature-driven parsing cannot represent its dynamic target-class rebinding. Configure
`auto_comp_cfg` programmatically (or via YAML consumed by the session API) instead of CLI flags.

## Lightning precision plugins only convert floating-point tensors

When composing with the Lightning adapter, note that Lightning precision plugins convert inputs
via `apply_to_collection(..., function=...)` targeting **floating-point tensors only**. Integer
tensors — token IDs, attention masks, embedding indices (`LongTensor`) — pass through unconverted,
which is exactly what embedding layers require. If you see mixed-precision surprises, audit the
float tensors in your batch; the integer inputs are not the issue.

## PyTorch profiling: `profiler_cfg` and `--trainer.profiler`

Each training-loop interface configures the PyTorch profiler its own way, and the two accept the same summary options.

**Core loop.** A module's `profiler_cfg` runs the core runners' train, test and analysis loops under
`torch.profiler.profile`, stepped once per batch:

```yaml
profiler_cfg:
  which: pytorch
  pytorch_profiler_cfg:
    activities: [cpu, cuda]
    schedule: {wait: 1, warmup: 1, active: 3}  # torch.profiler.schedule's arguments
    record_shapes: true
    profile_memory: true
    trace_dir: profiler_logs        # default: <core_log_dir>/pytorch_profiler
    filename: rte_profile           # write <trace_dir>/<phase>-rte_profile.txt
    sort_by_key: cpu_time_total
    row_limit: 50
    max_name_column_width: 100
```

Every window the schedule records is written as a TensorBoard trace to `trace_dir`. With `filename` set, the loop also
writes a table of the last recorded window's operators to `<trace_dir>/<phase>-<filename>.txt`, where `<phase>` is
`train`, `test` or `analysis` (a distributed run appends `-<rank>`). `sort_by_key` (default `cpu_time_total`),
`row_limit` (default 20) and `max_name_column_width` (default 55) shape that table, and setting them without
`filename` is refused. They are the names Lightning's `PyTorchProfiler` and finetuning-scheduler's example
`ExtendedPyTorchProfiler` use, so a configuration moves between the two interfaces unchanged, with one difference:
Lightning accepts a fixed set of sort keys (`cpu_time_total`, `cuda_time_total`, `count` and the memory columns among
them), while the core loop accepts any column of the table, `self_cpu_time_total` included. The core CLI takes the
same settings as flags:

```bash
interpretune --run_command test --config <session config> \
  --profiler_cfg.which pytorch \
  --profiler_cfg.pytorch_profiler_cfg '{"activities": ["cpu"], "filename": "rte_profile", "row_limit": 50}'
```

**Lightning.** A Lightning `Trainer` runs its own loops, so a Lightning composition refuses `profiler_cfg` and uses
Lightning's own profiler argument, in a configuration's `trainer` block or as `--trainer.profiler`:

```yaml
trainer:
  profiler:
    class_path: lightning.pytorch.profilers.PyTorchProfiler
    init_args:
      dirpath: profiler_logs
      filename: rte_profile         # write <dirpath>/<stage>-rte_profile.txt
      sort_by_key: cpu_time_total
      row_limit: 50
    dict_kwargs:                    # passed to torch.profiler.profile
      record_shapes: true
      profile_memory: true
```

Lightning's `PyTorchProfiler` takes `schedule` only as a callable, which a configuration file cannot hold. For a
schedule and `max_name_column_width` from configuration, point `class_path` at
`fts_examples.profiling.extended_profiler.ExtendedPyTorchProfiler` from finetuning-scheduler's examples, which adds
`schedule_cfg` (the arguments of `torch.profiler.schedule`) and `max_name_column_width` to the same `init_args`.

Enable one profiler per run: the PyTorch profiler or the memory profiler below, not both. `ITConfig` refuses a
configuration that enables both.

## Memory profiling: `memprofiler_cfg`

The `memprofiler` extension runs on finetuning-scheduler's profiling core (`finetuning_scheduler.profiling`), the same
profiler finetuning-scheduler uses, so its configuration is the core's schema field for field. It needs
finetuning-scheduler, which interpretune's `tune` extra installs (`pip install 'interpretune[tune]'`); enabling it without
the extra is refused, naming the extra.

```yaml
memprofiler_cfg:
  enabled: true
  collect_funcs:
    default: [training_step, test_step]  # added to every collection type below
    accelerator: []      # accelerator statistics (CUDA, ROCm, XPU)
    allocator_history: []  # allocator history snapshots, with `allocator_history: true`
    cpu: []              # host memory (RSS, VMS)
    fsdp: []             # FSDP2 memory tracking, with `track_fsdp_mem: true`
  allocator_history: true
  schedule:
    warmup_iters: 1      # iterations count from 1; these are skipped
    max_iter: 4          # collect through this iteration (unset: one iteration after the warmup)
  dump_memorystats_yaml: true  # rank_<r>_memory_stats.yaml under <core_log_dir>/memprofiler (or save_dir)
```

A step method decorated with `MemProfilerHooks.memprofilable` is profiled once per call: a snapshot at its start and
end, keyed `rank.fn_name.iter_idx.iter_ctx` (for example `0.training_step.2.end`). Collection a host cannot perform (an
accelerator statistic without an accelerator, allocator history on a backend without it) is disabled with a warning. The
remaining fields (memory and saved-tensor hooks, the FSDP2 tracker's depth, units and root module, report aggregation)
are documented on finetuning-scheduler's `MemProfilerCfg`. Run the PyTorch profiler and the memory profiler in separate
runs (see the design rationale).

In the core training loop, one profiled `training_step` call is a whole training iteration: the forward pass, the
backward pass and, at the end of an accumulation window, the optimizer step. A snapshot taken at the end of the call
therefore sees the gradients and the optimizer state. A `memprofilable` step method that runs inside that call is not
profiled a second time. Under Lightning, `training_step` remains the forward pass alone, since Lightning runs the
backward pass and the optimizer step itself.

To profile a run under the `fsdp2` placement, set `track_fsdp_mem: true` and list `training_step` under
`collect_funcs.fsdp`:

```yaml
memprofiler_cfg:
  enabled: true
  track_fsdp_mem: true
  collect_funcs:
    default: []
    fsdp: [training_step]
  schedule:
    warmup_iters: 0
    max_iter: 6
```

Each rank then writes one report per profiled iteration to `<core_log_dir>/memprofiler`, together with a
`rank_<r>_fsdp_memory_report.json` that holds the per-category peak and the module-wise forward and backward states of
every iteration. The tracker is built at the first profiled iteration, over the sharded module and the loop's
optimizer. An empty `fsdp_mem_tracker_root_module` names the root `fully_shard` was applied to, which under the core
loop is the interpretune module itself; any other path is resolved under the module's model. In a phased run
(`FTSCoreCallback`), each report entry records the schedule depth it ran at, and the tracker is rebuilt over the new
optimizer at each phase transition.
