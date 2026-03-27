# Benchmark Test Suite

End-to-end benchmarks that validate experiment reproducibility by running interpretune configs via the CLI and checking accuracy against the [benchmark registry](benchmark_registry.yaml).

## CLI Modes

The benchmark runner supports two CLI modes, auto-detected from the registry entry:

- **Lightning CLI** (`adapter_ctx` includes `lightning`): Uses `interpretune --lightning_cli test --config <yaml>` with a full Lightning Trainer. Batch limiting via `--trainer.limit_test_batches`.
- **Core CLI** (`adapter_ctx` uses `core`): Uses `interpretune --run_command test --config <yaml>` with the `SessionRunner`. Batch limiting via `--run_cfg.limit_test_batches`. Core configs use `ClassificationMixin.on_test_epoch_end()` for accuracy reporting.

To explicitly set the CLI mode for a registry entry, add `cli_mode: core` or `cli_mode: lightning`. If omitted, the mode is inferred from `adapter_ctx`.

## Quick Start

```bash
# Activate environment
cd ~/repos/interpretune && source /mnt/cache/$USER/.venvs/it_latest/bin/activate

# Run all benchmarks via pytest
IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py -v

# Run a single benchmark via the CLI runner
python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l

# Run via special_tests.sh harness
./tests/special_tests.sh --mark_type=benchmark
```

## Structure

```
tests/benchmarks/
├── README.md                   # This file
├── benchmark_registry.yaml     # Expected accuracy targets per config
├── benchmark_utils.py          # Shared diagnostic utilities
├── test_benchmarks.py          # Pytest parametrized tests (RunIf benchmark mark)
├── run_benchmarks.py           # Standalone CLI runner with debug support
└── debug_utils/                # Per-experiment diagnostic scripts
    └── rte_boolq/
        ├── dbg_rte_boolq.py        # Entailment-specific diagnostics
        ├── debug_batch_collapse.py  # Batch collapse debug tool
        └── debug_it_pipeline.py     # Pipeline debug tool
```

## Registry Format

The registry (`benchmark_registry.yaml`) uses a nested structure:

```yaml
benchmarks:
  <experiment_name>:           # e.g., rte_boolq
    <benchmark_id>:            # e.g., gemma2_2b_it_l
      config_path: ...         # YAML config path relative to repo root
      expected_accuracy: 0.809 # Target metric
      tolerance: 0.03          # Acceptable deviation from target
      cli_mode: lightning      # Optional: "lightning" or "core" (auto-detected from adapter_ctx)
      adapter_ctx: [...]       # Adapters composing the ITSession
      backends: [...]          # Model/analysis backends
      description: "..."       # Human-readable description
      last_validated: "..."    # YYYYMMDD_HH24MMSS local time
      commit_sha: "..."        # Short git SHA at validation time
      salient_pkg_versions:    # Package versions at validation time
        interpretune: "0.1.0.dev... (fork:speediedan/interpretune, branch:circuit-tracer-backend, sha:abcd123)"
        circuit_tracer: "0.4.0 (fork:speediedan/circuit-tracer, sha:14cc3e8)"
        torch: "..."
        # ... etc
      notes: "..."             # Additional context
      tags: [...]              # Filtering tags
      debug_utils_module: ...  # Optional: experiment-specific debug module
```

`salient_pkg_versions` keeps plain versions for ordinary PyPI installs and appends best-effort git provenance when it is
recoverable:

- Editable installs use the live checkout to record `fork`, `branch`, and `sha`.
- Git-backed non-editable installs use `direct_url.json` to record the source fork and pinned commit SHA.
- Packages without git provenance remain plain version strings.

## Running Benchmarks

### Via pytest (CI / coverage)

Benchmarks use the `benchmark` RunIf mark and the `IT_RUN_BENCHMARK_TESTS` environment variable:

```bash
# All benchmarks
IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py -v

# Filter by experiment
IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py -k "rte_boolq" -v

# Filter by specific config
IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py -k "gemma2_2b_it_l" -v
```

### Via CLI runner (interactive)

```bash
# Specific benchmark
python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l

# All benchmarks for an experiment
python tests/benchmarks/run_benchmarks.py --experiment rte_boolq

# All benchmarks
python tests/benchmarks/run_benchmarks.py --all

# With debug diagnostics
python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l --debug

# Update registry after validated run
python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l --update-registry

# Force-update registry (bypasses clean working tree check)
python tests/benchmarks/run_benchmarks.py --all --force-update-registry

# Quick smoke test (limited batches)
python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l --limit-batches 5
```

### Via special_tests.sh harness

```bash
# All benchmarks
./tests/special_tests.sh --mark_type=benchmark

# Filtered
./tests/special_tests.sh --mark_type=benchmark --filter_pattern='gemma2_2b_it_l'
```

## Adding a New Experiment

1. **Create the experiment config** under `src/it_examples/config/experiments/<experiment>/`.

2. **Register in `benchmark_registry.yaml`**:
   ```yaml
   benchmarks:
     <experiment_name>:
       <benchmark_id>:
         config_path: src/it_examples/config/experiments/<experiment>/<config>.yaml
         expected_accuracy: null  # Set after first validated run
         tolerance: 0.03
         cli_mode: lightning     # Or "core" for non-Lightning configs
         adapter_ctx: [lightning]
         backends: []
         description: "Description"
         last_validated: null
         tags: [baseline]
         debug_utils_module: <experiment_name>  # Optional
   ```

3. **Validate and update** the expected accuracy:
   ```bash
   python tests/benchmarks/run_benchmarks.py --benchmark <experiment>/<id> --update-registry
   ```
   This requires a clean working tree (or `--force-update-registry`). The registry entry will be
   updated with the accuracy, commit SHA, and salient package versions.

4. **(Optional) Add experiment-specific diagnostics** under `debug_utils/<experiment_name>/`:
   ```
   debug_utils/<experiment_name>/
   ├── __init__.py
   └── dbg_<experiment_name>.py   # Must accept --config and --output args
   ```

## Debug Diagnostics

When a benchmark fails, use `--debug` to run diagnostics before the benchmark:

```bash
python tests/benchmarks/run_benchmarks.py --benchmark rte_boolq/gemma2_2b_it_l --debug -v
```

This runs:
- **Experiment-specific diagnostics** (if `debug_utils_module` is set in the registry entry)
- **Shared diagnostics** from `benchmark_utils.py` (model info, tokenizer, dataset, generation sanity checks)

Diagnostics logs are written to `/tmp/benchmark_debug/` by default (override with `--log-dir`).

### RTE/BoolQ-specific tools

```bash
# Entailment mapping and prediction diagnostics
python tests/benchmarks/debug_utils/rte_boolq/dbg_rte_boolq.py \
  --config src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_zs_test.yaml

# Batch collapse debugging
python tests/benchmarks/debug_utils/rte_boolq/debug_batch_collapse.py

# Full pipeline debug
python tests/benchmarks/debug_utils/rte_boolq/debug_it_pipeline.py
```

## Registry Commit Isolation

Registry updates should be committed in isolation so that each commit that changes `benchmark_registry.yaml` represents a complete, validated benchmark run with no unrelated code changes mixed in. This makes it possible to trace any registry entry back to the exact codebase state that produced it.

A **pre-commit hook** (`check-benchmark-registry-isolation`) enforces this: if `benchmark_registry.yaml` is staged, the only other file allowed in the same commit is `benchmark_update.allow`.

**Workflow:**
1. Commit all code changes first.
2. If benchmark tooling, docs, or `requirements/utils/collect_env_details.py` changed, commit that work before touching the registry.
3. Run benchmarks with `--update-registry` (requires clean tree) or `--force-update-registry`.
4. Commit the registry update separately so the commit only contains `benchmark_registry.yaml` plus any explicitly allowed docs or sentinel files.

For a clean full-suite refresh, prefer this sequence:

```bash
# First commit the tooling/docs changes
git commit -m "Improve benchmark registry provenance capture"

# Then regenerate the registry from a clean tree
python tests/benchmarks/run_benchmarks.py --all --update-registry

# Confirm the diff is registry-only and commit it separately
git status --short
git commit -m "Update benchmark registry"
```

The `benchmark_update.allow` sentinel file can be created to bypass the clean working tree check when `--update-registry` is used (without `--force`). It is automatically allowed in the same commit as the registry.
