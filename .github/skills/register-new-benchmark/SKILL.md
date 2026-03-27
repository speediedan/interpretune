---
name: register-new-benchmark
description: Registers a new experiment benchmark in the Interpretune benchmark suite. Covers experiment config YAML creation, registry entry, validation run, and optional debug utilities. Generalizable across experiments and adapter compositions.
license: Apache-2.0
metadata:
  author: speediedan
  version: '1.0'
compatibility: Requires bash, Python 3.10+, uv, GPU (CUDA), and access to the Interpretune repository with an activated development venv.
---

# Register New Benchmark Skill

This skill guides the process of adding a new experiment benchmark to the Interpretune benchmark suite. It covers creating the experiment config YAML, registering it in the benchmark registry, running a validation pass, and optionally adding experiment-specific debug utilities.

## When to Use This Skill

Use this skill when:

- Adding a new model/adapter combination to the benchmark suite
- Creating a new experiment config that should be tracked for reproducibility
- Extending an existing experiment (e.g., `rte_boolq`) with a new adapter stack or model variant
- Setting up a new experiment family (e.g., a new task beyond RTE/BoolQ)

## Reference Files

This skill builds on the benchmark infrastructure documented in:

- `tests/benchmarks/README.md` — Benchmark suite overview, registry format, running/debugging benchmarks
- `tests/benchmarks/benchmark_registry.yaml` — Registry of all tracked benchmarks
- `tests/benchmarks/run_benchmarks.py` — CLI runner for executing and validating benchmarks
- `tests/benchmarks/benchmark_utils.py` — Shared diagnostic utilities
- `.github/copilot-instructions.md` — Build and environment setup instructions

## Required User Inputs

Before running this skill, gather the following:

1. **Experiment name** — The experiment family (e.g., `rte_boolq`). Use an existing family if the task/dataset matches, or create a new one.
2. **Benchmark ID** — A unique identifier within the experiment (e.g., `gemma3_1b_it_l_ct_ns`). Convention: `{model}_{size}_{variant}_{adapters}`.
3. **Model** — HuggingFace model ID (e.g., `google/gemma-3-1b-it`)
4. **Adapter context** — List of adapters composing the ITSession (e.g., `[lightning, sae_lens]`, `[core, nnsight, circuit_tracer]`)
5. **Backends** — Model/analysis backends used (e.g., `[transformer_lens]`, `[nnsight]`, `[]`)
6. **Reference config** — An existing benchmark config YAML to use as a template. Choose the closest match by adapter stack.
7. **Task-specific details** — For new experiments: dataset task name, prompt config class, entailment mapping, tokenizer settings, etc.
8. **[Optional] Expected accuracy** — If known from a prior run; otherwise set to `null` and validate.

## Prerequisites

- Interpretune repository checked out locally
- Development venv built and activated (see `copilot-instructions.md`)
- Required model weights accessible (HuggingFace auth token set if gated)
- GPU available for benchmark validation
- Existing experiment module available (e.g., `it_examples.experiments.rte_boolq`) or plan to create one

## Step-by-Step Process

### Phase 1: Create Experiment Config YAML

**Goal:** Create a properly structured YAML config for the new benchmark.

1. **Identify the reference config** — Find the closest existing config to use as a template:

   ```bash
   ls src/it_examples/config/experiments/<experiment>/<model_family>/
   ```

   For adapter-specific fields, check configs with similar adapter stacks.

2. **Create the config directory** (if needed for a new model family):

   ```bash
   mkdir -p src/it_examples/config/experiments/<experiment>/<model_family>/
   ```

3. **Create the YAML config** by adapting the reference config. Key sections to update:

   **Session config:**
   ```yaml
   session_cfg:
     adapter_ctx: [<adapter_list>]
     datamodule_cls: <experiment_datamodule_class>
     module_cls: <experiment_module_class>
   ```

   **Datamodule config** — Update model path, tokenizer settings, prompt config:
   ```yaml
   datamodule_cfg:
     class_path: interpretune.config.datamodule.ITDataModuleConfig
     init_args:
       model_name_or_path: <hf_model_id>
       task_name: <dataset_task>
       cust_tokenization_pattern: <pattern>  # e.g., gemma-chat
       prompt_cfg:
         class_path: <prompt_config_class>
       tokenizer_kwargs:
         padding_side: left
         # ... model-specific tokenizer settings
   ```

   **Module config** — Update based on adapter stack:

   - **TransformerBridge (SAE-Lens + Bridge):** Use `ITLensBridgeConfig` as `tl_cfg`
   - **HookedTransformer (SAE-Lens + HT):** Use `ITLensFromPretrainedConfig` or `ITLensFromPretrainedNoProcessingConfig` with `use_bridge: false`
   - **NNsight:** Use `ITNNsightConfig` with `nnsight_cfg`
   - **Circuit Tracer:** Add `circuit_tracer_cfg` with backend, transcoder_set, analysis targets

   **Generation config** — Ensure `output_logits: true` for classification tasks:
   ```yaml
   generative_step_cfg:
     class_path: <generative_classification_config>
     init_args:
       enabled: true
       lm_generation_cfg:
         class_path: <generation_config_class>
         init_args:
           max_new_tokens: <N>
           output_logits: true  # Critical for accuracy parsing
   ```

   **Trainer config** — Include logger with `save_dir`:
   ```yaml
   trainer:
     precision: bf16-true
     devices: 1
     logger:
       class_path: lightning.pytorch.loggers.TensorBoardLogger
       init_args:
         name: <experiment_tag>
         save_dir: lightning_logs
   ```

4. **Validate the config parses correctly** (quick check without running):

   ```bash
   interpretune test --config <config_path> --print_config > /dev/null
   ```

### Phase 2: Register in Benchmark Registry

**Goal:** Add the new benchmark entry to `benchmark_registry.yaml`.

1. **Add the registry entry** under the appropriate experiment:

   ```yaml
   benchmarks:
     <experiment_name>:
       <benchmark_id>:
         config_path: src/it_examples/config/experiments/<experiment>/<model_family>/<config>.yaml
         expected_accuracy: null  # Will be set after validation
         tolerance: 0.03
         adapter_ctx: [<adapters>]
         backends: [<backends>]
         description: "<Human-readable description>"
         last_validated: null
         notes: >
           Initial registration. Pending first validation run.
         tags: [<relevant_tags>]
         debug_utils_module: <experiment_name>  # If debug utils exist for this experiment
   ```

2. **Choose appropriate tags** from existing conventions:

   | Tag | Meaning |
   |-----|---------|
   | `baseline` | No adapter beyond Lightning/core |
   | `sae` | Uses SAE (SAE-Lens) |
   | `bridge` | TransformerBridge mode |
   | `hooked_transformer` | HookedTransformer mode |
   | `nnsight` | NNsight backend |
   | `circuit_tracer` | Circuit Tracer integration |
   | `lightning` | Lightning adapter |
   | `hf_generation` | Uses HuggingFace generation |
   | `sae_lens` | SAE-Lens adapter |

### Phase 3: Validate the Benchmark

**Goal:** Run the benchmark and capture the baseline accuracy.

1. **Run with `--update-registry`** to automatically record the accuracy:

   ```bash
   cd ~/repos/interpretune && source /mnt/cache/$USER/.venvs/it_latest/bin/activate
   python tests/benchmarks/run_benchmarks.py --benchmark <experiment>/<benchmark_id> --update-registry
   ```

   This records the observed accuracy, the current commit SHA, and `salient_pkg_versions` metadata.
   The provenance field should retain best-effort git details for salient dependencies: editable installs
   should contribute live checkout metadata (`fork`, `branch`, `sha`), while git-backed non-editable installs
   should derive the source fork and pinned commit from `direct_url.json`.

2. **If the run fails**, debug systematically:

   - **Config parse errors**: Check YAML syntax, class paths, required fields (e.g., `save_dir` for TensorBoardLogger)
   - **Model loading errors**: Verify HF auth token, model ID, device settings
   - **Generation errors**: Ensure `output_logits: true`, check generation config compatibility
   - **Accuracy parse errors**: Check `benchmark_utils.parse_accuracy()` patterns match the output format

   Use `--debug` for detailed diagnostics:
   ```bash
   python tests/benchmarks/run_benchmarks.py --benchmark <experiment>/<benchmark_id> --debug
   ```

3. **Verify the accuracy is reasonable**:

   - Compare against the baseline (no-adapter) benchmark for the same model
   - SAE reconstruction typically drops accuracy 3-8pp from baseline
   - If accuracy is suspiciously low, check entailment mapping, generation config, padding

4. **Run a second time** to verify stability (especially for `do_sample=true` configs):

   ```bash
   python tests/benchmarks/run_benchmarks.py --benchmark <experiment>/<benchmark_id>
   ```

5. **Keep the registry update clean**:

   - Commit benchmark tooling, environment-collector, and documentation changes before running `--update-registry`.
   - Re-run the benchmark from a clean working tree.
   - Commit the resulting `benchmark_registry.yaml` diff separately so the registry commit stays traceable to one validated run.

6. **Update notes** in the registry with validation context:

   ```yaml
   notes: >
     <Backend> mode. <Key config details>. Observed <accuracy>% on validation.
     <Tolerance rationale if non-default>.
   ```

### Phase 4: Add Regression Test Entry (Automatic)

The pytest benchmark tests in `tests/benchmarks/test_benchmarks.py` are auto-parametrized from the registry. Once the registry entry exists with a non-null `expected_accuracy`, the benchmark is automatically included in:

```bash
IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py -v
```

No additional test code is needed unless the experiment requires custom assertion logic.

### Phase 5: (Optional) Add Debug Utilities

**Goal:** Add experiment-specific diagnostics for debugging failures.

1. **Create the debug utils module** (if one doesn't exist for this experiment):

   ```
   tests/benchmarks/debug_utils/<experiment_name>/
   ├── __init__.py
   └── dbg_<experiment_name>.py
   ```

2. **The debug script must accept `--config` and `--output` arguments** and write JSON diagnostics:

   ```python
   """Experiment-specific diagnostics for <experiment_name> benchmarks."""
   import argparse
   import json

   def run_diagnostics(config_path: str, output_path: str) -> None:
       results = {}
       # ... experiment-specific checks ...
       with open(output_path, 'w') as f:
           json.dump(results, f, indent=2)

   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--config", required=True)
       parser.add_argument("--output", required=True)
       args = parser.parse_args()
       run_diagnostics(args.config, args.output)
   ```

3. **Set `debug_utils_module`** in the registry entry to enable automatic diagnostics with `--debug`.

## Verification Checklist

- [ ] Config YAML parses without errors (`--print_config`)
- [ ] Benchmark runs successfully and produces accuracy output
- [ ] Registry entry is complete with non-null `expected_accuracy` and `last_validated`
- [ ] Notes describe the setup and validation context
- [ ] Accuracy is reasonable compared to baseline
- [ ] Second run confirms stability within tolerance
- [ ] `IT_RUN_BENCHMARK_TESTS=1 python -m pytest tests/benchmarks/test_benchmarks.py -k "<benchmark_id>" -v` passes

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `save_dir is required` | Missing `save_dir` in TensorBoardLogger | Add `save_dir: lightning_logs` |
| `Could not parse accuracy` | Output format not matching regex | Check `benchmark_utils.parse_accuracy()` patterns |
| `output_logits not found` | Missing `output_logits: true` in generation config | Add to `lm_generation_cfg` |
| CUDA OOM | Model too large for single GPU | Adjust `device_map`, reduce batch size, or use offloading |
| `use_bridge=True` warning | `ITLensFromPretrainedConfig` used with `use_bridge=True` | Switch to `ITLensBridgeConfig` for Bridge mode |
| Very low accuracy | Wrong entailment mapping or prompt config | Verify entailment labels match model's Yes/No tokens |
