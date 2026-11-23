# Analysis Runner Usage

**Status:** Working design document
**Audience:** Contributors working with Interpretune's analysis pipeline

## Overview

Interpretune provides two runner implementations in `src/interpretune/runners/`:

| Runner | Module | Phases | Use Case |
|--------|--------|--------|----------|
| `SessionRunner` | `core.py` | `train`, `test` | Standard training and evaluation workflows |
| `AnalysisRunner` | `analysis.py` | `train`, `test`, `analysis` | Extends `SessionRunner` with analysis workflows |

Both runners operate in the **core (non-Lightning) framework** context. The Lightning adapter delegates loop orchestration to its own `Trainer` and does not currently use these runners.

## SessionRunner

`SessionRunner` is a barebones trainer that orchestrates training and testing when no framework adapter is specified during `ITSession` composition.

### Lifecycle

1. **`__init__`** — Accepts a `SessionRunnerCfg` (or dict), validates supported commands, calls `it_init()`.
2. **`it_init()`** — Dispatches the framework-independent `it_init` function which prepares the data module, model, and optimizers.
3. **`_run(phase, loop_fn)`** — Sets the current `CorePhases` enum, calls the loop function with the runner config, then dispatches `it_session_end()`.
4. **`it_session_end()`** — Fires phase-specific session-end hooks.

### Loop Functions

- **`core_train_loop`** — Multi-epoch training with optional validation. Manages `model.train()` / `model.eval()` transitions, optimizer steps, and epoch-level hooks.
- **`core_test_loop`** — Single-epoch evaluation under `torch.inference_mode()`. Prints accumulated `log()` / `log_dict()` metrics at epoch end.

### Entry Points

```python
runner = SessionRunner(run_cfg)
runner.test()   # Dispatches core_test_loop
runner.train()  # Dispatches core_train_loop
```

## AnalysisRunner

`AnalysisRunner` extends `SessionRunner` with the `analysis` phase and provides the primary interface for running analysis workflows.

### Additional Capabilities

- **`analysis` phase** — A `partialmethod` that dispatches `core_analysis_loop`.
- **`run_analysis(analysis_cfg)`** — High-level entry point accepting one or more `AnalysisCfg` objects.
- **`_run_analysis_cfg()`** — Wraps execution in an `activated_analysis_cfg` context manager and dispatches the analysis loop.
- **Dataset generation** — Uses `Dataset.from_generator()` with `analysis_store_generator` to produce HF `Dataset` objects from model inference steps.
- **Analysis hooks** — Fires `on_analysis_start`, `on_analysis_epoch_end`, and `on_analysis_end` hooks.

### core_analysis_loop

The analysis loop creates a streaming pipeline:

1. Calls `run_step(step_fn=..., as_generator=True)` which yields per-batch outputs.
2. Wraps the generator in `analysis_store_generator` to apply output formatting.
3. Passes to `Dataset.from_generator()` with features derived from the op's output schema via `schema_to_features()`.
4. Saves the resulting dataset to the analysis output store.
5. Returns the `AnalysisStoreProtocol` object.

### Entry Points

```python
runner = AnalysisRunner(run_cfg)
runner.test()              # Inherited from SessionRunner
runner.train()             # Inherited from SessionRunner
runner.run_analysis(cfg)   # Analysis-specific
```

## AnalysisCfg

`AnalysisCfg` is the dataclass that configures an analysis run:

| Field | Purpose |
|-------|---------|
| `output_store` | Where to save analysis results |
| `input_store` | Optional input dataset for multi-step composition |
| `step_fn` | Which step function to use (e.g., `"analysis_step"`) |
| `output_schema` | Schema defining the output dataset columns |
| `names_filter` | Optional filter for specific hook point names |
| `cache_dir` | Directory for caching intermediate results |

## Comparison: Core Test Loop vs Analysis Loop

| Aspect | `core_test_loop` | `core_analysis_loop` |
|--------|-----------------|---------------------|
| Output | Metrics printed at epoch end | HF `Dataset` saved to store |
| Inference mode | `torch.inference_mode()` | `torch.inference_mode()` |
| Hook dispatch | `on_test_batch_start`, `on_test_epoch_end` | `on_analysis_start`, `on_analysis_epoch_end`, `on_analysis_end` |
| Metric logging | `log()` / `log_dict()` accumulation | Not applicable |
| Return value | None | `AnalysisStoreProtocol` |
| Generator support | No | Yes (`as_generator=True`) |

## Execution Helpers: `execute_analysis_op` and `execute_analysis_step`

Beyond the `AnalysisRunner` loop, Interpretune exposes two lower-level helpers in
`interpretune.analysis.execution` for interactive and one-shot analysis workflows:

### `execute_analysis_op`

Executes a single analysis op on a module without entering the full runner loop:

```python
from interpretune.analysis.execution import execute_analysis_op

result = execute_analysis_op(
    module,
    batch,
    batch_idx=0,
    analysis_cfg=my_cfg,          # AnalysisCfg with a resolved op
    analysis_inputs=my_inputs,    # optional AnalysisInputs or dict
)
# result is an AnalysisBatch containing op outputs
```

Internally it:

1. Resolves the `analysis_cfg` (from the argument or `module.analysis_cfg`).
2. Enters an `activated_analysis_cfg` context that temporarily sets
   `module.analysis_cfg` and calls `init_analysis_cfgs`.
3. Builds merged `AnalysisInputs` from config-backed and caller-provided values.
4. Calls `active_cfg.op(module, analysis_batch, batch, batch_idx, ...)`.

### `execute_analysis_step`

Wraps `execute_analysis_op` with serialization — it executes the op, then
yields serialized rows via `resolved_cfg.save_batch(...)`:

```python
from interpretune.analysis.execution import execute_analysis_step

for row in execute_analysis_step(module, batch, batch_idx=0, analysis_cfg=my_cfg):
    print(row)  # serialized dict suitable for Dataset.from_generator
```

This is the same function the `AnalysisRunner` uses internally.  When
`AnalysisCfg.apply(module)` finds no manual `analysis_step`, it **generates** one
that delegates to `execute_analysis_step`.

### When to use each

| Function | Returns | Use Case |
|----------|---------|----------|
| `execute_analysis_op` | `AnalysisBatch` (in-memory) | Interactive exploration, debugging, multi-step composition |
| `execute_analysis_step` | Generator of serialized dicts | Building HF Datasets, feeding `core_analysis_loop` |
| `AnalysisRunner.run_analysis` | `AnalysisStoreProtocol` | Full pipeline: loop + hooks + dataset persistence |

### Relationship to the runner

```
AnalysisRunner.run_analysis()
  └─ core_analysis_loop()
       └─ run_step(as_generator=True)
            └─ generated analysis_step
                 └─ execute_analysis_step()
                      └─ execute_analysis_op()
                           └─ analysis_cfg.op(module, batch, ...)
```

## Notebook and Interactive Usage

### Top-level op wrappers

In notebooks and interactive scripts, prefer top-level op wrappers:

```python
import interpretune as it
import interpretune.analysis  # ensure op wrappers are registered

result = it.concept_direction(...)
result = it.compute_attribution_graph(...)
```

These are `OpWrapper` proxies registered on the `interpretune` module when
`interpretune.analysis` is first imported.  They lazily instantiate the
underlying `AnalysisOp` from the `DISPATCHER` on first call.

### Direct `execute_analysis_op` usage

For workflows that require explicit `AnalysisCfg` control (e.g., setting
`analysis_inputs`, `input_store`, or composing multiple ops), call
`execute_analysis_op` directly:

```python
from interpretune.analysis import AnalysisCfg, execute_analysis_op
import interpretune as it

cfg = AnalysisCfg(
    target_op=it.concept_direction,
    # ... other config
)
result = execute_analysis_op(module, batch, analysis_cfg=cfg, analysis_inputs=inputs)
```

This is the pattern the planned circuit-tracer cross-backend composition demo will use
(tracked in [#224](https://github.com/interpretune/interpretune/issues/224); see
`circuit_tracer_backend_support.md` — no committed notebook demonstrates it yet).

## Current Limitations

- **Lightning integration is partial:** `AnalysisRunner` extends `SessionRunner` (core framework). A Lightning-composed module with an `analysis_cfg` runs its analysis step under a `Trainer`: `Trainer.predict` routes each batch through the configured step (manual, or generated from the op) and returns its per-batch outputs, and `Trainer.test` runs the same setup while its batches still go through the module's own `test_step`. With `interpretune.adapters.lightning.AnalysisStoreCallback` in the `Trainer`'s callbacks, `Trainer.predict` also writes those rows as the analysis config's `AnalysisStore`, as `AnalysisRunner.run_analysis` does: the op's output features and format, `output_store.save_dir`, `output_store.dataset`, the `on_analysis_start`, `on_analysis_epoch_end` and `on_analysis_end` hooks, and the same run-scoped op state. Rows are streamed to disk as each batch yields them (`AnalysisStoreCallback(writer_batch_size=...)` sets how many are buffered, as `AnalysisRunnerCfg.writer_batch_size` does for the core loop). `Trainer.predict` also keeps every batch's outputs to return them, so pass `return_predictions=False` on a long run. It runs in one process. A module with no analysis op, a multi-process predict and a run that yields no rows are refused, and a run that raises leaves no partial store. Distributed runs, analysis checkpointing and Lightning logging backends under a `Trainer` are tracked in [#211](https://github.com/interpretune/interpretune/issues/211).
- **Single-process only** — Analysis loops do not support distributed execution.
- **Analysis hooks under Lightning:** the analysis hooks fire under a `Trainer` only with `AnalysisStoreCallback`, and a predict run is one epoch, so `on_analysis_epoch_end` fires once.

## Related Documents

- [Framework-Level Adapters](framework_level_adapters.md)
- [Custom Ops and Composition Guide](custom_ops_composition_guide.md)
- [Session, Module, and DataModule Usage Patterns](session_module_datamodule_usage.md)
