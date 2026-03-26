# Framework-Level Adapters

**Status:** Working design document
**Audience:** Contributors building or extending Interpretune workflow orchestration

## Overview

Interpretune supports two framework-level execution contexts:

1. **Core (default)** — A lightweight, framework-agnostic runner (`SessionRunner` / `AnalysisRunner`) that requires no external training framework.
2. **Lightning** — A full-featured adapter that composes Interpretune modules with PyTorch Lightning's `Trainer`, providing callbacks, distributed training, logging backends, and checkpointing.

Both contexts share the same `BaseITModule` protocol (hooks, mixins, `_call_itmodule_hook` dispatch), but differ in how loops are orchestrated, how metrics are reported, and what infrastructure is available.

## Architecture

### Module Composition (MRO)

All Interpretune modules inherit from `BaseITModule`:

```
BaseITModule(BaseITMixins, BaseITComponents, BaseITHooks, torch.nn.Module)
```

- **`BaseITMixins`** — Reusable behavior mixins (`ClassificationMixin`, `GenerativeStepMixin`, `HFFromPretrainedMixin`, `AnalysisStepMixin`).
- **`BaseITComponents`** — Core component attributes (`CoreHelperAttributes` for core modules; Lightning modules inherit from `LightningModule` instead).
- **`BaseITHooks`** — Lifecycle hooks (`setup`, `on_session_end`, `on_train_end`, etc.).

When the Lightning adapter is composed, the MRO becomes:

```
(LightningAdapter, BaseITModule, LightningModule)
```

Lightning's `log()` / `log_dict()` override the core stubs, and the `Trainer` replaces the core loop functions.

### Property Dispatch

`PropertyDispatcher._core_or_framework()` enables the same property code to resolve attributes from either:
- **Core**: `_it_state.<key>` (e.g., `_it_state._log_dir`, `_it_state._current_epoch`)
- **Lightning**: Trainer paths (e.g., `trainer.model._trainer.log_dir`, `trainer.current_epoch`)

This is configured via `_it_cls_metadata` on each adapter.

## Core Framework

### Runners

| Runner | Phases | Entry Points |
|--------|--------|-------------|
| `SessionRunner` | `train`, `test` | `runner.train()`, `runner.test()` |
| `AnalysisRunner` | `train`, `test`, `analysis` | `runner.run_analysis(cfg)` |

### Loop Functions

- **`core_train_loop`** — Iterates epochs: sets `model.train()`, fires `on_train_epoch_start`, runs training batches via `run_step`, optionally runs validation batches under `torch.inference_mode()`, fires `on_train_epoch_end`.
- **`core_test_loop`** — Sets `model.eval()`, runs test batches under `torch.inference_mode()`, prints accumulated metrics at epoch end, fires `on_test_epoch_end` (optional).
- **`core_analysis_loop`** — Generates an HF `Dataset` from `run_step(..., as_generator=True)`, saves to the analysis output store.

### Hook Dispatch

`run_step()` dispatches phase-specific hooks on the first step (`global_step == 0`):

| Step Function | Hook Fired | Optional? |
|--------------|------------|-----------|
| `test_step` | `on_test_batch_start` | Yes |
| `training_step` | `on_train_batch_start` | Yes |

All hooks are dispatched via `_call_itmodule_hook(module, hook_name=..., optional=True)`. Modules that don't implement a given hook simply skip it.

### Logging and Metrics

`CoreHelperAttributes` provides real `log()` and `log_dict()` methods:

- **`log(name, value)`** — Detaches tensors to CPU scalars and accumulates values in `_logged_metrics: dict[str, list[float]]`.
- **`log_dict(metric_dict)`** — Delegates to `log()` for each key.
- Extra kwargs (`prog_bar`, `sync_dist`, etc.) are accepted but ignored in core context.

At test epoch end, `core_test_loop` averages each metric list, prints the results, and clears the accumulator:

```
Test epoch end: {'accuracy': 0.8500, 'loss': 0.3500}
```

For any `compatibility_attrs` key that is **not** `log` or `log_dict`, a `_dummy_notify` stub is registered via `__getattr__`. This issues a one-time warning and returns a configured default, allowing framework-specific methods (e.g., `save_hyperparameters`) to degrade gracefully in core context.

### ClassificationMixin Integration

`ClassificationMixin.setup()` cooperatively calls `super().setup()` (reaching `BaseITHooks.setup()`), then initializes `classification_mapping` if configured. The mixin's `collect_answers()` method computes metrics and calls `self.log_dict(metric_dict)` — which routes to `CoreHelperAttributes.log_dict` in core context or `LightningModule.log_dict` in Lightning context.

## Lightning Framework

### What Lightning Provides Beyond Core

| Capability | Core | Lightning |
|-----------|------|-----------|
| Loop orchestration | Manual `core_*_loop` | `Trainer.fit()` / `Trainer.test()` |
| Logging | Accumulate-and-print | TensorBoard, W&B, CSV, etc. |
| Callbacks | None | Early stopping, checkpointing, LR monitor |
| Distributed training | Not supported | DDP, FSDP, DeepSpeed |
| Mixed precision | Manual `torch.autocast` | Precision plugins |
| Checkpointing | Not supported | Automatic model checkpointing |
| Progress bars | Not supported | Built-in progress tracking |
| Profiling | Not supported | Lightning profilers |

### Lightning Adapter Registration

The `LightningAdapter` registers in the `CompositionRegistry` under `Adapter.lightning` for both `module` and `datamodule` component keys. It maps core state attributes to Lightning trainer paths via `_it_cls_metadata`.

### Logging in Lightning Context

`LightningModule.log()` / `log_dict()` route to configured loggers with full support for `prog_bar`, `sync_dist`, `on_step` / `on_epoch`, and logger selection. `ClassificationMixin.collect_answers()` calls `self.log_dict(metric_dict, prog_bar=True, sync_dist=True)` — these kwargs are honored by Lightning and silently accepted by core.

## Design Principles

1. **Framework-agnostic module definitions** — Module definitions (e.g., `RTEBoolqSteps`) should NOT contain framework-specific hooks or accumulation logic. Use `ClassificationMixin` for prediction accumulation and metric reporting.

2. **Hook dispatch via `_call_itmodule_hook(..., optional=True)`** — Handles missing hooks gracefully. Modules do not need no-op stubs for hooks they don't implement.

3. **Cooperative `super()` chains** — Mixins like `ClassificationMixin.setup()` always call `super().setup()` to ensure the full MRO is traversed. `BaseITHooks.setup()` is the terminal hook handler.

4. **Single `log` / `log_dict` API** — User code calls `self.log()` or `self.log_dict()` regardless of context. Core accumulates and prints; Lightning routes to loggers.

## Analysis Execution Helpers

The analysis pipeline provides execution helpers that work independently of the
framework adapter:

| Helper | Context | Purpose |
|--------|---------|---------|
| `execute_analysis_op` | Core only | Execute a single analysis op on a module, returns `AnalysisBatch` |
| `execute_analysis_step` | Core only | Execute + serialize an analysis op, yields rows for `Dataset.from_generator` |
| `AnalysisRunner.run_analysis` | Core only | Full loop with hooks, persistence, and dataset generation |
| Top-level op wrappers (`it.concept_direction(...)`) | Core only | Lazy `OpWrapper` proxies registered on the `interpretune` module |

All four entry points ultimately call the configured `AnalysisCfg.op(...)`.
Running analysis workflows through a Lightning `Trainer` is planned future work.

See [Analysis Runner Usage](analysis_runner_usage.md) for detailed API documentation
and notebook usage patterns.

## Current Limitations

- **Analysis runner is core-only** — `AnalysisRunner` extends `SessionRunner` and is not yet integrated with the Lightning adapter. Running analysis workflows through a Lightning `Trainer` is planned future work.
- **No distributed support in core** — Multi-GPU/multi-node training requires the Lightning adapter.
- **No callback system in core** — Early stopping, checkpointing, etc. require Lightning.

## Related Documents

- [Session, Module, and DataModule Usage Patterns](session_module_datamodule_usage.md)
- [Analysis Runner Usage](analysis_runner_usage.md)
- [Protocol Architecture](protocol_architecture_working_design.md)
- [Developing Adapters](../.github/instructions/developing_adapters.instructions.md)
