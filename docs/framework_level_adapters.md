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

- **`core_train_loop`**: Iterates epochs: sets `model.train()`, fires `on_train_epoch_start` (optional), runs training batches via `run_step` (stepping step-interval lr schedulers after each optimizer step), optionally runs validation batches under `torch.inference_mode()` (or with gradients, see below), averages and prints the epoch's logged metrics, fires `on_train_epoch_end` (optional), then steps epoch-interval lr schedulers. It returns its `CoreLoopContext`.
- **`core_test_loop`** — Sets `model.eval()`, runs test batches under `torch.inference_mode()`, prints accumulated metrics at epoch end, fires `on_test_epoch_end` (optional).
- **`core_analysis_loop`** — Generates an HF `Dataset` from `run_step(..., as_generator=True)`, saves to the analysis output store.

### Core Training Loop Controls

`SessionRunnerCfg` fields that `core_train_loop` honors, beyond the batch limits and `max_epochs`:

- **`max_steps`**: An optimizer-step budget (`-1` for none), whichever of it and `max_epochs` is reached first. The epoch in which the budget is spent still validates and runs its epoch-end hooks. Optimizer steps are counted on the loop context (`ctx.optimizer_step`), because `module.global_step` counts every step `run_step` dispatches, validation and test steps included.
- **`validate_with_grad`**: Validate with gradients enabled rather than under `torch.inference_mode()`, for a validation objective computed with `torch.autograd.grad`. Off by default.
- **`precision`**: The step functions' autocast, with Lightning's names: `"bf16-mixed"` runs them under `torch.autocast(dtype=torch.bfloat16)` around fp32 master weights (the optimizer steps the fp32 parameters), `"bf16-true"` does the same for a module whose weights are already bfloat16, and `"32"` never autocasts. The default, `None`, keeps the module dtype's meaning: autocast exactly when the module's dtype is bfloat16, so an existing run is unchanged. A precision that contradicts the module's dtype is refused by name (`"bf16-mixed"` on bfloat16 weights is `"bf16-true"`), and training, validation and test steps all take the same precision, as Lightning applies its precision to every phase. This is the loop's autocast only: under an `fsdp2` placement it is separate from FSDP2's `MixedPrecisionPolicy`.
- **`accumulate_grad_batches`**: Accumulate gradients over this many training batches per optimizer step, each batch's loss scaled by the reciprocal, so an accumulated step matches one step on the window's mean loss. The optimizer, step-interval lr schedulers, `on_train_step_end` and `max_steps` all count optimizer steps, and a partial window at an epoch's end steps before validation, keeping the reciprocal scale (as in Lightning: a short final window steps proportionally less). A tuning scheduler that counts steps through its optimizer's post-hook therefore sees only real steps. Under a data-parallel placement every batch's backward still synchronizes gradients, so accumulation there saves optimizer steps, not communication.
- **`callbacks`**: Objects with any of `CoreCallback`'s methods (`interpretune.runners.core`): `on_train_start(ctx)`, `on_train_epoch_start(ctx)`, `on_train_step_end(ctx, outputs)`, `on_validation_end(ctx, metrics)`, `on_train_epoch_end(ctx, metrics)` and `on_train_end(ctx)`, plus `state_dict()` / `load_state_dict(state)` for checkpoints. A callback ends training by setting `ctx.should_stop`, which is how early stopping is written:

```python
class EarlyStopOnValLoss:
    def __init__(self, patience: int = 2):
        self.best, self.bad, self.patience = float("inf"), 0, patience

    def on_validation_end(self, ctx, metrics):
        if metrics["val_loss"] < self.best:
            self.best, self.bad = metrics["val_loss"], 0
        else:
            self.bad += 1
        ctx.should_stop = self.bad >= self.patience
```

**LR schedulers** returned by `configure_optimizers` (or built from `lr_scheduler_init`) step at their configured `interval` (`"step"` or `"epoch"`) and `frequency`. A `ReduceLROnPlateau` scheduler steps with its `monitor` metric from the epoch's logged metrics, and is refused when it names no monitor or the metric was not logged (unless its config sets `strict=False`, which skips the step with a warning).

**The optimizer is read from the module at every step** (`ctx.optimizer`), never captured once, so a callback may replace it, as a phased fine-tuning schedule that reinitializes the optimizer does.

**Checkpoints.** `checkpoint_state(module, ctx)` returns the module's `state_dict`, every optimizer's and lr scheduler's state, the Python, torch, CUDA and (when installed) numpy RNG states, the loop counters and each callback's `state_dict`. `load_checkpoint_state(module, checkpoint, ctx)` restores it. The module's optional `on_save_checkpoint(checkpoint)` and `on_load_checkpoint(checkpoint)` hooks may extend it. The state references live tensors, as `state_dict()` does, so save it with `torch.save`; inside a callback, `ctx.checkpoint_state()` is the shortcut. Callbacks' state loads first, so a callback may rebuild the optimizers the checkpoint's optimizer state then loads into. To resume a run, pass a checkpoint taken in a callback's `on_train_epoch_end` (or the path it was saved to) as `resume_from` (`SessionRunnerCfg.resume_from`): the loop loads it once every callback's `on_train_start` has run, completes that epoch's epoch-interval lr scheduler step, and continues at the next epoch.

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

At each training epoch end (after validation) `core_train_loop`, and at test epoch end `core_test_loop`, averages each metric list, prints the results, and clears the accumulator. The training loop also passes the averages to its callbacks:

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
| Callbacks | A minimal `CoreCallback` list (step, validation-end and epoch boundaries) | Early stopping, checkpointing, LR monitor |
| Distributed training | Not supported | DDP, FSDP, DeepSpeed |
| Mixed precision | `precision` (`bf16-mixed`, `bf16-true`, `32`) via `torch.autocast` | Precision plugins |
| Checkpointing | `checkpoint_state` / `load_checkpoint_state` (module, optimizers, lr schedulers, RNG), saved by a callback | Automatic model checkpointing |
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
- **Core callbacks are minimal**: The core loop exposes step, validation-end and epoch boundaries and checkpoint state, but ships no ready-made early-stopping or checkpointing callbacks; Lightning provides those.

## Related Documents

- [Session, Module, and DataModule Usage Patterns](session_module_datamodule_usage.md)
- [Analysis Runner Usage](analysis_runner_usage.md)
- [Protocol Architecture](protocol_architecture_working_design.md)
- [Developing Adapters](../.claude/skills/developing-adapters/SKILL.md)
