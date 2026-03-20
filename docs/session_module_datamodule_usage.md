# Session, Module, and DataModule Usage Patterns

**Status:** Draft guidance
**Audience:** users and contributors building workflows on top of Interpretune

## Purpose

Interpretune currently supports two broad workflow styles:

- CLI or runner-driven batch workflows
- notebook-driven interactive workflows

Both should use the same underlying protocol and composition patterns whenever possible. This guide documents the current recommended patterns and the areas that are still being improved.

## Core Roles

### `ITDataModule`

Use a datamodule when you need:

- tokenizer setup
- dataset loading and split management
- collator construction
- consistent dataloader behavior for training, evaluation, or batch analysis

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/base/datamodules.py`

### `BaseITModule` and `ITModule`

Use a module when you need:

- model initialization
- framework adapter composition
- training, test, or predict hooks
- analysis-step integration and analysis config ownership

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/base/modules.py`
- `/home/speediedan/repos/interpretune/src/interpretune/protocol.py`

### `ITSession`

Use a session when you want Interpretune to compose your datamodule and module using the adapters in `adapter_ctx`.

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/session.py`

## Recommended Patterns Today

### Pattern 1: Full session for CLI or batch analysis

Use this when your workflow is naturally dataloader-driven.

Recommended when:

- your analysis runs over a dataset split
- you want `AnalysisRunner` to drive the loop
- you want outputs persisted through `AnalysisStore`

General shape:

```python
session = ITSession(session_cfg)
runner = AnalysisRunner({"it_session": session, ...})
store = runner.run_analysis(...)
```

### Pattern 2: Full session plus direct dataloader access in notebooks

Use this when you still want datamodule-defined batches, but you want notebook control over which batches or examples to inspect.

Recommended when:

- you want to reuse tokenizer and dataloader setup
- you only want a small subset of batches
- you need interactive inspection between analysis steps

General shape:

```python
session = ITSession(session_cfg)
test_dl = session.datamodule.test_dataloader()
for batch_idx, batch in enumerate(test_dl):
    ...
```

This is currently the most practical notebook pattern, but it is also where the framework still shows some friction. The ongoing IG-7 refactor is intended to make this path more first-class.

### Pattern 3: Manual `analysis_step` for module-specific logic

Use a manual `analysis_step` when:

- your module needs custom pre-processing before ops run
- you want to mix custom module logic with built-in ops
- you need tighter control than the generated analysis-step path provides

See the current example in:

- `/home/speediedan/repos/interpretune/src/it_examples/experiments/rte_boolq.py`

## Best Practices

### Keep task-specific logic in the module, backend-specific logic in adapters or backends

Good split:

- module owns task semantics and analysis orchestration
- adapter owns framework integration
- backend owns execution or analysis-package specifics

### Reuse the datamodule when batch semantics matter

Even in notebooks, prefer reusing the datamodule's tokenizer, collator, and dataloaders rather than recreating them ad hoc.

### Prefer `AnalysisStore` for durable intermediate artifacts

If a notebook-derived artifact should be reused later, store it in a framework-compatible way instead of leaving it as an ephemeral Python object.

### Use generated analysis steps only when the default execution model fits

Generated analysis steps are best when:

- the op pipeline can run directly over each batch
- no custom orchestration is required
- values are naturally batch-derived

If you need aggregate or run-scoped values, write the orchestration explicitly for now.

## Current Limitations

### Module-only sessions are not yet ergonomic enough

This is a known design gap and an active refactor target.

### Notebook-static analysis values are not first-class yet

Values such as concept groups or aggregate concept directions should not have to pretend to be row-scoped inputs. The current framework still has friction here.

### Default demo-oriented module composition is still thin

Many examples still rely on task-specific module classes even when a lighter default composed module would be sufficient.

## Guidance for new work

### If you are building a reusable dataset-backed workflow

- create a real `ITDataModule`
- create a focused module or use an existing one
- keep your analysis pipeline in ops or composite ops
- persist outputs in `AnalysisStore`

### If you are building a notebook demo or exploratory analysis

- still prefer using a real session so adapters, tokenizer, and backend plumbing are correct
- reuse the datamodule's dataloader when possible
- keep aggregate notebook logic isolated so it can later move into a helper or framework path

### If you need behavior the generated analysis-step path cannot express cleanly

- write a manual `analysis_step`
- keep it small
- delegate as much as possible to built-in ops rather than reimplementing backend behavior

## Near-Term Expected Improvements

The active design direction is to add:

- module-only `ITSession` support
- a shared analysis execution helper usable from both runners and notebooks
- explicit scoped analysis inputs
- a default module composition path for demos and examples

When those land, this guide should be updated to move the current notebook caveats into a cleaner recommended workflow.
