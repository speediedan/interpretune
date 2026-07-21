# Custom Ops Composition Guide

**Status:** Draft guidance
**Audience:** contributors creating custom analysis ops, custom op collections, or hub-shareable analysis workflows

## Purpose

Interpretune's analysis system is most valuable when custom ops behave like built-in ops:

- composable
- schema-aware
- backend-aware without being backend-entangled
- serializable through `AnalysisStore` where appropriate

This guide documents the current best practices for writing custom ops that compose cleanly across existing model and analysis backends.

## Core Design Rule

Write ops against the Interpretune protocol surface, not against one concrete backend unless the op is explicitly backend-specific.

That means:

- prefer generic batch and module inputs
- use backend capability validation
- route package-specific graph or intervention behavior through an analysis backend seam when one exists

## Anatomy of an op

An op should define:

- a name and description
- input schema
- output schema
- optional required capabilities
- one implementation function

Relevant code:

- `src/interpretune/analysis/ops/base.py`
- `src/interpretune/analysis/ops/dispatcher.py`
- `src/interpretune/analysis/ops/definitions.py`

## Preferred Notebook And Script Surface

For user-facing notebooks, examples, and ad hoc research scripts, prefer the top-level op wrappers on `interpretune` instead of reaching into the dispatcher directly.

Preferred pattern:

```python
import interpretune as it
import interpretune.analysis  # registers top-level op wrappers

analysis_batch = it.AnalysisBatch(prompts=[prompt])
analysis_batch = it.model_fwd_w_cache_latent_models(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=0)
analysis_batch = it.logit_diffs_cache(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=0)
```

Avoid this in notebook or experiment code unless you are extending dispatcher internals themselves:

```python
op = DISPATCHER.get_op("logit_diffs_cache")
analysis_batch = op(module, analysis_batch, batch, batch_idx)
```

Why this is preferred:

- it matches the public API surface we expect users to learn
- it keeps notebook code aligned with in-tree example usage
- it avoids local dispatcher plumbing in research harnesses that are not actually implementing new dispatch behavior

## Best Practices

### 1. Make schemas explicit

Your input and output schemas are part of the contract.

Do:

- declare every required upstream field explicitly
- prefer Arrow-native typed columns when practical
- use structured serialization patterns for richer objects that cannot be represented naturally as typed columns

Avoid:

- hiding large structured outputs in JSON strings unless there is no better short-term option

### 2. Keep implementation logic small and composable

An op should do one coherent piece of work.

Prefer:

- several small ops composed into a pipeline

Over:

- one large op that mixes caching, aggregation, intervention, logging, and formatting

### 2.1 Let `AnalysisOp` own scoped batch context by default

When an op runs through the normal dispatcher / `AnalysisOp` surface, scoped
`AnalysisBatch` lookup is already bound for that execution. That means op
implementations should prefer:

- `analysis_batch.get(...)`
- `analysis_batch.require(...)`
- shared execution helpers such as `execute_analysis_op(...)`

Avoid building new per-op context decorators as the default pattern.

(The former `with_analysis_batch_context(...)` compatibility shim has been removed — direct
`*_impl(...)` calls that intentionally bypass `AnalysisOp` no longer need a context wrapper.)

### 3. Use capability checks instead of backend-name checks

Prefer:

- required capabilities
- model backend interfaces
- analysis backend interfaces

Avoid:

- branching on adapter names or concrete class names inside generic ops

### 4. Keep backend-specific logic behind the backend seam

If an op needs package-specific graph or intervention behavior, prefer extending the analysis backend interface instead of importing a specific backend package into a generic op.

This is especially important for:

- circuit-tracer graph hydration and decomposition
- intervention spec construction
- package-specific prompt and target conversion

### 5. Design for persistence when the output has reuse value

Ask:

- should this output be reusable across sessions?
- should it be shareable through a hub workflow?
- should it be inspectable as dataset columns?

If the answer is yes, design the output schema accordingly.

## Composition Patterns

### Pattern 1: Producer op then consumer op

Example:

- produce `concept_direction`
- consume it in `compute_attribution_graph`

This is the normal composition pattern and should remain the default.

### Pattern 2: Composite op for stable workflows

If a sequence is reused often, define a composite op rather than duplicating notebook orchestration.

### Pattern 3: Aggregate workflow feeding later ops

If an analysis result is derived across multiple batches and then reused later, prefer storing it through `AnalysisStore` or a framework-level aggregate helper rather than threading it through notebook-local state only.

## Cross-Backend Guidance

### Model-level backends

Examples:

- TransformerLens
- NNsight

When your op depends on execution features such as hooks or gradients, rely on the model backend capability surface.

### Analysis-level backends

Examples:

- circuit-tracer

When your op depends on richer analysis object semantics, rely on the analysis backend surface.

## Testing Guidance

### Minimum expected tests

- schema validation
- required capability validation
- correct behavior on at least one supported backend path
- persistence or serialization behavior if the op produces reusable artifacts

### Prefer focused tests over overly broad notebook-only validation

Notebook tests are useful, but they should not be the only correctness signal.

Good test targets include:

- `tests/core/test_analysis_ops_base.py`
- `tests/core/test_analysis_ops_dispatcher.py`
- `tests/core/test_analysis_ops_definitions.py`
- `tests/core/test_cross_backend_compat.py`

### Add round-trip tests when serialization matters

If an op output is meant to survive storage and reload, add a round-trip test through `AnalysisStore`.

## Hub-Oriented Guidance

The long-term direction is for ops, stores, adapters, and configured modules to be more easily shareable.

Design custom ops so they are compatible with that future:

- keep config explicit
- avoid hidden runtime dependencies on notebook globals
- avoid implicit local-path assumptions
- prefer stable schema contracts

## Current Open Gaps

### Prefer AnalysisBatch-scoped lookup for mixed batch, run, and store inputs

The current IG-7 execution path binds scoped input resolution directly onto `AnalysisBatch`.

Prefer:

- using `analysis_batch.field_name` as the primary access pattern for declared or required inputs
- using `analysis_batch.get("field_name")` when the value is genuinely optional
- using `analysis_batch.require("field_name")` when the value is mandatory
- overriding `scopes=` only when custom precedence is genuinely needed
- treating notebook variables and aggregate artifacts as `run` scope instead of relying on list indexing heuristics

Example:

```python
group_a = list(analysis_batch.concept_group_a)
target_ids = analysis_batch.require("logit_target_ids")
custom_value = analysis_batch.get("foo", scopes=("analysis_batch", "run", "store"))
```

Attribute-style access is execution-time resolution only. It uses the currently bound scope precedence:

- `analysis_batch`
- `batch`
- `run`
- `row`
- `store`

If the active op input schema declares a default value, attribute access will also use that default before raising.

Avoid:

- adding new direct `_value_for_batch(...)` style logic inside ops
- manually constructing resolver handles in op implementations unless you are extending framework internals
- assuming that every list-like value coming from an input store is row-scoped

`get_analysis_value(...)` and `get_analysis_resolver(...)` still exist during the transition, but new op code should prefer the `AnalysisBatch` access surface.

### Serialization and formatter boundary

This lookup API is an execution-time convenience only.

It does not change how `AnalysisStore` persists data or how the custom datasets formatter materializes rows, batches, or columns. The existing serialization path still lives in:

- `src/interpretune/analysis/core.py`
- `src/interpretune/analysis/formatters.py`
- `src/interpretune/analysis/ops/auto_columns.py`

That means:

- `analysis_batch.field_name`, `analysis_batch.get(...)`, and `analysis_batch.require(...)` resolve against already-bound row, batch, run, and store objects
- `AnalysisStore` still owns dataset-backed column access, `set_format(...)`, and custom tensorization behavior
- op authors should treat scoped lookup as a read layer over already-prepared inputs, not as a new persistence mechanism

In particular, this does not change Hugging Face Dataset semantics:

- string access is still column access on `AnalysisStore`
- integer or slice access is still row or row-range access on the underlying dataset
- `_format_columns(...)` and the Interpretune dataset formatter still control how persisted columns are materialized back into tensors or lists

Keep the conceptual split clear:

- `analysis_batch` means the execution-time resolved input surface for one op call
- `batch` means the dataloader batch argument passed into the op
- `AnalysisStore` means the persisted dataset-backed artifact layer

### Aggregate analysis patterns need a cleaner framework home

Do not hard-code runner-specific assumptions into custom ops just to support aggregate workflows. Keep aggregation orchestration in helpers or workflow code until the framework-level path lands.

## Practical Rule of Thumb

If a custom op would be difficult to use from both a CLI runner and a notebook with only minor orchestration differences, its abstraction boundary is probably wrong.

Aim for ops that:

- operate on declared inputs
- expose declared outputs
- let the framework decide how those inputs are sourced and how those outputs are persisted
