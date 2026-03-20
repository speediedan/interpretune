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

- `/home/speediedan/repos/interpretune/src/interpretune/analysis/ops/base.py`
- `/home/speediedan/repos/interpretune/src/interpretune/analysis/ops/dispatcher.py`
- `/home/speediedan/repos/interpretune/src/interpretune/analysis/ops/definitions.py`

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

- `/home/speediedan/repos/interpretune/tests/core/test_analysis_ops_base.py`
- `/home/speediedan/repos/interpretune/tests/core/test_analysis_ops_dispatcher.py`
- `/home/speediedan/repos/interpretune/tests/core/test_analysis_ops_definitions.py`
- `/home/speediedan/repos/interpretune/tests/core/test_cross_backend_compat.py`

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

### Analysis value resolution is still evolving

If your op currently needs both `analysis_batch` values and `input_store` fallback, keep the logic localized and expect a future migration toward a first-class scoped resolver.

### Aggregate analysis patterns need a cleaner framework home

Do not hard-code runner-specific assumptions into custom ops just to support aggregate workflows. Keep aggregation orchestration in helpers or workflow code until the framework-level path lands.

## Practical Rule of Thumb

If a custom op would be difficult to use from both a CLI runner and a notebook with only minor orchestration differences, its abstraction boundary is probably wrong.

Aim for ops that:

- operate on declared inputs
- expose declared outputs
- let the framework decide how those inputs are sourced and how those outputs are persisted
