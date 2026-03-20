# Interpretune Protocol Architecture Working Design

**Status:** Working document
**Audience:** contributors, early users, and future agents working on framework design

## Purpose

Interpretune is trying to solve a specific problem:

- users want to compose multiple interpretability frameworks and backends
- those frameworks expose different execution models and object types
- the same analysis pipeline should ideally work across those backends with minimal user rewrites

The current design achieves a meaningful amount of this already, but it is still pre-MVP and some abstractions remain more implicit than they should be. This document describes the current architecture accurately, highlights where it is strong today, and records the design questions that are actively shaping the next refactor steps.

## Core Architectural Layers

### 1. Session and composition layer

The root composition object is `ITSession`.

Its job is to:

- combine a datamodule and a module into one interpretune session
- enrich those components with the adapters listed in `adapter_ctx`
- make the resulting objects conform as closely as possible to the framework protocols

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/session.py`
- `/home/speediedan/repos/interpretune/src/interpretune/adapter_registry.py`
- `/home/speediedan/repos/interpretune/src/interpretune/protocol.py`

The current session model is still biased toward a paired `ITDataModule` plus `ITModule` workflow.

### 2. Protocol layer

Interpretune uses structural protocols to define the surfaces that core components should expose.

Important protocol families include:

- module and datamodule protocols
- analysis configuration and analysis store protocols
- model backend and analysis backend protocols
- analysis batch protocols

This layer lets Interpretune type and reason about composed objects without hard-coding a single concrete class hierarchy.

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/protocol.py`

### 3. Adapter layer

Adapters enrich base modules and datamodules with framework-specific behavior.

Examples:

- `transformer_lens`
- `nnsight`
- `sae_lens`
- `circuit_tracer`
- `lightning`

The adapter system is the main reason Interpretune can compose multiple model-level and analysis-level frameworks without exposing their raw integration details everywhere.

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/adapters/`

### 4. Backend layer

Interpretune distinguishes two backend concepts:

- `ModelBackend`: execution-oriented capabilities such as batched hooks and gradients
- `AnalysisBackend`: analysis-adapter functionality such as graph hydration, decomposition, tokenizer and embedding resolution, and feature intervention helpers

This split is important. It keeps execution concerns separate from richer analysis semantics.

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/analysis/backends/__init__.py`

### 5. Analysis op layer

`AnalysisOp` and `CompositeAnalysisOp` provide the main composition model for latent-space analysis.

An op:

- declares input and output schemas
- optionally declares required capabilities
- runs against a `module`, `analysis_batch`, `batch`, and `batch_idx`

Composite ops chain these operations while preserving a schema-driven execution contract.

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/analysis/ops/base.py`
- `/home/speediedan/repos/interpretune/src/interpretune/analysis/ops/dispatcher.py`
- `/home/speediedan/repos/interpretune/src/interpretune/analysis/ops/definitions.py`

### 6. Persistence layer

`AnalysisStore` is the primary durable interchange layer for analysis outputs.

It is used to:

- persist analysis results as datasets
- reload them for later ops
- bridge analysis runs across sessions or environments

This layer matters for hub workflows and for future reusable analysis artifacts.

Relevant code:

- `/home/speediedan/repos/interpretune/src/interpretune/analysis/core.py`
- `/home/speediedan/repos/interpretune/docs/analysis_store_serialization.md`

## Current Strengths

### Strong compositional story

The combination of adapters, backends, protocols, and analysis ops is already enough to support non-trivial cross-framework workflows.

### Good separation between generic ops and backend-specific analysis logic

Recent work on the `AnalysisBackend` seam has improved the design. Circuit-tracer-specific graph details live behind the analysis backend instead of leaking into every op.

### Schema-aware persistence

The schema system and `AnalysisStore` give the framework a path to durable, reusable analysis outputs rather than notebook-only transient results.

### Generated analysis-step convenience

For CLI-style workflows, `AnalysisCfg.apply()` and generated `analysis_step` methods make simple analysis configurations easy to run without writing custom step code.

## Current Weaknesses

### Session assumptions are still too narrow

The current session design still assumes that a datamodule and module pair is the normal case. That works well for training and batch evaluation, but it is awkward for:

- module-only analysis
- notebook exploration over a few hand-selected batches
- simple demo usage that does not justify a custom module or datamodule subclass

### Analysis value access is too implicit

Current ops often resolve values through helper functions rather than through a first-class execution context.

The most visible example is `get_analysis_value(...)`, which falls back from `analysis_batch` to `input_store`, and then relies on container heuristics for batch indexing.

This is the main reason notebook-static list values such as concept groups do not fit naturally today.

### CLI batch execution is more ergonomic than notebook execution

The framework's main happy path today is:

- initialize session
- attach analysis config
- let the runner drive analysis through `test_dataloader()`

Notebook workflows can still use the same pieces, but they often have to reach into lower-level internals or recreate parts of the execution pattern manually.

### Aggregate analysis is not first-class yet

Interpretune can serialize per-batch analysis outputs well enough, but aggregate workflows such as “derive a concept direction from several batches and reuse it later” still rely more on notebook conventions than on a dedicated framework path.

## Current Usage Patterns

### Good patterns

- use adapters to keep backend-specific integration logic out of user modules
- use composite ops for reusable analysis pipelines
- use `AnalysisStore` for durable artifacts rather than ad hoc notebook-only objects
- use analysis backends for graph and intervention details that are specific to a package

### Anti-patterns or fragile patterns

- treating `get_analysis_value(...)` as the long-term public abstraction for value resolution
- encoding rich structured metadata as JSON strings when Arrow-native typed columns are feasible
- designing notebook workflows around runner internals instead of a shared public analysis execution helper
- assuming all list-like analysis values are row-scoped

## Key Open Design Questions

### 1. What is the right public abstraction for analysis value lookup?

Current state:

- helper-based fallback from `analysis_batch` to `input_store`
- row indexing inferred from container type

Open question:

- should analysis value resolution become a first-class protocol or execution-context method?

Current answer in progress:

- yes, this should become explicit and scope-aware

### 2. How should interactive analysis relate to runner-driven analysis?

Open question:

- should notebook workflows use a separate execution API or the same shared execution path as generated and manual `analysis_step` calls?

Current answer in progress:

- use one shared execution model and make notebook access first-class

### 3. How should module-only sessions work?

Open question:

- can `ITSession` and the runner layer tolerate `datamodule is None` cleanly?

Current answer in progress:

- they should, especially for analysis and demo workflows

### 4. What should the default demo and example module story be?

Open question:

- do all examples need custom `ITModule` subclasses, or should Interpretune provide a basic default composition path for common analysis-only workflows?

Current answer in progress:

- provide a default module composition path

### 5. How should larger analysis artifacts be serialized?

Open question:

- when should a result be Arrow-native typed columns, and when should it be a richer hydrated object with explicit decompose and hydrate helpers?

Current answer in progress:

- keep the object serialization path, but make it more explicit and reusable across future ops

### 6. How do local registries evolve into hub-backed registries?

Related issues:

- #124 AnalysisStore upload and download
- #125 adapter upload and download
- #128 ITModule and ITDataModule bundle upload and download

These are not just distribution questions. They depend on stable semantics for sessions, artifacts, configs, and reusable analysis values.

## Near-Term Design Direction

The current preferred direction is:

- explicit scoped analysis inputs
- shared execution helper for CLI and notebooks
- module-only session support
- default demo-friendly module composition
- first-class aggregate analysis flows

That direction is described in the current IG-7 design proposal in the workstream design folder.

## How to use this document

- read this before making significant architecture changes
- update it whenever a major design question is resolved or reframed
- prefer recording active design tensions here instead of leaving them only in issues or notebook comments

This document is intentionally a working design reference, not a polished marketing overview.
