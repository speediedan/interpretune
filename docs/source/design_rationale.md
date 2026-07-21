# Design Rationale Notes

Short rationale notes answering "why does interpretune do it this way?" — kept together so the
reasoning survives refactors.

## Why an interoperability protocol (and why now)

In a world where **code is no longer a scarce resource**, effective abstract interface design is
of heightened importance — both because interface friction is the new rate-limiting constraint on
progress, and because we increasingly operate at higher levels of abstraction. The corollary for
interpretability tooling: ensuring compatibility between any two *particular* frameworks has
diminished in importance, but **defining an effective protocol for interpretability-framework
interoperability has become imperative**. That is the bet Interpretune makes: a protocol layer
over which frameworks and methods compose, with Interpretune helping validate more robust
causal/mechanistic guarantees while working at those higher levels of abstraction with AI world
models.

A considered follow-on ([#6](https://github.com/speediedan/interpretune/issues/6)): extracting the
core protocol out of Interpretune into a standalone, neutrally-named distribution once the MVP
stabilizes the protocol surface — deliberately **not** an MVP blocker (renaming/extraction churn
buys users nothing pre-alpha), but new public surfaces are designed with that extraction in mind.

## Why HF-head-first model loading (rather than TL `from_pretrained`)

Interpretune prefers loading the HuggingFace model **with its task head** first and then handing
it to TransformerLens, rather than using TL's own `from_pretrained` path. Rationale: the HF path
preserves the exact task-head weights and configuration used in fine-tuning/evaluation flows
(interpretune sessions frequently mix tuning and analysis), keeps a single authoritative
weight-loading path across adapter compositions (the same HF model feeds NNsight and
circuit-tracer substrates), and avoids relying on TL's re-implementation of head loading for
architectures where it lags upstream. Direct TL head-addition support has correspondingly not
been prioritized.

## Chat-templated prompt construction is opt-in

Prompt configs construct chat prompts manually by default (deterministic, versioned prompt
construction independent of tokenizer-bundled templates). An **opt-in** `apply_chat_template_fn`
path delegates to the tokenizer's `apply_chat_template()` where fidelity to the model's own
template is preferred (e.g. the Gemma prompt configs and the dashboard RTE pretokenization
utility use it). Manual construction remains the default so prompt provenance stays explicit.

## Run the PyTorch profiler and `memprofiler` independently

The PyTorch profiler and interpretune's `memprofiler` extension should be run **independently,
not simultaneously**: both hook allocation/timing surfaces, and concurrent operation produces
unreliable measurements in both. Profile time and memory in separate runs.

## Why `rss_diff`-based CPU-memory assertions are disabled

Test assertions on `rss_diff` (per-phase RSS deltas) are disabled: RSS deltas are dominated by
allocator/OS effects (arena reuse, lazy reclaim) and are not stable assertion targets across
runners. The retained alternatives: absolute RSS totals for coarse regression detection,
`cumul_out_bytes` (cumulative output-tensor bytes) for deterministic growth accounting, and
`saved_tensors_hooks`-based accounting for autograd-retained memory. `rss_diff` collection itself
remains available for interactive diagnosis.

## `required_ops` schema semantics (op compilation contract)

Two contract semantics govern composite-op compilation (exercised by
`test_compile_with_resolved_required_ops`):

1. A parent (composite) op's input schema must carry the required inputs of its children — child
   required-input needs propagate upward at compilation, so a composition is executable iff its
   root schema is satisfied.
2. Child output-schema keys that the parent does not override are prunable from the parent's
   output schema — the compiled parent exposes only its declared outputs, not the union of all
   child outputs.

A relaxation option (allowing parents to implicitly re-expose child outputs) has been considered
but is deliberately not implemented — explicit schemas keep compiled plans auditable.

## `.pyi` stubs are generated for native ops only

`scripts/generate_op_stubs.py` generates type stubs (`src/interpretune/__init__.pyi`) for
**native ops only** — user-defined/composed ops are dispatched dynamically and do not get stubs
by default. (A future option may extend generation to all registered ops.) The stale-stubs CI
check compares committed stubs against regeneration output, so run the script after changing
native op definitions.
