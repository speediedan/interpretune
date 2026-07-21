# Circuit-Tracer Backend Support

## Overview

The circuit-tracer adapter integrates the [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) library
into Interpretune's multi-backend architecture. It supports generating attribution graphs that map causal pathways
through a model using transcoders and replacement models.

## Analysis Ops

Five dispatcher-registered analysis ops form a complete circuit-tracer attribution pipeline:

| Op Name | Description | Key Outputs |
|---------|-------------|-------------|
| `concept_direction` | Compute semantic concept direction from contrastive prompts | Concept embedding vector |
| `compute_attribution_graph` | Generate attribution graph via `ReplacementModel.attribute()` | Graph nodes, edges, attribution targets |
| `graph_node_influence` | Score feature nodes by influence on target logits | Per-node influence scores |
| `extract_top_features` | Select top-K features by influence score | Ranked feature list with metadata |
| `feature_intervention_forward` | Verify causal effect via `feature_intervention()` | Pre/post intervention logit diffs |

### Pipeline Flow

```
concept_direction → compute_attribution_graph → graph_node_influence
                                                        ↓
                              feature_intervention_forward ← extract_top_features
```

Ops compose via the standard `AnalysisRunner` / `AnalysisOpDispatcher` system. Each op's outputs are stored
in `AnalysisStore` and available as inputs for downstream ops.

`extract_top_features` now also supports constrained `(layer, feature_id)` requests that were absent from the original
graph rows. In those cases it synthesizes candidate rows using same-layer positions when available, falls back to
global positions otherwise, and carries forward activation baselines so `feature_intervention_forward` can still build
meaningful intervention tuples. Optional activation overrides can be supplied by notebook experiment configs when a
specific feature should use a fixed intervention activation rather than a baseline heuristic.

When constrained selection is active, `extract_top_features` also preserves at least one representative row per
requested `(layer, feature_id)` pair before final top-N truncation. That guarantee is what keeps a synthesized missing
feature available to `feature_intervention_forward` even when other requested features occupy more positions.

For direct tensor steering outside the feature-intervention path, concept-direction notebook experiments now also use
`model_fwd_intervention` with explicit intervention mappings. This allows the runtime concept-direction vector to be
applied at non-default hook points such as `blocks.0.hook_in` in modes like `project`.

For hook naming, prefer the canonical TransformerBridge-style patterns documented in
[intervention_hook_pattern_support.md](intervention_hook_pattern_support.md).
Interpretune now expands the supported canonical and legacy HookedTransformer spellings in both directions before
backend resolution, but the portable cross-backend subset is still intentionally smaller than the full
TransformerLens v3 hook surface.

## Backend Support

| Backend | Status | Model Types |
|---------|--------|-------------|
| NNsight (`NNSightReplacementModel`) | Supported | Gemma-2, Gemma-3 |
| TransformerLens (`TransformerLensReplacementModel`) | Supported | Gemma-2 |

Backend selection is automatic based on the module's adapter context. Configurations with
`(core, nnsight, circuit_tracer)` use NNsight; `(core, transformer_lens, circuit_tracer)` use TransformerLens.

**TransformerBridge limitation (tracked)**: the TransformerLens circuit-tracer backend requires the
legacy `HookedTransformer` path — upstream circuit-tracer's `TransformerLensReplacementModel`
subclasses `HookedTransformer` directly, so TransformerLens v3 `TransformerBridge` mode is not
supported (`use_bridge: false` in the CT TL registry entries). Tracked in
[interpretune#223](https://github.com/speediedan/interpretune/issues/223) — revisit when upstream
circuit-tracer gains bridge support; historically this limitation (plus minor NNsight efficiencies and keeping
experimental variables fixed) is why the `tests/nb_experiments` concept-direction experimentation
standardized on the NNsight backend.

## Backend Compatibility Matrix

Model-backend capability surface (interpretune#201 task 5). Both backends implement the full
seven-method `ModelBackend` surface; the differences are in *how* and in edge-case behavior:

| Capability (`ModelBackend` method) | TL (HookedTransformer) | TL (TransformerBridge) | NNsight |
| --- | --- | --- | --- |
| `fwd` (minimal forward) | ✅ | ✅ | ✅ (trace-wrapped so input-key mapping applies) |
| `fwd_w_cache` (activation caching) | ✅ | ✅ | ✅ (envoy reads via `nnsight.save`) |
| `fwd_w_cache_and_latent_models` (latent-model splice + cache) | ✅ `run_with_cache_with_saes` | ✅ `run_with_cache_with_saes` | ✅ trace-scoped splice; ⚠️ requesting the **base** activation at a spliced site double-reads the envoy input (`OutOfOrderError`) — cache the SAE stage subhooks instead |
| `fwd_w_hooks_and_latent_models` (custom hooks + splice) | ✅ | ✅ | ✅ (edit-context interventions) |
| `fwd_w_hooks_batched` (multi-config hook batching) | ⚠️ sequential fallback (one pass per config; `configs_per_pass` no-op) | ⚠️ sequential fallback | ✅ native batched invokes |
| `fwd_w_grads_and_latent_models` (gradient capture + splice) | ✅ | ✅ | ✅ |
| `fwd_w_intervention` (direction/feature interventions) | ✅ (legacy-alias gaps: e.g. `unembed.hook_in` needs the `ln_final.hook_normalized` fallback — see interpretune#223) | ✅ (same alias caveat) | ✅ |

Lifecycle contract (validated by `tests/core/test_adapters_sae_lens.py::TestLatentModelHandleLifecycle`
across all three substrates): splice paths leave **no residual state** — baseline outputs are
restored after any spliced call, repeated spliced invocations are stable (no hook/edit
accumulation), and the SAE-Lens `acts_to_saes` registry is empty before and after where exposed.
Additional constraint: TransformerBridge cannot be initialized from a config dict alone
(config-based init always uses `HookedSAETransformer` — see the adapter development guide's
Pattern 3).

## Demo Notebooks

| Notebook | Description |
|----------|-------------|
| `ct_analysis_backend_demo.ipynb` | Full 5-op pipeline on Gemma-2-2b; `backend` parameter selects NNsight (default) or TransformerLens, `dashboard_mode` selects public neuronpedia.org or local dev-webapp dashboard links — all modes exercised by parameterized notebook tests |
| `ct_concept_steering_demo.ipynb` | Concept-direction-mediated, sign-aware multi-feature steering (feature-mediated + direct-hook paths) on the proven orange example; `BACKEND` + `DASHBOARD_MODE` parameterized (public gemma-2-2b default; gemma-3-1b-it + local-262k + locally generated explanations as a papermill param set) |
| `circuit_tracer_adapter_example_basic.ipynb` | Basic adapter usage (graph generation, Neuronpedia upload) |

## Cross-Backend Composition (planned demo — not yet in-tree)

Cross-backend *mixing* is deferred to a subsequent workstream (the RTE-focused research direction
continues in [interpretune#220](https://github.com/speediedan/interpretune/issues/220)); no
committed notebook currently demonstrates it — the planned demo is tracked in
[interpretune#224](https://github.com/speediedan/interpretune/issues/224). The planned demo design, which `AnalysisStore`
persistence already supports, composes independent sessions:

1. **Stage A** — GPT-2 SAE analysis via TransformerBridge: runs `logit_diffs_base`, persists store to disk
2. **Stage B** — Gemma-2 CT analysis via NNsight: runs full 5-op pipeline, collects results
3. **Stage C** — CPU-only enrichment: reloads Store A from disk, combines summaries from both backends

Each stage would run in its own session with independent teardown, demonstrating that `AnalysisStore`
persistence (`dataset.save_to_disk()` / `AnalysisStore(dataset=path)`) enables cross-session
composition. The archived single-session RTE concept-direction flow remains recoverable from git
history (`48f371b`) if a runnable reference is needed before that demo lands.

## Related Files

| File | Purpose |
|------|---------|
| `src/interpretune/adapters/circuit_tracer.py` | Adapter implementation |
| `src/interpretune/config/circuit_tracer.py` | `CircuitTracerConfig` dataclass |
| `src/interpretune/analysis/ops/definitions.py` | Op definitions (backend-agnostic) |
| `src/interpretune/analysis/backends/circuit_tracer.py` | CT backend implementation |
| `src/it_examples/notebooks/dev/circuit_tracer_examples/` | Demo notebooks (dev) |
| `tests/examples/test_notebooks.py` | Notebook tests |
