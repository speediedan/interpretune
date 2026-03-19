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

## Backend Support

| Backend | Status | Model Types |
|---------|--------|-------------|
| NNsight (`NNSightReplacementModel`) | Supported | Gemma-2, Gemma-3 |
| TransformerLens (`TransformerLensReplacementModel`) | Supported | Gemma-2 |

Backend selection is automatic based on the module's adapter context. Configurations with
`(core, nnsight, circuit_tracer)` use NNsight; `(core, transformer_lens, circuit_tracer)` use TransformerLens.

## Demo Notebooks

| Notebook | Description |
|----------|-------------|
| `ct_analysis_backend_demo.ipynb` | Full 5-op pipeline on Gemma-2-2b via NNsight backend |
| `ct_cross_backend_demo.ipynb` | Multi-model composition: GPT-2 SAE (TransformerBridge) + Gemma-2 CT (NNsight) with cross-backend AnalysisStore enrichment |
| `circuit_tracer_adapter_example_basic.ipynb` | Basic adapter usage (graph generation, Neuronpedia upload) |

## Cross-Backend Composition

The cross-backend demo illustrates how `AnalysisStore` enables composition across independent sessions:

1. **Stage A** — GPT-2 SAE analysis via TransformerBridge: runs `logit_diffs_base`, persists store to disk
2. **Stage B** — Gemma-2 CT analysis via NNsight: runs full 5-op pipeline, collects results
3. **Stage C** — CPU-only enrichment: reloads Store A from disk, combines summaries from both backends

Each stage runs in its own session with independent teardown, demonstrating that `AnalysisStore`
persistence (`dataset.save_to_disk()` / `AnalysisStore(dataset=path)`) enables cross-session composition.

## Related Files

| File | Purpose |
|------|---------|
| `src/interpretune/adapters/circuit_tracer.py` | Adapter implementation |
| `src/interpretune/config/circuit_tracer.py` | `CircuitTracerConfig` dataclass |
| `src/interpretune/analysis/ops/definitions.py` | Op definitions (backend-agnostic) |
| `src/interpretune/analysis/backends/circuit_tracer.py` | CT backend implementation |
| `src/it_examples/notebooks/dev/circuit_tracer_examples/` | Demo notebooks (dev) |
| `tests/examples/test_notebooks.py` | Notebook tests |
