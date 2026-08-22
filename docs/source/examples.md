# About These Examples

Runnable example notebooks. Every page below is rendered from the same notebook you can clone and
run yourself — they live in the repository at `src/it_examples/notebooks/publish/`.

```{note}
The notebooks shipped in the repository are stored **without outputs**, so a fresh clone gives you a
clean notebook to execute. The pages here are rendered from copies executed ahead of time, so you
can read the results without running anything. Notebooks are never executed when these docs are
built.
```

Several examples need optional adapters (TransformerLens, NNsight, SAELens, circuit-tracer), a GPU,
and in some cases gated model weights. See {doc}`usage/developer_multi_repo_setup` for environment
setup.

## Gallery

| Notebook | Adapters | Demonstrates | Colab |
|---|---|---|---|
| {doc}`SAELens tutorial <notebooks/saelens_adapter_example/saelens_adapter_example>` | `sae_lens` | The SAELens adapter end to end: SAE-spliced forward passes, `logit_diffs_latent` analysis, per-latent attribution. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/saelens_adapter_example/saelens_adapter_example.ipynb) |
| {doc}`Circuit Tracer tutorial <notebooks/circuit_tracer_examples/circuit_tracer_adapter_example_basic>` | `circuit_tracer` | Attribution-graph generation basics with the circuit-tracer adapter. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/circuit_tracer_examples/circuit_tracer_adapter_example_basic.ipynb) |
| {doc}`CT analysis backend demo <notebooks/circuit_tracer_examples/ct_analysis_backend_demo>` | `circuit_tracer`, `nnsight`, `transformer_lens` | The analysis-ops pipeline running a full semantic concept intervention through the circuit-tracer analysis backend. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/circuit_tracer_examples/ct_analysis_backend_demo.ipynb) |
| {doc}`Concept-direction steering <notebooks/circuit_tracer_examples/ct_concept_steering_demo>` | `circuit_tracer`, `nnsight`, `transformer_lens` | Concept-direction-mediated, sign-aware, multi-feature steering (the `orange` color-vs-fruit sense disambiguation example). | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/circuit_tracer_examples/ct_concept_steering_demo.ipynb) |
| {doc}`Concept-direction steering (local Neuronpedia) <notebooks/circuit_tracer_examples/ct_concept_steering_demo_local_np>` | `circuit_tracer`, `nnsight`, `transformer_lens` | The same steering workflow against a locally hosted Neuronpedia instance. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/circuit_tracer_examples/ct_concept_steering_demo_local_np.ipynb) |
| {doc}`Neuronpedia integration <notebooks/neuronpedia_example/circuit_tracer_w_neuronpedia_example>` | `circuit_tracer`, `nnsight` | Generating and sharing a model attribution graph via the Neuronpedia extension (Gemma-3-1B-IT). | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/neuronpedia_example/circuit_tracer_w_neuronpedia_example.ipynb) |
| {doc}`Attribution analysis <notebooks/attribution_analysis/attribution_analysis>` | `circuit_tracer`, `transformer_lens` | Gradient- and ablation-based attribution composed with attribution-graph analysis and Neuronpedia lookups. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/attribution_analysis/attribution_analysis.ipynb) |
| {doc}`Op collections <notebooks/example_op_collections/op_collection_example>` | hub only (no session) | Publishing, pulling, and loading analysis-op collections — hub and local — via `HubAnalysisOpManager`. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/example_op_collections/op_collection_example.ipynb) |
| {doc}`Hub op opt-in <notebooks/example_op_collections/bundled_ops_hub_optin>` | hub only (no session) | How op names resolve before and after opting into a hub collection over the bundled ops (`prefer_ops`). | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/example_op_collections/bundled_ops_hub_optin.ipynb) |
| {doc}`Shared-analysis round-trip <notebooks/shared_analysis/shared_analysis_roundtrip>` | `circuit_tracer`, `nnsight`, `transformer_lens` | End-to-end shared analysis: publish attribution-graph steering results as an `AnalysisStore`, then replicate them as a second user. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/speediedan/interpretune/blob/main/src/it_examples/notebooks/publish/shared_analysis/shared_analysis_roundtrip.ipynb) |

The **Adapters** column names the non-`core` adapter contexts each session composes (`core` participates in
every session); "hub only" notebooks exercise the hub client layer without constructing a session.
