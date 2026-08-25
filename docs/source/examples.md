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

## What each notebook needs

The base install covers three of these. The rest additionally need the `git-deps` group, because they
use the circuit-tracer adapter and no circuit-tracer release carries the surface interpretune uses.

```bash
uv pip install -e ".[examples]"                    # SAELens tutorial + the two hub-only notebooks
uv pip install -e ".[examples]" --group git-deps   # everything else
```

| Notebook | Install | GPU | Model access |
|---|---|---|---|
| Op collections | `.[examples]` | not required | HF token — it publishes a collection to your namespace |
| Hub op opt-in | `.[examples]` | not required | HF token |
| SAELens tutorial | `.[examples]` | bf16 CUDA | none — `gpt2` |
| Circuit Tracer tutorial | `+ git-deps` | bf16 CUDA | **gated** — `google/gemma-2-2b` |
| CT analysis backend demo | `+ git-deps` | bf16 CUDA | **gated** — `google/gemma-2-2b` |
| Concept-direction steering | `+ git-deps` | bf16 CUDA | **gated** — `google/gemma-2-2b` |
| Attribution analysis | `+ git-deps` | bf16 CUDA | **gated** — `google/gemma-2-2b` |
| Shared-analysis round-trip | `+ git-deps` | bf16 CUDA | **gated** — `google/gemma-2-2b`, plus an HF token to publish |
| Neuronpedia integration | `+ git-deps` | bf16 CUDA | **gated** — `google/gemma-3-1b-it`, plus a Neuronpedia API key |
| Concept-direction steering (local Neuronpedia) | `+ git-deps` | bf16 CUDA | **gated** — `google/gemma-3-1b-it`, plus a locally hosted Neuronpedia |

```{note}
**Gated** here means the weights are public but require accepting a licence on the Hugging Face model
page and then authenticating. Export the token as `HF_GATED_PUBLIC_REPO_AUTH_KEY` — the variable
these example configs name — or log in with `hf auth login`. An unset variable is not fatal: the
configs fall back to any ambient Hugging Face credential and warn, rather than raising during
session construction.
```

`papermill` and `nbmake` (in the `test` group) execute these notebooks in CI. Running one yourself
needs neither.

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
