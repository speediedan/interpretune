# Examples

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

## Neuronpedia dashboards: public or local

The concept-direction steering demo comes in two variants, so neither has to branch on which
substrate you have:

- **`ct_concept_steering_demo`** — the default. Resolves feature dashboards and explanations from
  the public [neuronpedia.org](https://www.neuronpedia.org), so it needs only a GPU and the model
  weights. Start here.
- **`ct_concept_steering_demo_local_np`** — the same analysis against a **local** Neuronpedia stack
  with dashboards you generated yourself, plus optional locally generated feature explanations and a
  user-curated feature-steering step. To produce the dashboards it reads, see
  {doc}`usage/neuronpedia_dashboard_pipeline`.

```{toctree}
:maxdepth: 1
:glob:

notebooks/**
```
