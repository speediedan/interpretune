# Interpretune

```{image} _static/images/logos/logo_interpretune.svg
:alt: Interpretune
:width: 420px
:class: only-light
```

**A flexible framework for collaborative AI world model analysis and tuning.**

Interpretune is an AI world model analysis framework that enables a wide range of
interpretability methods and packages to leverage **composable, shareable analysis operations and
state**, accelerating collaborative world model analysis and tuning with PyTorch — letting both
humans and agents inspect the mechanistic and causal faithfulness of model reasoning at mutually
intelligible levels of abstraction.

Interpretune composes adapters at **multiple levels of abstraction** — the *framework* level
(core PyTorch, Lightning), the *interpretability latent-model* level (TransformerLens, NNsight),
and the *analysis* level (circuit-tracer, SAE-Lens) — over a shared session/protocol layer (see
the {doc}`adapter development guide <usage/adapter_development_guide>`). This composition pattern
is what lets researchers **collaborate across interpretability frameworks**: analytical
primitives, artifacts, and patterns written once run over many substrate combinations.

Analysis flows are built from composable operations — e.g. `extract_top_features`,
`gradient_attribution`, `ablation_attribution`, `feature_intervention`, `graph_prune`,
`concept_direction`, `compute_attribution_graph` — compiled over the active adapter composition,
with results captured in shareable {doc}`AnalysisStore <concepts>` datasets.

```{note}
Interpretune is **pre-MVP**: APIs are subject to change. The {doc}`roadmap <roadmap>` describes
the path to the initial alpha release.
```

A note on terminology: throughout these docs, "world model" is used in the **epistemic/semantic**
sense — the internal representations and beliefs a model encodes about the world — as studied in
LLM interpretability, rather than the (related but distinct) predictive **visual world models** of
embodied-agent and model-based-RL research. The initial MVP focuses on LLMs; fuller multimodal
support is planned (see the {doc}`roadmap <roadmap>`).

```{toctree}
:caption: Getting Oriented
:maxdepth: 1

concepts
roadmap
configuration
design_rationale
```

```{toctree}
:caption: Usage Guides
:maxdepth: 1

usage/session_module_datamodule_usage
usage/analysis_runner_usage
usage/custom_ops_composition_guide
usage/interpretune_intervention_apis
usage/adapter_development_guide
usage/framework_level_adapters
usage/circuit_tracer_backend_support
usage/neuronpedia_dashboard_pipeline
usage/analysis_store_serialization
usage/cache_behavior
usage/generation_precedence
usage/analysis_injection_usage
usage/developer_multi_repo_setup
```

```{toctree}
:caption: Design Notes
:maxdepth: 1

design/protocol_architecture_working_design
design/intervention_hook_pattern_support
design/resource_management
design/tl_config_hierarchy_overview
design/tl_style_naming_implementation
design/ht_bridge_parity_behavior
design/fts_transformerlens_integration
```

```{toctree}
:caption: API Reference
:maxdepth: 1

api
```

## Who maintains Interpretune?

Interpretune's original author and initial maintainer,
[@speediedan](https://github.com/speediedan) (Daniel Dale), has been contributing to PyTorch and
Lightning for more than half a decade, is a contributor to circuit-tracer, TransformerLens, and
SAE-Lens among other frameworks, and is the author of research packages including
[finetuning-scheduler](https://github.com/speediedan/finetuning-scheduler).

**Contributors are enthusiastically welcome!** The
[IT-MVP milestone](https://github.com/speediedan/interpretune/milestone/1) is the best place to
find priority items and good first issues — thank you in advance for contributing to the
open-source interpretability ecosystem.
