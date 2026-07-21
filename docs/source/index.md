# interpretune

**A flexible framework for collaborative AI world model analysis and tuning.**

Interpretune composes interpretability adapters (TransformerLens, SAE-Lens, NNsight,
circuit-tracer, Lightning) over a shared session/protocol layer so that analysis operations —
activation caching, latent-model splicing, attribution graphs, concept-direction interventions —
can be written once and executed across backends, with results captured in shareable
{doc}`AnalysisStore <concepts>` datasets.

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
