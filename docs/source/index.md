# Interpretune

```{image} _static/images/logos/logo_interpretune.svg
:alt: Interpretune
:width: 420px
:class: only-light
```

**A flexible framework for collaborative AI world model analysis and tuning.**

Interpretune is a (pre-MVP) AI world model analysis framework. It strives to provide composable, shareable latent-space analysis mutually intelligible to humans and agents. By granting a wide range of interpretability
methods and packages access to composable, shareable analysis operations and state it accelerates novel,
collaborative, world model analysis and tuning with PyTorch. Both humans and agents can inspect
and refine the mechanistic and causal faithfulness of model reasoning at mutually intelligible
levels of abstraction. Core framework goals:

- transparent, causally faithful reasoning
- augmented model self-reflection
- world-model-guided collaborative tuning

Interpretune composes adapters at multiple levels of abstraction over a shared
session/protocol layer (see the
{doc}`adapter development guide <usage/adapter_development_guide>`):

- the *framework* level — core PyTorch, Lightning
- the *interpretability latent-model* level — TransformerLens, NNsight
- the *analysis* level — circuit-tracer, SAELens

This composition pattern is what will allow users to collaborate across interpretability
frameworks: analytical primitives, artifacts, and patterns written once run over many
substrate combinations.

Analysis flows are built from composable operations — e.g. `extract_top_features`,
`gradient_attribution`, `ablation_attribution`, `feature_intervention`, `graph_prune`,
`concept_direction`, `compute_attribution_graph`. Operations are compiled over the active
adapter composition, and results are captured in shareable {doc}`AnalysisStore <concepts>`
datasets.

```{note}
Interpretune is **pre-MVP**: APIs are subject to change. The {doc}`roadmap <roadmap>` describes
the path to the initial alpha release.
```

A note on terminology: throughout these docs, "world model" is used in the **epistemic/semantic**
sense — the internal representations and beliefs a model encodes about the world — as studied in
interpretability, rather than the (related but distinct) predictive **visual world models** of
embodied-agent and model-based-RL research. The initial MVP focuses on LLMs; fuller multimodal
support is planned (see the {doc}`roadmap <roadmap>`).

## Quickstart

Interpretune uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and create an out-of-tree venv (keeps parallel envs easy)
git clone https://github.com/speediedan/interpretune.git && cd interpretune
export VENV_BASE=${HOME}/.venvs IT_TARGET_VENV=it_latest
uv venv ${VENV_BASE}/${IT_TARGET_VENV} --python 3.13
source ${VENV_BASE}/${IT_TARGET_VENV}/bin/activate

# Install with development dependencies
# (the git-deps group becomes optional once circuit-tracer is published on PyPI)
uv pip install -e ".[test,examples,lightning,profiling]" --group git-deps dev
```

### Basic usage

Compose a session from an adapter context, initialize it, then run analysis operations. The same op
runs unchanged under a different adapter context — that portability is the point:

```python
import interpretune as it
from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY

dm_cfg, m_cfg, dm_cls, m_cls = MODULE_EXAMPLE_REGISTRY.get("gemma2.rte_demo.circuit_tracer")
session = it.ITSession(it.ITSessionConfig(
    adapter_ctx=(it.Adapter.core, it.Adapter.nnsight, it.Adapter.circuit_tracer),
    datamodule_cfg=dm_cfg, module_cfg=m_cfg, datamodule_cls=dm_cls, module_cls=m_cls,
))
it.it_init(**session)
result = it.intervention_from_concept(session.module, ...)
```

### Where to go next

- New to the framework? Start with {doc}`concepts` — sessions, protocols, ops, and `AnalysisStore`.
- Want something runnable? The {doc}`examples` are executable notebooks.
- Building your own analysis? See {doc}`usage/custom_ops_composition_guide`.
- Adding a backend? See {doc}`usage/adapter_development_guide`.

For advanced builds (locked CI requirements, multi-repo from-source composition), see
{doc}`usage/developer_multi_repo_setup`.

```{toctree}
:caption: Getting Oriented
:maxdepth: 1
:hidden:

concepts
roadmap
configuration
design_rationale
```

```{toctree}
:caption: Core Workflow
:maxdepth: 1
:hidden:

usage/session_module_datamodule_usage
usage/analysis_runner_usage
usage/analysis_store_serialization
usage/cache_behavior
usage/generation_precedence
```

```{toctree}
:caption: Composing Analysis
:maxdepth: 1
:hidden:

usage/custom_ops_composition_guide
usage/interpretune_intervention_apis
usage/analysis_injection_usage
```

```{toctree}
:caption: Adapters & Backends
:maxdepth: 1
:hidden:

usage/adapter_development_guide
usage/framework_level_adapters
usage/circuit_tracer_backend_support
```

```{toctree}
:caption: Dashboards & Neuronpedia
:maxdepth: 1
:hidden:

usage/neuronpedia_dashboard_pipeline
usage/verifying_dashboard_generation
```

```{toctree}
:caption: Examples
:maxdepth: 1
:hidden:
:glob:

examples
notebooks/**
```

```{toctree}
:caption: Development
:maxdepth: 1
:hidden:

usage/developer_multi_repo_setup
```

```{toctree}
:caption: Design Notes
:maxdepth: 1
:hidden:

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
:hidden:

api
```

## Who maintains Interpretune?

Interpretune's original author and initial maintainer,
[@speediedan](https://github.com/speediedan) (Daniel Dale), has been contributing to PyTorch and
Lightning for more than half a decade, is a contributor to circuit-tracer, TransformerLens, and
SAE-Lens among other frameworks, and is the author of research packages including
[finetuning-scheduler](https://github.com/speediedan/finetuning-scheduler).

**Contributors are enthusiastically welcomed!** The
[IT-MVP milestone](https://github.com/speediedan/interpretune/milestone/1) is the best place to
find priority items and good first issues — thank you in advance for contributing to the
open-source interpretability ecosystem.
