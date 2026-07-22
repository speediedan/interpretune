<div align="center">

<img src="docs/source/_static/images/logos/logo_interpretune.svg" alt="Interpretune" width="420"/>

**A flexible framework for collaborative AI world model analysis and tuning.**

*Composable, shareable latent-space analysis mutually intelligible to humans and agents.*

[![CI](https://github.com/speediedan/interpretune/actions/workflows/ci_test-full.yml/badge.svg)](https://github.com/speediedan/interpretune/actions/workflows/ci_test-full.yml)
[![Docs](https://readthedocs.org/projects/interpretune/badge/?version=latest)](https://interpretune.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/speediedan/interpretune/branch/main/graph/badge.svg)](https://codecov.io/gh/speediedan/interpretune)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/speediedan/interpretune/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)

</div>

> **Status: pre-MVP**, working toward an initial alpha release — active development, APIs subject
> to change. This README is written especially for upstream maintainers and prospective
> collaborators: it summarizes what interpretune is, the constructs it's built on, and the
> [roadmap](#roadmap) from here to alpha.

## What is interpretune?

Interpretune is an AI world model analysis framework that enables a wide range of
interpretability methods and packages to leverage **composable, shareable analysis operations and
state**, accelerating collaborative world model analysis and tuning with PyTorch. It lets both
humans and agents inspect and refine the mechanistic and causal faithfulness of model reasoning
at mutually intelligible levels of abstraction.

**What that unlocks**: transparent, causally faithful reasoning with greater confidence in
conclusions; model self-reflection; inter-agent latent-space collaboration; and more
sample-efficient, world-model-guided collaborative tuning.

Interpretune composes adapters at **multiple levels of abstraction** — the *framework* level
(core PyTorch, [Lightning](https://github.com/Lightning-AI/pytorch-lightning)), the
*interpretability latent-model* level ([TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
[NNsight](https://github.com/ndif-team/nnsight)), and the *analysis* level
([circuit-tracer](https://github.com/safety-research/circuit-tracer),
[SAE-Lens](https://github.com/decoderesearch/SAELens)) — over a shared session/protocol layer.
This composition pattern is what lets researchers collaborate across interpretability frameworks:
analysis operations (e.g. `extract_top_features`, `gradient_attribution`, `ablation_attribution`,
`feature_intervention`, `graph_prune`, `concept_direction`, `compute_attribution_graph`) are
**written once and executed across backends**, with results captured as shareable datasets.

A note on terminology: we use "world model" in the **epistemic/semantic** sense — the internal
representations, concepts, and beliefs a model encodes about the world, as studied in AI
interpretability — related to but distinct from the predictive *visual world models* of
embodied-agent/model-based-RL research. The initial MVP focuses on LLMs; fuller multimodal support
is planned.

## Core constructs (60-second tour)

| Construct | What it is |
| --- | --- |
| **`ITSession`** | Composes a datamodule + module with an ordered **adapter context** (e.g. `(core, transformer_lens, circuit_tracer)` or `(core, nnsight, circuit_tracer)`); the result is an "interpretunable" module whose capabilities are the composition of the selected adapters. |
| **Protocols** | Structural contracts (`interpretune.protocol`) that make the same analysis code executable across adapter compositions. |
| **Latent-model abstractions** | Analysis targets are **latent models** at an appropriate abstraction level — SAEs/transcoders today, but the protocol language (`LatentAnalysisTargets`, latent-model handles, ops like `model_fwd_w_cache_latent_models`) deliberately generalizes beyond SAEs. |
| **Analysis ops** | Composable, schema-declared operations (`it.concept_direction`, `it.compute_attribution_graph`, `it.intervention_from_concept`, …) compiled into execution plans and dispatched over the active backends. |
| **`AnalysisStore`** | The key sharing abstraction: serialized, schema-described datasets capturing analysis artifacts (activations, attributions, graphs, intervention results) — making world-model analyses **exchangeable datasets** rather than one-off notebook state. |
| **Extensions** | Cross-cutting capabilities: memory profiling, debug generation, Neuronpedia dashboard/explanation integration. |

A minimal flavor of the composition pattern (see the
[docs](https://interpretune.readthedocs.io/en/latest/) and `src/it_examples/notebooks/` demos for
runnable versions):

```python
import interpretune as it
from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY

dm_cfg, m_cfg, dm_cls, m_cls = MODULE_EXAMPLE_REGISTRY.get("gemma2.rte_demo.circuit_tracer")
session = it.ITSession(it.ITSessionConfig(
    adapter_ctx=(it.Adapter.core, it.Adapter.nnsight, it.Adapter.circuit_tracer),
    datamodule_cfg=dm_cfg, module_cfg=m_cfg, datamodule_cls=dm_cls, module_cls=m_cls,
))
it.it_init(**session)
# the same op runs unchanged under the transformer_lens adapter context
result = it.intervention_from_concept(session.module, ...)
```

## Installation

Interpretune uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/speediedan/interpretune.git && cd interpretune

# Set preferred venv base directory and name and create out-of-tree venv (recommended to enable multiple parallel envs)
export VENV_BASE=${HOME}/.venvs
export IT_TARGET_VENV=it_latest
uv venv ${VENV_BASE}/${IT_TARGET_VENV} --python 3.13 && source ${VENV_BASE}/${IT_TARGET_VENV}/bin/activate

# Install the package with all necessary development dependencies
# Note: the git-deps group is optional once circuit-tracer is published on PyPI
uv pip install -e ".[test,examples,lightning,profiling]" --group git-deps dev
```

For advanced development builds (locked CI requirements, multi-repo from-source composition), use
the build script:

```bash
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest
# from-source example:
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest \
  --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"
```

## Roadmap

### In flight: coordinated upstream Scalable Dashboard PRs

A coordinated multi-repo PR set upstreaming scalable dashboard generation validated end-to-end
with Interpretune: peak-memory-controlled generation with per-stage CUDA accounting and
pretokenization records (SAEDashboard), supporting surfaces in SAELens, the circuit-tracer
attribution-target abstraction, and Neuronpedia import/coverage support — with the dashboard
benchmark suite in this repo providing quantified evidence. *(Coordination PR link forthcoming.)*

**On ownership**: the upstream repos — principally Neuronpedia, together with SAEDashboard — own
the core dashboard APIs and generation protocol and will continue to own them; Interpretune
provides **one example** generation/import orchestration pipeline demonstrating the improved
scalability, example-aligned customizability, and (soon) direct shareability/streamability.

### Next: MVP milestone — shareable analysis artifacts

The [MVP milestone](https://github.com/speediedan/interpretune/milestone/1) centers on making
Interpretune artifacts shareable on the Hub — most importantly **AnalysisStore upload/download**
(analyses as exchangeable, reproducible, composable datasets), **adapter/session shareability**
(an analysis is runnable, not just readable), and **cross-backend support hardening**
([#201](https://github.com/speediedan/interpretune/issues/201),
[#224](https://github.com/speediedan/interpretune/issues/224)). This is the same hub-artifact
pattern as the streamable-dashboards effort — one consistent story for dashboards, explanations,
and analysis results.

### Following: hub-resident, streamable dashboards

Dashboards and locally-generated feature explanations become **Hugging Face Hub artifacts that
viewers stream on demand**, optionally disintermediating the Neuronpedia DB for user-generated
dashboards (on neuronpedia.org and local dev stacks alike) — same upstream-ownership framing as
above.

### Research directions

RTE cross-backend research ([#220](https://github.com/speediedan/interpretune/issues/220)),
Jacobian-space (J-lens) analysis support
([#225](https://github.com/speediedan/interpretune/issues/225)), **self-interpretability**
(interpretability that accelerates model advancement via self-reflection, including internal
reflection heuristics for more sample-efficient RL exploration), epistemic-coherence objectives,
reflective cognition for counterfactuals via latent-state evaluation, meta-latent world-model
interfaces, and multimodal world-model support — see the
[roadmap docs](https://interpretune.readthedocs.io/en/latest/roadmap.html).

## Documentation

- [Latest docs](https://interpretune.readthedocs.io/en/latest/) — concepts, usage guides, design
  notes, API reference
- Demo notebooks: `src/it_examples/notebooks/` (circuit-tracer analysis + concept-steering demos,
  SAE-Lens and NNsight adapter examples, with Colab badges on published copies)

## Testing

```bash
# Standard suite
python -m pytest src/interpretune tests -v

# Special phases (standalone GPU, profiling)
./tests/special_tests.sh --mark_type=standalone
./tests/special_tests.sh --mark_type=profile_ci
```

CI runs the full matrix (Linux/macOS/Windows CPU + a self-hosted GPU pipeline with
CUDA/standalone/profiling phases); coverage is tracked on
[codecov](https://codecov.io/gh/speediedan/interpretune). See `tests/README.md` for the testing
guide and `requirements/utils/lock_ci_requirements.sh` for CI-locked requirements regeneration.

## Contributing

Contributors are enthusiastically welcomed — Interpretune is built to be a
community effort, and the transition from pre-MVP to alpha is the perfect moment to get involved.
The [IT-MVP milestone](https://github.com/speediedan/interpretune/milestone/1) is the best place
to find priority items and good first issues, the
[issue tracker](https://github.com/speediedan/interpretune/issues) tags documentation and
`help wanted` items, and the [roadmap](#roadmap) above orients where things are headed. Thank you
in advance for contributing to the open-source interpretability ecosystem — every issue, doc fix,
and PR genuinely helps.

## License

[Apache 2.0](./LICENSE)
