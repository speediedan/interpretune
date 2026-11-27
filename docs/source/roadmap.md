# Roadmap

Interpretune is **pre-MVP**, working toward an initial alpha release. This roadmap is organized
by priority, opening with work actively in flight; the {doc}`concepts <concepts>` page defines the
constructs referenced here.

## In flight

### Coordinated upstream Scalable Dashboard PRs

A coordinated multi-repo PR set upstreaming the scalable-dashboard-generation work validated
end-to-end with Interpretune: peak-memory-controlled dashboard generation with per-stage CUDA
accounting and pretokenization records (SAEDashboard), supporting surfaces in SAELens, the
circuit-tracer attribution-target abstraction, and Neuronpedia-side import/coverage support — with
the dashboard benchmark suite in this repo providing the quantified evidence. The upstream PR set is
tracked in [#231](https://github.com/interpretune/interpretune/issues/231).

**On ownership**: the upstream repos — principally Neuronpedia, together with SAEDashboard — own
the core dashboard APIs and generation protocol today and will continue to own them. Interpretune
provides **one example** generation/import orchestration pipeline that exercises the proposed
scalable, more easily customizable, and ultimately streamable dashboard generation — demonstrating
the improvements, not redefining the interfaces.

### Documentation build-out

The documentation site (this site) is newly bootstrapped: a coherence pass over the converted
guides, notebook example rendering, and API-reference polish are in progress.

## Next: MVP milestone — shareable analysis artifacts

The [MVP milestone](https://github.com/interpretune/interpretune/milestone/1) centers on making
Interpretune's artifacts **shareable on the Hugging Face Hub** — most importantly:

- **AnalysisStore hub upload/download**: analyses as exchangeable datasets (the paradigm described
  in {doc}`concepts <concepts>`), enabling reproduction and composition of latent model analyses
  across researchers. This connects directly to the hub-based dashboard-availability pattern —
  one consistent hub-artifact story for dashboards, explanations, and analysis results.
- **Adapter/session shareability**: uploadable session and adapter configurations so an analysis
  is runnable, not just readable.
- **Cross-backend support hardening**: the dual-backend abstraction-layer validation has shipped
  ([#201](https://github.com/interpretune/interpretune/issues/201): latent model handle lifecycle,
  backend compatibility matrix); broadening cross-backend demo/e2e coverage
  ([#224](https://github.com/interpretune/interpretune/issues/224)) is deferred.

## Hub-resident, streamable dashboards

Shipped: generated dashboard corpora publish to Hugging Face Hub buckets, laid out so one feature can
be read without downloading the corpus, and a consumer finds them per trusted publisher and imports
them into a local Neuronpedia (the
[pipeline guide](../neuronpedia_dashboard_pipeline.md#publishing-generated-dashboards-to-the-hub) has
the commands).

Following, building on the upstream Scalable Dashboard PRs
([#231](https://github.com/interpretune/interpretune/issues/231)): viewers stream those corpora on
demand, optionally disintermediating the Neuronpedia DB for user-generated dashboards (both on
neuronpedia.org and local dev stacks), subject to the same upstream-ownership framing above. Locally
generated feature explanations are not yet part of a published corpus.

### Standardized hub-based dashboard discovery

Shipped ([#262](https://github.com/interpretune/interpretune/issues/262)): published corpora are discovered through
Hub collections whose items are buckets, with each corpus's own `dashboards.json` as the only description of what it
holds. A client's trust list is the collections it reads (the project's own by default), and a publisher registers a
corpus by adding its bucket to a collection, with no client release. Trust is per publisher for now, and the
permission model is still evolving (a finer, sub-publisher posture may follow); AnalysisStore discovery and trust are
tracked in [#1008](https://github.com/interpretune/interpretune/issues/1008). The pipeline guide's
[dashboard-corpus collection](../neuronpedia_dashboard_pipeline.md#hub-dashboard-discovery) section has the API.

## Research directions

- **RTE cross-backend research** ([#667](https://github.com/interpretune/interpretune/issues/667);
  attribution-graph auto-pruning and tuning in [#220](https://github.com/interpretune/interpretune/issues/220)):
  the recognizing-textual-entailment research program that motivated the cross-backend demo
  infrastructure — resuming on the hardened dual-backend substrate.
- **Jacobian-space (J-lens) analysis support**
  ([#225](https://github.com/interpretune/interpretune/issues/225) in-tree,
  [#273](https://github.com/interpretune/interpretune/issues/273) as a separately published op
  collection): the read/probe/steer ops, the `concept_basis="jlens_*"` selector for
  `concept_direction`, per-feature J-space signatures, and the `patch` / `clamp` /
  `reject` write ops have shipped, and the separately published collection is
  the first genuinely non-bundled proving ground for hub registration. Open research:
  J-space attribution graphs ([#683](https://github.com/interpretune/interpretune/issues/683)),
  RTE attribution graphs with a learned pruner and graph-guided fine-tuning
  ([#667](https://github.com/interpretune/interpretune/issues/667)), and fine-tuning defined through the
  lens ([#666](https://github.com/interpretune/interpretune/issues/666)). The demo and docs refresh
  ([#426](https://github.com/interpretune/interpretune/issues/426)) is nearly done.
- **Self-interpretability**: interpretability that accelerates model advancement via
  self-reflection — in addition to serving as the bridge for human access to AI world models.
  Internal model-reflection heuristics offered by Interpretune could enduringly improve RL
  exploration efficiency (effectively better sample efficiency).
- **Epistemic coherence**: coherence-oriented auxiliary objectives and evaluations treating
  world model consistency as a first-class tuning signal.
- **Reflective cognition for counterfactuals**: latent-state evaluation as a mechanism for models
  to assess counterfactuals via their own latent states.
- **Meta-latent interfaces**: SAE meta-latents (and successors) as decomposition levels in a
  human/machine world model interface.
- **Multimodal world models**: extending beyond the LLM-focused MVP.

## Considered (not MVP-blocking)

- **Core-protocol extraction** ([#6](https://github.com/interpretune/interpretune/issues/6)):
  generalize/extract the interoperability protocol out of Interpretune into a standalone
  distribution once the MVP stabilizes the protocol surface. Represented here deliberately as
  *considered* — the MVP proceeds without it, while new public surfaces are designed
  extraction-friendly. See the {doc}`design rationale <design_rationale>` for the fuller
  interoperability-protocol argument.
