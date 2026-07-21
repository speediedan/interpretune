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
the dashboard benchmark suite in this repo providing the quantified evidence. Coordination PR:
*link forthcoming once the interpretune coordination PR opens*.

**On ownership**: the upstream repos — principally Neuronpedia, together with SAEDashboard — own
the core dashboard APIs and generation protocol today and will continue to own them. Interpretune
provides **one example** generation/import orchestration pipeline that exercises the proposed
scalable, more easily customizable, and ultimately streamable dashboard generation — demonstrating
the improvements, not redefining the interfaces.

### Documentation build-out

The documentation site (this site) is newly bootstrapped: a coherence pass over the converted
guides, notebook example rendering, and API-reference polish are in progress.

## Next: MVP milestone — shareable analysis artifacts

The [MVP milestone](https://github.com/speediedan/interpretune/milestone/1) centers on making
Interpretune's artifacts **shareable on the Hugging Face Hub** — most importantly:

- **AnalysisStore hub upload/download**: analyses as exchangeable datasets (the paradigm described
  in {doc}`concepts <concepts>`), enabling reproduction and composition of world-model analyses
  across researchers. This connects directly to the hub-based dashboard-availability pattern —
  one consistent hub-artifact story for dashboards, explanations, and analysis results.
- **Adapter/session shareability**: uploadable session and adapter configurations so an analysis
  is runnable, not just readable.
- **Cross-backend support hardening**: completing the dual-backend abstraction-layer validation
  ([#201](https://github.com/speediedan/interpretune/issues/201) — latent-model handle lifecycle,
  backend compatibility matrix) and broadening cross-backend demo/e2e coverage
  ([#224](https://github.com/speediedan/interpretune/issues/224)).

## Following: hub-resident, streamable dashboards

Building on the upstream Scalable Dashboard PRs: dashboards and locally-generated feature
explanations become Hugging Face Hub artifacts that viewers stream on demand, optionally
disintermediating the Neuronpedia DB for user-generated dashboards (both on neuronpedia.org and
local dev stacks) — subject to the same upstream-ownership framing above.

## Research directions

- **RTE cross-backend research** ([#220](https://github.com/speediedan/interpretune/issues/220)):
  the recognizing-textual-entailment research program that motivated the cross-backend demo
  infrastructure — resuming on the hardened dual-backend substrate.
- **Jacobian-space (J-lens) analysis support**
  ([#225](https://github.com/speediedan/interpretune/issues/225)): J-space read/probe/steer ops
  and per-feature J-space signatures as the principled generalization of 1-D logit-diff output
  projections, co-designed with AnalysisStore hub sharing.
- **Self-interpretability**: interpretability that accelerates model advancement via
  self-reflection — in addition to serving as the bridge for human access to AI world models.
  Internal model-reflection heuristics offered by Interpretune could enduringly improve RL
  exploration efficiency (effectively better sample efficiency).
- **Epistemic coherence**: coherence-oriented auxiliary objectives and evaluations treating
  world-model consistency as a first-class tuning signal.
- **Reflective cognition for counterfactuals**: latent-state evaluation as a mechanism for models
  to assess counterfactuals via their own latent states.
- **Meta-latent interfaces**: SAE meta-latents (and successors) as decomposition levels in a
  human/machine world-model interface.
- **Multimodal world models**: extending beyond the LLM-focused MVP.

## Considered (not MVP-blocking)

- **Core-protocol extraction** ([#6](https://github.com/speediedan/interpretune/issues/6)):
  generalize/extract the interoperability protocol out of Interpretune into a standalone
  distribution once the MVP stabilizes the protocol surface. Represented here deliberately as
  *considered* — the MVP proceeds without it, while new public surfaces are designed
  extraction-friendly. See the {doc}`design rationale <design_rationale>` for the fuller
  interoperability-protocol argument.
