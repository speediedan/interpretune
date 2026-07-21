# Roadmap

Interpretune is **pre-MVP**, working toward an initial alpha release. This roadmap is written for
upstream maintainers and prospective collaborators; the {doc}`concepts <concepts>` page defines
the constructs referenced here.

## Wave 1 (in flight): Scalable Dashboards PR wave

A coordinated multi-repo PR set upstreaming the scalable-dashboard-generation work validated in
this repo — peak-memory-controlled dashboard generation with per-stage CUDA accounting and
pretokenization records (SAEDashboard), supporting surfaces in SAELens, the circuit-tracer
attribution-target abstraction, and Neuronpedia-side import/coverage support — with the dashboard
benchmark suite in this repo providing the quantified evidence. Interpretune's role: the
generation/import orchestration pipeline, benchmark tooling, and the dual-backend circuit-tracer
analysis integration that consumes the dashboards.

## Wave 2: hub-resident, streamable dashboards

Make dashboards **hub-resident and streamable**, optionally disintermediating the Neuronpedia DB
for user-generated dashboards (both on neuronpedia.org and local dev stacks): dashboards and
locally-generated feature explanations become Hugging Face Hub artifacts that viewers stream on
demand. Wave 2 also carries Jacobian-space (J-lens) analysis support as an adjacent research
thread.

## MVP milestone: shareable analysis artifacts

The [MVP milestone](https://github.com/speediedan/interpretune/milestone/1) centers on making
interpretune's artifacts **shareable on the Hugging Face Hub** — most importantly:

- **AnalysisStore hub upload/download**: analyses as exchangeable datasets (the paradigm described
  in {doc}`concepts <concepts>`), enabling reproduction and composition of world-model analyses
  across researchers. This connects directly to the hub-based dashboard-availability pattern —
  one consistent hub-artifact story for dashboards, explanations, and analysis results.
- **Adapter/session shareability**: uploadable session and adapter configurations so an analysis
  is runnable, not just readable.
- Supporting hardening: latent-model handle lifecycle validation, cross-backend artifact parity,
  and the backend-compatibility documentation matrix.

## Research directions (longer horizon)

- **Epistemic coherence**: coherence-oriented auxiliary objectives and evaluations that treat
  world-model consistency as a first-class training/tuning signal, addressing translation
  shortcomings between latent representations and human-legible structure.
- **Reflective cognition for counterfactuals**: using latent-state evaluation to impart a form of
  reflective cognition — models assessing counterfactuals via their own latent states rather than
  purely via sampled continuations.
- **Meta-latent interfaces**: SAE meta-latents (and successors) as decomposition levels in a
  human/machine world-model interface.
- **Multimodal world models**: extending beyond the LLM-focused MVP to multimodal models.
