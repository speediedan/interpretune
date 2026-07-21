# Core Concepts

> Interpretune: mechanistically informed AI model tuning.
> A flexible framework for collaborative AI world model analysis and tuning.

The intention of the framework: offer the **appropriate level of abstraction** for world model
analysis — seamlessly composing a wide range of interpretability frameworks and methods while
operating on the latent spaces of any PyTorch model and producing composable, shareable
artifacts. Interpretune strives to compose analysis operations at levels of abstraction that
**bridge human and agent world models**.

This page summarizes the constructs the rest of the documentation (and the
{doc}`roadmap <roadmap>`) builds on. It is written for prospective collaborators who want the
shape of the framework without reading the full usage guides.

## World models, in the epistemic sense

Interpretune is oriented around analyzing and tuning the **world models of AI systems** — used
here in the *epistemic/semantic* sense: the internal representations, concepts, and beliefs a
model encodes about the world, as studied by LLM interpretability research. This is related to,
but distinct from, the *visual/predictive world models* of embodied-agent and model-based-RL
research (models that predict future environment states). When these docs say "world model," they
mean the former. The initial MVP focuses on LLMs; fuller multimodal-model support is planned.

## Session composition: `ITSession`, adapters, and protocols

An {py:class}`~interpretune.session.ITSession` composes a datamodule and a module with an
**adapter context** — an ordered tuple such as `(core, transformer_lens, circuit_tracer)` or
`(core, nnsight, circuit_tracer)` — producing an "interpretunable" module whose capabilities are
the composition of the selected adapters. Adapters exist for TransformerLens (including
TransformerBridge), SAE-Lens, NNsight, circuit-tracer, and Lightning (training orchestration).
The protocol layer (`interpretune.protocol`) defines the structural contracts that make the same
analysis code executable across adapter compositions.

## Latent-model abstractions (not just SAEs)

Analysis operations target **latent models** at an appropriate level of abstraction — SAEs and
transcoders are the most common instances today, but the protocol language is deliberately
general: `LatentAnalysisTargets`, latent-model FQNs, latent-model handles, and operations like
`model_fwd_w_cache_latent_models` compose over any latent-model construct that satisfies the
protocols. Documentation and APIs use this latent-model language except where a statement is
genuinely SAE-specific. One motivating direction: SAE meta-latents as one decomposition level in
a broader human/machine world-model interface.

## Analysis operations and the op compiler

Analysis flows are built from composable **ops** (`it.concept_direction`,
`it.compute_attribution_graph`, `it.intervention_from_concept`, …) declared with input/output
schemas and compiled into execution plans. Ops are dispatched over the active adapter composition,
so a concept-direction intervention runs identically over a TransformerLens or NNsight substrate.
Native ops ship with generated `.pyi` stubs for IDE support.

## `AnalysisStore`: shareable world-model analysis artifacts

The **AnalysisStore** is a key abstraction of the interpretune protocol: a serialized,
schema-described dataset capturing the artifacts of analysis runs — activations, latent
attributions, attribution graphs, intervention results — in a backend-agnostic format. The
paradigm it enables: world-model analyses become **exchangeable datasets** rather than one-off
notebook state, so researchers can reproduce, extend, and *compose* each other's analyses. Making
AnalysisStores (and adapter/session configurations) uploadable/downloadable via the Hugging Face
Hub is the centerpiece of the MVP milestone — the same hub-resident pattern that underpins
streamable dashboard availability (see the {doc}`roadmap <roadmap>`).

## Extensions

Cross-cutting capabilities ship as extensions: memory profiling (`memprofiler`), debug generation
(`debug_generation`), and Neuronpedia integration (dashboard generation/import pipelines and
feature-explanation flows).
