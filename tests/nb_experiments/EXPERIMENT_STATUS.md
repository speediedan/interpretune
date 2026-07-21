# Notebook Experimentation Status

**Date:** 2026-07-11 (7c-amendments update; prior full revision 2026-07-07)
**Scope:** `tests/nb_experiments/` (shared notebook harness + `concept_direction/` experiment family)
**Last experimental checkpoint:** 2026-04-27 (`f53f89a` — "Checkpoint sign-aware feature selection and concept-direction validation")

## Purpose and Relationship to the Dashboard-Infrastructure Work

This directory holds the notebook-based concept-direction experimentation that motivated the
2026-05 → 2026-07 "scalable dashboards" workstream (the `neuronpedia_dashboard_pipeline` in
Interpretune plus the associated SAEDashboard / SAELens / Neuronpedia PR wave). The research
question is whether useful, steerable concept directions can be extracted from Gemma models
(gemma-2-2b, gemma-3-1b, gemma-3-4b; PT and IT variants) via two complementary approaches —
token-embedding differences ("embed") and answer-position latent-state differences ("store") —
and intervened on through GemmaScope transcoder features selected with circuit-tracer
attribution. The prompt/task framing throughout is classification-style (capitals/states,
dog/cat, Ohio entities, the color-vs-fruit sense of "orange"), sitting alongside Interpretune's
core RTE/BoolQ classification benchmark task.

By late April 2026 the experimentation had reached the point where the next steps (sign-aware
feature subsets, calibration of store-direction geometry, per-feature semantic validation on
gemma-3-1b-it) required **granular feature-level inspection that upstream Neuronpedia could not
provide**: no transcoder dashboards existed for gemma-3-1b-it at all (only a few res-16k layers),
and the gemma-3-4b-it 16k set covered just 12 of 34 layers (see `experiment_summary.md` items 10
and DQ-2). The only pre-existing feature semantics we had for the 1B lineages were two
hand-labeled 16k features (L16/155 "ripe fruit", L25/4973 "not color"). Closing that gap —
generating full 262k-width, task-relevant (RTE/BoolQ prompt corpus) dashboards for gemma-3-1b-it
locally, at practical cost — is what the ~2-month dashboard-infrastructure effort delivered
(see "Dashboard capability" below). Experimentation in this directory has been paused since
2026-04-27 while that infrastructure was built; nothing here is abandoned.

## Directory Layout

- `tests/nb_experiments/` — experiment-agnostic harness: layered YAML config loading
  (`config.py`), model/session registry (`session.py`, `configs/model_specs.yaml`), papermill
  launcher (`nb_experiment_launcher.py`), shared pipeline helpers (`nb_harness_utils.py`,
  `pipeline_patterns.py`), and local-Neuronpedia graph upload/cleanup (`local_graph_cleanup.py`).
- `concept_direction/` — the experiment family: source notebook
  (`concept_direction_template.ipynb` + `concept_direction.py`), per-experiment YAML configs,
  living analysis docs, and backend-parity test suites.
- `concept_direction/analysis/` — post-hoc analysis notebooks/launchers (cross-run comparison,
  latent dynamics, intervention drift) and their configs.
- Generated artifacts are written outside the repo to `/tmp/it_concept_direction_experiments/`
  (experiment notebooks) and `/tmp/it_concept_direction_experiments/analysis/` (analysis
  notebooks); they are timestamped, disposable execution artifacts, not source.

## Current Experiment Inventory

| Experiment / artifact | Question it addresses | Latest artifact (date) | Status |
|---|---|---|---|
| V1–V12 waves: capitals_states + dog/cat/cat_dog across gemma-2-2b / gemma-3-1b / gemma-3-4b, PT+IT, 16k+262k (`concept_direction_template.ipynb` + `configs/`) | Do embed vs store directions agree? Do interventions steer the target logit gap? Direction reversal symmetry, prompt sensitivity (OQ-I), direct projection vs feature mediation, context-enhanced extraction | Results tables in `concept_direction/experiment_summary.md` (last updated 2026-04-01; waves ran 2025-03 → 2026-03/04) | **Complete** — findings distilled in `experiment_summary.md` §1–§3; per-wave summaries were pruned at the 2026-04-25 consolidation (dangling `archived_analysis/` links replaced with consolidation notes, 2026-07-07) |
| Ohio entities on gemma-3-4b-it (`cp_ohio_entities_gemma_it.yaml`, `gemma3_4b_it_local_oqi_reasoning_oh_2975_15708*.yaml`) | Prompt-anchor sensitivity and 4B direct-op parity for an ambiguous entity-membership concept | `/tmp/.../gemma3_4b_it_local_oqi_reasoning_oh_2975_15708_20260425_145236.ipynb` (2026-04-25) | **Closed out** 2026-04-24 ("Ohio 4B parity closeout" in `concept_direction_analysis.md`); retained as cross-concept reference. Pinned `2975_15708` configs removed 2026-07-07 (stale-analysis cleanup); `cp_ohio_entities_gemma_it.yaml` + `base_oqi_reasoning_oh.yaml` remain |
| Orange (color vs fruit) pinned-feature lineage (`*_155_4973*.yaml`) | Debug-intervention validation against two hand-labeled 16k features | `/tmp/.../gemma3_1b_it_local_color_fruit_orange_155_4973_20260425_144518.ipynb` (2026-04-25) | **Superseded** by the structured `fs_l10_n5` selection lineage; pinned `155_4973`/`_4973` configs and their pinned-feature parity tests removed 2026-07-07 (stale-analysis cleanup) |
| Orange sign-aware structured selection (`*_fs_l10_n5*.yaml`: `signed_fs`, `s5`, `s5_any`, `s5_pos_noact_constrained`, `any_act_noconstrained` variants) | OQ-B: is mixed-sign cancellation the store direction's primary failure mode? Sweeps over `score_sign`, activation-function application, and layer constraints (layers ≥ 10, top-5) | `/tmp/.../gemma3_1b_it_local_color_fruit_orange_fs_l10_n5_20260427_144856.ipynb` (2026-04-27 14:56) | **Current** — active frontier at pause |
| Answer-basis follow-up (`*_answer_basis.yaml` + `analysis/answer_basis_analysis_experiment_set.yaml`) | Does projecting context into the answer-state basis recover answer-position parity for the store direction? | `/tmp/.../analysis/gemma3_1b_it_latent_dynamics_color_fruit_orange_fs_l10_n5_answer_basis_20260427_174607.ipynb` (2026-04-27 17:49 — latest run overall) | **Current** — answered: real geometric change but worsens store outcome; keep as diagnostic, not default |
| Cross-run analysis notebook (`analysis/concept_direction_analysis.ipynb` + launcher) | Regenerates the embed-vs-store comparison surface (probe separations, feature tables, Neuronpedia links) from reference-test reports + notebook outputs | `/tmp/.../analysis/concept_direction_analysis_answer_basis_20260427_144640.ipynb` (2026-04-27) | **Current** |
| Latent-dynamics notebook (`analysis/concept_direction_latent_dynamics_analysis.ipynb` + launcher) | Per-example latent-state geometry: projected vs selected store states, paired-rejection residuals, norm inflation | Answer-basis run above (2026-04-27 17:49) | **Current** |
| Intervention drift / graph parity tooling (`analysis/intervention_drift_analysis.{py,ipynb}`, `analysis/oqi_debug_session_ablation.py`, `intervention_graph_parity_testing.md`) | Fingerprints tensor/graph drift between notebook-debug and standalone circuit-tracer paths | Used through the 2026-04-24/25 parity closeout | **Current** (debug tooling, kept warm by unit tests) |
| Upstream CT manual parity (`analysis/extract_upstream_ct_semantic_reference.py`, `analysis/UPSTREAM_CT_PARITY_DEBUG.md`) | Manual three-way check: upstream circuit-tracer vs IT native vs IT analysis-op | 2026-03-17 (pre-consolidation) | **Stale** — superseded in practice by `tests/core/test_analysis_backend_parity.py`; archived to private notes 2026-07-07 |
| `concept_direction_analysis.md` | Living quality-analysis log (Ohio, orange, bat follow-ups; working hypotheses; recommended next steps) | 2026-04-27 (answer-basis section) | **Current — primary "where we left off" document** |
| `experiment_summary.md` | Synthesized findings + outstanding-question registry (OQ-A…OQ-I) for the V1–V12 waves | 2026-04-01 | **Current as historical record**; predates the Apr 19–27 gemma-3-1b-it work (dangling `archived_analysis/` links replaced with consolidation notes, 2026-07-07) |
| `wide_transcoder_support_plan.md` | 262k transcoder support/runtime plan for the *attribution* (circuit-tracer) path | 2026-05-16 | **Partially outdated** — its "no full-layer dashboards" framing is resolved by the dashboard pipeline; its attribution-runtime bottleneck analysis (~51 min 262k attribution runs) remains valid and unaddressed |

## Where We Left Off (2026-04-27)

Distilled latest insights, in rough order of importance:

1. **Embed and store select materially different feature subspaces everywhere.** Normalized
   direction cosine is near zero (0.02–0.24) across all model/concept combinations; the one
   apparent convergence (dog/cat Jaccard=1.0) was a data-integrity artifact, corrected in V9.
2. **The store direction's weakness is geometric, not an extraction artifact.** V11 direct
   projection showed pipeline effects are mostly feature-mediated (and can *invert* the raw
   direction signal); V12 context-enhanced extraction was numerically equivalent to standard
   answer-position extraction, closing the extraction-mode hypothesis.
3. **The parity target was deliberately reframed (2026-04-21).** Stop trying to maximize
   embed/store feature overlap; instead quantify and exploit the differing latent-space dynamics
   — the recurring pattern is "similar high-salience frontier, different normalized geometry,
   2–3 orders of magnitude store raw-norm inflation", i.e. a calibration problem in how store
   rows are weighted/aggregated, not a wholly different concept being discovered.
4. **Answer-basis projection is a real change but not the fix.** On the orange `fs_l10_n5`
   surface it halves the raw store norm yet drives store gap delta from `+3.91` to `−32.25` and
   Jaccard to 0, and the formulation algebraically cancels `context_enhanced_scale`. Verdict:
   keep as diagnostic; the next variant must change the formula (hybrid/interpolated or
   post-normalization blend), not just the scale.
5. **Sign-aware feature selection is implemented and checkpointed** (`f53f89a`: `signed_influence`
   score source, `score_sign` pos/neg/any, activation-function and layer constraints), with first
   sweeps run 2026-04-26/27. Systematic OQ-B conclusions (positive-only amplification vs
   negative-only ablation) were still pending when work paused.

## Pre-Merge Goal and Merge Sequencing (revised 2026-07-07)

The merge path for this work is: **`circuit-tracer-backend` merges to `main` first (after being made
independently green), then the dashboard-infrastructure branch rebases onto the new main** — the
dashboard branch is a direct linear descendant of `circuit-tracer-backend`, so the rebase is
mechanically trivial once the CT branch lands.

**Revised pre-merge experimental goal (deliberately narrow).** We have already demonstrated (Orange
sign-aware structured selection, 2026-04-26/27) that our sign-aware feature-set-based intervention
machinery can steer behavior for both the IT and non-IT gemma-3-1b variants. The goal before the
`circuit-tracer-backend` merge is to consolidate that into a **clean, reproducible demonstration** on
the simple `orange` (color-vs-fruit) and/or `bird_mammal_bat` example — in **both embed and store
contexts** — so the branch lands with clear experimental discoveries rather than a raw checkpoint. All
broader embed-vs-store-space experimentation (calibration surfaces, hybrid answer-basis formulations,
OQ-A/OQ-C, cross-concept sweeps) is **explicitly deferred** until after the circuit-tracer-backend and
then dashboard-change merges.

Remaining work for a clean `circuit-tracer-backend` commit:

1. **Demonstration runs**: rerun the sign-aware `fs_l10_n5` orange lineage (and optionally a new
   `gemma3_1b_it_local_bird_mammal_bat.yaml` — recommended 2026-04 but never created) end-to-end on
   current heads for both embed and store contexts on gemma-3-1b (IT + PT), capturing SUMMARY_RECORD
   artifacts and a short written readout of the steering effect sizes. Semantic validation of the
   selected features now uses the local 262k dashboards (layers 9/23 live; build more layers as
   needed at ~40-45 min/layer).
2. **Harden the latent-dynamics notebooks for inclusion in the merge**
   (`analysis/concept_direction_latent_dynamics_analysis.ipynb` + launcher, and the cross-run
   `concept_direction_analysis.ipynb`): parameter/default cleanup, deterministic seeds, a CI-safe
   smoke path (papermill execution with a tiny config), and a docs pass so reviewers can run them.
3. **Green the branch**: full interpretune suite + coverage harness on `circuit-tracer-backend`
   without `--allow-failures` (the branch's last validation was failure-tolerant and is ~2.5 months
   stale), absorbing any dependency drift.
4. **Distill the living analysis docs** (`concept_direction_analysis.md`, `experiment_summary.md`):
   mark the deferred follow-up threads as post-merge work (the queued items below), so the
   at-merge state reads as "concluded phase + named next directions".

### 7c Amendments (2026-07-11 — folding back the original `ct_backend_implementation_plan.md` demo/doc objectives)

The original CT-backend plan's IG-3.2 (document model-level intervention APIs), IG-5 (demo
notebooks + docs + notebook tests), and IG-6 (concept-direction demo) objectives are re-scoped as
follows for the 7c execution ahead:

1. **Ground-truth reset (post-merges, pre-#220)**: many notes in `experiment_summary.md` and
   `concept_direction_analysis.md` (some propagated into this file) are stale experimental
   directions. After this workstream and the dashboard wave (which we will not delay for this), the
   current experimental status must be reset/reverified from ground truth and the experimental
   priorities distilled prior to and as part of implementing
   [#220](https://github.com/speediedan/interpretune/issues/220) (issue updated 2026-07-11 with
   this prerequisite).
2. **Intervention-capabilities overview**: distill the current embed- and store-based intervention
   capabilities (sign-aware feature selection, layer filters, amplification/scaling and basis
   configuration controls, store-based intervention features) into a high-level overview document
   in `tests/nb_experiments/` (`intervention_capabilities_overview.md`), refreshing
   `docs/interpretune_intervention_apis.md` where stale.
3. **Archive the RTE cross-backend demo**: the RTE-focused
   `src/it_examples/notebooks/dev/circuit_tracer_examples/ct_cross_backend_demo.ipynb` is archived
   (archived to maintainer-side private notes) — the ambitious
   RTE-based research resumes under #220. Cross-backend *mixing* is deferred to a subsequent
   workstream to keep the CT-backend merge scope narrow and avoid delaying the dashboard PRs.
4. **Refocused steering demo (replaces the cross-backend demo, renamed accordingly)**: a
   single-backend, proven-example demo (`orange` color-vs-fruit) of store-based AND embed-based
   concept-direction-mediated, sign-aware, multi-feature steering on `gemma-3-1b-it`, using the
   local 262k **Monology** dashboards and locally generated feature explanations for the selected
   features — proving the full current pipeline end-to-end (dashboards → explanations → selection
   → steering with verified activation effects).
   **✅ DONE (2026-07-15)**: `ct_concept_steering_demo.ipynb` activated (Phases 1-4: session,
   store-path steering with target-gap assertion, dashboard refs + `ensure_local_feature_explanations`
   coverage, embed-path `model_fwd_intervention` steering with store-vs-embed direction cosine);
   papermill gate removed — `test_ct_concept_steering_notebook` PASSED on the 4090 against the
   imported 262k Monology substrate (explanation generation disabled under test).
   **⤴ REVISED (2026-07-16)**: the activated gemma-3-1b-it notebook was rebuilt on **gemma-2-2b +
   `gemmascope-transcoder-16k`** with a `DASHBOARD_MODE` parameter (`"public"` neuronpedia.org /
   `"local"` dev-webapp links) so reviewers can inspect the referenced feature dashboards publicly;
   the gemma-3-1b-it + local-262k + locally-generated-explanations flow returns as a papermill
   parameter set of the same notebook (`GENERATE_MISSING_LOCAL_EXPLANATIONS` capability retained for
   maintainers/local-DB developers). Motivation: the first fully executed gemma-3-1b-it run surfaced
   chat-answer-position token-variant pathologies (see "Token-Variant (▁-prefix) Effects" below) and
   its 262k dashboards are local-only, making the demo unreviewable without a local substrate.
   **Accuracy correction found during the rebuild**: the activated notebook's "store path" label was
   wrong — with no `concept_cache_key`/store rows wired into `intervention_from_concept`, BOTH
   phases derive the embed-basis direction (pipeline-vs-direct cosine +1.0000), so the demo actually
   contrasts *feature-mediated* vs *direct-hook* steering of the same direction; the rebuilt
   notebook relabels the phases accordingly and defers true store-basis directions to the
   nb_experiments harness (`_compute_store_direction`). The interim frozen `*_gemma2_ref.ipynb`
   copies were removed 2026-07-17 once the updated demos were validated (recoverable from git
   history: `475006f` / `48f371b`).
5. **`ct_analysis_backend_demo.ipynb` refresh**: switch to `gemma-3-1b-it` + the local 262k
   Monology source set; demonstrate the core concept-direction op pipeline AND the intervention
   APIs; remove references to the removed RTE concept-direction demo (point at the new steering
   demo instead). Between the two notebooks, cover analysis-op usage, native+hub op composition,
   embed- and store-based steering, the feature-selection API, and local feature explanations —
   while minimizing duplication and keeping each digestible.
   **✅ DONE (2026-07-15)**: retargeted to the `gemma3.rte_demo.circuit_tracer_w_neuronpedia`
   registry entry (262k transcoder set; chat-template-rendered prompt; gemma-3 key tokens;
   `neuronpedia_model="gemma-3-1b-it"`); RTE-demo references replaced with a pointer at the
   steering demo; `test_ct_analysis_backend_notebook[ct_analysis_backend_nnsight]` PASSED on
   the 4090. Intervention-API depth intentionally lives in `ct_concept_steering_demo.ipynb`
   (this demo keeps the composite-pipeline focus).
   **⤴ REVISED (2026-07-16)**: the gemma-3-1b-it retarget was rolled back — the executed run's
   steered top-5 was unintelligible (` `, `Con`, `▁▁▁▁`) with Austin at ~1e-10 pre-intervention at
   the chat answer position, versus the fully sensible gemma-2-2b behavior (▁Austin 42%→64.5%, gap
   +2.625→+5.375, top-3 features exactly matching the upstream-parity snapshot in
   `tests/upstream_parity/UPSTREAM_CT_PARITY_DEBUG.md`). The demo is re-anchored on the last
   gemma-2-2b version with a `dashboard_mode` public/local parameter; frozen reference copies
   (`ct_analysis_backend_demo_gemma2_ref.ipynb`, `ct_cross_backend_demo_gemma2_ref.ipynb` + tests)
   were restored for upstream-parity comparison while enhanced variants are bootstrapped.
6. **Full Monology 262k production dashboard run**: end-to-end all-layer dashboards for
   `gemma-3-1b-it` × `gemmascope-2-transcoder-262k` over the Monology corpus at the standard
   **24,576-prompt** scale (vs the 2,490-prompt benchmark shape) — both the semantic substrate for
   the demos above and a "production" validation artifact for the dashboard-scaling PRs (run doc
   maintained in maintainer-side private notes).

Deferred (post-merge) experimental threads, retained from the 2026-04 queue: pre-normalization store
norm/weight/paired-rejection comparison across Ohio/orange/bat with the `context_enhanced_scale`
1.0-vs-10.0 pair; narrowed `specific_features` follow-up on the `fs_l10_n5` shared later-layer
survivors (dashboard-validated); the two anti-target orange Color exemplars (`blue`, `yellow`); a
non-canceling hybrid answer-basis projection rule; OQ-A (continuation-style PT store prompts) and
OQ-C (scale-factor sweep aggregation). These are secondary to the auto-pruning direction below.

## Token-Variant (▁-prefix) Effects in Steering Results (2026-07-16)

A recurring pattern across the orange (color-vs-fruit) gemma-3-1b-it runs (reference artifact:
`/tmp/it_concept_direction_experiments/gemma3_1b_it_local_color_fruit_orange_fs_l10_n5_20260427_144856.ipynb`)
and the rolled-back gemma-3-1b-it `ct_analysis_backend_demo` run:

- **Why the "Original Top 5 Tokens" are never space-prefixed**: the evaluation prompts are
  chat-template rendered, so the measured answer position immediately follows `<start_of_turn>model\n`.
  After a newline, SentencePiece produces the *bare* token variant (no leading `▁`), so a
  well-calibrated model concentrates next-token mass on bare variants there (`Color` 99.957%,
  `Fruit`, `Colour`, `COLOR`, `color` — all bare).
- **Why the most-shifted tokens are consistently ▁-prefixed** (`▁Fruit` vs `Fruit`): the
  interventions act through transcoder-feature decoder directions (and concept directions) whose
  geometry reflects pretraining-corpus statistics, where concept words occur overwhelmingly
  *mid-sentence* as space-prefixed variants. Amplifying those features therefore boosts the
  ▁-variant unembedding directions, overriding the positional (start-of-line) context the model
  otherwise conditions on — e.g. the embed-path intervention moved `▁Fruit` from 2.24e-13 to 98.9%
  while the positionally appropriate bare `Fruit` only reached 1.1%.
- **The store-based intervention shows this most strongly** (its post-intervention top-5 was
  *entirely* ▁-prefixed: `▁Color`/`▁color` 49.5%/49.5%, `▁fruit`, `▁COLOR`, `▁Fruit`). Notably this
  is **not** explained by store-prompt formatting: the store cache's prompt-alignment debug shows
  the cached answer positions also hold the *bare* variant (`'Fruit'` id 95269 after
  `<start_of_turn>model\n`), i.e. the store direction was aggregated at bare-variant positions yet
  still steers hardest toward ▁-variants.

**Outstanding experimental question (OQ-J, added 2026-07-16)**: why does the *store-based* path
exhibit the strongest ▁-variant bias when its answer-position activations were cached at
bare-variant positions? Candidate factors to isolate: (a) store-selected features sit in earlier
layers (11–13) than embed-selected ones (~25), leaving more downstream propagation through
corpus-statistics-shaped features; (b) the store top features carry negative signs — de-amplification
side effects may suppress the positional "start-of-line" circuitry rather than promote the concept;
(c) direction aggregation across store rows may cancel positional components while preserving
concept components. A controlled follow-up should compare variant-mass distributions when steering
with (i) embed vs store directions, (ii) positive-only vs negative-only feature sets, and (iii)
early-layer vs late-layer selections on the same prompt.

Current mitigation (measurement-side, not a fix): the steering demo and harness select the
plain-vs-space-prefixed answer variant by pre-intervention logit (`best_variant_id()`-style
selection) and report both variants in key-token tables.

**OQ-K (added 2026-07-17): input/output semantic decoupling of attribution-selected steering
features.** Steered-feature explanations are consistently NOT concept-related (gemma-3 262k:
"printers"/"formal structured text"/…; gemma-2 16k: only the top feature fruit-related) even though
steering works — conclusion (see the 2026-07-17 section of `concept_direction_analysis.md`):
max-act explanations describe the *input* side (when a feature fires) while `signed_influence`
selection optimizes the *output* side (decoder projection onto the target logit diff), and
answer-position attribution biases selection toward late-layer "say-X-next"/redundancy-suppression
machinery (exemplar: gemma-2 L25/16131 fires on fruit contexts yet its top negative logits are all
`Fruit` variants — a repetition-suppression motif the sign-aware selector correctly suppresses).
Follow-up: quantify the decoupling per OQ-K; the steering demo's Phase 5 (user-curated features)
provides the input-semantics-selected comparison arm, and Phase 6 (added 2026-07-18) demonstrates
the decoupling mechanics directly via graph hydration (per-feature input concept-position
activation shares vs signed decoder projections) with a decoder-space UMAP visualization; the
full reframe plan — including the 16k-vs-262k decoder-mass-splitting quantification (top-1000
share 17.5% vs 1.53% at L19; ~13-16x per-rank dilution; shape scale-invariant) and 262k runtime
feasibility constraints (≈31.4 GB bf16 all layers vs 24 GiB VRAM; offload-only, minutes-scale
attribution → off-demo) — lives in the "Steering-Demo Feature-Semantics" section of
`concept_direction_analysis.md`.
**Finding 4 (2026-07-18, quantified & revised 2026-07-19)**: attribution-target selection favors
non-concept features exactly when the probed surface token is *shared by both contrast categories*
(a polysemous probe like `orange`), because both categories' detectors then fire on the same token
instances with opposing target-direction projections and mutually cancel — NOT because the
categories' context words overlap more (measured near-parity: bilateral fractions 0.173 vs 0.117;
pole-word cancellation 0.920 vs 0.983). The token-disjoint capitals−states scenario (every
concept-bearing token belongs to exactly one category; probe cue `Dallas` unambiguous) retains a
coherent concept-feature channel at the answer position (cancellation 0.865, net/gross ≈ 13.5%,
led by 23/12237 "locations") and selects 8/9 input-aligned features. Exemplar opposition at the
`orange` token: 25/9975 +17.8 toward Fruit vs suppressor-motif 25/16131 −14.1 (full definitions,
table, and the committed probe script reference in `concept_direction_analysis.md` Finding 4).
**J-space (Jacobian-lens) probes** for these questions are designed in the same analysis section
and tracked in [interpretune#225](https://github.com/speediedan/interpretune/issues/225)
(workspace-level polysemy-cancellation readout; per-feature J-space signatures as the principled
generalization of the 1-D output projections; workspace-flip vs logit-flip ordering;
`concept_direction(..., basis="jlens")`).
**Correction (2026-07-17, later)**: the gemma-3 arm of this observation was primarily an artifact —
the demo's `NEURONPEDIA_SOURCE_SET` said 262k while the registry's runtime transcoders are the
16k-width gemma-scope-2 set, so gemma-3 links/explanations were resolved in the wrong feature space
(caught by Phase 5's out-of-bounds curated index). Preset/tests corrected to the local
`gemmascope-2-transcoder-16k` set; the OQ-K decoupling conclusions rest on the correctly-matched
gemma-2 arm. Follow-up hardening: validate feature indices against runtime transcoder width in the
ref-building path.

**Local-service outstanding issues (2026-07-17, tracked for Wave 2 / pre-PR sweeps):**
1. `search-explanations` returns nothing locally: the flow is OpenAI-embedding-backed
   (`text-embedding-3-large` query embeddings + pgvector over `Explanation.embedding`) — the local
   webapp lacks an OpenAI key AND all 68 locally imported/generated gemma-3-1b-it explanations have
   NULL `embedding`. Fix options: embedding backfill at local generation/import time, or an NP-side
   non-embedding (ILIKE/trigram) fallback when embeddings are absent.
2. `search-topk-by-token` page: custom source sets didn't populate because the importer never sets
   `Source.inferenceEnabled` (fixed live in the local DB 2026-07-17 — 147 sources flagged; the
   importer-side fix is tracked with the NP-utils residuals); actual searches additionally require
   the local inference server to be running.
3. `<0xF0>` byte-fallback tokens still top-activate several 262k features (not 16k) and `<unusedN>`
   tokens remain prevalent in both local sets vs public dashboards — candidates/hypotheses tracked
   in maintainer-side private notes (2026-07-17 dashboard-quality addendum).

## Instruction-Tuned vs Base Model Substrate Guidance (2026-07-16)

Distilled from `concept_direction/README.md`, `concept_direction/experiment_summary.md`, and the
prompt-parity migration work:

1. **Match the transcoder training substrate to the run model.** Gemma Scope (16k) transcoders were
   trained on **base** gemma-2 models; running them under gemma-2-2b-it (or other IT variants)
   degrades attribution and especially *intervention* reliability — feature decoder directions no
   longer align with the IT model's computation. Use base gemma-2-2b with the 16k set (the public
   demo configuration), or an IT model only with transcoders trained on that IT model (e.g.
   gemma-scope-2 `gemmascope-2-transcoder-262k` for `gemma-3-1b-it`).
2. **Properly formed prompts for IT models are necessary but not sufficient.** Chat-template
   rendering fixes gross miscalibration (an IT model given a raw completion prompt behaves
   conversationally — e.g. `The` as top-1 rather than the answer token), but does not resolve the
   token-variant and feature-alignment effects above, and can make simple completion-style examples
   degenerate (with the properly rendered chat prompt, `Austin` is already top-1 at 0.970 for the
   capitals/states example on gemma-3-1b-it, leaving no useful steering headroom).
3. **Prefer base-model completion prompts for pedagogical steering demos**; reserve IT-model
   substrates for experiments that specifically target IT behavior and use IT-trained transcoders.

## Primary Post-Merge Direction: Attribution-Graph Auto-Pruning & Tuning

The primary new experimentation direction after the circuit-tracer-backend and dashboard merges is an
**objective-driven attribution-graph auto-pruning and tuning capability** (two phases):

- **Phase 1 — auto-pruned, intervention-bearing graphs.** Starting from a naively-pruned attribution
  graph and a user-provided target (a token embedding, a logit-difference direction, or any
  residual-stream-space tensor — the existing `AttributionTargets` abstraction), iteratively co-adjust
  graph pruning and a small set of node-activation adjustments to maximize similarity between the
  target and the latent state at the output node (`unembed.hook_in`), subject to interpretability
  regularization: minimize node/edge count, minimize relative activation-adjustment magnitude, and
  prefer shorter/sparser circuits without oversimplifying. The output is a pruned, visually manageable
  graph plus an `InterventionDict` change set that reproduces the desired behavioral change. Our
  intervention-based activation/logit expectation-value validation tests already show that under the
  appropriate linearization conditions (zero_softcap, frozen non-linearities, pre-activation capture)
  node-adjustment effects propagate predictably — so the optimization can search the linearized regime
  cheaply, interleaving real forward passes / graph reconstructions to verify progress. Staged plan:
  (1) candidate objective functions/constraints; (2) linearized-regime search algorithms that avoid a
  forward pass per candidate; (3) toy-model validation; (4) gemma-3-1b-it baseline efficacy; (5)
  iterate (efficiency, custom kernels, user-constrained search).
- **Phase 2 — graph-constrained tuning.** Use the pruned graphs + intervention sets as training
  signal: fine-tune the model so it produces the desired behavior *subject to interpretable graph
  constraints* (richer gradient signal than output-token loss alone), aiming for simpler, more
  coherent, more human-alignable internal "world models" with retained mechanistic fidelity; RTE is
  the first tuning target. Extensions: self-reflective/self-correcting variants using world-model
  coherence/surprise-minimization proxies.

**Repo-preparation plan (what nb_experiments needs to support this, post-merges):**

1. The demonstration + hardening items above land first (they provide the validated intervention
   machinery — sign-aware selection, `InterventionSpec`/`InterventionDict`, linearization contexts —
   that Phase 1 optimizes over).
2. Keep the graph/intervention parity suites (`test_analysis_backend_parity.py`, intervention drift
   tooling) green through the merges — they are the correctness harness the linearized-regime search
   will rely on; optionally port the retired pinned-feature adjacency-trace check to the `fs_l10_n5`
   lineage.
3. Add a `graph_pruning/` experiment family skeleton beside `concept_direction/` (config schema
   reusing the layered-YAML harness; toy-model fixture first per staged-plan step 3).
4. Wire dashboard-backed semantic validation into the loop (feature labels for surviving graph nodes
   via the local 262k dashboards) so pruned-graph interpretability claims are inspectable.

Tracking issue: [speediedan/interpretune#220](https://github.com/speediedan/interpretune/issues/220)
(filed 2026-07-07 from this formulation).

## Stale Analysis Slated for Removal (confirmed — executed 2026-07-07; items 3–4 archived rather than removed)

In-repo candidates:

1. `concept_direction/configs/gemma3_1b_it_local_color_fruit_orange_155_4973.yaml`,
   `..._155_4973_streaming.yaml`, `..._4973.yaml`, and
   `analysis/configs/gemma3_1b_it_latent_dynamics_color_fruit_orange_155_4973.yaml` — the
   hand-picked 16k-feature lineage is fully superseded by structured `fs_l10_n5` selection.
   **Executed 2026-07-07 (removed).** The pinned-feature parity tests that consumed these configs
   (`test_analysis_backend_parity_gemma3_1b_it_concept_direction_paths[orange_155_4973]`,
   `..._feature_intervention_wrapper[feature_25_4973]`, `..._selected_feature_adjacency_trace[feature_25_4973]`)
   were retired with them; sign-aware wrapper coverage continues via the `fs_l10_n5_s5_any` variant.
   Follow-up (optional): port the selected-feature adjacency-trace check to the `fs_l10_n5` lineage.
2. `concept_direction/configs/gemma3_4b_it_local_oqi_reasoning_oh_2975_15708.yaml` and
   `..._streaming.yaml` — Ohio 4B parity was closed out 2026-04-24; the config is a pinned
   debug surface for a resolved investigation. **Executed 2026-07-07 (removed).** The
   `test_analysis_backend_parity_gemma3_4b_it_ohio_reference_graph_sanity` test was retired with it;
   `analysis/oqi_debug_session_ablation.py` (and its unit test) now default to
   `base_oqi_reasoning_oh.yaml`.
3. `analysis/extract_upstream_ct_semantic_reference.py` + `analysis/UPSTREAM_CT_PARITY_DEBUG.md`
   — 2026-03-17 manual upstream-CT parity workflow; superseded by the automated
   `tests/core/test_analysis_backend_parity.py` suite. **Archived to private notes 2026-07-07**
   (not removed outright).
4. `concept_direction/prompt_parity_migration_guide.md` — the migration it documents is complete
   (AQ-1: render modes token-identical for Gemma 3 IT; `gemma_dataclass` de-prioritised).
   **Archived to private notes 2026-07-07** (not removed outright).
5. Dangling references (fix rather than remove): `experiment_summary.md` links to
   `archived_analysis/V9–V12_experimental_summary.md` and `README.md` links to
   `experimental_summaries.md`, none of which survived the 2026-04-25 consolidation commit.
   **Fixed 2026-07-07** — dangling links replaced with plain-text consolidation notes.

Generated-artifact candidates under `/tmp/it_concept_direction_experiments/` (disposable by
policy; 83 files, 2026-04-22 → 2026-04-27): everything predating the 2026-04-27 afternoon wave
is superseded — the `155_4973` runs and 4B Ohio run (04-25), the early `signed_fs` runs
(04-25 evening), and the intermediate 04-26 `fs_l10_n5(_s5/_any)` iterations. The reference set
to keep is the 2026-04-27 `fs_l10_n5` reruns (14:10/14:56), the sign-variant sweep notebooks
(11:05–12:12), and the answer-basis wave (14:46–17:49).

## Dashboard Capability That Unblocks the Next Iteration

The blocker recorded in `experiment_summary.md` (DQ-2: "no neuronpedia dashboards exist for 1B
transcoders, blocking feature inspection") is now resolved locally:

- `interpretune.utils.neuronpedia_dashboard_pipeline` (+ `scripts/launch_neuronpedia_dashboard_pipeline.py`)
  provides a file-backed, resumable generate → convert → import pipeline into a local Neuronpedia
  DB, with YAML/`EXTENDS` configs, per-layer locks, and stall/kill diagnostics. See
  `docs/neuronpedia_dashboard_pipeline.md`.
- Target source set: `gemmascope-2-transcoder-262k-rte` for `gemma-3-1b-it` — full 262k-width
  transcoder dashboards built over the task-relevant pretokenized RTE/BoolQ chat-template prompt
  corpus (`gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts`), i.e. feature
  dashboards grounded in the same task framing as the concept-direction experiments.
- Throughput: the `columnar_gpu` route reaches ~6.6k features/min end-to-end (generation +
  local DB import, 2048×128 RTE config; 2026-07-06 benchmark package) — roughly **40–45 minutes
  per 262k-width layer**, about a 10× end-to-end improvement over the legacy route (~650
  features/min ≈ 6.7 h/layer), with 100% per-feature activation-row parity against the preserved
  baseline. Cross-layer generation/import overlap (2026-07-05) further hides the import wall.
- Layers 9 and 23 of `gemmascope-2-transcoder-262k-rte` are already generated, imported, and
  webapp-validated in the local DB (2026-07-05 full-layer builds: 524,288 features in 91.2 min
  total), alongside the earlier `gemmascope-2-transcoder-16k` set (all 26 layers).
- This makes the pending `specific_features` / sign-aware semantic-validation steps above
  practical for the first time on gemma-3-1b-it, and also supersedes the
  `wide_transcoder_support_plan.md` §1 dashboard-availability rationale (its attribution-runtime
  plan remains open, tracked separately).
