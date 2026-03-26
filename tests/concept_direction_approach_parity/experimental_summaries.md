# Concept Direction Experimental Summaries

This document ties together three things for the notebook harness work in this directory:

1. The evaluated hypotheses in [HYPOTHESES.md](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/HYPOTHESES.md)
2. The six outstanding questions from the V3 anomaly review
3. The concrete notebook parameterizations we want to run, compare, and log

It is intended to be the operational companion to [HYPOTHESES.md](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/HYPOTHESES.md):

- `HYPOTHESES.md` defines the broader explanatory claims we want to test.
- This file maps those claims to concrete experiment configs, expected observations, and analysis notes.

## Outstanding Questions

| ID | Outstanding question | Related hypotheses | Primary configs | Current evidence | What would count as progress? |
|---|---|---|---|---|---|
| OQ1 | Why does the Gemma 2 base capitals/states embed direction selectively help Austin by suppressing Dallas and Texas instead of behaving like broad disruption? | H1, H2, H6 | `gemma2_base_capitals_states`, `gemma3_pt_capitals_states` | V3 shows base capitals/states embed Δ = +2.97 with Austin roughly stable while Dallas and Texas drop sharply. | Show whether lower scales or sign-aware feature subsets preserve the selective effect, and whether Gemma 3 PT reproduces the same pattern. |
| OQ2 | Why does the Gemma 2 base dog/cat embed direction steer toward cat even though the concept direction is dogs minus cats? | H1, H2, H6 | `gemma2_base_dog_cat`, `gemma3_pt_dog_cat` | V3 shows base dog/cat embed Δ = -4.34 and broad suppression where cat falls less than dog. | Identify whether this is driven by scale regime, mixed-sign feature selection, or concept-specific token geometry. |
| OQ3 | Why does the IT capitals/states intervention look like broad suppression, and why does that suppression favor Dallas over Austin? | H1, H3, H7 | `gemma2_it_capitals_states`, `gemma3_it_capitals_states` | V3 shows IT capitals/states embed Δ = -2.06 with all key logits dropping by 5 to 20 points. | Show whether model-matched Gemma 3 IT transcoders reduce suppression, and whether scale or sign-aware interventions recover targeted steering. |
| OQ4 | Is the store-based direction actually more semantically aligned, or is it simply selecting more influential and therefore more disruptive features? | H4, H5 | all eight baseline configs; compare embed vs store in each | V3 shows store often produces larger magnitude deltas with low cosine overlap relative to embed. | Quantify whether store features have higher activations, stronger influence, denser causal mass, or cleaner sign structure than embed features. |
| OQ5 | How much of the observed behavior is caused by sign-agnostic ranking from `abs_()` versus the underlying semantics of the selected features? | H2, H5 | all baseline configs plus sign-aware reruns | V3 negated-direction tests are identical because feature selection is sign-agnostic. | Show that positive-only amplification or negative-only ablation changes the gap more cleanly than the mixed-sign intervention. |
| OQ6 | Which effects are Gemma 2 specific and which generalize across model family, tokenizer, and transcoder regime? | H3, H7, H6 | Gemma 2 vs Gemma 3 matched pairs | We only have V3 Gemma 2 evidence so far. | Reproduce or falsify the same anomaly classes on Gemma 3 PT and IT with the unified harness. |

## Experiment Configurations

The unified harness is driven by flat YAML files in [configs](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/configs).
Executed notebooks are written to [generated_experiments](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/generated_experiments).

### Baseline matrix

| Config | Model family | Variant | Concept pair | Purpose |
|---|---|---|---|---|
| `gemma2_base_capitals_states.yaml` | Gemma 2 | base | capitals_states | Baseline reproduction of the selective Austin-helping case |
| `gemma2_it_capitals_states.yaml` | Gemma 2 | it | capitals_states | Baseline reproduction of the IT suppression case |
| `gemma3_pt_capitals_states.yaml` | Gemma 3 | pt | capitals_states | Gemma 3 PT comparison to Gemma 2 base |
| `gemma3_it_capitals_states.yaml` | Gemma 3 | it | capitals_states | Gemma 3 IT comparison to Gemma 2 IT |
| `gemma2_base_dog_cat.yaml` | Gemma 2 | base | dog_cat | Base anomaly where cat is favored |
| `gemma2_it_dog_cat.yaml` | Gemma 2 | it | dog_cat | IT dog/cat baseline for embed/store disagreement |
| `gemma3_pt_dog_cat.yaml` | Gemma 3 | pt | dog_cat | Gemma 3 PT comparison for concept-specific behavior |
| `gemma3_it_dog_cat.yaml` | Gemma 3 | it | dog_cat | Gemma 3 IT comparison for model-matched transcoders |

## Mapping Questions To Experiments

| Question | First-pass experiment set | Key notebook phases to inspect |
|---|---|---|
| OQ1 | `gemma2_base_capitals_states`, `gemma3_pt_capitals_states` | Phase 5 summary, Phase 6 scale sweep, Phase 8 sign-aware selection, Phase 9 direction probes |
| OQ2 | `gemma2_base_dog_cat`, `gemma3_pt_dog_cat` | Phase 5 summary, Phase 6 scale sweep, Phase 8 sign-aware selection, Phase 9 direction probes |
| OQ3 | `gemma2_it_capitals_states`, `gemma3_it_capitals_states` | Phase 5 summary, Phase 6 scale sweep, Phase 8 sign-aware selection |
| OQ4 | all baseline configs | Phase 4 store direction, Phase 5 summary, Phase 7 progressive ablation |
| OQ5 | all baseline configs | Phase 7 progressive ablation, Phase 8 sign-aware selection |
| OQ6 | Gemma 2 vs Gemma 3 matched pairs | Phase 5 summary and aggregate comparison across executed notebooks |

## Observation Log

| Date | Config | Question(s) | Key observations | Analysis note | Follow-up |
|---|---|---|---|---|---|
| 2026-03-23 | `gemma2_base_capitals_states_20260323_184412.ipynb` | OQ1, OQ4, OQ5 | Embed intervention improves the Austin minus Dallas gap by `+2.97`, with `2.0x` still helping Austin (`+0.36` logit) while suppressing Texas by `-2.53`; `50x` becomes destructive. Store direction is nearly gap-neutral (`+0.005`) despite decent probe separation. Top-100 embed ablation sharply lowers Austin (`-3.88`) and raises Texas (`+1.94`). | This still looks like a narrow, feature-mediated Austin rescue rather than a broad “capitalness” direction. The ablation result suggests the top embed features are causally necessary for the selective effect, while the store direction remains more semantically aligned than causally potent. | Rerun with explicit prompt-render mode control and sign-aware subsets to test whether positive-only features preserve the selective Austin gain without the Texas collapse. |
| 2026-03-23 | `gemma2_it_capitals_states_20260323_184939.ipynb` | OQ3, OQ4, OQ5 | Both embed and store directions worsen the Austin minus Dallas gap (`-2.06` and `-2.22`), even though store-side prompt predictions are `8/8`. At `2.0x`, embed suppresses Austin, Dallas, and Texas together; `50x` is catastrophically suppressive. Top-100 ablation barely hurts Austin (`-0.19`) but strongly boosts Texas (`+7.68`). Positive-only features at `10x` boost Austin by `+1.30` but boost Texas even more by `+5.26`. | The IT behavior is not just generic collapse; it is a broad suppression regime with residual feature mass that still prefers state evidence when amplified. The prompt-conditioned store direction keeps the task intact but does not recover the desired semantic split. | Compare tokenizer-based chat rendering against Gemma dataclass rendering to see whether prompt formatting, not only direction geometry, is contributing to the Dallas-favoring suppression pattern. |
| 2026-03-23 | `gemma3_pt_capitals_states_20260323_190458.ipynb` | OQ1, OQ4, OQ6 | Gemma 3 PT reproduces the base-model anomaly class: embed direction improves the target gap by `+6.06`, store direction is weakly positive (`+0.375`), and store prompt predictions remain `0/8`. The `2.0x` sweep already suppresses Austin, Dallas, and Texas, while `50x` fully destroys the answer tokens. Top-100 ablation raises Texas to rank `0`. Probe separation remains positive for both embed (`0.498`) and store (`0.152`). | The cross-family pattern is now clearer: base/PT models can show useful concept-direction geometry without having the prompt-level behavior needed for correct classification. That makes the gap improvements less trustworthy as evidence of genuinely task-aligned steering. | Check whether upstream Gemma 3 stabilizers such as `zero_softcap()` change the prompt-level baseline and whether Neuronpedia model routing should use `gemma-3-1b` for better feature inspection consistency. |
| 2026-03-23 | `gemma3_it_capitals_states_20260323_190849.ipynb` | OQ3, OQ5, OQ6 | Gemma 3 IT keeps `8/8` prediction accuracy, but both embed and store directions slightly worsen the target gap (`-0.5` and `-0.5625`). At `2.0x` and `50x`, Austin and Dallas both drop while generic label tokens such as `▁City` and `▁city` rise sharply. Top-100 ablation is mild, and positive-only feature amplification is nearly inert. Feature overlap between embed and store is high (`Jaccard = 0.667`), but both directions have weak store-side separation (`0.117`). | Compared with Gemma 2 IT, Gemma 3 IT looks less violently suppressive but still fails to produce a usable capitals-over-states intervention. High feature overlap with low steering efficacy suggests the issue is sign structure or prompt-conditioned activation context rather than missing feature agreement. | Run the Gemma dataclass chat-render variant and compare the rendered prompts directly; if behavior is unchanged, the next debugging step should focus on Gemma 3 backend configuration and upstream-style attribution stabilization rather than prompt wrapping alone. |

## First-Wave Capitals/States Readout

### Stable patterns across the first four runs

- Base/PT models show stronger embed-direction gap improvements than IT models, but both base/PT runs also fail the prompt-conditioned prediction sanity check (`0/8` for the store-direction prompt set).
- IT models preserve prompt-conditioned prediction accuracy (`8/8`) yet concept-direction interventions mostly reduce or destabilize the intended target gap.
- Progressive ablation repeatedly pushes mass toward state tokens, which argues that the top-ranked features are not simply “generic capitals” features.
- Direction probes show positive group separation in every run, so simple geometric separability is not sufficient to predict successful intervention behavior.

### Immediate implications for the updated harness

- Prompt construction needs to be part of the experiment surface, not a hidden notebook detail.
- The target token pair needs to be configuration-owned so the notebook can stop silently deriving evaluation targets from concept-pair names.
- Direct top-level op wrappers are a better fit for this harness than dispatcher lookups because the notebook is acting as a user-facing experiment script rather than a dispatcher testbed.

## Prompt Render Comparison

### 2026-03-23 rerun readout

| Pair | Tokenizer notebook | Dataclass notebook | Comparison readout |
|---|---|---|---|
| Gemma 2 IT capitals/states | `gemma2_it_capitals_states_20260323_162514.ipynb` | `gemma2_it_capitals_states_gemma_dataclass_20260323_163248.ipynb` | Both runs preserve `8/8` prediction accuracy, but the dataclass render changes the intervention regime materially: embed gap delta moves from `+0.107` to `+1.089`, store gap delta moves from `-0.118` to `+0.173`, cosine similarity drops from `0.161` to `0.035`, and feature overlap drops from `0.429` to `0.176`. |
| Gemma 3 IT capitals/states | `gemma3_it_capitals_states_20260323_163935.ipynb` | `gemma3_it_capitals_states_gemma_dataclass_20260323_165023.ipynb` | Both runs preserve `8/8` prediction accuracy and keep the same broad conclusion that IT intervention hurts the target gap, but the dataclass render makes the negative effect larger: embed gap delta moves from `-0.500` to `-0.969`, store gap delta moves from `-0.562` to `-1.000`, cosine similarity rises from `0.165` to `0.224`, and feature overlap stays fixed at `0.667`. |

### Preliminary interpretation

- Prompt wrapping is not a cosmetic detail for these IT runs; it changes both the intervention magnitude and the selected feature pool.
- Gemma 2 IT is especially sensitive to render choice. The dataclass path produces a more favorable capitals-over-states result than the tokenizer path, even though both keep task accuracy intact.
- Gemma 3 IT is less sensitive in kind but still sensitive in degree. Switching to the dataclass render does not recover a useful intervention; it amplifies the same negative trend.
- The original Gemma 3 dataclass failure was a prompt-prefix compatibility issue, not a direction-geometry issue: the dataclass render omitted the tokenizer BOS prefix expected by the Gemma 3 NNsight and circuit-tracer path. The harness now normalizes that prefix before attribution runs.

## Analysis Conventions

- Record executed notebook paths from [generated_experiments](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/generated_experiments) so results can be traced back to a stamped artifact.
- When possible, cite the phase number in the executed notebook rather than paraphrasing from memory.
- Keep “observation” distinct from “analysis”:
  - Observation: what the notebook printed or plotted
  - Analysis: what that implies for the question or hypothesis
- Prefer paired comparisons:
  - Gemma 2 base vs Gemma 3 PT
  - Gemma 2 IT vs Gemma 3 IT
  - Embed vs store within the same config
  - Mixed-sign vs sign-aware interventions within the same config

## Current Starting Point

- [V3_ANALYSIS.md](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/V3_ANALYSIS.md) provides the Gemma 2 baseline anomaly set.
- [HYPOTHESES.md](/home/speediedan/repos/interpretune/tests/concept_direction_approach_parity/HYPOTHESES.md) provides the explanatory hypotheses.
- The unified notebook harness and launcher are now the preferred execution path for follow-up runs.