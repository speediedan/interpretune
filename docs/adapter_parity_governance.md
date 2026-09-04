# Adapter Parity Governance

This document governs which execution backends hold a **tight numerical parity contract** with each other, which are validated separately, and why the split falls where it does.

It began as an account of HookedTransformer, TransformerBridge and NNsight. The reasoning generalized, so the document did: any adapter — bundled or hub-delivered — is placed by the same question, and the answer determines what a test may legitimately assert about it.

## Why This Matters

The NNsight analysis generalization work made backend comparisons a first-class part of the development process. That exposed an important distinction:

- TransformerBridge and NNsight both execute the native HuggingFace forward pass.
- HookedTransformer runs a separate TransformerLens forward implementation over converted weights.

That architectural split matters for both implementation decisions and test expectations.

## The Two Forward Families

### HuggingFace-native family

TransformerBridge, NNsight and interp-engine all stay on the HuggingFace model path.

- TransformerBridge wraps the HF model for TransformerLens-compatible analysis access.
- NNsight traces the HF model directly through its deferred execution system.
- interp-engine hooks the HF modules directly; its architecture facts (sandwich norms, residual multipliers, parallel residual) are detected *structurally* from the real model rather than from converted weights or names.
- All three therefore inherit the same native attention-mask semantics, pad-position handling, and position-id behavior.

**Placement is decided by which forward runs, not by who ships the adapter.** interp-engine arrives as a hub-delivered component rather than a bundled one, and that is irrelevant to this classification — a bundled adapter running its own forward would sit in the other family, and a third-party adapter on the HF path sits here.

### HookedTransformer family

HookedTransformer converts weights into a TransformerLens-native model and executes a different forward path.

- Attention masking uses TransformerLens' multiplicative boolean masking path.
- Pad-position embeddings are explicitly zeroed.
- Position-id and BOS/PAD edge-case handling differ from the HuggingFace path.

These are not incidental implementation details. They produce meaningfully different intermediate activations and, on padded inputs, materially different logits.

## What We Observed

The backend investigation established three stable divergence regimes.

| Scenario | Observed behavior |
|---|---|
| No attention mask | HookedTransformer and TransformerBridge are bit-for-bit identical |
| Unpadded input with an all-ones mask | Divergence stays very small, around `6e-5` |
| Real padded inputs with active masking | Divergence grows substantially through the stack |

The padded regime is the important one for Interpretune, because the analysis pipeline frequently operates on real tokenized batches with padding and mask-aware model execution.

## Why Padded Batches Diverge

The branch work showed that the large Bridge vs HookedTransformer gap is driven by forward-path differences at masked positions rather than generic float32 noise.

The main contributors are:

- different attention-mask arithmetic
- different treatment of pad-position embeddings
- different position-id handling at padded locations
- GPT-2 BOS/PAD overlap behavior in the HookedTransformer path
- layer-by-layer amplification of those early residual-stream differences

In practice, those differences can remain modest for derived scalar metrics such as logit differences while still being large for raw logits and latent activation patterns.

## How That Informed the Analysis Backend Design

The multi-backend refactor did not treat all wrappers as interchangeable.

Instead, the backend layer now reflects the fact that there are two different forward families:

- `NNsightModelBackend` and TransformerBridge-based analysis both target HuggingFace-native execution.
- HookedTransformer remains supported, but it is understood as a separate execution path with known architectural divergence on padded workloads.

This separation allowed the analysis ops, cache abstractions, and parity fixtures to be generalized without forcing a false assumption that all wrappers should numerically match under the same tolerances.

## Current Testing Strategy

Interpretune currently treats HuggingFace-native execution as the tight parity path.

### Active tight parity — a convergence set, not a pair

TransformerBridge ↔ NNsight ↔ interp-engine is the backend parity contract.

- all operate on the HF forward path
- differences come primarily from hook and tracing mechanisms rather than model semantics
- tight tolerances are appropriate here

This is the parity signal that protects the generalized latent-model analysis ops.

**Why a third member is worth more than more coverage.** A *pair* that agrees cannot distinguish "both correct" from "both wrong in the same way" — the two implementations share the HF forward, so a shared misreading of it is invisible to the comparison. A third independent implementation over that same forward converts agreement into **convergence**: three agreeing is evidence about the forward; two agreeing is evidence about the two. Adding members strengthens the *existing* contract rather than running a parallel one beside it.

**Optional members are gated on being installed, and the skip carries a reason.** A hub-delivered adapter is not present in every environment. Its parity leg is evaluated through the same declared-requirements predicate the rest of the rails use, so an absent backend produces "unavailable here, because X is not installed" rather than a bare skip — the distinction between *this comparison could not run* and *this comparison does not exist*.

### Three layers, tested separately, because their failures mean different things

Parity is not one claim. A backend can observe the forward correctly and still edit it wrongly, and an
analysis op can diverge for reasons neither of those explains. The suite is therefore split by layer, and
the split is diagnostic rather than organizational:

| Layer | Question | Why it is separate |
| --- | --- | --- |
| **1. Capture** | Does the backend see the same activation? | A capture divergence *explains* a divergence at layers 2 and 3. Testing it separately localizes the fault instead of reporting three failures for one cause — and the intervention failure would be the misleading one to chase. |
| **2. Intervention** | Does the backend apply the same edit, **to the same positions**? | Shapes and plausibility do not distinguish a last-token edit from a whole-prompt one. This is the layer where a silent wrong-answer defect lives. |
| **3. Analysis ops** | Do ops built on those backends produce the same values? | An op composes many calls; a divergence here is only interpretable once 1 and 2 are known good. |

**Layer 2's discriminator is causality, not introspection**, which is what makes it work identically for
every participant. In a causal LM, editing the residual stream at position `p` can only affect positions
`>= p` downstream. So a last-token intervention moves exactly the final position of a later activation, and
a whole-prompt intervention moves all of them. "Which positions were steered" — an internal question each
backend answers differently, and some cannot answer at all — becomes one observable that every backend
exposes and that the HF reference exposes too.

**Assert the position SET, not a count, and never the values alone.** A whole-prompt intervention produces
entirely plausible activations, so a value comparison passes under the defect. A count is no better: a bug
steering position 0 under a last-token spec gives a count of 1, and a whole-prompt intervention that
happens to be inert somewhere also gives 1 while operating on the wrong scope. Both are the same failure —
the arithmetic agrees while the operation differs. **And "changed" must name a tolerance**, or the set goes
noisy from nondeterminism at ~1e-9 and the pressure becomes to loosen the assertion until it passes, which
is how a guard decays into decoration.

**Validate the discriminator in both directions before trusting it.** A test that only checks the
last-token case cannot distinguish "the instrument works" from "the instrument always returns the last
position". The whole-prompt case is what demonstrates it can tell the two apart.

### Matching names is not matching tensors

**A parity comparison must be between points that mean the same thing, and hook NAMES do not establish that.**

interp-engine's `mlp_in` and `attn_in` are the sublayer's *post-norm* inputs. TransformerLens' `hook_mlp_in` and `hook_attn_in` fire on the *pre-norm* residual. Those differ by a whole normalization — measurably so: on gemma-3-1b-it the two tensors reach a cosine of ~0.17 and differ in norm by up to three orders of magnitude.

A parity test that resolves names through a translation layer, without checking that the layer preserves meaning, will compare a post-norm tensor against a pre-norm one and report a divergence that is real but is not the divergence it claims to measure. **The dangerous response is the natural one: loosen the tolerance until it passes** — which buries a genuine defect under a threshold chosen to hide it, and leaves a green test asserting nothing.

So: compare points established to be semantically identical, and keep the correctness of any name translation as a *separate* assertion with its own test. A parity suite and a mapping suite answer different questions, and merging them lets each one's failure be mistaken for the other's.

**A measured instance, and the reason the family boundary is drawn where it is.** The TransformerLens participant in the convergence set must be **TransformerBridge**, not `HookedTransformer.from_pretrained`. The latter is the legacy path: it converts weights, folding LayerNorm and centering writing weights by default. Those transforms preserve model *outputs* while changing the residual stream — so its `hook_resid_pre` is a differently-scaled tensor wearing the same name as the bridge's. Substituting it fails the convergence assertion while interp-engine and NNsight pass, and the reflex when exactly one participant disagrees is to widen the tolerance, which would have buried the real signal: the family is defined by *executing the HF forward*, and the legacy path does not.

### Separate HookedTransformer validation

HookedTransformer is still covered, but not by the same numerical parity contract.

- HookedTransformer-specific functionality is validated in its own dedicated tests
- former Bridge ↔ HookedTransformer bounded-divergence parity checks were removed
- the padded-batch divergence was large enough that wide tolerances stopped providing useful regression signal

That change reduced false equivalence in the tests and made the remaining parity checks sharper and more actionable.

## Current Source of Truth

For the generalized analysis work in this branch, HuggingFace-native results are the reference behavior.

That means:

- TransformerBridge is the reference implementation inside the TransformerLens-facing path
- NNsight is expected to closely match that behavior
- HookedTransformer remains a supported legacy-compatible execution path, but not the numerical reference for padded-backend parity

This is the practical source-of-truth model used by the current codebase and test suite.

## User Impact

For developers and researchers using Interpretune today, the consequence is straightforward:

- if you want backend parity expectations across the generalized analysis stack, compare NNsight against TransformerBridge
- if you use HookedTransformer, expect valid analysis behavior but do not assume raw numerical parity with HuggingFace-native execution on padded workloads
- when debugging divergences, first identify which forward family the run belongs to
- when configuring TransformerBridge mode in SAELensConfig, use `ITLensBridgeConfig` as `tl_cfg` (not `ITLensFromPretrainedConfig` with `use_bridge=True` — that will emit a misconfiguration warning since the config fields are HookedTransformer-specific)

That framing aligns the implementation, the tests, and the observed model behavior.
