# The Jacobian lens: the paper's method and Interpretune's implementation

This page states, operation by operation, how Interpretune's Jacobian-lens surface relates to the method of the
[workspace paper](https://transformer-circuits.pub/2026/workspace/index.html): what is implemented, where, in which
basis, and what remains. It is the single place to look before claiming that a result reproduces the paper or
extends it.

**How to read the status column, and how this page stays true.** Every row states the current behaviour in one
sentence, so the row is true today whether or not anyone updates it later; where a gap exists the row also names an
open issue. A row that said only "pending #N" is forbidden here, because it becomes false the day #N closes and
nothing on the page would show it. This page therefore degrades into *incomplete* rather than into *wrong*. Each
named issue carries a checklist item to update its row, and a scheduled check in the docs link-check workflow fails when a row cites an issue that has closed.

## Paper capability against Interpretune status

| Paper capability | Interpretune today | Where |
|---|---|---|
| Fit a Jacobian lens $J_\ell$ | Not implemented; pre-fitted lenses are consumed from the `neuronpedia/jacobian-lens` artifacts. Fitting is out of scope for #225 as written. | `resolve_jlens`, `jlens_layer_for_percentile` in `analysis/optools.py` |
| Read J-lens tokens from an activation | Implemented: the readout calls the model's own final-norm module and unembedding, so it is the paper's formula rather than a reconstruction. | `jlens_read` in `analysis/ops/bundled/jlens/` |
| Read through a J++ lens | Implemented for pre-fitted lenses: `jlens_kind="jpp"` resolves a J++ artifact through the same seam, checked against its sidecar's declared fit method, and `jlens_read` defaults to the J++ readout's semantic-token mask and RMS divisor. Fitting J++ lenses is not implemented. See "The J++ lens" below. | `resolve_jlens`, `jlens_read` |
| Probe one concept | Implemented as a cosine against the concept tokens' directions in the norm-aware basis; the paper's unfolded convention is available through `jlens_apply_final_norm=False`. | `jlens_concept_probe` |
| Decompose an activation into sparse J-space concepts | Implemented: nonnegative sparse decomposition by gradient pursuit over the J-lens dictionary, with its residual. | `jlens_sparse_inventory` |
| Add a J-lens direction | Implemented: the model-level `add` mode takes any direction, and `concept_direction` builds a single J-lens direction in the basis its `concept_basis` names (`jlens_folded` or `jlens_unfolded`, with no default). | `model_fwd_intervention`, mode `add`; `concept_direction` |
| Patch one J-lens coordinate into another | Implemented: the `patch` mode swaps the activation's coordinates along a pair of directions through the pseudoinverse; the pair is built in a stated basis. On a backend with activation capture and `add` but no `patch` (its steering parameters are static), the swap has an exact two-pass form: capture $h$, compute the static update $V(\sigma(c) - c)$ with $c = V^{+} h$, and add it at scale 1. | `model_fwd_intervention`, mode `patch`; `jlens_direction_rows` |
| Clamp coordinates to a clean-pass value | Implemented as bound-into-range: the `clamp` mode bounds the activation's pseudoinverse coordinates along a stacked basis into `[lo, hi]`, leaving in-range coordinates untouched; a bandless or inverted band is refused. A clean-pass-value hold is a two-pass construction over it. | `model_fwd_intervention`, mode `clamp` |
| Ablate J-space directions | Implemented: the `reject` mode removes the activation's component in the span (`alpha = 1` full removal), the complement of `project`. | `model_fwd_intervention`, mode `reject` |
| Reproduce the paper's rates | Reproduced in shape on the paper's own two-hop swap prompts on gemma-3-1b-it and gemma-3-4b-it: intermediate swaps peak at mid depth and collapse late, answer swaps peak late, and an unrelated-entity control swap stays near zero. On gemma-3-4b-it the same holds through the chat template, with the control less specific at mid-to-late layers. The polysemy probe (the paper's Finding 4) finds cancellation at the shared word in the lens readout on gemma-2-2b and on gemma-3 from 1b to 12b, across three wordings. Read through fixed-direction coordinates instead, it depends on the basis. gemma-3-27b is not measured, and on gemma-3 why the coordinate statistic changes with the basis is not established. | an experiment component outside this repository; the steering demo notebooks for single prompts |
| Attribute toward what the lens reads | Implemented for token targets: `attribution_target_source='jlens_readout'` attributes the top-k tokens a preceding `jlens_read` returns as ordinary final-layer logit targets, and records the lens layer and token ids with the graph. The lens read itself is attributed at the lens layer: a `concept_direction` built with a J-lens `concept_basis` becomes a target read at the output of the lens layer, whose edges match fully frozen interventions. | `compute_attribution_graph` in `analysis/ops/bundled/circuit_tracer/` |
| Validate an intervention to first order | Implemented: `intervention_first_order_check` runs downstream of `model_fwd_intervention`, predicts the metric change from the applied edit and the metric gradient at the clean site, and reports predicted, measured and residual per named basis for every write mode. | `intervention_first_order_check`; `tests/core/test_intervention_first_order_check.py` |
| Record which basis produced a result | Implemented: bundled J-lens ops stamp `jlens_basis` on their outputs, and the steering demos build patch pairs with the basis-recording collection revision, so a result names the basis that produced it. | `jlens_basis_name` in `analysis/optools.py` |

## Three places normalization can appear

The paper and Interpretune agree on where the lens lives and disagree only on a shorthand.

| Stage | Object | Paper | Interpretune |
|---|---|---|---|
| Lens fitting | $J_\ell$ | Jacobian from an intermediate residual to a later residual stream | Same; consumed pre-fitted |
| Readout | $W_U \mathrm{norm}(J_\ell h)$ | Final normalization, then unembedding | Same, calling the real modules |
| Fixed vocabulary basis for probing and writing | $v_t$ | Names the rows of $W_U J_\ell$ the J-lens vectors | Derives the norm-aware direction when the readout's normalization makes the two differ |

The fitted $J_\ell$ is a residual-to-residual transport map; normalization and unembedding come after it. The
paper computes its full readout with the normalization in place and then names $W_U J_\ell$ as the vectors it
probes, steers, patches and ablates with. That shorthand drops the normalization the readout contains, and where the
final norm's learned scale is anisotropic the two are not parallel: the scale changes direction, not only magnitude.

## What folding means

For an RMSNorm final norm with effective elementwise scale $s$, token $t$'s pre-softmax score is

$$
z_t = \frac{\langle (W_U[t] \odot s) J_\ell, h \rangle}{\mathrm{rms}(J_\ell h)}
$$

so the fixed direction that governs the normalized readout is the **folded** vector
$v_t^{\mathrm{folded}} = (W_U[t] \odot s) J_\ell$, while the paper's written shorthand is the **unfolded**
$v_t^{\mathrm{paper}} = W_U[t] J_\ell$. The denominator is a scalar shared by every token at one position, so it
changes logit magnitude and softmax temperature but not within-position ranking. For LayerNorm the fixed direction
also carries centering, $C (W_U[t] \odot s) J_\ell$ with $C = I - \tfrac{1}{d} \mathbf{1} \mathbf{1}^{\mathsf{T}}$,
so the two norm families cannot share one "multiply the unembedding by a scale" implementation. The derivation, the
measurements on gemma-2-2b and gemma-3-1b-it, and the decision rule are in
[Folding the final norm into lens directions](jlens_norm_folding.md); this page only names the two bases and
which one each operation uses.

Interpretune names the two bases `jlens_unfolded` and `jlens_folded`, selected by
`jlens_apply_final_norm`, and the folding is a per-model decision rather than a default: it is essential on
gemma-3-1b-it and roughly three times weaker at late layers on gemma-2-2b, because a uniform scale cancels exactly in
patching and only its anisotropy can matter.

## Per-operation comparison

Each row gives the paper's explicit construction, what the unfolded and folded implementations compute, the claim
each can honestly support, and what Interpretune does today. The bases are not interchangeable: a folded and an
unfolded result differ by a direction, not by a coefficient, so a result must state which produced it.

| Operation | Paper's explicit form | Unfolded | Folded | Honest claim | Interpretune today |
|---|---|---|---|---|---|
| Read | $\mathrm{softmax}(W_U \mathrm{norm}(J_\ell h))$ | Bare $W_U J_\ell h$ omits the learned scale and, for LayerNorm, centering | Folded dot products divided by $\mathrm{rms}(J_\ell h)$ are an exact rewrite for RMSNorm | Call the actual norm module; treat unfolded scores as a labelled approximation | `jlens_read` calls the real modules |
| Probe | Score or cosine against $v_t$ | Literal paper convention | Norm-aware fixed direction | Offer both, labelled | `jlens_concept_probe`, both bases by `jlens_apply_final_norm` |
| Add | $h + \alpha v_t$ | The paper's written operation | A different direction, readout-aligned; no scalar $\alpha$ repairs an anisotropic rotation | Folded for "the readout direction", unfolded for replication | `add` mode takes either; the source op states the basis (`concept_basis`) |
| Ablate | Project out $P_V = V V^{+}$ | Paper-basis subspace | Norm-aware subspace; a different column span | Must state the basis | `reject` mode (`model_fwd_intervention`); the pair-building op records the basis |
| Patch | $c = V^{+} h$, $h' = h + V(\sigma(c) - c)$ | The paper's literal coordinate swap | A different plane and pseudoinverse; uniform scale cancels, anisotropic scale does not | Make the basis explicit in the API and the result | `patch` mode, pair built in the stated basis; result basis recorded |
| Clamp | Hold coordinates at a clean-pass value | Clamp paper-basis coordinates | Clamp norm-aware coordinates | One basis across the clean and intervened passes | `clamp` mode bounds into `[lo, hi]`; a clean-pass hold is a two-pass construction over it |
| Validate | No fold choice specified | Validate $\Delta h$ in the unfolded basis | Validate $\Delta h$ in the folded basis | Validate each basis separately with $\nabla_h m^{\mathsf{T}} \Delta h$ | Mechanics pinned in tests; the per-basis instrument is `intervention_first_order_check` |

Two corrections the paper's own text implies, both carried here as requirements on future work: the top $k$ active
concepts should come from the sparse nonnegative decomposition rather than the top $k$ dot products (the bundled
`jlens_sparse_inventory` does this), and a pseudoinverse is required for any multi-vector write, because applying
two one-vector operations in sequence cannot reproduce a coordinate swap when the vectors overlap.

## The decision rule

1. A result states its basis. `jlens_unfolded` and `jlens_folded` are declared configurations of the intervention
   surface, never defaults; an op that does not state one is refused rather than defaulted, and a backend that
   cannot honour the requested one refuses it. This is the rule the capability vocabulary in
   [Interpretune Intervention APIs](interpretune_intervention_apis.md) carries for every configuration axis.
1. Default according to the claim: paper reproduction uses `jlens_unfolded`; "the direction governing the normalized
   readout" uses `jlens_folded`.
1. Folded and unfolded behavioural results are never compared as though they differed by a coefficient.
   Measured on the shipped gpt2-small lens at layer 8 (Paris-vs-London concept direction, collection
   revision `0731326e`): $1 - \lvert \cos \rvert = 0.0681$. The bases differ in practice, not only in
   construction, so the comparison the rule forbids would also be wrong numerically.
1. Each selected basis is validated with its own first-order prediction: `intervention_first_order_check` reports
   predicted, measured and residual under the basis the run names, so a large residual is a finding about that
   basis rather than a coefficient to be explained away.

## The J++ lens

The [J++ lens](https://github.com/safety-research/jpp_lens) (Ayonrinde and Lindsey, "J++ Lens: Jacobian
filtering enables more faithful workspace readouts") is a later method built on the paper's lens. Interpretune
reads J++ lenses but does not fit them.

**What changes at fit time.** Three things, in the post's terms. *Jacobian filtering:* a router assigns
fitting activations to $K$ clusters, each cluster's Jacobians are averaged into an expert, and per-layer weights
combine the experts, down-weighting Jacobians from activations whose readouts are poor. *Layer-wise relevance
propagation* in the backward pass when the Jacobians are computed, which detaches each RMSNorm's
$1/\mathrm{rms}$ factor. *Readout filtering:* non-semantic tokens are dropped from the readout.

**What is unchanged at read time.** A J++ lens is still one map $J_\ell$ per layer with the paper's shape, read
as $W_U \mathrm{norm}(J_\ell h)$, at the same cost. Everything on this page about the readout, the two bases and
folding applies to it unchanged. The folding page's J++ section covers the one difference that matters there.

**The supervised component.** The per-layer weights are fitted to maximize readout recall on graded items. A
J++ lens's readout score on items like those it was weighted on is therefore not a held-out measurement. Score
it with the task held out of the weight fit, as the measurements in [J-Space Usage](jlens_usage.md) do.

**The arbitrary per-layer scale.** Each layer's J++ map carries an arbitrary scale, so raw J++ readout
magnitudes do not compare across layers or against a J-lens. `jlens_read` divides by the final norm's RMS by
default for a J++ lens. The divisor leaves the ranking within a position unchanged.

**The post's claims against what was measured here.** The post reports a large readout gain and swaps at least
as strong as the J-lens's, on models from 9B to 284B parameters, with the swap claim measured on one model. On
two smaller models, gemma-3-1b-it and Qwen3.5-4B, the readout gain reproduces in direction and is smaller, and
the swap claim does not hold on our two-hop battery: the J-lens swaps better on both. The numbers, and the
resulting choice of kind per use, are in [J-Space Usage](jlens_usage.md).

## Maintenance

The one row above that names an issue cites #225, which carries a checklist item to update its row when it
closes. The docs link-check workflow, which runs weekly and on demand against the live web, reads
this page's issue citations and fails on any that has closed, so a stale row cannot survive a week unnoticed.
The page states current behaviour in every row for the reason given at the top: it must be true between those runs
as well.
