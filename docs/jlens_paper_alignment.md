# The Jacobian lens: the paper's method and interpretune's implementation

This page states, operation by operation, how interpretune's Jacobian-lens surface relates to the method of the
[workspace paper](https://transformer-circuits.pub/2026/workspace/index.html): what is implemented, where, in which
basis, and what remains. It is the single place to look before claiming that a result reproduces the paper or
extends it.

**How to read the status column, and how this page stays true.** Every row states the current behaviour in one
sentence, so the row is true today whether or not anyone updates it later; where a gap exists the row also names an
open issue. A row that said only "pending #N" is forbidden here, because it becomes false the day #N closes and
nothing on the page would show it. This page therefore degrades into *incomplete* rather than into *wrong*. Each
named issue carries a checklist item to update its row, and a scheduled check in the docs link-check workflow fails
by name when a row cites an issue that has closed.

## Paper capability against interpretune status

| Paper capability | Interpretune today | Where |
|---|---|---|
| Fit a Jacobian lens $J_\ell$ | Not implemented; pre-fitted lenses are consumed from the `neuronpedia/jacobian-lens` artifacts. Fitting is out of scope for #225 as written. | `resolve_jlens`, `jlens_layer_for_percentile` in `analysis/optools.py` |
| Read J-lens tokens from an activation | Implemented: the readout calls the model's own final-norm module and unembedding, so it is the paper's formula rather than a reconstruction. | `jlens_read` in `analysis/ops/bundled/jlens/` |
| Probe one concept | Implemented as a cosine against the concept tokens' directions in the norm-aware basis; the paper's unfolded convention is available through `jlens_apply_final_norm=False`. | `jlens_concept_probe` |
| Decompose an activation into sparse J-space concepts | Implemented: nonnegative sparse decomposition by gradient pursuit over the J-lens dictionary, with its residual. | `jlens_sparse_inventory` |
| Add a J-lens direction | The model-level `add` mode exists and takes any direction; a single-vector J-lens source for it is the collection's concern (#420 introduces the `basis=` selector that names which basis produced the direction). | `model_fwd_intervention`, mode `add` |
| Patch one J-lens coordinate into another | Implemented: the `patch` mode swaps the activation's coordinates along a pair of directions through the pseudoinverse; the pair is built in a stated basis. | `model_fwd_intervention`, mode `patch`; `jlens_direction_rows` |
| Clamp coordinates to a clean-pass value | Not implemented. A clean-pass clamp needs a clean reference activation and one basis across both passes; a static projection bound cannot express it. Open: #423. | none |
| Ablate J-space directions | Not implemented. Ablation projects *out* of a subspace; the `project` mode projects *onto* one. Open: #423. | none |
| Reproduce the paper's rates | Not yet: the evidence is the concept-steering demos on gemma-2-2b and gemma-3-1b-it, not the paper's battery. Open: #425. | the two steering demo notebooks |
| Validate an intervention to first order | Not implemented as an instrument: the patch mechanics are validated in tests, and the per-basis first-order prediction of the metric change is open as #539. | `tests/core/test_jlens_patch_validation.py` |
| Record which basis produced a result | The bundled J-lens ops record `jlens_basis` on their outputs; the model-level intervention result does not yet record the basis of the pair it was given. Open: #540. | `jlens_basis_name` in `analysis/optools.py` |

## Three places normalization can appear

The paper and interpretune agree on where the lens lives and disagree only on a shorthand.

| Stage | Object | Paper | Interpretune |
|---|---|---|---|
| Lens fitting | $J_\ell$ | Jacobian from an intermediate residual to a later residual stream | Same; consumed pre-fitted |
| Readout | $W_U \operatorname{norm}(J_\ell h)$ | Final normalization, then unembedding | Same, calling the real modules |
| Fixed vocabulary basis for probing and writing | $v_t$ | Names the rows of $W_U J_\ell$ the J-lens vectors | Derives the norm-aware direction when the readout's normalization makes the two differ |

The fitted $J_\ell$ is a residual-to-residual transport map; normalization and unembedding come after it. The
paper computes its full readout with the normalization in place and then names $W_U J_\ell$ as the vectors it
probes, steers, patches and ablates with. That shorthand drops the normalization the readout contains, and where the
final norm's learned scale is anisotropic the two are not parallel: the scale changes direction, not only magnitude.

## What folding means

For an RMSNorm final norm with effective elementwise scale $s$, token $t$'s pre-softmax score is

$$
z_t = \frac{\langle (W_U[t] \odot s) J_\ell, h \rangle}{\operatorname{rms}(J_\ell h)}
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

Interpretune names the two bases `jlens_paper` (unfolded) and `jlens_norm_aware` (folded), selected by
`jlens_apply_final_norm`, and the folding is a per-model decision rather than a default: it is essential on
gemma-3-1b-it and roughly three times weaker at late layers on gemma-2-2b, because a uniform scale cancels exactly in
patching and only its anisotropy can matter.

## Per-operation comparison

Each row gives the paper's explicit construction, what the unfolded and folded implementations compute, the claim
each can honestly support, and what interpretune does today. The bases are not interchangeable: a folded and an
unfolded result differ by a direction, not by a coefficient, so a result must state which produced it.

| Operation | Paper's explicit form | Unfolded | Folded | Honest claim | Interpretune today |
|---|---|---|---|---|---|
| Read | $\operatorname{softmax}(W_U \operatorname{norm}(J_\ell h))$ | Bare $W_U J_\ell h$ omits the learned scale and, for LayerNorm, centering | Folded dot products divided by $\operatorname{rms}(J_\ell h)$ are an exact rewrite for RMSNorm | Call the actual norm module; treat unfolded scores as a labelled approximation | `jlens_read` calls the real modules |
| Probe | Score or cosine against $v_t$ | Literal paper convention | Norm-aware fixed direction | Offer both, labelled | `jlens_concept_probe`, both bases by `jlens_apply_final_norm` |
| Add | $h + \alpha v_t$ | The paper's written operation | A different direction, readout-aligned; no scalar $\alpha$ repairs an anisotropic rotation | Folded for "the readout direction", unfolded for replication | `add` mode takes either; the source op states the basis (#420) |
| Ablate | Project out $P_V = V V^{+}$ | Paper-basis subspace | Norm-aware subspace; a different column span | Must state the basis | Not implemented (#423) |
| Patch | $c = V^{+} h$, $h' = h + V(\sigma(c) - c)$ | The paper's literal coordinate swap | A different plane and pseudoinverse; uniform scale cancels, anisotropic scale does not | Make the basis explicit in the API and the result | `patch` mode, pair built in the stated basis; result basis recorded once #540 lands |
| Clamp | Hold coordinates at a clean-pass value | Clamp paper-basis coordinates | Clamp norm-aware coordinates | One basis across the clean and intervened passes | Not implemented (#423) |
| Validate | No fold choice specified | Validate $\Delta h$ in the unfolded basis | Validate $\Delta h$ in the folded basis | Validate each basis separately with $\nabla_h m^{\mathsf{T}} \Delta h$ | Mechanics pinned in tests; the per-basis instrument is #539 |

Two corrections the paper's own text implies, both carried here as requirements on future work: the top $k$ active
concepts should come from the sparse nonnegative decomposition rather than the top $k$ dot products (the bundled
`jlens_sparse_inventory` does this), and a pseudoinverse is required for any multi-vector write, because applying
two one-vector operations in sequence cannot reproduce a coordinate swap when the vectors overlap.

## The decision rule

1. A result states its basis. `jlens_paper` and `jlens_norm_aware` are declared configurations of the intervention
   surface, never defaults; an op that does not state one is refused rather than defaulted, and a backend that
   cannot honour the requested one refuses it by name. This is the rule the capability vocabulary in
   [Interpretune Intervention APIs](interpretune_intervention_apis.md) carries for every configuration axis.
1. Default according to the claim: paper reproduction uses `jlens_paper`; "the direction governing the normalized
   readout" uses `jlens_norm_aware`.
1. Folded and unfolded behavioural results are never compared as though they differed by a coefficient.
1. Each selected basis is validated with its own first-order prediction, once #539 lands; until then the claim a
   result can make is the one its tests pin.

## Maintenance

The rows above that name an issue are #225, #420, #423, #425, #539 and #540. Each carries a checklist item to update
its row when it closes. The docs link-check workflow, which runs weekly and on demand against the live web, reads
this page's issue citations and fails by name on any that has closed, so a stale row cannot survive a week unnoticed.
The page states current behaviour in every row for the reason given at the top: it must be true between those runs
as well.
