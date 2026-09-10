# Folding the final norm into lens directions

A Jacobian-lens readout is $\mathrm{softmax}(W_U \cdot \mathrm{norm}(J h))$. A common shorthand
describes the lens directions as "the rows of $W_U J$", which drops the $\mathrm{norm}$ the readout
contains. The two agree only
in a special case, and where they disagree the difference is large enough to decide whether a
concept-steering intervention works at all.

This page derives the faithful form, states what does and does not cancel, and gives the per-model
decision rule with the measurements behind it. The composition itself lives in one place,
`interpretune.analysis.optools.fold_norm_into_unembed_rows`, so a caller picks a policy rather than
re-deriving a convention.

## The derivation

**RMSNorm.** With elementwise scale $s$, the norm is $\mathrm{norm}(x) = s \odot x / \mathrm{rms}(x)$.
Pushing $s$ through the dot product:

$$
W_U[c] \cdot \mathrm{norm}(x) = (W_U[c] \odot s) \cdot x / \mathrm{rms}(x)
$$

So the readout's own direction for token $c$ is $W_U[c] \odot s$, and composing with the lens gives
$v_c = (W_U[c] \odot s) J$. The $1/\mathrm{rms}(x)$ factor is a positive per-input scalar: it changes
the logit's magnitude and never its direction, so it is correctly ignored when what you want is a
basis. The shorthand is exact when $s$ is identically 1, and only then.

**LayerNorm.** A LayerNorm additionally subtracts the mean, so
$\mathrm{norm}(x) = s \odot (x - \mu(x) \mathbf{1}) / \sigma(x) + b$ with $\mu$ the mean and
$\sigma$ the standard deviation. Pushing $s$ through as before and then absorbing the centering:

$$
\begin{aligned}
(W_U[c] \odot s) \cdot (x - \mu(x)\mathbf{1})
  &= (W_U[c] \odot s) \cdot x - \mu(x)\bigl((W_U[c] \odot s) \cdot \mathbf{1}\bigr) \\
  &= C(W_U[c] \odot s) \cdot x,
  \qquad C = I - \mathbf{1}\mathbf{1}^\top / d
\end{aligned}
$$

so the faithful direction carries a centering projector as well as the scale. The learned bias $b$
adds an input-independent logit offset and drops out of a direction entirely.

**Why the centering is not optional in the way a rescaling is.** A uniform rescaling of a lens basis
cancels exactly under a patch-mode swap: scaling $V$ to $cV$ scales the pseudoinverse to $V^+/c$, so
the reconstruction is unchanged. Centering is not a rescaling. It removes an additive uniform
component, which moves the direction and therefore the plane the swap happens in. The two are easy to
conflate, and conflating them predicts that centering is harmless, which it is not.

**Per-family conventions, and they do not follow the family name.** `gemma`, `gemma2` and `gemma3`
RMSNorms apply `(1 + weight)`; `gemma3n` and the whole `gemma4` line apply `weight` directly, as do
other HF RMSNorms. TransformerLens folds the offset at conversion for the families that have one, and
its bridge path applies the same offset when exposing the parameter, so its stored weight is already
the applied scale either way.

The naming is a trap worth stating outright, because it produced a real defect: `gemma3` takes the
offset and `gemma3n` does not, so no name prefix separates them. Getting this wrong is silent, since
every path still produces a plausible direction.

**So the seam does not decide it by name at all: it reads the scale out of the norm module.** Since
`rms(c * 1) = c` for any constant `c`, evaluating the module on a constant vector returns the
elementwise scale the module actually applies, whatever convention its family uses. There is no
membership test to be exact about and no unrecognized family to warn on, because an unrecognized family
is precisely the case this handles without special treatment. A norm that cannot be probed is refused by
name rather than resolved to either convention, since choosing wrong is invisible in the output.

The split above is not incidental detail retained for completeness. It is the reason the mechanism is a
measurement rather than a declaration: a table would have to be right about every family that exists and
every family that will exist, and being wrong about one is silent.

## What the measurements say

**Folding is not uniformly beneficial, and the naive predictor is wrong.** A four-model sweep in
patch mode at scale 1.0 on a concept-contrast task:

| model | $\mathrm{rms}(s - 1)$ | $\cos$(folded, unfolded) | late layers |
| --- | --- | --- | --- |
| gemma-2-2b | 2.54 | 0.20 to 0.52 | unfolded about 3x stronger (L24 +17.77 vs +5.69; both flip) |
| gemma-3-1b-it | 8.79 | 0.65 to 0.87 | folded essential (unfolded never flips) |
| gpt2 | 1.48 | 0.49 to 0.70 | at the capability floor for this task |
| pythia-70m | 12.61 | 1.000 | folding is provably a no-op here |

pythia-70m has the largest scale deviation of the four and a folded-versus-unfolded cosine that rounds
to 1.000. **An earlier version of this page attributed that to the uniform-cancellation result above.
That attribution is wrong, and it is corrected here rather than annotated below**, because the row it
explains is still correct and a reader who accepts the explanation stops looking.

Three measurements rule it out. Pythia's final norm is a **LayerNorm**, so folding applies the centering
projector as well as the scale, which is precisely not the bare rescale the cancellation argument
covers: this page argues two paragraphs above that centering moves the direction where a rescale does
not. The scale is not uniform either, since $\mathrm{rms}(s-1)$ is 12.61 and the folded-versus-unfolded
cosine measured **before** composing with the lens is 0.987. The near-parallelism appears only **after**
composition, at 0.99988 to 0.99993.

So the correct reading is narrower and the cause is open: for this checkpoint the lens composition
collapses a direction difference that is plainly present beforehand, and why it does is not established
here. What the row still supports is the conclusion it was cited for. The size of $s - 1$ predicts
nothing; only the anisotropy of $s$, and specifically its alignment with the task pathway, can matter.

**The mechanism is coordinate alignment, confirmed by destroying it.** Folding with a permuted copy
of the model's own scale vector preserves its distribution and destroys its coordinate
correspondence. Across three seeds this collapses folded behavior to unfolded on both gemma models:

| model | layer | true-folded | unfolded | permuted (3 seeds) |
| --- | --- | --- | --- | --- |
| gemma-2-2b | 24 | +5.69 | +17.77 | +16.6 / +16.0 / +17.1 |
| gemma-3-1b-it | 24 | +13.90 | -0.29 | +0.12 / +3.42 / -15.12 |

The distribution of the scale is inert. The specific correspondence between the scale's coordinates
and the task pathway carries the entire effect, in both directions. One permuted seed landing at
-15.12 also shows that a wrong alignment is worse than no folding, which is the strongest argument
for keeping this a deliberate per-model choice rather than a default.

**Why the two gemma models order oppositely.** Decomposing by the top decile of coordinates by scale
magnitude: on gemma-3-1b-it the task-pathway gradient is about 5x enriched in those coordinates
(0.487 against a 0.10 baseline) while activation variance there is about 10x depleted (0.009). That
is a quiet but sensitive channel, and folding aims the swap plane into it (pole-vector energy 0.62
folded against 0.27 unfolded). On gemma-2-2b every split sits near baseline.

The gemma-2-2b result is a magnitude story rather than a reversal of mechanism: folded aligns
*better* with the task pathway (cosine 0.81 against 0.62), but unfolded's displacement is 3.4x larger
(norm 98 against 29) and wins on the product. On gemma-3-1b-it unfolded fails for a different reason,
which is worth distinguishing: the concept contrast is nearly absent in the unfolded basis (coordinate
norm 869 yet displacement norm 6.1) and anti-aligned with the task pathway (cosine -0.67). Not weak,
wrong.

## The decision rule

Fold when the model stores the task-relevant contrast in the coordinates the final norm amplifies.
In practice:

1. **Default to folding**, because it is the readout-faithful form and because the failure it avoids
   (a direction the readout cannot express) is worse than the one it risks.
2. **Check, do not assume.** The cheap check is first-order: for a two-pole swap with displacement
   $D$, compare $D \cdot \nabla_h(\mathrm{logit}_{\text{target}} - \mathrm{logit}_{\text{other}})$
   between the folded and unfolded bases and take the larger. Measured against real patched deltas this predicted three of four cases within
   3%, the fourth being the largest displacement, where a first-order estimate is expected to drift.
3. **Keep the option.** The opt-out is the stronger setting on at least one major model, so it is a
   supported configuration rather than legacy. Op collections building lens directions expose this as
   `jlens_apply_final_norm`.
4. **Never mix bases.** The pair and the model must share a residual basis. TransformerLens weight
   processing changes that basis, so a folded vector built from a processed `W_U` would double-count
   the fold; see the weight-processing note in `interpretune_intervention_apis.md`.

## A note on method

The prototype implemented both variants because the paper's formal readout and its operational
shorthand disagreed about the norm within the same section. The discrepancy itself was the prompt,
and the measurement above is the payoff. When a source's formal definition and its working shorthand
differ, implementing both and measuring costs little and occasionally decides the result.

The same section is also a reminder to check which claims rest on measurement. An earlier draft of
this material asserted that unfolded vectors were far weaker on gemma-2-2b. No such measurement
existed: only the folded variant had been run on that model, and the sweep above found the opposite.

## Not yet measured

The TransformerLens row of the seam is asserted by construction rather than measured against a real
TransformerLens model, and the interaction between weight processing and folding is documented as a
constraint rather than quantified. Both are tracked with the cross-backend validation work rather
than here, because both need the two backends running against the same lens artifacts.
