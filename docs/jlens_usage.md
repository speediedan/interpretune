# J-Space Usage

How to read, probe, and steer with Jacobian lenses in interpretune. For the
folding derivation and per-model decision rule see the norm-folding page; for
how the operations map onto the paper see the paper-alignment page. This page
is the practical surface: which op to call and what each choice means.

## Reading: `jlens_read`

`jlens_read` renders the ranked vocabulary tokens an activation is disposed to
produce, at chosen (layer, position) sets: the lens maps the activation through
the fitted Jacobian and the readout applies the unembedding with the final
norm folded in. The logit lens is the special case where the lens is the
identity. Lenses resolve from the `neuronpedia/jacobian-lens` repository by
probing its layout (never by formatting a path): the resolver matches the
artifact stem against the model name and confirms the match against the
artifact's sidecar before loading anything. A caller that already holds a
checkpoint passes its path directly and skips discovery.

Layers are dense model layers; lenses exist only for the fitted subset, so a
layer is either named explicitly (refused by name when unfitted) or selected
as a percentile over the fitted set (85% depth is the steering default —
late enough for naming tasks, below the final layers).

## Probing and inventory

Single-concept probing is cosine similarity against chosen J-lens vectors
(`jlens_concept_probe`). A discrete concept inventory
(`jlens_sparse_inventory`) decomposes an activation over at most ~25 J-lens
vectors by sparse nonnegative gradient pursuit: the J-space is a union of
low-dimensional cones, and the inventory names which ones are active. The
published measurements behind the technique: the J-space carries 5–10% of
activation variance yet most of the causal weight for verbal report.

## The `concept_basis` selector

`concept_direction` takes `concept_basis` naming where the concept vector
comes from, with no default: paper reproduction wants the unfolded basis
(`jlens_paper`) while the readout direction wants the norm-aware one
(`jlens_norm_aware`), and silently picking either would corrupt the other
use. `embed` and `store` remain for the non-lens bases.

## Steering and the folding default

Steering (`patch`, `clamp`, `reject` intervention modes) applies through the
folded basis by default. Folding aims the edit into the coordinates the final
norm amplifies, so it helps exactly where the model stores the task contrast:
essential on gemma-3-1b-it (unfolded vectors never flip at any layer), ~3x
weaker at late layers on gemma-2-2b (both forms flip). Go through
`resolve_unembed_and_norm_scale` rather than hand-rolling either side: the
scale convention differs per model family, and a mismatched fold is wrong in
a way no magnitude adjustment repairs.
