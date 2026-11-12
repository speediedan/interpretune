# J-Space Usage

How to read, probe, and steer with Jacobian lenses in interpretune. For the
folding derivation and per-model decision rule see
[Folding the final norm into lens directions](jlens_norm_folding.md); for how
the operations map onto the paper see [the paper-alignment page](jlens_paper_alignment.md).
This page is the practical surface: which op to call and what each choice means.

## Reading: `jlens_read`

`jlens_read` renders the ranked vocabulary tokens an activation is disposed to
produce, at chosen (layer, position) sets: the lens maps the activation through
the fitted Jacobian and the readout applies the unembedding with the final
norm folded in. The logit lens is the special case where the lens is the
identity. Lenses resolve from the `neuronpedia/jacobian-lens` repository by
probing its layout (never by formatting a path): the resolver matches the
artifact stem against the model name and confirms the match against the
artifact's sidecar before loading anything. A caller that already holds a
checkpoint passes its path directly and skips discovery. The resolution is
`resolve_jlens`, and `resolve_jlens_layer` picks the layer; both live in
`interpretune.analysis.optools`, beside `resolve_unembed_and_norm_scale` and
`jlens_direction_rows`, the one construction every J-lens direction goes through.

Layers are dense model layers; lenses exist only for the fitted subset, so a
layer is either named explicitly (refused when unfitted) or selected
as a percentile over the fitted set. 85% depth is the default for every J-lens
op: late enough for naming tasks, below the final layers.

## Lens kinds: the J-lens and J++

Two kinds of lens share one file format: the J-lens (`jlens`, the default) and the
[J++ lens](https://github.com/safety-research/jpp_lens) (`jpp`). Both are one fitted map per layer, read
through the final norm and the unembedding in the same way. They differ only in how the maps were fitted, so
the tensors cannot tell you which one you hold. Pass `jlens_kind="jpp"` to any J-lens op (as a keyword or on
the analysis batch) to resolve a J++ lens. The resolver looks in the repository's `jpp/` directories and
checks the artifact's sidecar, refusing a lens whose declared fit method contradicts the kind you
asked for. `neuronpedia/jacobian-lens` publishes J++ lenses for some of its models, among them Qwen3-4B and
the Qwen3.5 family, but not for every model that has a J-lens: gemma-3-1b-it has none there.

`jlens_read` defaults its readout to the lens kind, because the two kinds' reference readouts differ:

| setting | J-lens default | J++ default | what it does |
| --- | --- | --- | --- |
| `jlens_readout_mask` | `none` | `semantic` | `semantic` ranks only tokens whose decoded string has a letter or digit |
| `jlens_include_rms_scale` | `False` | `True` | divides by the final norm's RMS, as the norm itself does |

Either can be overridden, and the result records the settings it used. The RMS divisor never changes the
ranking within one position. It matters for magnitudes: a J++ map carries an arbitrary per-layer scale, so raw
J++ scores are not comparable across layers or lenses, and turning the divisor off with a J++ lens warns.
Compare readout magnitudes across layers or lenses only with the divisor on, and never compare raw scores
between a J-lens and a J++ lens.

**Which kind to use, as measured here.** We fitted J++ lenses with the post's code for two models below its
9B to 284B range and validated them against the published J-lenses:

- **Readouts favour J++.** Held-out macro recall@10, scored the way the post scores it: gemma-3-1b-it 0.302
  for the J-lens against 0.388 for J++, almost all of the gain on the typo task
  ([results](https://github.com/speediedan/interpretune/issues/784#issuecomment-6618055568)); Qwen3.5-4B 0.340 against 0.457, with about 60% of the gain
  surviving when each task is held out of the weight fit ([results](https://github.com/speediedan/interpretune/issues/925#issuecomment-6715805780)). On
  Qwen3.5-4B the held-out J++ readout falls below the J-lens in the last quarter of layers and at layers 3
  and 4, so use it for single-token readouts at layers 5 to 22.
- **Swaps and steering favour the J-lens.** On the two-hop swap battery the J-lens swaps better than J++ on
  both models, significantly at mid-depth bands ([gemma-3-1b-it](https://github.com/speediedan/interpretune/issues/784#issuecomment-6623062231),
  [Qwen3.5-4B](https://github.com/speediedan/interpretune/issues/925#issuecomment-6729102253)). On the post's own `probe-swap` runner Qwen3.5-4B's J++ is
  level with the J-lens. Build steering and swap directions from a J-lens.

Those two models are the evidence. The post reports swaps at least as strong as the J-lens's on one model at
the top of its size range, and the J++ lenses on `neuronpedia/jacobian-lens` (Qwen3-4B among them) were
fitted separately and have not been validated here. On any other model, measure before choosing a kind for
steering.

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
(`jlens_unfolded`) while the readout direction wants the norm-folded one
(`jlens_folded`), and silently picking either would corrupt the other
use. `embed` and `store` remain for the non-lens bases.

## Steering and the folding default

The intervention modes (`patch`, `clamp`, `reject` and the rest) are
basis-agnostic: they apply whatever directions they are given, so the basis is
chosen where the directions are built. `jlens_concept_probe` and
`jlens_sparse_inventory` fold by default (`jlens_apply_final_norm`);
`concept_direction` makes you choose, and folded is the recommended choice.
Build the directions from a J-lens rather than a J++ lens (see "Lens kinds" above); how folding applies to J++
maps is in the folding page's J++ section.
Folding aims the edit into the coordinates the final norm amplifies, so it
helps exactly where the model stores the task contrast: essential on
gemma-3-1b-it (unfolded vectors never flip at any layer), ~3x weaker at late
layers on gemma-2-2b (both forms flip). Which side of that line a new model
falls on is measured, not predicted: the decision rule's first-order check
compares the two bases before you commit to one. Go through
`resolve_unembed_and_norm_scale` rather than hand-rolling either side: the
scale convention differs per model family, and a mismatched fold is wrong in
a way no magnitude adjustment repairs. A site after the final norm, such as the
unembed's input, is fold-independent; see "Sites after the final norm" in the
folding page.
