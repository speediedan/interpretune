# The activation-point vocabulary

Written for a reader outside this project. Interpretune, TransformerLens, interp-engine, SAEDashboard and
circuit-tracer each name the same tensors differently, and every pair that talks carries its own translation.
This document states one vocabulary with two levels of name and one level of resolution, the component maps that
carry the architecture-specific part, and the reconciliation it would ask of interp-engine. It is written so it
can be lifted into a shared specification if the projects that read it agree.

## Two levels of name

**Semantic points** say what a tensor *is* in the forward pass and survive an architecture change. They are
interp-engine's point names, with three additions.

| semantic point | meaning | interp-engine | legacy TransformerLens spelling |
| --- | --- | --- | --- |
| `resid_pre` | the block's input | `resid_pre` | `blocks.{i}.hook_resid_pre` |
| `resid_mid` | the residual after the attention write, the second norm's input | `resid_mid` | `blocks.{i}.hook_resid_mid` |
| `resid_post` | the block's output | `resid_post` | `blocks.{i}.hook_resid_post` |
| `attn_in` | the attention sublayer's argument (the first norm's output) | `attn_in` | `blocks.{i}.attn.hook_in` |
| `mlp_in` | the MLP's argument (the second norm's output) | `mlp_in` | `blocks.{i}.mlp.hook_in` |
| `attn_out`, `mlp_out` | the raw sublayer outputs | `attn_out`, `mlp_out` | `blocks.{i}.attn.hook_out`, `blocks.{i}.mlp.hook_out` |
| `attn_contribution`, `mlp_contribution` *(added)* | what the sublayer adds to the residual: the post-norm output on a sandwich-norm model, the raw output otherwise | `attn_out_post` / `attn_out`, by architecture | `blocks.{i}.hook_attn_out`, `blocks.{i}.hook_mlp_out` |
| `z` | attention output before the output projection | `z` | `blocks.{i}.attn.o.hook_in` |
| `cross_attn_in`, `cross_attn_out` *(added)* | an encoder-decoder block's third sublayer: its argument, and its raw output | none | `blocks.{i}.hook_cross_attn_in`, `blocks.{i}.cross_attn.hook_out` |
| `cross_attn_contribution` *(added)* | what cross-attention adds to the residual, post-norm where one exists | none | `blocks.{i}.hook_cross_attn_out` |
| `resid_after_{k}` *(added)* | the residual after sublayer k's write (0-based); `resid_after_0` is `resid_mid` | none | `blocks.{i}.hook_resid_after_{k}` |
| `unembed_in` *(added as a name)* | the final norm's output, the unembed's input | `final_norm` | `unembed.hook_in` |
| `logits` *(added)* | the output distribution | `lm_head` | `unembed.hook_out` |

**Component points** say *where* a tensor is in a specific architecture's module tree, spelled in the
TransformerBridge grammar: `blocks.{i}.{component}.hook_{slot}` inside the block stack, `{component}.hook_{slot}`
outside it. Every component has `hook_in` and `hook_out`; a norm additionally has `hook_normalized` (the
normalized tensor before the learned gain) and `hook_scale` (the per-token denominator). A norm is therefore
three tensors, and no two of them are aliases.

A contribution is keyed by the sublayer's **kind name** and routes to that sublayer's post-norm by its position in the
document's `properties.sublayers` (sublayer k reads from `ln{k+1}_post` where the block has one), so a third sublayer
kind, cross-attention or a declared mixer, needs no new vocabulary machinery.

Both levels parse to one record, `ActivationPoint(component, slot, layer, contribution, subhook, caution)`, whose
string form is the component spelling. The `hook_resid_*` and contribution names are semantic, not legacy; two
legacy names, `hook_mlp_in` and `hook_attn_in`, parse with a **caution** because they name the residual *before*
the block norm and are routinely read as the sublayer's post-norm argument, one whole normalization away.

## One level of resolution

A point resolves against a **component map** to a tensor position, or to a stated refusal:

```
TensorRef(module_path, io, tuple_output, derivation)      a PyTorch module's input or output; `derivation` set for the
                                                          two norm intermediates, which no module emits
Unresolvable(reason, alternatives)                        a value, never an exception path
```

Refusal as a value is deliberate: two implementations grew a sentinel for "does not translate" independently
(interp-engine's unmapped-hook marker, Interpretune's unmappable-hook reasons), which is the signal that it belongs
in the schema. A consumer enumerating support renders it; a consumer that needs the tensor raises with the reason.

## Component maps

A point resolves through the **component map** of its model's architecture: one short document naming, for each
component in the bridge grammar (`blocks.{i}.ln2`, `blocks.{i}.attn.o`, `unembed`), the PyTorch module it wraps
(with `{i}` for the layer) and its **kind**. The slot rule is per kind, not per row, which is what collapses a table
of hooks into a dozen lines and keeps every alias consistent with the point it names.

The maps are venngram's. Their schema, its reader and its evolution policy are documented with venngram's maps
([the maps overview](https://github.com/speediedan/venngram#maps-base-and-overlay) and the schema module,
[`venngram.maps.schema`](https://github.com/speediedan/venngram/blob/main/src/venngram/maps/schema.py)), and this page
does not restate them. Interpretune bundles no map and writes none: `interpretune.analysis.points.component_map_for`
reads `venngram.maps.component_map_for`, the same lookup a tunable view resolves through, so one architecture has
one map whatever reads it. That lookup takes, in order, a map registered in the running process
(`venngram.maps.register`, or a pulled `hookmaps` component, below), then the best row the configured registries
index ([`speediedan/venngram-registry`](https://huggingface.co/datasets/speediedan/venngram-registry) by default). The
offline path is venngram's cache: the last registry revision read is served with a warning when the registry cannot
be reached, and a registry never read before is refused. There is no bundled fallback.

Row names follow the registry. Qwen3.5's linear-attention output projection is `blocks.{i}.linear_attn.out`; the
old spelling `linear_attn.o` is refused with the new name in the message, and has no alias.

### Bound and unbound maps

A map may mark a block-relative row `optional`: it applies on the layers where its module exists. That is how a
hybrid stack is written, because which layers have full attention depends on the checkpoint's depth, not on its
architecture (Qwen3.5 puts full attention on every fourth layer of whatever depth it has). An optional row therefore
has no layer coverage until it is read against a model, and Interpretune keeps the two states apart:

- **Unbound** (`component_map_for`): every row without `optional` resolves as written. An optional row resolves to
  an `Unresolvable` whose reason says it *resolves only against a model*. It is never assumed to cover layer 0, which
  on a hybrid stack is a plausible module path that does not exist.
- **Bound** (`bind_component_map(cmap, model)`): the map is first checked against the model with venngram's check
  (every row's module exists on every layer it covers and on none it excludes, an optional row's on at least one
  layer, every derived property agrees with the model), and a map that is wrong about its model is refused there,
  listing every problem. Each optional row then becomes the list of layers its module was found on, which the
  resolver reads like any other per-row layer list. A point on a layer the row does not cover is refused naming
  the layers it does.

The backends bind where they hold the model: the nnsight backend binds its hook resolver before its first trace
(`HookNameResolver.bind`), the TransformerLens backend binds to the HF model its bridge wraps when it declares what
it can capture, and the conformance suite's HF reference binds to the model it loads. A table derived from a map is
cached per bound map, so two checkpoints of one architecture at different depths never share one. Consumers without
a model (the capture inventory, the conformance cases) read the unbound map; the inventory of base spellings is
layer-free, so it is the same bound or not, and resolving an optional point is where the refusal appears.

**Supplying a map.** A `hookmaps` hub component carries maps as data (no trust gate):
`it.hub.pull_hookmaps("<org>/<repo>")` fetches and registers, `it.hub.load_hookmaps` registers from the cache. Its
documents are read through venngram's reader, so any schema venngram reads is accepted, and they are registered
with `venngram.maps.register`, ahead of the registries. A map for an architecture that already has one must agree
with it, per row on module, kind, `optional` and layers, and on the properties, declared kinds and stacks, or the
load is refused naming both sources; `replace=True` shadows the served map deliberately, and an identical map is a
no-op. A cached third-party map therefore never silently displaces the registry's, and never silently loses to it.

## Entering and leaving the supported set

**Entering.** An architecture enters through venngram's process: a map with a verified row in venngram's registry,
verified there against the architecture without weights. Interpretune's suite then holds it to two checks that make
support measured rather than asserted. A tiny builder for its transformers class binds the map to the class on every
run, and the oracle comparison against TransformerLens' adapter for the architecture, built from the same tiny config
with no weights and no Hub call, compares every row both sides name. Rows TransformerLens has no counterpart for are
listed with the reason: the multimodal maps' vision stacks and projectors, which TransformerLens bridges as one opaque
component, and Qwen3.5's linear-attention submodules, which its gated delta net bridge does not expose. A supported
architecture without a verified registry row at the CI warm generation fails the suite, naming it, so a registry
regression is a named failure rather than a fallback.

**Leaving.** An architecture leaves through the registry's deprecation: its map gains `deprecated_since` and
`replacement`, keeps loading with a warning, and strict lookup refuses it. A map pulled as a `hookmaps` component
for an architecture the registry no longer serves becomes the source for it with nothing to conflict against.

## What this asks of interp-engine

Four reconciliations, two of them already measured and drafted as issues:

| item | today | asked |
| --- | --- | --- |
| `blocks.{i}.hook_mlp_in`, `blocks.{i}.hook_attn_in` | mapped to the post-norm `mlp_in` / `attn_in` | refuse, naming `mlp.hook_in` / `attn.hook_in` for the sublayer argument and `hook_resid_mid` / `hook_resid_pre` for the pre-norm residual (the tensors differ by a whole normalization: cos 0.088 on gemma-3-1b-it layer 5) |
| `blocks.{i}.hook_in`, `unembed.hook_in` | unmapped ("no canonical point") | map to `resid_pre.{i}` and `final_norm` (identical tensors, measured at 0.0) |
| `lm_head` | listed as unmapped to TransformerLens | map to `unembed.hook_out` (and `final_norm` to `unembed.hook_in`), so the output distribution has a name on both sides |
| `_POINT_TO_TLENS["attn_in"]` | emits the pre-norm block hook | emit `attn.hook_in`, as `mlp_in` already does |

And one of TransformerLens: `docs/source/content/model_structure.md` states that a norm's `hook_normalized` and
`hook_scale` are aliases of `hook_out`, and that `hook_mlp_in` aliases two differently shaped tensors; the
implementation fires three distinct tensors. That paragraph is the plausible common ancestor of the same defect in
several downstream tables.

## TransformerLens aliases

TransformerLens spellings that are neither the semantic names above nor component points (`attn.hook_z`,
`hook_q`, `hook_embed`) are served through an explicit alias table (alias, canonical, level, an optional
deprecated-since, replacement, optional caution). They are accepted, not deprecated: TransformerLens's own bridge
carries them as current vocabulary, so the only thing Interpretune expresses about them is a preference for
canonical names. A parse reports the canonical spelling and the alias it came from; a strict mode refuses aliases
as a matter of style; adapters may register aliases for their own vocabulary rather than carrying private tables,
and mark one deprecated only when they are actually retiring it.
