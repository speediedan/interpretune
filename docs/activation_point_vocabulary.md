# The activation-point vocabulary

Written for a reader outside this project. Interpretune, TransformerLens, interp-engine, SAEDashboard and
circuit-tracer each name the same tensors differently, and every pair that talks carries its own translation.
This document states one vocabulary with two levels of name and one level of resolution, the data schema that
carries the architecture-specific part, and the reconciliation it would ask of interp-engine. It is written so it
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
| `unembed_in` *(added as a name)* | the final norm's output, the unembed's input | `final_norm` | `unembed.hook_in` |
| `logits` *(added)* | the output distribution | `lm_head` | `unembed.hook_out` |

**Component points** say *where* a tensor is in a specific architecture's module tree, spelled in the
TransformerBridge grammar: `blocks.{i}.{component}.hook_{slot}` inside the block stack, `{component}.hook_{slot}`
outside it. Every component has `hook_in` and `hook_out`; a norm additionally has `hook_normalized` (the
normalized tensor before the learned gain) and `hook_scale` (the per-token denominator). A norm is therefore
three tensors, and no two of them are aliases.

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
(interp-engine's unmapped-hook marker, interpretune's unmappable-hook reasons), which is the signal that it belongs
in the schema. A consumer enumerating support renders it; a consumer that needs the tensor raises with the reason.

## The component-map schema

One short document per architecture. The slot rule is per **kind**, not per row, which is what collapses a table
of hooks into a dozen lines and keeps every alias consistent with the point it names.

```yaml
architecture: GPT2LMHeadModel            # the model class name, today; other keys later
facts: {sandwich_norms: false}
components:                              # component path -> module path template + kind
  embed:              {module: "transformer.wte",               kind: embed}
  pos_embed:          {module: "transformer.wpe",               kind: embed}
  blocks.{i}:         {module: "transformer.h.{i}",             kind: block}
  blocks.{i}.ln1:     {module: "transformer.h.{i}.ln_1",        kind: norm}
  blocks.{i}.attn:    {module: "transformer.h.{i}.attn",        kind: attn}
  blocks.{i}.attn.o:  {module: "transformer.h.{i}.attn.c_proj", kind: linear}
  blocks.{i}.ln2:     {module: "transformer.h.{i}.ln_2",        kind: norm}
  blocks.{i}.mlp:     {module: "transformer.h.{i}.mlp",         kind: mlp}
  blocks.{i}.mlp.in:  {module: "transformer.h.{i}.mlp.c_fc",    kind: linear}
  blocks.{i}.mlp.out: {module: "transformer.h.{i}.mlp.c_proj",  kind: linear}
  ln_final:           {module: "transformer.ln_f",              kind: norm}
  unembed:            {module: "lm_head",                       kind: unembed}
```

Kinds: `block`, `attn`, `mlp`, `norm`, `linear`, `embed`, `unembed`. A sandwich-norm architecture adds
`ln1_post` and `ln2_post` rows and sets `facts.sandwich_norms`, which is what routes the contribution points.

Sources, in precedence order: a bundled document (five architectures ship with interpretune), a document pulled
as a `hookmaps` hub component (data, no trust gate; `it.hub.pull_hookmaps("<org>/<repo>")` fetches and registers,
`it.hub.load_hookmaps` registers from the cache), and a map derived from TransformerLens' own per-architecture
component mapping. Two sources for one architecture must agree, and a test says so. The TransformerLens-derived
map is the independent oracle the bundled documents are checked against, because a convergence test that resolves
both sides through the same resolver cannot see a wrong row.

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

## Legacy names

The legacy `HookedTransformer` vocabulary is deprecated except for the semantic names above. Deprecated spellings
are served through an explicit alias table (alias, canonical, level, deprecated-since, replacement, optional
caution); a parse reports the canonical spelling and the alias it came from; a strict mode refuses deprecated
aliases; adapters may register aliases for their own vocabulary rather than carrying private tables.
