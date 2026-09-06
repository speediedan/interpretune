# Intervention Hook Pattern Support

Hook names are parsed once, at the boundary, by `interpretune.analysis.points.parse`, and everything downstream
works with the parsed point rather than the string. This document is the caller-facing summary of that vocabulary
as it applies to `names_filter` (capture) and intervention patterns; the protocol itself, with the component-map
schema and the interp-engine reconciliation, is [`activation_point_vocabulary.md`](activation_point_vocabulary.md).

## Two levels of name

- **Semantic points** say what a tensor IS in the forward and survive an architecture change: `hook_resid_pre`,
  `hook_resid_mid`, `hook_resid_post`, `hook_attn_out`, `hook_mlp_out`. They are not legacy; they are what analysis
  code should ask for. `hook_attn_out` and `hook_mlp_out` are the sublayer's *contribution* to the residual: the
  raw module output on a pre-norm model and the post-norm output on a sandwich-norm model, exactly as
  TransformerLens fires them.
- **Component points** say WHERE a tensor is, in the TransformerBridge grammar: `blocks.{i}.ln2.hook_out`,
  `blocks.{i}.attn.o.hook_in`, `unembed.hook_out` (the output distribution), `ln_final.hook_normalized`. They are
  canonical by construction and resolve against a per-architecture component map
  (`interpretune/analysis/points/data/*.yaml`) to a tensor position, or to an `Unresolvable` carrying the reason.

Two semantic names parse with a **caution**: `hook_mlp_in` and `hook_attn_in` name the residual BEFORE the block
norm (they are the same tensor as `hook_resid_mid` / `hook_resid_pre`), one norm away from the sublayer's actual
argument (`mlp.hook_in`, `attn.hook_in`, which the vocabulary addresses at the norm's output). Consumers surface the
caution; whether to refuse the name is their policy.

## Deprecated spellings: the alias table

Legacy `HookedTransformer` names that are neither semantic nor component points are served through ONE table,
`interpretune.analysis.points.vocabulary.ALIASES`. An alias asserts only that a spelling MEANS a point; whether two
spellings name the same tensor is answered by resolution, per architecture, never by the table.

| deprecated spelling | means | level |
|---|---|---|
| `attn.hook_z` | `attn.o.hook_in` | component |
| `hook_q_input` / `hook_k_input` / `hook_v_input` | `attn.q.hook_in` / `attn.k.hook_in` / `attn.v.hook_in` | component |
| `hook_q` / `hook_k` / `hook_v` | `attn.q.hook_out` / `attn.k.hook_out` / `attn.v.hook_out` | component |
| `mlp.hook_pre` | `mlp.in.hook_out` (the up-projection's output, the pre-activation) | component |
| `mlp.hook_post` | `mlp.out.hook_in` | component |
| `hook_embed` / `hook_pos_embed` | `embed.hook_out` / `pos_embed.hook_out` | component |

`parse(name)` returns the canonical point with `alias` set to the spelling used, so a linter can say "you wrote X,
this means Y"; `parse(name, strict=True)` refuses every alias with `DeprecatedPointError` naming the replacement,
for configs that want to be canonical. Adapters register their own deprecated spellings with `register_alias`; a
registration that would shadow a semantic point, a component spelling or another alias is refused.

## What replaced the alias groups

Earlier versions carried `HOOK_ALIAS_GROUPS`, sets of names treated as one intervention site. Two groups asserted
identities that measurement refuted, so the mechanism is gone:

- `mlp.hook_in` was grouped with `hook_mlp_in`; they are one norm apart (cos 0.088 on gemma-3-1b-it layer 5).
- `attn.hook_out`, `hook_attn_out` and `hook_resid_mid` were one group. Measured with plain PyTorch hooks on the
  HF module at layer 5: on GPT-2, `attn.hook_out` and `hook_attn_out` are byte-identical while `hook_resid_mid` is
  a different tensor (cos 0.19); on Gemma-2, all three differ (`attn.hook_out` vs `hook_attn_out` cos 0.57,
  either vs `hook_resid_mid` cos 0.14 and 0.20). `hook_resid_mid` is `blocks.{i}.ln2.hook_in`; `hook_attn_out` is
  the contribution point and resolves to the post-attention norm's output where one exists; `attn.hook_out` is
  the raw module output.

Pattern expansion (`expand_intervention_patterns`) and TransformerLens capture (`names_filter`) now try every
**spelling of the same point** against the hooks the model exposes: the name as written, its canonical component
form, the semantic names of the same slot, and the registered aliases of each. A contribution point offers only
its own spelling, because the module it lives in depends on the architecture; a cautioned name is matched only
when written; a spelling the vocabulary does not know (`attn.hook_pattern`) is tried literally and nothing else.

## Wildcards

A layer wildcard is expanded after spelling normalization, so `blocks.*.hook_in`, `blocks.*.hook_resid_pre`,
`blocks.*.attn.o.hook_in` and `blocks.*.attn.hook_z` all match the same hooks. Any other wildcard placement is
matched literally against the model's hook names.

## Practical guidance

- Ask for semantic points when you mean a tensor's role (`hook_resid_pre`, `hook_mlp_out`) and component points
  when you mean a specific module (`blocks.5.ln2.hook_out`, `unembed.hook_in`).
- Write canonical spellings in new configs and validate with `parse(name, strict=True)`; keep a deprecated alias
  only while an older notebook is being migrated.
- A backend that cannot honour a point refuses by name (`Unresolvable`, or the backend's own error). Nothing is
  narrowed, widened or substituted for you, because every substitution produces plausible activations.
