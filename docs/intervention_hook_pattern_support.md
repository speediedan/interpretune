# Intervention Hook Pattern Support

> Working contract: this document records the hook-pattern subset that Interpretune currently treats as portable
> across intervention surfaces. It is not a claim that every TransformerLens v3 hook name already has matching
> cross-backend coverage in Interpretune.
>
> Test coverage note: we added focused regression coverage for the alias paths that recently broke
> (`hook_in` / `hook_out`, `attn.o.hook_in`, and constrained missing-feature retention), but we still need a broader
> pattern-by-pattern test sweep in a later pass.

## Resolution layers

Interpretune currently resolves intervention hook names through two layers:

1. `expand_intervention_patterns(...)` expands preferred canonical TransformerBridge-style hook names and supported
   legacy HookedTransformer aliases into the concrete hook names exposed by the active backend.
2. Backend-specific resolution then takes over:
   - TransformerLens / TransformerBridge uses the model hook registry plus its alias registry.
   - NNsight uses `src/interpretune/analysis/backends/hook_mapping.py`, which intentionally supports a smaller,
     explicitly curated portable subset.

For new configs, prefer the canonical names in the table below. Keep legacy aliases only when preserving older
notebooks or configs.

## Preferred portable patterns

| Preferred canonical pattern | Legacy aliases accepted | Current portable notes |
|---|---|---|
| `blocks.{i}.hook_in` | `blocks.{i}.hook_resid_pre` | Supported by TransformerLens and the NNsight resolver subset. Prefer this spelling in new configs. |
| `blocks.{i}.hook_out` | `blocks.{i}.hook_resid_post` | Supported by TransformerLens and the NNsight resolver subset. |
| `blocks.{i}.attn.hook_out` | `blocks.{i}.hook_attn_out`, `blocks.{i}.hook_resid_mid` | Use this as the preferred module-output spelling. `hook_resid_mid` remains accepted for backwards compatibility, but it should be treated as a legacy name. |
| `blocks.{i}.attn.o.hook_in` | `blocks.{i}.attn.hook_z` | Preferred cross-backend spelling for the attention output-projection input. |
| `blocks.{i}.mlp.hook_out` | `blocks.{i}.hook_mlp_out` | Portable today for GPT-2 and Llama-family NNsight mappings. Gemma-family NNsight flows still rely on the legacy `hook_mlp_out` path today. |
| `blocks.{i}.ln2.hook_out` | `blocks.{i}.ln2.hook_normalized`, `blocks.{i}.ln2.hook_scale` | Portable in the Gemma-family NNsight mappings; accepted by alias expansion elsewhere when the backend exposes the corresponding hook. |
| `unembed.hook_in` | none | Supported directly on TransformerBridge and NNsight models. **Legacy `HookedTransformer` models (e.g. the circuit-tracer TransformerLens backend) expose no `unembed.hook_in`** — use `ln_final.hook_normalized` (the pre-unembed input) there; adding a legacy alias expansion is tracked in [interpretune#223](https://github.com/speediedan/interpretune/issues/223). |

## Backend-specific alias families

`expand_intervention_patterns(...)` also tries supported canonical/legacy alias families for the following names when
the active backend exposes them:

- `embed.hook_out` ↔ `hook_embed`
- `pos_embed.hook_out` ↔ `hook_pos_embed`
- `attn.hook_in` ↔ `hook_attn_in`
- `attn.q.hook_in` / `attn.k.hook_in` / `attn.v.hook_in` ↔ `hook_q_input` / `hook_k_input` / `hook_v_input`
- `attn.q.hook_out` / `attn.k.hook_out` / `attn.v.hook_out` ↔ `hook_q` / `hook_k` / `hook_v`
- `attn.hook_pattern` ↔ `attn.hook_attention_weights`
- `attn.hook_hidden_states` ↔ `attn.hook_result`
- `mlp.hook_in` ↔ `hook_mlp_in`
- `ln1.hook_out` ↔ `ln1.hook_normalized` / `ln1.hook_scale`

Treat these as backend-specific until they are covered by the same explicit NNsight resolver subset and dedicated
tests.

## Wildcards

Wildcard patterns are expanded after alias normalization, so both canonical and legacy forms can be used with `*`.
Examples:

- `blocks.*.hook_in`
- `blocks.*.hook_out`
- `blocks.*.attn.hook_out`
- `blocks.*.attn.o.hook_in`

## Practical guidance

- Prefer canonical TransformerBridge-style names in notebook configs and explicit intervention mappings.
- Keep old HookedTransformer aliases only when updating older notebooks incrementally.
- When you need a hook outside the portable table above, treat it as backend-specific and validate it against the
  active backend before relying on cross-backend parity.
