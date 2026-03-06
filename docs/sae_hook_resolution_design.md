# SAE Hook Name Resolution Design

## Problem Statement

When SAEs are attached to a `TransformerBridge` model, their internal hooks
(e.g., `hook_sae_acts_post`, `hook_sae_input`) are registered at **canonical**
paths, but `sae.cfg.metadata.hook_name` retains the original **alias** name.
This creates a mismatch when code needs to look up SAE hooks in the cache or
target them with `fwd_hooks`.

### Concrete Example

Given an SAE configured with `hook_name="blocks.0.hook_resid_pre"`:

| Context | Hook Name |
|---------|-----------|
| `sae.cfg.metadata.hook_name` | `blocks.0.hook_resid_pre` (alias) |
| Actual hook registration in `add_sae()` | `blocks.0.hook_in` (canonical) |
| `hook_dict` compound key | `blocks.0.hook_in.hook_sae_acts_post` |
| Cache key after `run_with_cache_with_saes()` | `blocks.0.hook_in.hook_sae_acts_post` |

So `f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"` yields
`blocks.0.hook_resid_pre.hook_sae_acts_post` — which does **not** exist in
the cache or `hook_dict`.

### Why This Matters

- **HookedSAETransformer (HT)** doesn't have this problem because HT uses the
  alias names natively — there's no alias→canonical resolution step.
- **SAETransformerBridge** resolves aliases during `add_sae()` but doesn't
  expose the resolved names, forcing callers to replicate the resolution logic.
- **Current workaround**: A `_sae_hook_name()` test helper in interpretune's
  `test_adapters_sae_lens.py` that calls `model._resolve_hook_name()` directly.
  This leaks Bridge internals into tests and doesn't help non-test code.

## Approaches Evaluated

### Approach (a): Bridge Compatibility Mode + SAELensAttributeMixin Property

Add a `resolved_sae_hook_name()` method to interpretune's `SAELensAttributeMixin`
that checks for `_resolve_hook_name()` on the model.

- **Pro**: No upstream changes needed.
- **Con**: Couples interpretune's adapter layer to Bridge internals. Other SAELens
  consumers face the same problem. Doesn't fix the root cause.

### Approach (b): Upstream Fix in SAELens — `get_sae_hook_name()` on Bridge

Add a public method to `SAETransformerBridge` that exposes the resolved compound
hook name for any attached SAE's internal hook.

- **Pro**: Fixes the root cause at the right layer. Benefits all SAELens consumers.
  The Bridge already has the alias→canonical mapping — it just doesn't expose it.
- **Con**: Requires an SAELens PR.

### Approach (c): Generalize HookNameResolver in Interpretune

Extend interpretune's `HookNameResolver` (which already maps TL hook names to
HF module paths for NNsight) to also handle TL alias→canonical resolution.

- **Pro**: Provides a unified cross-backend hook resolution interface.
- **Con**: Conflates two different resolution concerns (TL↔HF paths vs TL
  alias→canonical). HookNameResolver is for cross-backend mapping; alias
  resolution is an intra-TL concern.

## Recommended Approach: (b) + (c) as Vision

### Layer 1 — SAELens Upstream (Immediate)

Add `get_sae_hook_name()` to `SAETransformerBridge`:

```python
def get_sae_hook_name(self, sae: SAE, internal: str = "hook_sae_acts_post") -> str:
    """Get the full resolved hook name for an SAE's internal hook point.

    Since SAETransformerBridge resolves alias names (e.g., 'blocks.0.hook_resid_pre')
    to canonical names (e.g., 'blocks.0.hook_in') during add_sae(), the actual
    hook registration path may differ from sae.cfg.metadata.hook_name.

    This method returns the resolved compound name as it appears in hook_dict
    and cache (e.g., 'blocks.0.hook_in.hook_sae_acts_post').

    Args:
        sae: The SAE whose hook name to resolve.
        internal: The SAE internal hook suffix (default: 'hook_sae_acts_post').

    Returns:
        The fully-resolved compound hook name.
    """
    base = sae.cfg.metadata.hook_name
    resolved = self._resolve_hook_name(base)
    return f"{resolved}.{internal}"
```

**Rationale**: The Bridge owns the alias→canonical mapping. Exposing it via a
public method is minimal, non-breaking, and serves all SAELens consumers.

### Layer 2 — Unified Vision (Future)

Extend interpretune's `HookNameResolver` to consume the Bridge's resolution
API as part of a unified cross-backend hook resolution interface. The resolver
would handle:

1. **TL alias→canonical** (Bridge): `blocks.0.hook_resid_pre` → `blocks.0.hook_in`
2. **TL canonical→HF module** (NNsight): `blocks.0.hook_in` → `transformer.h.0` envoy
3. **SAE compound names**: `blocks.0.hook_resid_pre.hook_sae_acts_post` →
   `blocks.0.hook_in.hook_sae_acts_post` (Bridge) or appropriate NNsight path

This provides a single interface that works across `TransformerBridge`,
`HookedTransformer`, `NNsight`, and `circuit_tracer` backends.

## Implementation Plan

### Phase 1: SAELens PR (This Session)

1. Add `get_sae_hook_name()` to `SAETransformerBridge` in
   `sae_lens/analysis/sae_transformer_bridge.py`
2. Add unit test in SAELens
3. Submit PR with existing uncommitted Bridge changes

### Phase 2: Interpretune Test Cleanup (This Session)

1. Replace `_sae_hook_name()` test helper with calls to `model.get_sae_hook_name()`
2. Run CI to validate

### Phase 3: Unified Hook Resolution (Future)

1. Extend `HookNameResolver` with alias resolution capability
2. Document cross-backend hook mapping in `transformer_bridge_architecture.md`
3. Integrate with analysis dispatcher

## Hook Access Patterns Across Backends

### HookedTransformer (Legacy)

```
SAE hook_name: "blocks.0.hook_resid_pre"         (alias = actual)
Cache key:     "blocks.0.hook_resid_pre.hook_sae_acts_post"
hook_dict key: "blocks.0.hook_resid_pre.hook_sae_acts_post"
```

No resolution needed — HT uses alias names natively.

### TransformerBridge

```
SAE hook_name: "blocks.0.hook_resid_pre"         (alias, from metadata)
Resolved:      "blocks.0.hook_in"                 (canonical, after _resolve_hook_name)
Cache key:     "blocks.0.hook_in.hook_sae_acts_post"
hook_dict key: "blocks.0.hook_in.hook_sae_acts_post"
```

Resolution needed — Bridge stores hooks at canonical paths.

### NNsight

NNsight doesn't use TL hook names directly. SAE hooks are accessed via
the NNsight envoy system:

```
HF module path:  "transformer.h.0"               (NNsight envoy)
TL equivalent:   "blocks.0.hook_resid_pre"        (mapped by HookNameResolver)
```

SAE attachment in NNsight follows a different path (direct model intervention
via `nnsight.trace()`), so the hook name resolution concern is primarily
TL-specific.

### Circuit Tracer

Circuit tracer uses either TL or NNsight as backend. Hook resolution
delegates to the chosen backend's conventions.

## Key Files

| File | Role |
|------|------|
| `sae_lens/analysis/sae_transformer_bridge.py` | `SAETransformerBridge` — alias resolution, SAE registration |
| `sae_lens/saes/sae.py` | `SAEMetadata` — stores `hook_name` as plain attribute |
| `interpretune/analysis/backends/hook_mapping.py` | `HookNameResolver` — TL↔HF path mapping |
| `interpretune/adapters/sae_lens.py` | `SAELensAttributeMixin` — IT adapter for SAE properties |
| `tests/core/test_adapters_sae_lens.py` | Current `_sae_hook_name()` workaround location |
