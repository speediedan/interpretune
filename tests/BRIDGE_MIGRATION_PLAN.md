# SAE Test Bridge Migration & Standalone Test Optimization Plan

**Session:** S10 — Bridge Migration, Standalone→Main Migration, Parity Optimization
**Date:** 2026-03-05
**Baseline:** 969 passed, 55 skipped, 0 failed (post-S9)

---

## Task 1: Bridge Migration for HT-Only SAE Tests

### Empirical Findings

Tested `SAETransformerBridge` with `enable_compatibility_mode(no_processing=True)` for GPT-2:

| Property | HookedTransformer | TransformerBridge |
|---|---|---|
| `sae.cfg.metadata.hook_name` | `blocks.0.hook_resid_pre` | `blocks.0.hook_resid_pre` |
| Resolved hook name | `blocks.0.hook_resid_pre` (identity) | `blocks.0.hook_in` |
| SAE registry key | `blocks.0.hook_resid_pre.hook_sae_acts_post` | `blocks.0.hook_in.hook_sae_acts_post` |
| Cache key for SAE hooks | `blocks.0.hook_resid_pre.hook_sae_acts_post` | `blocks.0.hook_in.hook_sae_acts_post` |

**Key finding:** Bridge resolves `blocks.0.hook_resid_pre` → `blocks.0.hook_in` via `_resolve_hook_name()`.
SAE internal hooks are registered under the **resolved** name, not the alias. This affects cache keys
and `run_with_hooks` targeting.

### Impact on Each Test

| Test | Issue | Fix Required |
|---|---|---|
| `test_run_with_saes_with_cache_fwd_bwd` | Cache assertion uses alias name → fails | Resolve base name before assertion |
| `test_run_with_cache_with_saes` | Same cache key mismatch | Same fix |
| `test_run_with_hooks_with_saes` | Hook targeting uses alias → hook doesn't fire | Resolve base name for fwd_hooks + assertion |
| `test_sl_module_warns` | No hook name issue — just needs Bridge fixture variant | Add Bridge parametrization |

### Solution: `_sae_hook_name` Helper

Add a small helper that resolves the SAE hook base name depending on the model type:

```python
def _sae_hook_name(model, sae_handle, internal: str = "hook_sae_acts_post") -> str:
    """Get the full hook name for an SAE's internal hook (works for both HT and Bridge)."""
    base = sae_handle.cfg.metadata.hook_name
    if hasattr(model, '_resolve_hook_name'):
        base = model._resolve_hook_name(base)
    return f"{base}.{internal}"
```

Then update:
- **core_l_run_w_pytest_cfg** → **core_l_bridge_run_w_pytest_cfg** for all 3 cache/hook tests
- Use `_sae_hook_name()` in assertions and hook targeting
- Add `sl_br_gpt2_w_ref_logits` to the parametrization

### test_sl_module_warns

This test verifies a warning when `sae_cfgs` is ablated. It currently uses
`get_it_session__l_sl_ht_gpt2__initonly` (requires Lightning). The test logic doesn't
depend on hook names — it only checks that the warning is raised. Migration plan:
- Change from direct fixture to parameterized, adding both HT and Bridge variants

---

## Task 2: Standalone → Main Collection Migration

### Current Standalone Tests (25 total)

| Category | Count | Tests | Migration Feasibility |
|---|---|---|---|
| Circuit Tracer Gemma2 (TL) | 7 | 5 init + 1 functionality + 1 lightning | **LOW** — Gemma2 ~5GB GPU + transcoder set |
| Circuit Tracer Gemma2 (NNsight) | 4 | 2 init + 2 functionality | **LOW** — Same memory constraints |
| NNsight Trace | 1 | `test_nnsight_trace_context` | **NONE** — sys.settrace() conflicts with coverage |
| TL Param Mapping | 2 | llama3, gemma2 | **LOW** — Large models (3B+), bf16_cuda required |
| Debug Generation | 2 | gemma3, llama3 | **LOW** — Large models, bf16_cuda + lightning |
| Notebooks | 6 | 3 categories × 2 | **NONE** — Integration tests, bf16_cuda |
| Parity FTS | 4 | FTS-dependent | **NONE** — Require FTS standalone env |

### Assessment

Most standalone tests are standalone due to **large model memory requirements** (Gemma2 ~5GB,
Llama3 ~7GB) or **framework conflicts** (NNsight sys.settrace, FTS process isolation).
The `cleanup_cuda` fixture helps with post-test cleanup but doesn't solve the core issue:
these models exhaust memory when loaded alongside the rest of the test suite's session-scoped fixtures.

**Recommendation:** No additional standalone→main migrations in this session beyond what S9
already achieved (4 tests moved). The memory-constrained tests genuinely need isolation.

---

## Task 3: Parity Test Optimization

### Current State

67 parity tests across 5 modules, each creating independent ITSession via `parity_test()` →
`config_modules()`. The parity test pattern deliberately creates fresh sessions per test to
validate the full initialization→execution pipeline, which IS the point of parity testing.

### Optimization Opportunities

Parity tests validate end-to-end behavior and are intentionally heavyweight. Significant
refactoring would change what they test. Minor improvements possible:

1. **No-op for this session** — The parity test architecture is designed for independent
   validation. Shared fixtures would change test semantics (each test validates full lifecycle).

---

## Implementation Order

1. Add `_sae_hook_name` helper to test file
2. Update `core_l_run_w_pytest_cfg` → `core_l_bridge_run_w_pytest_cfg` for 3 cache/hook tests
3. Update assertions/hook targeting in 3 tests to use `_sae_hook_name()`
4. Parametrize `test_sl_module_warns` to include Bridge
5. Update NOTE comment explaining the change
6. Run CI validation

---

## Completion Status (S10)

### Implementation Results

All tasks executed successfully:

| Step | Status | Details |
|---|---|---|
| `_sae_hook_name` helper | ✅ Done | Lines 41-53 of test_adapters_sae_lens.py |
| Parametrization switch (3 tests) | ✅ Done | Removed `core_l_run_w_pytest_cfg`, all 3 tests use `core_l_bridge_run_w_pytest_cfg` |
| Assertion/hook targeting updates | ✅ Done | 2 cache assertions + 1 fwd_hooks targeting use `_sae_hook_name()` |
| `test_sl_module_warns` Bridge variant | ✅ Done | Parametrized with Lightning HT + Core Bridge |
| NOTE comment update | ✅ Done | Updated to explain Bridge compatibility via `_sae_hook_name()` |
| CI validation | ✅ Done | **973 passed, 55 skipped, 0 failed** (23min) |

### Net Changes

- **+5 new Bridge test cases**: `test_run_with_saes_with_cache_fwd_bwd[core_bridge]`, `test_run_with_cache_with_saes[core_bridge]`, `test_run_with_hooks_with_saes[core_bridge]`, `test_run_with_saes[core_bridge]` (already existed), `test_sl_module_warns[core_bridge]`
- **-1 HT-only test case**: `test_sl_module_warns` (was Lightning HT only, now parametrized)
- **Net +4 tests** vs S9 baseline (969 → 973 passed)
- **Removed**: `core_l_run_w_pytest_cfg` (HT-only parametrization, no longer needed)

### SAE Lens Test Suite

51/51 tests passed in 83.32s, including all Bridge variants.
