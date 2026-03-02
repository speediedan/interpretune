# TransformerLens Codebase Changes

Upstream TransformerLens changes made as part of the interpretune
nnsight-support workstream.  All changes are on the interpretune fork
(`speediedan/TransformerLens`) and are intended for upstream contribution.

---

## 1. `_reconstruct_attention()` — Boolean 4D Attention Mask Fix

**File:** `transformer_lens/model_bridge/generalized_components/joint_qkv_attention.py`
**Class:** `JointQKVAttentionBridge`
**Method:** `_reconstruct_attention()` (lines 331–393)

### Problem

When HuggingFace models generate a 4D boolean attention mask (via
`_update_causal_mask()`), the original `_reconstruct_attention()` code:

1. **Applied a redundant `tril` causal mask on top of HF's 4D mask** — HF's
   mask already encodes causal masking; the extra `tril` was harmless for
   float masks but incorrect for boolean masks.
2. **Added boolean values directly to float attention scores** — boolean
   `True`/`False` were silently cast to `1.0`/`0.0` and added to scores,
   producing `scores + 1` for attend positions and `scores + 0` for masked
   positions.  This is the opposite of the intended semantics (masked
   positions should receive large negative values).

The combined effect: pad tokens leaked into attention computation and all
attend-position scores were shifted by +1.  While the +1 shift is small
relative to typical score magnitudes, the pad-token leakage compounds
through 12+ layers, producing answer-position logit divergence of ~85 units
vs native HF on GPT-2 with left-padded 178-token RTE inputs.

### Root Cause

HuggingFace's `_update_causal_mask()` returns different mask types depending
on model configuration:

- **Boolean 4D** (`torch.bool`): `True` = attend, `False` = mask
- **Float 4D** (`torch.float32`): `0.0` = attend, `min_dtype` = mask

The original code assumed float masks only.  Models that emit boolean 4D
masks (e.g., GPT-2 with `GptBigCodeAttention`-style masking, or any model
where the mask hasn't been converted to float by an intermediate layer)
triggered the bug.

### Fix

```python
def _reconstruct_attention(self, q, k, v, **kwargs):
    # ... shape handling unchanged ...

    scale = head_dim ** (-0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    attention_mask = kwargs.get("attention_mask", None)
    if attention_mask is not None:
        # HF's 4D mask already includes causal masking — use it directly
        # instead of redundant tril.
        if attention_mask.shape[-1] != seq_len:
            attention_mask = attention_mask[..., :seq_len]
        if attention_mask.shape[-2] != seq_len:
            attention_mask = attention_mask[..., :seq_len, :]

        # Convert boolean → additive float (True=0.0, False=min_dtype).
        # Using min_dtype rather than -inf avoids NaN from softmax on
        # fully-masked (padding) rows.
        if attention_mask.dtype == torch.bool:
            min_dtype = torch.finfo(attn_scores.dtype).min
            attention_mask = torch.where(
                attention_mask,
                torch.zeros((), dtype=attn_scores.dtype, device=attn_scores.device),
                torch.full((), min_dtype, dtype=attn_scores.dtype, device=attn_scores.device),
            )
        attn_scores = attn_scores + attention_mask
    else:
        # Fallback: simple causal mask when no HF mask is provided
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

    attn_scores = self.hook_attn_scores(attn_scores)
    # ... softmax, dropout, output projection unchanged ...
```

### Key Design Decisions

1. **`min_dtype` instead of `-inf`**: Softmax of a row of all `-inf` produces
   NaN.  Fully-masked rows (all-padding prefix in causal attention) would
   hit this.  `torch.finfo(dtype).min` is large enough for masking but
   finite, producing a well-defined (near-zero) softmax output.

2. **No `tril` when HF mask is present**: HF's `_update_causal_mask()` bakes
   causal masking into its 4D output.  A redundant `tril` would be harmless
   for float masks (double-masking already-masked positions) but for boolean
   masks it would AND the causal pattern with the mask — correct in theory
   but unnecessary and fragile.  The clean separation (HF mask → use it;
   no mask → tril fallback) is more maintainable.

3. **Sequence-length trimming**: HF may cache masks for longer sequences
   than the current batch (`attention_mask.shape[-1] > seq_len`).  Trimming
   ensures shape compatibility.

### Verification

The fix was validated through:

- **5-way canonical comparison** (`debug/debug_4way_canonical_comparison.py`):
  HF sdpa, HF eager, NNsight, HookedTransformer, TransformerBridge — Bridge
  now matches HF outputs to ~1e-7 on both padded and unpadded inputs.
- **56 parity tests** in `tests/core/test_sae_backend_parity.py`:
  Bridge ↔ NNsight tight parity (atol=1e-4) across base logit_diffs, SAE,
  gradient attribution, and ablation attribution.
- **965 broader tests** passing (full interpretune test suite).
- **5 bridge diagnostic scripts** in `debug/bridge_*_diagnostic*.py`:
  Layer-by-layer divergence analysis confirming mask handling is the sole
  source of Bridge↔HT differences.

### Related Diagnostics

| Script | Purpose |
|--------|---------|
| `debug/bridge_compat_diagnostic.py` | Baseline: unpadded inputs, `enable_compatibility_mode` effect |
| `debug/bridge_compat_diagnostic_padded.py` | Isolation: no mask → identical; mask → all divergence |
| `debug/bridge_compat_diagnostic_correct.py` | Full IT config (left-pad, pad=eos): ~46–93 logit-unit diff |
| `debug/bridge_compat_layerwise.py` | Layer growth: Layer 0 ~6.7 → Layer 11 ~60.7 |
| `debug/bridge_same_mask_layerwise.py` | Same explicit mask: Layer 0 ~4.0 → Layer 11 ~210.5 |
| `debug/debug_4way_canonical_comparison.py` | 5-way comparison: HF sdpa/eager, NNsight, HT, Bridge |

### References

- [Bridge vs HT Divergence Analysis](bridge_ht_divergence_analysis.md)
- [HT vs Bridge Source of Truth](ht_vs_bridge_source_of_truth.md)
- [TransformerBridge Architecture](transformer_bridge_architecture.md)
- HF `_update_causal_mask()`: `transformers/models/gpt2/modeling_gpt2.py`
