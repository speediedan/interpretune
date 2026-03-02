# HookedTransformer vs TransformerBridge: Source of Truth Analysis

## Purpose

This document analyzes the attention masking code path differences between
HookedTransformer (HT) and TransformerBridge (Bridge) and recommends which
should be treated as the **source of truth** for numerical parity testing.

> **Validation (post-fix):** A bug in `JointQKVAttentionBridge._reconstruct_attention()`
> was discovered where HF's boolean 4D attention masks were mishandled (boolean
> values treated as additive floats, redundant tril applied over HF's 4D mask).
> After the fix, Bridge matches native HF output to float precision on all
> tokens — confirming that Bridge correctly preserves HF's canonical forward
> pass and is the appropriate source of truth.

## Attention Masking Code Paths

The two implementations handle attention masking fundamentally differently:

### HookedTransformer (Custom TL Forward)

1. **Masking arithmetic**: Multiplicative boolean masking —
   `final_mask = (causal_mask * attention_mask).bool()` followed by
   `torch.where(final_mask, attn_scores, -inf)`
   (source: `abstract_attention.py:548–587`).
2. **Pad position embeddings**: Explicitly zeroed via
   `torch.where(offset_padding_mask, 0, pos_embed)` in `PosEmbed.forward()`
   (source: `pos_embed.py:58–78`).
3. **Pad position IDs**: Filled with 0 via `masked_fill(~mask, 0)`.
4. **BOS/PAD overlap**: Special handling when `prepend_bos=True` and
   `bos_token_id == pad_token_id` — sets the last pad token's mask bit to 1,
   creating a 1-position mask mismatch vs explicit tokenizer masks.

### TransformerBridge / HuggingFace (Native HF Forward)

1. **Masking arithmetic**: `_reconstruct_attention()` intercepts HF's 4D
   attention mask from `_update_causal_mask()`.  When a 4D mask is present,
   it skips redundant tril causal masking and converts boolean masks to float
   via `torch.where(mask, 0.0, torch.finfo(dtype).min)`.  This preserves
   HF's native masking semantics exactly.
2. **Pad position embeddings**: Uses real `W_pos` values at pad positions
   (no zeroing).
3. **Pad position IDs**: Filled with 1 via `masked_fill(~mask, 1)`.
4. **BOS/PAD overlap**: No special handling — uses the tokenizer-generated
   attention mask directly.

## Implications for Numerical Parity

| Scenario | Divergence | Root Cause |
|----------|-----------|------------|
| No attention mask | 0.0 (bit-for-bit identical) | Same arithmetic, no masking code paths active |
| All-ones mask, unpadded | ~6e-5 | Masking code paths run identity operations — minimal overhead |
| Real mask, padded inputs | ~42–107 logit units | Different masking implementations compound through 12 layers |

The divergence on padded inputs is **purely architectural** — weight matrices
are bit-for-bit identical between the two implementations.

## Recommendation: TransformerBridge as Source of Truth

**TransformerBridge (HF native forward) should be the source of truth.**

### Rationale

1. **Canonical model behavior**: The HuggingFace model is the authoritative
   implementation of GPT-2 (and other architectures). TransformerBridge
   preserves the native HF forward pass without modification. Any analysis
   results should match what the model was designed to produce.

2. **HookedTransformer reimplements the forward pass**: HT copies weights
   into a custom architecture with a reimplemented forward. This introduces
   intentional behavioral differences (pad embedding zeroing, BOS/PAD overlap
   handling, multiplicative masking) that do not exist in the original model.

3. **NNsight also uses HF forward**: The NNsight backend wraps the raw HF
   model and runs the native forward pass, the same code path as
   TransformerBridge. Using Bridge as the reference aligns the parity
   hierarchy: Bridge ≈ NNsight (same forward) ≠ HT (reimplemented forward).

4. **Upstream acknowledgment**: TransformerLens itself acknowledges this
   divergence — bridge-vs-hooked comparison tests are **skipped** upstream
   with the note "architectural differences."

5. **Future-proof**: As new model architectures are added, they will be
   available via HF first. TransformerBridge and NNsight naturally support
   new architectures; HookedTransformer requires manual weight conversion
   and forward reimplementation.

### Test Structure Implications

With TransformerBridge as the source of truth:

- **Active (Tight Parity)**: Bridge ↔ NNsight — both use HF native forward;
  differences arise only from hook interception mechanism.
  Tests: `TestLogitDiffsBaseBackendParity`, etc. in
  `tests/core/test_sae_backend_parity.py`.

### Note on Removed Bounded Divergence Tests

Bridge ↔ HT bounded divergence tests (formerly "Tier 2") were removed.
The divergence is architecturally fundamental — different masking code paths,
different pad embedding treatment, and different autograd graphs produce
answer_logits divergence of ~85 units and alive-latent Jaccard overlap of 0%
on padded batches.  Tolerances wide enough to absorb this (atol=2–110)
provide no useful regression signal.  HT functional correctness is validated
by existing HT-specific tests elsewhere.

See [Bridge vs HT Divergence Analysis](bridge_ht_divergence_analysis.md) §
"Testing Strategy" for measured divergence data and full rationale.

## References

- [Bridge vs HT Divergence Analysis](bridge_ht_divergence_analysis.md)
- [TransformerBridge Architecture](transformer_bridge_architecture.md)
- TransformerLens bridge vs hooked tests: **SKIPPED** upstream
- SAELens bridge tests: `SAELens/tests/analysis/test_sae_transformer_bridge.py`
- Bridge attention fix: `joint_qkv_attention.py` — `_reconstruct_attention()` conditional 4D mask handling
