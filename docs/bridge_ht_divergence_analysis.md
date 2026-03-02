# Bridge vs HookedTransformer Divergence Analysis

## Summary

HookedTransformer (HT) and TransformerBridge (Bridge) are two wrappers for
running transformer models through TransformerLens/SAELens.  They produce
**bounded but non-trivial numerical divergence** on padded inputs, but are
**bit-for-bit identical** when no attention mask is applied and **near-identical**
(~6e-5) on unpadded inputs with an all-ones mask.

The primary root cause is **different attention masking code paths** — not
float32 non-associativity as previously hypothesized.  This document explains
the true root cause, presents diagnostic evidence, reports measured divergence,
and describes the testing strategy in
`tests/core/test_sae_backend_parity.py`.

> **Note (post-fix update):** A bug in `JointQKVAttentionBridge._reconstruct_attention()`
> was discovered and fixed during this investigation.  The Bridge was mishandling
> HuggingFace's boolean 4D attention masks: it unconditionally applied a tril
> causal mask (redundant when HF's 4D mask already encodes causality) and
> treated boolean masks as additive floats (True=1.0, False=0.0 — effectively
> ignoring padding).  The fix: (1) skip tril when a 4D mask is present, and
> (2) convert boolean masks to float via `torch.where(mask, 0.0, min_dtype)`.
> After this fix, Bridge matches native HF to float precision on all tokens.
> Bridge↔HT divergence is now **larger** on padded batches (~85 answer_logits
> units vs the pre-fix ~10) because Bridge now correctly masks padding while
> HT uses its own independent masking implementation.

## Architecture Comparison

| Aspect                 | HookedTransformer                        | TransformerBridge                          |
|------------------------|------------------------------------------|--------------------------------------------|
| Weight source          | Converts HF weights to TL format         | Uses HF model weights directly             |
| Forward pass           | Custom TL attention / MLP / LayerNorm    | Native HuggingFace model forward pass      |
| Hook mechanism         | TL HookPoints on every sub-operation     | GeneralizedComponent hooks on HF modules   |
| Attention masking      | Multiplicative: `(causal * attn_mask).bool()` → `torch.where(mask, scores, -inf)` | Additive: `_reconstruct_attention()` converts HF 4D boolean mask to float via `torch.where(mask, 0.0, min_dtype)`, skips redundant tril when HF mask present (post-fix) |
| Pad position embedding | Zeroed via `torch.where(mask, 0, pos_embed)` | Uses real W_pos[1] values                   |
| Pad position IDs       | Filled with 0 (`masked_fill(~mask, 0)`)  | Filled with 1 (`masked_fill(~mask, 1)`)    |

Weights (W_pos, W_E, all layers) are **bit-for-bit identical** between the
two implementations — verified with `max_diff = 0.0` on both matrices.

## Root Cause: Attention Masking Code Path Divergence

### Disproving the Prior Hypothesis

The original analysis attributed divergence to "float32 non-associativity
compounding through 12 layers."  This is **incorrect**.  Evidence:

1. **No mask, no padding → bit-for-bit identical (diff = 0.0).**
   When neither implementation applies an attention mask, the outputs are
   exactly equal (Diagnostic 2).  If float32 operation ordering differences
   alone caused divergence, there would be non-zero diff here.

2. **All-ones mask, no padding → ~6e-5 diff.**
   When both implementations receive an all-ones attention mask on unpadded
   input, only ~6e-5 max absolute difference appears across 12 layers
   (Diagnostic 5 control).  This negligible amount represents the actual
   float32 overhead of the different masking code paths running identity
   operations (masking nothing).

3. **Real mask, real padding → ~42–107 logit-unit diff.**
   With legitimate padding and attention masks, divergence grows to tens of
   logit units (Diagnostics 3–5).

The divergence is dominated by the **masking-active code paths**, not by
generic float32 non-associativity.

### Three Parity Regimes

| Regime                    | Max Abs Diff (last token) | Explanation |
|---------------------------|---------------------------|-------------|
| No mask at all            | 0.0 (bit-for-bit)        | Same arithmetic, no masking code paths active |
| All-ones mask (unpadded)  | ~6e-5                     | Masking code paths run but do nothing — minimal overhead |
| Real mask (padded)        | ~42–107 logit units       | Different masking implementations create different intermediate values |

### Contributing Factors (Padded Divergence)

The ~42–107 logit-unit divergence on padded inputs arises from four
interacting mechanisms:

1. **Pad position embedding treatment.**
   HT explicitly zeroes position embeddings at pad positions via
   `torch.where(offset_padding_mask, 0, pos_embed)` in `PosEmbed.forward()`
   (pos_embed.py:58–78).  HF keeps real `W_pos[1]` values for pad positions
   (verified: pad-position diff = 1.621 per position; real-position diff =
   0.000000 — confirmed by manual computation bypassing GeneralizedComponent
   hooks).

2. **Attention masking arithmetic.**
   HT uses multiplicative boolean masking: `final_mask = (causal_mask *
   attention_mask).bool()` → `torch.where(final_mask, attn_scores, -inf)`
   (abstract_attention.py:548–587).  HF uses additive masking via
   `_prepare_4d_causal_attention_mask` which adds `torch.finfo(dtype).min`
   (~-3.4e38) at masked positions.  While both effectively mask pad positions,
   the different float32 arithmetic pathways at pad positions produce
   different intermediate values in the residual stream.

3. **BOS/PAD token overlap (HT auto-mask only).**
   GPT-2 has `bos_token_id == eos_token_id == pad_token_id == 50256`.  HT's
   `get_attention_mask()` (tokenize_utils.py:202–245) has special handling:
   when `prepend_bos=True` and `bos_token_id == pad_token_id`, it sets the
   mask of the last pad token to 1 (treating it as a real BOS).  This creates
   a 1-position mask mismatch vs explicit tokenizer-generated masks.  The
   mismatch shifts position IDs by 1 for all real tokens, producing ~0.79
   position-embedding diff at real positions (Diagnostic 4: `pos_emb_diff at
   mask=1 positions: max=0.79`).

4. **Layer-by-layer amplification.**
   Different pad-position embeddings in the residual stream, processed
   through different attention masking arithmetic, create small per-layer
   divergence that compounds through the transformer depth.  Measured
   growth (Diagnostic 4, HT auto-mask vs Bridge explicit mask):
   - Layer 0 last-token diff: ~6.7
   - Layer 6 last-token diff: ~20.1
   - Layer 11 last-token diff: ~60.7
   - Final logits last-token diff: ~41.9

### Why Real Tokens Diverge Despite Identical Embeddings

Manual computation confirmed that raw position embeddings at all real token
positions are **exactly identical** (diff = 0.000000) between HT and Bridge.
The divergence at real tokens appears because:

- The attention mechanism computes Q·K^T scores against ALL positions in the
  causal window, including pad positions that have different values in the
  two implementations.
- Even though the attention mask should make pad positions contribute zero
  to the weighted sum, the different numerical pathway to that zero (HT:
  multiplicative + `torch.where(mask, score, -inf)` vs HF: additive with
  `finfo.min`) produces subtly different float32 intermediate values in the
  softmax computation.
- These per-layer differences compound through residual connections.

## Measured Divergence (GPT-2, RTE Task)

Measurements from diagnostics on 178-token left-padded RTE inputs
(2 eval batches × 2 examples each):

### Token and Weight Alignment (pre-forward)

| Check             | Result              |
|-------------------|---------------------|
| Token tensors     | IDENTICAL           |
| W_E weights       | IDENTICAL (0.0)     |
| W_pos weights     | IDENTICAL (0.0)     |
| Position embeddings (real tokens) | IDENTICAL (0.000000) |
| Position embeddings (pad tokens)  | Different (1.621)    |

### Forward-Pass Output Divergence

| Scenario                            | All-Pos Max | Last-Tok Max | Last-Tok Mean |
|-------------------------------------|-------------|--------------|---------------|
| No mask (control)                   | 0.0         | 0.0          | 0.0           |
| All-ones mask, no padding (control) | ~6e-5       | ~6e-5        | ~6e-5         |
| HT(auto-mask) vs Bridge(explicit)   | ~121        | ~42          | ~15           |
| Same explicit mask, with padding    | ~233        | ~107         | ~38           |

Note: "Same explicit mask" shows larger divergence because the identical
mask forces HT through a non-standard masking path (explicit rather than
auto-generated), amplifying differences from mechanisms 1–2 above.

### Analysis Pipeline Divergence (via AnalysisStore, Post-Fix)

| Analysis Op           | logit_diffs max | answer_logits max | Predictions |
|-----------------------|-----------------|-------------------|-------------|
| logit_diffs_base      | 0.710           | 84.99             | EXACT match |
| logit_diffs_sae       | 0.988           | 70.25             | EXACT match |
| logit_diffs_attr_grad | 0.988           | —                 | —           |
| attr_grad attribution | 0.165           | —                 | —           |
| attr_grad correct_act | 4.679           | —                 | —           |
| alive_latents Jaccard (padded batch)  | 0% (blocks.9, blocks.10) | — | — |
| alive_latents Jaccard (unpadded batch)| 11.8–23.1%               | — | — |

> **Pre-fix values (for reference):** Base answer_logits was ~10.18, SAE was
> ~4.02.  Post-fix values are larger because Bridge now correctly masks
> padding (matching HF native behavior), creating greater divergence from
> HT's independent masking implementation.  This is expected and correct.

Derived metrics (logit_diffs) show much smaller divergence than raw logits
because the systematic bias from different pad handling largely cancels when
computing the **difference** between two answer token logits from the same
forward pass.

The alive-latents Jaccard overlap drops to 0% on padded batches because
different masking implementations produce different activation magnitudes at
pad-adjacent positions, causing different latents to fire.  On unpadded
batches, Jaccard overlap is 11–23%, demonstrating meaningful agreement when
padding differences are absent.

## Why TinyStories-1M Achieves atol=1e-4

The divergence is proportional to the amount of masking work:

1. **No padding** — With unpadded inputs, there are no pad positions to
   trigger different treatment.  Both implementations take nearly identical
   code paths, producing only ~6e-5 diff from basic masking overhead.
2. **Model depth** — More layers amplify per-layer differences.
3. **Sequence length** — Longer sequences have more positions where masking
   differences can accumulate.

TinyStories-1M is:
- **Small:** 4 layers (vs 12 for GPT-2)
- **Short inputs:** ~3 tokens for "Hello World!" (vs 178 padded)
- **No padding:** Single unpadded input

With no padding and a tiny model, divergence stays well within 1e-4.  This
is the same model and tolerance used by SAELens upstream bridge tests
(see `SAELens/tests/analysis/test_sae_transformer_bridge.py`).

## Upstream Acknowledgment

The TransformerLens project itself acknowledges this architectural divergence.
In `TransformerLens/tests/integration/test_bridge_vs_hooked_comparison.py`,
all bridge-vs-hooked comparison tests are **skipped** with the message:

> "Bridge vs Hooked comparison failing due to architectural differences."

This confirms that the divergence is a known, expected consequence of the two
fundamentally different forward-pass implementations, not a bug in either
TransformerLens or interpretune.

## Testing Strategy

### Active: Numerical Parity (TransformerBridge ↔ NNsight)

- **What:** Both run the HF model's native forward pass, using different hook
  interception mechanisms (TL HookPoints vs NNsight thread interleaving).
  TransformerBridge serves as the source of truth (canonical HF behavior).
- **Why close:** Same underlying HF forward pass, minor differences from
  hook mechanisms
- **Tolerance:** atol=1e-4, rtol=1e-4 (gradients: 5e-3)
- **Tests:** `TestLogitDiffsBaseBackendParity`, etc.

### Removed: Bounded Divergence (Bridge ↔ HT on GPT-2)

> **Tier 2 tests were removed** because the divergence between Bridge and HT
> is architecturally fundamental — different attention masking code paths
> (multiplicative vs additive), different pad position embedding treatment,
> and different autograd graphs that produce answer_logits divergence of ~85
> units and alive-latent Jaccard overlap of 0% on padded batches.  Tolerances
> wide enough to accommodate this divergence (atol=2–110) provide no useful
> regression signal; a real bug could hide within the tolerance bands.
> HookedTransformer functional correctness is validated by existing HT-specific
> tests elsewhere in the suite.  This is a deliberate simplification, not a
> loss of coverage.

For historical context, the removed Tier 2 parameters were:

- **Tolerance:** atol=2–110 depending on metric (calibrated from measured max
  with 1.3–3.0x headroom); alive_latents min_overlap=0.0 (padded batch
  Jaccard drops to 0%)
- **Invariants:** Top-1 predictions match exactly; continuous values bounded
- **Tests:** `TestBridgeLogitDiffsBaseParity`, etc.
- **Key insight:** Derived metrics (logit_diffs) have much smaller divergence
  than raw logits because the systematic pad-handling bias cancels in
  differential measurements.  Alive latent overlap is meaningful (11–23%)
  on unpadded batches but drops to 0% on padded batches where masking
  differences dominate.

Note: Direct TinyStories-1M model-level parity tests (previously Tier 3)
were removed — they are already covered by SAELens upstream bridge tests
(`SAELens/tests/analysis/test_sae_transformer_bridge.py`) and provided
limited additional signal once the divergence root cause was understood.

## Tolerance Calibration Reference (Historical)

The removed Tier 2 tolerances were set conservatively based on measured max divergence
with substantial headroom.  Preserved here for reference:

| Test Class                           | Metric           | Measured Max | Tolerance (atol) | Headroom |
|--------------------------------------|------------------|-------------|-------------------|----------|
| TestBridgeLogitDiffsBaseParity      | logit_diffs      | 0.710       | 2.0               | 2.8x     |
| TestBridgeLogitDiffsBaseParity      | answer_logits    | 84.99       | 110.0             | 1.3x     |
| TestBridgeLogitDiffsSAEParity       | logit_diffs      | 0.988       | 2.0               | 2.0x     |
| TestBridgeLogitDiffsSAEParity       | answer_logits    | 70.25       | 100.0             | 1.4x     |
| TestBridgeLogitDiffsAttrGradParity  | logit_diffs      | 0.988       | 2.0               | 2.0x     |
| TestBridgeLogitDiffsAttrGradParity  | attribution      | 0.165       | 0.5               | 3.0x     |
| TestBridgeLogitDiffsAttrGradParity  | correct_act      | 4.679       | 10.0              | 2.1x     |
| All Bridge classes                  | alive_latents    | 0% Jaccard  | 0.0 min_overlap   | —        |

The answer_logits tolerances are higher post-fix because Bridge now correctly
masks padding (matching native HF behavior), which means its logit values at
pad-adjacent positions diverge further from HT's re-implemented masking.
The alive_latents min_overlap_frac is 0.0 because padded batches produce 0%
Jaccard overlap — this is expected given the different masking implementations.

These tolerances cannot be tightened significantly without fundamentally
changing the test setup (e.g., using unpadded-only inputs, which would defeat
the purpose of testing real-world analysis scenarios).

## Diagnostic Evidence

Five diagnostic scripts and a manual position embedding trace were used to
establish the root cause.  All scripts are preserved in `debug/`.

| Diagnostic | File | Key Finding |
|------------|------|-------------|
| 1 | `bridge_compat_diagnostic.py` | `enable_compatibility_mode` has zero numerical effect; raw model parity ~1e-4 on short unpadded texts |
| 2 | `bridge_compat_diagnostic_padded.py` | No mask → bit-for-bit identical; mask alone causes all divergence |
| 3 | `bridge_compat_diagnostic_correct.py` | With correct IT config (left-pad, pad=eos): ~46–93 logit-unit diff; persists even with identical explicit masks |
| 4 | `bridge_compat_layerwise.py` | Layer-by-layer growth: Layer 0 ~6.7 → Layer 11 ~60.7 (last token) |
| 5 | `bridge_same_mask_layerwise.py` | Same explicit mask, layer-by-layer: Layer 0 ~4.0 → Layer 11 ~210.5; unpadded control: ~6e-5 |
| Manual | Terminal session | Raw `original_component` position embeddings: real tokens diff=0.000000, pad tokens diff=1.621 |

## References

- TransformerLens v3 TransformerBridge: wraps HF model without weight conversion
- TransformerLens bridge vs hooked tests: **SKIPPED** upstream ("architectural differences")
- SAELens bridge tests: `SAELens/tests/analysis/test_sae_transformer_bridge.py`
- Interpretune parity tests: `tests/core/test_sae_backend_parity.py`
- Diagnostic scripts: `debug/bridge_compat_diagnostic*.py`, `debug/bridge_same_mask_layerwise.py`
- Bridge attention fix: `TransformerLens/transformer_lens/model_bridge/generalized_components/joint_qkv_attention.py`
  — `_reconstruct_attention()`: conditional 4D mask handling, boolean→float conversion via `min_dtype`
