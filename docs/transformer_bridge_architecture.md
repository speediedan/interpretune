# TransformerBridge Architecture and Implementation

## Overview

TransformerBridge is the v3 architecture in TransformerLens that wraps
HuggingFace models **without weight conversion**.  Unlike the legacy
HookedTransformer (which copies and reshapes weights into a custom TL
architecture), TransformerBridge provides hook points around the HF model's
native modules while letting the original HuggingFace forward pass execute
unmodified.

This document distils the key architectural details relevant to interpretune's
SAE analysis pipeline and explains how Bridge relates to HookedTransformer and
NNsight.

## Core Architecture

### Class Hierarchy

```
nn.Module
├── TransformerBridge                     # Bridge wrapper
│   └── SAETransformerBridge              # SAELens subclass (adds SAE attachment)
│       └── (interpretune wraps via __class__ swap)
└── HookedTransformer (legacy)
    └── HookedSAETransformer (SAELens subclass)
```

### Key Components

| Component | Role |
|-----------|------|
| `TransformerBridge` | Top-level wrapper; stores `original_model` (HF), `adapter`, `cfg`, hook registry |
| `ArchitectureAdapter` | Maps HF module paths to TL-style names; created by `ArchitectureAdapterFactory` |
| `GeneralizedComponent` | Base class for bridge components; provides `hook_in`/`hook_out` HookPoints |
| `TransformerBridgeConfig` | Config dataclass extending `TransformerLensConfig` with `architecture` field |

### Initialization Flow

```python
# In interpretune's _convert_hf_to_bridge():
tl_config = map_default_transformer_lens_config(hf_model.config)
architecture = determine_architecture_from_hf_config(hf_model.config)
bridge_config = TransformerBridgeConfig.from_dict(tl_config.__dict__)
bridge_config.architecture = architecture
adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
bridge = TransformerBridge(model=hf_model, adapter=adapter, tokenizer=tokenizer)
```

The `TransformerBridge.__init__()` method:

1. Stores the HF model as `original_model` (in `__dict__`, not as an `nn.Module` child)
2. Calls `set_original_components()` to map HF modules into `GeneralizedComponent` wrappers
3. Initializes the hook registry — creates `HookPoint` instances for each component
4. Registers hook aliases (e.g., `hook_embed` → `embed.hook_out`)
5. Sets up hook compatibility infrastructure

### Forward Pass

TransformerBridge delegates the forward pass entirely to the HF model:

```python
# TransformerBridge.forward() essentially does:
outputs = self.original_model(**inputs)
```

The HF model's native attention, MLP, and LayerNorm implementations execute
exactly as they would in any HuggingFace pipeline.  Hook points are attached
via PyTorch hooks on the GeneralizedComponent wrappers, intercepting
activations as they flow through the HF modules.

This is fundamentally different from HookedTransformer, which reimplements
every transformer sub-operation in TL's own code.

### Compatibility Mode

`enable_compatibility_mode()` optionally processes weights to match HT
conventions (fold LayerNorm, center weights, reshape attention matrices).
**Important:** This has **zero numerical effect** on forward pass outputs.
It only creates compatibility aliases and reshaped weight views for TL API
consumers; the forward pass still uses the original HF weights.

## Attention Masking: Bridge vs HookedTransformer

This is the primary source of numerical divergence between the two wrappers.

### HookedTransformer (Multiplicative Boolean)

```python
# In abstract_attention.py:
final_mask = (causal_mask * attention_mask).bool()  # elementwise AND
attn_scores = torch.where(final_mask, attn_scores, self.IGNORE)  # -inf at masked
```

- Constructs a boolean mask combining causal and padding masks
- Uses `torch.where` to set masked positions to `-inf`
- Zeroes pad position embeddings via `torch.where(mask, 0, pos_embed)`
- Pad position IDs are filled with 0

### TransformerBridge / HuggingFace (Additive Float)

```python
# In HF's attention implementation:
causal_mask = _prepare_4d_causal_attention_mask(attention_mask, ...)
# Adds torch.finfo(dtype).min (~-3.4e38) at masked positions
attn_scores = attn_scores + causal_mask
```

- Constructs a float mask with 0.0 at attend positions and `finfo.min` at masked
- Adds the mask directly to attention scores (additive, not boolean)
- Keeps real `W_pos` values at pad positions (no zeroing)
- Pad position IDs are filled with 1

### Numerical Consequences

Both approaches produce functionally equivalent attention (masked positions
get negligible softmax weight), but the different float32 arithmetic creates
small intermediate differences that compound through transformer layers.

| Scenario | Max Absolute Diff |
|----------|-------------------|
| No mask at all | 0.0 (bit-for-bit) |
| All-ones mask, unpadded | ~6e-5 |
| Real mask, padded (12-layer GPT-2, 178 tokens) | ~42–107 logit units |

See [docs/bridge_ht_divergence_analysis.md](bridge_ht_divergence_analysis.md)
for detailed measurements and root cause analysis.

## Bridge in Interpretune

### SAELens Integration

Interpretune wraps the Bridge as `SAETransformerBridge` (a SAELens class)
via `__class__` swap — the same pattern SAELens itself uses in
`SAETransformerBridge.boot_transformers()`:

```python
# In interpretune's _convert_hf_to_bridge():
bridge = TransformerBridge(model=hf_model, adapter=adapter, tokenizer=tokenizer)
bridge.__class__ = SAETransformerBridge
bridge._acts_to_saes = {}
bridge._transcoder_output_hooks = {}
```

This gives the bridge SAE-compatible methods (`add_sae`, `reset_saes`,
`run_with_saes`, `run_with_cache`) while preserving the HF-native forward.

### Configuration

Interpretune provides two Bridge-related config classes:

- **`ITLensFromPretrainedNoProcessingConfig`** with `use_bridge=True`:
  Standard Bridge initialization for most use cases.
- **`ITLensBridgeConfig`**: Fine-grained control over TransformerBridgeConfig
  overrides and `enable_compatibility_mode()` kwargs.

### Fine-Tuning Schedule Support

The `TransformerBridgeStrategyAdapter` provides parameter name translation
between HF-style names (`model.transformer.h.N`) and TL-style names
(`model.blocks.N`) for fine-tuning schedules.  It uses a `ModelView` abstraction
with `TLNamesModelView` and `CanonicalModelView` implementations.

## Bridge vs NNsight

Both TransformerBridge and NNsight wrap the HuggingFace model and use its
native forward pass.  The key differences are in hook mechanism:

| Aspect | TransformerBridge | NNsight |
|--------|-------------------|---------|
| Wrapping approach | GeneralizedComponent wrappers with HookPoints | Thread-based deferred execution |
| Forward pass | HF native (via original_model) | HF native (via wrapped model) |
| Hook mechanism | PyTorch hooks on wrapper modules | Source code extraction + thread interleaving |
| Access pattern | `model.blocks[i].hook_out` | `model.transformer.h[i].output` |
| Weight access | Via compatibility mode aliases (W_Q, W_K, etc.) | Not applicable (tracing, not weights) |

Because both execute the HF model's native forward pass, they should produce
identical numerical results for the same inputs.  This makes Bridge the
natural reference point for validating NNsight parity — they share the same
computation graph.

## Recommendations for Testing Strategy

### TransformerBridge as Source of Truth

We recommend TransformerBridge as the reference implementation for parity
testing because:

1. **Canonical HF behavior:** Bridge executes the model exactly as HuggingFace
   intended, using the upstream-maintained attention, MLP, and normalization code.
2. **NNsight alignment:** NNsight also wraps the HF model, so Bridge↔NNsight
   parity validates that interpretune's analysis pipeline correctly handles both
   hook mechanisms against the same underlying computation.
3. **Future-proof:** As HuggingFace evolves its attention implementations, Bridge
   automatically tracks those changes.
4. **Ecosystem standard:** TransformerLens v3 positions Bridge as the primary
   interface; HookedTransformer is maintained for backward compatibility.

### Bounded Divergence Tests

The Tier 2 tests (Bridge ↔ HT bounded divergence) should be retained
short-term as regression guards — they verify that the known divergence
stays within measured bounds and that top-1 predictions remain identical.
Medium-term, as HookedTransformer usage decreases, these tests can be
simplified to structural checks (matching predictions, matching column
names) without requiring numerical tolerance calibration.

### On TinyStories Testing

The Tier 3 TinyStories tests were designed to confirm that Bridge↔HT
divergence is from masking code paths on padded inputs, not integration
bugs.  Now that this has been thoroughly established (see the five diagnostic
scripts and the root cause analysis), these tests provide diminishing value
and can be removed.  The Bridge↔NNsight parity tests (same computation
graph, should be near-identical) provide a more directly useful correctness
signal.

## Key Source Files

| File | Description |
|------|-------------|
| `TransformerLens/.../bridge.py` | `TransformerBridge` class |
| `TransformerLens/.../generalized_components/base.py` | `GeneralizedComponent` base class |
| `SAELens/.../sae_transformer_bridge.py` | `SAETransformerBridge` (adds SAE attachment) |
| `interpretune/adapters/transformer_lens.py` | `_convert_hf_to_bridge()` |
| `interpretune/adapters/sae_lens.py` | SAE-specific bridge conversion |
| `interpretune/config/transformer_lens.py` | `ITLensBridgeConfig` |
