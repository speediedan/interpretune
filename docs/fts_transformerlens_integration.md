# FinetuningScheduler Integration with TransformerLens

This document describes how Interpretune integrates FinetuningScheduler (FTS) with TransformerLens architectures (HookedTransformer and TransformerBridge), including parameter naming conventions, schedule generation, and phase-based parameter thawing behavior.

## Overview

FinetuningScheduler enables flexible fine-tuning through YAML-based schedules that specify which parameters to thaw at each training phase. When integrated with TransformerLens, FTS must handle four distinct parameter naming conventions:

1. **HuggingFace Canonical** - Standard PyTorch naming (e.g., `transformer.h.9.attn.c_attn.weight`)
2. **TransformerLens - HookedTransformer** - Legacy TL naming (e.g., `blocks.9.attn.W_Q`) including LayerNorms in named_parameters()
3. **TransformerLens - TransformerBridge** - Modern names using canonical `_original_component` wrapper references
4. **TransformerLens - TransformerBridge TL Names Mode** - Modern TL-style naming

## Parameter Naming Conventions

### Base (HuggingFace GPT-2)
```yaml
# Pattern: model.transformer.h.{layer}.(mlp|attn|ln_{1|2}).(c_proj|c_fc|c_attn|weight|bias)
Examples:
  - model.transformer.h.9.attn.c_attn.weight
  - model.transformer.h.10.mlp.c_fc.bias
  - model.transformer.h.11.ln1.weight
  - model.transformer.h.11.ln2.bias
  - model.transformer.wte.weight (embeddings)
  - model.transformer.ln_f.weight (final LayerNorm)
```

### HookedTransformer
```yaml
# Pattern: model.blocks.{layer}.*
# Legacy TL names include LayerNorm params
Examples:
  - model.blocks.9.attn.W_Q
  - model.blocks.10.mlp.W_out
  - model.blocks.11.ln1.w
  - model.blocks.11.ln2.b
  - model.embed.W_E (embeddings)
  - model.unembed.W_U (unembedding)
```

### TransformerBridge (Canonical Mode)
```yaml
# Pattern: model.blocks.{layer}.*
# Includes _original_component wrappers for compatibility
Examples:
  - model.blocks.9._original_component.attn.qkv._original_component.weight
  - model.blocks.10._original_component.mlp.c_fc._original_component.weight
  - model.blocks.11._original_component.ln_1._original_component.weight
  - model.blocks.11._original_component.ln_2._original_component.bias
  - model.embed._original_component.weight
  - model.unembed._original_component.weight
```

### TransformerBridge (TL Names Mode)
```yaml
# Pattern: blocks.{layer}.* (no model. prefix)
# Uses TL-style names, implicitly includes LayerNorms
Examples:
  - blocks.9.attn.W_Q
  - blocks.10.mlp.W_out
  - embed.W_E
  - unembed.W_U
```

## Schedule Generation and Phase Structure

### Base GPT-2 (2-Phase Schedule)

**Schedule:**
```yaml
0:
  max_transition_epoch: 2
  params:
  - model.transformer.h.(9|1[0-1]).(mlp|attn|ln_(1|2)).(c_proj|c_fc|c_attn|weight|bias).*
1:
  lr: 1.0e-06
  max_transition_epoch: 3
  params:
  - model.transformer.h.([7-8]).(mlp|attn|ln_(1|2)).(c_proj|c_fc|c_attn|weight|bias).*
2:
  lr: 1.0e-06
  params:
  - model.transformer.h.([0-6](?!\d)).(mlp|attn|ln_(1|2)).(c_proj|c_fc|c_attn|weight|bias).*
  - model.transformer.(wpe|wte).weight
  - model.transformer.ln_f.*
```

**Parameter Counts:**
- **Phase 0** (blocks 9-11): 36 parameters
  - blocks 9-11, 3 blocks × (4 attn + 4 mlp + 4 ln params per block) = 36
  - Params: attn (c_attn, c_proj), mlp (c_fc, c_proj), ln_1, ln_2
- **Phase 1** (blocks 7-8): 24 additional parameters (60 total)
  - adds blocks 7-8, +24 params
- **Phase 2** (blocks 0-6 + embeddings): 88 additional parameters (148 total)
  - adds blocks 0-6 + embeddings, +88 params
  - Embeddings: wte, wpe, ln_f (weight + bias)

### HookedTransformer (3-Phase Schedule)

**Schedule:**
```yaml
0:
  max_transition_epoch: 2
  params:
    - model.blocks.(9|1[0-1]).*
1:
  lr: 1.0e-06
  max_transition_epoch: 3
  params:
    - model.blocks.([7-8]).*
2:
  lr: 1.0e-06
  params:
    - model.blocks.([0-6](?!\d)).*
    - model.(pos_embed|embed|unembed).*
```

**Parameter Counts:**
- **Phase 0** (blocks 9-11): 48 parameters
  - 3 blocks × 12 TL params/block + 12 LayerNorm params = 48
  - TL params per block: W_Q, W_K, W_V, W_O, b_Q, b_K, b_V, b_O, W_in, W_out, b_in, b_out
  - LayerNorm: ln1.w, ln1.b, ln2.w, ln2.b per block (note legacy TL names include explicit LayerNorms)
- **Phase 1** (blocks 7-8): 32 additional parameters (80 total)
  - 2 blocks × 12 TL params + 8 LayerNorm params = 32
- **Phase 2** (blocks 0-6 + embeddings): 116 additional parameters (196 total out of 198 total params)
  - 7 blocks × 12 TL params + 28 LayerNorm + 4 embed params = 116
  - Embeddings: W_E, W_pos, W_U, b_U (4 params)
  - 2 ln_final (ln_final.w and ln_final.b) params remain frozen since we did not specify them in the schedule
- there are 198 total parameters in named_parameters(), 50 more than in the original HF due to 4 extra attention params per block (4x12) + 2 unembed params

### TransformerBridge Canonical Mode (3-Phase Schedule)

**Schedule:**
```yaml
0:
  max_transition_epoch: 2
  params:
    - model.blocks.(9|1[0-1]).*
1:
  lr: 1.0e-06
  max_transition_epoch: 3
  params:
    - model.blocks.([7-8]).*
2:
  lr: 1.0e-06
  params:
    - model.blocks.([0-6](?!\d)).*
    - model.(pos_embed|embed|unembed).*
```

**Parameter Counts:**
- **Phase 0** (blocks 9-11): 54 parameters
  - 3 blocks × 14 params/block + 12 LayerNorm params = 54
  - Bridge params per block (via _original_component refs, e.g. `model.blocks.11._original_component.mlp._original_component.c_proj._original_component.bias`):
    - attn (10, 2 original joined qkv not used but still available): qkv.weight, qkv.bias, q/k/v/o.weight, q/k/v/o.bias
    - mlp (4): c_fc.weight, c_fc.bias, c_proj.weight, c_proj.bias
  - **+6 params vs HookedTransformer**: Joint QKV projections stored separately
- **Phase 1** (blocks 7-8): 36 additional parameters (90 total)
  - 2 blocks × 14 params + 8 LayerNorm params = 36
  - **+4 params vs HookedTransformer**: Additional joint QKV params
- **Phase 2** (blocks 0-6 + embeddings): 129 additional parameters (219 total)
  - 7 blocks × 14 params + 28 LayerNorm + 3 embed params = 129
  - **+13 params vs HookedTransformer**: 7 blocks × 2 extra QKV params - 1 untied unembed weight param (unembed only has bias as it remained tied to the embed weight since we specified `enable_compatibility_mode` with `no_processing`)
  - 2 ln_final params remain frozen since we did not specify them in the schedule

**Key Difference:** TransformerBridge canonical mode stores joint QKV projection parameters separately from the split Q, K, V views. Both are included in the parameter count, resulting in 2 additional parameters per attention layer (qkv.weight + qkv.bias). They are thawed but not used in training.

### TransformerBridge TL Names Mode (3-Phase Schedule)

**Schedule:**
```yaml
0:
  max_transition_epoch: 2
  params:
    - blocks.(9|1[0-1]).*
1:
  max_transition_epoch: 3
  params:
    - blocks.([7-8]).*
2:
  lr: 1.0e-06
  params:
    - blocks.([0-6](?!\d)).*
    - (pos_embed|embed|unembed).*
```

**Parameter Counts:**
- **Phase 0** (blocks 9-11): 48 parameters
  - Same as HookedTransformer: 3 blocks × 12 TL params + 12 LayerNorm = 48
- **Phase 1** (blocks 7-8): 32 additional parameters (80 total)
  - Same as HookedTransformer: 2 blocks × 12 TL params + 8 LayerNorm = 32
- **Phase 2** (blocks 0-6 + embeddings): 117 additional parameters (197 total)
  - Same as HookedTransformer: 7 blocks × 12 TL params + 28 LayerNorm + 3 embed (1 unembed bias) + 2 ln_final layers thawed (since `implicit_ln_thaw` was left to the default of `True`) = 117
  - 24 total parameters remain unfrozen overall, the 24 (unused) joint qkv parameters
- since we did not set enable_compatibility_mode, the unembed weight remains tied to the embed weight, so we have only embed.weight, pos_embed.weight and unembed.bias embedding params thawed
**Key Feature:** TL names mode uses TLNamesModelView to translate between canonical parameter names and TL-style names. The schedule uses TL nomenclature (e.g., `W_Q`, `W_K`) to orchestrate the thawing of the associated underlying canonical parameters during training. Joint QKV parameters remain available but are not thawed since we are driving our training with TL names and as the qkv joint parameters are not used in training they are not mapped to the TL style names.

## LayerNorm Handling

### HookedTransformer (Legacy)
- **Explicit in Schedule**: LayerNorm parameters (ln1.w, ln1.b, ln2.w, ln2.b, ln_final.w, ln_final.b) are included in named_parameters()
- **Regex Matching**: Schedule patterns like `blocks.9.*` explicitly match LayerNorm params
- **Phase Association**: LayerNorm params thaw when their regex pattern matches them in the schedule

### TransformerBridge (Modern v3)
- Two supported modes, depending on the provided bridge adapter configuration.

#### CanonicalModelView vs TLNamesModelView
- CanonicalModelView:
    - When use_tl_names=False (default), the fine-tuning schedule uses direct canonical parameter specification
  but may expose architectural differences reducing schedule portability
    - **Explicit Schedule Matching**: Schedule patterns used to match LayerNorm params and they are thawed via canonical named_parameters()
- TLNamesModelView:
    - When use_tl_names=True, provides standard TL-style interface for convenient cross-architecture schedules
    - LayerNorm parameters are NOT included in tl_named_parameters() (used to drive schedule in TL names mode) so we offer the implicit ln thawing option.
    - **implicit_ln_thaw**: Default True, automatically thaws LayerNorms when their block is thawed
    - **Regex-Based**: Wildcard patterns like `blocks.9.*` trigger implicit LayerNorm thawing for that block

## Checkpoint Restoration Behavior

With the additive penalty divergence mechanism (implemented for testing), all test configurations show consistent checkpoint restoration behavior:

- **Divergence Starts**: Epoch 2 (first epoch of phase 1)
- **Best Checkpoint**: Remains at depth 0 (phase 0) throughout training
- **Restoration**: All architectures correctly restore to the best checkpoint from phase 0

## Implementation Details

### TLNamesModelView
The `TLNamesModelView` class (in `transformer_lens.py`) provides:
- **Bidirectional mapping** between TL and canonical parameter names
- **Component tracing** via `data_ptr()` to match parameters by memory address
- **Implicit LayerNorm handling** - LayerNorm params excluded from mapping, caught by regex
- **Transform methods** for converting between naming conventions

### CanonicalModelView
The `CanonicalModelView` class (in `model_view.py`) provides:
- **Direct canonical naming** for base HuggingFace models
- **Standard schedule generation** without name transformations
- **Explicit parameter matching** via regex patterns

### Schedule Generation Flow

1. **Model Inspection**: Extract trainable parameters with their names
2. **Name Transformation** (TL Names mode only): Convert canonical → TL names via ModelView
3. **Regex Pattern Generation**: Create patterns for layer groups (e.g., `blocks.(9|1[0-1]).*`)
4. **Phase Definition**: Map patterns to training phases with transition epochs
5. **YAML Serialization**: Write schedule to `{ModuleName}_ft_schedule.yaml`

## Testing Strategy

All FTS integration tests validate:
- Correct parameter counts per phase
- Proper schedule generation for each architecture
- Checkpoint restoration at correct depths
- Consistent behavior across naming conventions

Test configurations:
- `train_cpu_32_l_fts`: Base GPT-2 (3-phase)
- `train_cuda_32_l_tl_ht_fts`: HookedTransformer (3-phase)
- `train_cuda_32_l_tl_bridge_fts`: TransformerBridge canonical (3-phase)
- `train_cuda_32_l_tl_bridge_tl_names_fts`: TransformerBridge TL names (3-phase)

## Parameter Count Differences Explained

### Why HookedTransformer has 48 params vs Base 36 params in Phase 0?

**Base (36 params):**
- 3 blocks × 12 canonical params = 36
- Attention: c_attn.weight, c_attn.bias, c_proj.weight, c_proj.bias (4 params - joint QKV projection)
- MLP: c_fc.weight, c_fc.bias, c_proj.weight, c_proj.bias (4 params)
- LayerNorm: ln_1.weight, ln_1.bias, ln_2.weight, ln_2.bias (4 params)

**HookedTransformer (48 params):**
- 3 blocks × 16 TL params = 48
- Attention: W_Q, W_K, W_V, W_O, b_Q, b_K, b_V, b_O (8 params - split Q/K/V projections)
- MLP: W_in, W_out, b_in, b_out (4 params)
- LayerNorm: ln1.w, ln1.b, ln2.w, ln2.b (4 params)
- **Key difference**: HookedTransformer has 8 attention params vs Base's 4 (split Q/K/V vs joint QKV)

### Why TransformerBridge Canonical has 54 params vs HookedTransformer 48 params?

**Additional 6 params:**
- 3 blocks × 2 params = 6
- Each attention layer stores joint QKV projection (qkv.weight, qkv.bias)
- Split Q, K, V parameters are materialized as new parameters for LinearBridge modules
- Joint and split QKV parameters do NOT share underlying storage - they are separate tensors
- Result: Bridge has both joint (2 params) and split (6 params) QKV parameters per block


## Related Documentation

- [TL Style Naming Implementation](tl_style_naming_implementation.md) - Details on TL parameter name transformations
- [TL Config Hierarchy](tl_config_hierarchy_overview.md) - TransformerLens configuration architecture
