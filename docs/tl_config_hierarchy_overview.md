# TransformerLens Config Hierarchy Overview

## Overview
This document provides an overview of the configuration hierarchy for TransformerLens v3 (TransformerBridge) and legacy (HookedTransformer) paths, and their integration with Interpretune configurations.

## TransformerLens Config Hierarchy

### Base Class: TransformerLensConfig
Common base class for both architectures, stored in `transformer_lens/config/TransformerLensConfig.py`:
- Defines core model dimensions: `d_model`, `d_head`, `n_layers`, `n_ctx`, `d_vocab`, `n_heads`
- Common to both HookedTransformerConfig and TransformerBridgeConfig

### HookedTransformerConfig (Legacy)
Extends `TransformerLensConfig` as a dataclass:
- **Location**: `transformer_lens/config/HookedTransformerConfig.py`
- **Usage**: Traditional TransformerLens interface with weight conversion
- **Key Fields**:
  - Model architecture: `d_mlp`, `act_fn`, `attn_only`, `parallel_attn_mlp`
  - Attention config: `use_attn_scale`, `use_qk_norm`, `use_local_attn`, `window_size`
  - Initialization: `init_mode`, `initializer_range`, `init_weights`, `seed`
  - Normalization: `normalization_type`, `eps`, `final_rms`
  - Advanced: `gated_mlp`, `rotary_dim`, `rotary_base`, `num_experts`, `experts_per_token`
  - Device/dtype: `device`, `dtype`, `n_devices`
  - Metadata: `model_name`, `checkpoint_index`, `checkpoint_value`, `tokenizer_name`

### TransformerBridgeConfig (v3)
Extends `TransformerLensConfig` as a regular class:
- **Location**: `transformer_lens/config/TransformerBridgeConfig.py`
- **Usage**: Wraps HF models without weight conversion, more memory efficient
- **Key Additions**:
  - `architecture`: Architecture identifier for adapter selection (e.g., "gpt2", "llama")
  - `tokenizer_prepends_bos`: Tokenizer behavior configuration
  - `default_padding_side`: Padding side configuration
  - `split_attention_weights`: Attention weight processing configuration
- **Compatibility**: Includes all HookedTransformerConfig fields for API compatibility
- **Note**: Despite including HookedTransformer fields, TransformerBridge doesn't perform weight conversion

## TransformerBridge Model Structure

### Key Attributes
```python
class TransformerBridge(nn.Module):
    def __init__(self, model, adapter, tokenizer):
        self.original_model = model  # The wrapped HF model
        self.adapter = adapter        # ArchitectureAdapter instance
        self.cfg = adapter.cfg        # TransformerBridgeConfig instance
        self.tokenizer = tokenizer
```

### Adapter Structure
```python
class ArchitectureAdapter:
    def __init__(self, original_model, cfg):
        self.original_model = original_model
        self.cfg = cfg  # TransformerBridgeConfig
        # ... component mappings ...
```

## Interpretune Config Hierarchy

For generation flag precedence and debug semantics, see `docs/generation_precedence.md`.

### Base: ITLensSharedConfig
Common configuration shared across both initialization modes:
- **Location**: `src/interpretune/config/transformer_lens.py`
- **Key Fields**:
  - `move_to_device`: Control device movement (default: True)
  - `default_padding_side`: Padding side for tokenizer (default: "right")
  - `use_bridge`: **Toggle between TransformerBridge (v3) and HookedTransformer (legacy)** (default: True)
- **Purpose**: IT-specific settings that don't map directly to TL configs

### ITLensFromPretrainedConfig
Extends `ITLensSharedConfig` for `from_pretrained` initialization:
- **Usage**: When loading pretrained HF models via model_name
- **Key Fields**:
  - `model_name`: HF model identifier (e.g., "gpt2-small")
  - Processing flags: `fold_ln`, `center_writing_weights`, `center_unembed`, `refactor_factored_attn_matrices`
  - `device`, `dtype`, `n_devices`: Device/dtype configuration
  - `hf_model`: Optional pre-instantiated HF model (IT handles instantiation)
  - `tokenizer`: Optional tokenizer (IT handles instantiation)
  - `fold_value_biases`, `default_prepend_bos`: Behavior flags
- **Note**: These configs are **not** directly convertible to HookedTransformerConfig or TransformerBridgeConfig

### ITLensCustomConfig
Extends `ITLensSharedConfig` for config-based initialization:
- **Usage**: When providing explicit TransformerLens config (currently only supports HookedTransformerConfig)
- **Key Field**:
  - `cfg`: HookedTransformerConfig or dict convertible to it
- **Limitation**: Currently requires HookedTransformerConfig, cannot accept TransformerBridgeConfig directly
- **Note**: When `use_bridge=True` with `ITLensCustomConfig`, initialization will not use TransformerBridge — Interpretune will warn and force `use_bridge=False`, falling back to the legacy HookedTransformer path.

### ITLensConfig
Top-level IT configuration encapsulating all settings:
- **Key Fields**:
  - `tl_cfg`: Either ITLensFromPretrainedConfig or ITLensCustomConfig
  - `hf_from_pretrained_cfg`: HFFromPretrainedConfig (for HF model loading)
  - Various inherited IT core configs
- **Internal State**:
  - `_load_from_pretrained`: Boolean tracking initialization mode
  - `_dtype`: Resolved dtype
- **Methods**:
  - `_translate_tl_config()`: Maps TL config fields to IT fields (e.g., `hf_model` → `model_name_or_path`)
  - `_sync_pretrained_cfg()`: Syncs HF and TL dtypes, validates device_map
  - `_disable_pretrained_model_mode()`: Disables pretrained settings for custom config mode

## Config Flow During Initialization

### TransformerBridge Path (use_bridge=True)
1. User provides `ITLensFromPretrainedConfig` with `use_bridge=True` (default)
2. IT loads HF model via `model_name` using `hf_from_pretrained_cfg`
3. `_convert_hf_to_bridge()` is called:
   ```python
   # Map ITLensFromPretrainedConfig fields to TransformerBridgeConfig
   bridge_config = map_to_tl_config(hf_model.config, tl_config)
   bridge_config.architecture = hf_model.config.architectures[0]

   # Create adapter with TransformerBridgeConfig
   adapter = ArchitectureAdapterFactory.create_adapter(hf_model, bridge_config)

   # Create TransformerBridge
   model = TransformerBridge(hf_model, adapter, tokenizer)

   # Preserve original HF config
   model.config = hf_model.config  # HF PretrainedConfig
   ```
4. After initialization:
   - `self.model = TransformerBridge instance`
   - `self.model.cfg = TransformerBridgeConfig instance` (from adapter)
   - `self.model.config = HF PretrainedConfig` (original HF config)
   - `self.model.adapter.cfg = TransformerBridgeConfig instance` (same as model.cfg)

### HookedTransformer Path (use_bridge=False)
1. User provides `ITLensFromPretrainedConfig` with `use_bridge=False`
2. IT loads HF model via `model_name` using `hf_from_pretrained_cfg`
3. `_convert_hf_to_tl()` is called:
   ```python
   # Convert using TL's from_pretrained with weight conversion
   model = HookedTransformer.from_pretrained_no_processing(
       model_name=model_name,
       hf_model=hf_model,
       **filtered_kwargs
   )

   # Preserve original HF config
   model.config = hf_model.config  # HF PretrainedConfig
   ```
4. After initialization:
   - `self.model = HookedTransformer instance`
   - `self.model.cfg = HookedTransformerConfig instance` (created by TL)
   - `self.model.config = HF PretrainedConfig` (original HF config)

### Config-based Path (ITLensCustomConfig)
1. User provides `ITLensCustomConfig` with `cfg=HookedTransformerConfig`
2. `_load_from_pretrained = False` is set
3. `tl_config_model_init()` is called:
   ```python
   # Create HookedTransformer from config
   model = HookedTransformer(cfg=tl_cfg)
   ```
4. After initialization:
   - `self.model = HookedTransformer instance`
   - `self.model.cfg = HookedTransformerConfig instance` (provided by user)
   - No `self.model.config` (no original HF config)
   - **Cannot use TransformerBridge path** (requires HF model)

## Config Serialization Requirements

### Current _capture_hyperparameters Logic
Located in `src/interpretune/adapters/transformer_lens.py:_capture_hyperparameters()`:

```python
def _capture_hyperparameters(self) -> None:
   """Capture and serialize hyperparameters for model checkpointing.

   Current behavior:
   1. Serialize the actual TL model configuration (HookedTransformerConfig or TransformerBridgeConfig)
      derived from the initialized model instance (`self.model.cfg`) and store it under the
      `tl_model_cfg` key in the session `_init_hparams` so it can be used for reproducible recreation.
   2. Add a `_used_bridge` flag when possible to capture whether the bridge (v3) path was used.
   3. Store IT-specific TL settings under `it_tl_cfg` so IT-level configuration fields are preserved.
   4. Call the superclass implementation to capture the original HF `PretrainedConfig` (hf_preconversion_config).
   """

   # capture the Marshal-able TransformerLens model cfg from the runtime model instance
   tl_model_cfg = self._make_config_serializable(self.model.cfg, ["device", "dtype"])

   # Add architecture flag for clarity (used_bridge toggles the bridge vs legacy path)
   if hasattr(tl_model_cfg, "__dict__"):
      tl_model_cfg.__dict__["_used_bridge"] = self.it_cfg.tl_cfg.use_bridge

   # Save the serialized TransformerLens model config for checkpointing and reproduction
   self._it_state._init_hparams.update({"tl_model_cfg": tl_model_cfg})

   # Serialize IT-specific TL settings so they are available for the initialization flow
   self._it_state._init_hparams.update({"it_tl_cfg": self.it_cfg.tl_cfg})

   # Delegate to superclass to capture the original HF PretrainedConfig (hf_preconversion_config)
   super()._capture_hyperparameters()
```

### What Needs to be Serialized

#### For TransformerBridge Path (use_bridge=True):
1. **Original HF PretrainedConfig** (already preserved via `self.model.config`):
   - Source: `hf_model.config` (HuggingFace PretrainedConfig)
   - Purpose: Complete HF model configuration, required for reproducible recreation
   - Access: `self.model.config`

2. **TransformerBridgeConfig (runtime TL model config)** (serialized under `tl_model_cfg`):
   - Source: `self.model.cfg` (the authoritative TL config created during model initialization)
   - Purpose: TransformerLens v3 configuration including architecture info and device/dtype
   - Access: `self.model.cfg` or `self.model.adapter.cfg`
   - Stored in `_init_hparams` as `tl_model_cfg` (serializable via `_make_config_serializable`)

3. **ITLensFromPretrainedConfig** (IT-level settings)
   - Source: `self.it_cfg.tl_cfg` (the IT wrapper providing high-level runtime choices)
   - Purpose: IT-specific settings (fold_ln, center_writing_weights, use_bridge, etc.)
   - Stored under `_init_hparams` key `it_tl_cfg` so they are available for recreation and diagnostics

#### For HookedTransformer Path (use_bridge=False):
1. **Original HF PretrainedConfig** (already preserved via `self.model.config`):
   - Source: `hf_model.config` (HuggingFace PretrainedConfig)
   - Purpose: Complete HF model configuration
   - Access: `self.model.config`

2. **HookedTransformerConfig (runtime TL model config)** (serialized under `tl_model_cfg`):
   - Source: `self.model.cfg` (HookedTransformerConfig created during initialization)
   - Purpose: HookedTransformer model configuration details (d_mlp, activations, etc.)
   - Access: `self.model.cfg`
   - Stored in `_init_hparams` as `tl_model_cfg` for reproducibility

3. **ITLensFromPretrainedConfig** (IT-level settings)
   - Source: `self.it_cfg.tl_cfg`
   - Purpose: IT-specific settings
   - Stored under `_init_hparams` key `it_tl_cfg`

## Current Serialization Practice

We capture the runtime TL model configuration (`self.model.cfg`) — which is the authoritative
source of truth for the TransformerLens configuration used at runtime — and stores it in `_init_hparams` as `tl_model_cfg`.
This ensures that whether the module was initialized via a pretrained HF model (TransformerBridge path) or via a HookedTransformer
config, the true TL model configuration is captured and preserved.

Key points:
- The TF model's `self.model.cfg` (HookedTransformerConfig or TransformerBridgeConfig) is serialized and saved as
   `tl_model_cfg` in the `_init_hparams` map.
- A `_used_bridge` flag is stored alongside `tl_model_cfg` to clarify whether the bridge (v3) path was used.
- IT-specific TL settings are saved under the `it_tl_cfg` key so that high-level IT configuration choices are preserved.
- The superclass call continues to capture the original HF `PretrainedConfig` via `super()._capture_hyperparameters()`.

This approach simplifies config reconstruction at runtime and avoids the type confusion that previously existed when attempting to
serialize IT wrapper configs as if they were actual TL configs.


## Config Type Summary

| Config Class | Purpose | Initialization | Serialization Target |
|-------------|---------|----------------|---------------------|
| **TransformerLensConfig** | Base class for TL configs | N/A (abstract) | N/A |
| **HookedTransformerConfig** | Legacy TL config | Created by TL's from_pretrained | `self.model.cfg` |
| **TransformerBridgeConfig** | V3 TL config with architecture info | Created by map_to_tl_config + adapter | `self.model.cfg` |
| **ITLensSharedConfig** | Base IT TL settings | User provides | `self.it_cfg.tl_cfg` |
| **ITLensFromPretrainedConfig** | IT settings for from_pretrained | User provides | `self.it_cfg.tl_cfg` |
| **ITLensCustomConfig** | IT settings for config-based init | User provides | `self.it_cfg.tl_cfg` |
| **HF PretrainedConfig** | Original HF model config | Loaded with HF model | `self.model.config` |
