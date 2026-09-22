# TransformerLens Config Hierarchy Overview

## Overview
This document provides an overview of the TransformerLens configuration hierarchy and its integration with Interpretune configurations.

TransformerLens 4.0 removed the legacy `Hooked*` model stack, so `TransformerBridge` is the only model path and `TransformerBridgeConfig` the only TL config. The `use_bridge` flag that selected between them is retired: it had one reachable value left, and passing it now fails by name rather than being accepted and ignored.

## TransformerLens Config Hierarchy

### Base Class: TransformerLensConfig
Stored in `transformer_lens/config/TransformerLensConfig.py`:
- Defines core model dimensions: `d_model`, `d_head`, `n_layers`, `n_ctx`, `d_vocab`, `n_heads`

### TransformerBridgeConfig
Extends `TransformerLensConfig` as a regular class:
- **Location**: `transformer_lens/config/TransformerBridgeConfig.py`
- **Usage**: Wraps HF models without weight conversion, more memory efficient
- **Key Additions**:
  - `architecture`: Architecture identifier for adapter selection (e.g., "gpt2", "llama")
  - `tokenizer_prepends_bos`: Tokenizer behavior configuration
  - `default_padding_side`: Padding side configuration
  - `split_attention_weights`: Attention weight processing configuration
- **Compatibility**: Carries the fields the removed `HookedTransformerConfig` declared, so configs written
  against it keep working
- **Note**: The bridge does not convert weights. `enable_compatibility_mode()` applies the processing the
  legacy path did by default (LayerNorm folding, `center_writing_weights`, `center_unembed`); omit it to work
  with raw HF weights

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
Common configuration shared across the initialization modes:
- **Location**: `src/interpretune/config/transformer_lens.py`
- **Key Fields**:
  - `move_to_device`: Control device movement (default: True)
  - `default_padding_side`: Padding side for tokenizer (default: "right")
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
- **Note**: These configs are **not** directly convertible to `TransformerBridgeConfig`

### ITLensCustomConfig
Extends `ITLensSharedConfig` for config-based initialization:
- **Usage**: When providing an explicit TransformerLens config rather than loading pretrained weights
- **Key Field**:
  - `cfg`: `TransformerBridgeConfig` or dict convertible to it
- **Note**: No longer special-cased. It was previously forced onto the legacy path because a bridge could not
  be built without an HF model; `TransformerBridge.boot_native` removes that premise, so a custom config
  builds a bridge like any other.
- **Note**: `boot_native` does NOT infer TransformerLens' `-1` vocab sentinels from the tokenizer the way
  `HookedTransformer.__init__` did, so the adapter resolves `d_vocab` / `d_vocab_out` before booting. A config
  that omitted `d_vocab` and relied on inference would otherwise fail inside `nn.Embedding`.

### ITLensBridgeConfig
Extends `ITLensSharedConfig` for explicit TransformerBridge (v3) configuration:
- **Usage**: When you want fine-grained control over TransformerBridge initialization and compatibility mode settings
- **Key Fields**:
  - `model_name`: The HF model name/path for TransformerBridge (e.g., "gemma-2-2b-it")
  - `enable_compatibility_mode`: Whether to call enable_compatibility_mode() on the bridge after instantiation (default: False)
  - `enable_compatibility_mode_kwargs`: Optional kwargs for enable_compatibility_mode() (e.g., `fold_ln`, `fold_value_biases`)
  - `transformer_bridge_config_overrides`: Optional kwargs to pass to TransformerBridgeConfig constructor
  - `device`, `dtype`: Device/dtype configuration
- **Note**: This is the bridge-native config, and the one to use when you want compatibility mode or
  `TransformerBridgeConfig` overrides. `ITLensFromPretrainedConfig` also produces a bridge; it simply
  expresses the from-pretrained loading path instead.

### ITLensConfig
Top-level IT configuration encapsulating all settings:
- **Key Fields**:
  - `tl_cfg`: `ITLensFromPretrainedConfig`, `ITLensCustomConfig` or `ITLensBridgeConfig`
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

### Pretrained Path
1. User provides `ITLensFromPretrainedConfig` (or `ITLensBridgeConfig`)
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

### Config-based Path (ITLensCustomConfig)
1. User provides `ITLensCustomConfig` with `cfg=TransformerBridgeConfig`
2. `_load_from_pretrained = False` is set
3. `tl_config_model_init()` is called:
   ```python
   # Resolve TL's -1 vocab sentinels from the tokenizer first: boot_native passes cfg.d_vocab
   # straight into nn.Embedding and does not infer it the way HookedTransformer.__init__ did.
   self._resolve_vocab_sentinels(cfg, tokenizer)

   # Build a bridge around a randomly-initialized TL-native model: no HF model, no Hub call
   model = TransformerBridge.boot_native(cfg, tokenizer=tokenizer)
   ```
4. After initialization:
   - `self.model = TransformerBridge instance`
   - `self.model.cfg = TransformerBridgeConfig instance` (provided by user)
   - No `self.model.config` (no original HF config)

## Config Serialization Requirements

### Current _capture_hyperparameters Logic
Located in `src/interpretune/adapters/transformer_lens.py:_capture_hyperparameters()`:

```python
def _capture_hyperparameters(self) -> None:
   """Capture and serialize hyperparameters for model checkpointing.

   Current behavior:
   1. Serialize the actual `TransformerBridgeConfig` derived from the initialized model instance
      (`self.model.cfg`) and store it under the `tl_model_cfg` key in the session `_init_hparams` so it
      can be used for reproducible recreation.
   2. Store IT-specific TL settings under `it_tl_cfg` so IT-level configuration fields are preserved.
   3. Call the superclass implementation to capture the original HF `PretrainedConfig` (hf_preconversion_config).
   """

   # capture the Marshal-able TransformerLens model cfg from the runtime model instance
   tl_model_cfg = self._make_config_serializable(self.model.cfg, ["device", "dtype"])

   # Save the serialized TransformerLens model config for checkpointing and reproduction
   self._it_state._init_hparams.update({"tl_model_cfg": tl_model_cfg})

   # Serialize IT-specific TL settings so they are available for the initialization flow
   self._it_state._init_hparams.update({"it_tl_cfg": self.it_cfg.tl_cfg})

   # Delegate to superclass to capture the original HF PretrainedConfig (hf_preconversion_config)
   super()._capture_hyperparameters()
```

### What Needs to be Serialized

1. **Original HF PretrainedConfig** (already preserved via `self.model.config`):
   - Source: `hf_model.config` (HuggingFace PretrainedConfig)
   - Purpose: Complete HF model configuration, required for reproducible recreation
   - Access: `self.model.config`
   - Absent on the config-only path (`ITLensCustomConfig`), which never loads an HF model

2. **TransformerBridgeConfig (runtime TL model config)** (serialized under `tl_model_cfg`):
   - Source: `self.model.cfg` (the authoritative TL config created during model initialization)
   - Purpose: TransformerLens configuration including architecture info and device/dtype
   - Access: `self.model.cfg` or `self.model.adapter.cfg`
   - Stored in `_init_hparams` as `tl_model_cfg` (serializable via `_make_config_serializable`)

3. **The IT-level tl_cfg** (`ITLensFromPretrainedConfig`, `ITLensBridgeConfig` or `ITLensCustomConfig`)
   - Source: `self.it_cfg.tl_cfg` (the IT wrapper providing high-level runtime choices)
   - Purpose: IT-specific settings (`fold_ln`, `center_writing_weights`, `enable_compatibility_mode`, ...)
   - Stored under `_init_hparams` key `it_tl_cfg` so they are available for recreation and diagnostics

## Current Serialization Practice

We capture the runtime TL model configuration (`self.model.cfg`) — which is the authoritative
source of truth for the TransformerLens configuration used at runtime — and stores it in `_init_hparams` as `tl_model_cfg`.
This ensures that whether the module was initialized from pretrained HF weights or from a config alone
(`boot_native`), the true TL model configuration is captured and preserved.

Key points:
- The model's `self.model.cfg` (a `TransformerBridgeConfig`) is serialized and saved as `tl_model_cfg` in the
   `_init_hparams` map.
- IT-specific TL settings are saved under the `it_tl_cfg` key so that high-level IT configuration choices are preserved.
- The superclass call continues to capture the original HF `PretrainedConfig` via `super()._capture_hyperparameters()`.

This approach simplifies config reconstruction at runtime and avoids the type confusion that previously existed when attempting to
serialize IT wrapper configs as if they were actual TL configs.


## Config Type Summary

| Config Class | Purpose | Initialization | Serialization Target |
|-------------|---------|----------------|---------------------|
| **TransformerLensConfig** | Base class for TL configs | N/A (abstract) | N/A |
| **TransformerBridgeConfig** | The TL config, with architecture info | Created by map_to_tl_config + adapter, or supplied for `boot_native` | `self.model.cfg` |
| **ITLensSharedConfig** | Base IT TL settings | User provides | `self.it_cfg.tl_cfg` |
| **ITLensFromPretrainedConfig** | IT settings for the from_pretrained path | User provides | `self.it_cfg.tl_cfg` |
| **ITLensBridgeConfig** | IT settings for a bridge-native config (compatibility mode, overrides) | User provides | `self.it_cfg.tl_cfg` |
| **ITLensCustomConfig** | IT settings for config-based init | User provides | `self.it_cfg.tl_cfg` |
| **HF PretrainedConfig** | Original HF model config | Loaded with HF model | `self.model.config` |
