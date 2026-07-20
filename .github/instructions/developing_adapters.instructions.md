---
applyTo: "**/src/interpretune/adapters/**"
---

# Developing Adapters for Interpretune

## Overview

This instruction file provides guidance for developing adapters within the interpretune framework. Follow these patterns when creating new adapters or modifying existing ones.

## Key Files to Reference

- `docs/adapter_development_guide.md` - Comprehensive adapter development guide
- `src/interpretune/adapters/nnsight.py` - Reference implementation for model wrapper adapters
- `src/interpretune/adapters/transformer_lens.py` - Reference implementation for model wrapper adapters
- `src/interpretune/adapters/circuit_tracer.py` - Reference implementation for analysis adapters

## Adapter Development Checklist

### 1. Protocol Enum
- [ ] Add adapter to `Adapter` enum in `src/interpretune/protocol.py`

### 2. Configuration
- [ ] Create config dataclass in `src/interpretune/config/` (e.g., `my_adapter.py`)
- [ ] Inherit from `ITSerializableCfg` for serialization support
- [ ] Implement `__post_init__` for validation
- [ ] Export from `src/interpretune/config/__init__.py`
- [ ] Create `ITMyAdapterConfig(ITConfig)` if extending ITConfig

### 3. Adapter Module
- [ ] Create adapter module in `src/interpretune/adapters/` (e.g., `my_adapter.py`)
- [ ] Create `MyAttributeMixin` for property access
- [ ] Create `BaseMyModule(BaseITModule)` for model initialization
- [ ] Create `MyAdapter` with `register_adapter_ctx` classmethod
- [ ] Create composed module classes for each adapter combination

### 4. Registration
- [ ] Register all adapter combinations in `register_adapter_ctx`
- [ ] Add module to `adapter_modules` tuple in `_light_register.py`
- [ ] Add lazy exports to `adapters/__init__.py` `_LAZY_ADAPTER_ATTRS`

### 5. Testing
- [ ] Create parity test file in `tests/parity_acceptance/test_it_<adapter>.py`
- [ ] Add config aliases to `tests/parity_acceptance/cfg_aliases.py`
- [ ] Add expected results to `tests/parity_acceptance/expected.py`
- [ ] Add expected warnings to `tests/warns.py`
- [ ] Add profiling entries to `profile_memory_footprints.yaml`

## Composition Patterns

### Mixin Order (MRO)
Place classes in this order for proper method resolution:
1. Backend-specific mixins (e.g., `MyBackendMixin`)
2. Attribute mixins (e.g., `MyAttributeMixin`)
3. Base adapter modules (e.g., `BaseMyModule`)
4. Framework adapters (e.g., `LightningAdapter`)
5. Base module (`BaseITModule`)
6. Framework modules (e.g., `LightningModule`)

### Example Composition
```python
class MyLightningModule(
    MyBackendMixin,          # 1. Backend-specific
    MyAttributeMixin,        # 2. Attribute access
    BaseMyModule,            # 3. Base adapter
    LightningAdapter,        # 4. Framework adapter
    BaseITModule,            # 5. Base module
    LightningModule,         # 6. Framework module
): ...
```

## NNsight Adapter Patterns

### Batched NNsight analysis

NNsight analysis operations chunk batched hook configurations through the
backend layer rather than a public `NNsightForwardContext` helper:

```python
from interpretune.analysis.backends.nnsight import NNsightModelBackend

backend = NNsightModelBackend(hook_resolver=resolver, configs_per_pass=8)

logits_per_config = backend.fwd_w_hooks_batched(
    model=nnsight_model,
    batch=batch,
    latent_model_handles=latent_model_handles,
    hook_configs=hook_configs,
    configs_per_pass=8,
)
```

`configs_per_pass` is the current branch name for the old per-trace invoke
limit concept. Use smaller values when debugging memory pressure.

### Hook Name Mapping (TL ↔ NNsight)

The `HookNameResolver` translates between TransformerLens hook names and NNsight module paths:

- TL hook: `blocks.0.hook_resid_pre` → NNsight path: `model.transformer.h.0` (resolved via architecture mapping)
- The resolver uses `_TL_HOOK_NNSIGHT_MAP` with regex patterns for standard HookedTransformer hooks

### Dual-Backend SAE Analysis

SAE analysis operations (`logit_diffs_sae`, `logit_diffs_attr_grad`, `logit_diffs_attr_ablation`) support both:
- **TransformerBridge backend**: SAE splicing via TL hook-based injection
- **NNsight backend**: SAE splicing via thread-interleaved tracing

Backend selection is automatic via `get_backend_for_module()` — if the module has `nnsight_cfg`, NNsight backend is used.

## Framework-Agnostic Logging

Adapters must support the `log()` / `log_dict()` contract used by mixins like `ClassificationMixin`:

- **Core context**: `CoreHelperAttributes` provides real `log()` / `log_dict()` methods that accumulate metrics in `_logged_metrics`. The core runner prints averaged metrics at test epoch end.
- **Lightning context**: `LightningModule.log()` / `log_dict()` route to configured loggers with full `prog_bar`, `sync_dist`, and `on_step` / `on_epoch` support.
- **User code** calls `self.log()` / `self.log_dict()` regardless of context — the MRO determines which implementation runs.

When developing a new framework adapter, ensure it provides `log()` / `log_dict()` methods or inherits them from the framework module class.

## Hook Lifecycle

Lifecycle hooks are dispatched via `_call_itmodule_hook(module, hook_name=..., optional=True)`:

- **`setup()`**: `ClassificationMixin.setup()` → `super().setup()` → `BaseITHooks.setup()`. Classification mapping init lives in the mixin; datamodule and directory setup live in `BaseITHooks`.
- **`on_session_end()`**: Lives in `BaseITHooks`. Called by `on_train_end`, `on_test_end`, `on_predict_end`.
- **Phase-specific batch hooks**: `on_test_batch_start`, `on_train_batch_start` — fired by `run_step()` on the first step only, both optional.

New adapters should not override these hooks unless they need adapter-specific behavior. The cooperative `super()` chain ensures all mixins in the MRO are invoked.


## Common Mistakes to Avoid

1. **Circular imports**: Use `TYPE_CHECKING` for type hints
2. **Missing registrations**: Register all adapter combinations
3. **Wrong MRO order**: Mixins before base classes
4. **Incomplete exports**: Add to both lazy exports and light registration
5. **Missing config sync**: Handle model_name_or_path synchronization
