# TransformerLens-Style Parameter Naming Implementation

## Overview

This document describes the implementation of TL-style parameter naming support in FinetuningScheduler (FTS) for use with TransformerBridge models. This feature fosters greater cross-architecture schedule interoperability, allowing users to define fine-tuning schedules using cleaner, more intuitive parameter names from TransformerLens's standardized naming convention.

## Motivation

TransformerBridge models (TransformerLens v3) expose parameters through two interfaces:

1. **Canonical naming** (`named_parameters()`): Uses `_original_component` wrapper prefixes but provides access to canonical parameter names.
   - GPT-2 Example:
   - Parameter name: `model.blocks.9._original_component.attn.q._original_component.weight`
   - Total: 221 parameters (includes LayerNorms and potentially unused parameters like joint QKV params)

2. **TL-style naming** (`tl_named_parameters()`): Clean, standardized names that abstract away architecture-specific details
   - GPT-2 Example:
   - Parameter name: `blocks.9.attn.W_Q`
   - Total: 148 parameters (excludes LayerNorms and maps to used canonical parameters)

The canonical names are necessary for the optimizer's internal state and provide more granular control, but TL-style names are more intuitive for schedule authoring. The TransformerBridgeStrategyAdapter now supports defining fine-tuning schedules using both naming conventions, allowing users to choose the approach that best fits a given use case.

## Architecture

### Phase 1: FTS Core Changes

**Modified Files:**
- `finetuning-scheduler/src/finetuning_scheduler/strategy_adapters/base.py`
- `finetuning-scheduler/src/finetuning_scheduler/fts.py`
- `finetuning-scheduler/src/finetuning_scheduler/fts_supporters.py`

**Changes:**
1. Moved `gen_ft_schedule` from static method in `ScheduleImplMixin` to instance method on `StrategyAdapter` base class
2. Updated both call sites to use `self.strategy_adapter.gen_ft_schedule()`
3. Added deprecation warning (deprecated in 2.10.0, removal in 2.12.0) using `rank_zero_warn`

**Rationale:** Strategy adapters need to customize schedule generation to handle model-specific naming conventions.

### Phase 2: TransformerBridgeStrategyAdapter Implementation

**Modified File:**
- `interpretune/src/interpretune/adapters/transformer_lens.py`

**Key Components:**

#### 1. Initialization (`__init__`)
```python
def __init__(self, use_tl_names: bool = False):
    super().__init__()
    self.use_tl_names = use_tl_names
    self._tl_to_canonical_map: Dict[str, List[str]] = {}
    self._canonical_to_tl_map: Dict[str, str] = {}
```

#### 2. Parameter Mapping (`_build_param_mapping`)
- Uses `tensor.data_ptr()` for robust matching (handles views and aliases correctly)
- Builds bidirectional mappings:
  - **TL→canonical** (1:many): One TL name may map to multiple canonical names (e.g., views)
  - **canonical→TL** (1:1): Each canonical name maps to exactly one TL name
- Called during initialization if `use_tl_names=True`

#### 3. Schedule→Optimizer Translation (`fts_optim_transform`)
- Translates TL-style parameter patterns in schedules to canonical names for optimizer
- Handles regex patterns: `r"blocks.9.*"` → `r"model.blocks.9.*"`
- Preserves non-TL parameters unchanged (e.g., manually added classifier heads)

#### 4. Optimizer→Schedule Translation (`logical_param_translation`)
- Translates canonical parameter names back to TL-style for logging/display
- Uses `_canonical_to_tl_map` for efficient lookup
- Falls back to canonical names for unmapped parameters

#### 5. ModelView Abstraction
- Encapsulates parameter naming transformations in a separate ModelView class
- Two implementations:
  - **CanonicalModelView**: Identity transformation (default)
  - **TLNamesModelView**: TL-style naming with bidirectional mapping
- Each ModelView delegates to base StrategyAdapter methods:
  - `gen_schedule()` → `StrategyAdapter.gen_ft_schedule(self.adapter, dump_loc)`
  - `validate_schedule()` → `StrategyAdapter.validate_ft_sched(self.adapter)`
- This pattern avoids callable parameter passing and provides clean delegation

### Phase 3: Test Infrastructure

**Modified Files:**
- `interpretune/tests/conftest.py`
- `interpretune/tests/parity_acceptance/cfg_aliases.py`
- `interpretune/tests/parity_acceptance/test_it_fts.py`
- `interpretune/tests/parity_acceptance/expected.py`

**Test Components:**

1. **Schedule Transform** (`tl_bridge_multiphase_explicit_tl_names`):
   - Defines parameter patterns using TL-style names
   - Example: `r"blocks.(9|1[0-1]).*"` instead of `r"model.blocks.(9|1[0-1]).*"`

2. **Test Configuration** (`l_tl_bridge_gpt2_fts_multiphase_tl_names`):
   - Configures FTS with `strategy_adapter_cfg={'use_tl_names': True}`
   - Uses custom strategy adapter via entry point
   - Reuses existing test infrastructure (DivergeTestITModule, etc.)

3. **Parity Test** (`test_parity_fts[train_cuda_32_l_tl_bridge_fts_tl_names]`):
   - Validates that training dynamics work correctly with TL-style naming
   - Verifies checkpoint compatibility
   - Ensures optimizer state dict uses canonical names internally
   - NOTE: Parameter counts may differ from HookedTransformer due to:
     * LayerNorm handling (explicit vs implicit_ln_thaw)
     * Unembed weight tying (depends on enable_compatibility_mode)
     * Joint vs split QKV parameters in underlying architecture

## Usage

### Enabling TL-Style Naming

```python
from interpretune.adapters.transformer_lens import TransformerBridgeStrategyAdapter
from finetuning_scheduler import FinetuningScheduler

# Create strategy adapter with TL-style naming
strategy_adapter_cfg = {'use_tl_names': True}

# Pass to FTS callback
fts_callback = FinetuningScheduler(
    strategy_adapter_cfg=strategy_adapter_cfg,
    # ... other config
)
```

### Schedule Example (TL-Style)

```yaml
0:
  params:
    - blocks.(9|1[0-1]).*  # Layers 9-11
1:
  params:
    - blocks.([7-8]).*     # Layers 7-8
2:
  params:
    - blocks.([0-6](?!\d)).*  # Layers 0-6
    - (pos_embed|embed|unembed).*      # Embeddings
```

### Schedule Example (Canonical - Default)

```yaml
0:
  params:
    - model.blocks.(9|1[0-1]).*  # Layers 9-11
1:
  params:
    - model.blocks.([7-8]).*     # Layers 7-8
2:
  params:
    - model.blocks.([0-6](?!\d)).*  # Layers 0-6
    - model.(pos_embed|embed|unembed).*      # Embeddings
```

## Implementation Details

### Parameter Mapping Strategy

The implementation uses `tensor.data_ptr()` for matching because:
1. **Handles views correctly**: Different parameter names may reference the same underlying memory
2. **Robust to name changes**: Matching is based on actual tensor data, not string heuristics
3. **Efficient**: O(1) lookup via dictionary mapping

Example mapping:
```python
# TL→canonical (1:many) (in practice, in most cases 1:1)
{
    'blocks.9.attn.W_Q': [
        'model.blocks.9._original_component.attn.q._original_component.weight'
    ],
    'blocks.9.attn.b_Q': [
        'model.blocks.9._original_component.attn.q._original_component.bias'
    ],
}

# canonical→TL (1:1)
{
    'model.blocks.9._original_component.attn.q._original_component.weight': 'blocks.9.attn.W_Q',
    'model.blocks.9._original_component.attn.q._original_component.bias': 'blocks.9.attn.b_Q',
}
```

### Translation Functions

**fts_optim_transform** (Schedule→Optimizer):
```python
def fts_optim_transform(self, orig_pl: List[List[str]]) -> List[List[str]]:
    """Translate TL-style patterns to canonical names for optimizer."""
    # Delegate to the active ModelView for transformation
    return self.model_view.transform_to_canonical(orig_pl, inspect_only=False)
```

**logical_param_translation** (Optimizer→Schedule):
```python
def logical_param_translation(self, param_names: List[str]) -> List[str]:
    """Translate canonical names to TL-style for logging/display."""
    # Delegate to the active ModelView for transformation
    return self.model_view.transform_from_canonical(param_names)
```

**ModelView Delegation Pattern:**
```python
class CanonicalModelView(ModelView):
    def gen_schedule(self, dump_loc: Union[str, os.PathLike]) -> Optional[os.PathLike]:
        """Generate schedule with canonical naming."""
        from finetuning_scheduler.strategy_adapters.base import StrategyAdapter
        return StrategyAdapter.gen_ft_schedule(self.adapter, dump_loc)

    def validate_schedule(self) -> Tuple[int, int]:
        """Validate schedule with canonical naming."""
        from finetuning_scheduler.strategy_adapters.base import StrategyAdapter
        return StrategyAdapter.validate_ft_sched(self.adapter)

class TLNamesModelView(ModelView):
    def gen_schedule(self, dump_loc: Union[str, os.PathLike]) -> Optional[os.PathLike]:
        """Generate schedule using TL-style naming."""
        # Custom implementation using tl_named_parameters()
        ...

    def validate_schedule(self) -> Tuple[int, int]:
        """Validate schedule with TL-style diagnostics."""
        # Log mapping diagnostics
        rank_zero_debug(...)

        # Delegate to base StrategyAdapter
        from finetuning_scheduler.strategy_adapters.base import StrategyAdapter
        return StrategyAdapter.validate_ft_sched(self.adapter)
```

## Testing

### Parity Test Strategy

The parity test validates that TL-style naming mode produces identical training dynamics:

1. **Training Metrics**: Loss, accuracy, etc. should match exactly
2. **Callback State**: FTS depth transitions, parameter counts should be identical
3. **Checkpoint Compatibility**: Saved checkpoints should restore correctly

### Test Configuration

```python
# Test alias: train_cuda_32_l_tl_bridge_fts_tl_names
FTSParityTest(
    alias="train_cuda_32_l_tl_bridge_fts_tl_names",
    cfg=FTSParityCfg(
        **l_tl_bridge_gpt2_fts_multiphase_tl_names,
        **cuda
    ),
    marks="cuda_alone",
)
```

## Backward Compatibility

### Default Behavior
- `use_tl_names=False` (default): Uses canonical naming (backward compatible)
- All existing schedules continue to work unchanged

### Deprecation
- Static `ScheduleImplMixin.gen_ft_schedule()` deprecated in 2.10.0
- Will be removed in 2.12.0
- Uses `rank_zero_warn` for deprecation notice

### Migration Path
1. Users can opt-in to TL-style naming via `strategy_adapter_cfg`
2. No breaking changes to existing functionality
3. Clear deprecation warnings guide users to new API

## Future Work

### Potential Enhancements
1. **Auto-detection**: Automatically enable `use_tl_names` for TransformerBridge models
2. **Mixed naming**: Support mixing TL-style and canonical names in same schedule
3. **Custom mappings**: Allow users to define custom name translation rules
4. **Validation**: Enhance schedule validation to catch more common error patterns

### Related Features
1. **FSDP support**: Ensure TL-style naming works with FSDP strategy
2. **Documentation**: User guide with examples and best practices

## References

- [FinetuningScheduler Documentation](https://finetuning-scheduler.readthedocs.io/)
