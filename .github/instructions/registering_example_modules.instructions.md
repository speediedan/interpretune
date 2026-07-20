---
applyTo: "**/example_module_registry.yaml"
---

# Registering Example Modules in Interpretune

## Overview

The `MODULE_EXAMPLE_REGISTRY` (defined in `src/it_examples/example_module_registry.yaml`) is the central registry for adapter configurations used in tests and examples. Session fixtures depend on this registry to resolve base configurations for different adapter combinations.

## Why Register Example Modules?

1. **Session Fixtures**: `it_session`, `it_session_cfg`, and `analysis_session` fixtures use `gen_session_cfg()` which calls `MODULE_EXAMPLE_REGISTRY.get(test_cfg)` to resolve base configurations.

2. **Test Organization**: Registered modules can be used by multiple tests without duplicating configuration.

3. **Adapter Validation**: The registry validates that adapter combinations are supported and properly configured.

## Registry Entry Structure

Each entry in `example_module_registry.yaml` follows this structure:

```yaml
# this key nomenclature isn't strictly enforced, can deviate where sensible
{model_src_key}.{model_cfg_key}.{adapter_name}:
  reg_info:
    model_src_key: {model_key}       # e.g., model this configuration uses "gpt2", "gemma2", "llama3"
    model_cfg_key: {specific_cfg_key}        # e.g., "rte", "rte_demo", "rte_base_test"
    adapter_combinations:            # List of supported adapter tuples
      - [core, {adapter}]            # Core adapter combination, there can be many different compositions supported
      - [lightning, {adapter}]       # Lightning adapter combination example, not every adapter supports lightning
    description: {description}       # Human-readable description
  shared_config:
    task_name: {dataset_task}        # Task name for dataset loading
    model_name_or_path: {model_id}   # HuggingFace model ID
    tokenizer_id_overrides:          # Optional tokenizer overrides
      pad_token_id: {id}
    tokenizer_kwargs:                # Tokenizer configuration
      model_input_names: ['input']
      padding_side: left
      add_bos_token: false
  registered_cfg:
    datamodule_cfg:                  # ITDataModuleConfig settings
      prompt_cfg:
        class_path: {prompt_config_class}
      signature_columns: ['input', 'labels']
      text_fields: ["premise", "hypothesis"]
      enable_datasets_cache: True
      train_batch_size: 2
      eval_batch_size: 2
    module_cfg:                      # ITConfig (or adapter-specific config)
      class_path: {module_config_class}
      init_args:
        # Adapter-specific configuration...
    datamodule_cls:
      class_path: tests.modules.FingerprintTestITDataModule  # could also be desired test class, or example class
    module_cls:
      class_path: tests.modules.TestITModule  # could also be another desired test class or example class
```

## Step-by-Step: Adding a New Adapter Registration

### Step 1: Identify the Registry Key Pattern

The registry key follows the pattern: `{model_src_key}.{model_cfg_key}.{adapter_name}`

Examples:
- `gpt2.rte.transformer_lens` - GPT-2 with TransformerLens adapter
- `gpt2.rte.sae_lens` - GPT-2 with SAE-Lens adapter
- `gpt2.rte.nnsight` - GPT-2 with NNsight adapter
- `gemma2.rte_base_test.circuit_tracer_tl` - Gemma2 with Circuit Tracer (TL backend)

### Step 2: Define Adapter Combinations

Adapter combinations determine which adapter compositions are supported:

```yaml
adapter_combinations:
  - [core, nnsight]              # Basic NNsight
  - [lightning, nnsight]         # Lightning + NNsight
  - [core, transformer_lens]     # Basic TransformerLens
  - [lightning, transformer_lens]  # Lightning + TransformerLens
  - [core, transformer_lens, circuit_tracer]  # Circuit Tracer with TL backend
```

### Step 3: Configure the Module Config

The `module_cfg` section varies by adapter. Key adapter configs:

**TransformerLens:**
```yaml
module_cfg:
  class_path: interpretune.config.module.ITConfig
  init_args:
    tl_cfg:
      class_path: interpretune.config.transformer_lens.ITLensFromPretrainedConfig
```

**SAE-Lens:**
```yaml
module_cfg:
  class_path: interpretune.config.module.ITConfig
  init_args:
    tl_cfg:
      class_path: interpretune.config.transformer_lens.ITLensFromPretrainedNoProcessingConfig
    sae_cfgs:
      - class_path: interpretune.config.sae_lens.SAELensFromPretrainedConfig
        init_args:
          release: gpt2-small-res-jb
          sae_id: blocks.0.hook_resid_pre
```

**NNsight:**
```yaml
module_cfg:
  class_path: interpretune.config.nnsight.ITNNsightConfig
  init_args:
    nnsight_cfg:
      class_path: interpretune.config.nnsight.NNsightConfig
      init_args:
        model_name: openai-community/gpt2
        device_map: cpu
        torch_dtype: float32
        dispatch: true
```

**Circuit Tracer:**
```yaml
module_cfg:
  class_path: interpretune.config.module.ITConfig
  init_args:
    tl_cfg:
      class_path: interpretune.config.transformer_lens.ITLensFromPretrainedNoProcessingConfig
    circuit_tracer_cfg:
      class_path: interpretune.config.circuit_tracer.CircuitTracerConfig
      init_args:
        backend: "transformerlens"
        transcoder_set: "gemma"
```

### Step 4: Create Test Configuration Class

In `tests/core/cfg_aliases.py`, create a configuration class that matches the registry entry:

```python
from tests.base_defaults import BaseCfg

@dataclass(kw_only=True)
class CoreNNsightGPT2(BaseCfg):
    """Core NNsight adapter with GPT-2 for unit testing.

    Registered in example_module_registry.yaml as gpt2.rte.nnsight.
    """

    phase: str = "test"
    model_src_key: str | None = "gpt2"      # Must match registry
    model_cfg_key: str = "rte"               # Must match registry
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.nnsight)  # Must match registry
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(
            model_name="openai-community/gpt2",
            device_map="cpu",
            torch_dtype="float32",
            dispatch=True,
        )
    )
```

**Critical**: The `(model_src_key, model_cfg_key, phase, adapter_ctx)` tuple must match an entry in the registry.

### Step 5: Register Fixture in conftest.py

Add the fixture configuration to `FIXTURE_CFGS` in `tests/conftest.py`:

```python
from tests.core.cfg_aliases import CoreNNsightGPT2, LightningNNsightGPT2

FIXTURE_CFGS = {
    # ... existing entries ...
    "ns_gpt2": FixtureCfg(
        test_cfg=CoreNNsightGPT2,
        scope="class",  # class-scoped for sharing across test methods
        variants={
            "it_session": [FixtPhase.initonly, FixtPhase.setup],
            "it_session_cfg": [FixtPhase.cfgonly],
        },
    ),
    "l_ns_gpt2": FixtureCfg(
        test_cfg=LightningNNsightGPT2,
        scope="class",
        variants={
            "it_session": [FixtPhase.initonly, FixtPhase.setup],
            "it_session_cfg": [FixtPhase.cfgonly],
        },
    ),
}
```

### Step 6: Verify Registration

Test that the registry lookup works:

```bash
cd /home/speediedan/repos/interpretune && \
source /mnt/cache/speediedan/.venvs/it_latest/bin/activate && \
python -c "
from it_examples.example_module_registry import MODULE_EXAMPLE_REGISTRY
from interpretune.protocol import Adapter

# Test lookup
key = ('gpt2', 'rte', 'test', (Adapter.core, Adapter.nnsight))
result = MODULE_EXAMPLE_REGISTRY.get(key)
print(f'Registry lookup succeeded: {result is not None}')
"
```

## Complete Example: NNsight Adapter Registration

### Registry Entry (`example_module_registry.yaml`)

```yaml
gpt2.rte.nnsight:
  reg_info:
    model_src_key: gpt2
    model_cfg_key: rte
    adapter_combinations:
      - [core, nnsight]
      - [lightning, nnsight]
    description: Basic NNsight example, GPT2 with supported adapter compositions
  shared_config:
    task_name: pytest_rte_hf
    model_name_or_path: openai-community/gpt2
    tokenizer_id_overrides:
      pad_token_id: 50256
    tokenizer_kwargs:
      model_input_names: ['input']
      padding_side: left
      add_bos_token: false
  registered_cfg:
    datamodule_cfg:
      prompt_cfg:
        class_path: it_examples.experiments.rte_boolq.RTEBoolqPromptConfig
      signature_columns: ['input', 'labels']
      text_fields: ["premise", "hypothesis"]
      enable_datasets_cache: True
      train_batch_size: 2
      eval_batch_size: 2
    module_cfg:
      class_path: interpretune.config.nnsight.ITNNsightConfig
      init_args:
        auto_comp_cfg:
          class_path: interpretune.config.shared.AutoCompConfig
          init_args:
            module_cfg_name: RTEBoolqConfig
            module_cfg_mixin:
              class_path: it_examples.experiments.rte_boolq.RTEBoolqEntailmentMapping
              import_only: True
            target_adapters: nnsight
        hf_from_pretrained_cfg:
          class_path: interpretune.config.mixins.HFFromPretrainedConfig
          init_args:
            pretrained_kwargs:
              device_map: cpu
              dtype: float32
            model_head: transformers.GPT2LMHeadModel
        nnsight_cfg:
          class_path: interpretune.config.nnsight.NNsightConfig
          init_args:
            model_name: openai-community/gpt2
            device_map: cpu
            torch_dtype: float32
            dispatch: true
            tokenizer_kwargs:
              padding_side: left
              add_bos_token: false
    datamodule_cls:
      class_path: tests.modules.FingerprintTestITDataModule
    module_cls:
      class_path: tests.modules.TestITModule
```

## Troubleshooting

### KeyError: Module not found in registry

If you see:
```
KeyError: "A module registered with `('gpt2', 'rte', 'test', (<Adapter.core: 'core'>, <Adapter.nnsight: 'nnsight'>))` was not found in the registry."
```

**Check:**
1. The registry entry exists in `example_module_registry.yaml`
2. The `model_src_key`, `model_cfg_key`, and `adapter_combinations` match your test config
3. YAML syntax is correct (indentation, colons, etc.)

### Adapter Not Recognized

If the adapter isn't recognized:
1. Ensure the adapter is registered in `interpretune.protocol.Adapter` enum
2. Verify the adapter module is in `src/interpretune/adapters/`
3. Check the adapter registration in `CompositionRegistry`

### Tests Marked Standalone Still Required

If tests require `@RunIf(standalone=True)` despite registry entry:
1. Verify the test config class has correct `model_cfg_key` attribute
2. Check that fixture variants include `it_session_cfg` with `cfgonly` phase
3. Ensure fixture scope is appropriate (`class` or `module` for shared fixtures)

## Related Files

- `src/it_examples/example_module_registry.yaml` - Main registry file
- `src/it_examples/example_module_registry.py` - Registry loader
- `tests/configuration.py` - `gen_session_cfg()` function
- `tests/conftest.py` - `FIXTURE_CFGS` dictionary
- `tests/core/cfg_aliases.py` - Test configuration classes
- `.github/instructions/fixture_usage.instructions.md` - Fixture usage guide
- `docs/framework_level_adapters.md` - Framework-level adapter comparison (core vs Lightning)

## Dual-Backend Analysis Registration

Analysis operations can run on multiple backends (TransformerBridge, NNsight). For backend parity testing, register separate analysis config classes that share the same `model_src_key` and `model_cfg_key` but differ in `adapter_ctx`:

### Bridge SAE Analysis Config

```python
@dataclass(kw_only=True)
class CoreSLBridgeGPT2LogitDiffsSAE(BaseCfg):
    """TransformerBridge backend for SAE logit_diffs analysis."""
    model_src_key: str | None = "gpt2"
    model_cfg_key: str = "rte"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens, Adapter.transformer_lens)
    tl_cfg: ITLensFromPretrainedNoProcessingConfig = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gpt2-small", default_padding_side="left", use_bridge=True  # Bridge for parity testing
        )
    )
    analysis_cfg: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(ops=["logit_diffs_sae"]))
```

### NNsight SAE Analysis Config

```python
@dataclass(kw_only=True)
class CoreSLNNsightGPT2LogitDiffsSAE(BaseCfg):
    """NNsight backend for SAE logit_diffs analysis."""
    model_src_key: str | None = "gpt2"
    model_cfg_key: str = "rte"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.sae_lens, Adapter.nnsight)
    nnsight_cfg: NNsightConfig | None = field(
        default_factory=lambda: NNsightConfig(model_name="openai-community/gpt2", ...)
    )
    analysis_cfg: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(ops=["logit_diffs_sae"]))
```

Both resolve to the same `gpt2.rte` registry entry but use different adapter stacks. The registry entry must list both `[core, sae_lens, transformer_lens]` and `[core, sae_lens, nnsight]` in `adapter_combinations`.

## Module Definition Guidelines

When creating `module_cls` classes for registry entries:

- **No framework-specific hooks** — Module definitions should not implement `on_test_epoch_end` or accumulation logic. Use `ClassificationMixin` for metrics.
- **Use `self.log()` / `self.log_dict()`** — These route to `CoreHelperAttributes` (core) or `LightningModule` (Lightning) automatically via MRO.
- **`ClassificationMixin.setup()`** handles `classification_mapping` init — no need to duplicate in module `setup()`.
- See `docs/framework_level_adapters.md` for the full core vs Lightning comparison.
