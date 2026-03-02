# Interpretune Adapter Development Guide

This guide documents the patterns and best practices for developing adapters within the interpretune framework.

## Overview

Interpretune uses a composition-based adapter system that allows for flexible integration with different ML frameworks (TransformerLens, NNsight, SAE-Lens) and training systems (PyTorch Lightning). The adapter composition system enables mixing and matching different capabilities through multiple inheritance composition.

## Core Concepts

### 1. Adapter Types

Adapters fall into several categories:

| Adapter Type | Purpose | Examples |
|-------------|---------|----------|
| **Core** | Base functionality | `CoreAdapter`, `ITModule` |
| **Framework** | Training/execution framework | `LightningAdapter` |
| **Model Wrapper** | Model access patterns | `TransformerLensAdapter`, `NNsightAdapter` |
| **Analysis** | Analysis capabilities | `SAELensAdapter`, `CircuitTracerAdapter` |

### 2. Composition Registry

The `CompositionRegistry` manages adapter combinations:

```python
from interpretune.adapters import CompositionRegistry
from interpretune.protocol import Adapter

class MyAdapter:
    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        adapter_ctx_registry.register(
            Adapter.my_adapter,  # Primary adapter identifier
            component_key="module",  # Component type: 'module', 'datamodule', 'module_cfg'
            adapter_combination=(Adapter.core, Adapter.my_adapter),  # Composition tuple
            composition_classes=(MyModule,),  # Classes to compose
            description="My adapter for XYZ functionality",
        )
```

### 3. Adapter Combinations

Adapters are composed using tuples:
- `(Adapter.core, Adapter.transformer_lens)` - Core + TransformerLens
- `(Adapter.lightning, Adapter.nnsight)` - Lightning + NNsight
- `(Adapter.lightning, Adapter.transformer_lens, Adapter.circuit_tracer)` - Lightning + TL + CT

## Adapter Architecture Patterns

### Pattern 1: Attribute Mixin

Mixins provide consistent property access across compositions:

```python
class MyAttributeMixin:
    """Mixin providing property access for adapter-specific attributes."""

    it_cfg: ITConfig  # Type hint for composition

    @property
    def my_config(self) -> MyConfig | None:
        """Get adapter configuration from ITConfig."""
        if hasattr(self.it_cfg, "my_cfg"):
            return self.it_cfg.my_cfg
        return None

    @property
    def device(self) -> torch.device | None:
        """Get the device from model or state."""
        # Adapter-specific device resolution
        pass
```

### Pattern 2: Base Module

Base modules provide model initialization and core functionality:

```python
class BaseMyModule(BaseITModule):
    """Base module for adapter integration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._my_model_handle: Any | None = None

    def auto_model_init(self) -> None:
        """Initialize model using adapter-specific logic."""
        if self.my_config is not None:
            self._init_my_model()
        else:
            self.model_init()  # Fallback to base

    def _init_my_model(self) -> None:
        """Adapter-specific model initialization."""
        from my_library import MyModel

        cfg = self.my_config
        model_name = cfg.model_name or self.it_cfg.model_name_or_path

        self.model = MyModel(model_name, **cfg.get_init_kwargs())
```

### Pattern 3: Adapter Registration

Adapters register themselves with the composition registry:

```python
class MyAdapter(MyAttributeMixin):
    """Main adapter class for registration and composition."""

    @classmethod
    def register_adapter_ctx(cls, adapter_ctx_registry: CompositionRegistry) -> None:
        # Core compositions
        adapter_ctx_registry.register(
            Adapter.my_adapter,
            component_key="module",
            adapter_combination=(Adapter.core, Adapter.my_adapter),
            composition_classes=(MyModule,),
        )

        # Lightning compositions
        adapter_ctx_registry.register(
            Adapter.my_adapter,
            component_key="module",
            adapter_combination=(Adapter.lightning, Adapter.my_adapter),
            composition_classes=(
                MyAttributeMixin,
                BaseMyModule,
                LightningAdapter,
                BaseITModule,
                LightningModule,
            ),
        )
```

### Pattern 4: Composed Module Classes

Final module classes combine all components:

```python
class MyModule(MyAdapter, CoreHelperAttributes, BaseMyModule):
    """Composed module for (core, my_adapter) combination."""
    ...

# For more complex compositions
class MyLightningModule(
    MyAttributeMixin,
    BaseMyModule,
    LightningAdapter,
    BaseITModule,
    LightningModule,
):
    """Composed module for (lightning, my_adapter) combination."""
    ...
```

## Configuration Patterns

### Pattern 1: Adapter Config Dataclass

```python
from dataclasses import dataclass
from interpretune.config.shared import ITSerializableCfg

@dataclass(kw_only=True)
class MyConfig(ITSerializableCfg):
    """Configuration for my adapter."""

    model_name: str | None = "default-model"
    device_map: str | None = None
    some_option: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Validation logic here
        pass

    def get_init_kwargs(self) -> dict[str, Any]:
        """Generate kwargs for model initialization."""
        kwargs = {"some_option": self.some_option}
        if self.device_map:
            kwargs["device_map"] = self.device_map
        return kwargs
```

### Pattern 2: Extended ITConfig

```python
@dataclass(kw_only=True)
class ITMyConfig(ITConfig):
    """ITConfig extended with my adapter config."""

    my_cfg: MyConfig

    def __post_init__(self) -> None:
        if not self.my_cfg:
            raise MisconfigurationException("my_cfg is required")

        # Config synchronization logic
        self._sync_config()
        super().__post_init__()
```

### Pattern 3: Model Wrapper Selection (SAE Lens)

The SAE Lens adapter supports multiple TL model wrappers via the ``model_wrapper`` field on
``SAELensConfig``.  This pattern enables adapters to dispatch to different model initialization
paths from a single configuration:

| ``model_wrapper`` value | Model class | Notes |
|-------------------------|-------------|-------|
| ``"transformer_bridge"`` (default) | ``SAETransformerBridge`` | Wraps HF model without weight conversion; more memory efficient |
| ``"hooked_transformer"`` | ``HookedSAETransformer`` | Legacy path with weight conversion |

**Dispatch implementation** (in ``BaseSAELensTLModule``):

```python
def _convert_hf_to_tl(self) -> None:
    model_wrapper = getattr(self.it_cfg, "model_wrapper", "transformer_bridge")
    if model_wrapper == "transformer_bridge":
        self._convert_hf_to_bridge()
    else:
        self._convert_hf_to_hooked()
```

**Key constraints:**

- ``model_wrapper`` is only meaningful when ``backend="transformerlens"`` — the NNsight
  backend ignores it.
- TransformerBridge requires an HF model instance — it **cannot** be initialized from a
  config dict alone.  Config-based initialization (``ITLensCustomConfig``) always uses
  ``HookedSAETransformer`` regardless of ``model_wrapper``.
- When using the hooked path, set ``model_wrapper="hooked_transformer"`` explicitly in
  your test or production config since the default is now ``"transformer_bridge"``.

**Configuration example (Bridge — default):**

```python
from interpretune.config import SAELensConfig, ITLensBridgeConfig

cfg = SAELensConfig(
    tl_cfg=ITLensBridgeConfig(model_name="gpt2-small", default_padding_side="left"),
    sae_cfgs=[SAELensFromPretrainedConfig(release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre")],
)
```

**Configuration example (legacy hooked path):**

```python
cfg = SAELensConfig(
    model_wrapper="hooked_transformer",
    tl_cfg=ITLensFromPretrainedNoProcessingConfig(model_name="gpt2-small", use_bridge=False),
    sae_cfgs=[SAELensFromPretrainedConfig(release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre")],
)
```

## Light Registration

Adapters must be registered for light import in `_light_register.py`:

```python
# In _light_register.py
adapter_modules: tuple[str, ...] = (
    "interpretune.adapters.core",
    "interpretune.adapters.lightning",
    "interpretune.adapters.transformer_lens",
    "interpretune.adapters.sae_lens",
    "interpretune.adapters.nnsight",  # Add new adapter
    "interpretune.adapters.circuit_tracer",
)
```

## Lazy Exports

Add lazy exports in `adapters/__init__.py`:

```python
_LAZY_ADAPTER_ATTRS = {
    # ... existing exports ...
    "MyAdapter": "interpretune.adapters.my_adapter.MyAdapter",
    "MyModule": "interpretune.adapters.my_adapter.MyModule",
    "MyAttributeMixin": "interpretune.adapters.my_adapter.MyAttributeMixin",
    "BaseMyModule": "interpretune.adapters.my_adapter.BaseMyModule",
}
```

## Protocol Enum

Add adapter to the Protocol enum:

```python
# In protocol.py
class Adapter(AutoStrEnum):
    core = auto()
    lightning = auto()
    transformer_lens = auto()
    sae_lens = auto()
    nnsight = auto()
    circuit_tracer = auto()
    my_adapter = auto()  # Add new adapter
```

## Testing Patterns

### Parity Test Configuration

```python
from dataclasses import dataclass
from tests.base_defaults import BaseAugTest, BaseCfg

@dataclass(kw_only=True)
class MyParityCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.my_adapter)
    model_src_key: str | None = "gpt2"
    my_cfg: MyConfig | None = MyConfig()

@dataclass
class MyParityTest(BaseAugTest):
    result_gen: Callable | None = partial(collect_results, my_parity_results)
```

### Test Parametrization

```python
MY_CONFIGS = (
    MyParityTest(alias="test_cpu_32", cfg=MyParityCfg(phase="test")),
    MyParityTest(alias="test_cuda_32", cfg=MyParityCfg(phase="test", **cuda), marks="cuda"),
    MyParityTest(alias="train_cpu_32", cfg=MyParityCfg()),
)

@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(MY_CONFIGS, unpack=False))
def test_parity_my_adapter(recwarn, tmp_path, request, test_alias, test_cfg):
    parity_test(test_cfg, test_alias, EXPECTED[test_alias] or {}, tmp_path)
```

## Best Practices

1. **Use TYPE_CHECKING for imports**: Heavy library imports should be inside `if TYPE_CHECKING:` blocks
2. **Provide fallbacks**: Allow graceful degradation when optional features aren't available
3. **Capture hyperparameters**: Override `_capture_hyperparameters()` to log adapter config
4. **Handle device placement**: Implement device-related properties consistently
5. **Document compositions**: Clearly document what adapter combinations are supported
6. **Test all combinations**: Add tests for each registered adapter combination

## Common Pitfalls

1. **Circular imports**: Use lazy imports and TYPE_CHECKING to avoid
2. **Missing registrations**: Ensure all adapter combinations are registered
3. **Composition order**: MRO (Method Resolution Order) matters - place mixins first
4. **Config synchronization**: Handle cases where multiple configs need to stay in sync
5. **Type annotations**: Mixins need explicit type hints since they expect composed classes

## Example: Full Adapter Implementation

See the following files for complete examples:
- `src/interpretune/adapters/nnsight.py` - NNsight adapter
- `src/interpretune/adapters/transformer_lens.py` - TransformerLens adapter
- `src/interpretune/config/nnsight.py` - NNsight configuration
- `tests/parity_acceptance/test_it_ns.py` - NNsight parity tests
