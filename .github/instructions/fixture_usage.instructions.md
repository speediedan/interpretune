---
applyTo: "**/tests/**"
---

# Fixture Usage Instructions for Interpretune Tests

## Overview

Interpretune uses a sophisticated fixture generation system that creates test fixtures dynamically from configuration objects. This approach maximizes test fidelity and code reuse while maintaining flexibility for different adapter combinations and test scenarios.

## Core Principles

1. **Prefer Real Objects Over Mocks**: Use small real objects (via fixtures) for maximum test fidelity, except for trivial edge cases
2. **Maximize Fixture Reuse**: Leverage existing fixtures from `conftest.py` rather than creating new ones unless necessary
3. **Scope Appropriately**: Use fixture scopes (`session`, `module`, `class`, `function`) to balance performance and isolation
4. **Test Method Logic, Not Framework**: Focus unit tests on method behavior; use integration tests for full module lifecycle

## Critical: MODULE_EXAMPLE_REGISTRY Dependency

**IMPORTANT**: Most session fixtures (`it_session`, `it_session_cfg`, `analysis_session`) depend on the `MODULE_EXAMPLE_REGISTRY` defined in `src/it_examples/example_module_registry.yaml`.

### How Fixture Resolution Works
 - session fixture factory example
```
conftest.py fixture factories
         ↓
it_session_fixture_factory() / it_session_cfg_fixture_factory()
         ↓
config_modules() [tests/configuration.py]
         ↓
gen_session_cfg()
         ↓
MODULE_EXAMPLE_REGISTRY.get(test_cfg)  ← REGISTRY LOOKUP HERE
         ↓
Returns: (base_itdm_cfg, base_it_cfg, dm_cls, m_cls)
```

### Registry Lookup Keys

The registry lookup uses the test configuration's attributes to form a lookup key:
- `(model_src_key, model_cfg_key, phase, adapter_ctx)` tuple

**Example**: A config with `model_src_key="gpt2"`, `model_cfg_key="rte"`, `phase="test"`, `adapter_ctx=(Adapter.core, Adapter.nnsight)` looks up:
```
('gpt2', 'rte', 'test', (<Adapter.core: 'core'>, <Adapter.nnsight: 'nnsight'>))
```

### Implications for New Adapters

**If your adapter combination is not registered in `example_module_registry.yaml`:**
1. Session fixtures (`it_session`, `it_session_cfg`) will fail with `KeyError`
2. You must register the adapter combination in the registry YAML

**To add a new adapter combination**, see `.github/instructions/registering_example_modules.instructions.md`.

## Fixture Architecture

### Fixture Configuration System

Fixtures are defined in `FIXTURE_CFGS` dictionary in `conftest.py` using the `FixtureCfg` dataclass:

```python
FIXTURE_CFGS = {
    "config_key": FixtureCfg(
        test_cfg=ConfigClass,           # Configuration class to instantiate
        scope="module",                  # pytest scope (session/module/class/function)
        variants={                       # Variants to generate
            "it_session": [FixtPhase.setup],
            "it_session_cfg": [FixtPhase.cfgonly],
        }
    ),
}
```

### Fixture Types

1. **ITSession Fixtures** - Full session with module and datamodule:
   - Pattern: `get_it_session__{config_key}__{phase}`
   - Returns: `ITSessionFixture(it_session, test_cfg)`
   - Example: `get_it_session__l_tl_bridge_gpt2__setup`

2. **ITModule Fixtures** - Module only (with mock datamodule):
   - Pattern: `get_it_module__{config_key}__{phase}`
   - Returns: ITModule instance
   - Example: `get_it_module__core_gpt2__setup`

3. **ITSessionCfg Fixtures** - Configuration only (no instantiation):
   - Pattern: `get_it_session_cfg__{config_key}`
   - Returns: ITSessionCfg object
   - Example: `get_it_session_cfg__sl_ht_gpt2`

4. **AnalysisSession Fixtures** - Session with analysis runner and results:
   - Pattern: `get_analysis_session__{config_key}__{phase}_{runphase}`
   - Returns: `AnalysisSessionFixture(result, it_session, runner, run_config, test_cfg)`
   - Example: `get_analysis_session__sl_ht_gpt2_logit_diffs_sae__initonly_runanalysis`

5. **Fine-Tuning Schedule Fixtures** - Fine-tuning schedules:
   - Pattern: `get_ft_schedule__{config_key}__setup`
   - Returns: dict of schedules
   - Example: `get_ft_schedule__l_tl_bridge_gpt2_sched__setup`

### Fixture Phases

Fixtures support different initialization phases via `FixtPhase` enum:

- `cfgonly` - Configuration object only
- `initonly` - Module/session initialized but no hooks called
- `prepare_data` - `prepare_data()` hook called
- `setup` - `setup()` hook called (most common for tests)
- `configure_optimizers` - Optimizers configured

Analysis fixtures also support `RunPhase`:
- `cfgonly` - Configuration only
- `initrunner` - Runner initialized
- `runanalysis` - Analysis executed

## Adding New Fixtures

### Step 1: Define Configuration Class

Create a test configuration class in `tests/core/cfg_aliases.py` or similar.

**Pattern 1: Registry-Based Configuration (for configs registered in `example_module_registry.yaml`):**

Registry entries are looked up using `ModuleRegistry.get()` which accepts:
- **Tuple**: `(model_src_key, model_cfg_key, phase, adapter_ctx)` - e.g., `("gemma2", "rte_base_test", "test", (Adapter.core, Adapter.circuit_tracer))`
- **String**: Registry key like `"gemma2.rte_demo.circuit_tracer"`
- **RegKeyQueryable**: Object with `model_src_key`, `model_cfg_key`, `phase`, `adapter_ctx` attributes (like BaseCfg)

```python
from tests.base_defaults import BaseCfg

@dataclass(kw_only=True)
class CircuitTracerTLGemma2(BaseCfg):
    """Circuit Tracer with TransformerLens backend on Gemma2."""

    phase: str = "test"
    model_src_key: str | None = "gemma2"
    model_cfg_key: str = "rte_base_test"
    device_type: str = "cuda"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.circuit_tracer)
    tl_cfg: ITLensCfg | None = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gemma-2-2b", default_padding_side="left", use_bridge=False
        )
    )
    generative_step_cfg: GenerativeClassificationConfig | None = field(
        default_factory=lambda: GenerativeClassificationConfig(
            enabled=True,
            lm_generation_cfg=TLensGenerationConfig(max_new_tokens=1, output_logits=True, return_dict_in_generate=True),
        )
    )
    circuit_tracer_cfg: CircuitTracerConfig | None = field(
        default_factory=lambda: CircuitTracerConfig(
            backend="transformerlens",  # though TL is default we explicitly set backend here for clarity
            transcoder_set="gemma",
            analysis_target_tokens=['▁Dallas','▁Austin'],
            max_feature_nodes=8192,
            offload='cpu',
            verbose=True,
            )
    )


```

**Pattern 2: Direct Model Configuration (for smaller models or custom configs):**

This pattern uses `model_src_key` directly to build configuration without registry lookup.

```python
from tests.base_defaults import BaseCfg

@dataclass(kw_only=True)
class LightningTLGPT2(BaseCfg):
    """TransformerLens with GPT2 using direct model_src_key."""

    model_src_key: str | None = "gpt2"
    adapter_ctx: Sequence[Adapter | str] = (Adapter.lightning, Adapter.transformer_lens)
    tl_cfg: ITLensFromPretrainedNoProcessingConfig = field(
        default_factory=lambda: ITLensFromPretrainedNoProcessingConfig(
            model_name="gpt2-small", default_padding_side="left", use_bridge=False
        )
    )
```

### Step 2: Register in FIXTURE_CFGS

Add entry to `FIXTURE_CFGS` in `conftest.py`:

```python
FIXTURE_CFGS = {
    # ... existing entries ...
    "ct_tl_gemma2": FixtureCfg(
        test_cfg=CircuitTracerTLGemma2,
        scope="module",  # Use scope appropriate for usage (smaller fixtures usually can have wider scope)
        variants={
            "it_session": [FixtPhase.setup],  # Generate setup fixture
        }
    ),
}
```

### Step 3: Use in Tests

The fixture is auto-generated and available as `get_it_session__ct_tl_gemma2__setup`:

```python
class TestCircuitTracerTLBackend:
    def test_replacement_model_type(self, get_it_session__ct_tl_gemma2__setup):
        it_session = get_it_session__ct_tl_gemma2__setup.it_session
        assert isinstance(it_session.module.replacement_model, TransformerLensReplacementModel)
```
It can also be used in parameterized tests (along with other generated fixtures)
```python
    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__ct_tl_gemma2__setup"),
        ],
        ids=["transformerlens"],
    )
    def test_backend_property_access(self, session_fixture, request):
        """Verify backend property returns correct value from config."""
        it_session = request.getfixturevalue(session_fixture).it_session

        # Access backend through circuit_tracer_cfg
        backend = it_session.module.circuit_tracer_cfg.backend
        assert backend == "transformerlens"
```

## Fixture Parameterization Patterns

### Pattern 1: Direct Fixture Usage

Use when testing a specific configuration:

```python
class TestSpecificConfig:
    def test_feature(self, get_it_session__config_key__setup):
        session = get_it_session__config_key__setup.it_session
        # Test with specific configuration
        assert session.module.some_property == expected_value
```

### Pattern 2: Parameterized Fixtures

Use when testing across multiple configurations:

```python
# Define parameterization in conftest.py or test file
ct_backend_fixtures = {
    "argvalues": [
        pytest.param("ct_tl_gemma2"),
        pytest.param("ct_nnsight_gemma2"),
    ],
    "ids": ["transformerlens", "nnsight"],
}

class TestCircuitTracerBackends:
    @pytest.mark.parametrize("fixture_name", **ct_backend_fixtures)
    def test_backend_loading(self, fixture_name, request):
        # Use request fixture to get the actual fixture by name
        it_session = request.getfixturevalue(f"get_it_session__{fixture_name}__setup").it_session
        # Test works with any backend
        assert it_session.module.replacement_model is not None
```

### Pattern 3: Class-Scoped Fixture with Method Reuse

Use for expensive setups shared across multiple tests:

```python
@pytest.fixture(scope="class")
def sl_ht_gpt2_w_ref_logits(get_it_session__sl_ht_gpt2__initonly):
    fixture = get_it_session__sl_ht_gpt2__initonly
    sl_test_module = fixture.it_session.module
    return sl_test_module, TestClassSAELens.get_ref_logits(sl_test_module)


@pytest.fixture(scope="class")
def l_sl_ht_gpt2_w_ref_logits(get_it_session__l_sl_ht_gpt2__initonly):
    fixture = get_it_session__l_sl_ht_gpt2__initonly
    sl_test_module = fixture.it_session.module
    return sl_test_module, TestClassSAELens.get_ref_logits(sl_test_module)


core_l_run_w_pytest_cfg = {
    "argvalues": [
        pytest.param("sl_ht_gpt2_w_ref_logits"),
        pytest.param("l_sl_ht_gpt2_w_ref_logits", marks=RunIf(lightning=True)),
    ],
    "ids": ["core", "lightning"],
}

class TestClassSAELens:
    # ... other test methods and class level variables etc
    @pytest.mark.parametrize("session_fixture", **core_l_run_w_pytest_cfg)
    def test_run_with_saes(self, request, session_fixture):
        sl_test_module, original_logits = request.getfixturevalue(session_fixture)
        assert len(sl_test_module.model.acts_to_saes) == 0
        logits_with_saes = sl_test_module.model.run_with_saes(TestClassSAELens.prompt, saes=sl_test_module.sae_handles)
        assert len(sl_test_module.model.acts_to_saes) == 0
        assert not torch.allclose(logits_with_saes, original_logits)
```

## Module Registry Integration

### Adding example module config registry entries

Below we add an rte task example that composes transformer_lens and circuit_tracer adapters update `example_module_registry.yaml`:

```yaml
gemma2.rte.circuit_tracer_tl:
  reg_info:
    model_src_key: gemma2
    model_cfg_key: rte_base_test
    adapter_combinations:
      - [core, transformer_lens, circuit_tracer]
      - [lightning, transformer_lens, circuit_tracer]
    description: Circuit Tracer with TransformerLens backend, Gemma2-2b (non-instruction tuned)
  shared_config:
    task_name: rte
    os_env_model_auth_key: HF_GATED_PUBLIC_REPO_AUTH_KEY
    model_name_or_path: google/gemma-2-2b
    tokenizer_kwargs:
        model_input_names: ['input']
        padding_side: left
        add_bos_token: true
  registered_cfg:
    datamodule_cfg:
      prompt_cfg:
        class_path: it_examples.experiments.rte_boolq.RTEBoolqPromptConfig
      signature_columns: ['input', 'labels']
      text_fields: ["premise", "hypothesis"]
      enable_datasets_cache: False
      train_batch_size: 2
      eval_batch_size: 2
    module_cfg:
      class_path: interpretune.config.module.ITConfig
      init_args:
        auto_comp_cfg:
          class_path: interpretune.config.shared.AutoCompConfig
          init_args:
            module_cfg_name: RTEBoolqConfig
            module_cfg_mixin:
              class_path: it_examples.experiments.rte_boolq.RTEBoolqEntailmentMapping
              import_only: True
        generative_step_cfg:
          class_path: it_examples.experiments.rte_boolq.RTEBoolqGenerativeClassificationConfig
          init_args:
            enabled: True
            lm_generation_cfg:
              class_path: interpretune.config.transformer_lens.TLensGenerationConfig
              init_args:
                max_new_tokens: 1
                output_logits: true
                return_dict_in_generate: true
        hf_from_pretrained_cfg:
          class_path: interpretune.config.mixins.HFFromPretrainedConfig
          init_args:
            pretrained_kwargs:
              device_map: cpu
              dtype: float32
            model_head: transformers.Gemma2ForCausalLM
        tl_cfg:
          class_path: interpretune.config.transformer_lens.ITLensFromPretrainedNoProcessingConfig
          init_args:
            model_name: gemma-2-2b
            default_padding_side: left
            use_bridge: false  # circuit_tracer requires HookedTransformer, not TransformerBridge
        circuit_tracer_cfg:
          class_path: interpretune.config.circuit_tracer.CircuitTracerConfig
          init_args:
            backend: "transformerlens"
            transcoder_set: "gemma"
            max_n_logits: 10
            analysis_target_tokens: ['▁Dallas','▁Austin']
            desired_logit_prob: 0.95
            max_feature_nodes: 8192
            batch_size: 256
            offload: 'cpu'
            verbose: true
            use_neuronpedia: false
    datamodule_cls:
      class_path: tests.modules.FingerprintTestITDataModule
    module_cls:
      class_path: tests.modules.TestITModule
```

## Testing Best Practices

### Unit Tests: Test Method Logic

Unit tests should focus on method behavior, not complex initialization:

```python
class TestCircuitTracerTLBackendInitialization:
    """Test TransformerLens backend-specific initialization in CircuitTracerAdapter.

    These tests use the (core, transformer_lens, circuit_tracer) adapter combination.
    """

    @pytest.mark.parametrize(
        "session_fixture",
        [
            pytest.param("get_it_session__ct_tl_gemma2__setup"),
        ],
        ids=["transformerlens"],
    )
    def test_backend_property_access(self, session_fixture, request):
        """Verify backend property returns correct value from config."""
        it_session = request.getfixturevalue(session_fixture).it_session

        # Access backend through circuit_tracer_cfg
        backend = it_session.module.circuit_tracer_cfg.backend
        assert backend == "transformerlens"
```

### Integration Tests: Test Full Workflows

# TODO: update this description

```python
@dataclass(kw_only=True)
class TLParityCfg(BaseCfg):
    adapter_ctx: Sequence[Adapter | str] = (Adapter.core, Adapter.transformer_lens)
    model_src_key: str | None = "cust"


@dataclass
class TLParityTest(BaseAugTest):
    result_gen: Callable | None = partial(collect_results, tl_parity_results)


PARITY_TL_CONFIGS = (
    TLParityTest(alias="test_cpu_32", cfg=TLParityCfg(phase="test", model_src_key="gpt2")),
    TLParityTest(
        alias="test_cpu_32_l",
        cfg=TLParityCfg(
            phase="test",
            **w_l_tl,
        ),
        marks="lightning",
    ),
    TLParityTest(alias="test_cuda_32", cfg=TLParityCfg(phase="test", **req_det_cuda), marks="cuda"),
    TLParityTest(alias="test_cuda_32_l", cfg=TLParityCfg(phase="test", **req_det_cuda, **w_l_tl), marks="cuda_l"),
    TLParityTest(alias="train_cpu_32", cfg=TLParityCfg()),
    TLParityTest(alias="train_cpu_32_l", cfg=TLParityCfg(**w_l_tl), marks="lightning"),
    TLParityTest(alias="train_cuda_32", cfg=TLParityCfg(**req_det_cuda), marks="cuda"),
    TLParityTest(alias="train_cuda_32_l", cfg=TLParityCfg(**req_det_cuda, **w_l_tl), marks="cuda_l"),
)

EXPECTED_PARITY_TL = {cfg.alias: cfg.expected for cfg in PARITY_TL_CONFIGS}


@pytest.mark.usefixtures("make_deterministic")
@pytest.mark.parametrize(("test_alias", "test_cfg"), pytest_factory(PARITY_TL_CONFIGS, unpack=False))
def test_parity_tl(recwarn, tmp_path, request, test_alias, test_cfg):
    if test_cfg.req_deterministic:
        request.getfixturevalue("make_deterministic")
    state_log_mode = IT_GLOBAL_STATE_LOG_MODE  # one can manually set this to True for a local test override
    expected_results = EXPECTED_PARITY_TL[test_alias] or {}
    expected_warnings = TL_LIGHTNING_CTX_WARNS if Adapter.lightning in test_cfg.adapter_ctx else TL_CTX_WARNS
    parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warnings)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
```

### Avoid Complex Module Mocking

When testing adapters with complex initialization:

**DON'T:**
```python
@pytest.fixture
def complex_module():
    # Attempting to mock all the initialization is fragile
    module = MagicMock(spec=BaseCircuitTracerModule)
    module.circuit_tracer_cfg = CircuitTracerConfig()
    # This breaks easily when initialization logic changes
    return module
```

**DO:**
```python
@pytest.fixture
def circuit_tracer_session(get_it_session__ct_tl_gemma2__setup):
    # Use real fixture, deepcopy if needed for isolation
    return deepcopy(get_it_session__ct_tl_gemma2__setup)
```

## Common Patterns from Existing Tests

### Pattern: Architecture-Specific Expectations

From `test_adapters_transformer_lens.py`:

```python
@dataclass
class ArchitectureExpectations:
    """Expected parameter structure for a specific model architecture."""
    model_name: str
    n_layers: int
    # ... other expectations ...

LLAMA3_EXPECTATIONS = ArchitectureExpectations(
    model_name="Llama-3.2-3B-Instruct",
    n_layers=28,
    # ... architecture-specific values ...
)

@pytest.mark.standalone
class TestLlama3ParameterMapping:
    def test_llama3_tl_param_structure(self, get_it_session__l_tl_bridge_llama3__setup):
        session = get_it_session__l_tl_bridge_llama3__setup.it_session
        validate_tl_parameter_mapping(session.module.model, LLAMA3_EXPECTATIONS)
```

### Pattern: Deepcopy Session-Scoped Fixtures

From `test_adapters_sae_lens.py`:

```python
@pytest.fixture(scope="class")
def sl_ht_gpt2_w_ref_logits(get_it_session__sl_ht_gpt2__initonly):
    # Deepcopy to avoid interference between tests
    it_s = deepcopy(get_it_session__sl_ht_gpt2__initonly)
    # Modify copy for specific test needs
    it_s.it_session.module.compute_reference_logits()
    return it_s

class TestSAELens:
    def test_with_modified_session(self, sl_ht_gpt2_w_ref_logits):
        # Tests use modified copy, original fixture unaffected
        assert sl_ht_gpt2_w_ref_logits.it_session.module.reference_logits is not None
```

### Pattern: Conditional Parameterization

From `test_adapters_sae_lens.py`:

```python
core_l_run_w_pytest_cfg = {
    "argvalues": [
        pytest.param("sl_ht_gpt2_w_ref_logits"),
        pytest.param("l_sl_ht_gpt2_w_ref_logits", marks=RunIf(lightning=True)),
    ],
    "ids": ["core", "lightning"],
}

class TestSAELensAdapters:
    @pytest.mark.parametrize("fixture_name", **core_l_run_w_pytest_cfg)
    def test_across_adapters(self, fixture_name, request):
        session = request.getfixturevalue(fixture_name).it_session
        # Test works with both core and lightning adapters
```

## Troubleshooting

### Issue: Fixture Not Found

**Error:** `fixture 'get_it_session__my_config__setup' not found`

**Solution:** Ensure:
1. Config key is in `FIXTURE_CFGS`
2. Variant includes the phase you're requesting
3. Fixture generation loop has run (fixtures auto-generate on module load)

### Issue: Module Initialization Errors

**Error:** `AttributeError: object has no attribute 'X'` during fixture setup

**Solution:**
- Use simpler fixtures (e.g., `cfgonly` instead of `setup`)
- Test method logic separately from initialization
- Move to integration tests with `@pytest.mark.standalone`

### Issue: Fixture Scope Conflicts

**Error:** Tests interfere with each other using shared fixture

**Solution:**
- Use `deepcopy()` on session/module-scoped fixtures
- Change fixture scope to `function` for isolation
- Create test-specific derived fixtures

## Summary

1. **Leverage Existing Fixtures**: Check `conftest.py` before creating new fixtures
2. **Use Module Registry**: Define backend variants in `example_module_registry.yaml`
3. **Follow Patterns**: Reference existing adapter tests for proven approaches
4. **Test Appropriately**: Unit tests for logic, integration tests for workflows
5. **Scope Wisely**: Balance performance (session/module) vs isolation (function)
6. **Avoid Complex Mocks**: Use real objects via fixtures for better test fidelity
