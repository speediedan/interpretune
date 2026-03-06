# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

#### NNsight Adapter
- **New NNsight adapter** (`interpretune.adapters.nnsight`) enabling model interpretability via the NNsight deferred-execution tracing framework
- `NNsightConfig` and `NNsightAttributeMixin` for NNsight-specific configuration and model setup
- `BaseNNsightModule` with automatic model initialization via `auto_model_init` factory
- Support for both local execution and remote NDIF tracing via `nnsight_remote` and `ndif_api_key` config fields
- 6 new adapter compositions: `(core, nnsight)`, `(lightning, nnsight)`, `(core, sae_lens, nnsight)`, `(lightning, sae_lens, nnsight)`, `(core, circuit_tracer)` with NNsight backend, `(lightning, circuit_tracer)` with NNsight backend
- Comprehensive test suite with parity acceptance tests (`test_it_ns.py`) and profiling tests (`test_ns_profiling`)

#### Analysis Backend Abstraction
- **New backend-agnostic analysis system** (`interpretune.analysis.backends`) separating analysis operations from framework-specific implementation
- `ModelBackend` protocol with `TransformerLensBackend` and `NNsightBackend` implementations
- `HookNameResolver` for bidirectional TL↔NNsight hook name mapping with architecture-aware translation
- `NNsightForwardContext` for multi-invoke batched tracing with memory-efficient chunking (`max_invokes_per_trace`)
- `ActivationCacheAdapter` providing unified cache interface across backends
- Backend auto-detection from module adapter context

#### SAE-Lens Dual-Backend Support
- SAE-Lens adapter now supports both TransformerLens and NNsight backends via `backend` field in `SAELensConfig`
- `SAELensNNsightModuleMixin` for NNsight-based SAE hook dispatch using `model.sae()` with `hook=True`
- `SAEHookResolver` for mapping SAE hook points across backends
- `model_wrapper` field supporting `"transformer_bridge"` (default) and `"hooked_transformer"` model wrappers
- Backward-compatible: existing TL-based configs continue to work without changes

#### Circuit-Tracer Adapter Enhancements
- Multi-backend support: circuit-tracer now works with both TransformerLens (`HookedTransformer`) and NNsight (`LanguageModel`) backends
- `CircuitTracerNNsightModule` with `NNSightReplacementModel` integration for NNsight-backed circuit tracing
- `ReplacementModel.from_pretrained` factory integration for backend-agnostic model loading
- Gemma3 support via NNsight backend with full HF hub transcoder paths (`mwhanna/gemma-scope-2-1b-pt/...`)
- Remote NDIF execution support for circuit-tracer analysis
- `_load_replacement_model` method with backend-specific initialization and pretrained_kwargs forwarding

#### Analysis Operations
- Backend parameter added to `run_analysis_op`, `fwd_w_hooks_batched`, `logit_diffs`, and `cache_activations` ops
- `BackendInfo` for passing backend context through the analysis pipeline
- Analysis ops dispatch to appropriate backend based on module adapter context
- NNsight-compatible caching via `NNsightActivationCache`

#### Configuration
- `NNsightConfig` dataclass with `model_name`, `device_map`, `torch_dtype`, `dispatch`, `tokenizer_kwargs`
- `ITNNsightConfig` extending `ITConfig` for NNsight-specific module configuration
- `SAELensConfig` gains `backend`, `model_wrapper`, `nnsight_cfg` fields
- `CircuitTracerConfig` gains `nnsight_remote`, `ndif_api_key`, `model_name` fields
- `RunAnalysisConfig` updated with backend-aware dispatching

#### Testing Infrastructure
- `RunIf(bf16_cuda=True)` marker for tests requiring CUDA with bf16 support (Gemma2 model tests)
- `RunIf(optional=True)` marker for optional tests (NDIF remote, Gemma3)
- `cleanup_cuda` fixture for GPU memory management between tests
- New fixture configurations for NNsight sessions, NNsight analysis, and circuit-tracer backends
- NNsight parity acceptance tests with expected results and profiling baselines
- Example module registry entries for NNsight adapter combinations

### Changed
- `CompositionRegistry` updated to handle NNsight adapter combinations and dual-backend circuit-tracer
- `BaseITModule.auto_model_init` refactored to support both HuggingFace and NNsight model initialization paths
- `AnalysisRunner` now auto-detects and uses appropriate backend based on module's adapter context
- Analysis op `definitions.py` refactored to use backend abstraction instead of direct TL imports
- `protocol.py` updated with `Adapter.nnsight` enum value and NNsight-related protocol extensions
- Memory profiler updated to handle NNsight model memory accounting
- Debug generation extension updated for NNsight backend compatibility
- Circuit-tracer adapter significantly expanded from TL-only to multi-backend architecture (381+ lines added)
- SAE-Lens adapter expanded from TL-only to dual-backend with hook resolution (455+ lines added)

### Fixed
- Circuit-tracer `isinstance` assertions replaced with class-name checks to avoid false failures from pytest `importlib` import mode causing module double-loading
- Coverage collection fix for NNsight's `sys.settrace()` disabling behavior
- HuggingFace transformers v5 compatibility with configurable test reruns (`--reruns 2 --reruns-delay 5`)
- Dataset cache reset handling to prevent cache-dependent expected value divergence
- Tokenizer usage pattern fix for `force_prepare_data` enabled scenarios

## [0.1.0] - 2025-XX-XX

### Added

### Fixed

### Changed

### Deprecated
