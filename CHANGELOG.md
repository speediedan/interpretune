# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Phase-specific collection manifests and a ROCm failure-triage classifier for the CUDA-marked and standalone test phases (`tests/gpu_ci/phase_manifest.py`), with a committed measured manifest and build-day commands; ROCm execution remains pending AMD hardware ([#794](https://github.com/speediedan/interpretune/issues/794))

- In-tree latent projection/ablation device-placement acceptance tests: a single-device CPU reference compared against single- and split-device CUDA placements, asserting placement and parameter/optimizer movement (`tests/gpu_ci/test_latent_device_placement.py`) ([#795](https://github.com/speediedan/interpretune/issues/795))

- The latent device-placement tests split a real model: gpt2 under an explicit dict `device_map` across two devices with `gpt2-small-res-jb` SAEs spliced in, compared against the same run on one GPU, with a split positive control, exact single-device comparisons, device assertions on the outputs of cross-device operations, and an Adam step whose state is checked against its parameters' devices; phase collection manifests now record any collection narrowing (`extra_args`) and uncommitted or untracked files under the collection roots (`git_dirty`, `git_dirty_paths`) ([#795](https://github.com/speediedan/interpretune/issues/795), [#794](https://github.com/speediedan/interpretune/issues/794))

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

#### Circuit-Tracer Analysis Ops
- Five new circuit-tracer analysis ops registered in the dispatcher: `concept_direction`, `compute_attribution_graph`, `graph_node_influence`, `extract_top_features`, `feature_intervention_forward`
- Full pipeline from semantic concept direction → attribution graph → node influence → top feature extraction → causal intervention verification
- Graph decomposition and serialization into `AnalysisStore` with per-latent-model indexing
- Cross-backend composition: CT analysis results compose with SAE-lens `AnalysisStore` for combined analysis

#### Demo Notebooks
- **CT Analysis Backend Demo** (`ct_analysis_backend_demo.ipynb`): End-to-end pipeline running all 5 CT analysis ops via the dispatcher on Gemma-2-2b with NNsight backend
- **CT Cross-Backend Demo** (`ct_cross_backend_demo.ipynb`): Multi-model composition showing GPT-2 SAE analysis (TransformerBridge) and Gemma-2 CT analysis (NNsight) with cross-backend `AnalysisStore` enrichment
- Updated basic circuit-tracer notebook with next-steps referencing new demos
- Updated SAE-Lens notebook with cross-backend composition footer

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

#### Core Training Loop
- LR schedulers step on the core path at their configured `interval` (`step` or `epoch`) and `frequency`; `ReduceLROnPlateau` steps with its `monitor` metric and is refused, naming the metric, when that metric was not logged (#761)
- `SessionRunnerCfg.validate_with_grad` validates with gradients enabled (for objectives using `torch.autograd.grad`) instead of under `torch.inference_mode()` (#761)
- Per-epoch training and validation metrics: `log()`/`log_dict()` values are averaged, printed and reset at each training epoch end, and passed to callbacks (#761)
- A minimal callback list (`SessionRunnerCfg.callbacks`, `CoreCallback`, `CoreLoopContext`) with step, validation-end and epoch boundaries; a callback stops training by setting `ctx.should_stop` (#761)
- `checkpoint_state` / `load_checkpoint_state`: module, optimizer, lr scheduler, RNG and loop state, plus optional `on_save_checkpoint` / `on_load_checkpoint` module hooks and callback state (#761)

#### Tuning through finetuning-scheduler views
- Each module composition declares a finetuning-scheduler `TunableView` (`build_tunable_view()`, with cooperative `tunable_roles()`), attached as `fts_tunable_view` after model init: the identity view with PEFT `adapter:<name>` roles by default, `TransformerBridgeView` in TransformerLens names for TransformerLens compositions, `latent:sae` for spliced SAEs, and refusals with reasons for remote or undispatched nnsight, circuit-tracer, TransformerLens bridges holding quantized or packed parameters, and compositions whose `build_tunable_view()` returns `None` (#758, #759)
- `interpretune.testing.conformance.TuningConformance` runs finetuning-scheduler's view conformance cases against a composition, selected by its declared support (#760)
- The default optimizer (`optimizer_parameters()`) covers an attached, trainable view's root parameters, less `fitted` and `instrument` tensors: every view is rooted at the interpretune module, so trainable modules registered beside `module.model` (interventions, spliced SAEs) are optimized with no adapter extending the hook. Compositions without a view, or with a refusing one, keep the model's own parameters (#761)
- Local, dispatched nnsight compositions register their HF module on the module as `hf_model` through the cooperative `register_tunable_modules()` hook (run after every init path), so the optimizer, checkpoints and the identity view reach its parameters; remote (NDIF) and undispatched compositions are refused with their reasons (#774)

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
- SAEs spliced at init (`add_saes_on_init=True`) on the TransformerBridge path are now trained: they are registered on the module as `attached_saes`, built into the default optimizer through the new cooperative `optimizer_parameters()` hook, and saved in its `state_dict`. Before, the bridge kept them outside every module tree, so their parameters received gradients but were never stepped, and nothing reported it. The test `req_grad_mask` now resolves SAE parameter names and fails on a name it cannot resolve instead of silently ignoring it (#772)
- The core training loop honors `SessionRunnerCfg.max_steps` (counting optimizer steps), which it previously accepted and ignored (#761)
- The core training loop's `on_train_epoch_start` / `on_train_epoch_end` hooks are optional, as its batch hooks are; a module without them raised `AttributeError` (#761)
- The core training loop reads the optimizer from the module at every step rather than once, so a replaced optimizer is used (#761)
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
