# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- Phase-specific collection manifests and a ROCm failure-triage classifier for the CUDA-marked and standalone test phases (`tests/gpu_ci/phase_manifest.py`), with a committed measured manifest and build-day commands; ROCm execution remains pending AMD hardware ([#794](https://github.com/speediedan/interpretune/issues/794))

- ROCm triage buckets measured on the first dual-vendor parity runs: `device-fault` (device-side asserts, launch failures, HSA exceptions; not ROCm evidence by itself), `resource-calibration` and `numerics-tolerance`, plus `--classify-junit`, which reads a phase's JUnit file in execution order and attributes a sticky device-context cascade to its first fault (`device-fault-cascade`). The circuit-tracer conformance commuting control now reports its measured difference ([#794](https://github.com/speediedan/interpretune/issues/794))

- ROCm calibrations from the first AMD parity reruns, each applied only on ROCm: a `reserved_bytes.all.peak` override for `test_l_profiling.test_cuda_32` (`expected_mem_rocm`, the HIP caching allocator reserving about 20% more while allocated bytes agree), and conformance tolerances for the circuit-tracer commuting control (`ROCM_COMMUTE_ATOL`) and for batched against sequential hooks (a bounded fraction of logits beyond the padded tolerance, `ROCM_BATCHED_MISMATCH_FRACTION`, capped by `ROCM_BATCHED_ATOL`), each about twice the measured spread. The memory-stat check reports every stat that differs in one failure ([#794](https://github.com/speediedan/interpretune/issues/794))

- The cached test subsets (`pytest_rte_hf`, `pytest_rte_tl`) are keyed by tokenizer as well as task. gpt2, gemma2, gemma3 and llama3 compositions shared one subset, which was reused rather than rebuilt, so on a fresh datasets cache a gpt2 session could read another tokenizer's ids (measured: 236828 against a 50257-token vocabulary) and fault the device in its first embedding lookup, taking every later GPU test in the process with it on both vendors ([#794](https://github.com/speediedan/interpretune/issues/794))

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

### Changed

- The GPU pipelines run on either GPU agent through an any-vendor template (`.azure-pipelines/templates/gpu-job.yml`), modelled on finetuning-scheduler's: a host job that picks its digest-pinned image and device options from the agent's `GPU_VENDOR` and runs every step inside the job's container (`.azure-pipelines/scripts/ctr.sh`). `gpu-tests.yml` admits either vendor (a manual run can pin one with `vendor`); `readme-flow-gpu.yml` demands NVIDIA as a release job until `README.md` documents a ROCm install. An AMD agent takes one whole-server lease stage, since its cards are identical ([#703](https://github.com/speediedan/interpretune/issues/703))

- The ROCm CI image (`dockers/base-rocm`) installs PyTorch's ROCm wheel on plain Ubuntu, with no ROCm base image: the wheel's `rocm-sdk` dependencies carry the HIP runtime, plus an `amdsmi` path file for device discovery and a `torch-excludes.txt` that keeps the lock's CUDA torch from replacing it. `dockers/docker_images_main.sh` builds and pushes it beside the CUDA image ([#703](https://github.com/speediedan/interpretune/issues/703))
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
- The core training loop dispatches `on_train_batch_end` as optional, like every other hook in `run_step`: a module without it (the hub's RTE demo module, for one) failed at the first optimizer step ([#817](https://github.com/speediedan/interpretune/issues/817))
- `ITMeta` builds a class created with no build context over a composition it already built as an ordinary subclass, so FSDP2's `fully_shard` (which wraps a module by subclassing its class) can shard an interpretune module, the root unit included; a partial or invalid build context is still refused by name ([#820](https://github.com/speediedan/interpretune/issues/820))
- A ROCm tolerance for the circuit-tracer edge-prediction conformance case's active-feature check (`ROCM_EDGE_ATOL`, 2.6e-3 against 1e-3 elsewhere): on the AMD agent one of 3,030 active features measured 1.29e-3 off its edge prediction in two runs, the same element both times ([#703](https://github.com/speediedan/interpretune/issues/703))
- The GPU job warms the circuit-tracer transcoders (`mntss/clt-gemma-2-2b-426k`, `mwhanna/gemma-scope-transcoders`) from a GPU-only manifest (`tests/gpu_ci/hf_warm_manifest_gpu.yaml`) before its test phases, with retries, when the plan runs the notebook tests that load them (`plan_gpu_run.py --covers`). They were downloaded lazily inside the circuit-tracer notebook kernels, and on the AMD agent a partial cross-layer transcoder download (about 71 GB of `*.incomplete` blobs accumulated) crashed `test_attribution_analysis_notebook[analysis_inj_salient_logits_CLT]` on every run ([#703](https://github.com/speediedan/interpretune/issues/703))
- Memory profiling on a host without CUDA no longer fails at session end: `MemProfiler.dump_memory_stats()` creates the snapshot directory that setup creates only for CUDA allocator history, which is disabled when CUDA is unavailable ([#821](https://github.com/speediedan/interpretune/issues/821))
- The NNsight + Lightning memory-profiling cases (`test_ns_profiling.test_cpu_32_l`, `test_cuda_32_l`), skipped since they were added for a reported ~2x footprint, run again: measured on torch 2.15 they match their core twins (CUDA allocated peak equal to the byte, CPU `rss_diff` within 2%) ([#810](https://github.com/speediedan/interpretune/issues/810))
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
