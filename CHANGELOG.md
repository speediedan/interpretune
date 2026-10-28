# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `CircuitTracerConfig.tune_base` lets a circuit-tracer composition tune its base model as `base` under transcoders that stay `instrument` unless promoted; base tensors are not promotable. Attribution now runs with the replacement model frozen, since circuit-tracer attributes with `backward()`: a thawed embedding failed it outright and other thawed base parameters accumulated attribution gradients into `.grad`. Attributing with thawed base parameters warns, suggesting re-fitting the transcoders when their fit has degraded ([#813](https://github.com/speediedan/interpretune/issues/813))
- circuit-tracer compositions are tunable through their transcoders (`interpretune.adapters.circuit_tracer.tunable_view.CircuitTracerView`), replacing the refusal they shipped with: every transcoder tensor starts as `instrument` and is promotable to `latent:transcoder`, base-model parameters are refused by name, and `offload='disk'` refuses the composition. `CircuitTracerConfig.lazy_delta_rank` gives lazily loaded transcoder weights a trainable low-rank correction that every circuit-tracer read (whole-matrix and decoder-row) applies, so the memory-saving lazy layouts stay tunable. The interp-engine replacement model is no longer registered twice on the module, and the nnsight backend's transcoders are registered once. `interpretune.adapters.circuit_tracer.drift.TranscoderFitMeter` measures per-layer transcoder drift: fraction of variance unexplained, $L_0$, and dead and dense feature counts ([#787](https://github.com/speediedan/interpretune/issues/787), [#709](https://github.com/speediedan/interpretune/issues/709))
- Data-parallel placement for the core training and test loops (`SessionRunnerCfg.placement`: `single`, `ddp` or `fsdp2`, launched with `torchrun`; `interpretune.runners.distributed`). Both data-parallel placements apply FSDP2's `fully_shard` in place (`ddp` over a replicate mesh), so no wrapper stands between callers and `module.model`. The runner builds the session one rank at a time with data prepared once per host, broadcasts rank 0's weights, shards (`fsdp2` by the classes `shard_units` names, deepest first), rebuilds each loader over a `DistributedSampler` and averages logged metrics across ranks. Mismatched launches, layer splits, shared accelerators, non-contiguous parameters and loaders that cannot be rebuilt faithfully are refused by name. Two-rank CPU runs of both placements match one process training on their combined batch, step for step. `it_init` gains `prepare_data` and `configure_optimizers` switches, and `it_configure_optimizers` builds the optimizers on its own ([#20](https://github.com/speediedan/interpretune/issues/20), [#803](https://github.com/speediedan/interpretune/issues/803))
- Memory-profiling delta pairs for the analysis adapters (`tests/profiling_delta`): `sae_lens_train`/`sae_lens_test` against SAELens's own `SAETransformerBridge` with the same SAE attached through `add_sae`, `nnsight_test` against a bare nnsight `LanguageModel`, and `circuit_tracer_test` against a bare circuit-tracer `ReplacementModel` attributing the same prompt. Their bounds, and those of `tl_train`/`tl_test`, are frozen on both vendors ([#825](https://github.com/speediedan/interpretune/issues/825), [#823](https://github.com/speediedan/interpretune/issues/823))

- Latent models for tuning (`interpretune.tuning.latents`): any `nn.Module` with `encode` and `decode` (an SAE, a transcoder, a block-sparse featurizer) can be spliced in at a site (`LatentSite`: the module it reads, input or output, and the module whose output it rewrites, by replacement, replacement with a detached error term, or observation), registered on the module as `latent_models.<name>`, and scheduled by finetuning-scheduler as `role:latent:<name>` (or held as an `instrument` until a schedule promotes it). `LatentModelsMixin` declares the roles through `tunable_roles()` on any backend. A new Tuning page describes the seam [#812](https://github.com/speediedan/interpretune/issues/812)

- Learned interventions for tuning (`interpretune.tuning.interventions`): an `nn.Module` that rewrites a site's output (`SteeringVector`, a zero-initialized learned direction, is the first kind) can be spliced in at a module path, registered on the module as `interventions.<name>`, and scheduled by finetuning-scheduler as `role:intervention:<kind>` (`role:intervention:steer` for steering vectors), or applied untrained as an `instrument`. `InterventionsMixin` declares the roles through `tunable_roles()` on any backend, beside `LatentModelsMixin`. A new Tuning page describes the seam [#812](https://github.com/speediedan/interpretune/issues/812)

- Memory-profiling deltas (`tests/profiling_delta`): each sampled composition runs beside a matched upstream arm (plain PyTorch and Hugging Face, a vanilla `LightningModule`, or the same `TuningScheduler` in a plain loop), one subprocess per arm, and the extended profiling tier gates on the interpretune-minus-upstream delta in peak and retained bytes, per-phase leak slope and phase-transition step. Bounds are frozen per vendor from NVIDIA and AMD measurements (`freeze.py`); planted leak and overhead controls prove the gate can fail; phased runs also assert exact structural invariants; release and tag builds run it report-only ([#823](https://github.com/speediedan/interpretune/issues/823), [#811](https://github.com/speediedan/interpretune/issues/811))

- Phase reports for the GPU pipeline, ported from finetuning-scheduler: each GPU phase's manifest is collected on the CPU before the GPU work, every pytest process the phase starts records its outcomes (`tests/gpu_ci/phase_plugin.py`), and a report per phase counts them against the manifest, collected beside passed, skipped and failed, under the vendor the agent reported (`tests/gpu_ci/phase_report.py`). A failing phase fails the job, and so does an empty one on a full run; the manifests, records and reports are published as the job's phase evidence. A release pipeline (`.azure-pipelines/gpu-release.yml`) runs one vendor-demanded leg per vendor and checks both legs' evidence (`tests/gpu_ci/release_eligibility.py`): one full, clean run per vendor, with a reviewed reason for every skip (`tests/gpu_ci/reviewed_exclusions.json`) ([#703](https://github.com/speediedan/interpretune/issues/703))

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

#### Tuning through finetuning-scheduler views
- Each module composition declares a finetuning-scheduler `TunableView` (`build_tunable_view()`, with cooperative `tunable_roles()`), attached as `tunable_view` after model init: the identity view with PEFT `adapter:<name>` roles by default, `TransformerBridgeView` in TransformerLens names for TransformerLens compositions, `latent:sae` for spliced SAEs, and refusals with reasons for remote or undispatched nnsight, circuit-tracer, TransformerLens bridges holding quantized or packed parameters, and compositions whose `build_tunable_view()` returns `None` (#758, #759)
- `interpretune.testing.conformance.TuningConformance` runs finetuning-scheduler's view conformance cases against a composition, selected by its declared support (#760)
- The default optimizer (`optimizer_parameters()`) covers an attached, trainable view's root parameters, less `fitted` and `instrument` tensors: every view is rooted at the interpretune module, so trainable modules registered beside `module.model` (interventions, spliced SAEs) are optimized with no adapter extending the hook. Compositions without a view, or with a refusing one, keep the model's own parameters (#761)
- `interpretune.tuning.core_loop.FTSCoreCallback` drives finetuning-scheduler's `TuningScheduler` from the core training loop (`SessionRunnerCfg.callbacks`): the tuning scheduler is built over the session's module and its view, its optimizer and lr schedulers are installed as the module's, and its epoch, step and end boundaries, transitions and stops follow the loop's. A tuning scheduler rooted at another module is refused ([#762](https://github.com/speediedan/interpretune/issues/762))
- Local, dispatched nnsight compositions register their HF module on the module as `hf_model` through the cooperative `register_tunable_modules()` hook (run after every init path), so the optimizer, checkpoints and the identity view reach its parameters; remote (NDIF) and undispatched compositions are refused with their reasons (#774)
- A top-level Tuning docs section: the adapter-agnostic Tuning page, ending in a summary of what each composition declares (with the issues tracking each refused composition's tuning paths), and a tuning page per packaged adapter (core/HF, TransformerLens, sae_lens, nnsight, circuit-tracer, interp-engine) to hold its example usage (#812)

- An RTE phased fine-tuning example that runs one finetuning-scheduler schedule from the interpretune CLI alone, on either training-loop interface: the core loop (`interpretune --run_command train`) with `FTSCoreCallback`, a `PlateauTransition` and a `BestCheckpoint`, or Lightning (`interpretune --lightning_cli fit`) with `FinetuningScheduler`, `FTSEarlyStopping` and `FTSCheckpoint`, configured alike. The `rte` component gains an `rte_ft` variant (`rte_ft.gemma3.core`, #697's schedule for gemma-3-1b-it in fp32 on one GPU, and the CPU-sized `rte_ft.gpt2.core`), and ships its schedules (`ft_schedules/`) and one run configuration per interface (`runs/`) beside `rte_boolq.py`. A CPU test runs both GPT-2 configurations through the CLI and checks that the two interfaces thaw the same parameters each epoch, transition in the same epochs and keep the same best checkpoint ([#765](https://github.com/speediedan/interpretune/issues/765))

- The CLI runs a session from a cached Hub component configuration: `session_ref: <repo_id>:<config_key>[@<revision>]` (in a configuration file or on the command line) seeds the session, and the configuration files' session keys merge over it. `hub://<org>/<repo>[@<revision>]/<path>` names a file a cached component ships, in `--config` arguments and in configuration values such as a schedule path, and resolves cache-only. A key-less `interpretune.hub.pull` also fetches the manifest's `extra_files`, which these references read ([#847](https://github.com/speediedan/interpretune/issues/847))

- `FTSCoreCallback` builds `tuning_scheduler_kwargs` values given as `class_path` mappings (a transition policy, a checkpoint policy, a list of gates) when it builds the tuning scheduler, so a core-loop fine-tuning run's policies come from its configuration while the callback's arguments stay plain configuration ([#847](https://github.com/speediedan/interpretune/issues/847))

- A fine-tuning benchmark: gemma-3-1b-it's RTE phased fine-tuning on the core loop, trained through the CLI and scored on its last validation accuracy. A registry entry can name its run command (`run_command`, `test` by default), its timeout (`timeout_s`) and a `hub://` run configuration ([#848](https://github.com/speediedan/interpretune/issues/848))

- A run chooses its finetuning-scheduler view in configuration: `it_cfg.tunable_view` (a `class_path` and `init_args`) is instantiated with the module as the view's root, ahead of the composition's `build_tunable_view()`, and a view class taking `roles` receives the composition's roles. A TransformerLens composition can then schedule its own (physical) parameter names (finetuning-scheduler's `CanonicalView`) or turn the implicit norm thaw off (`TransformerBridgeView` with `implicit_norm_thaw: false`) with no subclass, as the retired strategy adapter's `model_view_cfg` allowed. `TransformerBridgeView.to_physical` resolves the physical name of a parameter no TransformerLens name reaches (a norm TransformerLens does not name, a fused projection) to itself, so a schedule can name it exactly. A phased parity run covers each mode, and the committed phase collection manifest is re-measured ([#841](https://github.com/speediedan/interpretune/issues/841))

### Changed

- The finetuning-scheduler schedules under `tests/benchmarks/configs/rte_boolq/ft_schedules/`, which nothing referenced, are retired: the GPT-2 three-phase schedule continues as the `rte` component's `ft_schedules/rte_ft.gpt2.yaml` in current idioms (each later phase sets its `lr`, phase 0 is bounded by `max_transition_epoch`), beside #697's gemma-3-1b-it schedule ([#14](https://github.com/speediedan/interpretune/issues/14))

- `SessionRunnerCfg.callbacks` instantiates an entry given as configuration (`class_path` and `init_args`), so a core-loop FTS run builds `FTSCoreCallback` from its parsed configuration alone: the configuration dumps and reloads unchanged and reproduces the tuning scheduler's `config_hash`. `schedule` and plain-data `tuning_scheduler_kwargs` are configuration-expressible; object-valued arguments (`tuning_scheduler_factory`, optimizer factories, policies, gates) stay Python-only ([#809](https://github.com/speediedan/interpretune/issues/809))

- The core loop resumes from its own checkpoint: `resume_from` (`SessionRunnerCfg.resume_from`) takes a `checkpoint_state` taken in a callback's `on_train_epoch_end`, or the path it was saved to, loads it after every callback's `on_train_start`, completes that epoch's epoch-interval lr scheduler step and continues at the next epoch. `load_checkpoint_state` now loads callbacks' state before the module's and the optimizers'. A core-loop FTS run resumes this way: `FTSCoreCallback`'s state is finetuning-scheduler's `TuningScheduler.resume_state()`, which marks the resume point with its checkpoint policy, and loading it rebuilds the optimizer phase by phase before the loop loads the optimizer's state (finetuning-scheduler pinned at `87e1d6d3`, its #144) ([#809](https://github.com/speediedan/interpretune/issues/809))

- TransformerLens compositions tune through their `TransformerBridgeView` under either loop: `TransformerBridgeStrategyAdapter`, its `TLNamesModelView`, the `ModelView` base (`interpretune.adapters.model_view`) and the `finetuning_scheduler.strategy_adapters` entry point (`transformerbridge`) are removed, with no shim. Drop `custom_strategy_adapters` and `strategy_adapter_cfg` from a `FinetuningScheduler` configuration: TransformerLens names resolve through the attached view, and a composition that names the module's own parameters overrides `build_tunable_view()` to return the identity view. The callback reports thawed parameters as physical tensors, so a TransformerLens-names schedule now counts the fused QKV weights its q/k/v splits are views of ([#763](https://github.com/speediedan/interpretune/issues/763))

- finetuning-scheduler moves from the `lightning` extra to a new `tune` extra (with `peft` and `bitsandbytes`, which the core HF path's LoRA and quantized loading import), and the `lightning` extra names `lightning` and includes `tune` ([#764](https://github.com/speediedan/interpretune/issues/764))

- Views attach as `tunable_view`, the tunable view protocol's attribute in finetuning-scheduler (pinned at `4ef546f2`, its #137), and an attached object that is not a view raises its `TunableViewError` ([#758](https://github.com/speediedan/interpretune/issues/758))

- The TransformerLens parameter-mapping tests run against the view on every architecture they covered (GPT-2 unprocessed and processed, Llama 3, Gemma 2), with finetuning-scheduler's mapping cases per architecture, and retire the strategy-adapter copy of those checks ([#760](https://github.com/speediedan/interpretune/issues/760))

- The docs use the full window width on every page, as finetuning-scheduler's do: both sidebars are pinned at fixed widths and the article takes the rest, rather than widening only notebook and table pages up to a 70rem cap

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
- A Lightning checkpoint of a session whose prompt config uses the tokenizer's chat template (`ChatTemplatePromptConfig`) loads with `weights_only=True` again: the datamodule's hyperparameters held the bound tokenizer, which Lightning refused, so restoring a best checkpoint (at a finetuning-scheduler phase transition, for one) failed. A pickled prompt config now leaves the binding out, and a deep copy shares the bound tokenizer rather than copying it ([#765](https://github.com/speediedan/interpretune/issues/765))
- `configure_optimizers` returns no lr schedulers, rather than `[None]`, when `lr_scheduler_init` is empty (Lightning refused the `None` scheduler, and the core loop kept an `LRSchedulerConfig` wrapping `None`), and returns `None` when `optimizer_init` is empty ([#765](https://github.com/speediedan/interpretune/issues/765))
- The core training loop dispatches `on_train_batch_end` as optional, like every other hook in `run_step`: a module without it (the hub's RTE demo module, for one) failed at the first optimizer step ([#817](https://github.com/speediedan/interpretune/issues/817))
- `ITMeta` builds a class created with no build context over a composition it already built as an ordinary subclass, so FSDP2's `fully_shard` (which wraps a module by subclassing its class) can shard an interpretune module, the root unit included; a partial or invalid build context is still refused by name ([#820](https://github.com/speediedan/interpretune/issues/820))
- A ROCm tolerance for the circuit-tracer edge-prediction conformance case's active-feature check (`ROCM_EDGE_ATOL`, 2.6e-3 against 1e-3 elsewhere): on the AMD agent one of 3,030 active features measured 1.29e-3 off its edge prediction in two runs, the same element both times ([#703](https://github.com/speediedan/interpretune/issues/703))
- The GPU job warms the circuit-tracer transcoders (`mntss/clt-gemma-2-2b-426k`, `mwhanna/gemma-scope-transcoders`) from a GPU-only manifest (`tests/gpu_ci/hf_warm_manifest_gpu.yaml`) before its test phases, with retries, when the plan runs the notebook tests that load them (`plan_gpu_run.py --covers`). They were downloaded lazily inside the circuit-tracer notebook kernels, and on the AMD agent a partial cross-layer transcoder download (about 71 GB of `*.incomplete` blobs accumulated) crashed `test_attribution_analysis_notebook[analysis_inj_salient_logits_CLT]` on every run ([#703](https://github.com/speediedan/interpretune/issues/703))
- Memory profiling on a host without CUDA no longer fails at session end: `MemProfiler.dump_memory_stats()` creates the snapshot directory that setup creates only for CUDA allocator history, which is disabled when CUDA is unavailable ([#821](https://github.com/speediedan/interpretune/issues/821))
- The NNsight + Lightning memory-profiling cases (`test_ns_profiling.test_cpu_32_l`, `test_cuda_32_l`), skipped since they were added for a reported ~2x footprint, run again: measured on torch 2.15 they match their core twins (CUDA allocated peak equal to the byte, CPU `rss_diff` within 2%) ([#810](https://github.com/speediedan/interpretune/issues/810))
- `test_l_profiling[test_cuda_bf16]` reported 2.6x its expected memory only because of the rerun artifact below; its first attempt retained 33,554,432 B (32 MiB) less than expected at the end of the test. Its fp32 twin dropped by the same 32 MiB, so it is not weights (the default cuBLAS workspace is that size). The memory-profiling expectations are re-measured on torch 2.15 on both vendors: the two HF test cases, and the TransformerLens cases after the q/k/v fix below (132 MB less allocated at the end of the test case, 156 MiB less in training). ROCm keeps overrides only where its measurement differs beyond tolerance (the HF test cases' allocated and reserved peaks) ([#819](https://github.com/speediedan/interpretune/issues/819))
- The TransformerLens and SAE Lens compositions no longer hold a second copy of every attention layer's query, key and value weights on the device (about 81 MiB on gpt2, growing with the model). The bridge's split q/k/v weights are views of the fused attention weight, and the adapters moved the bridge to its device after wrapping, which converts each parameter separately and gives every view its own storage. The HF model now moves before it is wrapped, as `boot_transformers` does; `test_tl_bridge_keeps_qkv_views` checks the views on CPU and CUDA ([#824](https://github.com/speediedan/interpretune/issues/824))
- The NNsight memory-profiling cases on ROCm (`test_ns_profiling.test_cuda_32` and its Lightning twin) reported about three times gpt2's footprint (1.50 GB allocated against 0.53 GB). Two causes: the cases lacked the ROCm `reserved_bytes.all.peak` override that `test_l_profiling.test_cuda_32` carries (the HIP caching allocator reserves 689,963,008 B, measured identically for native nnsight), so the first attempt failed on reserved bytes alone, and `special_tests.sh`'s reruns then measured again in a process still holding each failed attempt's model. The cases take the override, and the memory-profiling tests (`test_l_profiling`, `test_ns_profiling`, `test_tl_profiling`) are no longer rerun, so a footprint failure reports its first attempt ([#810](https://github.com/speediedan/interpretune/issues/810))
- SAEs spliced at init (`add_saes_on_init=True`) on the TransformerBridge path are now trained: they are registered on the module as `attached_saes`, built into the default optimizer through the new cooperative `optimizer_parameters()` hook, and saved in its `state_dict`. Before, the bridge kept them outside every module tree, so their parameters received gradients but were never stepped, and nothing reported it. The test `req_grad_mask` now resolves SAE parameter names and fails on a name it cannot resolve instead of silently ignoring it (#772)
- The core training loop honors `SessionRunnerCfg.max_steps` (counting optimizer steps), which it previously accepted and ignored (#761)
- The core training loop's `on_train_epoch_start` / `on_train_epoch_end` hooks are optional, as its batch hooks are; a module without them raised `AttributeError` (#761)
- The core training loop reads the optimizer from the module at every step rather than once, so a replaced optimizer is used (#761)
- Circuit-tracer `isinstance` assertions replaced with class-name checks to avoid false failures from pytest `importlib` import mode causing module double-loading
- Coverage collection fix for NNsight's `sys.settrace()` disabling behavior
- HuggingFace transformers v5 compatibility with configurable test reruns (`--reruns 2 --reruns-delay 5`)
- Dataset cache reset handling to prevent cache-dependent expected value divergence
- Tokenizer usage pattern fix for `force_prepare_data` enabled scenarios
- The `sae_lens_test` delta pair compares equal work: the native SAELens arm receives interpretune's generate kwargs, which the root wrapper's `*args, **kwargs` signature had hidden, so it generated the bridge's default 10 tokens against interpretune's one and its retained attention patterns differed by about 24 MiB. Its bounds are now frozen on both vendors ([#838](https://github.com/speediedan/interpretune/issues/838))

## [0.1.0] - 2025-XX-XX

### Added

### Fixed

### Changed

### Deprecated
