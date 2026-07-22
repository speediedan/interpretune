# CLAUDE.md - Interpretune

## Project Overview

Interpretune is a flexible framework for collaborative AI world model analysis and tuning. **Pre-MVP stage** — features and APIs are subject to change.

**Stack:** Python 3.10+ (CI on 3.13), PyTorch 2.7.1+, transformer_lens >= 3.0.0, sae_lens, datasets, jsonargparse

## Tooling Reliability

- Pylance-backed tooling can hang or time out on longer operations. If a Pylance request stalls, switch to direct file inspection, search, or terminal-based validation rather than waiting on the language server indefinitely.

## Build & Dev Environment

Uses `uv` for dependency management. For Venvs, use `/mnt/cache/$USER/.venvs/` (preferred for hardlink perf), fall back to `~/.venvs/` if that path isn't available.

Place the venv on the same filesystem as the UV cache when possible. `--venv-dir=/mnt/cache/$USER/.venvs` is the preferred build-script pattern because it avoids UV hardlink warnings and matches the standalone-process wrappers used in this repo.

### Common Environment Variables

```bash
export IT_VENV_BASE=/mnt/cache/${USER}/.venvs
export IT_TARGET_VENV=it_latest
export IT_REPO_DIR=${HOME}/repos/interpretune
```

These are used by dev scripts and can be set in your shell profile. `IT_VENV_BASE` controls where venvs are created/found, `IT_TARGET_VENV` names the active venv, and `IT_REPO_DIR` points to the working tree.

```bash
# Build dev environment
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=${IT_TARGET_VENV}

# Activate
source ${IT_VENV_BASE:-~/.venvs}/${IT_TARGET_VENV}/bin/activate

# With custom venv location (recommended)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --venv-dir=/mnt/cache/$USER/.venvs

# CPU-only (CI)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --torch-backend=cpu

# From-source packages
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest \
  --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1"
```

### From-Source Builds (multi-repo integrated env)

The integrated dev env (`it_latest`) builds the salient upstream deps **editable from source** via repeated
`--from-source="pkg:path:extras:ENVVAR=...:FLAGS=..."` directives. Canonical full rebuild (SAEDashboard,
SAELens, circuit-tracer, TransformerLens, nnsight all from local checkouts):

```bash
./scripts/build_it_env.sh --repo-home=${IT_REPO_DIR} --target-env-name=it_latest --venv-dir=${IT_VENV_BASE} \
  --from-source="sae_dashboard:${HOME}/repos/SAEDashboard:dev:UV_EXCLUDE=${IT_REPO_DIR}/requirements/ci/excludes.txt:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt" \
  --from-source="sae_lens:${HOME}/repos/SAELens:dev:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt:FLAGS=-r ${IT_REPO_DIR}/requirements/ci/sl_uv_requirements.txt" \
  --from-source="circuit_tracer:${HOME}/repos/circuit-tracer:dev:UV_EXCLUDE=${IT_REPO_DIR}/requirements/ci/excludes.txt:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt" \
  --from-source="transformer-lens:${HOME}/repos/TransformerLens" \
  --from-source="nnsight:${HOME}/repos/nnsight:all:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt"
```

Rules of thumb:

- `UV_EXCLUDE` (excludes file) is **required** when interpretune, circuit-tracer, AND transformer-lens are
  all from source, so exactly one directive controls the transformer-lens install. `UV_OVERRIDE` alone
  suffices when only two of the three are from source.
- SAELens is Poetry-legacy (not PEP 621), so uv needs an exported requirements file passed via `FLAGS=-r`.
  The vendored copy lives at `requirements/ci/sl_uv_requirements.txt`; regenerate it whenever SAELens
  `pyproject.toml`/lock changes (`poetry export --all-groups --all-extras` → post-process).
- nnsight should be installed first among the from-source set (its vllm-era pins occasionally need the
  override file).
- **After every rebuild**: recreate circuit-tracer's untracked `temp_hf_override.txt` (stash/restore or
  per the circuit-tracer admin notes), then validate with `python requirements/utils/collect_env_details.py`
  (also feeds `salient_pkg_versions` provenance used by the benchmark registry).

### Git Dependency Caching & Pins

When a from-source package brings in git-pinned dependencies, UV caches those resolved commits and later editable installs should respect them. Interpretune relies on the `override-dependencies` configuration in `pyproject.toml` so editable installs can coexist with those pinned git requirements instead of fighting them.

Where pins live and how to refresh them:

- `pyproject.toml` `[tool.uv] override-dependencies` — the transformer-lens git SHA pin (the highest-risk
  bump: as of 2026-07 the pin is ~260+ commits behind TL main; probe a TL bump in a scratch env and run the
  dashboard parity gates + core suite before folding it into `it_latest`).
- `requirements/ci/requirements.txt` — the CI lock; regenerate with `./requirements/utils/lock_ci_requirements.sh`.
- Any rebuild that changes benchmark-relevant deps (torch, transformer_lens, nnsight, sae_lens,
  sae_dashboard) must end with the dashboard parity gates green and a fresh benchmark wave from clean
  committed heads (see the Neuronpedia Dashboard Pipeline section below).

## Testing

```bash
# Basic test run (mirrors ci_test-full.yml)
cd /home/runner/work/interpretune/interpretune && python -m pytest src/interpretune tests -v

# With coverage
python -m pytest --cov=src/interpretune --cov-append --cov-report= src/interpretune tests -v
python -m coverage report

# Standalone tests (use inline env vars, NOT export, to avoid marker conflicts)
IT_RUN_STANDALONE_TESTS=1 python -m pytest tests/core/test_transformer_lens.py::TestGemma2ParameterMapping -v
unset IT_RUN_STANDALONE_TESTS

# Standalone test harness
./tests/special_tests.sh --mark_type=standalone
./tests/special_tests.sh --mark_type=standalone --filter_pattern='ParameterMapping'

# Profiling tests
IT_RUN_PROFILING_TESTS=1 python -m pytest tests/parity_acceptance/test_it_l.py::test_l_profiling -v
unset IT_RUN_PROFILING_TESTS
```

**⚠️ Standalone marks must be at the test METHOD level, not the class level.**
`pytest_collection_modifyitems` in `tests/conftest.py` uses `item.own_markers` — which only contains
markers on the test *function* itself, not inherited from a parent class.  Class-level `@RunIf(standalone=True)`
decorators are invisible to the standalone collection filter, so those tests are silently excluded from
standalone runs.

- Always apply `@RunIf(standalone=True)` (or `marks="standalone"` / `marks="l_standalone"`) to **individual
  test methods**, never to the class.
- For memory-intensive tests that previously used standalone as a workaround, prefer
  `@pytest.mark.usefixtures("cleanup_memory")` at the method level — this triggers `gc.collect()` after
  each test without sacrificing cross-platform CI signal.
- **TODO/BUG:** Fix `pytest_collection_modifyitems` in `tests/conftest.py` to use `item.iter_markers()`
  instead of `item.own_markers` so class-level standalone marks are properly collected.

Test reruns (`--reruns 2 --reruns-delay 5`) are used in CI for transient httpx/HF timeouts.

**⚠️ CRITICAL: Running Tests in Background to Avoid Truncation/OOM Kills**

The basic test suite runs ~30 minutes; the full suite with `gen_it_coverage.sh` ~50 minutes. **Always run tests in the background with `nohup`/`disown`** — foreground terminal sessions and piped output (`tee`) will be killed by terminal lifecycle management or OOM before completion. Set prudent timeouts and check progress periodically.

```bash
# RECOMMENDED: Run basic test suite in the background (avoids truncation/OOM kills)
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
nohup python -m pytest src/interpretune tests -v > /tmp/it_test_results.txt 2>&1 &
disown
echo "Test PID: $!"

# Monitor progress (check periodically)
tail -20 /tmp/it_test_results.txt
grep -c "PASSED\|FAILED\|ERROR" /tmp/it_test_results.txt
# Check if process is still running
ps -p <PID> -o pid,etime,rss,cmd

# IMPORTANT: Kill the process before starting another test run to avoid conflicts
kill <PID>  # or: pkill -f "pytest.*src/interpretune"
```

**Full coverage run (background via harness):**

```bash
# Generate coverage with no rebuild (recommended for quick iteration)
# Outputs full logs to /tmp dir with timestamp for later inspection
${IT_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${IT_REPO_DIR}/scripts/gen_it_coverage.sh \
  --repo-home=${IT_REPO_DIR} \
  --target-env-name=${IT_TARGET_VENV} \
  --venv-dir=${IT_VENV_BASE} \
  --no-rebuild-base

# Preferred for debugging: use --allow-failures to continue past failures and --no-reruns to expedite debugging
${IT_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${IT_REPO_DIR}/scripts/gen_it_coverage.sh \
  --repo-home=${IT_REPO_DIR} \
  --target-env-name=${IT_TARGET_VENV} \
  --venv-dir=${IT_VENV_BASE} \
  --no-rebuild-base \
  --allow-failures \
  --no-reruns

# Monitor coverage progress
tail -f $(ls -rt /tmp/gen_it_coverage_it_* | tail -1)
```

**Watcher rules for background runs (added after two missed completions, 2026-07-22):**

1. **Never `pgrep -f` a pattern your own watcher command contains** — the watcher matches
   itself and waits forever (a watcher polling `pgrep -f "gen_it_coverage.sh"` IS a process
   whose command line contains `gen_it_coverage.sh`). Use the bracket trick
   (`pgrep -f "[g]en_it_coverage.sh"`) or, better, watch the concrete PID.
2. **Prefer PID-based watches over output-grep watches.** `manage_standalone_processes.sh`
   prints the wrapper PID at launch — capture it and wait with
   `until ! kill -0 <PID> 2>/dev/null; do sleep 60; done`. Output-grep watchers silently hang
   when the terminal line's wording changes (a `grep -qE "COVERAGE RUN COMPLETE|..."` watcher
   missed a run that ended with `Exiting with status code 1`).
3. **Cover every terminal state.** A run can end green, end red, or die — the watch condition
   must fire for all three (PID-exit does this for free); then read the log tail to classify.

Note: coverage collection takes ~50 minutes (longer with `--run-all-and-examples`).

**Pre-PR-wave gate**: before opening a coordinated multi-repo PR wave (Scalable Dashboards Phase 7 and
similar), a full `gen_it_coverage.sh` pass is mandatory, not optional. Run it in
`--run-all-and-examples` mode (adds optional profiling, optional-other, and example tests to the phase
split); use `--allow-failures` + `--resource-debug` for diagnosing passes, but the final gate run must be
free of real failures. It must run *after* refreshing the
registered benchmarks (`tests/benchmarks/` — see Commit & PR Requirements below and
`tests/benchmarks/README.md`; this is distinct from the neuronpedia dashboard benchmark suite under
`scripts/run_dashboard_benchmark_suite.py`), so `IT_RUN_BENCHMARK_TESTS=1` runs inside the coverage pass
validate against current `expected_accuracy`/`salient_pkg_versions`, not stale targets from a prior
dependency state. After the local gate is green, all REMOTE checks must also pass: the GitHub Actions
matrix (`ci_test-full`, `type-check`) and the self-hosted Azure GPU pipeline
(`.azure-pipelines/gpu-tests.yml`) — trigger via a ready-for-review PR targeting `main` (releases the
Azure PR trigger; approve the gated run per `.github/skills/az-pipelines-debug/SKILL.md`) or a manual
pipeline run.

If the coverage harness reports a conflicting pytest process, check whether it is still active before starting a second run. Old hung collectors are usually safe to kill with `pkill -f "pytest.*--collect-only"`; recently started runs should usually be allowed to finish.

The local coverage harness now mirrors the Azure GPU pipeline's phase split:
- `Testing: standard` runs CPU-only with `CUDA_VISIBLE_DEVICES=''`
- `Testing: standard gpu cuda-marked` reruns only regular CUDA / bf16-marked tests with `IT_RUN_CUDA_TESTS=1`
- `Testing: standalone gpu` and `Testing: CI Profiling` remain separate special-test phases

When debugging fixture or CUDA memory growth locally, add `--resource-debug` to `gen_it_coverage.sh` or
`special_tests.sh`. That exports the canonical `IT_RESOURCE_DEBUG=1`
flags used by `tests/conftest.py`, `tests/analysis_resource_utils.py`, and the coverage harness summary
parser for per-test / per-fixture / per-GPU logging.

If semantic concept-direction intervention parity drifts unexpectedly, use the normal gate in
`tests/core/test_analysis_backend_parity.py` first. For deeper upstream sanity checking, run
`tests/upstream_parity/extract_upstream_ct_semantic_reference.py` and consult `tests/upstream_parity/UPSTREAM_CT_PARITY_DEBUG.md`
for the current three-way upstream/native/op reference snapshot. Keep this as a manual debugging
tool rather than part of the regular test suite.

## Neuronpedia Dashboard Pipeline & Benchmarks

The repo carries the production pipeline and benchmark tooling for the Scalable Dashboards workstream:

- **Pipeline**: `src/interpretune/utils/neuronpedia_dashboard_pipeline.py` (+ `neuronpedia_db_utils.py`);
  docs in `docs/neuronpedia_dashboard_pipeline.md` (multi-GPU workers, batch-level resume markers,
  `--runner-overlap-batch-packaging`, `--overlap-local-db-import`, `--layer-list`).
- **Profiling/benchmarks**: `scripts/profile_neuronpedia_dashboard_generation.py` (per-leg presets),
  `scripts/run_dashboard_benchmark_suite.py` (3-way/scaling/full waves + reviewer packaging; usage in
  `scripts/dashboard_benchmark_suite_usage.md`), `scripts/sweep_neuronpedia_dashboard_configs.py`
  (batch-shape probes). Tests: `tests/core/test_neuronpedia_dashboard_pipeline.py`,
  `tests/core/test_dashboard_benchmark_artifacts.py`.
- **Env vars**: always set the large-cache locations for pipeline/benchmark runs —
  `IT_NP_CACHE=/mnt/cache_extended/$USER/.cache/huggingface/interpretune/neuronpedia`,
  `HF_HOME`/`HF_DATASETS_CACHE`/`HF_HUB_CACHE` under `/mnt/cache_extended/$USER/.cache/huggingface`.
- **Reproducibility policy**: commit all four repos (interpretune, SAEDashboard, SAELens, neuronpedia)
  before regenerating a packaged 3-way benchmark wave; add a benchmark-regeneration note to those commit
  messages; the package manifest records the `SD-*/SL-*/NP-*/IT-*` lineage and refuses dirty trees.
- **GPU run hygiene**: run waves/legs via `nohup ... & disown` with PID-based waiters; before a wave, kill
  stale GPU holders by exact PID from `nvidia-smi --query-compute-apps` (pytest HF-teardown hangs are the
  usual culprits; never `pkill -f pytest` from a command line that itself matches the pattern).

## Linting & Code Quality

Ruff is used via pre-commit (not installed standalone):

```bash
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files
pre-commit run --all-files
```

**Note:** `tests/*_parity/` directories are currently excluded from pre-commit (imported research code).

## Code Style

- **Line length:** 120 chars
- **Type hints:** `from __future__ import annotations` (modern syntax)
- **Docstrings:** docformatter-compatible (wrap summaries at 115, descriptions at 120)
- **Ruff rules:** E, W, F enabled; E731, F722 ignored (jaxtyping compat)
- **McCabe complexity:** max 10

## Architecture

### Source Layout

```
src/interpretune/           # Main package (version from src/__about__.py)
├── adapters/               # Framework integrations (TL, SAE, Lightning, CircuitTracer, NNsight)
├── analysis/               # Analysis tools with ops system and dispatcher
├── base/components/cli.py  # CLI entry point
├── config/                 # Dataclass-based configuration system
├── extensions/             # MemProfiler, DebugGeneration, NeuronpediaIntegration
├── runners/                # SessionRunner, AnalysisRunner
├── utils/                  # Helpers, exceptions, repr utilities
├── protocol.py             # Type protocols (ITModuleProtocol, ITDataModuleProtocol)
├── registry.py             # ModuleRegistry, RegisteredCfg
└── session.py              # ITSession
src/it_examples/            # Example experiments, configs, notebooks
tests/                      # 744 test functions
├── core/                   # Core tests
├── benchmarks/             # End-to-end experiment benchmarks (see tests/benchmarks/README.md)
├── *_parity/               # Research parity tests
└── examples/               # Example-based tests
```

### Key Patterns

- **Lazy imports:** Heavy deps (transformer_lens, lightning, neuronpedia) loaded lazily via `_light_register.py` and PEP 562 `__getattr__`. Import-time optimization is critical.
- **Protocol-based:** ITModuleProtocol, ITDataModuleProtocol, AnalysisOpProtocol
- **Composition:** `CompositionRegistry` manages adapter combinations (core, lightning, transformer_lens, sae_lens, circuit_tracer, nnsight)
- **Backend-agnostic analysis ops:** Keep backend-specific imports out of `src/interpretune/analysis/ops/definitions.py`. If an op needs backend-specific behavior, extend the `AnalysisBackend` seam rather than importing backend code directly into the op definition.
- **Framework agnosticism:** Module definitions (e.g. `RTEBoolqSteps`) should NOT contain framework-specific hooks or accumulation logic. Use `ClassificationMixin` for prediction accumulation and metric reporting. Hook dispatch via `_call_itmodule_hook(..., optional=True)` handles missing hooks gracefully.
- **Framework-agnostic logging:** `CoreHelperAttributes` provides real `log()` / `log_dict()` methods that accumulate metrics in `_logged_metrics`. The core runner prints averaged metrics at test epoch end. Lightning modules use `LightningModule.log()` / `log_dict()` instead. User code calls `self.log()` / `self.log_dict()` regardless of context.
- **ClassificationMixin.setup():** Cooperatively calls `super().setup()` then initializes `classification_mapping` if configured. `collect_answers()` computes metrics and calls `self.log_dict()` — no custom accumulation logic needed.
- **Generation config:** Use `HFGenerationConfig` (applies params to `model.generation_config`) for HF-backed models. Use `CoreGenerationConfig` (passes params as `generate_kwargs`) only for `HookedTransformer` models. The NNsight adapter applies `HFGenerationConfig.model_config` to the underlying HF model via `_apply_generation_config()`.
- **User-facing op calls:** In notebooks, examples, and one-off experiment scripts, prefer `import interpretune as it` plus direct top-level op wrappers such as `it.concept_direction(...)` or `it.compute_attribution_graph(...)`. Ensure `interpretune.analysis` has been imported once so those wrappers are registered. Avoid `DISPATCHER.get_op(...)` outside dispatcher-internals work.
- **Class metadata:** Core classes use `_it_cls_metadata` (ITClassMetadata frozen dataclass)
- **Repr helpers:** `state_to_dict()`, `state_to_summary()`, `state_repr()` via `_obj_summ_map`
- **Type stubs:** `.pyi` files auto-generated by `scripts/generate_op_stubs.py` — must stay in sync

### TransformerLens v3

- **TransformerBridge (default, `use_bridge=True`):** Wraps HF models without weight conversion, more memory efficient
- **Legacy HookedTransformer (`use_bridge=False`):** Traditional TL with weight conversion
- `ITLensCustomConfig` with `use_bridge=True` is ignored (TransformerBridge requires HF model, not config-only init)

### Config Hierarchy

- `ITLensConfig` (top-level) → `ITLensSharedConfig` (base with IT settings)
- `ITLensFromPretrainedConfig` (from_pretrained path) | `ITLensCustomConfig` (config-based, HookedTransformer only)
- TL side: `TransformerLensConfig` → `HookedTransformerConfig` | `TransformerBridgeConfig`
- `_capture_hyperparameters()` serializes: hf_preconversion_config, tl_model_cfg, it_tl_cfg

Serialization details matter here:

- `hf_preconversion_config` is the original HF `PretrainedConfig`
- `tl_model_cfg` is the actual TL config from `self.model.cfg`
- `it_tl_cfg` is the Interpretune-specific TL config from `self.it_cfg.tl_cfg`

`ITLensCustomConfig` with `use_bridge=True` is ignored; config-based initialization remains HookedTransformer-only because TransformerBridge requires an HF model instance. See `docs/config_hierarchy_analysis.md` when working on this area.

## CI/CD

- **GitHub Actions** (`ci_test-full.yml`): Ubuntu 22.04, Windows 2022, macOS 14, Python 3.13, 90-min timeout
- **Azure GPU pipeline** (`.azure-pipelines/gpu-tests.yml`): Self-hosted, requires admin approval, only for ready-for-review PRs. Use CPU CI as long as possible before requesting GPU runs.
- The self-hosted approval gate can be driven from the shell when `AZURE_DEVOPS_EXT_PAT` is present. Prefer PAT-backed Azure DevOps CLI or REST calls over manual UI approval when you need to release a queued GPU run during active debugging.
- A queued build can stay in `notStarted` until its approval is granted. Check the build first, then inspect pending approvals before changing runner configuration:
  ```bash
  az pipelines build show --id <build_id> --organization https://dev.azure.com/speediedan --project interpretune -o table
  curl -sS -u ":${AZURE_DEVOPS_EXT_PAT}" \
    "https://dev.azure.com/speediedan/interpretune/_apis/pipelines/approvals?state=pending&api-version=7.1-preview.1"
  ```
- Preferred approval management: the multi-mode helper script in the distributed-insight repo
  (`project_admin/shared_admin_scripts/az_pipeline_agent_scripts/manage-approvals.sh`), auth via
  `ADO_MCP_AUTH_TOKEN` or `AZURE_DEVOPS_EXT_PAT`:
  ```bash
  ./manage-approvals.sh -o speediedan -p interpretune -m list
  ./manage-approvals.sh -o speediedan -p interpretune -m approve -i "<approval_id>" -c "Approved via CLI for GPU validation."
  ./manage-approvals.sh -o speediedan -p interpretune -m reject-all   # dispose stale pending gates
  ```
  (Rejecting a gate completes the build as `failed` — the normal terminal state for a rejected approval.)
- Fallback: approve a pending run with a PATCH to the approvals endpoint:
  ```bash
  curl -sS -X PATCH -u ":${AZURE_DEVOPS_EXT_PAT}" \
    -H "Content-Type: application/json" \
    -d '[{"approvalId":"<approval_id>","status":"approved","comment":"Approved via CLI for GPU validation."}]' \
    "https://dev.azure.com/speediedan/interpretune/_apis/pipelines/approvals?api-version=7.1-preview.1"
  ```
- Agent-stack recovery: infrastructure failures on the self-hosted runner (e.g. the pipeline dying at
  "Initialize containers" with `stat -c %g /var/run/docker.sock` errors after a host reboot) are fixed
  by restarting the rootless-docker + agent stack; this exact command is authorized for agents via a
  NOPASSWD sudoers entry:
  ```bash
  sudo /opt/az_pipeline_agent/restart-stack.sh
  ```
- The build-level queue shown by `az pipelines build show` may still display `Azure Pipelines` even when the YAML job uses the self-hosted `Default` pool. Treat approval state and actual worker dispatch as the source of truth before editing the pool stanza.
- The current GPU test flow is intentionally phase-split to reduce peak memory while preserving CUDA coverage:
  1. `Testing: standard` runs CPU-only with `CUDA_VISIBLE_DEVICES=''`
  2. `Testing: standard gpu cuda-marked` runs regular CUDA-gated tests under `IT_RUN_CUDA_TESTS=1`
  3. `Testing: standalone gpu` runs standalone GPU tests
  4. `Testing: CI Profiling` runs `profile_ci` GPU tests
- **Coverage target:** 90% on commits, 50% on patches (`.codecov.yml`)
- **Torch prerelease:** Configured via `requirements/ci/torch-pre.txt` (version, CUDA target, channel)
- **Dependencies:** Locked in `requirements/ci/requirements.txt`; regenerate with `./requirements/utils/lock_ci_requirements.sh`

## Pre-MVP Backwards Compatibility

Interpretune is **pre-MVP**. Internal op signatures, batch protocols, and pipeline composition may change without deprecation notices. Do not add backwards-compatibility shims (fallback code paths, silent coercions, etc.) to preserve caller assumptions that predate the current design. If an op's contract changes, update all in-tree callers and tests to match the new contract directly.

## Commit & PR Requirements

- Before each commit, run the basic local test command (`python -m pytest src/interpretune tests -v`) plus pre-commit. If the session is still blocked after reasonable debugging effort, an intermediate checkpoint commit is acceptable as long as the current blocker is documented.
- All tests passing (`python -m pytest src/interpretune tests -v`)
- All pre-commit hooks passing
- CPU coverage >= existing coverage
- Docstrings for all public functions/classes
- `tests/benchmarks/benchmark_registry.yaml` updates should be committed in isolation.
- For benchmark refreshes, commit tooling/docs changes first, then run `python tests/benchmarks/run_benchmarks.py --all --update-registry` from a clean tree so the registry diff lands alone. Use `--force-update-registry` only when you intentionally need to bypass the clean-tree check.
- `salient_pkg_versions` should retain best-effort git provenance for salient dependencies: editable installs should include live checkout metadata (`fork`, `branch`, `sha`), while git-backed non-editable installs should derive the source fork and pinned commit from `direct_url.json`.
- If benchmark tooling, the environment collector, or benchmark docs change, commit that work first. Only then run `python tests/benchmarks/run_benchmarks.py --all --update-registry` from a clean tree and commit the resulting registry-only diff.
- Unit tests for new functionality
- **All non-skipped CI workflows must be green before merging to `main`** — that means the FULL surface,
  not just the locally-run pytest phases: the hosted GitHub workflows (Test full across all three OSes,
  Stale Stubs and Type Checks — BOTH halves: pyright AND `generate_op_stubs.py` freshness — PyPI dry-run,
  regen-ci-req report, benchmark-registry isolation) AND every phase of the gated Azure GPU pipeline
  (standard, **standard gpu cuda-marked**, standalone, profile_ci). Lesson from the Session-31 CT merge
  (2026-07-20): a branch was greened locally on three phases but the cuda-marked phase (whose tests hide
  among the locally-skipped set) plus the hosted stubs/pyright/fixture-scope surfaces were never
  exercised, so `main` went red on merge. Run the cuda-marked phase locally via
  `IT_RUN_CUDA_TESTS=1 python -m pytest tests -v` (inline env var; these tests SKIP silently without it —
  they hide inside the "skipped" count) with `HF_GATED_PUBLIC_REPO_AUTH_KEY`/`HF_TOKEN` available, and
  check stub freshness with `python scripts/generate_op_stubs.py` + a clean `git diff` before any merge.
- **No AI-attribution trailers** (`Co-Authored-By: Claude ...`, "Generated with ..." lines) in commit
  messages or PR bodies — they are commit noise (and `.claude/settings.json` no longer adds them). When
  rebasing the long-lived multi-repo workstream branches (interpretune, SAEDashboard, SAELens,
  circuit-tracer, neuronpedia) ahead of opening PRs, strip these trailers from prior commits as part of
  the rebase; a full-history sweep outside those rebases is not required.

## Multi-Session Workstream Cadence (plan + completion log)

Long-running multi-session workstreams (e.g. the Phase 7 PR-packaging plan in the maintainer's private
notes) follow this documentation cadence:

- The working plan's status section stays LEAN: a short current-status paragraph, a compact per-session
  index table (session / focus / landed SHAs / CI builds), and the proposed next-session order.
- Every session appends full detail to the companion completion log: a Part 1 detailed slice entry plus
  a Part 2 end-of-session status snapshot (ending with the pushed multi-repo heads).
- During a session it is fine to write a dated addendum into the plan's status section, but addenda are
  periodically CONSOLIDATED into the index (after verifying the log carries all their detail); superseded
  next-session orders are dropped rather than accumulated — Part 2 snapshots preserve that history.

## Worktrees & Parallel Environments

- A long-lived `~/repos/it-release` worktree with a matching `it_release` venv (see
  `scripts/build_it_env.sh`, `scripts/gen_it_coverage.sh`) exists for future releases and doubles as a
  ready second worktree/env pair for parallel deep debugging. Prefer reusing it when it meets
  requirements (avoids setup cost and detached-worktree proliferation); create a fresh detached worktree
  when isolation from its state matters.
- Overlay-venv recipe for evaluating another branch against an existing heavy env: `python -m venv` a new
  env, `pip install -e <worktree> --no-deps` into it, then bridge the base env with an executable `.pth`
  (`import site; site.addsitedir('<base site-packages>')` — a plain-directory `.pth` will NOT propagate
  the base env's editable installs). Put the overlay venv's `bin` on `PATH` for tests that spawn console
  scripts (e.g. `test_it_cli.py`).

## Notebook Publishing

**⚠️ CRITICAL: Notebook tests run against PUBLISHED notebooks, not dev notebooks.** If you edit any dev notebook, you **MUST** run `python scripts/publish_notebooks.py --force` before testing or committing. The pre-commit hook only triggers when `.ipynb` files in `dev/` are staged — if the dev notebook was changed in a prior commit or by another tool, the published copy will be stale and notebook tests will fail.

Dev notebooks live in `src/it_examples/notebooks/dev/` and are auto-published to `src/it_examples/notebooks/publish/` via `scripts/publish_notebooks.py`.

- **Pre-commit hook** (`publish-notebooks`): Triggered on changes to `^src/it_examples/notebooks/dev/.*\.ipynb$`. Automatically strips `remove-cell` tagged cells, adds Colab badges + install cells, fixes import paths, and tracks hashes in `.notebook_hashes.json`.
- **CLI flags:** `--dry-run` (preview), `--check-only` (CI validation), `--force` (republish all). Launch.json has debug configs for each mode.
- **Workflow:** Edit dev notebooks only → commit → pre-commit publishes automatically. Never edit publish notebooks directly.
- **Verification:** After any notebook changes, run `python scripts/publish_notebooks.py --check-only` to verify published notebooks are in sync.

## Elevated-Access Blockers

When work is blocked by a command that requires elevated access or interactive local auth (sudo, an
interactive login, a gpg/pinentry prompt, a locked credential store), do not just work around it silently:
include in the response summary a concrete proposed solution that would let Claude execute that command (or a
scripted group of commands) autonomously in the future. The established pattern is a root-owned wrapper
script + sudoers `NOPASSWD` directive + `~/.claude/settings.json` `permissions.allow` entry — e.g.
`Bash(sudo /opt/az_pipeline_agent/restart-stack.sh)` paired with
`speediedan ALL=(ALL) NOPASSWD: /opt/az_pipeline_agent/restart-stack.sh`. Propose that pattern (or a
better-fitting design) so it can be implemented and validated in a subsequent session.

## Important Caveats

- **Git dependencies** (transformer_lens, sae_lens, circuit_tracer, nnsight) are pinned to specific commits in `pyproject.toml`
- **Type checking:** Enabled for all `src/` files except `src/it_examples/utils/raw_graph_analysis.py`; all `tests/` files are excluded
- **Full test suite requires ML dependencies** — tests will fail without proper env setup
- **Import guards:** `tests/core/test_adapters_import_time.py` prevents accidental eager imports
- **No test-environment bandaids in application code:** When test failures stem from environment issues (e.g., `isinstance()` failures due to importlib double-loading modules), fix the problem in the test infrastructure or use an app-level registry pattern (like `CT_BACKEND_REGISTRY` in `circuit_tracer.py`) — never degrade application code with workarounds for test-specific problems

## Detailed Instruction Files

The following files in `.github/instructions/` provide in-depth guidance for specific development tasks. Consult them when working in the relevant areas:

| File | Applies to | When to use |
|------|-----------|-------------|
| `analysis_injection.instructions.md` | `src/it_examples/**` | Modifying runtime hook injection for upstream package analysis (config YAML, analysis points, patcher logic) |
| `developing_adapters.instructions.md` | `src/interpretune/adapters/**` | Creating or modifying adapters — covers protocol enum, config, MRO composition order, registration checklist |
| `fixture_usage.instructions.md` | `tests/**` | Writing or debugging test fixtures — covers fixture types/phases, MODULE_EXAMPLE_REGISTRY dependency, parameterization patterns |
| `registering_example_modules.instructions.md` | `example_module_registry.yaml` | Adding adapter combinations to the central registry — registry key patterns, config structure, verification steps |
| `updating_expected_results.instructions.md` | `tests/parity_acceptance/**` | Updating expected test baselines — state logging, memory footprint YAML, profiling update script |
