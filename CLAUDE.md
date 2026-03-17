# CLAUDE.md - Interpretune

## Project Overview

Interpretune is a flexible framework for collaborative AI world model analysis and tuning. **Pre-MVP stage** — features and APIs are subject to change.

**Stack:** Python 3.10+ (CI on 3.13), PyTorch 2.7.1+, transformer_lens >= 3.0.0, sae_lens, datasets, jsonargparse

## Build & Dev Environment

Uses `uv` for dependency management. For Venvs, use `/mnt/cache/$USER/.venvs/` (preferred for hardlink perf), fall back to `~/.venvs/` if that path isn't available.

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

# Note: Coverage collection takes approximately 50 minutes
```

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
├── *_parity/               # Research parity tests
└── examples/               # Example-based tests
```

### Key Patterns

- **Lazy imports:** Heavy deps (transformer_lens, lightning, neuronpedia) loaded lazily via `_light_register.py` and PEP 562 `__getattr__`. Import-time optimization is critical.
- **Protocol-based:** ITModuleProtocol, ITDataModuleProtocol, AnalysisOpProtocol
- **Composition:** `CompositionRegistry` manages adapter combinations (core, lightning, transformer_lens, sae_lens, circuit_tracer, nnsight)
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
- Approve a pending run with a PATCH to the approvals endpoint:
  ```bash
  curl -sS -X PATCH -u ":${AZURE_DEVOPS_EXT_PAT}" \
    -H "Content-Type: application/json" \
    -d '[{"approvalId":"<approval_id>","status":"approved","comment":"Approved via CLI for GPU validation."}]' \
    "https://dev.azure.com/speediedan/interpretune/_apis/pipelines/approvals?api-version=7.1-preview.1"
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

## Commit & PR Requirements

- All tests passing (`python -m pytest src/interpretune tests -v`)
- All pre-commit hooks passing
- CPU coverage >= existing coverage
- Docstrings for all public functions/classes
- Unit tests for new functionality

## Notebook Publishing

**⚠️ CRITICAL: Notebook tests run against PUBLISHED notebooks, not dev notebooks.** If you edit any dev notebook, you **MUST** run `python scripts/publish_notebooks.py --force` before testing or committing. The pre-commit hook only triggers when `.ipynb` files in `dev/` are staged — if the dev notebook was changed in a prior commit or by another tool, the published copy will be stale and notebook tests will fail.

Dev notebooks live in `src/it_examples/notebooks/dev/` and are auto-published to `src/it_examples/notebooks/publish/` via `scripts/publish_notebooks.py`.

- **Pre-commit hook** (`publish-notebooks`): Triggered on changes to `^src/it_examples/notebooks/dev/.*\.ipynb$`. Automatically strips `remove-cell` tagged cells, adds Colab badges + install cells, fixes import paths, and tracks hashes in `.notebook_hashes.json`.
- **CLI flags:** `--dry-run` (preview), `--check-only` (CI validation), `--force` (republish all). Launch.json has debug configs for each mode.
- **Workflow:** Edit dev notebooks only → commit → pre-commit publishes automatically. Never edit publish notebooks directly.
- **Verification:** After any notebook changes, run `python scripts/publish_notebooks.py --check-only` to verify published notebooks are in sync.

## Important Caveats

- **Git dependencies** (transformer_lens, sae_lens, circuit_tracer, nnsight) are pinned to specific commits in `pyproject.toml`
- **Type checking:** Enabled for all `src/` files except `src/it_examples/utils/raw_graph_analysis.py`; all `tests/` files are excluded
- **Full test suite requires ML dependencies** — tests will fail without proper env setup
- **Import guards:** `tests/core/test_import_time_and_adapters.py` prevents accidental eager imports
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
