# Copilot Instructions for Interpretune

## Repository Overview

**Interpretune** is a flexible, powerful framework for collaborative AI world model analysis and tuning. This project is in **pre-MVP** stage - features and APIs are subject to change.

**Key Technologies:**
- Python 3.10+ (CI tests on 3.13)
- PyTorch 2.7.1+ with transformers ecosystem
- Core deps: transformer_lens >= 3.0.0 (TransformerBridge support), sae_lens, datasets, jsonargparse
- Optional: PyTorch Lightning, W&B, circuit-tracer, neuronpedia

**Repository Size:** ~200 files, primarily Python, with YAML configs and shell scripts

## TransformerLens v3 Integration

**TransformerBridge (v3, default):**
- Wraps HuggingFace models without weight conversion
- More memory efficient (no weight duplication)
- Better HF ecosystem compatibility
- Enabled by default via `use_bridge=True` in tl_cfg

**Legacy HookedTransformer:**
- Traditional TL interface with weight conversion
- Available via `use_bridge=False` in tl_cfg
- Maintained for backward compatibility

**Implementation:**
- `_convert_hf_to_bridge()`: TransformerBridge initialization
- `_convert_hf_to_tl()`: Legacy HookedTransformer initialization
- Config-based initialization always uses HookedTransformer (TransformerBridge requires HF model)

### Configuration Hierarchy

**TransformerLens Configs:**
- `TransformerLensConfig`: Base class (d_model, n_layers, etc.)
- `HookedTransformerConfig`: Legacy config extending base (dataclass)
- `TransformerBridgeConfig`: V3 config extending base with architecture field

**Interpretune Configs:**
- `ITLensSharedConfig`: Base with shared and IT-specific settings (`move_to_device`, `use_bridge`)
- `ITLensFromPretrainedConfig`: For from_pretrained initialization (fold_ln, model_name, etc.)
- `ITLensCustomConfig`: For config-based initialization (requires HookedTransformerConfig or one constructed from a dict)
- `ITLensConfig`: Top-level IT config encapsulating all settings

**Config Serialization:**
Three types of configs are serialized by `_capture_hyperparameters()`:
1. `hf_preconversion_config`: Original HF PretrainedConfig (via superclass)
2. `tl_model_cfg`: Actual TL config from `self.model.cfg` (HookedTransformerConfig or TransformerBridgeConfig)
3. `it_tl_cfg`: IT-specific settings from `self.it_cfg.tl_cfg` (ITLensFromPretrainedConfig or ITLensCustomConfig)

**Important Limitations:**
- `ITLensCustomConfig` with `use_bridge=True` will be ignored; IT will warn and force `use_bridge=False`.
- TransformerBridge requires HF model, cannot be initialized from config alone
- Config-based path (`ITLensCustomConfig`) only supports HookedTransformer

See `docs/config_hierarchy_analysis.md` for detailed configuration relationship analysis.

## Code Standards

### Required Before Each Commit
- Unless guidance in a comment or a pull request or target issue description states otherwise, always run our basic tests (which mirror the `ci_test-full.yml` workflow) in your local environment and ensure all tests are passing before committing, for example:
```bash
cd /home/runner/work/interpretune/interpretune && python -m pytest src/interpretune tests -v
```
- Ensure all pre-commit hooks pass.
- If the copilot session is still failing despite trying to get tests and pre-commit hooks passing for some time, it's okay to commit your intermediate work with a comment about the present challenge to be dealt with in a subsequent session.

### Requirement for Each Pull Request
- All pull requests must pass the CI checks.
- Ensure that the code is well-documented, with docstrings for all public functions and classes.
- Write unit tests for new functionality and ensure existing tests pass.
- Ensure the cpu coverage reported by our `ci_test-full.yml` workflow is >= the existing coverage.

## Build and Validation Commands

### Environment Setup
Development environment uses `uv` for fast, reliable dependency management:

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create development environment (creates traditional venv at ~/.venvs/it_latest)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest

# Activate the environment
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate

# Run commands directly (no need for 'uv run')
python --version
python -m pytest tests/
```

### Development Environment Scripts
Use the provided build script for automated setup:

```bash
# Standard development build (recommended for dev work)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest

# Quick editable install (without locked requirements)
uv pip install -e ".[test,examples,lightning,profiling]" --group git-deps dev

# Venv Location Options (for hardlink performance and standalone process wrappers):
#
# OPTION 1 (Recommended for standalone process wrappers): Use --venv-dir to set BASE directory
# The venv will be created at: <venv-dir>/<target-env-name>
# This is most robust when using with manage_standalone_processes.sh:
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --venv-dir=/mnt/cache/username/.venvs
# Creates venv at: /mnt/cache/username/.venvs/it_latest
#
# OPTION 2: Use IT_VENV_BASE environment variable to set base directory
# This approach uses IT_VENV_BASE as base + target_env_name:
export IT_VENV_BASE=/mnt/cache/username/.venvs
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest
# Creates venv at: /mnt/cache/username/.venvs/it_latest
#
# OPTION 3: Use default (~/.venvs/<target_env_name>) - simplest but may cause hardlink warnings
# If UV cache is on different filesystem, you'll see "Failed to hardlink files" warnings
# Creates venv at: ~/.venvs/it_latest
#
# Why placement matters: UV uses hardlinks for fast installs, but hardlinks only work within
# the same filesystem. Placing venv on same filesystem as UV cache ensures fast installs and
# no warnings. Example UV cache location: /mnt/cache/username/.cache/uv

# Build with CPU-only torch (for CI environments)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --torch-backend=cpu

# Torch prerelease builds (configured via requirements/ci/torch-pre.txt):
# If torch-pre.txt exists, the build script will automatically use the prerelease version if not all of the config
# lines are commented out
# torch-pre.txt format (3 lines): version, CUDA target, channel (nightly or test)
# Example torch-pre.txt: (no # in front of lines, these lines below)
2.10.0.dev20250122
cu128
nightly

# Build with single package from source (no extras)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"

# Build with package from source with extras
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all"

# Build with package from source with extras and environment variable
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1"

# Build with multiple packages from source (using multiple --from-source flags - cleaner!)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest \
  --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1" \
  --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"

# Build with multiple packages from source (using semicolon separator - also supported)
# it_latest with sae-lens, nnsight, and circuit-tracer from source using multiple --from-source flags
# and multiple env variables in some cases and support for specific uv build flags via the FLAGS keyword
./scripts/build_it_env.sh --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest --from-source="sae_lens:${HOME}/repos/SAELens:all:UV_OVERRIDE=${HOME}/repos/interpretune/requirements/ci/overrides.txt:FLAGS=-r ~/repos/distributed-insight/project_admin/interpretune/adapter_reference/sae_lens/admin_scripts/sl_poetry_requirements.txt" --from-source="nnsight:${HOME}/repos/nnsight:all:UV_OVERRIDE=${HOME}/repos/interpretune/requirements/ci/overrides.txt" --from-source="circuit_tracer:${HOME}/repos/circuit-tracer:dev:UV_EXCLUDE=${HOME}/repos/interpretune/requirements/ci/excludes.txt:UV_OVERRIDE=${HOME}/repos/interpretune/requirements/ci/overrides.txt"

# Important: When using with manage_standalone_processes.sh wrapper, use --venv-dir:
~/repos/interpretune/scripts/manage_standalone_processes.sh --use-nohup scripts/build_it_env.sh \
  --repo-home=${HOME}/repos/interpretune --target-env-name=it_latest \
  --venv-dir=/mnt/cache/speediedan/.venvs/it_latest \
  --from-source="finetuning_scheduler:~/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1"
```

**Important: Git Dependency Caching and override-dependencies**

When installing from-source packages that specify git dependencies (e.g., finetuning-scheduler with USE_CI_COMMIT_PIN=1 pinning Lightning to a specific commit), UV's caching ensures correct behavior:

1. From-source packages are installed FIRST with all their dependencies
2. UV caches git dependencies by their fully-resolved commit hash
3. When interpretune is subsequently installed, UV respects the cached commit-pinned versions
4. The [tool.uv] override-dependencies in pyproject.toml replaces interpretune's git URL dependencies with version constraints, allowing editable installations to satisfy requirements

See [UV's dependency caching docs](https://docs.astral.sh/uv/concepts/cache/#dependency-caching) for details on git dependency caching behavior.

### Linting and Code Quality
**Always run linting before committing (assumes activated venv):**

```bash
# Activate your environment first
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate

# Run ruff linting (configured in pyproject.toml)
# we don't have ruff installed as a separate package but use it via pre-commit (with the --fix flag)
# there are two phases, the check and format, run each separately
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files

# Run pre-commit hooks (includes ruff, docformatter, yaml checks)
pre-commit run --all-files
```

**Expected Ruff Issues:** The `tests/*_parity/` directories contain imported research code with many linting violations - these are intentionally excluded from pre-commit checks.

### Testing
**Test command (assumes activated venv):**
```bash
# Activate your environment first
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate

# Basic test run (requires full dependencies)
cd /home/runner/work/interpretune/interpretune && python -m pytest src/interpretune tests -v

# With coverage (using pytest-cov ensures coverage starts before test collection imports)
python -m pytest --cov=src/interpretune --cov-append --cov-report= src/interpretune tests -v
python -m coverage report

# Test collection only (to check test discovery)
pytest --collect-only
```

**Test Reruns for Transient Failures:**

All CI and local coverage scripts include `--reruns 2 --reruns-delay 5` by default to handle transient `httpx` read timeouts with HuggingFace `transformers` v5. The `pytest-rerunfailures` plugin is included in our test dependencies.

Local scripts (`gen_it_coverage.sh`, `special_tests.sh`, `analyze_test_coverage.py`) support configurable rerun behavior via `--no-reruns`, `--reruns=N`, and `--reruns-delay=N` flags. Use `--allow-failures` with `gen_it_coverage.sh` and `special_tests.sh` to continue collecting coverage past test failures — preferred for faster debugging iterations.

**⚠️ Dependency Note:** Full test suite requires ML dependencies. Tests will fail without proper environment setup.

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

# Preferred for debugging: use --allow-failures to continue past failures
${IT_REPO_DIR}/scripts/manage_standalone_processes.sh --use-nohup \
  ${IT_REPO_DIR}/scripts/gen_it_coverage.sh \
  --repo-home=${IT_REPO_DIR} \
  --target-env-name=${IT_TARGET_VENV} \
  --venv-dir=${IT_VENV_BASE} \
  --no-rebuild-base \
  --allow-failures

# Monitor coverage progress
tail -f $(ls -rt /tmp/gen_it_coverage_it_* | tail -1)

# Note: Coverage collection takes approximately 50 minutes
```

**Test timing:** Most tests run quickly (<30s), but some integration tests may take 1-2 minutes.

**Test structure:** Tests are in `tests/` with special subdirectories:
- `tests/core/` - core functionality tests
- `tests/*_parity/` - research parity tests (excluded from pre-commit)

## Project Layout and Architecture

### Source Code Structure
```
src/interpretune/           # Main package
├── (version metadata forwarded from top-level `src/__about__.py`)
├── adapters/               # Integration adapters
│   ├── transformer_lens.py
│   ├── sae_lens.py
│   ├── lightning.py
│   └── circuit_tracer.py
├── base/                   # Core base classes
│   └── components/
│       └── cli.py          # CLI entry point
├── analysis/               # Analysis tools
├── config/                 # Configuration system
├── runners/                # Execution runners
└── utils/                  # Utility functions

src/it_examples/            # Example experiments
├── config/                 # YAML configuration files
│   ├── experiments/        # Experiment configs
│   └── ft_schedules/       # Fine-tuning schedules
└── experiments/            # Python experiment modules
```

### Configuration Files
- `pyproject.toml` - Main project config, dependencies, ruff/pytest settings
- `.pre-commit-config.yaml` - Code quality hooks
- `requirements/` - Pinned dependency files
  - `base.txt` - Core dependencies
  - `test.txt`, `examples.txt`, etc. - Optional dependency groups

### Key Entry Points
- Console script: `interpretune` → `interpretune.base.components.cli:bootstrap_cli`

### Notebook Publishing

**⚠️ CRITICAL: Notebook tests run against PUBLISHED notebooks, not dev notebooks.** If you edit any dev notebook, you **MUST** run `python scripts/publish_notebooks.py --force` before testing or committing. The pre-commit hook only triggers when `.ipynb` files in `dev/` are staged — if the dev notebook was changed in a prior commit or by another tool, the published copy will be stale and notebook tests will fail.

Dev notebooks live in `src/it_examples/notebooks/dev/` and are auto-published to `src/it_examples/notebooks/publish/` via `scripts/publish_notebooks.py`.

- **Pre-commit hook** (`publish-notebooks`): Triggered on changes to `^src/it_examples/notebooks/dev/.*\.ipynb$`. Automatically strips `remove-cell` tagged cells, adds Colab badges + install cells, fixes import paths, and tracks hashes in `.notebook_hashes.json`.
- **CLI flags:** `--dry-run` (preview), `--check-only` (CI validation), `--force` (republish all). Launch.json has debug configs for each mode.
- **Workflow:** Edit dev notebooks only → commit → pre-commit publishes automatically. Never edit publish notebooks directly.
- **Verification:** After any notebook changes, run `python scripts/publish_notebooks.py --check-only` to verify published notebooks are in sync.

## CI and Validation Pipeline

### GitHub Actions Workflow
**File:** `.github/workflows/ci_test-full.yml`

**Triggers:** Push/PR to main, changes to source/test files
**Platforms:** Ubuntu 22.04, Windows 2022, macOS 14 (Python 3.13)
**Timeout:** 90 minutes

**CI Process:**
1. Check for torch prerelease configuration (torch-pre.txt)
2. Install torch prerelease if configured (otherwise uses stable with --torch-backend=cpu)
3. Install interpretune in editable mode with git dependencies
4. Install locked CI requirements (all PyPI packages)
5. Run pytest with coverage
6. Resource monitoring (Linux only)
7. Upload artifacts on failure

**CI Installation Flow:**
```bash
# Step 1: (If torch-pre.txt exists) Install torch prerelease
uv pip install --prerelease=if-necessary-or-explicit "torch==${TORCH_PRE_VERSION}+cpu" --index-url "https://download.pytorch.org/whl/${TORCH_PRE_CHANNEL}/cpu"

# Step 2: Install interpretune editable + git dependencies
uv pip install -e . --group git-deps

# Step 3: Install all locked PyPI dependencies (with --torch-backend=cpu for stable torch)
uv pip install -r requirements/ci/requirements.txt --torch-backend=cpu
```

**Development Installation Flow (build_it_env.sh):**
```bash
# Step 1: Install torch (prerelease from torch-pre.txt or stable via --torch-backend)
# Prerelease: uv pip install --prerelease=if-necessary-or-explicit "torch==${VERSION}" --index-url "https://download.pytorch.org/whl/${CHANNEL}/${CUDA}"
# Stable: uv pip install torch --torch-backend=auto

# Step 2: Install interpretune editable + git dependencies
uv pip install -e . --group git-deps

# Step 3: Install locked CI requirements
uv pip install -r requirements/ci/requirements.txt

# Step 4: Install from-source packages (if specified)
# These override any PyPI/git versions for development
```

**Torch Prerelease Configuration (torch-pre.txt):**
Create `requirements/ci/torch-pre.txt` with 3 lines (no comments inline):
```
2.10.0.dev20250122
cu128
nightly
```
- Line 1: Torch version (e.g., `2.10.0.dev20250122`)
- Line 2: CUDA target (e.g., `cu128`, `cpu`)
- Line 3: Channel (`nightly` or `test`)

**Environment Variables for CI:**
- `IT_CI_LOG_LEVEL` - Defaults to "INFO", set to "DEBUG" for verbose logging
- `CI_RESOURCE_MONITOR` - Set to "1" to enable resource logging
- `IT_USE_CT_COMMIT_PIN` - Controls circuit-tracer installation method

### Azure self-hosted GPU pipeline (new)

We now have a separate Azure DevOps pipeline that runs GPU/standalone tests on a self-hosted runner: `.azure-pipelines/gpu-tests.yml`.
- This pipeline is intentionally restrictive: it only triggers for PRs that are marked "ready for review" and must be explicitly approved by a repository administrator before the self-hosted GPU job will run (currently: speediedan).
- Because self-hosted GPU capacity is limited, aim to rely on feedback from the normal GitHub Actions CPU CI workflow for as long as possible while iterating on an issue. Defer switching the PR to "ready for review" until you believe GPU testing is necessary. Copilot should prefer this conservative approach when suggesting CI runs or opening PRs.

Note: the GPU pipeline runs only when a PR is ready for review and an admin approves the run — do not expect it to run automatically for draft PRs or early-stage work.

### Manual Validation Steps
Set environment context variables (developer-specific paths):

```bash
export IT_VENV_BASE=/mnt/cache/${USER}/.venvs
export IT_TARGET_VENV=it_latest
export IT_REPO_DIR=${HOME}/repos/interpretune  # Example: adjust to your local repo path

cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate

# Run ruff linting (configured in pyproject.toml)
# we don't have ruff installed as a separate package but use it via pre-commit (with the --fix flag)
# there are two phases, the check and format, run each separately
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files

# Run pre-commit hooks (includes ruff, docformatter, yaml checks)
pre-commit run --all-files
```

### Coverage Collection

Use the `gen_it_coverage.sh` script to collect comprehensive test coverage locally. This script runs all tests including standalone and special tests, then generates a coverage report.

**Handling conflicting processes:**

The `manage_standalone_processes.sh` harness checks for conflicting pytest processes before starting. If you encounter a conflict:
- If the process is hung or old (>40 minutes), kill it: `pkill -f "pytest.*--collect-only"`
- If recently started (<40 minutes), wait a few minutes for it to complete naturally

**Monitoring progress:**

```bash
# Tail the most recent coverage log
tail -f `ls -rt /tmp/gen_it_coverage_it_* | tail -1`
```

**Common coverage commands:**

```bash
# Generate coverage with no rebuild (recommended for quick iteration)
~/repos/interpretune/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/interpretune/scripts/gen_it_coverage.sh \
  --repo-home=${HOME}/repos/interpretune \
  --target-env-name=it_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --no-rebuild-base

# Preferred for debugging: use --allow-failures to continue past failures and see all results
~/repos/interpretune/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/interpretune/scripts/gen_it_coverage.sh \
  --repo-home=${HOME}/repos/interpretune \
  --target-env-name=it_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --no-rebuild-base \
  --allow-failures

# Generate coverage with rebuild (use when dependencies changed)
~/repos/interpretune/scripts/manage_standalone_processes.sh --use-nohup \
  ~/repos/interpretune/scripts/gen_it_coverage.sh \
  --repo-home=${HOME}/repos/interpretune \
  --target-env-name=it_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs

# Note: Coverage collection takes approximately 40 minutes
```

**Flags:**

- `--no-rebuild-base`: Skips environment rebuild (faster, use when dependencies haven't changed)
- `--allow-failures`: Continue collecting coverage even if tests fail; preferred for debugging to see all failures at once
- `--venv-dir`: Base directory for venvs (recommended: `/mnt/cache/${USER}/.venvs` for hardlink performance)
- `--run-all-and-examples`: Include additional example tests (extends runtime)
- `--torch-backend=cpu`: Force CPU-only torch (useful for testing without GPU)

**Output:**

- Coverage report written to `/tmp/current_interpretune_coverage.out`
- Detailed logs in `/tmp/gen_it_coverage_it_latest_<timestamp>.log`
- HTML coverage report in `htmlcov/` directory

### Updating dependencies

When updating dependencies, edit `pyproject.toml` and regenerate locked requirements:

```bash
# Edit pyproject.toml to update version constraints

# Regenerate locked CI requirements
./requirements/utils/lock_ci_requirements.sh

# Rebuild your development environment
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest

# Or update manually in an activated environment
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate
uv pip install --upgrade <package-name>

# After updating, test thoroughly
python -m pytest tests/ -v
```

Notes:
- Dependencies are specified in `pyproject.toml` with optional extras and dependency groups
- CI uses locked requirements (requirements/ci/requirements.txt) for reproducibility
- Development can use either locked requirements (via build script) or direct installation
- Always run the full CI after dependency changes to validate compatibility across platforms

### Type-checking caveat

We currently only exclude one file from type checking in the /src tree ("src/it_examples/utils/raw_graph_analysis.py") while all files in /tests are excluded.

## Special Dependencies and Known Issues

### Circuit-Tracer Dependency
**Note:** circuit-tracer is installed directly from git as specified in `pyproject.toml`. The commit is pinned in the examples optional dependencies: `circuit-tracer @ git+https://github.com/speediedan/circuit-tracer.git@004f1b28...`. When you run `uv pip install -e ".[examples]"`, uv resolves and installs this git dependency automatically.

### Dependency Constraints
- **torch** requires 2.7.1+ for newer features
- **setuptools** requires 77.0.0+ for PEP 639 support (used in build system)

### Import Dependencies
- `transformer_lens` and `sae_lens` have complex initialization requirements
- Some imports are conditional based on available packages
- Circuit-tracer imports are wrapped in try/catch blocks

## Development Guidelines

### Code Style
- **Line length:** 120 characters (configured in ruff)
- **Import style:** Sort within sections, first-party packages listed
- **Type hints:** Use modern syntax with `from __future__ import annotations`
- **Docstrings:** Use docformatter-compatible format

### Testing Guidelines
- Test files mirror `src/` structure in `tests/`
- Use pytest fixtures from `conftest.py`
- Add coverage for new functionality

#### Running Special Tests (Standalone and Profiling)

Some tests require special environment setup and are marked with `@pytest.mark.standalone` or involve profiling. These tests can be run using the `special_tests.sh` harness or manually with environment variables.

Set environment context variables (developer-specific paths):

```bash
export IT_VENV_BASE=/mnt/cache/${USER}/.venvs
export IT_TARGET_VENV=it_latest
export IT_REPO_DIR=${HOME}/repos/interpretune  # Example: adjust to your local repo path
```

**Using the test harness:**
```bash
export IT_REPO_DIR=${HOME}/repos/interpretune  # Example: adjust to your local repo path
# Run all standalone tests
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
 ./tests/special_tests.sh --mark_type=standalone

# Run specific standalone test by filter pattern
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
./tests/special_tests.sh --mark_type=standalone --filter_pattern='test_attribution_analysis_notebook[analysis_inj_salient_logits_SLT]'
# another example, filtering by partial test class name
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
./tests/special_tests.sh --mark_type=standalone --filter_pattern='ParameterMapping'
# Run profiling tests
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
./tests/special_tests.sh --mark_type=profiling
```

**Manual execution (without harness):**

Set environment context variables (developer-specific paths):

```bash
export IT_VENV_BASE=/mnt/cache/${USER}/.venvs
export IT_TARGET_VENV=it_latest
export IT_REPO_DIR=${HOME}/repos/interpretune  # Example: adjust to your local repo path
```

Then run specific tests using **inline environment variables** (not export) to avoid marker conflicts:

```bash
# Run specific standalone test
# IMPORTANT: Use inline variable assignment (VAR=value command), not export
# This prevents marker conflicts when multiple test environment variables are set
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
IT_RUN_STANDALONE_TESTS=1 python -m pytest tests/examples/test_notebooks.py::test_attribution_analysis_notebook[analysis_inj_salient_logits_SLT] -v
unset IT_RUN_STANDALONE_TESTS

# another example using inline variable assignment with multiple standalone tests invoked separately
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
IT_RUN_STANDALONE_TESTS=1 python -m pytest tests/core/test_transformer_lens.py::TestGemma2ParameterMapping::test_gemma2_tl_param_structure -v
IT_RUN_STANDALONE_TESTS=1 python -m pytest tests/core/test_transformer_lens.py::TestLlama3ParameterMapping::test_llama3_tl_param_structure -v
unset IT_RUN_STANDALONE_TESTS

# Run specific profiling ci test
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
IT_RUN_PROFILING_TESTS=1 python -m pytest tests/parity_acceptance/test_it_l.py::test_l_profiling[test_cuda_32_l] -v
unset IT_RUN_PROFILING_TESTS

# Run specific profiling (non-ci, ones that don't run by default with ci) test
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
IT_RUN_PROFILING_TESTS=2 python -m pytest tests/parity_acceptance/test_it_tl.py::test_tl_profiling[test_cuda_32] -v
unset IT_RUN_PROFILING_TESTS

# Run specific optional tests
export IT_RUN_OPTIONAL_TESTS=1 && \
cd ${IT_REPO_DIR} && \
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate && \
python -m pytest tests/parity_acceptance/test_it_fts.py::test_parity_fts[train_cuda_32_l_fts] -v  || true && \
unset IT_RUN_OPTIONAL_TESTS
```

**Important Notes:**
- **Always use inline environment variables** (`VAR=value command`) instead of `export` for test markers
- Using `export` can cause marker filtering conflicts when multiple test environment variables are set
- Environment variables like `IT_VENV_BASE`, `IT_TARGET_VENV`, and `IT_REPO_DIR` are developer-specific
- Adjust these paths to match your local development environment
- The `special_tests.sh` harness handles environment setup automatically
- Standalone tests may take longer and require more resources than regular tests

### Configuration
- YAML configs in `src/it_examples/config/`
- Use dataclasses for configuration objects
- Support jsonargparse CLI integration

**Trust these instructions** - only search for additional information if these instructions are incomplete or incorrect for your specific task. The repository structure and build process are complex, but following these guidelines will minimize exploration and failed commands.
