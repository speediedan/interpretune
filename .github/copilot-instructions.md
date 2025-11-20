# Copilot Instructions for Interpretune

## Repository Overview

**Interpretune** is a flexible, powerful framework for collaborative LLM world model analysis and tuning. This project is in **pre-MVP** stage - features and APIs are subject to change.

**Key Technologies:**
- Python 3.10+ (CI tests on 3.12)
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
source ~/.venvs/it_latest/bin/activate

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

# Build with specific PyTorch nightly version
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --torch_dev_ver=dev20240201

# Build with PyTorch test channel
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --torch_test_channel

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
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all:USE_CI_COMMIT_PIN=1;circuit_tracer:${HOME}/repos/circuit-tracer"

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

**From-Source Package Version Requirements:**

When installing packages from source (especially transformer-lens), ensure the package version in the source repo satisfies dependent package requirements:
- circuit-tracer requires `transformer-lens>=v2.16.0`
- TransformerLens repo default version (with the old v2 poetry install) is 0.0.0 in pyproject.toml (set by CI pipeline on release)
- For local development, update TransformerLens version to 2.16.1 or higher: `sed -i 's/version="0\.0\.0"/version="2.16.1"/' ~/repos/TransformerLens/pyproject.toml`
- This ensures circuit-tracer's dependency is satisfied without UV upgrading transformer-lens to PyPI version
- This should not be necessary when installing transformer_lens from source with versions >= 3.0.0 as uv is used

### Linting and Code Quality
**Always run linting before committing (assumes activated venv):**

```bash
# Activate your environment first
source ~/.venvs/it_latest/bin/activate

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
source ~/.venvs/it_latest/bin/activate

# Basic test run (requires full dependencies)
cd /home/runner/work/interpretune/interpretune && python -m pytest src/interpretune tests -v

# With coverage
python -m coverage run --append --source src/interpretune -m pytest src/interpretune tests -v
python -m coverage report

# Test collection only (to check test discovery)
pytest --collect-only
```

**⚠️ Dependency Note:** Full test suite requires ML dependencies. Tests will fail without proper environment setup.

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

## CI and Validation Pipeline

### GitHub Actions Workflow
**File:** `.github/workflows/ci_test-full.yml`

**Triggers:** Push/PR to main, changes to source/test files
**Platforms:** Ubuntu 22.04, Windows 2022, macOS 14 (Python 3.12)
**Timeout:** 90 minutes

**CI Process:**
1. Install interpretune in editable mode with git dependencies
2. Install locked CI requirements (all PyPI packages)
3. Run pytest with coverage
4. Resource monitoring (Linux only)
5. Upload artifacts on failure

**CI Installation Flow:**
```bash
# Step 1: Install interpretune editable + git dependencies
uv pip install -e . --group git-deps

# Step 2: Install all locked PyPI dependencies
uv pip install -r requirements/ci/requirements.txt
```

**Development Installation Flow (build_it_env.sh):**
```bash
# Step 1: Install interpretune editable + git dependencies
uv pip install -e . --group git-deps

# Step 2: Install locked CI requirements
uv pip install -r requirements/ci/requirements.txt

# Step 3: Install from-source packages (if specified)
# These override any PyPI/git versions for development
```

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

```bash
# Activate your environment first
source ~/.venvs/it_latest/bin/activate

# Run ruff linting (configured in pyproject.toml)
# we don't have ruff installed as a separate package but use it via pre-commit (with the --fix flag)
# there are two phases, the check and format, run each separately
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files

# Run pre-commit hooks (includes ruff, docformatter, yaml checks)
pre-commit run --all-files
```

### Updating dependencies

When updating dependencies, edit `pyproject.toml` and regenerate locked requirements:

```bash
# Edit pyproject.toml to update version constraints

# Regenerate locked CI requirements
./requirements/utils/lock_ci_requirements.sh

# Rebuild your development environment
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest

# Or update manually in an activated environment
source ~/.venvs/it_latest/bin/activate
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

Full repository type-checking is a work in progress. Current local checks may only include a subset of files (for example, `src/interpretune/adapters/lightning.py`). Expect that type-checking will cover most files in future updates; don't assume exhaustive static type guarantees yet.

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
- Avoid modifying `*_parity/` test directories (research code)

### Configuration
- YAML configs in `src/it_examples/config/`
- Use dataclasses for configuration objects
- Support jsonargparse CLI integration

**Trust these instructions** - only search for additional information if these instructions are incomplete or incorrect for your specific task. The repository structure and build process are complex, but following these guidelines will minimize exploration and failed commands.
