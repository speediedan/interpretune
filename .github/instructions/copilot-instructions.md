# Copilot Instructions for Interpretune

## Repository Overview

**Interpretune** is a flexible, powerful framework for collaborative LLM world model analysis and tuning. This project is in **pre-MVP** stage - features and APIs are subject to change.

**Key Technologies:**
- Python 3.10+ (CI tests on 3.12)
- PyTorch 2.7.1+ with transformers ecosystem
- Core deps: transformer_lens, sae_lens, datasets, jsonargparse
- Optional: PyTorch Lightning, W&B, circuit-tracer

**Repository Size:** ~200 files, primarily Python, with YAML configs and shell scripts

## Code Standards

### Required Before Each Commit
- Ensure all pre-commit hooks pass.

### Requirement for Each Pull Request
- All pull requests must pass the CI checks.
- Ensure that the code is well-documented, with docstrings for all public functions and classes.
- Write unit tests for new functionality and ensure existing tests pass.
- Ensure the cpu coverage reported by our `ci_test-full.yml` workflow is >= the existing coverage.

## Build and Validation Commands

### Environment Setup

Always install dependencies in order to avoid conflicts. We now maintain optional, committed CI pinned requirements and a small helper to generate them; follow the flow below depending on whether you want a standard developer install or to reproduce CI pinned installs.

Developer (default, fast): install from `pyproject.toml` / editable install

```bash
# Basic development setup
python -m pip install --upgrade pip setuptools wheel build
python -m pip install -e '.[test]'

# Full development with examples (may fail due to circuit-tracer dependency)
python -m pip install -e '.[test,examples,lightning]'

# If circuit-tracer install fails, use the built-in tool after basic install:
pip install interpretune[examples]
interpretune-install-circuit-tracer
```

Reproducible CI-style install (preferred for CI or to reproduce pinned builds):

```bash
# (1) Regenerate pinned inputs on a canonical builder (requires pip-tools to run pip-compile):
# python -m pip install pip-tools toml
# python requirements/regen_reqfiles.py --mode pip-compile --ci-output-dir=requirements/ci
# or locally: pip-compile requirements/ci/requirements.in --output-file requirements/ci/requirements.txt

# (2) Install from the generated pinned file
pip install -r requirements/ci/requirements.txt

# (3) Optionally apply post-upgrades (controlled by repo vars / env), see below
export APPLY_POST_UPGRADES=1
pip install --upgrade -r requirements/post_upgrades.txt
```

**⚠️ Known Issue:** Full dependency install may timeout due to large ML packages. Install basic deps first, then add extras incrementally.

### Development Environment Scripts
For complex setups, use the provided build script:

```bash
# Standard development build (recommended for dev work)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest

# Build without circuit-tracer commit pin
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --no_commit_pin
```

### Linting and Code Quality
**Always run linting before committing:**

```bash
# Run ruff linting (configured in pyproject.toml)
ruff check .
ruff check . --fix  # Auto-fix issues

# Run pre-commit hooks (includes ruff, docformatter, yaml checks)
pre-commit run --all-files
```

**Expected Ruff Issues:** The `tests/*_parity/` directories contain imported research code with many linting violations - these are intentionally excluded from pre-commit checks.

### Testing
**Test command:**
```bash
# Basic test run (requires full dependencies)
pytest src/interpretune tests -v

# With coverage (as used in CI)
coverage run --source src/interpretune -m pytest src/interpretune tests -v
coverage report

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
├── __about__.py            # Version info (0.1.0dev)
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
  - `ci_constraints.txt` - CI version pins
  - `test.txt`, `examples.txt`, etc. - Optional dependency groups

### Key Entry Points
- Console script: `interpretune` → `interpretune.base.components.cli:bootstrap_cli`
- Circuit-tracer installer: `interpretune-install-circuit-tracer`

## CI and Validation Pipeline

### GitHub Actions Workflow
**File:** `.github/workflows/ci_test-full.yml`

**Customizing CI for OS-specific Issues:**
When working on an issue that involves only a specific OS, you can customize the `ci_test-full.yml` workflow to run only that OS (and optionally only a subset of tests). See the `bugfix/debug_windows_tl_direct_attr` branch for an example of this approach.

**Triggers:** Push/PR to main, changes to source/test files
**Platforms:** Ubuntu 22.04, Windows 2022, macOS 14 (Python 3.12)
**Timeout:** 90 minutes

**CI Process:**
1. Install dependencies with constraints
2. Run pytest with coverage
3. Resource monitoring (Linux only)
4. Upload artifacts on failure

**Environment Variables for CI:**
- `IT_CI_LOG_LEVEL` - Defaults to "INFO", set to "DEBUG" for verbose logging
- `CI_RESOURCE_MONITOR` - Set to "1" to enable resource logging
- `IT_USE_CT_COMMIT_PIN` - Controls circuit-tracer installation method

### Manual Validation Steps
```bash
# Lint check
ruff check .

# Type checking (limited scope for now, we intend to exapand this as we get closer to releasing the MVP)
pyright src/interpretune/adapters/lightning.py

# Test with resource monitoring (Linux)
./scripts/ci_resource_monitor.sh &
pytest src/interpretune tests -v
```

## Special Dependencies and Known Issues


### Circuit-Tracer Dependency
**Issue:** circuit-tracer is not on PyPI, requires git-based install

**Solutions:**
1. Use built-in installer: `interpretune-install-circuit-tracer`
2. Manual install: `pip install git+https://github.com/speediedan/circuit-tracer.git@<commit>`
3. Environment variable control: `IT_USE_CT_COMMIT_PIN=1`

### Dependency Constraints and Required Upgrades
- **datasets** pinned to 2.21.0 for sae_lens compatibility and reducing the dependency graph during installation, but **Interpretune requires the latest `datasets` and `fsspec` for full functionality**. After any dependency install, always run:
  - `pip install --upgrade datasets`
  - `pip install --upgrade fsspec`
  This is required until sae-lens provides support for datasets >= 4.0 (see [ci_test-full.yml](../workflows/ci_test-full.yml)).
- **torch** requires 2.7.1+ for newer features
- **setuptools** requires 77.0.0+ for PEP 639 support

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
