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
Always install dependencies in order to avoid conflicts:

```bash
# Basic development setup
python -m pip install --upgrade pip setuptools setuptools-scm wheel build
python -m pip install -r requirements/ci/requirements.txt -r requirements/ci/platform_dependent.txt
python -m pip install -e '.[test,examples,lightning]'

# If circuit-tracer install fails, use the built-in tool after basic install:
pip install interpretune[examples]
interpretune-install-circuit-tracer
```

**⚠️ Known Issue:** Full dependency install may timeout due to large ML packages. Install basic deps first, then add extras incrementally.

### Development Environment Scripts
For complex setups, use the provided build script:

```bash
# Standard development build (recommended for dev work)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest

# Build with a circuit-tracer commit pin
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest --ct-commit-pin
```

### Linting and Code Quality
**Always run linting before committing:**

```bash
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
**Test command:**
```bash
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
- Circuit-tracer installer: `interpretune-install-circuit-tracer`

## CI and Validation Pipeline

### GitHub Actions Workflow
**File:** `.github/workflows/ci_test-full.yml`

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

### Azure self-hosted GPU pipeline (new)

We now have a separate Azure DevOps pipeline that runs GPU/standalone tests on a self-hosted runner: `.azure-pipelines/gpu-tests.yml`.
- This pipeline is intentionally restrictive: it only triggers for PRs that are marked "ready for review" and must be explicitly approved by a repository administrator before the self-hosted GPU job will run (currently: speediedan).
- Because self-hosted GPU capacity is limited, aim to rely on feedback from the normal GitHub Actions CPU CI workflow for as long as possible while iterating on an issue. Defer switching the PR to "ready for review" until you believe GPU testing is necessary. Copilot should prefer this conservative approach when suggesting CI runs or opening PRs.

Note: the GPU pipeline runs only when a PR is ready for review and an admin approves the run — do not expect it to run automatically for draft PRs or early-stage work.

### Manual Validation Steps
```bash
# Lint check
ruff check .

# Type checking (limited scope)
pyright src/interpretune/adapters/lightning.py

# Test with resource monitoring (Linux)
./scripts/ci_resource_monitor.sh &
pytest src/interpretune tests -v
```

### Regenerating stable CI dependency pins

When updating top-level requirements or periodically refreshing CI pins, use the repository helper to regenerate and compile the CI requirement files. This workflow updates `requirements/*` and writes compiled CI pins to `requirements/ci`.

Run these commands from your repo home after activating ensuring you've activated any relevant venv (e.g. `source ~/.venvs/${target_env_name}/bin/activate`):

```bash
python requirements/regen_reqfiles.py && \
python requirements/regen_reqfiles.py --mode pip-compile --ci-output-dir=requirements/ci
```

Notes:
- Regenerating pins may change CI dependency resolution — run the full CI (or at least the CPU GitHub Actions CI) after updating pins to validate. Don't update pins aggressively, this is done periodically anyway, focus mostly on the issue at hand without changing the CI pins unless you think it is related to the issue.

### Type-checking caveat

Full repository type-checking is a work in progress. Current local checks may only include a subset of files (for example, `src/interpretune/adapters/lightning.py`). Expect that type-checking will cover most files in future updates; don't assume exhaustive static type guarantees yet.

## Special Dependencies and Known Issues

### Circuit-Tracer Dependency
**Issue:** circuit-tracer is not on PyPI, requires git-based install

**Solutions:**
1. Use built-in installer: `interpretune-install-circuit-tracer`
2. Manual install: `pip install git+https://github.com/speediedan/circuit-tracer.git@<commit>`
3. Environment variable control: `IT_USE_CT_COMMIT_PIN=1`

### Dependency Constraints
- **datasets** pinned to 2.21.0 for sae_lens compatibility
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
