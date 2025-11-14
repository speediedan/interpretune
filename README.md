# Interpretune

Interpretune: A flexible, powerful framework for collaborative LLM world model analysis and tuning.

> **Status:** Pre-MVP (Minimum Viable Product) – Active development, not yet feature complete.

## Overview
Interpretune aims to provide tools and infrastructure for:
- Analyzing large language model (LLM) world models
- Collaborative research with shareable/composable experimentation
- Flexible tuning and interpretability workflows

## Installation

Interpretune uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Quick Start

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/speediedan/interpretune.git
cd interpretune

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install interpretune in editable mode with all dependencies
# Note: git-deps group is optional once circuit-tracer is published on PyPI
uv pip install -e ".[test,examples,lightning,profiling]" --group git-deps dev

# Run tests
pytest tests/ -v
```

### Development Setup

For advanced development workflows, use the provided build script which supports locked CI requirements and from-source packages:

```bash
# Standard development build (uses locked CI requirements)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest

# Activate the created environment
source ~/.venvs/it_latest/bin/activate

# Build with packages from source (useful for development)
./scripts/build_it_env.sh --repo_home=${PWD} --target_env_name=it_latest \
  --from-source="finetuning_scheduler:${HOME}/repos/finetuning-scheduler:all" \
  --from-source="circuit_tracer:${HOME}/repos/circuit-tracer"
```

### Locked Requirements for CI

CI workflows use locked requirements for reproducibility:

```bash
# Install using locked CI requirements (CI approach)
uv pip install -e . --group git-deps
uv pip install -r requirements/ci/requirements.txt

# Regenerate locked requirements (after updating pyproject.toml)
./requirements/utils/lock_ci_requirements.sh
```

## Project Status
This project is in the **pre-MVP** stage. Features and APIs are subject to change. Contributions and feedback are welcome as the framework evolves.

## License
See [LICENSE](./LICENSE) for details.
