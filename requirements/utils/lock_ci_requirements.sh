#!/bin/bash
# Simple wrapper around uv pip compile for CI requirements locking
# This replaces the complex regen_reqfiles.py with a straightforward uv-based approach
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CI_DIR="${REPO_ROOT}/requirements/ci"

# Ensure output directory exists
mkdir -p "${CI_DIR}"

echo "Generating locked CI requirements from pyproject.toml..."

# uv pip compile can read directly from pyproject.toml
# We include:
# - Base dependencies (always included from [project.dependencies])
# - Optional dependencies: examples, lightning
# - Dependency groups: test, profiling, dev
#
# Note: git-deps group is excluded from locking because git URLs cannot be
# included in universal lock files. It will be installed separately.

uv pip compile \
    "${REPO_ROOT}/pyproject.toml" \
    --extra examples \
    --extra lightning \
    --group dev \
    --group test \
    --group profiling \
    --output-file "${CI_DIR}/requirements.txt" \
    --upgrade \
    --no-strip-extras \
    --universal

echo "✓ Generated ${CI_DIR}/requirements.txt"
echo ""
echo "Note: git-deps group (git URL dependencies) is installed separately in CI"
