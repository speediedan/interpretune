# Circuit-Tracer Installation for Interpretune

## Overview

Circuit-tracer is an optional but frequently used dependency for Interpretune that may have dependency conflicts with other interpretune dependencies.

Once circuit-tracer is available on PyPI directly via either the official Anthropic repo or a fork packaged for use with Interpretune, `circuit-tracer` can be installed via interpretune with the examples extra (`pip install interpretune[examples]`), which will include circuit-tracer. Making circuit-tracer available on PyPI is currently an outstanding [issue](https://github.com/speediedan/interpretune/issues/<TODOADD_LINK>).

Once circuit-tracer is available on PyPI, the need for an additional installation below will be eliminated. In the meantime, here are the current options for installing it:

## Installation Methods

### Method 1: Using the Built-in Installation Tool (Recommended)

Install circuit-tracer with the built-in tool after installing interpretune with the examples extra:

```bash
# Install interpretune with examples dependencies
pip install interpretune[examples]

# Install circuit-tracer using the built-in tool
interpretune-install-circuit-tracer
```

This will:
1. Install circuit-tracer dependencies that may conflict with other interpretune dependencies (like safetensors and numpy)
2. Install circuit-tracer from the pinned commit (default behavior)

### Method 2: Manual Installation

If you prefer manual control:

```bash
# Install prerequisites
pip install safetensors>=0.5.0 numpy>=2.0.0

# Install circuit-tracer from a specific commit
pip install --no-deps git+https://github.com/speediedan/circuit-tracer.git@6a05a1612f6eea60e3acf51b8e10a205ce9e8650
```

## Configuration Options

### Using Commit Pin (Default)

```bash
interpretune-install-circuit-tracer
```

This installs circuit-tracer from a specific tested commit for maximum compatibility.

### Using Version-based Installation

```bash
interpretune-install-circuit-tracer
```

This attempts to install circuit-tracer from PyPI (currently not available, will fail).

### Environment Variables

You can also control the installation behavior using environment variables:

```bash
# Force commit-based installation
export IT_USE_CT_COMMIT_PIN=1
interpretune-install-circuit-tracer

# Force version-based installation
export IT_USE_CT_COMMIT_PIN=0
interpretune-install-circuit-tracer
```

## Development Setup

For development environments, the build scripts handle circuit-tracer installation automatically:

```bash
# Standard development build (uses commit pin)
./scripts/build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest

# Development build without CT commit pin
./scripts/build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest

# Development build with local circuit-tracer source
./scripts/build_it_env.sh --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest --ct_from_source=${HOME}/repos/circuit-tracer
```

## Troubleshooting

### Installation Fails

If circuit-tracer installation fails:

1. Ensure you have git installed (required for git-based pip installs)
2. Check your network connection to GitHub
3. Try installing prerequisites manually first:
   ```bash
   pip install safetensors>=0.5.0 numpy>=2.0.0
   ```

### Import Errors

If you get import errors when using circuit-tracer functionality:

1. Verify circuit-tracer is installed: `pip list | grep circuit-tracer`
2. Try reimporting interpretune: `python -c "import interpretune; print('OK')"`
3. Check for version conflicts: `pip check`

### Version Conflicts

If you have dependency conflicts:

1. Create a fresh virtual environment
2. Install interpretune first, then circuit-tracer
3. Consider using `--no-deps` flag during installation

## Why This Approach?

This approach provides several benefits:

1. **Optional Dependency**: Circuit-tracer is only installed when needed
2. **PyPI Compatibility**: Interpretune can be published to PyPI without git dependencies
3. **Flexible Installation**: Users can choose commit-based or version-based installation
4. **Development Friendly**: Supports local development workflows

When circuit-tracer becomes available on PyPI, the version-based installation will work seamlessly without requiring any changes to user workflows.
