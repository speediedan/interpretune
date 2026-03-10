---
applyTo: "**/tests/parity_acceptance/**"
---

# Updating Expected Test Results

## Overview

This instruction file provides guidance for updating expected test results in the parity_acceptance test suite. Expected results may need updating when:
- Memory footprints change due to dependency updates
- Model architectures are updated
- New test configurations are added
- Profiling tests produce new baseline values

## Key Files

- `tests/parity_acceptance/expected.py` - Test result definitions and mappings
- `tests/parity_acceptance/profile_memory_footprints.yaml` - Memory profiling expected values
- `tests/results.py` - `TestResult`, `MemProfResult` classes

## Capturing New Expected Values

### Step 1: Run Tests with State Logging

To capture new expected values, run tests with `IT_GLOBAL_STATE_LOG_MODE=1`:

```bash
IT_GLOBAL_STATE_LOG_MODE=1 python -m pytest \
  tests/parity_acceptance/test_it_ns.py::test_parity_ns[test_cpu_32] -v
```

This will output the actual test results to the console, which can be used to update expected values.

### Step 2: Update Memory Footprints YAML

For profiling tests, update `profile_memory_footprints.yaml`:

```yaml
test_ns_profiling.test_cuda_32:
  mem_results:
    expected_mem:
      allocated_bytes.all.current: <actual_value>
      allocated_bytes.all.peak: <actual_value>
      npp_diff: <actual_value>
      reserved_bytes.all.peak: <actual_value>
    phase: test
    src: cuda
```

### Step 3: Update expected.py Results

For non-profiling tests, update the result dictionaries in `expected.py`:

```python
ns_parity_results = {
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    # Add new entries as needed
}
```

## Using the Profiling Update Script

For batch updates to memory footprints, use the `update_profiling_memory_stats.sh` script:

```bash
# Standard usage with default venv location (IT_VENV_BASE or ~/.venvs)
./scripts/update_profiling_memory_stats.sh \
  --repo-home=${HOME}/repos/interpretune \
  --target-env-name=it_latest \
  --working-dir=/tmp

# With custom venv base directory (recommended when using shared cache for hardlinks)
# NOTE: The script will compute the full venv path as <venv-dir>/<target-env-name>.
./scripts/update_profiling_memory_stats.sh \
  --repo-home=${HOME}/repos/interpretune \
  --target-env-name=it_latest \
  --working-dir=/tmp \
  --venv-dir=/mnt/cache/${USER}/.venvs
```

Behavior notes:
- If `--venv-dir` is provided, the script uses `<venv-dir>/<target-env-name>/bin/activate` to activate the venv.
- If `--venv-dir` is not provided, the script falls back to the `IT_VENV_BASE` environment variable (if set) and then to `~/.venvs`.
- This mirrors the approach used by the repository's `build_it_env.sh`, and the venv resolution uses the same `determine_venv_path()` helper from `scripts/infra_utils.sh`.

The script runs profiling tests with `IT_GLOBAL_STATE_LOG_MODE=1` for three marker types:
1. `profile_ci` - Profiling tests that run in CI (`IT_RUN_PROFILING_TESTS=1`)
2. `profile` - Extended profiling tests (`IT_RUN_PROFILING_TESTS=2`)
3. `optional` - Optional tests (`IT_RUN_OPTIONAL_TESTS=1`)

The script generates diffs showing memory footprint changes for each marker type.

## Result Structure

### TestResult Fields
- `exact_results`: Tuple of (phase, key, value) for exact matches
- `close_results`: Tuple of (phase, key, value) for approximate matches
- `tolerance_map`: Dict of tolerances for specific keys

### MemProfResult Keys
- `rss_diff`: RSS memory difference (CPU)
- `npp_diff`: Non-parameter packed bytes diff
- `allocated_bytes.all.current`: Current CUDA allocation
- `allocated_bytes.all.peak`: Peak CUDA allocation
- `reserved_bytes.all.peak`: Peak CUDA reservation

## Validation

After updating expected values:

1. Run the specific test to verify:
```bash
python -m pytest tests/parity_acceptance/test_it_ns.py::test_parity_ns[test_cpu_32] -v
```

2. Run the full test suite to ensure no regressions:
```bash
python -m pytest tests/parity_acceptance/ -v
```

3. Check coverage is maintained:
```bash
python -m coverage run -m pytest tests/ && python -m coverage report
```
