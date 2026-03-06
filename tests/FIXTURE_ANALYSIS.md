# Dynamic Fixture Benchmark & Performance Analysis

This tool provides an enumeration and cursory isolated analysis of all pytest fixtures in the Interpretune test suite to facilitate developer fixture selection and test management/analysis. It is especially useful with respect to our set of dynamically generated fixtures.

## Features

✅ **Dynamic Discovery**: Automatically discovers all generated fixtures from `conftest.py`

✅ **Categorization**: Groups fixtures by Real Models (gpt2, llama3, gemma2) vs Custom Test Models vs Config-Only

✅ **Scope Analysis**: Tracks fixture scopes (session, class, function)

✅ **Performance Benchmarking**: Measures init time, memory usage, and GPU memory for each fixture

✅ **Pytest startup baseline**: Establishes a baseline pytest startup time for comparison to isolated fixture init time
(note there are numerous heavy package imports that occur during pytest startup that can be profiled using the guide in `tests/PROFILING.md`)

## Performance Measurement

The benchmark uses a comparative approach to measure fixture overhead:
- **Baseline Test**: Measures pytest startup time with minimal test that imports core modules
- **Fixture Test**: Measures time/memory when fixture is actually instantiated
- **Memory Δ**: Difference in peak memory usage between baseline and fixture tests
- **GPU Δ**: Difference in GPU memory allocation (when GPU available)

## Usage

```bash
export IT_VENV_BASE=/mnt/cache/${USER}/.venvs
export IT_TARGET_VENV=it_latest
export IT_REPO_DIR=${HOME}/repos/interpretune

# From the repository root:
cd ${IT_REPO_DIR}
source ${IT_VENV_BASE}/${IT_TARGET_VENV}/bin/activate  # Activate your environment
python tests/dynamic_fixture_benchmark.py

# For faster iteration during development:
python tests/dynamic_fixture_benchmark.py --max-fixtures=10

# For background execution:
${IT_REPO_DIR}/scripts/manage_standalone_processes.sh --use_nohup ${IT_REPO_DIR}/tests/dynamic_fixture_benchmark.py
```

## Generated Artifacts

- **`tests/fixture_benchmark_report.md`** - Main enumeration/analysis report is dynamically generated/overwritten on each run
- **`tests/profiling_artifacts/pytest_startup_speedscope_<timestampe>.json`** - Pytest startup baseline speedscope profile (see `tests/PROFILING.md` for more)
- log of the benchmark generation `/tmp/dynamic_fixture_benchmark.py_<timestamp>_wrapper.out`

---
