# Dynamic Fixture Benchmark & Performance Analysis

This tool provides comprehensive analysis of all pytest fixtures in the Interpretune test suite, including performance benchmarking with memory and GPU measurement.

## Features

✅ **Dynamic Discovery**: Automatically discovers all generated fixtures from `conftest.py`
✅ **Smart Categorization**: Groups fixtures by Real Models (gpt2, llama3, gemma2) vs Custom Test Models vs Config-Only
✅ **Usage Analysis**: Counts actual fixture usage across all test files
✅ **Scope Analysis**: Tracks fixture scopes (session, class, function)
✅ **Performance Benchmarking**: Measures init time, memory usage, and GPU memory for each fixture
✅ **Import Profiling**: Generates pytest startup flamegraphs to identify slow imports
✅ **Optimization Insights**: Identifies high-impact optimization targets based on usage and performance
✅ **Cleanup Recommendations**: Finds unused fixtures that could be removed

## Generated Report Includes

1. **Summary Statistics** - Overall fixture counts and usage percentages by category and scope
2. **Detailed Analysis Table** - Complete fixture inventory with usage counts, performance metrics, and metadata
3. **Performance Insights** - Top candidates for optimization based on usage, scope, init time, and memory consumption
4. **Cleanup Recommendations** - Unused fixtures that could be considered for removal
5. **Import Profiling Artifacts** - Raw importtime data and flamegraph visualizations for pytest startup analysis

## Performance Measurement

The benchmark uses a comparative approach to measure fixture overhead:
- **Baseline Test**: Measures pytest startup time with minimal test that imports core modules
- **Fixture Test**: Measures time/memory when fixture is actually instantiated
- **Memory Δ**: Difference in peak memory usage between baseline and fixture tests
- **GPU Δ**: Difference in GPU memory allocation (when GPU available)
- **Import Profiling**: Records module import times during pytest startup for flamegraph generation

## Usage

```bash
# From the repository root:
cd /path/to/interpretune
source ~/.venvs/it_latest/bin/activate  # Activate your environment
python tests/dynamic_fixture_benchmark.py

# For faster iteration during development:
python tests/dynamic_fixture_benchmark.py --max-fixtures=10

# For background execution with detailed logging:
./scripts/manage_standalone_processes.sh tests/dynamic_fixture_benchmark.py --repo_home=${HOME}/repos/interpretune --target_env_name=it_latest
```

## Generated Artifacts

This generates several outputs:
- **`tests/fixture_benchmark_report.md`** - Main analysis report with performance data
- **`tests/profiling_artifacts/pytest_importtime_raw_*.txt`** - Raw pytest import timing data
- **`tests/profiling_artifacts/pytest_importtime_flamegraph_*.txt`** - Flamegraph visualization for pytest startup imports

The report includes direct links to import profiling artifacts for easy access to performance data.

## Key Insights from Latest Report

- **47 total fixtures** analyzed (27 generated, 20 static)
- **55.3% fixture utilization** (26 used, 21 unused)
- **100% usage rate** for Custom Model and Config-Only fixtures
- **93.3% usage rate** for Real Model fixtures
- **0% usage rate** for Static fixtures (many are autouse or internal)

## High-Impact Optimization Targets

Session-scope fixtures with heavy usage that would benefit most from optimization:

- `get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis` (17 uses)
- `get_it_session_cfg__tl_cust` (11 uses)
- `get_it_session__tl_gpt2_debug__setup` (10 uses)
- `get_it_session__core_cust__setup` (8 uses)

These fixtures are instantiated once per session but used heavily across multiple tests, making them prime candidates for performance optimization.
