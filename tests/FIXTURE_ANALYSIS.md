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

## S9 Fixture & Standalone Test Analysis (2026-03-05)

### Fixture Cleanup Summary

As part of S9.9, unused or underutilized fixtures were audited. Changes applied to `conftest.py`:

**Commented out (unused):**
| Fixture Key | Reason |
|---|---|
| `l_gemma2_debug` | No test references; debug generation tests use other fixtures |
| `l_ct_nnsight_gemma2` | Pending circuit-tracer nnsight backend (Q4/Q5 design scope) |
| `l_ct_nnsight_gemma3` | Same — future backend work |

**Trimmed (removed unused variants):**
| Fixture Key | Removed Variants | Remaining |
|---|---|---|
| `ns_gpt2` | `initonly` | `setup`, `cfgonly` |
| `l_ns_gpt2` | `initonly`, `cfgonly` | `setup` |
| `sl_ns_gpt2` | `cfgonly` | `initonly`, `setup` |
| `sl_ht_gpt2_analysis` | `initonly` | `setup` |

**Deliberately kept:**
- `sl_br_gpt2` — both `initonly` and `cfgonly` variants used by Bridge migration tests

**Imports commented:** `LightningGemma2Debug`, `LightningCTNNsightGemma2`, `LightningCTNNsightGemma3`

### TransformerBridge SAE Test Migration

4 new Bridge parametrizations were added to `test_sl_module_init` in
`tests/core/test_adapters_sae_lens.py`:

| Param ID | Fixture | add_sae |
|---|---|---|
| `bridge_slcust_noadd` | `get_it_session_cfg__sl_br_gpt2` | False |
| `bridge_slcust_add` | `get_it_session_cfg__sl_br_gpt2` | True |
| `bridge_slgpt2_noadd` | `get_it_session_cfg__sl_br_gpt2` | False |
| `bridge_slgpt2_add` | `get_it_session_cfg__sl_br_gpt2` | True |

**Why only `test_sl_module_init`:** This test validates SAE loading/configuration
which works identically for Bridge because `SAETransformerBridge._acts_to_saes`
is keyed by `input_hook_alias` (the HT-convention name from `sae.cfg.metadata.hook_name`),
matching `normalized_sae_cfg_refs` output.

**Tests that must remain HT-only:** `test_run_with_cache_with_saes`,
`test_run_with_hooks_with_saes`, and `test_run_with_saes_with_cache_fwd_bwd` use
`run_with_cache` which returns cache keys using resolved hook names (not aliases).
Bridge resolves `hook_resid_pre` → `hook_in`, causing key mismatches with expected
HT-convention names.

### Standalone Test Categorization

29 tests were marked `@pytest.mark.standalone` (or `@RunIf(standalone=True)`).
After analysis, 4 were moved to the main collection. The remaining 25 require
standalone execution for the reasons documented below.

**Moved to main collection (4 tests):**

| Test | File | Reason safe to move |
|---|---|---|
| `test_framework` | `test_analysis_injection.py` | Pure Python, no model loading or GPU |
| `test_orchestrator_access` | `test_analysis_injection.py` | Pure Python config validation |
| `test_notebook_discovery` | `test_notebooks.py` | File path checks only |
| `test_op_collection_notebooks` | `test_notebooks.py` | Validates notebook cell metadata |

**Remaining standalone tests (25):**

| Category | Count | Tests | Reason |
|---|---|---|---|
| Circuit tracer (Gemma2) | 10 | `test_ct_session_backends[...]` (10 params) | Large model (Gemma2-2b), GPU required, transcoder downloads |
| NNsight trace | 1 | `test_nnsight_trace_context` | nnsight's `sys.settrace(None)` interferes with pytest/coverage tracers |
| TL parameter mapping | 2 | `test_llama3_tl_param_structure`, `test_gemma2_tl_param_structure` | Large gated models (3B+), GPU + bf16 |
| Debug generation | 2 | `test_debug_generation_llama3`, `test_debug_generation_gemma2` | Large gated models, GPU + bf16 |
| Notebook execution | 8 | `test_attribution_analysis_notebook[...]` (5), `test_ct_notebook[...]` (3) | GPU + bf16, model downloads, long runtime |
| FTS parity | 1 | `test_parity_fts[train_cuda_32_l_fts]` | Disk space requirements |
| Profiling | 1 | `test_tl_profiling[test_cuda_32]` | Memory measurement sensitivity |
