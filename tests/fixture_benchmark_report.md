# Interpretune Fixture Benchmark Report

**Generated:** 2025-09-01 14:28:50
**GPU Available:** True
**GPU Device:** NVIDIA GeForce RTX 4090
**Total Fixtures Analyzed:** 40
**Baseline Pytest Time:** 13.195s

## Import Profiling Artifacts

📊 **Import Analysis**:
- Raw timing data: [pytest_importtime_raw_20250901_140922.txt](/home/speediedan/repos/interpretune/tests/profiling_artifacts/pytest_importtime_raw_20250901_140922.txt)
- Flamegraph: [pytest_importtime_flamegraph_20250901_140922.txt](/home/speediedan/repos/interpretune/tests/profiling_artifacts/pytest_importtime_flamegraph_20250901_140922.txt)

## Summary Statistics

> **Note:** Fully contextualized fixture profiling (i.e. using --setup-show with a full test suite run) is
> planned and will be linked to here. The focus of this report is fixture enumeration and isolated
> analysis/benchmarking.

- **Total Fixtures:** 40


## Detailed Fixture Analysis

| Fixture Name | Type | Scope | Category | Init Time (s) | Memory Δ (MB) | GPU Δ (MB) | Status |
|--------------|------|-------|----------|---------------|---------------|------------|--------|
| get_it_session_cfg__core_cust | dynamic | session | Config Only | 0.000 | 0.2 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_cust | dynamic | session | Config Only | 0.000 | 0.6 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_gpt2 | dynamic | class | Config Only | 0.000 | -0.1 | 0.0 | ✅ Success |
| get_it_session_cfg__tl_cust | dynamic | session | Config Only | 0.000 | -0.4 | 0.0 | ✅ Success |
| get_it_module__core_cust__setup | dynamic | class | Custom Model | 0.906 | 7.7 | 0.0 | ✅ Success |
| get_it_session__core_cust__initonly | dynamic | session | Custom Model | 1.510 | 33.8 | 0.0 | ✅ Success |
| get_it_session__core_cust__setup | dynamic | session | Custom Model | 1.540 | 35.6 | 0.0 | ✅ Success |
| get_it_session__core_cust_force_prepare__initonly | dynamic | class | Custom Model | 1.256 | 33.9 | 0.0 | ✅ Success |
| get_it_session__core_cust_memprof__initonly | dynamic | class | Custom Model | 1.878 | 53.6 | 0.0 | ✅ Success |
| get_it_session__tl_cust__initonly | dynamic | session | Custom Model | 1.574 | 36.1 | 0.0 | ✅ Success |
| get_it_session__tl_cust__setup | dynamic | session | Custom Model | 1.753 | 35.9 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis | dynamic | class | Real Model | 44.061 | 1186.9 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis | dynamic | session | Real Model | 12.932 | 3069.1 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis | dynamic | session | Real Model | 10.734 | 1100.5 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis | dynamic | session | Real Model | 11.363 | 1191.3 | 0.0 | ✅ Success |
| get_it_session__core_gpt2_peft__initonly | dynamic | class | Real Model | 2.519 | 404.5 | 208.5 | ✅ Success |
| get_it_session__core_gpt2_peft_seq__initonly | dynamic | class | Real Model | 2.647 | 485.8 | 208.5 | ✅ Success |
| get_it_session__l_gemma2_debug__setup | dynamic | class | Real Model | 3.767 | 210.0 | 4986.5 | ✅ Success |
| get_it_session__l_llama3_debug__setup | dynamic | class | Real Model | 7.948 | 1027.8 | 2900.5 | ✅ Success |
| get_it_session__l_sl_gpt2__initonly | dynamic | class | Real Model | 4.861 | 975.6 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2__initonly | dynamic | class | Real Model | 4.854 | 992.1 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2_analysis__initonly | dynamic | session | Real Model | 4.639 | 976.9 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2_analysis__setup | dynamic | session | Real Model | 6.466 | 986.1 | 0.0 | ✅ Success |
| get_it_session__tl_gpt2_debug__setup | dynamic | class | Real Model | 2.965 | 681.3 | 0.0 | ✅ Success |
| clean_cli_env | static | function | Static | 0.000 | -0.6 | 0.0 | ✅ Success |
| cli_test_configs | static | session | Static | 0.000 | -0.7 | 0.0 | ✅ Success |
| cli_test_file_env | static | session | Static | 0.000 | -0.1 | 0.0 | ✅ Success |
| datadir | static | session | Static | 0.000 | 0.2 | 0.0 | ✅ Success |
| fts_patch_env | static | function | Static | 0.000 | -0.2 | 0.0 | ✅ Success |
| gpt2_ft_schedules | static | function | Static | 6.824 | 2721.3 | 0.0 | ✅ Success |
| huggingface_env | static | function | Static | 0.000 | -0.4 | 0.0 | ✅ Success |
| make_deterministic | static | session | Static | 0.000 | -0.3 | 0.0 | ✅ Success |
| make_it_module | static | class | Static | 0.000 | -0.3 | 0.0 | ✅ Success |
| mock_analysis_store | static | function | Static | 0.000 | 0.4 | 0.0 | ✅ Success |
| mock_dm | static | class | Static | 0.000 | 3.2 | 0.0 | ✅ Success |
| preserve_global_rank_variable | static | function | Static | 0.000 | 0.7 | 0.0 | ✅ Success |
| restore_env_variables | static | function | Static | 0.000 | -0.4 | 0.0 | ✅ Success |
| restore_grad_enabled_state | static | function | Static | 0.000 | -0.6 | 0.0 | ✅ Success |
| teardown_process_group | static | function | Static | 0.000 | 0.5 | 0.0 | ✅ Success |
| tmpdir_server | static | function | Static | 0.000 | -0.9 | 0.0 | ✅ Success |
