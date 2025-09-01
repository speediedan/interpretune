# Interpretune Fixture Benchmark Report

**Generated:** 2025-08-31 14:31:29
**GPU Available:** True
**GPU Device:** NVIDIA GeForce RTX 4090
**Total Fixtures Analyzed:** 44

## Summary Statistics

> **Note:** Fixture benchmarking is limited due to pytest fixture dependency
> injection requirements. This report focuses on fixture discovery, usage
> analysis, and categorization. For accurate performance benchmarking,
> fixtures should be tested within a proper pytest session context.

- **Total Fixtures:** 44
- **Used Fixtures:** 23 (52.3%)
- **Unused Fixtures:** 21

### By Category

| Category | Count | Used | Usage % |
|----------|-------|------|---------|
| Custom Model | 10 | 10 | 100.0% |
| Real Model | 14 | 13 | 92.9% |
| Static | 20 | 0 | 0.0% |

### By Scope

| Scope | Count | Used | Usage % |
|-------|-------|------|---------|
| class | 15 | 12 | 80.0% |
| session | 16 | 11 | 68.8% |
| function | 13 | 0 | 0.0% |

## Detailed Fixture Analysis

| Fixture Name | Uses | Type | Scope | Category | Init Time (s) | Memory Δ (MB) | GPU Δ (MB) | Status |
|--------------|------|------|-------|----------|---------------|---------------|------------|--------|
| get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis | 17 | dynamic | session | Real Model | 11.888 | 1095.7 | 0.0 | ✅ Success |
| get_it_session_cfg__tl_cust | 11 | dynamic | session | Custom Model | 0.000 | 0.3 | 0.0 | ✅ Success |
| get_it_session__tl_gpt2_debug__setup | 10 | dynamic | class | Real Model | 3.792 | 684.2 | 0.0 | ✅ Success |
| get_it_session__core_cust__setup | 8 | dynamic | session | Custom Model | 1.803 | 35.2 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis | 6 | dynamic | session | Real Model | 12.326 | 1177.5 | 0.0 | ✅ Success |
| get_it_module__core_cust__setup | 6 | dynamic | class | Custom Model | 1.799 | 6.8 | 0.0 | ✅ Success |
| get_it_session_cfg__core_cust | 6 | dynamic | session | Custom Model | 0.000 | -0.3 | 0.0 | ✅ Success |
| get_it_session__core_cust__initonly | 5 | dynamic | session | Custom Model | 2.249 | 34.0 | 0.0 | ✅ Success |
| get_it_session__core_gpt2_peft__initonly | 4 | dynamic | class | Real Model | 3.459 | 472.6 | 208.5 | ✅ Success |
| get_it_session__l_sl_gpt2__initonly | 4 | dynamic | class | Real Model | 5.742 | 1013.0 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2__initonly | 4 | dynamic | class | Real Model | 5.483 | 974.3 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2_analysis__setup | 4 | dynamic | session | Real Model | 7.089 | 1016.0 | 0.0 | ✅ Success |
| get_it_session__tl_cust__setup | 4 | dynamic | session | Custom Model | 2.238 | 36.9 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_gpt2 | 4 | dynamic | class | Real Model | 0.085 | 0.2 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis | 2 | dynamic | session | Real Model | 15.061 | 2915.1 | 0.0 | ✅ Success |
| get_it_session__core_cust_force_prepare__initonly | 2 | dynamic | class | Custom Model | 2.133 | 34.5 | 0.0 | ✅ Success |
| get_it_session__core_cust_memprof__initonly | 2 | dynamic | class | Custom Model | 2.715 | 51.9 | 0.0 | ✅ Success |
| get_it_session__core_gpt2_peft_seq__initonly | 2 | dynamic | class | Real Model | 3.319 | 455.1 | 208.5 | ✅ Success |
| get_it_session__l_gemma2_debug__setup | 2 | dynamic | class | Real Model | 6.951 | 208.4 | 4986.5 | ✅ Success |
| get_it_session__l_llama3_debug__setup | 2 | dynamic | class | Real Model | 10.736 | 1154.5 | 2900.5 | ✅ Success |
| get_it_session__tl_cust__initonly | 2 | dynamic | session | Custom Model | 2.183 | 36.1 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_cust | 2 | dynamic | session | Custom Model | 0.000 | 1.4 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis | 1 | dynamic | class | Real Model | 47.495 | 1124.2 | 0.0 | ✅ Success |
| clean_cli_env | 0 | static | function | Static | 0.000 | -0.7 | 0.0 | ✅ Success |
| cli_test_configs | 0 | static | session | Static | 0.002 | -0.0 | 0.0 | ✅ Success |
| cli_test_file_env | 0 | static | session | Static | 0.116 | 0.2 | 0.0 | ✅ Success |
| datadir | 0 | static | session | Static | 0.109 | 0.0 | 0.0 | ✅ Success |
| fts_patch_env | 0 | static | function | Static | 0.000 | -0.2 | 0.0 | ✅ Success |
| get_analysis_session | 0 | static | function | Static | N/A | N/A | N/A | ❌ Fixture function not found... |
| get_it_module | 0 | static | class | Static | N/A | N/A | N/A | ❌ Fixture function not found... |
| get_it_session | 0 | static | function | Static | N/A | N/A | N/A | ❌ Fixture function not found... |
| get_it_session__sl_gpt2_analysis__initonly | 0 | dynamic | session | Real Model | 5.368 | 975.2 | 0.0 | ✅ Success |
| get_it_session_cfg | 0 | static | function | Static | N/A | N/A | N/A | ❌ Fixture function not found... |
| gpt2_ft_schedules | 0 | static | function | Static | 8.055 | 2729.1 | 0.0 | ✅ Success |
| huggingface_env | 0 | static | function | Static | 0.000 | -0.1 | 0.0 | ✅ Success |
| make_deterministic | 0 | static | session | Static | 0.000 | 0.6 | 0.0 | ✅ Success |
| make_it_module | 0 | static | class | Static | 0.000 | 0.7 | 0.0 | ✅ Success |
| mock_analysis_store | 0 | static | function | Static | 0.000 | -0.3 | 0.0 | ✅ Success |
| mock_dm | 0 | static | class | Static | 0.000 | 2.4 | 0.0 | ✅ Success |
| preserve_global_rank_variable | 0 | static | function | Static | 0.000 | -0.5 | 0.0 | ✅ Success |
| restore_env_variables | 0 | static | function | Static | 0.052 | -0.7 | 0.0 | ✅ Success |
| restore_grad_enabled_state | 0 | static | function | Static | 0.000 | 0.3 | 0.0 | ✅ Success |
| teardown_process_group | 0 | static | function | Static | 0.000 | -0.2 | 0.0 | ✅ Success |
| tmpdir_server | 0 | static | function | Static | 0.536 | -1.1 | 0.0 | ✅ Success |

## Performance Insights

### Slowest Fixtures (Top 5)

- **get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis**: 47.495s (Category: Real Model)
- **get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis**: 15.061s (Category: Real Model)
- **get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis**: 12.326s (Category: Real Model)
- **get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis**: 11.888s (Category: Real Model)
- **get_it_session__l_llama3_debug__setup**: 10.736s (Category: Real Model)

### Most Memory Intensive (Top 5)

- **get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis**: 2915.1MB (Category: Real Model)
- **gpt2_ft_schedules**: 2729.1MB (Category: Static)
- **get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis**: 1177.5MB (Category: Real Model)
- **get_it_session__l_llama3_debug__setup**: 1154.5MB (Category: Real Model)
- **get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis**: 1124.2MB (Category: Real Model)

## Optimization Recommendations

### High-Impact Optimization Targets

These fixtures are heavily used and have session scope, making them prime candidates for optimization:

- **get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis**: 17 uses, Real Model
- **get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis**: 6 uses, Real Model
- **get_it_session__core_cust__initonly**: 5 uses, Custom Model
- **get_it_session__core_cust__setup**: 8 uses, Custom Model
- **get_it_session__sl_gpt2_analysis__setup**: 4 uses, Real Model
- **get_it_session__tl_cust__setup**: 4 uses, Custom Model
- **get_it_session_cfg__core_cust**: 6 uses, Custom Model
- **get_it_session_cfg__tl_cust**: 11 uses, Custom Model

### Unused Fixtures Consider for Removal

Found 21 unused fixtures that could potentially be removed:

- **get_it_session__sl_gpt2_analysis__initonly**: Real Model, session scope
- **mock_dm**: Static, class scope
- **make_it_module**: Static, class scope
- **clean_cli_env**: Static, function scope
- **cli_test_file_env**: Static, session scope
- **cli_test_configs**: Static, session scope
- **fts_patch_env**: Static, function scope
- **gpt2_ft_schedules**: Static, function scope
- **make_deterministic**: Static, session scope
- **datadir**: Static, session scope
- ... and 11 more
