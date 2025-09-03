# Interpretune Fixture Benchmark Report

**Generated:** 2025-09-03 14:15:27
**GPU Available:** True
**GPU Device:** NVIDIA GeForce RTX 4090
**Total Fixtures Analyzed:** 40
**Baseline Pytest Time:** 12.219s


## Summary Statistics

> **Note:** Fully contextualized fixture profiling (i.e. using --setup-show
> with a full test suite run) is planned and will be linked to here.
> The focus of this report is to enumerate dynamic fixtures, facilitate appropriate fixture/test
> alignment and provide isolated analysis/benchmarking.

- **Total Fixtures:** 40


## Detailed Fixture Analysis

| Fixture Name | Type | Scope | Category | Init Time (s) | Memory Δ (MB) | GPU Δ (MB) | Status |
|--------------|------|-------|----------|---------------|---------------|------------|--------|
| get_it_session_cfg__core_cust | dynamic | session | Config Only | 1.0 | 0.0 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_cust | dynamic | session | Config Only | 1.0 | 0.5 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_gpt2 | dynamic | class | Config Only | 0.8 | 0.0 | 0.0 | ✅ Success |
| get_it_session_cfg__tl_cust | dynamic | session | Config Only | 0.8 | 0.0 | 0.0 | ✅ Success |
| get_it_module__core_cust__setup | dynamic | class | Custom Model | 2.6 | 7.8 | 0.0 | ✅ Success |
| get_it_session__core_cust__initonly | dynamic | session | Custom Model | 3.7 | 32.7 | 0.0 | ✅ Success |
| get_it_session__core_cust__setup | dynamic | session | Custom Model | 3.1 | 34.9 | 0.0 | ✅ Success |
| get_it_session__core_cust_force_prepare__initonly | dynamic | class | Custom Model | 3.2 | 34.5 | 0.0 | ✅ Success |
| get_it_session__core_cust_memprof__initonly | dynamic | class | Custom Model | 3.7 | 44.2 | 0.0 | ✅ Success |
| get_it_session__tl_cust__initonly | dynamic | session | Custom Model | 2.9 | 34.5 | 0.0 | ✅ Success |
| get_it_session__tl_cust__setup | dynamic | session | Custom Model | 3.0 | 37.1 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis | dynamic | class | Real Model | 48.7 | 1187.6 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis | dynamic | session | Real Model | 15.2 | 3016.2 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis | dynamic | session | Real Model | 12.9 | 1132.1 | 0.0 | ✅ Success |
| get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis | dynamic | session | Real Model | 13.0 | 1163.5 | 0.0 | ✅ Success |
| get_it_session__core_gpt2_peft__initonly | dynamic | class | Real Model | 4.5 | 274.7 | 208.5 | ✅ Success |
| get_it_session__core_gpt2_peft_seq__initonly | dynamic | class | Real Model | 4.3 | 276.1 | 208.5 | ✅ Success |
| get_it_session__l_gemma2_debug__setup | dynamic | class | Real Model | 5.8 | 209.4 | 4986.5 | ✅ Success |
| get_it_session__l_llama3_debug__setup | dynamic | class | Real Model | 10.0 | 1153.9 | 2900.5 | ✅ Success |
| get_it_session__l_sl_gpt2__initonly | dynamic | class | Real Model | 7.1 | 988.0 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2__initonly | dynamic | class | Real Model | 6.7 | 1004.7 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2_analysis__initonly | dynamic | session | Real Model | 6.7 | 985.1 | 0.0 | ✅ Success |
| get_it_session__sl_gpt2_analysis__setup | dynamic | session | Real Model | 8.6 | 1014.7 | 0.0 | ✅ Success |
| get_it_session__tl_gpt2_debug__setup | dynamic | class | Real Model | 5.2 | 683.0 | 0.0 | ✅ Success |
| clean_cli_env | static | function | Static | 0.8 | 0.3 | 0.0 | ✅ Success |
| cli_test_configs | static | session | Static | 1.0 | 0.4 | 0.0 | ✅ Success |
| cli_test_file_env | static | session | Static | 0.6 | 0.1 | 0.0 | ✅ Success |
| datadir | static | session | Static | 1.0 | 0.0 | 0.0 | ✅ Success |
| fts_patch_env | static | function | Static | 1.0 | 0.0 | 0.0 | ✅ Success |
| gpt2_ft_schedules | static | function | Static | 8.8 | 2742.2 | 0.0 | ✅ Success |
| huggingface_env | static | function | Static | 0.8 | 1.0 | 0.0 | ✅ Success |
| make_deterministic | static | session | Static | 0.7 | 0.0 | 0.0 | ✅ Success |
| make_it_module | static | class | Static | 0.9 | 0.6 | 0.0 | ✅ Success |
| mock_analysis_store | static | function | Static | 0.7 | 0.4 | 0.0 | ✅ Success |
| mock_dm | static | class | Static | 0.8 | 3.1 | 0.0 | ✅ Success |
| preserve_global_rank_variable | static | function | Static | 1.0 | 0.4 | 0.0 | ✅ Success |
| restore_env_variables | static | function | Static | 0.9 | 0.0 | 0.0 | ✅ Success |
| restore_grad_enabled_state | static | function | Static | 0.9 | 1.6 | 0.0 | ✅ Success |
| teardown_process_group | static | function | Static | 1.1 | 0.1 | 0.0 | ✅ Success |
| tmpdir_server | static | function | Static | 1.4 | 0.0 | 0.0 | ✅ Success |
