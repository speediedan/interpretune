# Interpretune Fixture Benchmark Report

**Generated:** 2026-03-05 12:26:52
**GPU Available:** True
**GPU Device:** NVIDIA GeForce RTX 4090
**Total Fixtures Analyzed:** 58
**Baseline Pytest Time:** 15.969s


## Summary Statistics

> **Note:** Fully contextualized fixture profiling (i.e. using --setup-show
> with a full test suite run) is planned and will be linked to here.
> The focus of this report is to enumerate dynamic fixtures, facilitate appropriate fixture/test
> alignment and provide isolated analysis/benchmarking.

- **Total Fixtures:** 58


## Detailed Fixture Analysis

| Fixture Name | Type | Scope | Category | Init Time (s) | Memory Δ (MB) | GPU Δ (MB) | Status |
|--------------|------|-------|----------|---------------|---------------|------------|--------|
| get_it_session_cfg__core_cust | dynamic | session | Config Only | 0.5 | 0.4 | 0.0 | ✅ Success |
| get_it_session_cfg__ns_gpt2 | dynamic | class | Config Only | 0.7 | 0.0 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_br_gpt2 | dynamic | class | Config Only | 0.4 | 0.7 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_cust | dynamic | session | Config Only | 0.5 | 0.0 | 0.0 | ✅ Success |
| get_it_session_cfg__sl_ht_gpt2 | dynamic | class | Config Only | 0.7 | 0.0 | 0.0 | ✅ Success |
| get_it_session_cfg__tl_cust | dynamic | session | Config Only | 0.5 | 0.3 | 0.0 | ✅ Success |
| get_it_module__core_cust__setup | dynamic | class | Custom Model | 3.7 | 7.9 | 0.0 | ✅ Success |
| get_it_session__core_cust__initonly | dynamic | session | Custom Model | 3.4 | 50.7 | 0.0 | ✅ Success |
| get_it_session__core_cust__setup | dynamic | session | Custom Model | 21.3 | 60.3 | 0.0 | ✅ Success |
| get_it_session__core_cust_force_prepare__initonly | dynamic | class | Custom Model | 4.4 | 47.9 | 0.0 | ✅ Success |
| get_it_session__core_cust_memprof__initonly | dynamic | class | Custom Model | 4.1 | 58.5 | 0.0 | ✅ Success |
| get_it_session__l_gemma3_debug__setup | dynamic | class | Custom Model | 9.1 | 512.3 | 1907.2 | ✅ Success |
| get_it_session__tl_cust__initonly | dynamic | session | Custom Model | 3.5 | 49.0 | 0.0 | ✅ Success |
| get_it_session__tl_cust__setup | dynamic | session | Custom Model | 3.4 | 51.1 | 0.0 | ✅ Success |
| get_analysis_session__sl_br_gpt2_logit_diffs_attr_ablation__initonly_runanalysis | dynamic | class | Real Model | N/A | N/A | N/A | ❌ Timeout (>2min)... |
| get_analysis_session__sl_br_gpt2_logit_diffs_attr_grad__initonly_runanalysis | dynamic | session | Real Model | 32.7 | 2792.3 | 0.0 | ✅ Success |
| get_analysis_session__sl_br_gpt2_logit_diffs_base__initonly_runanalysis | dynamic | session | Real Model | 33.3 | 1066.8 | 0.0 | ✅ Success |
| get_analysis_session__sl_br_gpt2_logit_diffs_sae__initonly_runanalysis | dynamic | session | Real Model | 37.2 | 1170.5 | 0.0 | ✅ Success |
| get_analysis_session__sl_ht_gpt2_logit_diffs_attr_ablation__initonly_runanalysis | dynamic | class | Real Model | 93.7 | 1196.6 | 0.0 | ✅ Success |
| get_analysis_session__sl_ht_gpt2_logit_diffs_attr_grad__initonly_runanalysis | dynamic | session | Real Model | 34.9 | 3191.9 | 0.0 | ✅ Success |
| get_analysis_session__sl_ht_gpt2_logit_diffs_base__initonly_runanalysis | dynamic | session | Real Model | 44.7 | 1141.5 | 0.0 | ✅ Success |
| get_analysis_session__sl_ht_gpt2_logit_diffs_sae__initonly_runanalysis | dynamic | session | Real Model | 42.1 | 1179.4 | 0.0 | ✅ Success |
| get_analysis_session__sl_ns_gpt2_logit_diffs_attr_ablation__initonly_runanalysis | dynamic | class | Real Model | N/A | N/A | N/A | ❌ Timeout (>2min)... |
| get_analysis_session__sl_ns_gpt2_logit_diffs_attr_grad__initonly_runanalysis | dynamic | session | Real Model | 51.7 | 2545.4 | 0.0 | ✅ Success |
| get_analysis_session__sl_ns_gpt2_logit_diffs_base__initonly_runanalysis | dynamic | session | Real Model | 37.3 | 1083.2 | 0.0 | ✅ Success |
| get_analysis_session__sl_ns_gpt2_logit_diffs_sae__initonly_runanalysis | dynamic | session | Real Model | 40.4 | 1131.7 | 0.0 | ✅ Success |
| get_it_session__core_gpt2_peft__initonly | dynamic | class | Real Model | 5.2 | 286.6 | 202.1 | ✅ Success |
| get_it_session__core_gpt2_peft_seq__initonly | dynamic | class | Real Model | 5.0 | 290.8 | 201.8 | ✅ Success |
| get_it_session__l_llama3_debug__setup | dynamic | class | Real Model | 10.7 | 885.4 | 2900.5 | ✅ Success |
| get_it_session__l_ns_gpt2__setup | dynamic | class | Real Model | 7.7 | 82.7 | 0.0 | ✅ Success |
| get_it_session__l_sl_ht_gpt2__initonly | dynamic | class | Real Model | 10.0 | 1020.7 | 0.0 | ✅ Success |
| get_it_session__l_tl_bridge_gpt2__setup | dynamic | module | Real Model | 4.6 | 54.3 | 0.0 | ✅ Success |
| get_it_session__l_tl_bridge_gpt2_processed__setup | dynamic | module | Real Model | 5.7 | 1158.5 | 0.0 | ✅ Success |
| get_it_session__ns_gpt2__setup | dynamic | class | Real Model | 5.2 | 81.9 | 0.0 | ✅ Success |
| get_it_session__ns_gpt2_debug__setup | dynamic | class | Real Model | 4.6 | 81.6 | 0.0 | ✅ Success |
| get_it_session__sl_br_gpt2__initonly | dynamic | class | Real Model | 8.7 | 398.8 | 0.0 | ✅ Success |
| get_it_session__sl_ht_gpt2__initonly | dynamic | class | Real Model | 9.4 | 1036.3 | 0.0 | ✅ Success |
| get_it_session__sl_ht_gpt2_analysis__setup | dynamic | session | Real Model | 24.0 | 1024.0 | 0.0 | ✅ Success |
| get_it_session__sl_ns_gpt2__initonly | dynamic | class | Real Model | 9.3 | 420.9 | 0.0 | ✅ Success |
| get_it_session__tl_gpt2_debug__setup | dynamic | class | Real Model | 4.2 | 55.2 | 0.0 | ✅ Success |
| clean_cli_env | static | function | Static | 0.5 | 0.6 | 0.0 | ✅ Success |
| cleanup_cuda | static | function | Static | 1.0 | 0.0 | 0.0 | ✅ Success |
| cleanup_memory | static | function | Static | 1.0 | 0.0 | 0.0 | ✅ Success |
| cli_test_configs | static | session | Static | 0.5 | 0.3 | 0.0 | ✅ Success |
| cli_test_file_env | static | session | Static | 0.6 | 0.0 | 0.0 | ✅ Success |
| datadir | static | session | Static | 0.5 | 0.4 | 0.0 | ✅ Success |
| get_ft_schedule | static | function | Static | N/A | N/A | N/A | ❌ Fixture function not found... |
| huggingface_env | static | function | Static | 0.7 | 0.5 | 0.0 | ✅ Success |
| make_deterministic | static | function | Static | 0.3 | 1.0 | 0.0 | ✅ Success |
| make_deterministic_session | static | session | Static | 0.3 | 1.6 | 0.0 | ✅ Success |
| make_it_module | static | class | Static | 0.6 | 0.0 | 0.0 | ✅ Success |
| mock_analysis_store | static | function | Static | 0.6 | 0.4 | 0.0 | ✅ Success |
| mock_dm | static | class | Static | 1.3 | 2.5 | 0.0 | ✅ Success |
| preserve_global_rank_variable | static | function | Static | 0.7 | 0.2 | 0.0 | ✅ Success |
| restore_env_variables | static | function | Static | 0.7 | 0.0 | 0.0 | ✅ Success |
| restore_grad_enabled_state | static | function | Static | 0.7 | 0.0 | 0.0 | ✅ Success |
| teardown_process_group | static | function | Static | 0.8 | 0.4 | 0.0 | ✅ Success |
| tmpdir_server | static | function | Static | 1.3 | 0.0 | 0.0 | ✅ Success |
