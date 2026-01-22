from tests.results import TestResult, def_results, MemProfResult, load_memory_footprint_results
from functools import partial


# TODO: using result dicts in this module for now but may save/construct all TestResults from yaml files in a manner
# similar to ``mem_results`` to make programmatic management of expected results easier

l_parity_results = {
    "train_cpu_32": TestResult(
        exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, "loss", 15.520967),)
    ),
    "train_cpu_32_debug": TestResult(
        exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, "loss", 15.520967),)
    ),
    "train_cuda_32": TestResult(
        exact_results=def_results("cuda", 32, ds_cfg="train"), close_results=((0, "loss", 13.356880),)
    ),
    "train_cuda_bf16": TestResult(
        exact_results=def_results("cuda", "bf16", ds_cfg="train"), close_results=((0, "loss", 14.748956),)
    ),
    "train_cpu_bf16": TestResult(exact_results=def_results("cpu", "bf16", ds_cfg="train")),
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "predict_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=def_results("cuda", 32, ds_cfg="test")),
    "test_cuda_bf16": TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="test")),
}

cprof_results = partial(def_results, ds_cfg="train_prof")

########################################################################################################################
# NOTE [TransformerLens Profiling Parity Differences]:
# Salient differences between TransformerLens and Base Profiling Contexts
# ----------------------------------------------------------------------------------------------------------------------
# 1. The total memory allocated on cuda for a single forward (batch size 1) is ~150MB higher with TransformerLens than
#    with base. This is largely attributable to the fact we use a LM head (instead the smaller classification head) with
#    TransformerLens which involves essentially another copy of the wte (150MB) (an aside, inspecting the cuda allocator
#    state evolution, you'll see HF loads the wte embed last by whereas TransformerLens allocates the embed and unembed
#    layers prior to other layers in the state dict)
# 2. Because we are transforming the state dict on cpu with TransformerLens, it's expected that the `rss_diff`
#    with the first forward will not require substantial additional memory (3.6 GB already allocated prior to first
#    forward with TransformerLens vs 2.8GB rss with base) (one example of many why cuda memory profiling is preferred
#    and prioritized over cpu-based `rss_diff` profiling, `rss_diff` is primarily used as secondary/supplementary
#    metric except for special cases)
# 3. Note that npp (non-parameter packed) bytes is not a reliable activation memory proxy for TransformerLens as
#    currently defined. Many packed bytes that would normally be `Parameter`s are transformed by TransformerLens using
#    einsum into normal `Tensors`. As an example, `MLP.forward` will pass both the typical forward args and the
#    `Parameter`` `W.out` as regular tensors. `W.out` will be 9437184 bytes in this case (shape (1, 3072, 768)) and
#    despite the fact that the backend storage of the tensor is still shared, what is actually saved for backward is an
#    einsum-processed regular tensor version of the parameter `W.out`: see https://bit.ly/3tZsGBe
#    TODO: Consider disabling npp collection for TransformerLens unless activation checkpointing is supported and it's
#    decided that enhancing npp to be useful for TransformerLens-like transformations is sufficiently valuable
########################################################################################################################

memory_footprints = load_memory_footprint_results()


class MemResult(TestResult):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, mem_results=memory_footprints, *args, **kwargs)


profiling_results = {
    "test_l_profiling.test_cuda_32": MemResult(exact_results=cprof_results("cuda", 32, ds_cfg="test_prof")),
    "test_l_profiling.test_cuda_bf16": MemResult(exact_results=cprof_results("cuda", "bf16", ds_cfg="test_prof")),
    "test_l_profiling.train_cpu_32": MemResult(exact_results=cprof_results("cpu", 32)),
    "test_l_profiling.train_cpu_32_act": MemResult(exact_results=cprof_results("cpu", 32)),
    "test_l_profiling.train_cuda_32": MemResult(exact_results=cprof_results("cuda", 32)),
    "test_l_profiling.train_cuda_32_act": MemResult(exact_results=cprof_results("cuda", 32)),
    "test_tl_profiling.test_cpu_32": MemResult(exact_results=def_results("cpu", 32, ds_cfg="test_prof")),
    "test_tl_profiling.test_cuda_32": MemResult(exact_results=def_results("cuda", 32, ds_cfg="test_prof")),
    "test_tl_profiling.train_cpu_32": MemResult(exact_results=def_results("cpu", 32, ds_cfg="train_prof")),
    "test_tl_profiling.train_cuda_32": MemResult(exact_results=def_results("cuda", 32, ds_cfg="train_prof")),
    "test_l_profiling.test_cpu_32": MemResult(
        exact_results=cprof_results("cpu", 32, ds_cfg="test_prof"), tolerance_map={"rss_diff": (0.05, 1e08)}
    ),
    "test_l_profiling.train_cuda_bf16": MemResult(
        exact_results=cprof_results("cuda", "bf16"), tolerance_map={k: (0.1, 2e08) for k in MemProfResult.cuda_mem_keys}
    ),
    # NNsight profiling results
    "test_ns_profiling.test_cpu_32": MemResult(exact_results=def_results("cpu", 32, ds_cfg="test_prof")),
    "test_ns_profiling.test_cuda_32": MemResult(exact_results=def_results("cuda", 32, ds_cfg="test_prof")),
    "test_ns_profiling.train_cuda_32": MemResult(exact_results=def_results("cuda", 32, ds_cfg="train_prof")),
    # NOTE: NNsight Lightning profiling tests (_l variants) are temporarily skipped in test_it_ns.py
    # pending investigation of ~2x memory footprint difference compared to core variants.
    # This suggests possible duplicate model references or other memory overhead in Lightning+NNsight.
}

# TODO: using result dicts in this module for now but ultimately plan to save/construct TestResults from a yaml file
# here to make programmatic management of expected results easier

# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these core results for all supported parity test suffixes (e.g. '_l')
tl_parity_results = {
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=def_results("cuda", 32, ds_cfg="test")),
    "train_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="train")),
    "train_cuda_32": TestResult(exact_results=def_results("cuda", 32, ds_cfg="train")),
    "train_cpu_32_debug": TestResult(exact_results=def_results("cpu", 32, ds_cfg="train")),
}

# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these core results for all supported parity test suffixes (e.g. '_l')
sl_parity_results = {
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32)),
    "train_cpu_32": TestResult(exact_results=def_results("cpu", 32)),
    "train_cuda_32": TestResult(exact_results=def_results("cuda", 32)),
}

# NNsight expected results
# NNsight wraps HuggingFace models directly without weight conversion
# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these core results for all supported parity test suffixes (e.g. '_l')
ns_parity_results = {
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=def_results("cuda", 32, ds_cfg="test")),
    "train_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="train")),
    "train_cuda_32": TestResult(exact_results=def_results("cuda", 32, ds_cfg="train")),
}

# =====================================================================================
# FinetuningScheduler (FTS) Expected Results
# =====================================================================================
#
# These dictionaries validate checkpoint restoration and parameter thawing behavior
# across different TransformerLens architectures and naming conventions.
#
# RUNNING TESTS TO CAPTURE STATE:
# Run tests with IT_GLOBAL_STATE_LOG_MODE=1 to populate/update these expected results:
#   IT_GLOBAL_STATE_LOG_MODE=1 python -m pytest \
#     tests/parity_acceptance/test_it_fts.py::test_parity_fts[train_cpu_32_l_fts] -v
#
# PARAMETER COUNTS PER PHASE:
#
# Base GPT-2 (3-phase schedule):
#   Phase 0: 36 params  (blocks 9-11, 3 blocks × (4 attn + 4 mlp + 4 ln params per block))
#   Phase 1: 60 params  (adds blocks 7-8, +24 params)
#   Phase 2: 148 params (adds blocks 0-6 + embeddings, +88 params)
#
# HookedTransformer (3-phase schedule):
#   Phase 0: 48 params  (blocks 9-11, 3 blocks × (8 attn + 4 mlp + 4 ln params per block))
#   NOTE: +4 params per block vs Base GPT-2 due to split QKV projections (-2 qkv joint, +6 q/k/v split)
#   Phase 1: 80 params  (adds blocks 7-8, +32 params)
#   Phase 2: 196 params (adds blocks 0-6 + embeddings, +116 params)
#
# TransformerBridge Canonical (3-phase schedule):
#   Phase 0: 54 params  (blocks 9-11, 3 blocks × (10 attn + 4 mlp + 4 ln params per block))
#   NOTE: +2 params per block vs HookedTransformer due to thawed (but unused) joint QKV projections
#   Phase 1: 90 params  (adds blocks 7-8, +36 params)
#   Phase 2: 219 params (adds blocks 0-6 + embeddings, +129 params)
#
# TransformerBridge TL Names (3-phase schedule):
#   Phase 0: 48 params  (blocks 9-11, identical to HookedTransformer)
#   Phase 1: 80 params  (adds blocks 7-8, identical to HookedTransformer)
#   Phase 2: 197 params (adds blocks 0-6 + 2 ln_final - 1 unembed weight, +1 vs HookedTransformer)
#   NOTE: TL names mode bidirectionally translates canonical<->TL names using TL names for schedule generation and
#         orchestration. The 2 ln_final parameters are thawed because we left implicit_ln_thaw `True` and
#         we did not configure enable_compatibility_mode so the unembed weight remained tied to embed weight and
#         thus we have only an unembed bias.
#         +2 thawed ln_final params - 1 unembed weight param = net +1 param vs HookedTransformer
#
# WHY THE DIFFERENCES?
#
# Base vs HookedTransformer (+12 params in Phase 0):
#   - HookedTransformer has 8 attention params per block vs Base's 4
#   - Base: c_attn.weight, c_attn.bias, c_proj.weight, c_proj.bias (4 params)
#   - HookedTransformer: W_Q, W_K, W_V, W_O, b_Q, b_K, b_V, b_O (8 params)
#   - Split Q/K/V projections vs joint QKV (-2 qkv joint, +6 q/k/v split = net +4 per block)
#   - LayerNorm params: explicitly included in HookedTransformer schedules, explicitly included in Base schedules
#
# TransformerBridge Canonical vs HookedTransformer (+6 params in Phase 0):
#   - Bridge stores joint QKV projection (qkv.weight, qkv.bias per block)
#   - Split Q, K, V parameters are materialized as new parameters for LinearBridge modules
#   - Joint and split QKV parameters do NOT share underlying storage
#   - Result: 2 additional params per attention layer × 3 blocks = +6 params
#
# TransformerBridge TL Names vs HookedTransformer (parameter differences):
#   - TL names mode transforms canonical parameter names to TL-style names
#   - LayerNorms handled differently: explicit in HookedTransformer schedules, implicit_ln_thaw in Bridge
#   - Unembed parameters may differ (weight tying depends on enable_compatibility_mode)
#   - Using TL names provides standard interface for cross-architecture schedules
#   - CanonicalModelView (use_tl_names=False, default) allows direct canonical specification
#     but may expose architectural differences reducing schedule portability
#
# CHECKPOINT RESTORATION (with additive penalty divergence):
#   - Divergence starts at epoch 2 (first epoch of phase 1)
#   - Best checkpoint remains at depth 0 (phase 0) for all architectures
#   - All tests correctly restore to best checkpoint from phase 0 in epochs 3-4
#
# See docs/fts_transformerlens_integration.md for detailed documentation
# =====================================================================================

# Base GPT-2 FTS Results (3-phase schedule: 36 → 60 → 148 params)
l_fts_callback_results = {
    0: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 0,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 0,
        "curr_thawed_params": 36,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    1: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 1,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 36,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    2: {
        "curr_depth": 1,
        "depth_remaining": 1,
        "ft_epoch": 2,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 60,
        "optim_pg_len": 2,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    3: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 3,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 148,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    4: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 4,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 148,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
}

# HookedTransformer FTS Results (3-phase schedule: 48 → 80 → 196 params)
# TL-style naming with implicit LayerNorm thawing via regex wildcards
l_tl_ht_fts_multiphase_callback_results = {
    0: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 0,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 0,
        "curr_thawed_params": 48,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    1: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 1,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 48,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    2: {
        "curr_depth": 1,
        "depth_remaining": 1,
        "ft_epoch": 2,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 80,
        "optim_pg_len": 2,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    3: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 3,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 196,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    4: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 4,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 196,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
}

# TransformerBridge Canonical FTS Results (3-phase schedule: 54 → 90 → 219 params)
# Canonical HF naming with _original_component wrappers, +2 params/block vs HookedTransformer due to
# joint QKV projections
l_tl_bridge_fts_multiphase_callback_results = {
    0: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 0,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 0,
        "curr_thawed_params": 54,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    1: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 1,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 54,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    2: {
        "curr_depth": 1,
        "depth_remaining": 1,
        "ft_epoch": 2,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 90,
        "optim_pg_len": 2,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    3: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 3,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 219,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    4: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 4,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 219,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
}

# TransformerBridge TL Names FTS Results (3-phase schedule: 48 → 80 → 197 params)
# TL-style naming mode (blocks.9.attn.W_Q) with 1:1 mapping to canonical params
# Joint QKV params excluded from TL name mapping, implicitly thawed via regex wildcard patterns
l_tl_bridge_fts_tl_names_multiphase_callback_results = {
    0: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 0,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 0,
        "curr_thawed_params": 48,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    1: {
        "curr_depth": 0,
        "depth_remaining": 2,
        "ft_epoch": 1,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 48,
        "optim_pg_len": 1,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    2: {
        "curr_depth": 1,
        "depth_remaining": 1,
        "ft_epoch": 2,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 80,
        "optim_pg_len": 2,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    3: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 3,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 197,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    4: {
        "curr_depth": 2,
        "depth_remaining": 0,
        "ft_epoch": 4,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 197,
        "optim_pg_len": 3,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
}


# TODO: using result dicts in this module for now but ultimately plan to save/construct TestResults from a yaml file
# here to make programmatic management of expected results easier

# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these results for all supported parity test suffixes (e.g. '_l_fts')
fts_parity_results = {
    "train_cpu_32_l_fts": TestResult(
        exact_results=def_results("cpu", 32, ds_cfg="train"), callback_results=l_fts_callback_results
    ),
    "train_cuda_32_l_fts": TestResult(
        exact_results=def_results("cuda", 32, ds_cfg="train"), callback_results=l_fts_callback_results
    ),
    "train_cpu_32_l_tl_ht_fts": TestResult(
        exact_results=def_results("cpu", 32, ds_cfg="train"), callback_results=l_tl_ht_fts_multiphase_callback_results
    ),
    "train_cuda_32_l_tl_ht_fts": TestResult(
        exact_results=def_results("cuda", 32, ds_cfg="train"), callback_results=l_tl_ht_fts_multiphase_callback_results
    ),
    "train_cuda_32_l_tl_bridge_fts": TestResult(
        exact_results=def_results("cuda", 32, ds_cfg="train"),
        callback_results=l_tl_bridge_fts_multiphase_callback_results,
    ),
    "train_cuda_32_l_tl_bridge_tl_names_fts": TestResult(
        exact_results=def_results("cuda", 32, ds_cfg="train"),
        callback_results=l_tl_bridge_fts_tl_names_multiphase_callback_results,
    ),
}
