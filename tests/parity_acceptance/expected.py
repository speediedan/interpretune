from tests.results import TestResult, def_results, MemProfResult
from functools import partial


# TODO: using result dicts in this module for now but ultimately plan to save/construct TestResults from a yaml file
# here to make programmatic management of expected results easier


l_parity_results = {
    "train_cpu_32":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, 'loss', 15.520967),)),
    "train_cpu_32_debug":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, 'loss', 15.520967),)),
    "train_cuda_32":
    TestResult(exact_results=def_results("cuda", 32, ds_cfg="train"), close_results=((0, 'loss', 13.356880),)),
    "train_cuda_bf16":
    TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="train"), close_results=((0, 'loss', 13.402528),)),
    "train_cpu_bf16": TestResult(exact_results=def_results("cpu", "bf16", ds_cfg="train")),
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32,ds_cfg="test")),
    "predict_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=def_results("cuda", 32, ds_cfg="test")),
    "test_cuda_bf16": TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="test")),
}

cprof_results = partial(def_results, ds_cfg="train_prof")

l_profiling_parity_results = {
    "test_cpu_32":
    TestResult(exact_results=cprof_results("cpu", 32, ds_cfg="test_prof"), mem_results=("test", "cpu", (633864192,)),
               tolerance_map={"rss_diff": (0.05, 1e08)}),  # lightning ver requires a bit more
    "test_cuda_32":
    TestResult(exact_results=cprof_results("cuda", 32, ds_cfg="test_prof"),
               mem_results=("test", "cuda", (544706048, 641185280, 708837376, 0))),
    "test_cuda_bf16":
    TestResult(exact_results=cprof_results("cuda", "bf16", ds_cfg="test_prof"),
               mem_results=("test", "cuda", (301455872, 347033088, 371195904, 0))),
    "train_cpu_32":
    TestResult(exact_results=cprof_results("cpu", 32),
               mem_results=("train", "cpu", (561709056, 177550460))),
    "train_cpu_32_act":
    TestResult(exact_results=cprof_results("cpu", 32),
               mem_results=("train", "cpu", (648609792, 161311584))),
    "train_cuda_32":
    TestResult(exact_results=cprof_results("cuda", 32),
               mem_results=("train", "cuda", (1746485248, 2576384000, 2965372928, 309007392))),
    "train_cuda_32_act":
    TestResult(exact_results=cprof_results("cuda", 32),
               mem_results=("train", "cuda", (1581433344, 2573807616, 2923429888, 161007456))),
    "train_cuda_bf16":
    TestResult(exact_results=cprof_results("cuda", "bf16"),
               mem_results=("train", "cuda", (956976640, 1363192832, 1507852288, 190250016)),
               tolerance_map={k: (0.1, 2e08) for k in MemProfResult.cuda_mem_keys}),
               # TODO: as current allocated memory for Lightning is about 217 MB higher than core and npp is about 80 MB
               # lower than core, investigate the precise source of these divergences (not presently viewed as highest
               # priority given other values are nearly identical to core)
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

########################################################################################################################
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
tl_profiling_parity_results = {
    "test_cpu_32":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="test_prof"),
               mem_results=("test", "cpu", (71303168,))),  # see note #2 above
    "test_cuda_32":
    TestResult(exact_results=def_results("cuda", 32, ds_cfg="test_prof"),  # see note #1 above
               mem_results=("test", "cuda", (699944960, 750239232, 815792128, 0))),
    "train_cpu_32":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="train_prof"),
               mem_results=("train", "cpu", (70086656, 693365760))),  # see note #3 above
    "train_cuda_32":
    TestResult(exact_results=def_results("cuda", 32, ds_cfg="train_prof"),
               mem_results=("train", "cuda", (2332741632, 3352547840, 3619684352, 312485028))),
}

l_fts_callback_results = {
    0: {
        "curr_depth": 0,
        "depth_remaining": 1,
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
        "depth_remaining": 1,
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
        "depth_remaining": 0,
        "ft_epoch": 2,
        "current_ckpt_depth": 0,
        "best_ckpt_depth": 0,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 148,
        "optim_pg_len": 2,
        "ckpt_cback_current_ckpt_depth": 0,
        "ckpt_cback_best_ckpt_depth": 0,
    },
    3: {
        "curr_depth": 1,
        "depth_remaining": 0,
        "ft_epoch": 3,
        "current_ckpt_depth": 1,
        "best_ckpt_depth": 1,
        "best_ckpt_pgs_len": 1,
        "curr_thawed_params": 148,
        "optim_pg_len": 2,
        "ckpt_cback_current_ckpt_depth": 1,
        "ckpt_cback_best_ckpt_depth": 1,
    },
}

# currently these are the same, but they could well be different in future tests with enriched TL-specific functionality
l_tl_fts_callback_results = l_fts_callback_results

# TODO: using result dicts in this module for now but ultimately plan to save/construct TestResults from a yaml file
# here to make programmatic management of expected results easier

# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these results for all supported parity test suffixes (e.g. '_l_fts')
fts_parity_results = {
    "train_cpu_32_l_fts": TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"),
                                     callback_results=l_fts_callback_results),
    "train_cpu_32_l_tl_fts": TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"),
                               callback_results=l_tl_fts_callback_results),
    "train_cuda_32_l_fts": TestResult(exact_results=def_results("cuda", 32, ds_cfg="train"),
                                callback_results=l_fts_callback_results),
    "train_cuda_32_l_tl_fts": TestResult(exact_results=def_results("cuda", 32, ds_cfg="train"),
                                callback_results=l_tl_fts_callback_results),
}
