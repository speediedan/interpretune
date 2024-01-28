from tests.configuration import TestResult, def_results
from functools import partial

tl_results = partial(def_results, dataset_type="tl")

# TODO: using result dicts in this module for now but ultimately plan to save/construct TestResults from a yaml file
# here to make programmatic management of expected results easier

# we always check for basic exact match on device type and precision as well
# note our result mapping function uses these core results for all supported parity test suffixes (e.g. '_l')
tl_parity_results = {
    "test_cpu_32": TestResult(exact_results=tl_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=tl_results("cuda", 32, ds_cfg="test")),
    "train_cpu_32": TestResult(exact_results=tl_results("cpu", 32, ds_cfg="train")),
    "train_cuda_32": TestResult(exact_results=tl_results("cuda", 32, ds_cfg="train")),
    "train_cpu_32_debug": TestResult(exact_results=tl_results("cpu", 32, ds_cfg="train")),
}

########################################################################################################################
# Salient differences between TransformerLens and Base Profiling Contexts
# ----------------------------------------------------------------------------------------------------------------------
# 1. The total memory allocated on cuda for a single forward (batch size 1) is ~150MB higher with TransformerLens than
#    with base. This is largely attributable to the fact we use a LM head (instead the smaller classification head) with
#    TransformerLens which involves essentially another copy of the wte (150MB) (an aside, inspecting the cuda allocator
#    state evolution, you'll see HF loads thewte embed last by whereas TransformerLens allocates the embed and unembed
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
    TestResult(exact_results=def_results("cuda", 32, ds_cfg="test_prof"),
               mem_results=("test", "cuda", (699942912, 771789824, 834666496, 0))), # see note #1 above
    "train_cpu_32":
    TestResult(exact_results=def_results("cpu", 32),
               mem_results=("train", "cpu", (70086656, 693365760))),  # see note #3 above
    "train_cuda_32":
    TestResult(exact_results=def_results("cuda", 32),
               mem_results=("train", "cuda", (2361786368, 3353252352, 3619684352, 693365760))),
}
