from tests.configuration import TestResult, def_results
from tests.parity_acceptance.base.cfg_aliases import MemProfResult
from functools import partial

# TODO: using result dicts in this module for now but ultimately plan to save/construct TestResults from a yaml file
# here to make programmatic management of expected results easier


basic_parity_results = {
    "train_cpu_32":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, 'loss', 15.520967),)),
    "train_cpu_32_debug":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, 'loss', 15.520967),)),
    "train_cuda_32":
    TestResult(exact_results=def_results("cuda", 32, ds_cfg="train"), close_results=((0, 'loss', 13.356880),)),
    "train_cuda_bf16":
    TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="train"), close_results=((0, 'loss', 13.402528),)),
    "train_cpu_bf16": TestResult(exact_results=def_results("cpu", "bf16", ds_cfg="train")),
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=def_results("cuda", 32, dataset_type="gpt2", ds_cfg="test")),
    "test_cuda_bf16": TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="test")),
}

cprof_results = partial(def_results, dataset_type="gpt2")

profiling_parity_results = {
    "test_cpu_32":
    TestResult(exact_results=cprof_results("cpu", 32, ds_cfg="test_prof"), mem_results=("test", "cpu", (611057664,)),
               tolerance_map={"rss_diff": (0.05, 1e08)}),  # lightning ver requires a bit more
    "test_cuda_32":
    TestResult(exact_results=cprof_results("cuda", 32, ds_cfg="test_prof"),
               mem_results=("test", "cuda", (544712192, 619739648, 673185792, 0))),
    "test_cuda_bf16":
    TestResult(exact_results=cprof_results("cuda", "bf16", ds_cfg="test_prof"),
               mem_results=("test", "cuda", (301458944, 336683008, 352321536, 0))),
    "train_cpu_32":
    TestResult(exact_results=cprof_results("cpu", 32),
               mem_results=("train", "cpu", (561709056, 177550460))),
    "train_cpu_32_act":
    TestResult(exact_results=cprof_results("cpu", 32),
               mem_results=("train", "cpu", (646250496, 160481568))),
    "train_cuda_32":
    TestResult(exact_results=cprof_results("cuda", 32),
               mem_results=("train", "cuda", (1768124416, 2576384000, 2988441600, 316803408))),
    "train_cuda_32_act":
    TestResult(exact_results=cprof_results("cuda", 32),
               mem_results=("train", "cuda", (1582231040, 2574886912, 2925527040, 160177440))),
    "train_cuda_bf16":
    TestResult(exact_results=cprof_results("cuda", "bf16"),
               mem_results=("train", "cuda", (974026240, 1363192832, 1524629504, 200479032)),
               tolerance_map={k: (0.1, 2e08) for k in MemProfResult.cuda_mem_keys}),
               # TODO: as current allocated memory for Lightning is about 217 MB higher than core and npp is about 80 MB
               # lower than core, investigate the precise source of these divergences (not presently viewed as highest
               # priority given other values are nearly identical to core)
}
