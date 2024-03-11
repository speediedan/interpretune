from tests.configuration import TestResult, def_results
from tests.base.cfg_aliases import MemProfResult

# TODO: using result dicts in this module for now but ultimately plan to save/construct TestResults from a yaml file
# here to make programmatic management of expected results easier

basic_parity_results = {
    "train_cpu_32":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, 'loss', 8.9896163),)),
    "train_cpu_32_debug":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="train"), close_results=((0, 'loss', 8.9896163),)),
    "train_cuda_32":
    TestResult(exact_results=def_results("cuda", 32, ds_cfg="train"), close_results=((0, 'loss', 7.0788559),)),
    "train_cuda_bf16":
    TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="train"), close_results=((0, 'loss', 5.1562681),)),
    "train_cpu_bf16": TestResult(exact_results=def_results("cpu", "bf16", ds_cfg="train")),
    "test_cpu_32": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "test_cpu_32_tl": TestResult(exact_results=def_results("cpu", 32, ds_cfg="test")),
    "test_cuda_32": TestResult(exact_results=def_results("cuda", 32, ds_cfg="test")),
    "test_cuda_bf16": TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="test")),
}

profiling_parity_results = {
    "test_cpu_32":
    TestResult(exact_results=def_results("cpu", 32, ds_cfg="test_prof"), mem_results=("test", "cpu", (378863616,)),
               tolerance_map={"rss_diff": (0.05, 1e08)}),  # lightning ver requires a bit more
    "test_cuda_32":
    TestResult(exact_results=def_results("cuda", 32, ds_cfg="test_prof"),
               mem_results=("test", "cuda", (544712192, 556677120, 608174080, 0))),
    "test_cuda_bf16":
    TestResult(exact_results=def_results("cuda", "bf16", ds_cfg="test_prof"),
               mem_results=("test", "cuda", (301458944, 309889024, 318767104, 0))),
    "train_cpu_32":
    TestResult(exact_results=def_results("cpu", 32),
               mem_results=("train", "cpu", (561709056, 177550460))),
    "train_cpu_32_act":
    TestResult(exact_results=def_results("cpu", 32),
               mem_results=("train", "cpu", (396951552, 6098252))),
    "train_cuda_32":
    TestResult(exact_results=def_results("cuda", 32),
               mem_results=("train", "cuda", (1767844352, 2575136768, 2841640960, 162420092))),
    "train_cuda_32_act":
    TestResult(exact_results=def_results("cuda", 32),
               mem_results=("train", "cuda", (1582231040, 2574886912, 2778726400, 5794124))),
    "train_cuda_bf16":
    TestResult(exact_results=def_results("cuda", "bf16"),
               mem_results=("train", "cuda", (971867136, 1362683904, 1438646272, 123287392)),
               tolerance_map={k: (0.1, 2e08) for k in MemProfResult.cuda_mem_keys}),
               # TODO: as current allocated memory for Lightning is about 217 MB higher than core and npp is about 80 MB
               # lower than core, investigate the precise source of these divergences (not presently viewed as highest
               # priority given other values are nearly identical to core)
}
