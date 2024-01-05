import os
import random
from typing import Tuple
from collections import defaultdict

import torch

def make_deterministic(warn_only=False, fill_uninitialized_memory=True):
    # https://pytorch.org/docs/2.2/notes/randomness.html#reproducibility
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch._C._set_deterministic_fill_uninitialized_memory(fill_uninitialized_memory)
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

def dummy_step(*args, **kwargs) -> None:
    ...

def nones(num_n) -> Tuple:  # to help dedup config
    return (None,) * num_n

def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)
