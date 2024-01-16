from typing import Tuple
from collections import defaultdict


def dummy_step(*args, **kwargs) -> None:
    ...

def nones(num_n) -> Tuple:  # to help dedup config
    return (None,) * num_n

def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)
