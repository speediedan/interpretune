"""
Interpretune
=====================

TODO: add description here and appropriate import defaults

"""
import os
from interpretune.__about__ import *  # noqa: F401, F403
# TODO: remove this very temporary patch once jsonargparse is updated
from typing import Any
from jsonargparse import ArgumentParser

# In PyTorch 2.0+, setting this variable will force `torch.cuda.is_available()` and `torch.cuda.device_count()`
# to use an NVML-based implementation that doesn't poison forks.
# https://github.com/pytorch/pytorch/issues/83973
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"


def patch_jsonargparse_python_3_12_8():
    def _parse_known_args_patch(self: ArgumentParser, args: Any = None, namespace: Any = None) -> tuple[Any, Any]:
        namespace, args = super(ArgumentParser, self)._parse_known_args(args, namespace,
                                                                        intermixed=False)  # type: ignore
        return namespace, args

    setattr(ArgumentParser, "_parse_known_args", _parse_known_args_patch)

patch_jsonargparse_python_3_12_8()  # Required until fix https://github.com/omni-us/jsonargparse/issues/641
