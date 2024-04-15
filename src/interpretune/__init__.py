"""
Interpretune
=====================

TODO: add description here and appropriate import defaults

"""
import os
from interpretune.__about__ import *  # noqa: F401, F403
# from finetuning_scheduler.fts import FinetuningScheduler

# from finetuning_scheduler.fts_supporters import (  # isort: skip
#     FTSState,
#     FTSCheckpoint,
#     FTSEarlyStopping,
#     ScheduleImplMixin,
#     ScheduleParsingMixin,
#     CallbackDepMixin,
#     CallbackResolverMixin,
# )
# In PyTorch 2.0+, setting this variable will force `torch.cuda.is_available()` and `torch.cuda.device_count()`
# to use an NVML-based implementation that doesn't poison forks.
# https://github.com/pytorch/pytorch/issues/83973
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# __all__ = [
#     "FTSState",
#     "FTSCheckpoint",
#     "FTSEarlyStopping",
#     "ScheduleImplMixin",
#     "ScheduleParsingMixin",
#     "CallbackDepMixin",
#     "CallbackResolverMixin",
#     "FinetuningScheduler",
# ]
