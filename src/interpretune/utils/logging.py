import os
import sys
import logging
import warnings
from typing import Optional, Callable, Any, TypeVar, Union, Dict
from typing_extensions import ParamSpec, overload
from functools import wraps
from contextlib import contextmanager

from torch.utils.collect_env import get_env_info


log = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

CUDA_MAY_BE_INIT_MSG = "Unable to patch `get_cuda_module_loading_config`, CUDA may be initialized during env logging."

# generalizing this context manager in case we need to patch other env logging functions
@contextmanager
def patch_torch_env_logging_fn(module_name: str, fn_name: str, warn_msg: str):
    try:
        orig_fn = sys.modules[module_name].__dict__.pop(fn_name, None)
        sys.modules[module_name].__dict__[fn_name] = lambda : "not inspected"
        yield
    finally:
        sys.modules[module_name].__dict__[fn_name] = orig_fn

def maybe_patched_get_env_info(module_name: str, fn_name: str, warn_msg: str):
    try:
        with patch_torch_env_logging_fn(module_name, fn_name, warn_msg):
            sys_info = get_env_info()
    except:  # noqa: E722
        # if we are unable to patch our target fn for any unexpected reason, we continue with a warn
        rank_zero_warn(warn_msg)
        sys_info = get_env_info()
    return sys_info


def collect_env_info() -> Dict:
    """Collect environmental details, logging versions of salient packages for improved reproducibility.

    Returns:
        Dict: The dictionary of environmental details
    """
    # we patch `get_cuda_module_loading_config` to avoid initializing CUDA
    sys_info = maybe_patched_get_env_info("torch.utils.collect_env", "get_cuda_module_loading_config",
                                          CUDA_MAY_BE_INIT_MSG)
    sys_dict = sys_info._asdict()
    pip_dict = {name: ver for name, ver in [p.split("==") for p in sys_info._asdict()["pip_packages"].split("\n")]}
    sys_dict["pip_packages"] = pip_dict
    return sys_dict

################################################################################
# Locally-defined (possibly framework overridding) rank-zero logging functions
# originally based upon https://bit.ly/orig_fabric_logging_utils and
# https://bit.ly/lightning_core_utils
################################################################################


def _get_rank() -> Optional[int]:
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None

@overload
def rank_zero_only(fn: Callable[P, T]) -> Callable[P, Optional[T]]: ...


@overload
def rank_zero_only(fn: Callable[P, T], default: T) -> Callable[P, T]: ...


def rank_zero_only(fn: Callable[P, T], default: Optional[T] = None) -> Callable[P, Optional[T]]:
    """Wrap a function to call internal function only in rank zero.

    Function that can be used as a decorator to enable a function/method being called only on global rank 0.
    """

    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn

# add the attribute to the function but don't overwrite if it already exists
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank() or 0)

def _debug(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)


def _info(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)

@rank_zero_only
def rank_zero_info(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit info-level messages only on global rank 0."""
    _info(*args, stacklevel=stacklevel, **kwargs)

def _warn(message: Union[str, Warning], stacklevel: int = 2, **kwargs: Any) -> None:
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit warn-level messages only on global rank 0."""
    _warn(message, stacklevel=stacklevel, **kwargs)

rank_zero_deprecation_category = DeprecationWarning

def rank_zero_deprecation(message: Union[str, Warning], stacklevel: int = 5, **kwargs: Any) -> None:
    """Emit a deprecation warning only on global rank 0."""
    category = kwargs.pop("category", rank_zero_deprecation_category)
    rank_zero_warn(message, stacklevel=stacklevel, category=category, **kwargs)
