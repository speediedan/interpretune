import os
import sys
import logging
import warnings
from collections import namedtuple
from typing import Optional, Callable, Any, TypeVar, Union, Dict, Type
from typing_extensions import ParamSpec, overload
from functools import wraps
from platform import python_version
from pathlib import Path

import torch
from torch.utils import collect_env

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE

# for now, if lightning is not available, we assume rank zero is 0 since raw pytorch distributed logic hasn't been
# implemented yet

if _LIGHTNING_AVAILABLE:
    from lightning.fabric.utilities.rank_zero import _get_rank
else:
    _get_rank = lambda: None

_default_format_warning = warnings.formatwarning

# mostly adapted from lightning_utilities.core.rank_zero

log = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


# adapted from lightning.fabric.utilities.warnings
def _custom_format_warning(
    message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str] = None
) -> str:
    """Custom formatting that avoids an extra line in case warnings are emitted from the `rank_zero`-functions."""
    if _is_path_in_interpretune(Path(filename)):
        # The warning originates from the Interpretune package
        return f"{filename}:{lineno}: {message}\n"
    return _default_format_warning(message, category, filename, lineno, line)

def _is_path_in_interpretune(path: Path) -> bool:
    """Naive check whether the path looks like a path from the Interpretune package."""
    return "interpretune" in str(path.absolute())

warnings.formatwarning = _custom_format_warning

@overload
def rank_zero_only(fn: Callable[P, T]) -> Callable[P, Optional[T]]:
    ...


@overload
def rank_zero_only(fn: Callable[P, T], default: T) -> Callable[P, T]:
    ...


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


def _debug(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)


def _info(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
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


def rank_prefixed_message(message: str, rank: Optional[int]) -> str:
    """Add a prefix with the rank to a message."""
    if rank is not None:
        # specify the rank of the process being logged
        return f"[rank: {rank}] {message}"
    return message


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank() or 0)

# override PyTorch default, extending it to capture additional salient packages for reproducability
# https://github.com/pytorch/pytorch/blob/7c2489bdae5a96dc122c3bb7b42c18528bcfdc86/torch/utils/collect_env.py#L271
def get_pip_packages(run_lambda):
    """Returns `pip list` output.

    Note: will also find conda-installed pytorch
    and numpy packages.
    """
    # People generally have `pip` as `pip` or `pip3`
    # But here it is incoved as `python -mpip`
    def run_with_pip(pip):
        if collect_env.get_platform() == "win32":
            system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
            findstr_cmd = os.path.join(system_root, "System32", "findstr")
            grep_cmd = rf'{findstr_cmd} /R "numpy torch mypy transformers datasets"'
        else:
            grep_cmd = r'grep "torch\|numpy\|mypy\|transformers\|datasets"'
        return collect_env.run_and_read_all(run_lambda, pip + " list --format=freeze | " + grep_cmd)

    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    out = run_with_pip(sys.executable + " -mpip")

    return pip_version, out


def get_env_info():
    run_lambda = collect_env.run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if collect_env.TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        if not hasattr(torch.version, "hip") or torch.version.hip is None:  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
        else:  # HIP version
            cfg = torch._C._show_config().split("\n")
            hip_runtime_version = [s.rsplit(None, 1)[-1] for s in cfg if "HIP Runtime" in s][0]
            miopen_runtime_version = [s.rsplit(None, 1)[-1] for s in cfg if "MIOpen" in s][0]
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = "N/A"
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"

    sys_version = sys.version.replace("\n", " ")

    systemenv_kwargs = {
        "torch_version": version_str,
        "is_debug_build": debug_mode_str,
        "python_version": f"{sys_version} ({sys.maxsize.bit_length() + 1}-bit runtime)",
        "python_platform": collect_env.get_python_platform(),
        "is_cuda_available": cuda_available_str,
        "cuda_compiled_version": cuda_version_str,
        "cuda_runtime_version": collect_env.get_running_cuda_version(run_lambda),
        "nvidia_gpu_models": collect_env.get_gpu_info(run_lambda),
        "nvidia_driver_version": collect_env.get_nvidia_driver_version(run_lambda),
        "cudnn_version": collect_env.get_cudnn_version(run_lambda),
        "hip_compiled_version": hip_compiled_version,
        "hip_runtime_version": hip_runtime_version,
        "miopen_runtime_version": miopen_runtime_version,
        "pip_version": pip_version,
        "pip_packages": pip_list_output,
        "conda_packages": collect_env.get_conda_packages(run_lambda),
        "os": collect_env.get_os(run_lambda),
        "libc_version": collect_env.get_libc_version(),
        "gcc_version": collect_env.get_gcc_version(run_lambda),
        "clang_version": collect_env.get_clang_version(run_lambda),
        "cmake_version": collect_env.get_cmake_version(run_lambda),
        "caching_allocator_config": collect_env.get_cachingallocator_config(),
        "is_xnnpack_available": collect_env.is_xnnpack_available(),
        "cpu_info": collect_env.get_cpu_info(run_lambda),
    }
    # get_cuda_module_loading_config() initializes CUDA which we want to avoid so we bypass this inspection
    systemenv_kwargs["cuda_module_loading"] = "not inspected"
    return collect_env.SystemEnv(**systemenv_kwargs)


def collect_env_info() -> Dict:
    """Collect environmental details, logging versions of salient packages for improved reproducibility.

    Returns:
        Dict: The dictionary of environmental details
    """
    _ = namedtuple(
        "SystemEnv",
        [
            "torch_version",
            "is_debug_build",
            "cuda_compiled_version",
            "gcc_version",
            "clang_version",
            "cmake_version",
            "os",
            "libc_version",
            "python_version",
            "python_platform",
            "is_cuda_available",
            "cuda_runtime_version",
            "nvidia_driver_version",
            "nvidia_gpu_models",
            "cudnn_version",
            "pip_version",  # 'pip' or 'pip3'
            "pip_packages",
            "conda_packages",
            "hip_compiled_version",
            "hip_runtime_version",
            "miopen_runtime_version",
            "caching_allocator_config",
        ],
    )
    collect_env.get_pip_packages = get_pip_packages
    sys_info = get_env_info()
    sys_dict = sys_info._asdict()
    pip_dict = {name: ver for name, ver in [p.split("==") for p in sys_info._asdict()["pip_packages"].split("\n")]}
    sys_dict["pip_packages"] = pip_dict
    return sys_dict
