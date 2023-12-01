from typing import Any, Union, Optional, Dict, Tuple, Callable
import importlib
from functools import lru_cache
from importlib.util import find_spec
import operator
import platform
import sys
import torch
import pkg_resources
from packaging.version import Version


class MisconfigurationException(Exception):
    """Exception used to inform users of misuse with interpretune."""

@lru_cache()
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False

@lru_cache()
def module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> module_available('os')
    True
    >>> module_available('os.bla')
    False
    >>> module_available('bla.bla')
    False
    """
    module_names = module_path.split(".")
    if not package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


@lru_cache(1)
def num_cuda_devices() -> int:
    """Returns the number of available CUDA devices.

    Since we require torch > 2.0, we do not need to explicitly use the nvml-based check.
    """
    return torch.cuda.device_count()

def _import_class(class_path: str) -> Any:
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    return getattr(module, class_name)


# adapted from lightning_utilities/core/imports.py
def it_compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.

    >>> compare_version("torch", operator.ge, "0.1")
    True
    >>> compare_version("does_not_exist", operator.ge, "0.0")
    False
    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, pkg_resources.DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution(package).version)
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


_DOTENV_AVAILABLE = module_available("dotenv")
_LIGHTNING_AVAILABLE = package_available("lightning")
_FTS_AVAILABLE = package_available("finetuning_scheduler")
_BNB_AVAILABLE = package_available("bitsandbytes")

if _LIGHTNING_AVAILABLE:
    from lightning_utilities.core.imports import compare_version
else:
    compare_version = it_compare_version

# adapted from fabric/utilities/imports.py
_IS_WINDOWS = platform.system() == "Windows"

# There are two types of interactive mode we detect
# 1. The interactive Python shell: https://stackoverflow.com/a/64523765
# 2. The inspection mode via `python -i`: https://stackoverflow.com/a/6879085/1162383
_IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)


_TORCH_GREATER_EQUAL_2_0 = compare_version("torch", operator.ge, "2.0.0")
_TORCH_GREATER_EQUAL_2_1 = compare_version("torch", operator.ge, "2.1.0", use_base_version=True)
_TORCH_GREATER_EQUAL_2_2 = compare_version("torch", operator.ge, "2.2.0", use_base_version=True)
_TORCH_EQUAL_2_0 = _TORCH_GREATER_EQUAL_2_0 and not _TORCH_GREATER_EQUAL_2_1

_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)



def instantiate_class(init: Dict[str, Any], args: Optional[Union[Any, Tuple[Any, ...]]] = None) -> Any:
        """Instantiates a class with the given args and init. Accepts class definitions with a "class_path".

        Args:
            init: Dict of the form {"class_path":..., "init_args":...}.
            args: Positional arguments required for instantiation.

        Returns:
            The instantiated class object.
        """
        class_module, class_name, args_class = None, None, None
        shortcircuit_local = False
        kwargs = init.get("init_args", {})
        class_path = init.get("class_path", None)
        if args and not isinstance(args, tuple):
            args = (args,)
        if class_path:
            shortcircuit_local = False if "." in class_path else True
            if not shortcircuit_local:
                class_module, class_name = init["class_path"].rsplit(".", 1)
            else:  # class is expected to be locally defined
                args_class = globals()[init["class_path"]]
        else:
            raise MisconfigurationException("A class_path was not included in a configuration that requires one")
        if not shortcircuit_local:
            module = __import__(class_module, fromlist=[class_name])
            args_class = getattr(module, class_name)
        return args_class(**kwargs) if not args else args_class(*args, **kwargs)
