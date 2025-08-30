from typing import Any, Union, Optional, Dict, Tuple, Callable, List
import importlib
from functools import lru_cache
from importlib.util import find_spec
import operator
import pkg_resources
from packaging.version import Version

from interpretune.utils.exceptions import MisconfigurationException

# Lazy import torch to improve import performance
_torch = None


def _get_torch():
    """Get torch module, importing it lazily."""
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def instantiate_class(
    init: Dict[str, Any], args: Optional[Union[Any, Tuple[Any, ...]]] = None, import_only: bool = False
) -> Any:
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
        assert class_module is not None
        assert class_name is not None
        module = importlib.import_module(class_module)
        args_class = getattr(module, class_name)
    if import_only:
        assert args_class is not None
        return args_class
    else:
        assert args_class is not None
        return args_class(**kwargs) if not args else args_class(*args, **kwargs)


def resolve_funcs(cfg_obj: Any, func_type: str) -> List[Callable[..., Any]]:
    resolved_funcs = []
    funcs_to_resolve = getattr(cfg_obj, func_type)
    if not isinstance(funcs_to_resolve, list):
        funcs_to_resolve = [funcs_to_resolve]
    for func_or_qualname in funcs_to_resolve:
        if callable(func_or_qualname):
            resolved_funcs.append(func_or_qualname)  # TODO: inspect if signature is appropriate for custom hooks
        else:
            module = None
            func = None
            try:
                module, func = func_or_qualname.rsplit(".", 1)
                mod = importlib.import_module(module)
                resolved_func = getattr(mod, func, None)
                if callable(resolved_func):
                    resolved_funcs.append(resolved_func)
                else:
                    raise MisconfigurationException(f"Custom function {func} from module {module} is not callable!")
            except (AttributeError, ImportError) as e:
                err_msg = f"Unable to import and resolve specified function {func} from module {module}: {e}"
                raise MisconfigurationException(err_msg)
    return resolved_funcs


def _resolve_torch_dtype(dtype: Union[Any, str]) -> Optional[Any]:  # Use Any instead of torch.dtype
    torch = _get_torch()
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        return _str_to_torch_dtype(dtype)


def _str_to_torch_dtype(str_dtype: str) -> Optional[Any]:  # Use Any instead of torch.dtype
    torch = _get_torch()
    if hasattr(torch, str_dtype):
        return getattr(torch, str_dtype)
    elif hasattr(torch, str_dtype.split(".")[-1]):
        return getattr(torch, str_dtype.split(".")[-1])


def _import_class(class_path: str) -> Any:
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    return getattr(module, class_name)


################################################################################
# `lightning-utilities` compatible import helper functions
# largely copied from https://bit.ly/lightning_utils definitions
################################################################################


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


def compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
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


################################################################################
# Interpretune installation environment probes
################################################################################

################################################################################
# Interpretune installation environment probes
################################################################################


# Lazy evaluation using classes that act like module-level constants
class _LazyAvailability:
    def __init__(self, check_func):
        self._check_func = check_func
        self._cached_result = None
        self._evaluated = False

    def __bool__(self):
        if not self._evaluated:
            self._cached_result = self._check_func()
            self._evaluated = True
        return self._cached_result

    def __eq__(self, other):
        return bool(self) == other

    def __ne__(self, other):
        return bool(self) != other


_TORCH_GREATER_EQUAL_2_2 = _LazyAvailability(
    lambda: compare_version("torch", operator.ge, "2.2.0", use_base_version=True)
)
_DOTENV_AVAILABLE = _LazyAvailability(lambda: module_available("dotenv"))
_LIGHTNING_AVAILABLE = _LazyAvailability(lambda: package_available("lightning"))
_NEURONPEDIA_AVAILABLE = _LazyAvailability(lambda: package_available("neuronpedia"))
_CT_AVAILABLE = _LazyAvailability(lambda: package_available("circuit_tracer"))
_FTS_AVAILABLE = _LazyAvailability(lambda: module_available("finetuning_scheduler"))
_BNB_AVAILABLE = _LazyAvailability(lambda: package_available("bitsandbytes"))
_SL_AVAILABLE = _LazyAvailability(lambda: module_available("sae_lens"))
