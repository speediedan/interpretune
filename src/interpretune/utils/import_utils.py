from typing import Any, Callable
import importlib
import os
from functools import lru_cache
from importlib.util import find_spec
from importlib.metadata import version as get_version, PackageNotFoundError
import operator

from interpretune.utils.logging import rank_zero_warn
import torch
from packaging.version import InvalidVersion, Version

from interpretune.utils import MisconfigurationException


def instantiate_class(
    init: dict[str, Any], args: Any | tuple[Any, ...] | None = None, import_only: bool = False
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


def resolve_funcs(cfg_obj: Any, func_type: str) -> list[Callable[..., Any]]:
    """Resolve a config attribute into a list of callables, accepting callables or dotted qualnames.

    Configuration may name a hook either as an already-imported callable or as an importable
    ``module.attr`` string (the form a YAML/CLI config can express), and either as one value or a
    list. This normalizes all four shapes to a list of callables, importing where needed.

    Raises:
        MisconfigurationException: a qualname resolves to something non-callable, or its module or
            attribute cannot be imported -- surfaced here rather than at first call, so a typo in a
            config fails at setup instead of mid-run.
    """
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


def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype | None:
    """Resolve a dtype which may be a torch.dtype or a string to a torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        return _str_to_dtype(dtype)


def _str_to_dtype(str_dtype: str) -> torch.dtype | None:
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


def compare_version(package: str, op: Callable, version_str: str, use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.

    >>> compare_version("torch", operator.ge, "0.1")
    True
    >>> compare_version("does_not_exist", operator.ge, "0.0")
    False
    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, PackageNotFoundError):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try importlib.metadata to infer version
            pkg_version = Version(get_version(package))
    except (TypeError, InvalidVersion, PackageNotFoundError):
        # `__version__` is not a parseable version -- Sphinx mocks it, so return True and let all
        # summaries generate. BOTH exception types are required: packaging <= 26.2 let the underlying
        # TypeError escape, while 26.3+ catches it and re-raises `InvalidVersion` (a ValueError).
        # Catching only TypeError makes this raise on 26.3+, which is how CI caught it.
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version_str))


################################################################################
# Interpretune installation environment probes
################################################################################

_TORCH_GREATER_EQUAL_2_2 = compare_version("torch", operator.ge, "2.2.0", use_base_version=True)
_DOTENV_AVAILABLE = module_available("dotenv")
_LIGHTNING_AVAILABLE = package_available("lightning")
_NEURONPEDIA_AVAILABLE = package_available("neuronpedia")
_CT_AVAILABLE = package_available("circuit_tracer")
_FTS_AVAILABLE = module_available("finetuning_scheduler")
_BNB_AVAILABLE = package_available("bitsandbytes")
_SL_AVAILABLE = module_available("sae_lens")
_NNSIGHT_AVAILABLE = package_available("nnsight")
# local-checkout package (neuronpedia repo, utils/neuronpedia-utils) — not installable from PyPI
_NEURONPEDIA_UTILS_AVAILABLE = module_available("neuronpedia_utils")


def _resolve_env_auth_token(os_env_model_auth_key: str | None) -> str | None:
    """Resolve a configured auth-token environment variable, warning rather than raising when unset.

    Returns None when no variable is configured, and None WITH A WARNING when one is configured but
    absent from the environment. Deliberately not ``os.environ[...]`` (#354): a bare ``KeyError`` from
    deep inside model init names a variable and nothing else, and a user who followed the README and
    forgot to export a token gets no actionable message.

    Returning None is also the correct behavior rather than merely the gentler one: huggingface_hub
    falls back to the ambient cached credential (``huggingface-cli login``), so a gated-public repo the
    user already has access to keeps working. The warning exists because the config NAMED a variable
    that does not exist, which is worth knowing even when the fallback succeeds.
    """
    if not os_env_model_auth_key:
        return None
    env_var = os_env_model_auth_key.upper()
    token = os.environ.get(env_var)
    if token is None:
        rank_zero_warn(
            f"Configured auth-token environment variable {env_var!r} is not set. Proceeding without an explicit "
            "token: any ambient Hugging Face credential (e.g. from `huggingface-cli login`) still applies, but "
            f"access to gated resources will fail later if none exists. Export {env_var} to use an explicit token."
        )
    return token
