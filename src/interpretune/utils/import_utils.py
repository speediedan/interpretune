from typing import Any, Union, Optional, Dict, Tuple
import importlib
from functools import lru_cache
from importlib.util import find_spec


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


def _import_class(class_path: str) -> Any:
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    return getattr(module, class_name)

_DOTENV_AVAILABLE = module_available("dotenv")
_LIGHTNING_AVAILABLE = package_available("lightning")
_FTS_AVAILABLE = package_available("finetuning_scheduler")
_BNB_AVAILABLE = package_available("bitsandbytes")

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
