"""Utilities to dynamically load functions from Hub-downloaded modules."""

from __future__ import annotations
import os
import sys
import threading
import warnings
import importlib
import importlib.util
import hashlib
import shutil
import filecmp
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional, Union, List, Set

from huggingface_hub import try_to_load_from_cache
from transformers.dynamic_module_utils import get_relative_import_files, check_imports
from transformers.utils.hub import cached_file, extract_commit_hash, is_offline_mode
from transformers.utils import logging
from interpretune.analysis import IT_MODULES_CACHE, IT_DYNAMIC_MODULE_NAME
from interpretune.utils.logging import rank_zero_debug, rank_zero_warn

# Track paths we've added to sys.path to avoid duplicates
_added_op_paths: Set[str] = set()

logger = logging.get_logger(__name__)
_IT_REMOTE_CODE_LOCK = threading.Lock()

# Note: This module is initially a customization of a subset of functions from HuggingFace Transformers
# dynamic_module_utils.py (at this sha https://bit.ly/orig_transformers_dyn_mod_utils). In the future, it is hoped we
# can dynamically create patched versions of functions like get_cached_module_file and create_dynamic_module with our
# relevant env var patches to avoid this substantial HF code duplication/maintenance (or even convince HF to refactor
# these functions to support the custom patterns we require).


def init_it_modules() -> None:
    """Creates the cache directory for interpretune modules with an init, and adds it to the Python path."""
    # This function has already been executed if IT_MODULES_CACHE already is in the Python path.
    if IT_MODULES_CACHE in sys.path:
        return

    ensure_op_paths_in_syspath([IT_MODULES_CACHE])
    os.makedirs(IT_MODULES_CACHE, exist_ok=True)
    init_path = Path(IT_MODULES_CACHE) / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        importlib.invalidate_caches()


def create_dynamic_module_it(name: Union[str, os.PathLike]) -> None:
    """Creates a dynamic module in the interpretune cache directory for modules.

    Args:
        name (`str` or `os.PathLike`):
            The name of the dynamic module to create.
    """
    init_it_modules()
    dynamic_module_path = (Path(IT_MODULES_CACHE) / name).resolve()
    # If the parent module does not exist yet, recursively create it.
    if not dynamic_module_path.parent.exists():
        create_dynamic_module_it(dynamic_module_path.parent)
    os.makedirs(dynamic_module_path, exist_ok=True)
    init_path = dynamic_module_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()
        # It is extremely important to invalidate the cache when we change stuff in those modules
        importlib.invalidate_caches()


def get_function_in_module(
    function_name: str,
    module_path: Union[str, os.PathLike],
    *,
    force_reload: bool = False,
) -> Callable:
    """Import a module on the cache directory for modules and extract a function from it.

    Args:
        function_name (`str`): The name of the function to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.
        force_reload (`bool`, *optional*, defaults to `False`):
            Whether to reload the dynamic module from file if it already exists in `sys.modules`.
            Otherwise, the module is only reloaded if the file has changed.

    Returns:
        `Callable`: The function looked for.
    """
    name = os.path.normpath(module_path)
    if name.endswith(".py"):
        name = name[:-3]
    name = name.replace(os.path.sep, ".")
    module_file: Path = Path(IT_MODULES_CACHE) / module_path

    with _IT_REMOTE_CODE_LOCK:
        if force_reload:
            sys.modules.pop(name, None)
            importlib.invalidate_caches()
        cached_module: Optional[ModuleType] = sys.modules.get(name)
        module_spec = importlib.util.spec_from_file_location(name, location=module_file)
        if module_spec is None:
            raise ImportError(f"Could not create module spec for {name} from {module_file}")

        # Hash the module file and all its relative imports to check if we need to reload it
        module_files: list[Path] = [module_file] + sorted(map(Path, get_relative_import_files(module_file)))
        module_hash: str = hashlib.sha256(b"".join(bytes(f) + f.read_bytes() for f in module_files)).hexdigest()

        module: ModuleType
        if cached_module is None:
            module = importlib.util.module_from_spec(module_spec)
            # insert it into sys.modules before any loading begins
            sys.modules[name] = module
        else:
            module = cached_module
        # reload in both cases, unless the module is already imported and the hash hits
        if getattr(module, "__interpretune_module_hash__", "") != module_hash:
            if module_spec.loader is None:
                raise ImportError(f"Module spec for {name} has no loader")
            module_spec.loader.exec_module(module)
            module.__interpretune_module_hash__ = module_hash
        return getattr(module, function_name)


def get_cached_module_file_it(
    op_repo_name_or_path: Union[str, os.PathLike],
    module_file: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> str:
    """Downloads a module from a local folder or a distant repo and returns its path inside the cached interpretune
    module.

    Args:
        op_repo_name_or_path (`str` or `os.PathLike`):
            This can be either:
            - a string, the *repo id* of an analysis operations repo hosted inside a model repo on
              huggingface.co (e.g., "username/repo-name").
            - a path to a *directory* containing operation files saved locally.

        module_file (`str`):
            The name of the module file containing the function to look for.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded repo should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific repo version to use.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load files from local cache.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    Returns:
        `str`: The path to the module inside the cache.
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of "
            "Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    if is_offline_mode() and not local_files_only:
        rank_zero_debug("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Download and cache module_file from the repo `op_repo_name_or_path` or grab it if it's a local file.
    op_repo_name_or_path = str(op_repo_name_or_path)
    is_local = os.path.isdir(op_repo_name_or_path)
    cached_module = None
    if is_local:
        submodule = os.path.basename(op_repo_name_or_path)
    else:
        # Replace '.' with '/' for repo names to support HuggingFace repo conventions
        op_repo_name_or_path = op_repo_name_or_path.replace(".", "/")
        submodule = op_repo_name_or_path.replace("/", os.path.sep)
        cached_module = try_to_load_from_cache(
            op_repo_name_or_path,
            module_file,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            revision=_commit_hash,
            repo_type=repo_type,
        )
    new_files = []
    try:
        # Load from URL or cache if already cached
        resolved_module_file = cached_file(
            op_repo_name_or_path,
            module_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            repo_type=repo_type,
            _commit_hash=_commit_hash,
        )
        if not is_local and cached_module != resolved_module_file:
            new_files.append(module_file)
    except Exception as e:
        logger.error(
            f"Could not locate or download module file `{module_file}` using the provided repo"
            f" {op_repo_name_or_path}: {e}"
        )
        raise

    # Check we have all the requirements in our environment
    if resolved_module_file is None:
        raise RuntimeError(f"Failed to resolve module file {module_file} from {op_repo_name_or_path}")
    modules_needed = check_imports(resolved_module_file)

    # Now we move the module inside our cached dynamic modules.
    full_submodule = IT_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module_it(full_submodule)
    submodule_path = Path(IT_MODULES_CACHE) / full_submodule
    if submodule == os.path.basename(op_repo_name_or_path):
        # We copy local files to avoid putting too many folders in sys.path
        if not (submodule_path / module_file).exists() or not filecmp.cmp(
            resolved_module_file, str(submodule_path / module_file)
        ):
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        for module_needed in modules_needed:
            module_needed = f"{module_needed}.py"
            module_needed_file = os.path.join(op_repo_name_or_path, module_needed)
            if not (submodule_path / module_needed).exists() or not filecmp.cmp(
                module_needed_file, str(submodule_path / module_needed)
            ):
                shutil.copy(module_needed_file, submodule_path / module_needed)
                importlib.invalidate_caches()
    else:
        commit_hash = extract_commit_hash(resolved_module_file, _commit_hash)

        # TODO: might be able to remove this fallback
        if commit_hash is None:
            # If we cannot extract the commit hash, we assume it's a local module
            commit_hash = "local"

        # The module file will end up being placed in a subfolder with the git hash of the repo
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        full_submodule_module_file_path = os.path.join(full_submodule, module_file)
        create_dynamic_module_it(Path(full_submodule_module_file_path).parent)

        if not (submodule_path / module_file).exists():
            shutil.copy(resolved_module_file, submodule_path / module_file)
            importlib.invalidate_caches()
        # Make sure we also have every file with relative imports
        for module_needed in modules_needed:
            if not (submodule_path / f"{module_needed}.py").exists():
                get_cached_module_file_it(
                    op_repo_name_or_path,
                    f"{module_needed}.py",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                )
                new_files.append(f"{module_needed}.py")

    if len(new_files) > 0 and revision is None:
        new_files = "\n".join([f"- {f}" for f in new_files])
        url = f"https://huggingface.co/{op_repo_name_or_path}"
        rank_zero_warn(
            f"A new version of the following files was downloaded from {url}:\n{new_files}"
            f"\n. Make sure to double-check they do not contain any added malicious code. "
            f"To avoid downloading new versions of the code file, you can pin a revision."
        )

    return os.path.join(full_submodule, module_file)


def get_function_from_dynamic_module(
    function_reference: str,
    op_repo_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> Callable:
    """Extracts a function from a module file, present in the local folder or repository of an operations repo.

    <Tip warning={true}>

    Calling this function will execute the code in the module file found locally or downloaded from the Hub. It should
    therefore only be called on trusted repos.

    </Tip>

    Args:
        function_reference (`str`):
            The full name of the function to load, including its module and optionally its repo.
        op_repo_name_or_path (`str` or `os.PathLike`):
            This can be either:
            - a string, the *repo id* of an analysis operations repo hosted inside a model repo on
              huggingface.co (e.g., "username/repo-name").
            - a path to a *directory* containing operation files saved locally.

            This is used when `function_reference` does not specify another repo.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded repo should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint.
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load from local files.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).
        code_revision (`str`, *optional*, defaults to `"main"`):
            The specific revision to use for the code on the Hub, if the code lives in a different repository than the
            rest of the operations.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Callable`: The function, dynamically imported from the module.

    Examples:

    ```python
    # Download module `example_op_definitions.py` from huggingface.co and cache then extract the function
    # `trivial_test_op_impl` from this module.
    func = get_function_from_dynamic_module(
        "example_op_definitions.trivial_test_op_impl", "username/my-ops-repo"
    )

    # Download module `example_op_definitions.py` from a given repo and cache then extract the function
    # `trivial_test_op_impl` from this module.
    func = get_function_from_dynamic_module(
        "username/my-ops-repo--example_op_definitions.trivial_test_op_impl",
        "username/another-ops-repo"
    )
    ```
    """
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of "
            "Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # Catch the name of the repo if it's specified in `function_reference`
    if "--" in function_reference:
        repo_id, function_reference = function_reference.split("--")
    else:
        repo_id = op_repo_name_or_path
    module_file, function_name = function_reference.split(".")

    if code_revision is None and op_repo_name_or_path == repo_id:
        code_revision = revision

    # Get the cached module file
    final_module = get_cached_module_file_it(
        repo_id,
        module_file + ".py",
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=code_revision,
        local_files_only=local_files_only,
        repo_type=repo_type,
    )

    # Check if the module file was found
    if final_module is None:
        raise OSError(f"Could not locate the module file '{module_file}.py' in repository '{repo_id}'")

    return get_function_in_module(function_name, final_module, force_reload=force_download)


def ensure_op_paths_in_syspath(op_paths: List[Union[str, Path]]) -> None:
    """Ensure all operation paths are in sys.path for module discovery.

    Args:
        op_paths: List of paths that should be in sys.path for operation discovery.
                 Can be strings or Path objects.
    """
    global _added_op_paths

    for path_item in op_paths:
        if not path_item:
            continue

        # Convert to Path object for consistent handling
        path = Path(path_item).resolve()

        # Check if we've already processed this path
        path_str_resolved = str(path)
        if path_str_resolved in _added_op_paths:
            continue

        # Verify the path exists and is a directory
        if not path.exists():
            rank_zero_warn(f"Operation path does not exist: {path}")
            continue

        if not path.is_dir():
            rank_zero_warn(f"Operation path is not a directory: {path}")
            continue

        # Add to sys.path if not already present
        if path_str_resolved not in sys.path:
            sys.path.insert(0, path_str_resolved)
            _added_op_paths.add(path_str_resolved)
            rank_zero_debug(f"Added operation path to sys.path: {path_str_resolved}")
        else:
            # Track that we've seen this path even if it was already in sys.path
            _added_op_paths.add(path_str_resolved)


def remove_op_paths_from_syspath(op_paths: List[Union[str, Path]]) -> None:
    """Remove operation paths from sys.path.

    Args:
        op_paths: List of paths to remove from sys.path.
                 Can be strings or Path objects.
    """
    global _added_op_paths

    for path_item in op_paths:
        if not path_item:
            continue

        # Convert to Path object for consistent handling
        path = Path(path_item).resolve()
        path_str_resolved = str(path)

        # Remove from sys.path if we added it
        if path_str_resolved in _added_op_paths:
            try:
                sys.path.remove(path_str_resolved)
                _added_op_paths.remove(path_str_resolved)
                rank_zero_debug(f"Removed operation path from sys.path: {path_str_resolved}")
            except ValueError:
                # Path was not in sys.path, but was in our tracking set
                _added_op_paths.discard(path_str_resolved)


def cleanup_op_paths() -> None:
    """Remove all operation paths that we've added to sys.path."""
    global _added_op_paths

    paths_to_remove = list(_added_op_paths)
    for path_str in paths_to_remove:
        try:
            sys.path.remove(path_str)
            rank_zero_debug(f"Cleaned up operation path from sys.path: {path_str}")
        except ValueError:
            # Path was not in sys.path
            pass

    _added_op_paths.clear()


def get_added_op_paths() -> Set[str]:
    """Get the set of operation paths we've added to sys.path.

    Returns:
        Set of path strings that have been added to sys.path
    """
    return _added_op_paths.copy()
