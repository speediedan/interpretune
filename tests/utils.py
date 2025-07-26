from typing import Tuple, List, Dict, Optional, Union, Type, Any, Callable, NamedTuple
import importlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
import os
import psycopg
from dotenv import load_dotenv

import torch

from interpretune.session import ITSession
from interpretune.config import GenerativeClassificationConfig, CoreGenerationConfig

################################################################################
# Test Utility Functions
################################################################################

@dataclass(kw_only=True)
class ToyGenCfg(CoreGenerationConfig):
    output_logits: bool = True
    verbose: bool = True

def dummy_step(*args, **kwargs) -> None:
    ...

def nones(num_n) -> Tuple:  # to help dedup config
    return (None,) * num_n

def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)

# useful for manipulating segments of nested dictionaries (e.g. generating config file sets for CLI composition tests)
def set_nested(chained_keys: List | str, orig_dict: Optional[Dict] = None):
    orig_dict = {} if orig_dict is None else orig_dict
    chained_keys = chained_keys if isinstance(chained_keys, list) else chained_keys.split(".")
    reduce(lambda d, k: d.setdefault(k, {}), chained_keys, orig_dict)
    return orig_dict

def get_nested(target: Dict, chained_keys: List | str):
    chained_keys = chained_keys if isinstance(chained_keys, list) else chained_keys.split(".")
    return reduce(lambda d, k: d.get(k), chained_keys, target)

def get_model_input_dtype(precision):
    if precision in ("float16", "16-true", "16-mixed", "16", 16):
        return torch.float16
    if precision in ("bfloat16", "bf16-true", "bf16-mixed", "bf16"):
        return torch.bfloat16
    if precision in ("64-true", "64", 64):
        return torch.double
    return torch.float32

@contextmanager
def ablate_cls_attrs(object: object, attr_names: str| tuple):
    try:
        # (orig_obj_attach, orig_attr_handle): index by original object and handle of attribute we ablate
        ablated_attr_indices = {}
        if not isinstance(attr_names, tuple):
            attr_names = (attr_names,)
        for attr_name in attr_names:
            ablated_attr_indices[attr_name] = attr_resolve(object, attr_name)
        yield
    finally:
        for attr_name in reversed(ablated_attr_indices.keys()):
            setattr(ablated_attr_indices[attr_name][0], attr_name, ablated_attr_indices[attr_name][1])

def attr_resolve(object: object, attr_name: str):
    if not hasattr(object, attr_name):
        raise AttributeError(f"{object} does not have the requested attribute to ablate ({attr_name})")
    orig_attr_handle = getattr(object, attr_name)
    if object.__dict__.get(attr_name, '_indirect') != '_indirect':
        orig_obj_attach_handle = object
    else:
        orig_obj_attach_handle = indirect_resolve(object, attr_name, orig_attr_handle)
    ablation_index_entry = (orig_obj_attach_handle, orig_attr_handle)
    delattr(orig_obj_attach_handle, attr_name)
    return ablation_index_entry

def indirect_resolve(object: object, attr_name: str, orig_attr_handle: object):
    try:
        orig_attr_fqn = getattr(orig_attr_handle, "__qualname__", None) or orig_attr_handle.__class__.__qualname__
        orig_obj_attach = orig_attr_fqn[:-len(orig_attr_fqn.rsplit(".", 1)[-1]) - 1]
    except AttributeError as ae:
        raise AttributeError("Could not resolve the original object and attribute of the requested object and"
                             f" attribute pair: ({object}, {attr_name}). Received: {ae}")
    mod = importlib.import_module(orig_attr_handle.__module__)
    return getattr(mod, orig_obj_attach)

@contextmanager
def disable_genclassif(it_session: ITSession):
    try:
        orig_genclassif_cfg = it_session.module.it_cfg.generative_step_cfg
        it_session.module.it_cfg.generative_step_cfg = GenerativeClassificationConfig(enabled=False)
        yield
    finally:
        it_session.module.it_cfg.generative_step_cfg = orig_genclassif_cfg

def _unwrap_one(seq):
    return seq[0] if len(seq) == 1 else seq

def kwargs_from_cfg_obj(cfg_obj, source_obj, base_kwargs=None):
    """Dynamically extract a subset of configuration parameters from a source object based on an object's
    signature.

    Args:
        cfg_obj: The object (class or function) whose signature will be used to extract parameter names
        source_obj: The object from which to extract attribute values
        base_kwargs: Optional base dictionary to update with extracted values

    Returns:
        Dictionary with extracted configuration parameters
    """
    import inspect

    # Start with base kwargs if provided
    kwargs = base_kwargs or {}

    # Determine the signature to use based on the type of cfg_obj
    if inspect.isclass(cfg_obj):
        param_names = [
            param.name for param in inspect.signature(cfg_obj.__init__).parameters.values()
            if param.name not in ('self',)  # Exclude 'self'
        ]
    elif inspect.isfunction(cfg_obj):
        param_names = [
            param.name for param in inspect.signature(cfg_obj).parameters.values()
        ]
    else:
        raise TypeError("cfg_obj must be a class or a function")

    # Extract matching attributes from the source object
    for attr in param_names:
        if hasattr(source_obj, attr):
            kwargs[attr] = getattr(source_obj, attr)

    return kwargs

def get_super_method(cls_path_or_type: Union[str, Type], instance: Any, method_name: str) -> Callable:
    """Retrieves a method from a parent class by using standard super() resolution.

    This is useful for testing specific implementations of methods that might be overridden in subclasses.

    Args:
        cls_path_or_type: Either a fully-qualified dot-separated string path to a class or the class type itself
        instance: An instantiated object from which to fetch the method
        method_name: The name of the method to retrieve

    Returns:
        A handle on the target method found in the parent class

    Example:
        ```python
        from it_examples.experiments.rte_boolq import RTEBoolqModuleMixin

        # Get the BaseITModule version of standardize_logits
        base_standardize_logits = get_super_method(RTEBoolqModuleMixin, module, "standardize_logits")
        result = base_standardize_logits(logits)
        ```
    """
    try:
        # If cls_path_or_type is a string, import the class
        if isinstance(cls_path_or_type, str):
            try:
                module_path, class_name = cls_path_or_type.rsplit('.', 1)
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Failed to import {cls_path_or_type}: {str(e)}")
        else:
            cls = cls_path_or_type

        # Use standard super() resolution
        method = getattr(super(cls, instance), method_name)
        return method
    except AttributeError as e:
        raise AttributeError(f"Method '{method_name}' not found in parent classes of {cls.__name__}: {str(e)}")


class InOutComp(NamedTuple):
    """Named tuple for more explicit input and output access in comparisons."""
    input: Any
    output: Any

################################################################################
# CUDA utils
################################################################################

def _clear_cuda_memory() -> None:
    # strangely, the attribute function be undefined when torch.compile is used
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        # https://github.com/pytorch/pytorch/issues/95668
        torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()

def cuda_reset():
    if torch.cuda.is_available():
        _clear_cuda_memory()
        torch.cuda.reset_peak_memory_stats()

def sync_dev_graph_metadata():
    """Sync graph metadata from Neuronpedia production service to a local dev database.

    Args:
        model_id (str): The model ID associated with the graph.
        slug (str): The unique slug for the graph.
        username (str): The username to associate the graph metadata with in the local database.
    """
    model_id = os.environ.get("MODEL_ID")
    slug = os.environ.get("SLUG")
    username = os.environ.get("USERNAME")
    np_s3_user_graph_prefix = os.environ.get("NP_S3_USER_GRAPH_PREFIX")

    if not all([model_id, slug, username, np_s3_user_graph_prefix]):
        raise ValueError("One or more required environment variables are missing.")

    import neuronpedia
    from neuronpedia.np_graph_metadata import NPGraphMetadata
    # Load environment variables
    load_dotenv(".env.localhost_np")

    # Get Neuronpedia API key
    public_api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if not public_api_key:
        raise ValueError("NEURONPEDIA_API_KEY is not set in the environment.")

    # Fetch graph metadata from Neuronpedia
    with neuronpedia.api_key(public_api_key):
        graph_metadata = NPGraphMetadata.get(model_id, slug)

    # Connect to the local PostgreSQL database
    conn = psycopg.connect(
        dbname=os.environ.get("POSTGRES_DB"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
        host="localhost",
        port=5432
    )

    try:
        with conn.cursor() as cursor:
            # Get user ID from username
            cursor.execute("SELECT id FROM \"User\" WHERE name=%s;", (username,))
            user_id = cursor.fetchone()
            if not user_id:
                raise ValueError(f"User '{username}' not found in the database.")

            user_id = user_id[0]

            s3_url = f"{np_s3_user_graph_prefix}/{user_id}/{slug}.json"

            # Insert graph metadata into the GraphMetadata table
            cursor.execute(
                """
                INSERT INTO public."GraphMetadata" (
                    id, "modelId", slug, "promptTokens", prompt, "titlePrefix", url, "userId", "createdAt", "updatedAt",
                    "isFeatured") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(3), current_timestamp(3),
                    FALSE)
                ON CONFLICT (id) DO NOTHING;
                """,
                (
                    graph_metadata.id,
                    graph_metadata.model_id,
                    graph_metadata.slug,
                    graph_metadata.prompt_tokens,
                    graph_metadata.prompt,
                    graph_metadata.title_prefix,
                    s3_url,
                    user_id
                )
            )

            conn.commit()
            print(f"Graph metadata for slug '{slug}' synced successfully to local dev database.")

    except Exception as e:
        conn.rollback()
        print(f"Error syncing graph metadata: {e}")

    finally:
        conn.close()
