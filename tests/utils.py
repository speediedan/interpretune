from typing import Tuple, List, Dict, Optional
import importlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce

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
