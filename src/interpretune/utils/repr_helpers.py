from __future__ import annotations

from typing import Any, Dict, Callable
import torch
import os
# Note: callers should provide an `_obj_summ_map` dict mapping attribute_name -> label


def summarize_tensor(t: torch.Tensor) -> str:
    return f"Tensor(shape={tuple(t.shape)}, dtype={getattr(t, 'dtype', None)}, device={getattr(t, 'device', None)})"


def summarize_primitive(obj: Any, max_len: int = 20) -> str:
    s = repr(obj)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def summarize_container(obj: Any) -> str:
    try:
        return f"{obj.__class__.__name__}(len={len(obj)})"
    except Exception:
        return obj.__class__.__name__


def summarize_dict_keys(d: Dict, max_keys: int = 8) -> Dict:
    keys = list(d.keys())
    return {"len": len(keys), "keys": keys[:max_keys]}


def summarize_obj(obj: Any) -> Any:
    """Return a JSON-serializable small summary for `obj`.

    Rules:
    - torch.Tensor -> shape/dtype/device string
    - dict -> {len:int, keys:[...first keys...]}
    - list/tuple/set -> ClassName(len=N)
    - str -> truncated primitive
    - path-like (os.PathLike / pathlib.Path) -> string path
    - objects with __class__ -> class name
    """
    try:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return summarize_tensor(obj)
        # primitive numeric/bool types should return as-is
        if isinstance(obj, (int, float, bool)):
            return obj
        # bytes-like
        if isinstance(obj, (bytes, bytearray)):
            return repr(obj)[:20]
        # path-like
        if hasattr(obj, "__fspath__"):
            try:
                return os.fspath(obj)
            except Exception:
                return obj.__class__.__name__
        if isinstance(obj, dict):
            return summarize_dict_keys(obj)
        if isinstance(obj, (list, tuple, set)):
            return summarize_container(obj)
        if isinstance(obj, str):
            return summarize_primitive(obj)
        # fallback: prefer class name for custom objects
        if hasattr(obj, "__class__"):
            # small builtins already handled above
            # show as ClassName(...) to indicate a stateful/custom object
            try:
                return f"{obj.__class__.__name__}(...)"
            except Exception:
                return obj.__class__.__name__
        return summarize_primitive(obj)
    except Exception:
        return f"<error summarizing {obj.__class__.__name__}>"


def state_to_dict(
    obj: Any,
    *,
    custom_key_transforms: dict[str, Callable[[Any], Any]] | None = None,
) -> dict:
    """Generalized state summarizer: produce a dict for `obj` by reading `obj._obj_summ_map`.

    - obj must define `_obj_summ_map` as a dict mapping attribute_name -> label
    - custom_key_transforms: mapping of attribute_name -> function(value) for special handling
    """
    out: dict = {}
    custom_key_transforms = custom_key_transforms or {}

    obj_summ_map = getattr(obj, "_obj_summ_map", None)
    if not isinstance(obj_summ_map, dict):
        raise AttributeError("object must define _obj_summ_map as dict(attribute_name -> label)")

    # keys are the attribute names to read from the object
    keys = list(obj_summ_map.keys())

    for k in keys:
        try:
            v = getattr(obj, k)
            if k in custom_key_transforms:
                try:
                    out[k] = custom_key_transforms[k](v)
                except Exception:
                    out[k] = f"<error transform {k}>"
            else:
                out[k] = summarize_obj(v)
        except Exception as exc:
            out[k] = "<error: {}>".format(exc.__class__.__name__)
    return out


def state_to_summary(
    state_dict: dict,
    obj: Any,
    max_items: int = 10,
) -> str:
    """Generalized compact single-line summary builder.

    - state_dict: output from state_to_dict
    - mapping: list of (label, key) pairs where key corresponds to state_dict keys (may be private or short)
    Returns a parenthesized string like "(device=cpu, epoch=3, ...)" (no class name).
    """
    parts: list[str] = []
    obj_summ_map = getattr(obj, "_obj_summ_map", None)
    if not isinstance(obj_summ_map, dict):
        raise AttributeError("object must define _obj_summ_map as dict(attribute_name -> label)")

    # mapping is list of (label, key) in insertion order of the dict
    mapping = [(label, key) for key, label in obj_summ_map.items()]

    for label, key in mapping[:max_items]:
        val = state_dict.get(key, None)
        if isinstance(val, dict) and val.get("type"):
            display = f"{val.get('type')}({val.get('len')})"
        elif isinstance(val, (list, tuple)):
            display = f"len={len(val)}"
        else:
            display = val
        parts.append(f"{label}={display}")
    return f"({', '.join(parts)})"


def state_repr(summary: str, class_name: str) -> str:
    """Format a repr for class_name using the provided parenthesized summary.

    If formatting fails, return class_name(<unavailable>).
    """
    try:
        if not summary:
            return f"{class_name}(<unavailable>)"
        # summary is expected to be parenthesized like "(a=1, b=2)"
        return f"{class_name}{summary}"
    except Exception:
        return f"{class_name}(<unavailable>)"
