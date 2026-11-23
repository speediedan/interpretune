"""A prefix key/value cache handed to a model backend's traces, and the batch keys that carry it.

An attribution that scores one position can run its forward and backward pass on that position alone, reading the
rest of the sequence from a key/value cache built beforehand without gradients. A batch carries the cache under
``past_key_values``. A backend either honours it on every trace it runs, or refuses the batch by name: a dropped cache
runs the scored tokens without their context and returns plausible, wrong values.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

#: The batch keys that carry a prefix cache to the model's forward, beside the tokens and the attention mask.
PREFIX_CACHE_KEYS = ("past_key_values", "use_cache", "cache_position")


def carries_prefix_cache(batch: Any) -> bool:
    """Whether ``batch`` hands the model a prefix key/value cache."""
    return isinstance(batch, Mapping) and batch.get("past_key_values") is not None


def _replayable_layer_types() -> tuple[type, ...]:
    from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

    return DynamicLayer, DynamicSlidingWindowLayer


def fresh_prefix_cache(cache: Any) -> Any:
    """A copy of ``cache`` that one forward pass can extend without changing the cache the caller holds.

    A forward pass appends its own keys and values to the cache it is handed, and a trace may run the model more than
    once, so each forward starts from this copy and every one of them reads the same prefix. Only the cache's structure
    is copied: a dynamic layer extends itself by concatenating into new tensors and reassigning them, so the copy shares
    the prefix tensors and costs no memory.

    That holds only for the dynamic layers (``DynamicLayer``, ``DynamicSlidingWindowLayer``), so any other cache is
    refused by name rather than copied on a guess. A static cache writes into its preallocated tensors in place, so a
    copy sharing them would let one forward overwrite the prefix the next one reads.

    Raises:
        TypeError: ``cache`` is not a ``transformers`` ``Cache``, offloads its layers, or holds a layer of another kind.
    """
    from transformers.cache_utils import Cache

    if not isinstance(cache, Cache):
        raise TypeError(
            f"past_key_values must be a transformers Cache (a DynamicCache); got {type(cache).__name__}. Build the "
            "prefix with `use_cache=True` and pass the `past_key_values` the forward returns."
        )
    if getattr(cache, "offloading", False):
        raise TypeError(
            f"a prefix cache that offloads its layers ({type(cache).__name__} with offloading=True) moves its tensors "
            "between devices during a forward, so it cannot be replayed for every trace; build it without offloading"
        )
    replayable = _replayable_layer_types()
    if other := sorted({type(layer).__name__ for layer in cache.layers if type(layer) not in replayable}):
        raise TypeError(
            f"a prefix cache can hold only {[kind.__name__ for kind in replayable]} layers, which extend themselves "
            f"without writing into the prefix; got {other}. A static cache writes in place, so build the prefix as a "
            "DynamicCache."
        )
    fresh = copy.copy(cache)
    fresh.layers = [copy.copy(layer) for layer in cache.layers]
    return fresh


def prefix_cache_kwargs(batch: Mapping[str, Any]) -> dict[str, Any]:
    """The forward arguments that hand ``batch``'s prefix cache, freshly copied, to one forward pass.

    Empty when the batch carries no cache. ``use_cache`` defaults to ``True``, since a model told not to use its cache
    ignores the one it is handed.
    """
    if not carries_prefix_cache(batch):
        return {}
    kwargs = {key: batch[key] for key in PREFIX_CACHE_KEYS if batch.get(key) is not None}
    kwargs["past_key_values"] = fresh_prefix_cache(batch["past_key_values"])
    kwargs.setdefault("use_cache", True)
    return kwargs


def refuse_prefix_cache(batch: Any, backend: str) -> None:
    """Refuse, by name, a batch carrying a prefix cache that ``backend`` cannot honour.

    Raises:
        NotImplementedError: ``batch`` carries ``past_key_values``.
    """
    if carries_prefix_cache(batch):
        raise NotImplementedError(
            f"the {backend} backend cannot run a trace from a prefix key/value cache (`past_key_values` in the batch); "
            "dropping it would score the tokens without their context. Pass the whole sequence, or use the nnsight "
            "model backend, which runs each trace from the cache."
        )
