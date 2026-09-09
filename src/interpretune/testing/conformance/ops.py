"""The conformance suite's op implementations.

One op: store the capture so a case can read it back.
"""

from __future__ import annotations

from typing import Any

import torch


def store_capture_points_impl(module, analysis_batch, batch, batch_idx: int, **_kwargs):
    """Write every cached point into three flat columns: values, shape, names.

    The cache arrives from ``model_fwd_w_cache`` (declared in ``required_ops``); this op adds nothing to the
    forward and consults no backend, so it is pure over its inputs -- the same purity the bundled ops keep,
    asserted for the same reason. ``names_filter`` is a list on the cfg a caller wrote and a callable after
    setup resolves it, so the requested names are read off the cache, which the backend filtered to them.
    """
    import interpretune as it

    if getattr(analysis_batch, "cache", None) is None:
        analysis_batch = it.model_fwd_w_cache(module, analysis_batch, batch, batch_idx)
    cache: Any = analysis_batch.cache
    if cache is None:
        raise ValueError("model_fwd_w_cache produced no cache; nothing to store")
    names = sorted(cache.keys()) if hasattr(cache, "keys") else []
    tensors = []
    for name in names:
        value = cache[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"cached point {name!r} is a {type(value).__name__}, not a tensor")
        tensors.append(value.detach().to("cpu", torch.float32))
    # Points differ in shape (a norm's scale is [batch, pos, 1], an attention input is [batch, pos, heads, d_head],
    # the unembed input is [batch, pos, d_model]), so each is flattened and concatenated, with the shapes encoded
    # beside them as one int64 sequence: `[n_points, ndim_0, dims_0..., ndim_1, dims_1..., ...]`.
    shape_code: list[int] = [len(tensors)]
    for t in tensors:
        shape_code.append(t.ndim)
        shape_code.extend(int(d) for d in t.shape)
    values = torch.cat([t.flatten() for t in tensors]) if tensors else torch.empty(0)
    analysis_batch.update(
        captured_values=values,
        captured_shape=torch.tensor(shape_code, dtype=torch.int64),
        captured_point_names=names,
    )
    return analysis_batch


def captured_points(store: Any, index: int) -> dict[str, torch.Tensor]:
    """Rebuild ``{point name: tensor}`` for batch ``index`` from the three flat columns."""
    # Item access: the store serves only protocol-declared columns as attributes, and these are the suite's own.
    values = torch.as_tensor(store["captured_values"][index], dtype=torch.float32)
    code = [int(d) for d in torch.as_tensor(store["captured_shape"][index]).flatten().tolist()]
    names = list(store["captured_point_names"][index])
    if not names:
        return {}
    n_points, pos = code[0], 1
    assert n_points == len(names), f"{n_points} shapes for {len(names)} names"
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        ndim = code[pos]
        shape = code[pos + 1 : pos + 1 + ndim]
        pos += 1 + ndim
        numel = 1
        for d in shape:
            numel *= d
        out[name] = values[offset : offset + numel].reshape(shape)
        offset += numel
    return out
