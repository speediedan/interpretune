"""Which tensor dtypes an AnalysisStore column holds faithfully, and the refusals for the rest.

A store column's dtype is the one its op schema declares (``ColCfg.array_dtype or ColCfg.datasets_dtype``). Writing a
tensor of another dtype makes Arrow cast it, and Arrow's casts on this path are not uniformly safe: depending on the
column's shape and on the writer, a float64 tensor in a float32 column is rounded, a fractional value in an int64
column is truncated, any non-zero value in a bool column becomes ``True``, and a large integer in a float32 column is
rounded or refused. None of those raise in every case, so a store could hold values its producer never computed.

The rule here is value-based: a tensor may be written to a column of another dtype when casting it to the column's
dtype and back reproduces every value (a bfloat16 tensor in a float32 column always does), or when the cast only
rounds floats to the declared float precision, which is warned about once per column. Anything else (truncating a
fraction, booleanizing, losing integer precision, a finite value overflowing to infinity) is refused by
:func:`check_storable` with :class:`StoreDtypeError`, naming the column, both dtypes and the first changed value.

The measured round-trip table is in ``docs/analysis_store_serialization.md``.
"""

from __future__ import annotations

from typing import Any

import torch

#: Arrow value types a store column can declare, mapped to the tensor dtype a reader gets back. Arrow has no
#: bfloat16 type, so it is absent: a bfloat16 tensor is stored in a float32 column, which holds it exactly.
STORE_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

#: Declared dtypes Arrow cannot hold, with what to declare instead.
_UNSTORABLE_DECLARED = {
    "bfloat16": "float32 (every bfloat16 value is a float32 value, so the column holds it exactly)",
    "bf16": "float32 (every bfloat16 value is a float32 value, so the column holds it exactly)",
}


class StoreDtypeError(ValueError):
    """A column's declared dtype cannot hold a value written to it without changing that value."""


def validate_declared_dtype(col_name: str, dtype: str) -> str:
    """Return ``dtype`` for column ``col_name``, refusing by name a dtype Arrow has no type for.

    ``datasets`` refuses these too, but with a message that names neither the column nor a replacement.
    """
    normalized = str(dtype).removeprefix("torch.")
    if normalized in _UNSTORABLE_DECLARED:
        raise StoreDtypeError(
            f"Column {col_name!r} declares dtype {dtype!r}, which Arrow (and so an AnalysisStore) has no type for."
            f" Declare {_UNSTORABLE_DECLARED[normalized]}."
        )
    return dtype


def latent_model_column_dtype(handle_dtype: str | None, default: str = "float32") -> str:
    """The column dtype for a latent model's activations, from the dtype its config names.

    A latent model (an SAE, say) loaded in bfloat16 writes bfloat16 activations, and Arrow has no bfloat16 type; those
    columns are stored as float32, which holds every bfloat16 value exactly, so a reader gets the values the model
    produced, as float32. Any other dtype is used as named.
    """
    if not handle_dtype:
        return default
    normalized = str(handle_dtype).removeprefix("torch.")
    return "float32" if normalized in ("bfloat16", "bf16") else normalized


def _values_survive(tensor: torch.Tensor, target: torch.dtype) -> bool:
    """Whether casting ``tensor`` to ``target`` and back reproduces every value (NaN matches NaN)."""
    back = tensor.to(target).to(tensor.dtype)
    same = back == tensor
    if tensor.is_floating_point() and target.is_floating_point:
        same = same | (back.isnan() & tensor.isnan())
    return bool(same.all())


def _first_changed(tensor: torch.Tensor, target: torch.dtype) -> str:
    """The first value the cast changes, as ``a -> b``, so the refusal shows what would have been stored."""
    flat = tensor.reshape(-1)
    back = flat.to(target).to(tensor.dtype)
    changed = back != flat
    if tensor.is_floating_point() and target.is_floating_point:
        changed = changed & ~(back.isnan() & flat.isnan())
    idx = int(changed.nonzero()[0, 0])
    return f"{flat[idx].item()!r} -> {flat[idx].to(target).item()!r}"


_ROUNDING_WARNED: set[tuple[str, str, str]] = set()


def _is_float_rounding(tensor: torch.Tensor, target: torch.dtype) -> bool:
    """Whether the cast only rounds floats to the declared precision: no finite value becomes infinite.

    A schema that declares ``float32`` declares the precision its op stores, and a model run in float64 produces
    float64 values for it; rounding those is the declared conversion, so it is warned about, not refused. Overflow is
    not rounding: a finite value that becomes ``inf`` is refused like any other changed value.
    """
    if not (tensor.is_floating_point() and target.is_floating_point):
        return False
    return not bool((tensor.isfinite() & ~tensor.to(target).isfinite()).any())


def _warn_rounding_once(col_name: str, given: str, stored_dtype: str) -> None:
    from interpretune.utils import rank_zero_warn

    key = (col_name, given, stored_dtype)
    if key not in _ROUNDING_WARNED:
        _ROUNDING_WARNED.add(key)
        rank_zero_warn(
            f"Column {col_name!r} is declared {stored_dtype} and was given {given} values; they are stored rounded to"
            f" {stored_dtype}. Declare the column {given} to keep the full precision."
        )


def check_storable(col_name: str, value: Any, stored_dtype: str) -> None:
    """Refuse a value that column ``col_name`` (declared ``stored_dtype``) cannot hold without changing it.

    The one change accepted is a float rounded to a narrower declared float with no overflow, which is warned about
    once per column and dtype pair (see :func:`_is_float_rounding`).

    Tensors are checked wherever they sit in ``value``: directly, inside the mappings a per-hook or per-latent column
    writes, and inside lists. Columns whose dtype is not numeric (strings, objects) are not checked, and neither are
    non-tensor values, which ``datasets`` encodes without a cast this module can reason about.
    """
    target = STORE_TORCH_DTYPES.get(str(stored_dtype))
    if target is None:
        return
    if isinstance(value, torch.Tensor):
        if value.dtype == target or value.dtype == torch.bfloat16 and target == torch.float32:
            return
        if value.is_complex():
            raise StoreDtypeError(
                f"Column {col_name!r} is declared {stored_dtype} and was given a {_torch_name(value.dtype)} tensor: an"
                " AnalysisStore has no complex type. Store the real and imaginary parts as separate columns."
            )
        if value.numel() and not _values_survive(value, target):
            given = _torch_name(value.dtype)
            if _is_float_rounding(value, target):
                _warn_rounding_once(col_name, given, stored_dtype)
                return
            raise StoreDtypeError(
                f"Column {col_name!r} is declared {stored_dtype} and was given a {given} tensor whose values"
                f" {stored_dtype} cannot hold (first changed value: {_first_changed(value, target)}). Cast the tensor"
                f" in the op if that change is intended, or declare the column {_dtype_name(value.dtype)}."
            )
        return
    if isinstance(value, dict):
        for item in value.values():
            check_storable(col_name, item, stored_dtype)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_storable(col_name, item, stored_dtype)


def _torch_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_name(dtype: torch.dtype) -> str:
    """The store dtype name for a tensor dtype (``torch.bfloat16`` stores as float32)."""
    if dtype == torch.bfloat16:
        return "float32"
    for name, candidate in STORE_TORCH_DTYPES.items():
        if candidate == dtype:
            return name
    return _torch_name(dtype)
