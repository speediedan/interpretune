from abc import ABC
from typing import Optional, Tuple, Callable, Any, Mapping, Sequence, Union
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import fields, is_dataclass, FrozenInstanceError

import torch

################################################################################
# Useful Lightning utility functions used for core testing but not imported
# from Lightning to avoid dependency. Originally copied and/or adpated from:
# https://bit.ly/lightning_core_utils and https://bit.ly/lightning_fabric_utils
################################################################################



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


################################################################################
# Data movement utils
################################################################################

def is_namedtuple(obj: object) -> bool:
    """Check if object is type nametuple."""
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")

def is_dataclass_instance(obj: object) -> bool:
    """Check if object is dataclass."""
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
    include_none: bool = True,
    allow_frozen: bool = False,
    **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        allow_frozen: Whether not to error upon encountering a frozen dataclass instance.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection
    """
    if include_none is False or wrong_dtype is not None or allow_frozen is True:
        # not worth implementing these on the fast path: go with the slower option
        return _apply_to_collection_slow(
            data,
            dtype,
            function,
            *args,
            wrong_dtype=wrong_dtype,
            include_none=include_none,
            allow_frozen=allow_frozen,
            **kwargs,
        )
    # fast path for the most common cases:
    if isinstance(data, dtype):  # single element
        return function(data, *args, **kwargs)
    if isinstance(data, list) and all(isinstance(x, dtype) for x in data):  # 1d homogeneous list
        return [function(x, *args, **kwargs) for x in data]
    if isinstance(data, tuple) and all(isinstance(x, dtype) for x in data):  # 1d homogeneous tuple
        return tuple(function(x, *args, **kwargs) for x in data)
    if isinstance(data, dict) and all(isinstance(x, dtype) for x in data.values()):  # 1d homogeneous dict
        return {k: function(v, *args, **kwargs) for k, v in data.items()}
    # slow path for everything else
    return _apply_to_collection_slow(
        data,
        dtype,
        function,
        *args,
        wrong_dtype=wrong_dtype,
        include_none=include_none,
        allow_frozen=allow_frozen,
        **kwargs,
    )

def _apply_to_collection_slow(
    data: Any,
    dtype: Union[type, Any, Tuple[Union[type, Any]]],
    function: Callable,
    *args: Any,
    wrong_dtype: Optional[Union[type, Tuple[type, ...]]] = None,
    include_none: bool = True,
    allow_frozen: bool = False,
    **kwargs: Any,
) -> Any:
    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = _apply_to_collection_slow(
                v,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple_ = is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple_ or is_sequence:
        out = []
        for d in data:
            v = _apply_to_collection_slow(
                d,
                dtype,
                function,
                *args,
                wrong_dtype=wrong_dtype,
                include_none=include_none,
                allow_frozen=allow_frozen,
                **kwargs,
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple_ else elem_type(out)

    if is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        dfields = {}
        memo = {}
        for field in fields(data):
            field_value = getattr(data, field.name)
            dfields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in dfields.items():
            v = None
            if field_init:
                v = _apply_to_collection_slow(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    allow_frozen=allow_frozen,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            try:
                setattr(result, field_name, v)
            except FrozenInstanceError as e:
                if allow_frozen:
                    # Quit early if we encounter a frozen data class; return `result` as is.
                    break
                raise ValueError(
                    "A frozen dataclass was passed to `apply_to_collection` but this is not allowed."
                ) from e
        return result

    # data is neither of dtype, nor a collection
    return data


class _TransferableDataType(ABC):
    """A custom type for data that can be moved to a torch device via ``.to(...)``."""

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is _TransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented

_DEVICE = Union[torch.device, str, int]

def to_device(device: _DEVICE, obj: Union[torch.nn.Module, torch.Tensor, Any]) -> Union[torch.nn.Module, torch.Tensor,
                                                                                        Any]:
    r"""Move a :class:`torch.nn.Module` or a collection of tensors to the current device, if it is not already on
    that device.

    Args:
        obj: An object to move to the device. Can be an instance of :class:`torch.nn.Module`, a tensor, or a
            (nested) collection of tensors (e.g., a dictionary).

    Returns:
        A reference to the object that was moved to the new device.

    """
    if isinstance(obj, torch.nn.Module):
        obj.to(device)
        return obj
    return move_data_to_device(obj, device=device)

def move_data_to_device(batch: Any, device: _DEVICE) -> Any:
    """Transfers a collection of data to the given device. Any object that defines a method ``to(device)`` will be
    moved and all other objects in the collection will be left untouched.

    Args:
        batch: A tensor or collection of tensors or anything that has a method ``.to(...)``.
            See :func:`apply_to_collection` for a list of supported collection types.
        device: The device to which the data should be moved

    Return:
        the same collection but with all contained tensors residing on the new device.

    See Also:
        - :meth:`torch.Tensor.to`
        - :class:`torch.device`
    """
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:
        kwargs = {}
        # Don't issue non-blocking transfers to CPU
        if isinstance(data, torch.Tensor) and isinstance(device, torch.device) and device.type not in "cpu":
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        # user wrongly implemented the `_TransferableDataType` and forgot to return `self`.
        return data

    return apply_to_collection(batch, dtype=_TransferableDataType, function=batch_to)
