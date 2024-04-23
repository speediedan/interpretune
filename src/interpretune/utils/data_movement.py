from typing import Any, Union

import torch

################################################################################
# A few useful utility functions for data movement. Originally copied and/or
# adpated from: https://bit.ly/lightning_core_utils and
# https://bit.ly/lightning_fabric_utils
################################################################################

################################################################################
# Data movement utils
################################################################################

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

    return batch_to(data=batch)
