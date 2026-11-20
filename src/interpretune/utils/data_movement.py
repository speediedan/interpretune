from typing import Any

import torch

from interpretune.utils.exceptions import MisconfigurationException

################################################################################
# A few useful utility functions for data movement. Originally copied and/or
# adpated from: https://bit.ly/lightning_core_utils and
# https://bit.ly/lightning_fabric_utils
################################################################################

################################################################################
# Data movement utils
################################################################################

_DEVICE = torch.device | str | int


def to_device(device: _DEVICE, obj: torch.nn.Module | torch.Tensor | Any) -> torch.nn.Module | torch.Tensor | Any:
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


################################################################################
# Device requests
################################################################################


def _names_gpu(device: Any) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.strip().lower().split(":")[0] == "cuda"
    return False


def check_gpu_device_request(device: Any, *, field: str) -> None:
    """Refuse by name an explicitly requested GPU device that this torch cannot provide.

    ROCm builds of torch take the same ``cuda`` device string, so one check covers both vendors. A torch built without
    GPU support and a GPU build that sees no device need different remedies, so the message tells them apart. ``field``
    names the setting that asked, as the user wrote it.
    """
    if not _names_gpu(device) or torch.cuda.is_available():
        return
    if torch.version.cuda is None and getattr(torch.version, "hip", None) is None:
        reason = f"this torch build ({torch.__version__}) has no GPU support"
        gpu_remedy = "install a torch build for your accelerator (CUDA or ROCm)"
    else:
        reason = "torch sees no usable GPU (no driver loaded, or no device visible to this process)"
        gpu_remedy = "check the GPU driver and the visible devices (CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES)"
    raise MisconfigurationException(
        f"`{field}` asks for the GPU device {str(device)!r}, but {reason}. To run on the CPU, set `{field}` to 'cpu'"
        f" or leave it unset; to run on the GPU, {gpu_remedy}."
    )


def check_gpu_device_map(device_map: Any, *, field: str) -> None:
    """:func:`check_gpu_device_request` for a ``device_map``: a device string, or a mapping of modules to devices.

    An integer device index names a GPU too. The placement strategies (``"auto"``, ``"balanced"``, ...) are not a GPU
    request: they place on whatever is available.
    """
    if isinstance(device_map, dict):
        for value in device_map.values():
            check_gpu_device_map(value, field=field)
    elif isinstance(device_map, int) and not isinstance(device_map, bool):
        check_gpu_device_request(f"cuda:{device_map}", field=field)
    else:
        check_gpu_device_request(device_map, field=field)
