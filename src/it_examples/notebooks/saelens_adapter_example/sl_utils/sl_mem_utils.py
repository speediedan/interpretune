import gc

from typing import Tuple
import torch as t
#from openai import OpenAIError
from tabulate import tabulate


def get_tensor_size(obj):
    size = 0
    if t.is_tensor(obj):
        size += obj.element_size() * obj.nelement()
    return size


def get_tensors_size(obj):
    if isinstance(obj, t.nn.Module):
        return sum(get_tensor_size(p) for p in obj.parameters())
    if hasattr(obj, "state_dict"):
        return sum(get_tensor_size(t) for t in obj.state_dict().values())
    return get_tensor_size(obj)


def get_device(obj):
    if t.is_tensor(obj):
        return str(obj.device)
    if isinstance(obj, t.nn.Module):
        try:
            return str(next(iter(obj.parameters())).device)
        except StopIteration:
            return "N/A"
    return "N/A"


def print_memory_status():
    t.cuda.synchronize()
    allocated = t.cuda.memory_allocated(0)
    total = t.cuda.get_device_properties(0).total_memory
    free = total - allocated
    print(f"Allocated: {allocated / 1024**3:.2f} GB")
    print(f"Total:  {total / 1024**3:.2f} GB")
    print(f"Free:  {free / 1024**3:.2f} GB")


def profile_pytorch_memory(namespace: dict, n_top: int = 10, filter_device: str = None):
    print_memory_status()

    object_sizes = {}
    for name, obj in namespace.items():
        try:
            obj_type = (type(obj).__name__ if isinstance(obj, t.nn.Module) \
                         else f"Tensor {tuple(obj.shape)}" if t.is_tensor(obj) else None)
            if obj_type is None:
                continue
            device = get_device(obj)
            if filter_device and device != filter_device:
                continue
            size = get_tensors_size(obj)
            object_sizes[name] = (obj_type, device, size / (1024**3))
        except (ReferenceError):
            # ReferenceError: this object might have been garbage collected, so we don't care about it
            continue

    # Convert bytes to GB, sort by size & print
    sorted_sizes = sorted(object_sizes.items(), key=lambda x: x[1][2], reverse=True)[:n_top]
    table_data = [(name, obj_type, device, size) for name, (obj_type, device, size) in sorted_sizes]
    print(
        tabulate(
            table_data, headers=["Name", "Object", "Device", "Size (GB)"], floatfmt=".2f", tablefmt="simple_outline"
        )
    )


def find_cuda_tensors():
    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if t.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append(obj)
        except BaseException:
            pass
    return cuda_tensors

def find_cuda_tensors_by_type():
    tensors = find_cuda_tensors()
    tensors_by_type = {}
    for tensor in tensors:
        t = type(tensor)
        if t not in tensors_by_type:
            tensors_by_type[t] = 0
        tensors_by_type[t] += tensor.element_size() * tensor.nelement()
    return tensors_by_type

def summarize_cuda_tensors_by_shape():
    tensors = find_cuda_tensors()
    tensors_by_shape = {}
    for tensor in tensors:
        shape = tuple(tensor.shape)
        if shape not in tensors_by_shape:
            tensors_by_shape[shape] = [0, 0]
        tensors_by_shape[shape][0] += 1
        tensors_by_shape[shape][1] += tensor.nbytes

    summary = {shape: (count, size / (1024**2)) for shape, (count, size) in tensors_by_shape.items()}
    sorted_summary = dict(sorted(summary.items(), key=lambda x: x[1][1], reverse=True))
    return sorted_summary

def find_cuda_tensors_by_shape(shape: Tuple):
    tensors = find_cuda_tensors()
    tensors_by_shape = []
    for tensor in tensors:
        if tuple(tensor.shape) == shape:
            tensors_by_shape.append(tensor)
    return tensors_by_shape
