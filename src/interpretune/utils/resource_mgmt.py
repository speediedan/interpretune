"""Resource management utilities for CUDA memory and Python garbage collection.

Provides general-purpose helpers for managing CUDA tensor lifecycles and system memory cleanup. These are used by both
the main application (notebook workflows, experiment scripts) and the test infrastructure.
"""

from __future__ import annotations

import gc
import os
import shlex
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import psutil
import torch


def cleanup_python_cuda() -> None:
    """Run Python garbage collection and release CUDA caches if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def safe_clean_cuda(model: Any, *, min_bytes: int = 1 << 20) -> Iterator[None]:
    """Move *model* to CUDA; on exit free large transient CUDA tensors.

    Snapshots data_ptrs of all large dense CUDA tensors after the model arrives
    on CUDA. On exit, any *new* large tensor not in the snapshot has its storage
    replaced via ``set_(torch.empty(0))`` to free VRAM while Python references
    remain alive. ``gc.collect()`` + ``empty_cache()`` flush remaining
    allocations before the model is moved back to CPU.

    Handles ``ReferenceError`` from weakly-referenced objects that may be GC'd
    during iteration, making it safe for notebook and interactive use.

    Args:
        model: Object with a ``.to()`` method (typically ``torch.nn.Module``).
        min_bytes: Minimum tensor size to track (default 1 MiB).
    """
    if model is None or not torch.cuda.is_available():
        yield
        return

    model.to("cuda")

    def _is_large_dense_cuda(candidate: object) -> bool:
        try:
            return (
                isinstance(candidate, torch.Tensor)
                and candidate.is_cuda
                and candidate.layout == torch.strided
                and candidate.nbytes >= min_bytes
            )
        except ReferenceError:
            return False

    known_ptrs: set[int] = set()
    for candidate in gc.get_objects():
        if _is_large_dense_cuda(candidate):
            try:
                known_ptrs.add(candidate.data_ptr())
            except ReferenceError:
                continue

    try:
        yield
    finally:
        freed_ptrs: set[int] = set()
        for candidate in gc.get_objects():
            if not _is_large_dense_cuda(candidate):
                continue
            try:
                data_ptr = candidate.data_ptr()
            except ReferenceError:
                continue
            if data_ptr in known_ptrs or data_ptr in freed_ptrs:
                continue
            freed_ptrs.add(data_ptr)
            try:
                candidate.set_(torch.empty(0))
            except Exception:
                pass
        cleanup_python_cuda()
        try:
            model.to("cpu")
        except Exception:
            pass


def analysis_resource_debug_enabled() -> bool:
    """Return ``True`` when resource-debug logging is enabled."""

    return os.environ.get("IT_RESOURCE_DEBUG", "0") == "1"


def get_resource_snapshot(*, include_cuda: bool = True) -> dict[str, float | int | bool]:
    """Capture a compact process and optional CUDA resource snapshot."""

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    snapshot: dict[str, float | int | bool] = {
        "rss_gb": memory_info.rss / (1024**3),
        "vms_gb": memory_info.vms / (1024**3),
    }

    if include_cuda:
        cuda_available = torch.cuda.is_available()
        snapshot["cuda_available"] = cuda_available
        snapshot["cuda_device_count"] = torch.cuda.device_count() if cuda_available else 0
        if cuda_available:
            for device_idx in range(torch.cuda.device_count()):
                device_prefix = f"cuda_gpu{device_idx}"
                device_props = torch.cuda.get_device_properties(device_idx)
                snapshot[f"{device_prefix}_total_gb"] = device_props.total_memory / (1024**3)
                snapshot[f"{device_prefix}_current_allocated_gb"] = torch.cuda.memory_allocated(device_idx) / (1024**3)
                snapshot[f"{device_prefix}_current_reserved_gb"] = torch.cuda.memory_reserved(device_idx) / (1024**3)
                snapshot[f"{device_prefix}_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device_idx) / (1024**3)
                snapshot[f"{device_prefix}_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device_idx) / (1024**3)

    return snapshot


def _format_snapshot_value(value: float | int | bool | str) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, str):
        return shlex.quote(value) if any(char.isspace() for char in value) else value
    return str(value)


def _resource_parts(snapshot: Mapping[str, float | int | bool | str], *, delta_prefix: str = "") -> list[str]:
    return [f"{delta_prefix}{key}={_format_snapshot_value(value)}" for key, value in snapshot.items()]


def _existing_disk_target(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def log_resource_snapshot(
    label: str,
    *,
    paths: list[str | Path] | tuple[str | Path, ...] = (),
    prefix: str = "analysis_resource_debug",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit a compact RSS and disk-usage snapshot when resource debugging is enabled."""

    if not analysis_resource_debug_enabled():
        return

    parts: list[str] = []
    if metadata:
        parts.extend(_resource_parts({key: str(value) for key, value in metadata.items()}))
    parts.extend(_resource_parts(get_resource_snapshot()))

    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        disk_target = _existing_disk_target(path)
        disk_usage = shutil.disk_usage(disk_target)
        parts.extend(
            [
                f"path{index}={path}",
                f"used_gb{index}={disk_usage.used / (1024**3):.2f}",
                f"free_gb{index}={disk_usage.free / (1024**3):.2f}",
            ]
        )

    print(f"[{prefix}] {label}: " + " ".join(parts))


def log_resource_delta(
    label: str,
    *,
    before: dict[str, float | int | bool] | None,
    after: dict[str, float | int | bool] | None = None,
    prefix: str = "analysis_resource_debug",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit the current resource snapshot plus numeric deltas from a prior snapshot."""

    if not analysis_resource_debug_enabled():
        return

    current = after or get_resource_snapshot()
    parts: list[str] = []
    if metadata:
        parts.extend(_resource_parts({key: str(value) for key, value in metadata.items()}))
    parts.extend(_resource_parts(current))

    if before is not None:
        deltas: dict[str, float] = {}
        for key, value in current.items():
            prev = before.get(key)
            if isinstance(value, bool) or isinstance(prev, bool):
                continue
            if isinstance(value, (float, int)) and isinstance(prev, (float, int)):
                deltas[key] = float(value) - float(prev)
        parts.extend(_resource_parts(deltas, delta_prefix="delta_"))

    print(f"[{prefix}] {label}: " + " ".join(parts))
