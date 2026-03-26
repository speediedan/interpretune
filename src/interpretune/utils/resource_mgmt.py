"""Resource management utilities for CUDA memory and Python garbage collection.

Provides general-purpose helpers for managing CUDA tensor lifecycles and
system memory cleanup. These are used by both the main application (notebook
workflows, experiment scripts) and the test infrastructure.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Iterator

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
