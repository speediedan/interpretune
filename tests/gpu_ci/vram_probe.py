"""Per-test GPU memory probe: a pytest plugin, loaded with ``-p tests.gpu_ci.vram_probe``.

Driven by ``tests/gpu_ci/calibrate.py``, which runs each GPU test in its own process on an otherwise idle device.
Each test gets two measurements, because each misses something the other sees:

- **allocator peak** (``max_memory_reserved``): exact, in-process only, and blind to the CUDA context;
- **device-level peak** (``torch.cuda.mem_get_info`` polled from a thread): everything on the device, including the
  context and any child process (a notebook kernel, a CLI subprocess). It can miss a short spike, and it is only
  this test's footprint on an idle device, which the GPU lease provides.

Both come from the ``torch.cuda`` API, which ROCm builds of PyTorch provide too, so no vendor tool is needed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import torch

GIB = 2**30


def pytest_addoption(parser):
    group = parser.getgroup("vram", "per-test GPU memory measurement")
    group.addoption("--vram-report", default=None, help="append one JSON line per test with its measured GPU memory")
    group.addoption(
        "--vram-enforce",
        action="store_true",
        help="cap the caching allocator at each test's declared min_gpu_mem_gb, so an under-declared test runs out of "
        "memory instead of passing",
    )
    group.addoption("--vram-sample-hz", type=float, default=20.0, help="device-level sampling rate")


class _DeviceSampler(threading.Thread):
    """Track the peak of device-wide used memory (every process on the device) while a test runs."""

    def __init__(self, hz: float) -> None:
        super().__init__(daemon=True)
        self.interval = 1.0 / hz
        self.peak_used = 0
        self._halt = threading.Event()

    def _used(self) -> int:
        free, total = torch.cuda.mem_get_info(0)
        return total - free

    def run(self) -> None:
        while not self._halt.is_set():
            self.peak_used = max(self.peak_used, self._used())
            self._halt.wait(self.interval)

    def stop(self) -> int:
        self._halt.set()
        self.join()
        self.peak_used = max(self.peak_used, self._used())
        return self.peak_used


def _declared(item) -> float:
    return max(
        (float(m.kwargs.get("min_gpu_mem_gb") or 0) for m in item.iter_markers() if m.name == "skipif"), default=0.0
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    cfg = item.config
    report_path = cfg.getoption("--vram-report")
    if not (report_path or cfg.getoption("--vram-enforce")) or not torch.cuda.is_available():
        yield
        return
    # the context this process holds before the test runs: created here if nothing created it yet
    free, total = torch.cuda.mem_get_info(0)
    base_used = total - free
    declared = _declared(item)
    if cfg.getoption("--vram-enforce") and declared:
        # the declaration covers the whole device footprint, so the allocator gets it minus the context already held
        torch.cuda.set_per_process_memory_fraction(max(0.0, min(1.0, (declared * GIB - base_used) / total)), 0)
    torch.cuda.reset_peak_memory_stats(0)
    sampler = _DeviceSampler(cfg.getoption("--vram-sample-hz"))
    sampler.start()
    t0 = time.time()
    yield
    peak_used = sampler.stop()
    if cfg.getoption("--vram-enforce") and declared:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
    if not report_path:
        return
    alloc_peak = torch.cuda.max_memory_reserved(0)
    outcomes = [r.outcome for r in getattr(item, "_vram_reports", [])]
    row = {
        "nodeid": item.nodeid,
        "outcome": "failed" if "failed" in outcomes else ("skipped" if "skipped" in outcomes else "passed"),
        "seconds": round(time.time() - t0, 2),
        "device": torch.cuda.get_device_name(0),
        "device_total_gb": round(total / GIB, 2),
        "context_gb": round(base_used / GIB, 3),
        "allocator_peak_gb": round(alloc_peak / GIB, 3),
        "device_peak_gb": round(peak_used / GIB, 3),
        # what the test needs on an idle device: everything the device held at its peak, or the in-process peak
        # plus this process's context when the sampler missed a short spike
        "need_gb": round(max(peak_used, base_used + alloc_peak) / GIB, 3),
        "declared_gb": declared,
        "torch": torch.__version__,
    }
    with Path(report_path).open("a") as f:
        f.write(json.dumps(row) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not hasattr(item, "_vram_reports"):
        item._vram_reports = []
    item._vram_reports.append(outcome.get_result())
