"""The memory profiler extension: finetuning-scheduler's framework-neutral profiling core, connected to a module.

Collection, scheduling and reporting are the core's (``finetuning_scheduler.profiling``), so interpretune and
finetuning-scheduler profile with one implementation: accelerator statistics and allocator history on CUDA, ROCm and
XPU, host memory, forward and saved-tensor hooks, and FSDP2 memory tracking. This extension keeps the configuration on
``it_cfg.memprofiler_cfg`` and points the core at the module's torch model.

In the core training loop, one profiled ``training_step`` call is a whole training iteration: the forward pass, the
backward pass and, at an accumulation window's end, the optimizer step, so a snapshot's end and an FSDP2 report see
the gradients and optimizer state. A ``memprofilable`` step method called inside that region is not profiled again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from interpretune.config.extension_cfgs import (
    DefaultMemHooks as DefaultMemHooks,
    MemProfilerCfg as MemProfilerCfg,
    MemProfilerFuncs as MemProfilerFuncs,
    MemProfilerHooks as MemProfilerHooks,
    MemProfilerSchedule as MemProfilerSchedule,
)


class _ProfiledModel:
    """The core profiler's view of an interpretune module: its torch model, carrying the profiler's configuration.

    The core profiles the object it connects to: it reads ``memprofiler_cfg`` from it, registers hooks on its
    ``modules()`` and reads the hook attributes off it. An interpretune module keeps its configuration on ``it_cfg`` and
    builds its model after its extensions connect, and an NNsight model wraps the torch module in ``_module``, so the
    torch model is resolved on every access.
    """

    def __init__(self, it_module: Any, core_cfg: Any) -> None:
        self._it_module = it_module
        self.memprofiler_cfg = core_cfg

    @property
    def torch_model(self) -> torch.nn.Module:
        model = self._it_module.model
        if model is None:
            raise RuntimeError("The memory profiler collects from the module's model, which is not built yet.")
        return getattr(model, "_module", model)

    def modules(self):
        return self.torch_model.modules()

    def get_submodule(self, target: str) -> torch.nn.Module:
        # The core resolves ``fsdp_mem_tracker_root_module`` here. Under the core loop's ``fsdp2`` placement
        # ``fully_shard`` wraps the interpretune module that holds the model, so an empty path names that module.
        if not target and _fsdp_module(self._it_module):
            return self._it_module
        return self.torch_model.get_submodule(target)

    def __getattr__(self, name: str) -> Any:
        # the hook attributes the core collects (`save_hook_attrs`) live on the torch model's root module
        return getattr(self.torch_model, name)


class MemProfiler:
    """The memory profiler extension, delegating to finetuning-scheduler's ``MemProfiler``.

    Attributes the extension does not define (``memory_stats``, ``memprofiler_cfg``, ``schedule``, ...) are the core
    profiler's. ``memprofiler_cfg`` is therefore the core's validated configuration, built from ``it_cfg`` at
    :meth:`connect`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._core: Any = None
        self._region: str | None = None
        # the current training phase, set by the loop that runs a phased schedule (``FTSCoreCallback``): the core
        # rebuilds its ``FSDP2MemTracker`` when the phase changes and records each FSDP2 report's phase
        self.phase_source: Callable[[], int | None] | None = None

    def connect(self, obj_ref: Any) -> None:
        """Connect the core profiler to ``obj_ref``'s model, configured from ``obj_ref.it_cfg.memprofiler_cfg``."""
        from finetuning_scheduler.profiling import MemProfiler as CoreMemProfiler

        self._core = CoreMemProfiler()
        target = _ProfiledModel(obj_ref, obj_ref.it_cfg.memprofiler_cfg.to_core())
        self._core.connect(
            target,
            optimizer=lambda: (getattr(obj_ref, "optimizers", None) or [None])[0],
            log_dir=lambda: getattr(obj_ref, "core_log_dir", None),
            phase=self._phase,
        )

    def _phase(self) -> int | None:
        return self.phase_source() if self.phase_source is not None else None

    @contextmanager
    def profile(self, fn_name: str) -> Iterator[None]:
        """Profile one call of ``fn_name``: snapshots at its start and end, and FSDP2 tracking where configured.

        A call made inside another profiled call is measured by the enclosing one, and is not profiled itself: it
        would retake the enclosing call's snapshots, and entering the ``FSDP2MemTracker`` a second time fails. With
        ``track_fsdp_mem``, the first profiled call builds the ``FSDP2MemTracker`` over the sharded root and the
        loop's optimizer.
        """
        if self._region is not None:
            yield
            return
        if self._core.memprofiler_cfg.track_fsdp_mem and self._core.fsdp_mem_tracker is None:
            # built at the first profiled call: by then the loop has sharded the model and built its optimizer
            self._core.maybe_init_fsdp_mem_tracker()
        self._region = fn_name
        try:
            with type(self._core).memprofile_ctx(self._core, fn_name=fn_name):
                yield
        finally:
            self._region = None

    def dump_memory_stats(self) -> None:
        """Save the collected statistics (and FSDP2 reports) under the profiler's artifact directory."""
        if self._core.memprof_log_dir is None:
            self._core.init_memprof_log_dir()
        self._core.dump_memory_stats()

    def __getattr__(self, name: str) -> Any:
        core = self.__dict__.get("_core")
        if core is None:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r} before it is connected")
        return getattr(core, name)


def _fsdp_module(module: Any) -> bool:
    """Whether ``fully_shard`` has been applied to ``module``."""
    if not torch.distributed.is_available():
        return False
    from torch.distributed.fsdp import FSDPModule

    return isinstance(module, FSDPModule)
