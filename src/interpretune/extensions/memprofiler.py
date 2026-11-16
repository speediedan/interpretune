"""The memory profiler extension: finetuning-scheduler's framework-neutral profiling core, connected to a module.

Collection, scheduling and reporting are the core's (``finetuning_scheduler.profiling``), so interpretune and
finetuning-scheduler profile with one implementation: accelerator statistics and allocator history on CUDA, ROCm and
XPU, host memory, forward and saved-tensor hooks, and FSDP2 memory tracking. This extension keeps the configuration on
``it_cfg.memprofiler_cfg`` and points the core at the module's torch model.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
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

    def connect(self, obj_ref: Any) -> None:
        """Connect the core profiler to ``obj_ref``'s model, configured from ``obj_ref.it_cfg.memprofiler_cfg``."""
        from finetuning_scheduler.profiling import MemProfiler as CoreMemProfiler

        self._core = CoreMemProfiler()
        target = _ProfiledModel(obj_ref, obj_ref.it_cfg.memprofiler_cfg.to_core())
        self._core.connect(
            target,
            optimizer=lambda: (getattr(obj_ref, "optimizers", None) or [None])[0],
            log_dir=lambda: getattr(obj_ref, "core_log_dir", None),
        )

    def profile(self, fn_name: str) -> AbstractContextManager:
        """Profile one call of ``fn_name``: snapshots at its start and end, and FSDP2 tracking where configured."""
        return type(self._core).memprofile_ctx(self._core, fn_name=fn_name)

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
