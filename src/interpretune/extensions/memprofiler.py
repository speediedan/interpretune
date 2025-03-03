import os
import pickle
from dataclasses import fields, field, dataclass
from typing import Any
from collections.abc import Callable
from enum import Enum
from collections import defaultdict
from pathlib import Path

import torch
from psutil import Process

from interpretune.utils import rank_zero_warn, resolve_funcs, _get_rank, rank_zero_only
from interpretune.protocol import AutoStrEnum, CoreSteps
from interpretune.config import ITSerializableCfg


class DefaultMemHooks(AutoStrEnum):
    pre_forward = 'interpretune.extensions.memprofiler._hook_npp_pre_forward'
    post_forward = 'interpretune.extensions.memprofiler._hook_npp_post_forward'
    reset_state = 'interpretune.extensions.memprofiler._reset_memory_hooks_state'

@dataclass(kw_only=True)
class MemProfilerHooks(ITSerializableCfg):
    pre_forward_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.pre_forward.value])
    post_forward_hooks: list[str| Callable] = field(default_factory=lambda: [DefaultMemHooks.post_forward.value])
    # the provided reset_state_hooks will be called with the model and the `save_hook_attrs` list
    reset_state_hooks: list[str | Callable] = field(default_factory=lambda: [DefaultMemHooks.reset_state.value])

@dataclass(kw_only=True)
class MemProfilerFuncs(ITSerializableCfg): # can specify arbitrary list of `memprofilable` decorated function names
    cuda: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))
    cpu: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))
    cuda_allocator_history: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))

@dataclass(kw_only=True)
class MemProfilerSchedule(ITSerializableCfg):
    # keeping schedule simple as possible for now, may expand to accommodate more flexible schedules in the future
    warmup_steps: int = 0
    max_step: int | None = None

@dataclass(kw_only=True)
class MemProfilerCfg(ITSerializableCfg):
    enabled: bool = False
    cuda_allocator_history: bool = False
    schedule: MemProfilerSchedule = field(default_factory=MemProfilerSchedule)
    save_dir: str | os.PathLike | None = None
    enabled_funcs: MemProfilerFuncs = field(default_factory=MemProfilerFuncs)
    enable_memory_hooks: bool = True
    enable_saved_tensors_hooks: bool = True
    memory_hooks: MemProfilerHooks = field(default_factory=MemProfilerHooks)
    saved_tensors_funcs: list = field(default_factory=lambda: list(('interpretune.extensions.memprofiler._npp_hook',
                                                                    lambda x: x)))
    # if you add custom hooks, make sure to add the desired module state attributes to save to `save_hook_attrs`
    save_hook_attrs: list = field(default_factory=lambda: ["rss_pre_forward", "rss_post_forward", "rss_diff",
                                                           "npp_pre_forward", "npp_post_forward", "npp_diff"])
    # since we cannot reliably ascertain when all MemProfilerFuncs will be executed, memory hooks will
    # only be removed once the funcs in this list have reached `max_step`
    retain_hooks_for_funcs: list[str | Enum] = field(default_factory=lambda: list(step.name for step in CoreSteps))

    def __post_init__(self) -> None:
        if not torch.cuda.is_available() and self.enabled and any((self.enabled_funcs.cuda_allocator_history,
                                                                   self.enabled_funcs.cuda,
                                                                   self.cuda_allocator_history)):
            rank_zero_warn("Disabling CUDA memory profiling functionality since no CUDA device detected.")
            self.enabled_funcs.cuda, self.enabled_funcs.cuda_allocator_history = [], []
            self.cuda_allocator_history = False
        has_hooks = any(getattr(self.memory_hooks, ht.name) for ht in fields(self.memory_hooks))
        if self.enabled and not has_hooks:
            rank_zero_warn("MemProfilerCfg is configured to enable memory hooks but MemProfilerHooks does not have"
                           " any specified.")

# TODO: enable once these hooks are added
# @dataclass(kw_only=True)
# class PyTorchProfilerCfg(ITSerializableCfg):
#     # pytorch_profiler_enabled: bool = False
#     # pytorch_profiler_cfg: Dict[str, Any] = field(default_factory=dict)

# accessed in global scope to track non-parameter packed bytes (npp) as a simple proxy (ceiling) for activation memory
_npp_bytes = 0

def _hook_npp_pre_forward(module, *args, **kwargs):
    mem = module.mem_info_handle()
    global _npp_bytes
    module.npp_pre_forward = _npp_bytes
    module.rss_pre_forward = mem.rss
    return None

def _hook_npp_post_forward(module, *args, **kwargs):
    global _npp_bytes
    module.npp_post_forward = _npp_bytes
    module.npp_diff = module.npp_post_forward - module.npp_pre_forward
    mem = module.mem_info_handle()
    module.rss_post_forward = mem.rss
    rss_diff = module.rss_post_forward - module.rss_pre_forward
    module.rss_diff = rss_diff + (module.rss_diff if hasattr(module, "rss_diff") else 0)
    return None

def _reset_memory_hooks_state(model, reset_attrs: list[str]):
    global _npp_bytes
    _npp_bytes = 0
    for module in model.modules():
        for attr in reset_attrs:
            setattr(module, attr, 0)

def _npp_hook(x):
    global _npp_bytes
    if not isinstance(x, torch.nn.Parameter):
        _npp_bytes += x.nbytes
    return x

class MemProfiler:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.memory_stats = defaultdict(dict)
        self._enabled = {}
        self._module = None
        self._cuda_snapshot_dir = None
        self._curr_pid = None
        self._rank = _get_rank() or 0  # for future use, currently only single rank supported
        self._snap_indices = {}
        self._configured_hooks = {}
        self._saved_tensors_funcs = []
        self._hook_handles = defaultdict(list)
        self._done_prof_funcs = []

    def connect(self, obj_ref: Any) -> None:
        self._module = obj_ref
        self._curr_pid = Process(os.getpid())
        if self.memprofiler_cfg.enable_saved_tensors_hooks:
            self._saved_tensors_funcs = resolve_funcs(cfg_obj=self.memprofiler_cfg, func_type='saved_tensors_funcs')

    @property
    def memprofiler_cfg(self) -> MemProfilerCfg:
        return self._module.it_cfg.memprofiler_cfg

    @property
    def schedule(self) -> int:
        return self._module.it_cfg.memprofiler_cfg.schedule

    def remove_memprofiler_hooks(self) -> None:
        for handle_list in self._hook_handles.values():
            for handle in handle_list:
                handle.remove()

    def exec_reset_state_hooks(self) -> None:
        for hook in self._configured_hooks["reset_state_hooks"]:
            hook(self._module.model, self.memprofiler_cfg.save_hook_attrs)

    def add_memprofiler_hooks(self) -> None:
        # TODO: extend supported hook points (e.g. backward, etc.) and if/once supporting additional hook points,
        # use a hook_type to registration function mapping
        memory_hooks_cfg = self.memprofiler_cfg.memory_hooks
        for supported_hooks in fields(memory_hooks_cfg):
            if getattr(memory_hooks_cfg, supported_hooks.name):
                self._configured_hooks[supported_hooks.name] = resolve_funcs(cfg_obj=memory_hooks_cfg,
                                                                             func_type=supported_hooks.name)
        for module in self._module.model.modules():
            module.mem_info_handle = self._curr_pid.memory_info
            for hook_func in self._configured_hooks["pre_forward_hooks"]:
                self._hook_handles[hook_func].append(module.register_forward_pre_hook(hook_func))
            for hook_func in self._configured_hooks["post_forward_hooks"]:
                self._hook_handles[hook_func].append(module.register_forward_hook(hook_func))
        self.exec_reset_state_hooks()

    def init_cuda_snapshots_dir(self) -> None:
        self._cuda_snapshot_dir = self.memprofiler_cfg.save_dir or Path(self._module.core_log_dir) / "memprofiler"
        self._cuda_snapshot_dir = Path(self._cuda_snapshot_dir)  # ensure the dir is a Path
        self._cuda_snapshot_dir.mkdir(exist_ok=True, parents=True)

    def cuda_allocator_history_snap(self, snap_key: tuple) -> dict:
        cuda_snapshot_file = (self._cuda_snapshot_dir / f"cuda_alloc_{snap_key}.pickle")
        torch.cuda.memory._dump_snapshot(cuda_snapshot_file)

    def done(self, step_idx: int) -> bool:
        return self.schedule.max_step and step_idx >= self.schedule.max_step

    def _process_hooks(self, snap_key) -> None:
        if self.memprofiler_cfg.enable_memory_hooks:
            if len(self._hook_handles) == 0:
                self.add_memprofiler_hooks()
            collected = {attr: getattr(self._module.model, attr, None) for attr in self.memprofiler_cfg.save_hook_attrs}
            self.memory_stats[snap_key].update(collected)

    def _collect_snap(self, snap_key, reset_mem_hooks: bool = False) -> None:
        _, phase, *_ = snap_key
        snap_key = ".".join(map(str, snap_key))
        mem_cfg = self.memprofiler_cfg
        self._process_hooks(snap_key)
        if phase in mem_cfg.enabled_funcs.cpu:
            mem = self._curr_pid.memory_info()
            self.memory_stats[snap_key].update({"rss": mem.rss, "vms": mem.vms})
        if phase in mem_cfg.enabled_funcs.cuda:
            self.memory_stats[snap_key].update(torch.cuda.memory_stats())
        if phase in mem_cfg.enabled_funcs.cuda_allocator_history and mem_cfg.cuda_allocator_history:
            self.cuda_allocator_history_snap(snap_key)
        if mem_cfg.enable_memory_hooks and reset_mem_hooks:
            self.exec_reset_state_hooks()

    @property
    def _should_remove_hooks(self) -> bool:
        return all(func in self._done_prof_funcs for func in self.memprofiler_cfg.retain_hooks_for_funcs)

    def teardown_prof(self, phase: str, step_ctx: str) -> None:
        self._enabled[(phase, step_ctx)] = False
        if not any(self._enabled[(phase, step_ctx)] for step_ctx in ["start", "end"]):
            self._done_prof_funcs.append(CoreSteps[phase])
        if self.memprofiler_cfg.retain_hooks_for_funcs and self._should_remove_hooks:
            self.remove_memprofiler_hooks()
            self.memprofiler_cfg.enable_memory_hooks = False

    def gen_snap_keys(self, phase: str, step_ctx: str, epoch_idx: int | None = None,
                      step_idx: int | None = None) -> tuple[int, int, tuple]:
        # NOTE [Memprofiler Key Format]:
        # snap key format is rank.phase.epoch_idx.step_idx.step_ctx
        # e.g. 0.training_step.0.0.end keys hook output for the end of training step 0, epoch 0 for rank 0
        # 0.training_step.1.2.start keys mem stats for the start of training step 2, epoch 1 for rank 0
        epoch_idx = next(e_idx for e_idx in (epoch_idx, self._module.current_epoch) if e_idx is not None)
        if step_idx is None:
            step_idx = self._snap_indices[(phase, step_ctx)]
        return epoch_idx, step_idx, (self._rank, phase, epoch_idx, step_idx, step_ctx)

    def maybe_init_phase(self, phase: str, step_ctx: str) -> None:
        if not self._snap_indices.get((phase, step_ctx), None):
            self._snap_indices[(phase, step_ctx)] = 0
            self._enabled[(phase, step_ctx)] = True

    def snap(self, phase: str, step_ctx: str, epoch_idx: int | None = None, step_idx: int | None = None,
             reset_mem_hooks: bool = False) -> None:
        self.maybe_init_phase(phase, step_ctx)
        if not self._enabled[(phase, step_ctx)]:
            return
        epoch_idx, step_idx, snap_key = self.gen_snap_keys(phase, step_ctx, epoch_idx, step_idx)
        if step_idx >= self.schedule.warmup_steps:
            if not self.done(step_idx):
                self._collect_snap(snap_key, reset_mem_hooks)
            else:
                self.teardown_prof(phase, step_ctx)
        if self._enabled[(phase, step_ctx)]:
            self._snap_indices[(phase, step_ctx)] += 1

    @rank_zero_only
    def dump_memory_stats(self) -> None:
        # TODO: all gather memory stats in the future if/when multiple ranks are supported
        filename = self._cuda_snapshot_dir / "memory_stats.pickle"
        with open(filename, "wb") as f:
            pickle.dump(self.memory_stats, f)
