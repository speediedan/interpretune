from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from tests.orchestration import run_it
from tests.runif import RunIf
from interpretune.extensions import MemProfiler, MemProfilerSchedule, MemProfilerCfg, MemProfilerHooks
from interpretune.extensions.memprofiler import MemProfilerFuncs
from interpretune.utils import MisconfigurationException


class TestClassMemProfiler:
    @RunIf(min_cuda_gpus=1, min_gpu_mem_gb=1.0)
    def test_memprofiler_remove_hooks(self, get_it_session__core_cust_memprof__initonly):
        fixture = get_it_session__core_cust_memprof__initonly
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        memprof_module = it_session.module
        memprof_module.memprofiler.memprofiler_cfg.schedule = MemProfilerSchedule(warmup_iters=1, max_iter=3)
        memprof_module.memprofiler.memprofiler_cfg.retain_hooks_for_funcs = {"test_step"}
        run_it(it_session=it_session, test_cfg=test_cfg)

    def test_memprofiler_no_accelerator_warn(self):
        memprofiler_cfg = MemProfilerCfg(enabled=True, allocator_history=True)
        with patch("finetuning_scheduler.profiling.config.accelerator_backend", return_value=None):
            with pytest.warns(UserWarning, match="Disabling accelerator memory profiling"):
                core_cfg = memprofiler_cfg.to_core()
        assert core_cfg.allocator_history is False
        assert not core_cfg.collect_funcs.accelerator and not core_cfg.collect_funcs.allocator_history

    def test_memprofiler_no_hooks(self):
        memprofiler_cfg = MemProfilerCfg(
            enabled=True,
            memory_hooks=MemProfilerHooks(pre_forward_hooks=[], post_forward_hooks=[], reset_state_hooks=[]),
        )
        with pytest.warns(UserWarning, match="but MemProfilerHooks does not have"):
            memprofiler_cfg.to_core()

    def test_memprofiler_refused_without_finetuning_scheduler(self):
        """The profiler runs on finetuning-scheduler's core, an optional dependency; enabling it without the core
        is refused naming the extra that installs it."""
        with patch("interpretune.config.extension_cfgs._FTS_AVAILABLE", False):
            MemProfilerCfg()  # a disabled profiler needs nothing
            with pytest.raises(MisconfigurationException, match=r"interpretune\[tune\]"):
                MemProfilerCfg(enabled=True)

    def test_the_configuration_schema_is_the_cores(self):
        """Each configuration class carries the core's fields, so a configuration reads the same in both projects;
        the core's deprecated aliases are left out."""
        from finetuning_scheduler.profiling import config as core_cfg

        deprecated = set(core_cfg._DEPRECATED_FUNC_KEYS) | {"cuda_allocator_history"}
        for ours, core in (
            (MemProfilerCfg, core_cfg.MemProfilerCfg),
            (MemProfilerFuncs, core_cfg.MemProfilerFuncs),
            (MemProfilerSchedule, core_cfg.MemProfilerSchedule),
            (MemProfilerHooks, core_cfg.MemProfilerHooks),
        ):
            assert {f.name for f in fields(ours)} == {f.name for f in fields(core)} - deprecated, ours.__name__

    def test_to_core_composes_the_default_funcs(self):
        core_cfg = MemProfilerCfg(
            enabled=True, collect_funcs=MemProfilerFuncs(default=["test_step"], cpu=["training_step"])
        ).to_core()
        assert core_cfg.collect_funcs.cpu == {"test_step", "training_step"}
        assert core_cfg.schedule.max_iter == 1  # an unset max_iter collects one iteration after the warmup

    def test_memprofiler_collects_from_the_wrapped_model_and_dumps(self, tmp_path):
        """The core profiles the module's torch model (unwrapping an NNsight model's ``_module``), and the
        statistics are saved under ``<core_log_dir>/memprofiler``."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        memprofiler_cfg = MemProfilerCfg(enabled=True, collect_funcs=MemProfilerFuncs(default=["test_step"]))
        module = SimpleNamespace(
            it_cfg=SimpleNamespace(memprofiler_cfg=memprofiler_cfg),
            core_log_dir=tmp_path,
            model=SimpleNamespace(_module=model),
        )
        memprofiler = MemProfiler()
        memprofiler.connect(module)
        for _ in range(2):
            with memprofiler.profile("test_step"):
                model(torch.ones(1, 4))
        memprofiler.dump_memory_stats()
        with open(tmp_path / "memprofiler" / "rank_0_memory_stats.yaml") as f:
            stats = yaml.safe_load(f)
        # one iteration is collected (the default schedule), at its start and end, with the hook attributes the
        # forward hooks set on the root module
        assert set(stats) == {"0.test_step.1.start", "0.test_step.1.end"}
        assert "npp_diff" in stats["0.test_step.1.end"] and "rss" in stats["0.test_step.1.end"]

    def test_a_call_inside_a_profiled_call_is_not_profiled_again(self, tmp_path):
        """The core loop profiles a whole training iteration as one ``training_step`` call, and the step method it
        runs may be ``memprofilable`` itself: the inner call is measured by the outer one and never reaches the
        core, which would retake the outer call's snapshots and enter an ``FSDP2MemTracker`` a second time (the
        two-rank fsdp2 profile in ``test_core_loop_distributed.py`` fails that way without this)."""
        from finetuning_scheduler.profiling import MemProfiler as CoreMemProfiler

        model = torch.nn.Linear(4, 2)
        memprofiler_cfg = MemProfilerCfg(
            enabled=True,
            collect_funcs=MemProfilerFuncs(default=["training_step"]),
            schedule=MemProfilerSchedule(warmup_iters=0, max_iter=2),
        )
        module = SimpleNamespace(
            it_cfg=SimpleNamespace(memprofiler_cfg=memprofiler_cfg), core_log_dir=tmp_path, model=model
        )
        memprofiler = MemProfiler()
        memprofiler.connect(module)
        with patch.object(CoreMemProfiler, "memprofile_ctx", wraps=CoreMemProfiler.memprofile_ctx) as core_ctx:
            for _ in range(3):
                with memprofiler.profile("training_step"):
                    with memprofiler.profile("training_step"):
                        model(torch.ones(1, 4)).sum().backward()
        assert core_ctx.call_count == 3
        assert set(memprofiler.memory_stats) == {f"0.training_step.{i}.{at}" for i in (1, 2) for at in ("start", "end")}

    def test_the_phase_source_is_the_cores_phase(self):
        """The loop running a phased schedule (``FTSCoreCallback``) sets ``phase_source``; the core reads the phase
        through it when it records an FSDP2 report or rebuilds its tracker."""
        from finetuning_scheduler.profiling.memprofiler import _resolved

        module = SimpleNamespace(
            it_cfg=SimpleNamespace(memprofiler_cfg=MemProfilerCfg(enabled=True)), model=torch.nn.Linear(2, 2)
        )
        memprofiler = MemProfiler()
        memprofiler.connect(module)
        assert _resolved(memprofiler._core._phase) is None
        depth = [0]
        memprofiler.phase_source = lambda: depth[0]
        depth[0] = 1
        assert _resolved(memprofiler._core._phase) == 1

    def test_memprofiler_before_connect_names_the_missing_step(self):
        with pytest.raises(AttributeError, match="before it is connected"):
            MemProfiler().memory_stats
