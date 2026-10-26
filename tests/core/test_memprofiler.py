import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.orchestration import run_it
from tests.runif import RunIf
from interpretune.extensions import MemProfiler, MemProfilerSchedule, MemProfilerCfg, MemProfilerHooks
from interpretune.protocol import CoreSteps


class TestClassMemProfiler:
    @RunIf(min_cuda_gpus=1, min_gpu_mem_gb=1.0)
    def test_memprofiler_remove_hooks(self, get_it_session__core_cust_memprof__initonly):
        fixture = get_it_session__core_cust_memprof__initonly
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        memprof_module = it_session.module
        memprof_module.memprofiler.memprofiler_cfg.schedule = MemProfilerSchedule(warmup_steps=1, max_step=3)
        memprof_module.memprofiler.memprofiler_cfg.retain_hooks_for_funcs = [CoreSteps.test_step]
        run_it(it_session=it_session, test_cfg=test_cfg)

    def test_memprofiler_nocuda_warn(self):
        test_memprof_cfg = {"enabled": True, "cuda_allocator_history": True}
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.warns(UserWarning, match="Disabling CUDA memory profiling"):
                memprofiler_cfg = MemProfilerCfg(**test_memprof_cfg)
        assert memprofiler_cfg.cuda_allocator_history is False

    def test_memprofiler_no_hooks(self):
        test_memprof_cfg = {
            "enabled": True,
            "memory_hooks": MemProfilerHooks(pre_forward_hooks=[], post_forward_hooks=[], reset_state_hooks=[]),
        }
        with pytest.warns(UserWarning, match="but MemProfilerHooks does not have"):
            memprofiler_cfg = MemProfilerCfg(**test_memprof_cfg)
        assert memprofiler_cfg.cuda_allocator_history is False

    def test_memprofiler_dump_stats_without_cuda(self, tmp_path):
        """With CUDA unavailable, allocator history is disabled, so setup creates no snapshot directory; dumping
        the collected stats at session end must still work (it raised an ``AssertionError`` on hosts without
        CUDA)."""
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.warns(UserWarning, match="Disabling CUDA memory profiling"):
                memprofiler_cfg = MemProfilerCfg(enabled=True, cuda_allocator_history=True)
        module = SimpleNamespace(it_cfg=SimpleNamespace(memprofiler_cfg=memprofiler_cfg), core_log_dir=tmp_path)
        memprofiler = MemProfiler()
        memprofiler.connect(module)
        memprofiler.memory_stats["0.test.0.0.end"] = {"rss_diff": 1}
        memprofiler.dump_memory_stats()
        with open(tmp_path / "memprofiler" / "memory_stats.pickle", "rb") as f:
            assert pickle.load(f) == {"0.test.0.0.end": {"rss_diff": 1}}
