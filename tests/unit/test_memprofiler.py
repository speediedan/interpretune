from unittest.mock import patch

import pytest

from tests.orchestration import run_it
from tests.runif import RunIf
from interpretune.extensions import MemProfilerSchedule, MemProfilerCfg, MemProfilerHooks
from interpretune.protocol import CoreSteps

class TestClassMemProfiler:

    @RunIf(min_cuda_gpus=1)
    def test_memprofiler_remove_hooks(self, get_it_session__core_cust_memprof__initonly):
        fixture = get_it_session__core_cust_memprof__initonly
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        memprof_module = it_session.module
        memprof_module.memprofiler.memprofiler_cfg.schedule = MemProfilerSchedule(warmup_steps=1, max_step=3)
        memprof_module.memprofiler.memprofiler_cfg.retain_hooks_for_funcs = [CoreSteps.test_step]
        run_it(it_session=it_session, test_cfg=test_cfg)

    def test_memprofiler_nocuda_warn(self):
        test_memprof_cfg = {"enabled": True, "cuda_allocator_history": True}
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.warns(UserWarning, match="Disabling CUDA memory profiling"):
                memprofiler_cfg = MemProfilerCfg(**test_memprof_cfg)
        assert memprofiler_cfg.cuda_allocator_history is False

    def test_memprofiler_no_hooks(self):
        test_memprof_cfg = {"enabled": True,
                            "memory_hooks": MemProfilerHooks(pre_forward_hooks=[], post_forward_hooks=[],
                                                             reset_state_hooks=[])}
        with pytest.warns(UserWarning, match="but MemProfilerHooks does not have"):
            memprofiler_cfg = MemProfilerCfg(**test_memprof_cfg)
        assert memprofiler_cfg.cuda_allocator_history is False
