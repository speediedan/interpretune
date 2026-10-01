"""ProfilerCfg selection invariant (interpretune#11, slice 1)."""

from __future__ import annotations

import pytest

from interpretune.config.profiling import ProfilerCfg
from interpretune.extensions.memprofiler import MemProfilerCfg
from interpretune.utils import MisconfigurationException


def test_default_is_inert():
    cfg = ProfilerCfg()
    assert cfg.which == "none"


def test_single_backends_pass():
    assert ProfilerCfg(which="pytorch").which == "pytorch"
    assert ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"activities": ["cpu"]})
    mem = MemProfilerCfg(enabled=True)
    assert ProfilerCfg(which="memprofiler", memprofiler_cfg=mem).memprofiler_cfg is mem


def test_disabled_mem_section_is_inert():
    assert ProfilerCfg(memprofiler_cfg=MemProfilerCfg(enabled=False)).which == "none"


def test_two_active_profilers_refuse_by_name():
    with pytest.raises(MisconfigurationException, match="two active profilers"):
        ProfilerCfg(which="pytorch", memprofiler_cfg=MemProfilerCfg(enabled=True))
    with pytest.raises(MisconfigurationException, match="two active profilers"):
        ProfilerCfg(pytorch_profiler_cfg={"activities": ["cpu"]}, memprofiler_cfg=MemProfilerCfg(enabled=True))


def test_memprofiler_selected_without_section_refuses():
    with pytest.raises(MisconfigurationException, match="needs a memprofiler_cfg section"):
        ProfilerCfg(which="memprofiler")
