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


def test_memprofiler_selected_without_section_passes_structurally():
    """The bare selection is well-formed here; activation is decided at ITConfig level."""
    assert ProfilerCfg(which="memprofiler").which == "memprofiler"


class TestITConfigProfilerSelection:
    """The invariant covers the runtime-read configuration, not just the selector."""

    def test_default_config_passes(self):
        from interpretune.config.module import ITConfig

        assert ITConfig().profiler_cfg.which == "none"

    def test_pytorch_selection_refuses_without_runner_integration(self):
        from interpretune.config.module import ITConfig

        with pytest.raises(MisconfigurationException, match="no runner reads"):
            ITConfig(profiler_cfg=ProfilerCfg(which="pytorch"))
        with pytest.raises(MisconfigurationException, match="no runner reads"):
            ITConfig(profiler_cfg=ProfilerCfg(pytorch_profiler_cfg={"activities": ["cpu"]}))

    def test_nested_memprofiler_section_refuses_as_duplicate_source(self):
        from interpretune.config.module import ITConfig

        with pytest.raises(MisconfigurationException, match="duplicates it_cfg.memprofiler_cfg"):
            ITConfig(profiler_cfg=ProfilerCfg(which="memprofiler", memprofiler_cfg=MemProfilerCfg(enabled=True)))

    def test_memprofiler_selection_needs_the_extension_field(self):
        from interpretune.config.module import ITConfig

        with pytest.raises(MisconfigurationException, match="needs it_cfg.memprofiler_cfg enabled"):
            ITConfig(profiler_cfg=ProfilerCfg(which="memprofiler"))

    def test_memprofiler_selection_with_extension_enabled_passes(self):
        from interpretune.config.module import ITConfig

        cfg = ITConfig(
            profiler_cfg=ProfilerCfg(which="memprofiler"),
            memprofiler_cfg=MemProfilerCfg(enabled=True),
        )
        assert cfg.profiler_cfg.which == "memprofiler"
