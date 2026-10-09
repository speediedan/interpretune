"""PyTorch profiler selection for the core runners (interpretune#11)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from interpretune.config.profiling import ProfilerCfg
from interpretune.extensions.memprofiler import MemProfilerCfg
from interpretune.utils import MisconfigurationException

# One active step per cycle, two cycles: two profiled batches must write exactly two traces, which only happens when
# the loop steps the profiler once per batch.
TWO_WINDOWS = {"wait": 0, "warmup": 0, "active": 1, "repeat": 2}


def _traces(trace_dir):
    return sorted(trace_dir.glob("*.pt.trace.json"))


class TestProfilerCfg:
    def test_default_is_off(self):
        assert ProfilerCfg().which == "none"

    def test_pytorch_selection_with_settings_passes(self):
        cfg = ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"activities": ["cpu"], "record_shapes": True})
        assert cfg.which == "pytorch"

    def test_settings_without_the_selection_refuse(self):
        with pytest.raises(MisconfigurationException, match="never runs"):
            ProfilerCfg(pytorch_profiler_cfg={"activities": ["cpu"]})

    def test_unknown_setting_refuses_by_name(self):
        with pytest.raises(MisconfigurationException, match=r"unsupported keys \['on_trace_ready'\]"):
            ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"on_trace_ready": "x"})

    def test_unknown_activity_refuses_by_name(self):
        with pytest.raises(MisconfigurationException, match=r"unknown entries \['gpu'\]"):
            ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"activities": ["gpu"]})

    def test_torch_kwargs_translate_the_serializable_settings(self, tmp_path):
        cfg = ProfilerCfg(
            which="pytorch",
            pytorch_profiler_cfg={"activities": ["cpu"], "schedule": TWO_WINDOWS, "with_stack": True},
        )
        kwargs = cfg.torch_profiler_kwargs(default_trace_dir=tmp_path)
        assert kwargs["activities"] == [torch.profiler.ProfilerActivity.CPU]
        assert callable(kwargs["schedule"]) and callable(kwargs["on_trace_ready"])
        assert kwargs["with_stack"] is True
        assert "trace_dir" not in kwargs


class TestITConfigProfilerSelection:
    def test_pytorch_profiler_alone_passes(self):
        from interpretune.config.module import ITConfig

        assert ITConfig(profiler_cfg=ProfilerCfg(which="pytorch")).profiler_cfg.which == "pytorch"

    def test_memprofiler_alone_passes(self):
        from interpretune.config.module import ITConfig

        cfg = ITConfig(memprofiler_cfg=MemProfilerCfg(enabled=True))
        assert cfg.profiler_cfg.which == "none"

    def test_both_profilers_refuse(self):
        from interpretune.config.module import ITConfig

        with pytest.raises(MisconfigurationException, match="two profilers"):
            ITConfig(profiler_cfg=ProfilerCfg(which="pytorch"), memprofiler_cfg=MemProfilerCfg(enabled=True))


def _profiled_module(tmp_path, which="pytorch", pytorch_profiler_cfg=None):
    module = MagicMock()
    module.it_cfg = SimpleNamespace(
        profiler_cfg=ProfilerCfg(
            which=which, pytorch_profiler_cfg=pytorch_profiler_cfg if pytorch_profiler_cfg is not None else {}
        )
    )
    module.core_log_dir = str(tmp_path)
    module.batch_to_device = MagicMock(side_effect=lambda batch: batch)
    module.dtype = torch.float32
    module.model = MagicMock()
    module.optimizers = [MagicMock()]

    weight = torch.ones(2, requires_grad=True)  # the train loop backpropagates the step output

    def step(batch, batch_idx):
        return (batch["input"] @ weight).sum()

    module.training_step = MagicMock(side_effect=step)
    module.validation_step = MagicMock(return_value=None)
    module.test_step = MagicMock(side_effect=step)
    del module.on_test_epoch_end
    del module._logged_metrics
    return module


def _datamodule(n_batches=2):
    batches = [{"input": torch.ones(2, 2) * (i + 1)} for i in range(n_batches)]
    loader = MagicMock()
    loader.__iter__.side_effect = lambda: iter(batches)
    datamodule = MagicMock()
    datamodule.train_dataloader.return_value = loader
    datamodule.test_dataloader.return_value = loader
    del datamodule.val_dataloader
    return datamodule


class TestCoreRunnersProfile:
    """The selection drives a real profiler in the core loops, stepped once per batch."""

    def test_test_loop_writes_one_trace_per_scheduled_window(self, tmp_path):
        from interpretune.runners.core import core_test_loop

        module = _profiled_module(tmp_path, pytorch_profiler_cfg={"activities": ["cpu"], "schedule": TWO_WINDOWS})
        core_test_loop(module=module, datamodule=_datamodule(2), limit_test_batches=-1)
        assert len(_traces(tmp_path / "pytorch_profiler")) == 2

    def test_train_loop_writes_one_trace_per_scheduled_window(self, tmp_path):
        from interpretune.runners.core import core_train_loop

        trace_dir = tmp_path / "custom_traces"
        module = _profiled_module(
            tmp_path,
            pytorch_profiler_cfg={"activities": ["cpu"], "schedule": TWO_WINDOWS, "trace_dir": str(trace_dir)},
        )
        core_train_loop(
            module=module, datamodule=_datamodule(2), limit_train_batches=-1, limit_val_batches=-1, max_epochs=1
        )
        assert len(_traces(trace_dir)) == 2
        assert not (tmp_path / "pytorch_profiler").exists()

    def test_analysis_loop_writes_one_trace_per_scheduled_window(self, tmp_path):
        from interpretune.runners.analysis import analysis_store_generator

        module = _profiled_module(tmp_path, pytorch_profiler_cfg={"activities": ["cpu"], "schedule": TWO_WINDOWS})
        module.analysis_step = MagicMock(side_effect=lambda batch, batch_idx: iter([{"batch_idx": batch_idx}]))
        rows = list(analysis_store_generator(module=module, datamodule=_datamodule(2), max_epochs=1))
        assert [row["batch_idx"] for row in rows] == [0, 1]
        assert len(_traces(tmp_path / "pytorch_profiler")) == 2

    def test_unselected_profiler_writes_nothing(self, tmp_path):
        from interpretune.runners.core import core_test_loop

        module = _profiled_module(tmp_path, which="none")
        core_test_loop(module=module, datamodule=_datamodule(2), limit_test_batches=-1)
        assert not (tmp_path / "pytorch_profiler").exists()


def test_lightning_composition_refuses_the_core_runner_profiler():
    """A Lightning Trainer runs its own loops, so the selection would profile nothing there."""
    pytest.importorskip("lightning")
    from interpretune.adapters.lightning import LightningAdapter

    stub = SimpleNamespace(it_cfg=SimpleNamespace(profiler_cfg=ProfilerCfg(which="pytorch")))
    with pytest.raises(MisconfigurationException, match="--trainer.profiler"):
        LightningAdapter.setup(stub)
