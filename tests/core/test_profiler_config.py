"""PyTorch profiler selection for the core runners (interpretune#11)."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from interpretune.config.profiling import ProfilerCfg
from interpretune.extensions.memprofiler import MemProfilerCfg
from interpretune.utils import MisconfigurationException
from tests.runif import RunIf

# One active step per cycle, two cycles: two profiled batches must write exactly two traces, which only happens when
# the loop steps the profiler once per batch.
TWO_WINDOWS = {"wait": 0, "warmup": 0, "active": 1, "repeat": 2}
SUMMARY = {"filename": "prof", "sort_by_key": "self_cpu_time_total", "row_limit": 3, "max_name_column_width": 30}


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

    def test_summary_options_stay_out_of_the_torch_kwargs(self, tmp_path):
        cfg = ProfilerCfg(which="pytorch", pytorch_profiler_cfg={**SUMMARY, "record_shapes": True})
        kwargs = cfg.torch_profiler_kwargs(default_trace_dir=tmp_path)
        assert not set(kwargs) & {"filename", "sort_by_key", "row_limit", "max_name_column_width"}
        assert cfg.summary_options() == SUMMARY

    def test_summary_defaults_fill_in_when_only_a_filename_is_set(self):
        cfg = ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"filename": "prof"})
        assert cfg.summary_options() == {
            "filename": "prof",
            "sort_by_key": "cpu_time_total",
            "row_limit": 20,
            "max_name_column_width": 55,
        }
        assert ProfilerCfg(which="pytorch").summary_options() is None

    def test_summary_options_without_a_filename_refuse(self):
        with pytest.raises(MisconfigurationException, match=r"sets \['row_limit', 'sort_by_key'\] but no `filename`"):
            ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"row_limit": 5, "sort_by_key": "count"})

    @pytest.mark.parametrize("key", ["cpu_time_total", "cuda_time_total", "self_cuda_time_total", "xpu_time_total"])
    def test_accelerator_sort_keys_pass(self, key):
        """Torch maps an accelerator's name to `device` before sorting, so the accelerator spellings are
        columns."""
        ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"filename": "prof", "sort_by_key": key})

    def test_unknown_sort_key_refuses(self):
        with pytest.raises(MisconfigurationException, match="sort_by_key='gpu_time' is not a column"):
            ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"filename": "prof", "sort_by_key": "gpu_time"})

    @pytest.mark.parametrize("value", [0, -3, "50", True])
    def test_non_positive_or_non_integer_row_limit_refuses(self, value):
        with pytest.raises(MisconfigurationException, match="row_limit must be a positive integer"):
            ProfilerCfg(which="pytorch", pytorch_profiler_cfg={"filename": "prof", "row_limit": value})


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


def _run_phase(module, phase, n_batches):
    from interpretune.runners.analysis import analysis_store_generator
    from interpretune.runners.core import core_test_loop, core_train_loop

    datamodule = _datamodule(n_batches)
    if phase == "train":
        core_train_loop(
            module=module, datamodule=datamodule, limit_train_batches=-1, limit_val_batches=-1, max_epochs=1
        )
    elif phase == "test":
        core_test_loop(module=module, datamodule=datamodule, limit_test_batches=-1)
    else:  # an analysis step that runs operators, so the profiler has something to summarize
        module.analysis_step = MagicMock(
            side_effect=lambda batch, batch_idx: iter([{"x": (batch["input"] @ batch["input"]).relu().sum() * 2}])
        )
        list(analysis_store_generator(module=module, datamodule=datamodule, max_epochs=1))


def _table_rows(table):
    """The operator rows of a profiler summary table: the lines between its header rule and its closing rule."""
    lines = table.strip().splitlines()
    rules = [i for i, line in enumerate(lines) if line.startswith("---")]
    return lines[rules[1] + 1 : rules[2]]


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

    @pytest.mark.parametrize("phase", ["train", "test", "analysis"])
    def test_filename_writes_a_summary_named_by_phase(self, tmp_path, phase):
        module = _profiled_module(tmp_path, pytorch_profiler_cfg={"activities": ["cpu"], **SUMMARY})
        _run_phase(module, phase, n_batches=2)
        summaries = sorted((tmp_path / "pytorch_profiler").glob("*.txt"))
        assert [p.name for p in summaries] == [f"{phase}-prof.txt"]
        table = summaries[0].read_text(encoding="utf-8")
        assert "Self CPU" in table
        assert len(_table_rows(table)) == SUMMARY["row_limit"]

    def test_row_limit_bounds_the_summary(self, tmp_path):
        """Positive control for the row count above: a larger limit shows more of the same run's operators."""
        module = _profiled_module(tmp_path, pytorch_profiler_cfg={"activities": ["cpu"], **SUMMARY, "row_limit": 50})
        _run_phase(module, "test", n_batches=2)
        assert len(_table_rows((tmp_path / "pytorch_profiler" / "test-prof.txt").read_text())) > SUMMARY["row_limit"]

    def test_no_recorded_window_warns_and_writes_no_summary(self, tmp_path):
        never_active = {"wait": 5, "warmup": 0, "active": 1}
        module = _profiled_module(
            tmp_path, pytorch_profiler_cfg={"activities": ["cpu"], "schedule": never_active, "filename": "prof"}
        )
        with pytest.warns(UserWarning, match="recorded no operators during the test loop"):
            _run_phase(module, "test", n_batches=2)
        assert not list(tmp_path.rglob("*.txt"))

    def test_steps_without_operators_warn_and_write_no_summary(self, tmp_path):
        from interpretune.runners.analysis import analysis_store_generator

        module = _profiled_module(tmp_path, pytorch_profiler_cfg={"activities": ["cpu"], "filename": "prof"})
        module.analysis_step = MagicMock(side_effect=lambda batch, batch_idx: iter([{"batch_idx": batch_idx}]))
        with pytest.warns(UserWarning, match="recorded no operators during the analysis loop"):
            list(analysis_store_generator(module=module, datamodule=_datamodule(2), max_epochs=1))
        assert not list(tmp_path.rglob("*.txt"))

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


################################################################################
# The configuration guide's examples, run as written
################################################################################

CONFIGURATION_GUIDE = Path(__file__).parents[2] / "docs" / "source" / "configuration.md"


def _guide_yaml(first_key):
    """The configuration guide's YAML example whose top-level key is ``first_key``, parsed."""
    blocks = re.findall(r"```yaml\n(.*?)```", CONFIGURATION_GUIDE.read_text(encoding="utf-8"), flags=re.S)
    matches = [yaml.safe_load(b) for b in blocks if b.startswith(f"{first_key}:") and "profiler" in b]
    assert len(matches) == 1, f"expected one {first_key!r} profiler example in {CONFIGURATION_GUIDE.name}"
    return matches[0]


def test_guide_core_example_writes_its_summary(tmp_path):
    """The core loop example parses into ``ProfilerCfg`` and, run on CPU, writes the summary it promises."""
    profiler_cfg = _guide_yaml("profiler_cfg")["profiler_cfg"]
    cfg = ProfilerCfg(**profiler_cfg)
    opts = dict(cfg.pytorch_profiler_cfg, activities=["cpu"], trace_dir=str(tmp_path / "traces"))
    opts["schedule"] = {"wait": 0, "warmup": 0, "active": 1}  # one recorded window from two batches
    module = _profiled_module(tmp_path, pytorch_profiler_cfg=opts)
    _run_phase(module, "test", n_batches=2)
    summary = tmp_path / "traces" / f"test-{profiler_cfg['pytorch_profiler_cfg']['filename']}.txt"
    assert summary.is_file()
    assert _table_rows(summary.read_text(encoding="utf-8"))


def _lightning_cust_session():
    import copy

    from interpretune.protocol import Adapter
    from interpretune.session import ITSession, ITSessionConfig
    from tests.module_registry import TEST_MODULE_REGISTRY

    target = SimpleNamespace(model_src_key="cust", model_cfg_key="rte", adapter_ctx=(Adapter.lightning,))
    itdm_cfg, it_cfg, dm_cls, m_cls = TEST_MODULE_REGISTRY.get(target)
    return ITSession(
        ITSessionConfig(
            adapter_ctx=(Adapter.lightning,),
            datamodule_cfg=copy.deepcopy(itdm_cfg),
            module_cfg=copy.deepcopy(it_cfg),
            datamodule_cls=dm_cls,
            module_cls=m_cls,
        )
    )


def _lightning_trainer_from(trainer_block):
    """A Trainer built from a ``trainer`` configuration block as ``LightningCLI`` builds one: its parser
    instantiates the block's classes (the profiler among them), then the Trainer is constructed from the result."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.cli import LightningArgumentParser

    parser = LightningArgumentParser()
    parser.add_lightning_class_args(Trainer, "trainer")
    cfg = parser.parse_object({"trainer": trainer_block})
    return Trainer(**parser.instantiate_classes(cfg).trainer)


_CPU_TRAINER = {
    "accelerator": "cpu",
    "devices": 1,
    "logger": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "limit_test_batches": 2,
}


@RunIf(lightning=True)
@pytest.mark.parametrize(
    "profiler_class",
    [
        "lightning.pytorch.profilers.PyTorchProfiler",
        "fts_examples.profiling.extended_profiler.ExtendedPyTorchProfiler",
    ],
    ids=["pytorch_profiler", "extended_pytorch_profiler"],
)
def test_lightning_configures_the_pytorch_profiler_the_standard_way(tmp_path, profiler_class):
    """The guide's Lightning example, parsed as ``LightningCLI`` parses ``trainer.profiler``, profiles a Lightning
    composition's test loop and writes Lightning's summary.

    Both Lightning's profiler and finetuning-scheduler's example extension of it (which adds ``schedule_cfg`` and
    ``max_name_column_width``) are configured this way.
    """
    if profiler_class.startswith("fts_examples"):
        pytest.importorskip("fts_examples.profiling.extended_profiler")
    profiler = _guide_yaml("trainer")["trainer"]["profiler"]
    profiler["class_path"] = profiler_class
    profiler["init_args"]["dirpath"] = str(tmp_path / "profiler_logs")
    if profiler_class.startswith("fts_examples"):
        profiler["init_args"].update(schedule_cfg={"wait": 0, "warmup": 0, "active": 1}, max_name_column_width=40)
    trainer = _lightning_trainer_from({**_CPU_TRAINER, "default_root_dir": str(tmp_path), "profiler": profiler})
    module_path, class_name = profiler_class.rsplit(".", 1)
    assert type(trainer.profiler) is getattr(importlib.import_module(module_path), class_name)
    it_session = _lightning_cust_session()
    trainer.test(it_session.module, datamodule=it_session.datamodule)
    summary = tmp_path / "profiler_logs" / f"test-{profiler['init_args']['filename']}.txt"
    assert summary.is_file(), sorted(p.name for p in (tmp_path / "profiler_logs").iterdir())
    assert "Self CPU" in summary.read_text(encoding="utf-8")
