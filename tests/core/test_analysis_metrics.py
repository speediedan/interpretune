"""Analysis metrics: `log_analysis` values averaged per analysis epoch and reported by the loop running the analysis.

Under a Lightning ``Trainer`` analysis runs in ``predict``, whose hooks refuse ``self.log``, so analysis metrics have
their own accumulator: the core runner prints each epoch's averages, and ``AnalysisStoreCallback`` sends them to the
Trainer's loggers.
"""

from __future__ import annotations

import csv
from types import MethodType
from unittest.mock import MagicMock

import pytest
import torch

from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.base.components.mixins import AnalysisStepMixin
from interpretune.runners.analysis import analysis_store_generator


class _Recorder(AnalysisStepMixin):
    pass


class TestLogAnalysis:
    def test_values_are_averaged_per_epoch_and_kept_in_order(self):
        module = _Recorder()
        assert module.analysis_metrics == []
        module.log_analysis("score", 1.0)
        module.log_analysis("score", torch.tensor(3.0))
        module.log_analysis("rows", 4)
        assert module.flush_analysis_metrics() == {"score": 2.0, "rows": 4.0}
        module.log_analysis("score", 5.0)
        assert module.flush_analysis_metrics() == {"score": 5.0}
        assert module.analysis_metrics == [{"score": 2.0, "rows": 4.0}, {"score": 5.0}]

    def test_an_epoch_that_records_nothing_is_not_kept(self):
        module = _Recorder()
        assert module.flush_analysis_metrics() == {}
        assert module.analysis_metrics == []

    def test_reset_starts_a_run_afresh(self):
        module = _Recorder()
        module.log_analysis("score", 1.0)
        module.flush_analysis_metrics()
        module.log_analysis("score", 2.0)
        module.reset_analysis_metrics()
        assert module.analysis_metrics == []
        assert module.flush_analysis_metrics() == {}

    def test_a_tensor_with_more_than_one_value_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"'score' got a tensor of shape \(2,\)"):
            _Recorder().log_analysis("score", torch.tensor([1.0, 2.0]))


def _module_logging(values):
    """A mock analysis module whose step records ``values`` in turn, one per call, with the real accumulator."""
    module = MagicMock()
    for name in ("log_analysis", "flush_analysis_metrics", "reset_analysis_metrics"):
        setattr(module, name, MethodType(getattr(AnalysisStepMixin, name), module))
    calls = iter(values)

    def step(*args, **kwargs):
        module.log_analysis("score", next(calls))
        return [AnalysisBatch(logit_diffs=torch.tensor([0.5]))]

    module.analysis_step = MagicMock(side_effect=step)
    return module


def test_the_core_loop_prints_and_keeps_each_epoch_s_averages(capsys):
    datamodule = MagicMock()
    loader = MagicMock()
    loader.__iter__.side_effect = lambda: iter([{"input": torch.tensor([[1]])}, {"input": torch.tensor([[2]])}])
    datamodule.test_dataloader.return_value = loader
    module = _module_logging([1.0, 3.0, 10.0, 20.0])
    module.log_analysis("stale", 99.0)  # recorded before the run starts: the run's reset drops it
    rows = list(analysis_store_generator(module=module, datamodule=datamodule, step_fn="analysis_step", max_epochs=2))
    assert len(rows) == 4
    assert AnalysisStepMixin.analysis_metrics.fget(module) == [{"score": 2.0}, {"score": 15.0}]
    out = capsys.readouterr().out
    assert "Analysis epoch 0 end: {'score': 2.0000}" in out
    assert "Analysis epoch 1 end: {'score': 15.0000}" in out


def test_the_store_callback_sends_the_averages_to_the_trainer_s_loggers(tmp_path):
    """``self.log`` is refused in predict hooks; the callback reaches the loggers directly."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger

    from interpretune.adapters.lightning import AnalysisStoreCallback
    from tests.core.test_lightning_analysis_step import _analysis_cfg, _lightning_cust_session

    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = _analysis_cfg(ignore_manual=True)
    run_step = module._run_configured_step

    def recording_step(analysis_cfg, batch, batch_idx, dataloader_idx):
        # what an analysis step does: record a metric while it runs, inside the Trainer's predict loop
        module.log_analysis("batch_rows", float(len(batch["labels"])))
        module.log_analysis("batch_index", float(batch_idx))
        return run_step(analysis_cfg, batch, batch_idx, dataloader_idx)

    module._run_configured_step = recording_step
    logger = CSVLogger(tmp_path / "logs")
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=[AnalysisStoreCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_predict_batches=2,
    )
    trainer.predict(module, datamodule=it_session.datamodule, return_predictions=False)
    (epoch,) = module.analysis_metrics
    assert epoch["batch_index"] == 0.5 and epoch["batch_rows"] > 0
    with open(f"{logger.log_dir}/metrics.csv") as f:
        logged = list(csv.DictReader(f))
    assert [(float(r["analysis/batch_index"]), float(r["analysis/batch_rows"]), r["step"]) for r in logged] == [
        (0.5, epoch["batch_rows"], "0")
    ]
