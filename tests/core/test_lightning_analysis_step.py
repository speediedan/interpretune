"""Lightning analysis_step support (interpretune#91).

A Trainer loop never invokes the AnalysisRunner, so without these seams a Lightning-composed
module reaches its first batch with no analysis setup and no way to run its analysis_step.
Covers: setup hooks (predict/test start), predict_step routing when analysis is configured,
and passthrough when it is not. CPU Trainer throughout.
"""

from __future__ import annotations

from types import SimpleNamespace

from lightning.pytorch import Trainer

import interpretune as it
import interpretune.analysis  # (registers the it.<op> wrappers)
from interpretune.protocol import Adapter
from interpretune.session import ITSessionConfig, ITSession
from tests.module_registry import TEST_MODULE_REGISTRY


def _lightning_cust_session():
    """A Lightning-composed session on the tiny cust model (CPU)."""
    import copy

    target = SimpleNamespace(model_src_key="cust", model_cfg_key="rte", adapter_ctx=(Adapter.lightning,))
    itdm_cfg, it_cfg, dm_cls, m_cls = TEST_MODULE_REGISTRY.get(target)
    # The registry hydrates shared config objects: deepcopy so one test attaching an analysis_cfg
    # cannot leak it into the next test's session through the same it_cfg.
    itdm_cfg, it_cfg = copy.deepcopy(itdm_cfg), copy.deepcopy(it_cfg)
    session_cfg = ITSessionConfig(
        adapter_ctx=(Adapter.lightning,),
        datamodule_cfg=itdm_cfg,
        module_cfg=it_cfg,
        datamodule_cls=dm_cls,
        module_cls=m_cls,
    )
    return ITSession(session_cfg)


def _trainer(tmp_path, limit=1):
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_predict_batches=limit,
        limit_test_batches=limit,
    )


def _analysis_cfg(ignore_manual: bool = True):
    from interpretune import AnalysisCfg

    return AnalysisCfg(target_op=it.labels_to_ids, ignore_manual=ignore_manual)


def test_predict_step_routes_to_manual_analysis_step(tmp_path):
    """With analysis configured to keep a manual step, Trainer.predict executes the module's analysis_step."""
    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = _analysis_cfg(ignore_manual=False)

    calls = []

    def manual_step(batch, batch_idx, dataloader_idx=0):
        calls.append((batch_idx, dataloader_idx))
        return {"ran": True}

    # plain assignment: instance-dict functions do not bind, so this is called as-is
    module.analysis_step = manual_step

    out = _trainer(tmp_path).predict(module, datamodule=it_session.datamodule)
    assert calls, "analysis_step was never invoked during predict"
    assert calls[0][0] == 0
    assert isinstance(out, list) and out and out[0] == {"ran": True}


def test_ignore_manual_generates_the_step_from_the_op(tmp_path):
    """``ignore_manual=True`` means "ignore the existing analysis_step and generate one from the op".

    The Trainer path must honor it exactly as the AnalysisRunner does: a manual step present on the module is NOT
    what runs, and the generated step's output (the op's result for each batch) is what predict returns.
    """
    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = _analysis_cfg(ignore_manual=True)

    calls = []

    def manual_step(batch, batch_idx, dataloader_idx=0):
        calls.append(batch_idx)
        return {"ran": "manual"}

    module.analysis_step = manual_step

    out = _trainer(tmp_path).predict(module, datamodule=it_session.datamodule)
    assert not calls, "ignore_manual=True, yet the manual analysis_step ran instead of the generated one"
    assert isinstance(out, list) and out, "the generated analysis step returned nothing"
    assert out[0] != {"ran": "manual"}


def test_generated_step_runs_through_trainer_predict(tmp_path):
    """The main case the issue is about: no manual step at all, so the op generates one and predict runs it."""
    it_session = _lightning_cust_session()
    module = it_session.module
    if "analysis_step" in vars(module):
        del module.analysis_step
    module.analysis_cfg = _analysis_cfg(ignore_manual=True)

    out = _trainer(tmp_path).predict(module, datamodule=it_session.datamodule)
    assert isinstance(out, list) and out, "the generated analysis step returned nothing"
    assert module.analysis_cfg.applied_to(module), "the analysis config was never applied to the module"


def test_predict_step_falls_through_without_analysis_cfg(tmp_path):
    """Without analysis configured, predict takes Lightning's default path."""
    it_session = _lightning_cust_session()
    out = _trainer(tmp_path).predict(it_session.module, datamodule=it_session.datamodule)
    assert isinstance(out, list) and out, "default predict_step returned nothing"


def test_on_predict_start_wraps_predict_for_analysis(tmp_path):
    """The predict-start hook routes predict through the configured step without clobbering it."""
    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = _analysis_cfg()
    manual = module.analysis_step
    assert not getattr(module, "_it_predict_wrapped", False)
    _trainer(tmp_path).predict(module, datamodule=it_session.datamodule)
    assert getattr(module, "_it_predict_wrapped", False)
    # the task's own analysis_step is preserved (the generated one is added beside it); only predict dispatch is
    # wrapped (bound methods never preserve identity, so compare the underlying functions)
    assert module.analysis_step.__func__ is manual.__func__


def test_setup_without_a_log_directory_is_refused_by_name():
    """Outside a Trainer or runner there is nowhere to write analysis outputs: say so, rather than fail on a path."""
    import pytest

    from interpretune.utils import MisconfigurationException

    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = _analysis_cfg()
    with pytest.raises(MisconfigurationException, match="no log directory to write analysis outputs"):
        module.on_predict_start()


def test_analysis_store_callback_writes_the_store(tmp_path):
    """With ``AnalysisStoreCallback``, Trainer.predict writes the rows its analysis steps yield as the
    AnalysisStore.

    The store has one row per row predict returned, the core loop's features for the op, and is saved to the
    configured ``output_store.save_dir``, as ``AnalysisRunner.run_analysis`` leaves it.
    """
    from datasets import load_from_disk

    from interpretune.adapters.lightning import AnalysisStoreCallback
    from interpretune.runners.analysis import dataset_features_and_format

    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = _analysis_cfg(ignore_manual=True)
    trainer = _trainer(tmp_path, limit=2)
    trainer.callbacks.append(AnalysisStoreCallback())

    out = trainer.predict(module, datamodule=it_session.datamodule)
    rows = [row for batch_rows in out for row in batch_rows]
    assert rows, "predict returned no analysis rows"

    store = module.analysis_cfg.output_store
    assert store.dataset is not None, "the callback did not set the store's dataset"
    assert len(store.dataset) == len(rows)
    features, _, _ = dataset_features_and_format(module, {})
    assert set(store.dataset.column_names) == set(features)
    assert len(load_from_disk(str(store.save_dir))) == len(rows)
    assert getattr(module, "_it_analysis_row_writer", None) is None, "the writer was kept after the store was written"


def _store_session(tmp_path, limit=2, **callback_kwargs):
    """A cust session with a generated analysis step, and a Trainer carrying ``AnalysisStoreCallback``."""
    from interpretune.adapters.lightning import AnalysisStoreCallback

    it_session = _lightning_cust_session()
    it_session.module.analysis_cfg = _analysis_cfg(ignore_manual=True)
    trainer = _trainer(tmp_path, limit=limit)
    trainer.callbacks.append(AnalysisStoreCallback(**callback_kwargs))
    return it_session, trainer


def test_analysis_store_callback_streams_rows_to_disk(tmp_path, monkeypatch):
    """Rows reach the Arrow file during the run, at the configured batch size, not when the run ends."""
    from datasets.arrow_writer import ArrowWriter

    from interpretune.adapters.lightning import AnalysisStoreCallback
    from interpretune.runners.analysis import AnalysisRowWriter

    flushed, seen = [], []
    write_batch, finalize = ArrowWriter.write_examples_on_file, AnalysisRowWriter.finalize

    def counting_write(self):
        flushed.append(len(self.current_examples))
        return write_batch(self)

    def counting_finalize(self):
        seen.append(sum(flushed))
        return finalize(self)

    monkeypatch.setattr(AnalysisRowWriter, "finalize", counting_finalize)
    it_session, trainer = _store_session(tmp_path, limit=3, writer_batch_size=1)
    assert any(isinstance(c, AnalysisStoreCallback) for c in trainer.callbacks)
    monkeypatch.setattr(ArrowWriter, "write_examples_on_file", counting_write)
    trainer.predict(it_session.module, datamodule=it_session.datamodule, return_predictions=False)
    n_rows = len(it_session.module.analysis_cfg.output_store.dataset)
    assert n_rows == 3
    assert seen == [n_rows], "rows were still buffered when the run ended"


def test_analysis_store_callback_matches_the_rows_predict_returns(tmp_path):
    """The streamed store holds, value for value, what the steps yielded."""
    import torch

    it_session, trainer = _store_session(tmp_path, writer_batch_size=1)
    out = trainer.predict(it_session.module, datamodule=it_session.datamodule)
    rows = [row for batch_rows in out for row in batch_rows]
    dataset = it_session.module.analysis_cfg.output_store.dataset
    assert len(dataset) == len(rows) > 1
    compared = 0
    for row, stored in zip(rows, dataset):
        for name in dataset.column_names:
            value = row[name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(torch.as_tensor(stored[name]).reshape(value.shape), value), name
                compared += 1
    assert compared, "no tensor column was compared"


def test_analysis_store_callback_runs_the_analysis_lifecycle(tmp_path, monkeypatch):
    """The hooks and the op-state boundaries run in the core loop's order."""
    it_session, trainer = _store_session(tmp_path)
    module = it_session.module
    events = []
    for name in ("on_analysis_start", "on_analysis_epoch_end", "on_analysis_end"):
        original = getattr(module, name)
        monkeypatch.setattr(module, name, lambda n=name, o=original: (events.append(n), o())[1], raising=False)
    cfg_cls = type(module.analysis_cfg)
    reset, finalize = cfg_cls.reset_op_state, cfg_cls.finalize_op_state
    monkeypatch.setattr(
        cfg_cls,
        "reset_op_state",
        lambda self, *, epoch_boundary=False: (
            events.append(f"reset(epoch_boundary={epoch_boundary})"),
            reset(self, epoch_boundary=epoch_boundary),
        )[1],
    )
    monkeypatch.setattr(cfg_cls, "finalize_op_state", lambda self: (events.append("finalize"), finalize(self))[1])
    trainer.predict(module, datamodule=it_session.datamodule)
    assert events == [
        "reset(epoch_boundary=False)",
        "on_analysis_start",
        "on_analysis_epoch_end",
        "reset(epoch_boundary=True)",
        "on_analysis_end",
        "finalize",
    ]


def test_analysis_store_callback_leaves_no_partial_store_when_a_step_raises(tmp_path):
    """A run that raises removes its partial rows, restores the grad state and keeps no writer."""
    import pytest
    import torch

    it_session, trainer = _store_session(tmp_path, limit=2)
    module = it_session.module
    module.analysis_cfg = _analysis_cfg(ignore_manual=False)

    seen = []

    def failing_step(batch, batch_idx, dataloader_idx=0):
        seen.append(module._it_analysis_row_writer.path)
        assert seen[0].exists(), "the rows file was not open during the run"
        raise RuntimeError("step failed")

    module.analysis_step = failing_step
    grad_before = torch.is_grad_enabled()
    with pytest.raises(RuntimeError, match="step failed"):
        trainer.predict(module, datamodule=it_session.datamodule)
    assert seen and not seen[0].exists(), "the partial rows file was kept"
    assert module._it_analysis_row_writer is None
    assert module.analysis_cfg.output_store.dataset is None
    assert torch.is_grad_enabled() == grad_before


def test_analysis_store_callback_refuses_a_run_without_rows(tmp_path):
    import pytest

    from interpretune.utils import MisconfigurationException

    it_session, trainer = _store_session(tmp_path)
    module = it_session.module
    module.analysis_cfg = _analysis_cfg(ignore_manual=False)
    paths = []

    def empty_step(batch, batch_idx, dataloader_idx=0):
        paths.append(module._it_analysis_row_writer.path)
        return []

    module.analysis_step = empty_step
    with pytest.raises(MisconfigurationException, match="produced no analysis rows"):
        trainer.predict(module, datamodule=it_session.datamodule)
    assert paths and not paths[0].exists()
    assert module._it_analysis_row_writer is None


def test_analysis_store_callback_refuses_a_non_positive_writer_batch_size():
    import pytest

    from interpretune.adapters.lightning import AnalysisStoreCallback
    from interpretune.utils import MisconfigurationException

    with pytest.raises(MisconfigurationException, match="writer_batch_size must be a positive number"):
        AnalysisStoreCallback(writer_batch_size=0)


def test_analysis_store_callback_without_an_analysis_op_is_refused(tmp_path):
    import pytest

    from interpretune.adapters.lightning import AnalysisStoreCallback
    from interpretune.utils import MisconfigurationException

    it_session = _lightning_cust_session()
    trainer = _trainer(tmp_path)
    trainer.callbacks.append(AnalysisStoreCallback())
    with pytest.raises(MisconfigurationException, match="no analysis config with an op"):
        trainer.predict(it_session.module, datamodule=it_session.datamodule)


def test_predict_without_the_callback_writes_no_store(tmp_path):
    """Without the callback, predict returns the step outputs and leaves the store alone, as before."""
    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = _analysis_cfg(ignore_manual=True)
    _trainer(tmp_path).predict(module, datamodule=it_session.datamodule)
    assert module.analysis_cfg.output_store.dataset is None
