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
    assert getattr(module, "_it_analysis_rows", None) is None, "rows were kept after the store was written"


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
