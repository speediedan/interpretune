"""A shippable core module must survive a direct forward, which is what every Trainer loop calls (#717).

Only the backend forward path (the analysis ops) resolves batch input keys through aliases, so a config that declares
the TransformerLens bridge's `input` key for a module whose forward goes straight to an HF model passes every analysis
test and fails the first train, test or predict loop with HF's "You have to specify either input_ids or
inputs_embeds". Parity train/predict coverage uses only the cust test modules, whose forwards handle their own keys, so
this file covers the hub config a reader actually ships with: `rte_demo.gpt2.core`, built from the in-tree rte tree via
a local-publish revision (offline, no hub push).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import interpretune as it
from interpretune.hub.components import local_publish
from tests.rte_component import rte_entrypoint_src

RTE_DIR = Path(__file__).parents[2] / "src" / "it_examples" / "examples" / "rte"


def _core_session(adapters: tuple, dataset_dir: Path) -> it.ITSession:
    rev = local_publish(RTE_DIR, "speediedan/rte", entrypoint_src=rte_entrypoint_src())
    dm_cfg, m_cfg, dm_cls, m_cls = it.hub.load("speediedan/rte", "rte_demo.gpt2.core", revision=rev)
    # A per-test dataset directory: `prepare_data` rewrites the saved dataset, and on Windows a dataset another
    # session still has memory-mapped cannot be overwritten (OSError 22 on the arrow file).
    dm_cfg.dataset_path = dataset_dir
    return it.ITSession(
        it.ITSessionConfig(
            adapter_ctx=adapters, datamodule_cfg=dm_cfg, module_cfg=m_cfg, datamodule_cls=dm_cls, module_cls=m_cls
        )
    )


def test_core_module_direct_forward_accepts_its_own_batches(tmp_path):
    """The batches the datamodule yields must be ones the module's forward accepts."""
    session = _core_session((it.Adapter.core,), tmp_path / "rte_dataset")
    it.it_init(session.module, session.datamodule)
    batch = next(iter(session.datamodule.test_dataloader()))
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        out = session.module(**inputs)
    assert out.logits.shape[0] == batch["input_ids"].shape[0]


def test_lightning_trainer_predict_runs_on_the_core_module(tmp_path):
    """The loop #55's demo needs: Lightning's Trainer.predict over the shippable core config, on CPU."""
    pytest.importorskip("lightning")
    from lightning.pytorch import Trainer

    session = _core_session((it.Adapter.lightning,), tmp_path / "rte_dataset")
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_predict_batches=1,
    )
    out = trainer.predict(session.module, datamodule=session.datamodule)
    assert isinstance(out, list) and out, "Trainer.predict returned nothing"
