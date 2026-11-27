"""One process of the check of ``AnalysisStoreCallback`` under a multi-process Lightning ``Trainer``.

Run alone (``--mode single``) it writes the one-process reference; run by ``torch.distributed.run`` over two
processes (``--mode ddp``) each rank runs the same predict under ``strategy="ddp"`` on CPU. The predict covers the
tiny cust model's ten RTE examples in batches of two; the sampler deals each rank every other example, five each, so
each rank runs three batches, the last holding one example.
The generated ``logit_diffs_base`` step records each example's logit difference with ``log_analysis``, and a
``CSVLogger`` receives the epoch's averages. ``--mode unsharded`` turns Lightning's distributed sampler off, so each
rank would analyze every example; the callback must refuse it. ``--mode shard_fails`` makes rank 1's shard fail to
close; both ranks must raise, naming rank 1, rather than one of them waiting on the other.

Writes ``<out_dir>/<mode>_rank<rank>.json``: the store's rows, its columns, the analysis metrics, the logged metrics,
the predict batch size and the row files left in the store's cache directory (or the refusal's message).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

import interpretune as it
import interpretune.analysis  # (registers the it.<op> wrappers)
from interpretune import AnalysisCfg
from interpretune.adapters.lightning import AnalysisStoreCallback
from interpretune.utils import MisconfigurationException
from tests.core.test_lightning_analysis_step import _lightning_cust_session


def _plain(value):
    return value.tolist() if isinstance(value, torch.Tensor) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "ddp", "unsharded", "shard_fails"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    it_session = _lightning_cust_session()
    module = it_session.module
    module.analysis_cfg = AnalysisCfg(target_op=it.labels_to_ids, ignore_manual=True)
    run_step = module._run_configured_step

    def recording_step(analysis_cfg, batch, batch_idx, dataloader_idx):
        # each example's token sum: distinct per example, and exact in float, so the pooled average is exact too
        for value in batch["tokens"].sum(dim=-1).tolist():
            module.log_analysis("token_sum", float(value))
        return run_step(analysis_cfg, batch, batch_idx, dataloader_idx)

    module._run_configured_step = recording_step
    if args.mode == "shard_fails" and int(os.environ["RANK"]) == 1:
        from interpretune.runners.analysis import AnalysisRowWriter

        def failing_finalize(self):
            raise OSError("disk full")

        AnalysisRowWriter.finalize = failing_finalize
    distributed = args.mode != "single"
    logger = CSVLogger(args.out_dir / f"{args.mode}_logs", name="", version=0)
    trainer = Trainer(
        default_root_dir=args.out_dir / args.mode,
        accelerator="cpu",
        devices=2 if distributed else 1,
        strategy="ddp" if distributed else "auto",
        use_distributed_sampler=args.mode != "unsharded",
        logger=logger,
        callbacks=[AnalysisStoreCallback(writer_batch_size=1)],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    record: dict = {"rank": trainer.global_rank}
    try:
        trainer.predict(module, datamodule=it_session.datamodule, return_predictions=False)
    except (MisconfigurationException, RuntimeError) as err:
        record["refused"] = f"{type(err).__name__}: {err}"
    store = module.analysis_cfg.output_store  # set up by the predict run
    if "refused" in record:
        record["store_saved"] = store is not None and (Path(store.save_dir) / "dataset_info.json").exists()
    else:
        dataset = store.dataset.with_format(None)
        record.update(
            columns=sorted(dataset.column_names),
            rows=[{name: _plain(value) for name, value in row.items()} for row in dataset],
            analysis_metrics=module.analysis_metrics,
            batch_size=trainer.predict_dataloaders.batch_sampler.batch_size,
        )
        if trainer.global_rank == 0:
            with open(Path(logger.log_dir) / "metrics.csv", encoding="utf-8", newline="") as f:
                record["logged"] = list(csv.DictReader(f))
    rows_dir = Path(store.cache_dir) if store is not None else None
    record["row_files"] = sorted(p.name for p in rows_dir.glob("analysis-rows-*.arrow")) if rows_dir else []
    out = args.out_dir / f"{args.mode}_rank{trainer.global_rank}.json"
    out.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
