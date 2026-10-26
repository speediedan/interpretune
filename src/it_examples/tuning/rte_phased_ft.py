# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Phased fine-tuning on RTE with finetuning-scheduler, on the core loop or under Lightning (#765).

One session configuration (the ``speediedan/rte`` component's ``rte_ft`` variant) and one schedule
(``ft_schedules/rte_ft.<model>.yaml``) run on either host:

- ``core``: interpretune's core loop, with ``FTSCoreCallback`` driving finetuning-scheduler's ``TuningScheduler``, a
  ``PlateauTransition`` on ``val_loss`` and a ``BestCheckpoint``;
- ``lightning``: a Lightning ``Trainer`` with the ``FinetuningScheduler`` callback, ``FTSEarlyStopping`` and
  ``FTSCheckpoint``, configured with the same values.

Each run writes one JSON record: every epoch's schedule depth, thawed parameters and validation metrics, the final
depth, and the best checkpoint's depth and score. Two records of the same configuration should agree exactly on the
thawed sets, the transition epochs and the best depth. ``gemma3`` is interpretune#697's schedule for gemma-3-1b-it
(one 24 GB GPU, fp32); ``gpt2`` is small enough for a CPU::

    python -m it_examples.tuning.rte_phased_ft --host core --model gpt2 --out /tmp/rte_ft_core.json
    python -m it_examples.tuning.rte_phased_ft --host lightning --model gpt2 --out /tmp/rte_ft_lightning.json

The session configuration is materialized from the configs shipped with ``it_examples`` (a local publish of the
``rte`` component), and its entrypoint (``rte_boolq.py``) comes from the cached ``speediedan/rte`` snapshot: fetch it
once with ``it.hub.pull('speediedan/rte')`` and opt in to running it (``IT_TRUST_REMOTE_CODE=1``).
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

import interpretune as it

PKG_DIR = Path(__file__).parent
RTE_COMPONENT_DIR = PKG_DIR.parent / "examples" / "rte"
SCHEDULE_DIR = PKG_DIR / "ft_schedules"
MONITOR = "val_loss"
# per model: the `max_depth` the run stops at (-1 for every phase) and the default device
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "gpt2": {"max_depth": -1, "device": "cpu"},
    "gemma3": {"max_depth": 1, "device": "cuda"},
}


def load_schedule(model: str) -> dict[int, dict[str, Any]]:
    """The phase schedule for ``model``."""
    with open(SCHEDULE_DIR / f"rte_ft.{model}.yaml") as f:
        return yaml.safe_load(f)


def build_session(host: str, model: str, *, device: str, dataset_dir: Path | None = None) -> Any:
    """The ``rte_ft.<model>.core`` session, composed for ``host``."""
    from interpretune.hub.components import local_publish, resolve_component_manifest

    _, snapshot, _ = resolve_component_manifest("speediedan/rte")
    revision = local_publish(RTE_COMPONENT_DIR, "speediedan/rte", entrypoint_src=snapshot / "rte_boolq.py")
    cfg = it.hub.load("speediedan/rte", f"rte_ft.{model}.core", revision=revision)
    dm_cfg, m_cfg = copy.deepcopy(cfg.datamodule_cfg), copy.deepcopy(cfg.module_cfg)
    if dataset_dir is not None:
        dm_cfg.dataset_path = dataset_dir
    m_cfg.hf_from_pretrained_cfg.pretrained_kwargs["device_map"] = device
    adapter = it.Adapter.core if host == "core" else it.Adapter.lightning
    return it.ITSession(
        it.ITSessionConfig(
            adapter_ctx=(adapter,),
            datamodule_cfg=dm_cfg,
            module_cfg=m_cfg,
            datamodule_cls=cfg.datamodule_cls,
            module_cls=cfg.module_cls,
        )
    )


class EpochRecord:
    """Each epoch's depth, thawed parameters and validation metrics, from either host."""

    def __init__(self) -> None:
        self.epochs: list[dict[str, Any]] = []

    def start(self, epoch: int, depth: int, module: torch.nn.Module) -> None:
        thawed = sorted(n for n, p in module.named_parameters() if p.requires_grad)
        self.epochs.append({"epoch": epoch, "depth": depth, "thawed": thawed})

    def end(self, metrics: dict[str, Any]) -> None:
        self.epochs[-1].update({k: float(metrics[k]) for k in (MONITOR, "accuracy") if k in metrics})


def run_core(
    session: Any,
    schedule: dict,
    *,
    max_epochs: int,
    max_depth: int,
    limit_train_batches: int,
    limit_val_batches: int,
    patience: int,
    min_delta: float,
    ckpt_dir: str,
) -> dict[str, Any]:
    from finetuning_scheduler.core.checkpoint import BestCheckpoint
    from finetuning_scheduler.core.transitions import PlateauTransition
    from interpretune.config import SessionRunnerCfg
    from interpretune.tuning.core_loop import FTSCoreCallback

    record = EpochRecord()
    checkpoint = BestCheckpoint(dirpath=ckpt_dir, monitor=MONITOR)
    fts = FTSCoreCallback(
        schedule,
        tuning_scheduler_kwargs={
            "transition": PlateauTransition(monitor=MONITOR, patience=patience, min_delta=min_delta),
            "checkpoint": checkpoint,
            "max_depth": max_depth,
        },
    )

    class Recorder:
        def on_train_epoch_start(self, ctx: Any) -> None:
            record.start(ctx.epoch, fts.tuning_scheduler.curr_depth, ctx.module)

        def on_train_epoch_end(self, ctx: Any, metrics: dict[str, Any]) -> None:
            record.end(metrics)

    # `FTSCoreCallback` first, so an epoch is recorded after the transition it starts with
    run_cfg = SessionRunnerCfg(
        it_session=session,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        callbacks=[fts, Recorder()],
    )
    it.SessionRunner(run_cfg=run_cfg).train()
    ts = fts.tuning_scheduler
    return {
        "final_depth": ts.curr_depth,
        "best_depth": checkpoint.best_depth,
        "best_score": float(checkpoint.best_score),
        "epochs": record.epochs,
    }


def run_lightning(
    session: Any,
    schedule: dict,
    *,
    max_epochs: int,
    max_depth: int,
    limit_train_batches: int,
    limit_val_batches: int,
    patience: int,
    min_delta: float,
    ckpt_dir: str,
    device: str,
) -> dict[str, Any]:
    from lightning.pytorch import Callback, Trainer
    from finetuning_scheduler import FinetuningScheduler, FTSCheckpoint, FTSEarlyStopping

    record = EpochRecord()
    fts = FinetuningScheduler(ft_schedule=schedule, max_depth=max_depth)
    checkpoint = FTSCheckpoint(monitor=MONITOR, dirpath=ckpt_dir, save_top_k=1)

    class Recorder(Callback):
        def on_train_epoch_start(self, trainer: Trainer, pl_module: Any) -> None:
            record.start(trainer.current_epoch, fts.curr_depth, pl_module)

        def on_validation_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
            if record.epochs and not trainer.sanity_checking:
                record.end(dict(trainer.callback_metrics))

    trainer = Trainer(
        default_root_dir=ckpt_dir,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches if limit_train_batches > 0 else 1.0,
        limit_val_batches=limit_val_batches if limit_val_batches > 0 else 1.0,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            fts,
            FTSEarlyStopping(monitor=MONITOR, min_delta=min_delta, patience=patience),
            checkpoint,
            Recorder(),
        ],
    )
    trainer.fit(**session)
    return {
        "final_depth": fts.curr_depth,
        "best_depth": checkpoint.best_ckpt_depth,
        "best_score": float(checkpoint.best_model_score),
        "epochs": record.epochs,
    }


def run(
    host: str,
    model: str = "gpt2",
    *,
    max_epochs: int = 10,
    max_depth: int | None = None,
    limit_train_batches: int = -1,
    limit_val_batches: int = -1,
    patience: int = 1,
    min_delta: float = 0.001,
    seed: int = 0,
    device: str | None = None,
    dataset_dir: Path | None = None,
    ckpt_dir: str | None = None,
) -> dict[str, Any]:
    """Run the schedule for ``model`` on ``host`` (``"core"`` or ``"lightning"``) and return its record."""
    if host not in ("core", "lightning"):
        raise ValueError(f"host must be 'core' or 'lightning', got {host!r}")
    defaults = MODEL_DEFAULTS[model]
    device = device or defaults["device"]
    max_depth = defaults["max_depth"] if max_depth is None else max_depth
    schedule = load_schedule(model)
    ckpt_dir = ckpt_dir or tempfile.mkdtemp(prefix=f"rte_ft_{host}_")
    torch.manual_seed(seed)
    session = build_session(host, model, device=device, dataset_dir=dataset_dir)
    kwargs = dict(
        max_epochs=max_epochs,
        max_depth=max_depth,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        patience=patience,
        min_delta=min_delta,
        ckpt_dir=ckpt_dir,
    )
    result = (
        run_core(session, schedule, **kwargs)
        if host == "core"
        else run_lightning(session, schedule, **kwargs, device=device)
    )
    transitions = [e["epoch"] for prev, e in zip(result["epochs"], result["epochs"][1:]) if e["depth"] != prev["depth"]]
    return {
        "host": host,
        "model": model,
        "seed": seed,
        "max_depth": max_depth,
        "transition_epochs": transitions,
        **result,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", choices=("core", "lightning"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_DEFAULTS), default="gpt2")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=-1)
    parser.add_argument("--limit-val-batches", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        args.host,
        args.model,
        max_epochs=args.max_epochs,
        max_depth=args.max_depth,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
        dataset_dir=args.dataset_dir,
        ckpt_dir=args.ckpt_dir,
    )
    args.out.write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k != "epochs"}
    summary["epochs"] = [{k: (len(v) if k == "thawed" else v) for k, v in e.items()} for e in result["epochs"]]
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
