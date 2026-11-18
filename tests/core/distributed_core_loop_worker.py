"""One process of the core loop's distributed parity runs (``tests/core/test_core_loop_distributed.py``).

Run under ``torch.distributed.run`` as one rank of a ``ddp``, ``fsdp2`` or ``tp`` run, or with ``--placement single``
as the one-process reference, which trains on the global batch (the per-rank batch times ``--world-size``) that
data-parallel ranks share out; a tensor-parallel run's reference passes ``--world-size 1``, since its ranks share one
batch. Each rank seeds its model differently and the reference is seeded as rank 0: the runner broadcasts rank 0's
weights before placement, and without that broadcast the ranks would not match it.

With ``--phased``, finetuning-scheduler's ``TuningScheduler`` drives two phases through ``FTSCoreCallback``: the
last block first, then everything before it. With ``--latent-site``, a small latent model replaces the output of the
module at that path (``tuning.latents``), so it is trained, and its splice is exercised, with the model. With
``--transcoder-site``, it reads that module's input and replaces its output instead (a transcoder), and a phased run
tunes it alone first, then everything else. ``--device-type cuda`` places each rank (and the reference) on its own GPU.

Writes ``<out_dir>/<placement>_rank<rank>.json`` with the per-step training losses, the epoch metrics the callbacks saw,
each rank's recorded device, the parameter types and FSDP units after placement, (phased) the final depth and the
parameters trainable at the end, and (latent) the latent model's parameters after training, or the placement's
refusal.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn

PER_RANK_BATCH = 2


class TinyLatent(nn.Module):
    """A small sparse autoencoder: ``encode`` and ``decode`` make it a latent model."""

    def __init__(self, width: int, n_latents: int = 8) -> None:
        super().__init__()
        self.enc, self.dec = nn.Linear(width, n_latents), nn.Linear(n_latents, width)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


class LossRecorder:
    """Records each optimizer step's loss and the epoch metrics the loop hands its callbacks."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.epoch_metrics: list[dict[str, float]] = []

    def on_train_step_end(self, ctx, outputs):
        self.losses.append(float(outputs.detach()))

    def on_train_epoch_end(self, ctx, metrics):
        self.epoch_metrics.append(dict(metrics))


class SaveCheckpoint:
    """Saves a resumable checkpoint at the end of epoch ``at_epoch`` (after FTSCoreCallback, when phased)."""

    def __init__(self, checkpoint_dir: str, at_epoch: int = 0) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.at_epoch = at_epoch

    def on_train_epoch_end(self, ctx, metrics):
        if ctx.epoch == self.at_epoch:
            ctx.save_checkpoint(self.checkpoint_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", required=True)
    parser.add_argument("--world-size", type=int, default=2, help="the ranks the reference's global batch stands for")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fsdp-plan", type=json.loads, default=None, help="the FSDP2 plan, as JSON")
    parser.add_argument("--tp-plan", type=json.loads, default=None, help="the tensor-parallel plan, as JSON")
    parser.add_argument("--phased", action="store_true", help="tune in two phases with FTSCoreCallback")
    parser.add_argument("--max-epochs", type=int, default=None, help="override the run's epoch count")
    parser.add_argument("--save-checkpoint-dir", default=None, help="save a checkpoint at the first epoch's end")
    parser.add_argument("--resume-from", default=None, help="resume from a checkpoint directory")
    parser.add_argument("--out-name", default=None, help="output record stem (default: <placement>)")
    parser.add_argument("--latent-site", default=None, help="splice a latent model at this module path")
    parser.add_argument("--transcoder-site", default=None, help="splice a transcoder over this module")
    parser.add_argument("--device-type", default="cpu", help="cuda places each rank on its own device (LOCAL_RANK)")
    args = parser.parse_args()

    from interpretune.config import SessionRunnerCfg
    from interpretune.runners import SessionRunner
    from interpretune.utils import MisconfigurationException
    from tests.configuration import config_modules
    from tests.parity_acceptance.test_it_l import CoreCfg

    rank = int(os.environ.get("RANK", "0"))
    batch_size = PER_RANK_BATCH * (args.world_size if args.placement == "single" else 1)
    test_cfg = CoreCfg(
        device_type=args.device_type,
        force_prepare_data=False,
        max_epochs=args.max_epochs or (2 if args.phased else 1),
        limit_train_batches=3,
        limit_val_batches=2,
        dm_override_cfg={"train_batch_size": batch_size, "eval_batch_size": batch_size},
        # no dropout: each rank draws from its own RNG stream, the reference from one
        model_cfg={"model_args": {"max_seq_len": 200, "n_layers": 2, "dropout_p": 0.0}},
    )
    out_dir = Path(args.out_dir)
    torch.manual_seed(1 + rank)  # ranks initialize differently; placement must broadcast rank 0's weights
    if args.device_type == "cuda":
        # the session is built here, before the runner would make this rank's device current
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    it_session = config_modules(test_cfg, f"dist_{args.placement}", {}, out_dir / f"{args.placement}_rank{rank}")
    recorder = LossRecorder()
    callbacks: list = [recorder]
    fts = None
    if args.phased:
        from interpretune.tuning.core_loop import FTSCoreCallback

        fts = FTSCoreCallback()  # its schedule names the placed module's parameters, set once they exist
        callbacks.insert(0, fts)
    if args.save_checkpoint_dir:
        callbacks.append(SaveCheckpoint(args.save_checkpoint_dir))
    latent = None
    if args.latent_site:
        from interpretune.tuning.latents import LatentSite, attach_latent_model

        width = it_session.module.get_submodule(args.latent_site.rsplit(".", 1)[0]).w1.out_features
        latent = attach_latent_model(
            it_session.module, "probe", TinyLatent(width), LatentSite(args.latent_site, op="replace")
        )
    elif args.transcoder_site:
        from interpretune.tuning.latents import LatentSite, attach_latent_model

        width = it_session.module.get_submodule(args.transcoder_site).w1.in_features
        site = LatentSite(args.transcoder_site, read_from="input", op="replace")
        latent = attach_latent_model(it_session.module, "transcoder", TinyLatent(width).to(args.device_type), site)
    runner_cfg = SessionRunnerCfg(
        it_session=it_session,
        max_epochs=test_cfg.max_epochs,
        limit_train_batches=test_cfg.limit_train_batches,
        limit_val_batches=test_cfg.limit_val_batches,
        callbacks=callbacks,
        placement=args.placement,
        fsdp_plan=args.fsdp_plan,
        tp_plan=args.tp_plan,
        resume_from=args.resume_from,
    )
    out_path = out_dir / f"{args.out_name or args.placement}_rank{rank}.json"
    try:
        runner = SessionRunner(run_cfg=runner_cfg)
    except MisconfigurationException as err:
        if latent is None:
            raise
        out_path.write_text(json.dumps({"refused": str(err)}))
        return
    from torch.distributed.fsdp import FSDPModule

    module = it_session.module
    # after placement, so every rank holds rank 0's broadcast weights
    latent_init = {n: p.detach().flatten().tolist() for n, p in latent.named_parameters()} if latent else None
    param_types = sorted({type(p).__name__ for p in module.parameters()})
    fsdp_units = [name for name, sub in module.named_modules() if isinstance(sub, FSDPModule)]
    split_params = sorted(n for n, p in module.named_parameters() if type(p).__name__ == "DTensor")
    if fts is not None:
        trainable = [n for n, p in module.named_parameters() if p.requires_grad]
        if args.transcoder_site:
            first = [n for n in trainable if n.startswith("latent_models.")]
        else:
            last_block = max({n.rsplit(".", 2)[0] for n in trainable if ".layers." in n})
            first = [n for n in trainable if n.startswith(last_block + ".")]
        fts.schedule = {
            0: {"params": first, "max_transition_epoch": 1},
            1: {"params": [n for n in trainable if n not in first]},
        }
    runner.train()
    record = {
        "losses": recorder.losses,
        "epoch_metrics": recorder.epoch_metrics,
        "devices": list(runner.dist_run.devices) if runner.dist_run else [],
        "param_types": param_types,
        "fsdp_units": fsdp_units,
        "split_params": split_params if args.placement == "tp" else [],
    }
    if args.device_type == "cuda":
        record["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if latent is not None:  # whole on every rank: a latent model is never split
        record["latent_params"] = {n: p.detach().flatten().tolist() for n, p in latent.named_parameters()}
        record["latent_init"] = latent_init
    if fts is not None:
        assert fts.tuning_scheduler is not None
        record["depth"] = fts.tuning_scheduler.curr_depth
        record["trainable"] = sorted(n for n, p in module.named_parameters() if p.requires_grad)
    out_path.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
