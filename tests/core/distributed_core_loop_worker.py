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
tunes it alone first, then everything else. With ``--intervention-site``, a trainable intervention adds a learned vector
to the output of the block at that path (``tuning.interventions``), so its operand is trained with the model.
With ``--addressed``, the test model has a venngram component map, a latent model is spliced over the first block's
feed-forward, and a three-phase schedule written only in venngram addresses tunes the last block's MLP by layer and
component, then the latent model by role, then the first block's attention. ``--addressed promote`` attaches the latent
model frozen (its tensors are ``artifact``) and promotes it in the second phase.
``--device-type cuda`` places each rank (and the reference) on its own GPU.
``--memprof-fsdp`` profiles every training iteration with the memory profiler's ``FSDP2MemTracker``.

Writes ``<out_dir>/<placement>_rank<rank>.json`` with the per-step training losses, the epoch metrics the callbacks saw,
each rank's recorded device, the parameter types and FSDP units after placement, (phased) the final depth and the
parameters trainable at the end, (latent) the latent model's parameters after training, or the placement's refusal, and
(memprof) the path of the rank's FSDP2 memory report.
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


class TrainableRecorder:
    """Records the parameters trainable in each epoch, and the latent model's values at each epoch's end."""

    def __init__(self) -> None:
        self.module: nn.Module | None = None
        self.trainable: list[list[str]] = []

    def on_train_epoch_end(self, ctx, metrics):
        assert self.module is not None
        self.trainable.append(sorted(n for n, p in self.module.named_parameters() if p.requires_grad))


# the test `Transformer`'s component map, so its tunable view has layers and components to address
COMPONENT_MAP = {
    "schema_version": 4,
    "architecture": "Transformer",
    "components": {
        "embed": {"module": "tok_embeddings", "kind": "embed"},
        "pos_embed": {"module": "pos_embeddings", "kind": "embed"},
        "blocks.{i}": {"module": "layers.{i}", "kind": "block"},
        "blocks.{i}.attn": {"module": "layers.{i}.attention", "kind": "attn"},
        "blocks.{i}.attn.q": {"module": "layers.{i}.attention.wq", "kind": "linear"},
        "blocks.{i}.attn.k": {"module": "layers.{i}.attention.wk", "kind": "linear"},
        "blocks.{i}.attn.v": {"module": "layers.{i}.attention.wv", "kind": "linear"},
        "blocks.{i}.attn.o": {"module": "layers.{i}.attention.wo", "kind": "linear"},
        "blocks.{i}.mlp": {"module": "layers.{i}.feed_forward", "kind": "mlp"},
        "blocks.{i}.mlp.in": {"module": "layers.{i}.feed_forward.w1", "kind": "linear"},
        "blocks.{i}.mlp.out": {"module": "layers.{i}.feed_forward.w2", "kind": "linear"},
        "ln_final": {"module": "norm", "kind": "norm"},
        "unembed": {"module": "output", "kind": "unembed"},
    },
}

ADDRESSED_SCHEDULE = {
    0: {"params": ["layers:-1: & component:mlp.*"], "max_transition_epoch": 1},
    1: {"params": ["role:latent:probe"], "max_transition_epoch": 2},
    2: {"params": ["layers:0 & component:attn.*"]},
}


def register_addressed_model(frozen: bool) -> None:
    """Register the component map, and have the test module attach the latent model where a module attaches what it
    tunes: after its model exists and before its view is built."""
    from venngram.maps import from_document, register

    from interpretune.tuning.latents import LatentSite, attach_latent_model
    from tests.modules import TestITModule

    register(from_document(COMPONENT_MAP))

    def register_tunable_modules(self) -> None:
        width = self.model.layers[0].feed_forward.w1.in_features
        attach_latent_model(
            self, "probe", TinyLatent(width), LatentSite("model.layers.0.feed_forward", op="replace"), frozen=frozen
        )

    TestITModule.register_tunable_modules = register_tunable_modules


def full_probe_params(module: nn.Module) -> dict[str, list[float]]:
    """The attached latent model's parameters, gathered: each rank holds a shard of a split one."""
    return {
        n: (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().flatten().tolist()
        for n, p in module.latent_models["probe"].named_parameters()
    }


class SaveCheckpoint:
    """Saves a resumable checkpoint at the end of epoch ``at_epoch`` (after FTSCoreCallback, when phased)."""

    def __init__(self, checkpoint_dir: str, at_epoch: int = 0) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.at_epoch = at_epoch

    def on_train_epoch_end(self, ctx, metrics):
        if ctx.epoch == self.at_epoch:
            ctx.save_checkpoint(self.checkpoint_dir)


def unit_settings(module: torch.nn.Module) -> dict[str, dict[str, str | bool]]:
    """Each FSDP2 unit's mixed precision policy and whether composable activation checkpointing was applied to it,
    read from the sharded module (torch exposes neither publicly)."""
    from torch.distributed._composable.contract import _get_registry
    from torch.distributed.fsdp import FSDPModule

    settings = {}
    for name, unit in module.named_modules():
        if isinstance(unit, FSDPModule):
            policy = unit._get_fsdp_state()._fsdp_param_group.fsdp_params[0].mp_policy
            settings[name] = {
                "param_dtype": str(policy.param_dtype),
                "reduce_dtype": str(policy.reduce_dtype),
                "act_ckpt": "checkpoint" in (_get_registry(unit) or {}),
            }
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement", required=True)
    parser.add_argument("--world-size", type=int, default=2, help="the ranks the reference's global batch stands for")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fsdp-plan", type=json.loads, default=None, help="the FSDP2 plan, as JSON")
    parser.add_argument("--fsdp-default-kwargs", type=json.loads, default=None, help="every unit's kwargs, as JSON")
    parser.add_argument("--tp-plan", type=json.loads, default=None, help="the tensor-parallel plan, as JSON")
    parser.add_argument("--phased", action="store_true", help="tune in two phases with FTSCoreCallback")
    parser.add_argument("--max-epochs", type=int, default=None, help="override the run's epoch count")
    parser.add_argument("--save-checkpoint-dir", default=None, help="save a checkpoint at the first epoch's end")
    parser.add_argument("--resume-from", default=None, help="resume from a checkpoint directory")
    parser.add_argument("--out-name", default=None, help="output record stem (default: <placement>)")
    parser.add_argument("--latent-site", default=None, help="splice a latent model at this module path")
    parser.add_argument("--transcoder-site", default=None, help="splice a transcoder over this module")
    parser.add_argument("--intervention-site", default=None, help="add a learned vector to this block's output")
    parser.add_argument(
        "--addressed",
        nargs="?",
        const="roles",
        choices=("roles", "promote"),
        help="tune by venngram address in three phases, the second promoting a frozen latent model with `promote`",
    )
    parser.add_argument("--device-type", default="cpu", help="cuda places each rank on its own device (LOCAL_RANK)")
    parser.add_argument("--memprof-fsdp", action="store_true", help="profile every training step with FSDP2MemTracker")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle the training split (off by default: a one-process reference and the ranks' DistributedSampler "
        "shares draw different orders, so a run compared with one reads the split in order)",
    )
    args = parser.parse_args()

    from interpretune.config import SessionRunnerCfg
    from interpretune.runners import SessionRunner
    from interpretune.utils import MisconfigurationException
    from tests.configuration import config_modules
    from tests.parity_acceptance.test_it_l import CoreCfg

    rank = int(os.environ.get("RANK", "0"))
    memprofiler_cfg = None
    if args.memprof_fsdp:
        from interpretune.extensions.memprofiler import MemProfilerCfg, MemProfilerFuncs, MemProfilerSchedule

        # every training iteration of the run (two epochs of three batches when phased), FSDP2 tracking only
        memprofiler_cfg = MemProfilerCfg(
            enabled=True,
            track_fsdp_mem=True,
            collect_funcs=MemProfilerFuncs(default=[], fsdp=["training_step"]),
            enable_memory_hooks=False,
            enable_saved_tensors_hooks=False,
            schedule=MemProfilerSchedule(warmup_iters=0, max_iter=6),
        )
    batch_size = PER_RANK_BATCH * (args.world_size if args.placement == "single" else 1)
    test_cfg = CoreCfg(
        device_type=args.device_type,
        force_prepare_data=False,
        max_epochs=args.max_epochs or (3 if args.addressed else 2 if args.phased else 1),
        limit_train_batches=3,
        limit_val_batches=2,
        dm_override_cfg={
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "dataloader_kwargs": {"num_workers": 0, "pin_memory": False, "shuffle": args.shuffle},
        },
        # no dropout: each rank draws from its own RNG stream, the reference from one
        model_cfg={"model_args": {"max_seq_len": 200, "n_layers": 2, "dropout_p": 0.0}},
        memprofiler_cfg=memprofiler_cfg,
    )
    out_dir = Path(args.out_dir)
    torch.manual_seed(1 + rank)  # ranks initialize differently; placement must broadcast rank 0's weights
    if args.device_type == "cuda":
        # the session is built here, before the runner would make this rank's device current
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if args.addressed:
        register_addressed_model(frozen=args.addressed == "promote")
    it_session = config_modules(test_cfg, f"dist_{args.placement}", {}, out_dir / f"{args.placement}_rank{rank}")
    recorder = LossRecorder()
    callbacks: list = [recorder]
    fts = None
    trainable_recorder = None
    if args.addressed:
        from interpretune.tuning.core_loop import FTSCoreCallback

        # the schedule is addresses alone, so it is written before placement; the recorder reads each epoch's phase
        # before the tuning scheduler's own epoch end may start the next
        schedule = {depth: dict(phase) for depth, phase in ADDRESSED_SCHEDULE.items()}
        if args.addressed == "promote":
            schedule[1]["promote"] = {"role:artifact": "latent:probe"}
        fts = FTSCoreCallback(schedule)
        trainable_recorder = TrainableRecorder()
        trainable_recorder.module = it_session.module
        callbacks[:0] = [trainable_recorder, fts]
    elif args.phased:
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
    intervention = None
    if args.intervention_site:
        from interpretune.tuning.interventions import OpIntervention, attach_intervention

        width = it_session.module.get_submodule(f"{args.intervention_site}.feed_forward.w1").in_features
        intervention = OpIntervention("add", width).to(args.device_type)
        attach_intervention(it_session.module, "steer", intervention, args.intervention_site)
    runner_cfg = SessionRunnerCfg(
        it_session=it_session,
        max_epochs=test_cfg.max_epochs,
        limit_train_batches=test_cfg.limit_train_batches,
        limit_val_batches=test_cfg.limit_val_batches,
        callbacks=callbacks,
        placement=args.placement,
        fsdp_plan=args.fsdp_plan,
        fsdp_default_kwargs=args.fsdp_default_kwargs,
        tp_plan=args.tp_plan,
        resume_from=args.resume_from,
    )
    out_path = out_dir / f"{args.out_name or args.placement}_rank{rank}.json"
    try:
        runner = SessionRunner(run_cfg=runner_cfg)
    except MisconfigurationException as err:
        if latent is None:
            raise
        out_path.write_text(json.dumps({"refused": str(err)}), encoding="utf-8")
        return
    from torch.distributed.fsdp import FSDPModule

    module = it_session.module
    # after placement, so every rank holds rank 0's broadcast weights
    latent_init = {n: p.detach().flatten().tolist() for n, p in latent.named_parameters()} if latent else None
    param_types = sorted({type(p).__name__ for p in module.parameters()})
    fsdp_units = [name for name, sub in module.named_modules() if isinstance(sub, FSDPModule)]
    unit_policies = unit_settings(module) if args.fsdp_default_kwargs else {}
    probe_init = full_probe_params(module) if args.addressed else None
    split_params = sorted(n for n, p in module.named_parameters() if type(p).__name__ == "DTensor")
    if fts is not None and not args.addressed:
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
        "unit_settings": unit_policies,
        "split_params": split_params if args.placement == "tp" else [],
    }
    if args.device_type == "cuda":
        record["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if latent is not None:  # whole on every rank: a latent model is never split
        record["latent_params"] = {n: p.detach().flatten().tolist() for n, p in latent.named_parameters()}
        record["latent_init"] = latent_init
    if intervention is not None:  # gathered: each rank holds a shard of a split operand
        record["intervention_params"] = {
            n: (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().flatten().tolist()
            for n, p in intervention.named_parameters()
        }
    if trainable_recorder is not None:
        record["trainable_by_epoch"] = trainable_recorder.trainable
        record["latent_role"] = sorted(set(fts.tuning_scheduler.roles.values()))  # pyright: ignore
        record["probe_init"], record["probe_params"] = probe_init, full_probe_params(module)
    if args.memprof_fsdp:
        # this rank's FSDP2 memory report, written when the session ended
        log_dir = module.memprofiler.memprof_log_dir
        record["fsdp_mem_report"] = str(log_dir / f"rank_{rank}_fsdp_memory_report.json")
    if fts is not None:
        assert fts.tuning_scheduler is not None
        record["depth"] = fts.tuning_scheduler.curr_depth
        record["trainable"] = sorted(n for n, p in module.named_parameters() if p.requires_grad)
    out_path.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
