"""Distributed placement for the core training and test loops.

``SessionRunnerCfg.placement`` chooses one of three placements:

- ``single``: one process and no process group (the default).
- ``ddp``: every rank holds the whole model, and gradients are averaged across ranks. This is ``fully_shard`` over a
  ``(replicate, shard)`` mesh whose shard dimension has size one, so the parameters are replicated ``DTensor``\\ s and
  the module is wrapped in place: no wrapper module stands between callers and ``module.model``, hooks or a tunable
  view's names, and finetuning-scheduler sees the same kind of module under both data-parallel placements.
- ``fsdp2``: parameters, gradients and optimizer state are sharded across ranks. ``fully_shard`` is applied to every
  module ``SessionRunnerCfg.fsdp_plan`` selects, deepest first, then to the interpretune module as the root unit.

The plan is finetuning-scheduler's FSDP2 plan (``finetuning_scheduler.strategies.plan.shard_model``), so the core loop
takes the same keys (module names, regexes and ``class:`` selectors), per-unit ``fully_shard`` kwargs and aliases
(``act_ckpt``, ``cpu_offload_policy``) as finetuning-scheduler's own interfaces. Under ``ddp`` a plan is optional and
shards over the ``(replicate, shard)`` mesh.

A run is launched with ``torchrun``, which sets ``WORLD_SIZE``, ``RANK`` and ``LOCAL_RANK``. Each rank builds the
session in turn (rank 0 first, so dataset preparation is not raced), the ranks then agree on their devices and the
module is placed before its optimizers are built. Each rank's loaders are rebuilt over a ``DistributedSampler``, and the
loops average logged metrics across ranks, so callbacks, lr schedulers and transition policies see one global value on
every rank (ranks that disagree on a phase transition deadlock at the next collective).

Every combination this module cannot vouch for is refused by name: a placement that does not match the launch, a rank
whose parameters span several devices (a layer split under data parallelism), two ranks on one accelerator, a parameter
``fully_shard`` cannot shard, and a loader that cannot be rebuilt faithfully over a ``DistributedSampler``.
"""

from __future__ import annotations

import atexit
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from interpretune.utils import _FTS_AVAILABLE, MisconfigurationException, rank_zero_only, rank_zero_warn

Placement = Literal["single", "ddp", "fsdp2"]
PLACEMENTS: tuple[str, ...] = ("single", "ddp", "fsdp2")


@dataclass
class DistributedRun:
    """This process's place in a data-parallel run.

    Attributes:
        placement: ``ddp`` or ``fsdp2``.
        rank: The global rank.
        local_rank: The rank on this host, which selects its accelerator.
        world_size: The number of ranks.
        devices: Each rank's device, by rank, recorded when the module is placed (part of the run's provenance: a run
            whose ranks shared one accelerator would otherwise read as data parallel).
        mesh: The ``DeviceMesh`` the module was sharded over.
    """

    placement: str
    rank: int
    local_rank: int
    world_size: int
    devices: tuple[str, ...] = ()
    mesh: Any = field(default=None, repr=False)

    @property
    def is_rank_zero(self) -> bool:
        """Whether this is global rank 0."""
        return self.rank == 0

    @contextmanager
    def rank_zero_first(self) -> Iterator[None]:
        """Run the enclosed block on rank 0, then on the other ranks.

        Session builds write shared caches (a datamodule's prepared dataset), and concurrent builds race on them: one
        rank reads a file another is still writing. Rank 0 reaches the barrier even when its block raises, so the
        others are released (and ``torchrun`` stops the group once rank 0 exits).
        """
        import torch.distributed as dist

        if self.rank != 0:
            dist.barrier()
            yield
            return
        try:
            yield
        finally:
            dist.barrier()


def resolve_placement(placement: str, env: Mapping[str, str] | None = None) -> DistributedRun | None:
    """This process's :class:`DistributedRun` under ``placement``, or ``None`` for ``single``.

    Raises:
        MisconfigurationException: If ``placement`` is unknown, ``single`` is asked for in one of several ranks (each
            would train its own copy on the whole dataset), a data-parallel placement is asked for with fewer than two
            ranks, or the launcher set no ``RANK`` or ``LOCAL_RANK``.
    """
    env = os.environ if env is None else env
    if placement not in PLACEMENTS:
        raise MisconfigurationException(f"Unknown placement '{placement}': expected one of {', '.join(PLACEMENTS)}.")
    world_size = int(env.get("WORLD_SIZE", "1"))
    if placement == "single":
        if world_size > 1:
            raise MisconfigurationException(
                f"This process is one of WORLD_SIZE={world_size} ranks, but the runner's placement is 'single', so"
                " every rank would train its own copy of the model on the whole dataset. Set `placement` to 'ddp' or"
                " 'fsdp2', or launch a single process."
            )
        return None
    if world_size < 2:
        raise MisconfigurationException(
            f"Placement '{placement}' needs at least two ranks, but WORLD_SIZE is {env.get('WORLD_SIZE', 'unset')}."
            " Launch with torchrun (`torchrun --nproc-per-node <N> ...`), or use placement 'single'."
        )
    if missing := [key for key in ("RANK", "LOCAL_RANK") if key not in env]:
        raise MisconfigurationException(
            f"Placement '{placement}' found WORLD_SIZE={world_size} but no {' or '.join(missing)}: launch with"
            " torchrun, which sets all three."
        )
    return DistributedRun(placement, int(env["RANK"]), int(env["LOCAL_RANK"]), world_size)


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def init_process_group(run: DistributedRun) -> None:
    """Join (or create, from the launcher's environment) the process group ``run`` describes.

    When accelerators are present, this rank's (``LOCAL_RANK``) becomes the current device before the session is built,
    so a module placed on ``"cuda"`` lands on it, and the group gets both a ``gloo`` backend (for CPU tensors) and an
    ``nccl`` one. A group this function creates is destroyed at exit.

    Raises:
        MisconfigurationException: If a process group already exists with another rank or world size.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size()) != (run.rank, run.world_size):
            raise MisconfigurationException(
                f"A process group already exists as rank {dist.get_rank()} of {dist.get_world_size()}, but the launcher"
                f" environment says rank {run.rank} of {run.world_size}."
            )
    else:
        backend = "gloo"
        if torch.cuda.is_available():
            if run.local_rank < torch.cuda.device_count():
                torch.cuda.set_device(run.local_rank)
            backend = "cpu:gloo,cuda:nccl"
        dist.init_process_group(backend)
        atexit.register(_destroy_process_group)
    rank_zero_only.rank = run.rank  # type: ignore[attr-defined]  # rank-zero printing and warnings follow the group


def _check_devices(per_rank: Sequence[Sequence[str]], placement: str) -> tuple[str, ...]:
    for rank, devices in enumerate(per_rank):
        if len(devices) != 1:
            held = ", ".join(devices) or "no parameters"
            raise MisconfigurationException(
                f"Placement '{placement}' needs each rank's module on one device, but rank {rank} holds {held}. A layer"
                " split combined with data parallelism is not supported."
            )
    devices = tuple(devices[0] for devices in per_rank)
    if len({torch.device(d).type for d in devices}) > 1:
        raise MisconfigurationException(f"Placement '{placement}' found ranks on different device types: {devices}.")
    if torch.device(devices[0]).type != "cpu":
        seen: dict[str, int] = {}
        for rank, device in enumerate(devices):
            if device in seen:
                raise MisconfigurationException(
                    f"Ranks {seen[device]} and {rank} both placed the module on {device}, so this would not be a"
                    f" data-parallel run. Place each rank on its own device (the runner makes `cuda:LOCAL_RANK` current"
                    " before the build, so a module placed on 'cuda' lands there), not on a fixed index."
                )
            seen[device] = rank
    return devices


def _refuse_unshardable(module: torch.nn.Module) -> None:
    if names := [name for name, param in module.named_parameters() if not param.is_contiguous()]:
        shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
        raise MisconfigurationException(
            f"`fully_shard` cannot shard non-contiguous parameters, and this module has {len(names)}: {shown}. They are"
            " usually transposed views of a packed weight (TransformerLens's split q/k/v, for one); this backend"
            " cannot be placed with 'ddp' or 'fsdp2' until they are contiguous."
        )


def _resolve_plan(
    module: torch.nn.Module,
    fsdp_plan: Mapping[str, Mapping[str, Any] | None] | None,
    default_kwargs: Mapping[str, Any] | None,
    placement: str,
) -> dict[str, dict[str, Any]]:
    """``fsdp_plan`` expanded to the modules it selects, each with its ``fully_shard`` kwargs, deepest first.

    finetuning-scheduler applies a plan in the order its keys first select modules and refuses a module whose ancestor
    is already sharded, so ``class:Block`` listed before ``class:Instrument`` would fail for an instrument spliced
    inside a block. Ordering by depth makes the plan's key order immaterial. The module itself is sharded last, as the
    root.
    """
    if fsdp_plan is None:
        if placement == "ddp":
            return {}
        raise MisconfigurationException(
            "Placement 'fsdp2' needs `fsdp_plan`: the modules to shard as their own units, keyed by name, regex or"
            " `class:<name>` (for example `{'class:GPT2Block': {}}`, plus each spliced instrument's class so that it is"
            " gathered on its own). An empty plan shards the module as one unit, which gathers every parameter for the"
            " whole forward."
        )
    if not fsdp_plan:
        return {}
    from finetuning_scheduler.core.exceptions import FTSConfigurationError
    from finetuning_scheduler.strategies.plan import resolve_fsdp_plan

    try:
        resolved = resolve_fsdp_plan(module, fsdp_plan, default_kwargs, warn=rank_zero_warn)
    except FTSConfigurationError as err:
        raise MisconfigurationException(str(err)) from err
    resolved.pop("", None)  # the module itself is always the last, root unit
    return dict(sorted(resolved.items(), key=lambda item: item[0].count("."), reverse=True))


def _require_fts(fsdp_plan: Mapping[str, Any] | None, default_kwargs: Mapping[str, Any] | None) -> None:
    if (fsdp_plan or default_kwargs) and not _FTS_AVAILABLE:
        raise MisconfigurationException(
            "`fsdp_plan` and `fsdp_default_kwargs` are applied with finetuning-scheduler's FSDP2 plan, which is not"
            " installed. Install interpretune's `tune` extra, or shard the module as one unit with an empty plan."
        )


def place_module(
    module: torch.nn.Module,
    run: DistributedRun,
    fsdp_plan: Mapping[str, Mapping[str, Any] | None] | None = None,
    fsdp_default_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Agree on each rank's device, build the mesh and shard ``module`` in place as ``run.placement`` asks.

    Runs on every rank, after the session is built and before its optimizers are, since sharding replaces the
    parameters with ``DTensor``\\ s. Rank 0's parameters and buffers are broadcast to the other ranks first, as
    ``DistributedDataParallel`` does: ``fully_shard`` keeps each rank's own values, so ranks whose initialization
    differed (an unseeded random init) would otherwise train different replicas (``ddp``) or assemble one model from
    shards of different ones (``fsdp2``), with no error.

    Raises:
        MisconfigurationException: If a rank's module spans several devices or holds no parameters, ranks share an
            accelerator or differ in device type, a parameter is non-contiguous, ``fsdp_plan`` is missing (for
            ``fsdp2``) or has a key that selects no module, or a plan is given without finetuning-scheduler installed.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    local = sorted({str(param.device) for param in module.parameters()})
    per_rank: list[Any] = [None] * run.world_size
    dist.all_gather_object(per_rank, local)
    run.devices = _check_devices(per_rank, run.placement)
    _refuse_unshardable(module)
    _require_fts(fsdp_plan, fsdp_default_kwargs)
    units = _resolve_plan(module, fsdp_plan, fsdp_default_kwargs, run.placement)
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            dist.broadcast(tensor, src=0)
    device_type = torch.device(run.devices[run.rank]).type
    if run.placement == "ddp":
        mesh = init_device_mesh(device_type, (run.world_size, 1), mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh(device_type, (run.world_size,), mesh_dim_names=("shard",))
    if units or fsdp_default_kwargs:
        from finetuning_scheduler.core.exceptions import FTSConfigurationError
        from finetuning_scheduler.strategies.plan import shard_model

        try:
            shard_model(module, mesh, units, default_kwargs=fsdp_default_kwargs)
        except FTSConfigurationError as err:
            raise MisconfigurationException(str(err)) from err
    else:
        fully_shard(module, mesh=mesh)
    run.mesh = mesh


def distributed_dataloader(loader: Iterable, run: DistributedRun, seed: int = 0) -> Any:
    """``loader`` rebuilt over a ``DistributedSampler`` that gives this rank its share of the dataset.

    The rebuilt loader keeps the batch size, collate function, worker settings and ``drop_last``, and shuffles when the
    original did (with ``seed``, the same on every rank; call ``sampler.set_epoch`` each epoch). Without ``drop_last``,
    ``DistributedSampler`` pads the last round with repeated samples so every rank takes the same number of batches,
    so a metric averaged over an uneven dataset counts those samples twice.

    Raises:
        MisconfigurationException: If ``loader`` is not a ``DataLoader``, iterates an ``IterableDataset``, or uses a
            sampler or batch sampler other than PyTorch's sequential, random (without replacement) and batch samplers,
            which a ``DistributedSampler`` cannot reproduce.
    """
    from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler, SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    if not isinstance(loader, DataLoader):
        raise MisconfigurationException(
            f"Placement '{run.placement}' shards data by rebuilding the datamodule's loaders over a"
            f" `DistributedSampler`, which needs a `DataLoader`, not a {type(loader).__name__}."
        )
    if isinstance(loader.dataset, IterableDataset):
        raise MisconfigurationException(
            f"Placement '{run.placement}' cannot shard an `IterableDataset` ({type(loader.dataset).__name__}) by"
            " sampler: shard it by rank in the dataset itself."
        )
    sampler = loader.sampler
    faithful = type(loader.batch_sampler) is BatchSampler and (
        type(sampler) is SequentialSampler
        or (type(sampler) is RandomSampler and not sampler.replacement and sampler._num_samples is None)
    )
    if not faithful:
        raise MisconfigurationException(
            f"Placement '{run.placement}' cannot rebuild a loader whose sampler is a {type(sampler).__name__} (batch"
            f" sampler {type(loader.batch_sampler).__name__}) over a `DistributedSampler` without changing what it"
            " samples: use PyTorch's sequential or random sampler (without replacement), or shard the data by rank in"
            " the datamodule."
        )
    dist_sampler = DistributedSampler(
        loader.dataset,
        num_replicas=run.world_size,
        rank=run.rank,
        shuffle=type(sampler) is RandomSampler,
        seed=seed,
        drop_last=loader.drop_last,
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=dist_sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        prefetch_factor=loader.prefetch_factor if loader.num_workers > 0 else None,
        persistent_workers=loader.persistent_workers,
    )


def reduce_metrics(metrics: Mapping[str, float], run: DistributedRun) -> dict[str, float]:
    """``metrics`` averaged across ranks, the same on every rank.

    Each rank's values are already its per-epoch averages over the same number of batches (``DistributedSampler``
    equalizes them), so their mean is the global average.

    Raises:
        MisconfigurationException: If ranks logged different metric names, since a value missing on one rank cannot
            be averaged and a policy reading it would decide differently per rank.
    """
    import torch.distributed as dist

    keys = sorted(metrics)
    per_rank: list[Any] = [None] * run.world_size
    dist.all_gather_object(per_rank, keys)
    if any(rank_keys != keys for rank_keys in per_rank):
        differing = {rank: rank_keys for rank, rank_keys in enumerate(per_rank) if rank_keys != keys}
        raise MisconfigurationException(
            f"Ranks logged different metrics this epoch (rank {run.rank}: {keys}; differing: {differing}), so they"
            " cannot be averaged across ranks."
        )
    if not keys:
        return {}
    device = "cpu" if "gloo" in str(dist.get_backend()) else torch.device(torch.cuda.current_device())
    values = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    return {k: v / run.world_size for k, v in zip(keys, values.tolist())}
