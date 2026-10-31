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
import functools
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from interpretune.utils import _FTS_AVAILABLE, MisconfigurationException, rank_zero_only, rank_zero_warn

Placement = Literal["single", "ddp", "fsdp2"]
PLACEMENTS: tuple[str, ...] = ("single", "ddp", "fsdp2")
# the attribute holding rank 0's CPU copy of a model built on the meta device (`HFFromPretrainedConfig.meta_init`),
# which `place_module` loads into the sharded model and then drops
META_INIT_SOURCE = "_meta_init_source"
# the attribute holding the local directory of safetensors files every rank reads its own shards from
# (`HFFromPretrainedConfig.meta_init_load: dcp`)
META_INIT_CHECKPOINT = "_meta_init_checkpoint"


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

    A module built on the meta device (``HFFromPretrainedConfig.meta_init``) holds no values to broadcast. It is
    sharded as it is, materialized on each rank's device with ``to_empty``, and then loaded from rank 0's CPU copy of
    the checkpoint (see :func:`_load_from_rank0`), so no rank ever holds the whole model on its accelerator.

    Raises:
        MisconfigurationException: If a rank's module spans several devices or holds no parameters, ranks share an
            accelerator or differ in device type, a parameter is non-contiguous, ``fsdp_plan`` is missing (for
            ``fsdp2``) or has a key that selects no module, a plan is given without finetuning-scheduler installed, or
            a meta-built module cannot be loaded (see :func:`_load_from_rank0`).
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    on_meta = meta_built(module)
    local = [str(rank_device(run))] if on_meta else sorted({str(param.device) for param in module.parameters()})
    per_rank: list[Any] = [None] * run.world_size
    dist.all_gather_object(per_rank, local)
    run.devices = _check_devices(per_rank, run.placement)
    _refuse_unshardable(module)
    _require_fts(fsdp_plan, fsdp_default_kwargs)
    units = _resolve_plan(module, fsdp_plan, fsdp_default_kwargs, run.placement)
    if not on_meta:
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
    _register_direct_calls(module)
    if on_meta:
        kept = {name: buffer.clone() for name, buffer in module.named_buffers() if not buffer.is_meta}
        module.to_empty(device=rank_device(run))
        with torch.no_grad():
            for name, value in kept.items():
                buffer = module.get_buffer(name)
                buffer.copy_(value)
                dist.broadcast(buffer, src=0)
        if hasattr(module, META_INIT_CHECKPOINT):
            _load_from_checkpoint_files(module, run)
        else:
            _load_from_rank0(module, run)


def _register_direct_calls(module: torch.nn.Module) -> None:
    """Register the methods a composition calls on its submodules outside its own forward as FSDP2 forward methods.

    FSDP2 gathers a unit's parameters in hooks on ``forward``, so a submodule called directly (an nnsight model's
    Hugging Face module, which nnsight calls rather than the interpretune module) or through another method (a
    transcoder's ``encode``) would compute with sharded parameters. The module declares these calls with
    ``fsdp_direct_calls()``; each declared submodule must be its own unit for its methods to be registered.

    Raises:
        MisconfigurationException: If a declared submodule is not its own FSDP2 unit, naming the plan keys to add.
    """
    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    declare = getattr(module, "fsdp_direct_calls", None)
    entries: list[tuple[str, torch.nn.Module, tuple[str, ...]]] = list(declare()) if callable(declare) else []  # type: ignore[arg-type]
    missing: dict[str, list[str]] = {}
    for name, submodule, methods in entries:
        sharded = isinstance(submodule, FSDPModule)  # a flag, not a narrowing: the module stays a torch module
        if not sharded:
            missing.setdefault(type(submodule).__name__, []).append(name)
            continue
        _register_forward_methods(submodule, methods, register_fsdp_forward_method)
    if missing:
        names = [n for group in missing.values() for n in group]
        raise MisconfigurationException(
            f"This composition calls {', '.join(names[:3])}{' and more' if len(names) > 3 else ''} directly rather"
            " than through its forward, so FSDP2 gathers their parameters only if each is its own unit. Add"
            f" {', '.join(f'`class:{cls}`' for cls in sorted(missing))} to `fsdp_plan`."
        )


def _register_forward_methods(submodule: torch.nn.Module, methods: Sequence[str], register: Any) -> None:
    """Register ``methods`` of one FSDP2 unit as forward methods, gathering its parameters only at the outermost
    call.

    The declared methods call one another (a transcoder's ``decode_sparse`` calls ``_get_decoder_vectors``, its
    ``forward`` calls ``encode`` and ``decode``), and FSDP2 reshards a unit when any registered method returns. Without
    this guard the inner call's return reshards the unit while the outer call is still computing, which then reads a
    sharded parameter (``aten.add.Tensor got mixed torch.Tensor and DTensor``). So a call made while another registered
    call on the same unit is in flight, ``forward`` included, runs the original method.
    """
    depth = [0]

    def enter(*_: Any) -> None:
        depth[0] += 1

    def leave(*_: Any) -> None:
        depth[0] -= 1

    if "forward" in methods:
        submodule.register_forward_pre_hook(enter)
        submodule.register_forward_hook(leave, always_call=True)
    for method in methods:
        if method == "forward":
            continue
        inner = getattr(submodule, method)
        register(submodule, method)
        outer = getattr(submodule, method)

        @functools.wraps(inner)
        def guarded(*args: Any, _inner: Any = inner, _outer: Any = outer, **kwargs: Any) -> Any:
            if depth[0]:
                return _inner(*args, **kwargs)
            depth[0] += 1
            try:
                return _outer(*args, **kwargs)
            finally:
                depth[0] -= 1

        setattr(submodule, method, guarded)


def rank_device(run: DistributedRun) -> torch.device:
    """The device this rank's module belongs on: its accelerator (made current when the group was joined), else the
    CPU."""
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


@contextmanager
def params_on_meta() -> Iterator[None]:
    """Within the context, each parameter a module registers is moved to the meta device as it is registered, while
    buffers are built with their values on the CPU.

    A model built here holds no weights, but its non-persistent buffers (rotary frequencies, for one) are real; under
    ``torch.device("meta")`` they would be uninitialized with no checkpoint to load them from. Each parameter is
    allocated on the CPU only until it is registered, so the peak is one parameter, not the model. It patches
    ``torch.nn.Module.register_parameter`` for the duration, so it is not safe to build other modules concurrently.
    """
    register = torch.nn.Module.register_parameter

    def register_on_meta(self: torch.nn.Module, name: str, param: torch.nn.Parameter | None) -> None:
        register(self, name, param)
        # a parameter already on meta is re-registered as it is, so tying one module's weight to another's (as
        # `tie_weights` does by assignment) keeps them one object
        if param is not None and not param.is_meta:
            kwargs = {**param.__dict__, "requires_grad": param.requires_grad}
            self._parameters[name] = type(param)(param.to("meta"), **kwargs)

    torch.nn.Module.register_parameter = register_on_meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = register


def meta_built(module: torch.nn.Module) -> bool:
    """Whether ``module``'s parameters were built on the meta device, refusing a module only partly built there.

    Buffers may hold values on a meta-built module (see :func:`params_on_meta`); :func:`place_module` keeps them across
    materialization. A buffer left on the meta device of a module whose parameters are not is refused, since nothing
    would ever give it values.
    """
    params = dict(module.named_parameters())
    on_meta = [p.is_meta for p in params.values()]
    if any(on_meta) and not all(on_meta):
        names = [n for n, p in params.items() if not p.is_meta]
        raise MisconfigurationException(
            f"This module was built partly on the meta device: {len(names)} tensors are not ({', '.join(names[:3])}"
            f"{' and more' if len(names) > 3 else ''}). Materializing the meta part would also discard their values,"
            " so build the whole module on the meta device or none of it."
        )
    built = bool(params) and all(on_meta)
    if not built and (meta_buffers := [n for n, b in module.named_buffers() if b.is_meta]):
        raise MisconfigurationException(
            f"This module was built partly on the meta device: its parameters are not, but {len(meta_buffers)} buffers"
            f" are ({', '.join(meta_buffers[:3])}{' and more' if len(meta_buffers) > 3 else ''}), and nothing would"
            " give them values. Build the whole module on the meta device or none of it."
        )
    return built


def _checkpoint_renamer(model: torch.nn.Module) -> Any:
    """``rename(checkpoint_key, model_keys)``: the model key transformers' own checkpoint conversion maps a key to.

    Raises:
        MisconfigurationException: If the model's conversion mapping has weight converters (which merge or reshape
            tensors), since loading per-rank slices can only rename.
    """
    from transformers import PreTrainedModel, conversion_mapping, core_model_loading

    if not isinstance(model, PreTrainedModel):
        raise MisconfigurationException(
            f"`meta_init_load: dcp` renames checkpoint keys with transformers' conversion mapping, which needs a"
            f" `PreTrainedModel`, not a {type(model).__name__}. Use `meta_init_load: rank0`."
        )
    transforms = conversion_mapping.get_model_conversion_mapping(model)
    renamings = [t for t in transforms if isinstance(t, core_model_loading.WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, core_model_loading.WeightConverter)]
    if converters:
        raise MisconfigurationException(
            "`meta_init_load: dcp` reads each rank's slices of the checkpoint as they are stored, but"
            f" {type(model).__name__}'s checkpoint needs transformers' weight converters"
            f" ({', '.join(str(c.source_patterns) for c in converters[:3])}), which merge or reshape tensors. Use"
            " `meta_init_load: rank0`."
        )
    prefix = getattr(model, "base_model_prefix", None) or None

    def rename(key: str, model_keys: Mapping[str, Any]) -> str:
        return core_model_loading.rename_source_key(key, renamings, [], prefix, dict(model_keys))[0]

    return rename


def _load_target(module: torch.nn.Module) -> torch.nn.Module | None:
    """The torch module a meta-built model is loaded into: the module's ``meta_load_target()``, else ``module.model``.

    An adapter whose model is a wrapper rather than a torch module (an nnsight or circuit-tracer model) names the module
    it wraps, which holds every parameter the placement sharded.
    """
    target = getattr(module, "meta_load_target", None)
    model = target() if callable(target) else getattr(module, "model", None)
    return model if isinstance(model, torch.nn.Module) else None


def _load_from_checkpoint_files(module: torch.nn.Module, run: DistributedRun) -> None:
    """Load the load target's (see :func:`_load_target`) sharded parameters from the checkpoint's safetensors
    files, each rank reading only the slices it holds (``torch.distributed.checkpoint`` with
    ``HuggingFaceStorageReader``).

    Every rank checks the checkpoint's keys against the model before reading, from the same files, so all ranks refuse
    together. A parameter tied to one the checkpoint supplies (an LM head tied to the embedding) needs no key of its
    own, and keys the model declares ``from_pretrained`` should ignore (``_keys_to_ignore_on_load_unexpected`` and
    ``_keys_to_ignore_on_load_missing``) are ignored here too. Buffers are not read: :func:`place_module` kept their
    built values.

    Raises:
        MisconfigurationException: If the directory holds no safetensors checkpoint, a checkpoint key maps to no model
            key, a model parameter is supplied by no key, or the model needs weight converters.
    """
    import json

    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
    from torch.distributed.checkpoint.state_dict_loader import load

    model = _load_target(module)
    assert model is not None
    path = getattr(module, META_INIT_CHECKPOINT)
    index = os.path.join(path, "model.safetensors.index.json")
    single = os.path.join(path, "model.safetensors")
    if os.path.isfile(index):
        with open(index) as f:
            keys = list(json.load(f)["weight_map"])
    elif os.path.isfile(single):
        from safetensors import safe_open

        with safe_open(single, framework="pt") as f:
            keys = list(f.keys())
    else:
        raise MisconfigurationException(
            f"`meta_init_load: dcp` reads safetensors files, but {path} holds neither `model.safetensors` nor"
            " `model.safetensors.index.json`. Use `meta_init_load: rank0`."
        )
    rename = _checkpoint_renamer(model)
    targets = model.state_dict()
    params = dict(model.named_parameters())
    # patterns a model declares for keys `from_pretrained` ignores (old checkpoints' attention-mask buffers, for one)
    ignore_unexpected = getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
    ignore_missing = getattr(model, "_keys_to_ignore_on_load_missing", None) or []
    mapped = {key: rename(key, targets) for key in keys}
    ignored = {k for k, v in mapped.items() if v not in targets and any(re.search(p, k) for p in ignore_unexpected)}
    mapped = {k: v for k, v in mapped.items() if k not in ignored}
    unknown = sorted(k for k, v in mapped.items() if v not in targets)
    supplied = {id(params[v]) for v in mapped.values() if v in params}
    unsupplied = sorted(
        n
        for n, p in model.named_parameters(remove_duplicate=False)
        if id(p) not in supplied and not any(re.search(pattern, n) for pattern in ignore_missing)
    )
    if unknown or unsupplied:
        raise MisconfigurationException(
            "The checkpoint's files do not match the meta-built model, so loading them would leave parameters"
            f" uninitialized or misplaced. Keys no model key matches: {unknown[:5]}; parameters no key supplies:"
            f" {unsupplied[:5]}."
        )
    load({key: targets[name] for key, name in mapped.items()}, storage_reader=HuggingFaceStorageReader(path))
    object.__delattr__(module, META_INIT_CHECKPOINT)


def _load_from_rank0(module: torch.nn.Module, run: DistributedRun) -> None:
    """Load rank 0's CPU copy of the checkpoint into the load target (see :func:`_load_target`), sharded and
    materialized on every rank.

    Rank 0 checks the copy against the model first and every rank raises the same refusal, so no rank is left
    waiting in a collective. Buffers are broadcast from the copy too, since ``to_empty`` left them uninitialized and
    the non-persistent ones (rotary frequencies, for one) are in no state dict.

    Raises:
        MisconfigurationException: If rank 0 holds no copy (the module was built on the meta device by something other
            than ``meta_init``), or the copy is missing a key of the model, has one the model lacks, or differs from it
            in a shape (for one, embeddings resized after the build).
    """
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    model = _load_target(module)
    source = getattr(module, META_INIT_SOURCE, None)
    problem, full = None, {}
    if run.rank == 0:
        if model is None:
            wrapper = type(getattr(module, "model", None)).__name__
            problem = (
                f"The module was built on the meta device, but its model is a {wrapper} rather than a torch module and"
                " `meta_load_target()` names none to load the checkpoint into."
            )
        elif source is None:
            problem = (
                "The module was built on the meta device, but rank 0 holds no checkpoint to load it from. Build it"
                " with `HFFromPretrainedConfig.meta_init`, which loads one on rank 0's CPU."
            )
        else:
            full = source.state_dict()
            expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
            missing = sorted(set(expected) - set(full))
            unexpected = sorted(set(full) - set(expected))
            reshaped = sorted(k for k in set(expected) & set(full) if tuple(full[k].shape) != expected[k])
            if missing or unexpected or reshaped:
                problem = (
                    "Rank 0's checkpoint does not match the meta-built model, so loading it would leave parameters"
                    f" uninitialized or misplaced. Missing: {missing[:5]}; unexpected: {unexpected[:5]}; different"
                    f" shapes: {reshaped[:5]}."
                )
    box = [problem]
    dist.broadcast_object_list(box, src=0)
    if box[0] is not None:
        raise MisconfigurationException(box[0])
    assert model is not None
    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    set_model_state_dict(model, full, options=options)
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if run.rank == 0:
                assert source is not None
                buffer.copy_(source.get_buffer(name))
            dist.broadcast(buffer, src=0)
    if source is not None:
        object.__delattr__(module, META_INIT_SOURCE)


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
